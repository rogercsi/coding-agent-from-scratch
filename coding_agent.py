"""
LangGraph Industrial-Grade Agent
  - Memory & Session       : MemorySaver checkpointer + mem0 long-term memory (user-scoped, LLM-extracted)
  - Tool & File System     : read/write/edit/apply_diff/list/search + run_python/run_shell
  - RAG / Vector Search    : ChromaDB + all-MiniLM-L6-v2, Python-aware chunking, MD5 incremental index
  - Unified Diff Editing   : edit_file fuzzy SequenceMatcher fallback + apply_diff unified-diff tool
  - Async Full-Duplex REPL : graph.astream() + asyncio.Queue stdin reader + preemptive mid-turn cancel
  - MCP Client Adapter     : MCPToolRegistry — stdio & HTTP transports, JSON-RPC 2.0
  - Middleware             : pre/post stack (logging, rate-limit, skill-router)
  - Async Memory Factory   : background LLM extraction → structured fact upsert + conflict resolution
  - OpenTelemetry Tracing  : full span tree, token audit, prompt versioning, JSONL export
  - Interrupt & Resume     : human-in-the-loop gate (single / auto-approve)
  - Skill Middleware       : intent-based skill prompt injection
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import threading
import contextvars
import uuid
import weakref
import logging
import concurrent.futures.thread as _cft
from concurrent.futures import ThreadPoolExecutor


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose worker threads are daemon=True.

    Two guarantees:
    1. daemon=True   → threading._shutdown() atexit won't join them.
    2. Not in _cft._threads_queues → concurrent.futures _python_exit() atexit
       won't call t.join() either.
    3. shutdown(wait=…) is always non-blocking → asyncio's
       shutdown_default_executor() returns immediately instead of waiting for
       a thread that's blocked on sys.stdin.readline() or mem0 I/O.
    """

    def _adjust_thread_count(self) -> None:  # type: ignore[override]
        try:
            if self._idle_semaphore.acquire(timeout=0):
                return

            def _weakref_cb(_, q=self._work_queue):
                q.put(None)

            if len(self._threads) < self._max_workers:
                t = threading.Thread(
                    target=_cft._worker,
                    args=(weakref.ref(self, _weakref_cb),
                          self._work_queue,
                          self._initializer,
                          self._initargs),
                    daemon=True,
                )
                t.start()
                self._threads.add(t)
                # Intentionally NOT in _cft._threads_queues — skipped by _python_exit()
        except AttributeError:
            # CPython internals changed; fall back to parent (non-daemon threads).
            # shutdown() override still prevents blocking at interpreter exit.
            super()._adjust_thread_count()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        # Daemon threads die on exit anyway; skip the blocking t.join().
        super().shutdown(wait=False, cancel_futures=cancel_futures)
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field, create_model

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

# ── Optional OpenTelemetry SDK ────────────────────────────────────────────────
try:
    from opentelemetry import trace as _otel_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    _OTEL_SDK = True
except ImportError:
    _OTEL_SDK = False

# ── Config ────────────────────────────────────────────────────────────────────
# Load .env from the same directory as this file (ignored by git).
# Variables already in the environment take precedence over .env values.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not _api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set.\n"
        "Create coding-agent/.env with:\n"
        "  ANTHROPIC_API_KEY=your-key-here\n"
        "or export the variable before running."
    )

# Must be set BEFORE `from mem0 import ...` because mem0's telemetry module
# snapshots this env var at import time.
os.environ.setdefault("MEM0_TELEMETRY", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")

ROOT = Path(__file__).resolve().parents[1]   # mem0 repo root

# Async cancellation token — set by REPL when user interrupts a running turn.
# Async tools check this; None when running in sync context.
_cancel_ev: Optional[asyncio.Event] = None

# ── OpenTelemetry Tracer ──────────────────────────────────────────────────────

@dataclass
class _Span:
    span_id:    str
    parent_id:  Optional[str]
    name:       str
    trace_id:   str
    start_ns:   int
    end_ns:     int = 0
    status:     str = "OK"
    attributes: dict = field(default_factory=dict)


class AgentTracer:
    """
    OTel-compatible tracer.
    - Uses real opentelemetry-sdk if installed (OTEL_EXPORTER_OTLP_ENDPOINT controls export).
    - Always writes JSONL to ROOT/.traces/traces-YYYY-MM-DD.jsonl for ClickHouse / any OLAP.
    """

    PROMPT_VERSION = "1"   # bump when system prompt template changes

    def __init__(self):
        self._store: dict[str, list[_Span]] = {}
        self._lock  = threading.Lock()
        self._dir   = ROOT / ".traces"
        self._dir.mkdir(exist_ok=True)

        if _OTEL_SDK:
            provider = TracerProvider(resource=Resource({"service.name": "lg-agent"}))
            # Only wire up console exporter when explicitly requested
            if os.getenv("OTEL_CONSOLE"):
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            _otel_api.set_tracer_provider(provider)
            self._otel = _otel_api.get_tracer("lg-agent")
        else:
            self._otel = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start_trace(self) -> str:
        tid = uuid.uuid4().hex
        with self._lock:
            self._store[tid] = []
        return tid

    def start_span(self, trace_id: str, name: str, parent_id: Optional[str] = None) -> str:
        sid  = uuid.uuid4().hex[:16]
        span = _Span(span_id=sid, parent_id=parent_id, name=name,
                     trace_id=trace_id, start_ns=time.time_ns())
        with self._lock:
            self._store.setdefault(trace_id, []).append(span)
        return sid

    def end_span(self, trace_id: str, span_id: str,
                 attributes: Optional[dict] = None, error: Optional[str] = None):
        with self._lock:
            for s in self._store.get(trace_id, []):
                if s.span_id == span_id:
                    s.end_ns = time.time_ns()
                    s.status = "ERROR" if error else "OK"
                    if error:
                        s.attributes["error.message"] = error
                    if attributes:
                        s.attributes.update(attributes)
                    break

    def flush(self, trace_id: str, session_id: str, user_id: str):
        with self._lock:
            spans = self._store.pop(trace_id, [])
        if not spans:
            return

        total_in  = sum(s.attributes.get("input_tokens",  0) for s in spans)
        total_out = sum(s.attributes.get("output_tokens", 0) for s in spans)

        record = {
            "trace_id":       trace_id,
            "session_id":     session_id,
            "user_id":        user_id,
            "service":        "lg-agent",
            "prompt_version": self.PROMPT_VERSION,
            "timestamp_ms":   time.time_ns() // 1_000_000,
            "total_input_tokens":  total_in,
            "total_output_tokens": total_out,
            "spans": [
                {
                    "span_id":     s.span_id,
                    "parent_id":   s.parent_id,
                    "name":        s.name,
                    "start_ms":    s.start_ns  // 1_000_000,
                    "duration_ms": (s.end_ns - s.start_ns) // 1_000_000 if s.end_ns else 0,
                    "status":      s.status,
                    **s.attributes,
                }
                for s in spans
            ],
        }
        path = self._dir / f"traces-{time.strftime('%Y-%m-%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        logger.debug("Trace %s flushed | spans=%d in=%d out=%d",
                     trace_id[:8], len(spans), total_in, total_out)


_tracer = AgentTracer()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]
    session_id:        str
    user_id:           str
    memories:          list
    middleware_ctx:    dict
    observability:     dict
    skill_ctx:         dict
    requires_approval: bool
    approval_context:  str
    auto_approve:      bool
    trace_id:          str   # OTel trace ID for this turn
    root_span_id:      str   # root span of this turn

# ── mem0 Memory (persistent, LLM-managed extraction + dedup) ─────────────────

from mem0 import Memory as _Mem0Memory

def _build_mem0() -> _Mem0Memory:
    config = {
        "llm": {
            "provider": "anthropic",
            "config": {
                "model":   "claude-haiku-4-5-20251001",
                "api_key": os.environ.get("ANTHROPIC_API_KEY", "sk-placeholder"),
                # ANTHROPIC_BASE_URL read automatically by Anthropic Python client
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "all-MiniLM-L6-v2"},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "agent_memory_v1",
                "path": str(ROOT / ".mem0store"),
            },
        },
        "history_db_path": str(ROOT / ".mem0store" / "history.db"),
    }
    return _Mem0Memory.from_config(config)

_mem0: Optional["_Mem0Memory"] = None
_mem0_init_error: Optional[BaseException] = None

def _mem0_init_worker() -> None:
    global _mem0, _mem0_init_error
    try:
        _mem0 = _build_mem0()
    except BaseException as exc:
        _mem0_init_error = exc

_mem0_init_thread = threading.Thread(
    target=_mem0_init_worker, daemon=True, name="mem0-init",
)
_mem0_init_thread.start()

def _get_mem0() -> "_Mem0Memory":
    _mem0_init_thread.join()   # no-op after first call
    if _mem0_init_error is not None:
        raise RuntimeError("mem0 initialisation failed") from _mem0_init_error
    return _mem0  # type: ignore[return-value]

_extraction_pool = _DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="mem-extract")

_user_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="default")


def _get_current_user() -> str:
    return _user_ctx.get()


def _set_current_user(user_id: str) -> None:
    _user_ctx.set(user_id)

# ── Middleware ────────────────────────────────────────────────────────────────

class Middleware:
    def before(self, state: AgentState) -> dict: return {}
    def after (self, state: AgentState) -> dict: return {}


class LoggingMiddleware(Middleware):
    def before(self, state):
        trace_id    = _tracer.start_trace()
        root_span   = _tracer.start_span(trace_id, "turn")
        logger.info("[%s] turn-start | user=%s | history=%d",
                    state["session_id"], state["user_id"], len(state["messages"]))
        return {
            "trace_id":     trace_id,
            "root_span_id": root_span,
            "observability": {
                "start_time":          time.time(),
                "node_timings":        {},
                "errors":              [],
                "total_input_tokens":  0,
                "total_output_tokens": 0,
            },
        }

    def after(self, state):
        obs     = state.get("observability") or {}
        elapsed = time.time() - obs.get("start_time", time.time())
        timings = {k: round(v, 3) for k, v in obs.get("node_timings", {}).items()}
        in_tok  = obs.get("total_input_tokens",  0)
        out_tok = obs.get("total_output_tokens", 0)
        logger.info("[%s] turn-end | %.2fs | %s | tokens in=%d out=%d",
                    state["session_id"], elapsed, json.dumps(timings), in_tok, out_tok)
        _tracer.end_span(
            state.get("trace_id",     ""),
            state.get("root_span_id", ""),
            attributes={
                "session_id":  state.get("session_id", ""),
                "user_id":     state.get("user_id",    ""),
                "duration_ms": round(elapsed * 1000),
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                **{f"node_{k}_ms": round(v * 1000) for k, v in timings.items()},
            },
        )
        return {}


# ── Anthropic model pricing (USD per million tokens, 2025-06) ─────────────────
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8":            {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":          {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-5":          {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":  {"input":  0.25, "output":  1.25},
}
_DEFAULT_MODEL = "claude-sonnet-4-5"


@dataclass
class TenantMetrics:
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_cost_usd:      float = 0.0
    turn_count:          int   = 0
    tool_call_count:     int   = 0
    last_seen:           float = 0.0


class ObservabilityMiddleware(Middleware):
    """
    Multi-tenant token audit + cost estimation (post-turn).
    Writes a structured audit record to ROOT/.traces/audit-YYYY-MM-DD.jsonl.
    Maintains in-memory per-user aggregates accessible via .report(user_id).
    """

    def __init__(self):
        self._metrics: dict[str, TenantMetrics] = {}
        self._lock  = threading.Lock()

    def after(self, state: AgentState) -> dict:
        obs      = state.get("observability") or {}
        user_id  = state.get("user_id",    "default")
        model    = obs.get("model",         _DEFAULT_MODEL)
        in_tok   = obs.get("total_input_tokens",  0)
        out_tok  = obs.get("total_output_tokens", 0)
        pricing  = _MODEL_PRICING.get(model, _MODEL_PRICING[_DEFAULT_MODEL])
        cost_usd = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

        with self._lock:
            m = self._metrics.setdefault(user_id, TenantMetrics())
            m.total_input_tokens  += in_tok
            m.total_output_tokens += out_tok
            m.total_cost_usd      += cost_usd
            m.turn_count          += 1
            m.last_seen            = time.time()

        logger.info(
            "[AUDIT] user=%-12s trace=%s | in=%5d out=%5d | "
            "cost=$%.5f cumulative=$%.4f",
            user_id, state.get("trace_id","")[:8],
            in_tok, out_tok, cost_usd, m.total_cost_usd,
        )

        # Append to audit JSONL
        record = {
            "ts_ms":      time.time_ns() // 1_000_000,
            "trace_id":   state.get("trace_id", ""),
            "session_id": state.get("session_id", ""),
            "user_id":    user_id,
            "model":      model,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
            "cost_usd":      round(cost_usd, 8),
            "turn_number":   m.turn_count,
        }
        path = ROOT / ".traces" / f"audit-{time.strftime('%Y-%m-%d')}.jsonl"
        path.parent.mkdir(exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        return {}

    def report(self, user_id: str) -> dict:
        with self._lock:
            m = self._metrics.get(user_id)
        if not m:
            return {"user_id": user_id, "message": "no data yet"}
        return {
            "user_id":             user_id,
            "turns":               m.turn_count,
            "total_input_tokens":  m.total_input_tokens,
            "total_output_tokens": m.total_output_tokens,
            "total_cost_usd":      round(m.total_cost_usd, 6),
            "avg_cost_per_turn":   round(m.total_cost_usd / max(m.turn_count, 1), 6),
        }

    def all_reports(self) -> list[dict]:
        with self._lock:
            users = list(self._metrics)
        return [self.report(u) for u in users]


class MemoryProcessingMiddleware(Middleware):
    """Post-turn: delegate extraction + dedup to mem0 in a background thread."""

    def after(self, state: AgentState) -> dict:
        messages = list(state.get("messages", []))[-8:]
        if not messages:
            return {}
        uid  = state.get("user_id", "default")
        conv = []
        for m in messages:
            if isinstance(m, HumanMessage):
                conv.append({"role": "user",      "content": m.content if isinstance(m.content, str) else str(m.content)})
            elif isinstance(m, AIMessage):
                c = m.content
                if isinstance(c, list):
                    c = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                conv.append({"role": "assistant", "content": str(c)})
        _extraction_pool.submit(_get_mem0().add, conv, user_id=uid)
        return {}


class RateLimitMiddleware(Middleware):
    def __init__(self, rpm: int = 20):
        self._calls: dict[str, list] = {}
        self._rpm = rpm

    def before(self, state):
        uid  = state["user_id"]
        now  = time.time()
        hist = [t for t in self._calls.get(uid, []) if now - t < 60]
        hist.append(now)
        self._calls[uid] = hist
        if len(hist) > self._rpm:
            raise RuntimeError(f"Rate limit exceeded for '{uid}'")
        return {}


class SkillRouterMiddleware(Middleware):
    _MAP: dict[str, list] = {
        "file":     ["file","read","write","edit","dir","文件","目录","读取","写入","修改","编辑"],
        "code":     ["code","function","debug","程序","代码","bug","实现","函数","脚本"],
        "analysis": ["analyze","summarize","分析","总结","解释","explain","audit"],
    }

    def before(self, state):
        text   = (state["messages"][-1].content if state["messages"] else "").lower()
        active = [s for s, kws in self._MAP.items() if any(k in text for k in kws)]
        if active:
            logger.info("Skills activated: %s", active)
        return {"skill_ctx": {"active_skills": active, "results": {}}}


_observability_mw = ObservabilityMiddleware()   # keep ref for REPL /report command

_MIDDLEWARE: list[Middleware] = [
    LoggingMiddleware(),
    RateLimitMiddleware(rpm=20),
    SkillRouterMiddleware(),
    _observability_mw,          # post-turn: token audit + cost estimation
    MemoryProcessingMiddleware(),  # post-turn: two-phase memory extraction (async)
]

# ── Skill Registry ────────────────────────────────────────────────────────────

_SKILL_PROMPTS: dict[str, str] = {
    "file":     "You have full repo file access. Use edit_file for targeted changes, write_file for new files.",
    "code":     "You are an expert programmer. Write clean, idiomatic, minimal code.",
    "analysis": "You are an expert analyst. Use clear headers, evidence, and structure.",
}

# ── MCP Tool Registry ─────────────────────────────────────────────────────────

class _MCPStdioSession:
    """
    Persistent MCP stdio session (JSON-RPC 2.0 over subprocess stdin/stdout).
    Keeps the subprocess alive; a background reader thread dispatches responses.
    """

    def __init__(self, command: list[str], env: Optional[dict] = None):
        self._cmd  = command
        self._env  = {**os.environ, **(env or {})}
        self._proc: Optional[subprocess.Popen] = None
        self._pending:   dict[int, threading.Event] = {}
        self._responses: dict[int, Any]             = {}
        self._lock    = threading.Lock()
        self._counter = 0

    def start(self):
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=self._env,
        )
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "lg-agent", "version": "1.0"},
        })
        self._notify("notifications/initialized", {})

    def _reader(self):
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = msg.get("id")
            if req_id is not None:
                with self._lock:
                    ev = self._pending.get(req_id)
                    if ev is not None:
                        self._responses[req_id] = msg
                if ev:
                    ev.set()

    def _notify(self, method: str, params: dict):
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        self._proc.stdin.write(msg)
        self._proc.stdin.flush()

    def _rpc(self, method: str, params: dict, timeout: float = 15.0) -> Any:
        with self._lock:
            self._counter += 1
            req_id = self._counter
            ev = threading.Event()
            self._pending[req_id] = ev

        msg = json.dumps({"jsonrpc":"2.0","method":method,"params":params,"id":req_id}) + "\n"
        self._proc.stdin.write(msg)
        self._proc.stdin.flush()

        if not ev.wait(timeout):
            with self._lock:
                self._pending.pop(req_id, None)
                self._responses.pop(req_id, None)  # discard any late-arriving response
            raise TimeoutError(f"MCP {method} timed out after {timeout}s")

        resp = self._responses.pop(req_id)
        with self._lock:
            self._pending.pop(req_id, None)

        if "error" in resp:
            raise RuntimeError(f"MCP error: {resp['error']}")
        return resp.get("result", {})

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts  = [c["text"] for c in result.get("content", [])
                  if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(parts) or json.dumps(result)

    def close(self):
        if self._proc:
            self._proc.terminate()


class _MCPHTTPSession:
    """
    MCP over HTTP Streamable transport (JSON-RPC 2.0 POST).
    Compatible with MCP servers that expose a single HTTP endpoint.
    """

    def __init__(self, url: str, headers: Optional[dict] = None):
        self._url     = url.rstrip("/")
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._counter = 0

    def _rpc(self, method: str, params: dict) -> Any:
        import urllib.request
        self._counter += 1
        payload = json.dumps({
            "jsonrpc": "2.0", "method": method, "params": params, "id": self._counter,
        }).encode()
        req = urllib.request.Request(
            self._url, data=payload, headers=self._headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts  = [c["text"] for c in result.get("content", [])
                  if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(parts) or json.dumps(result)


def _json_type(t: str) -> type:
    return {"string": str, "integer": int, "number": float,
            "boolean": bool, "array": list, "object": dict}.get(t, str)


class MCPToolRegistry:
    """
    Dynamically load tools from MCP servers and expose them as LangChain tools.

    Usage:
        registry = MCPToolRegistry()
        registry.add_stdio("fs",  "npx", "-y", "@modelcontextprotocol/server-filesystem", "./workspace")
        registry.add_http("ext",  "http://localhost:8000/mcp")
        registry.connect()           # starts sessions
        extra_tools = registry.tools # LangChain BaseTool list
    """

    def __init__(self):
        self._configs:  list[dict] = []
        self._sessions: list[tuple[str, Any]] = []
        self._tools:    list[Any]  = []

    def add_stdio(self, name: str, command: str, *args: str,
                  env: Optional[dict] = None) -> "MCPToolRegistry":
        self._configs.append({"name": name, "transport": "stdio",
                               "command": [command, *args], "env": env or {}})
        return self

    def add_http(self, name: str, url: str,
                 headers: Optional[dict] = None) -> "MCPToolRegistry":
        self._configs.append({"name": name, "transport": "http",
                               "url": url, "headers": headers or {}})
        return self

    def connect(self) -> "MCPToolRegistry":
        for cfg in self._configs:
            try:
                if cfg["transport"] == "stdio":
                    sess = _MCPStdioSession(cfg["command"], cfg["env"])
                    sess.start()
                elif cfg["transport"] == "http":
                    sess = _MCPHTTPSession(cfg["url"], cfg["headers"])
                else:
                    continue
                self._sessions.append((cfg["name"], sess))
                logger.info("MCP connected: %s (%s)", cfg["name"], cfg["transport"])
            except Exception as exc:
                logger.error("MCP connect failed (%s): %s", cfg["name"], exc)

        self._tools = []
        for server_name, sess in self._sessions:
            try:
                for schema in sess.list_tools():
                    self._tools.append(self._wrap(server_name, sess, schema))
            except Exception as exc:
                logger.error("MCP list_tools failed (%s): %s", server_name, exc)

        if self._tools:
            logger.info("MCP loaded %d tools from %d servers",
                        len(self._tools), len(self._sessions))
        return self

    @property
    def tools(self) -> list:
        return list(self._tools)

    @staticmethod
    def _wrap(server_name: str, sess: Any, schema: dict):
        tool_name   = schema["name"]
        description = f"[MCP:{server_name}] {schema.get('description', '')}"
        props       = (schema.get("inputSchema") or {}).get("properties", {})
        required    = set((schema.get("inputSchema") or {}).get("required", []))

        if props:
            field_defs: dict[str, Any] = {}
            for fname, fschema in props.items():
                ftype = _json_type(fschema.get("type", "string"))
                fdesc = fschema.get("description", "")
                if fname in required:
                    field_defs[fname] = (ftype, Field(description=fdesc))
                else:
                    field_defs[fname] = (Optional[ftype], Field(default=None, description=fdesc))

            ArgsModel = create_model(f"{tool_name}_Args", **field_defs)

            def _fn(**kwargs) -> str:
                return sess.call_tool(tool_name, {k: v for k, v in kwargs.items() if v is not None})

            _fn.__name__ = tool_name
            return StructuredTool.from_function(
                func=_fn, name=tool_name, description=description, args_schema=ArgsModel,
            )
        else:
            def _fn_noargs() -> str:                      # type: ignore[misc]
                return sess.call_tool(tool_name, {})
            _fn_noargs.__name__ = tool_name
            _fn_noargs.__doc__  = description
            return tool(_fn_noargs)


# ── MCP: CodeGraph — structural code intelligence (call graph, impact, search) ─
# Requires: codegraph CLI installed (https://codegraph.sh) and
#           `codegraph init <repo_root>` run at least once.
# The subprocess is kept alive as a stdio JSON-RPC 2.0 server; no HTTP port needed.
_AGENT_DIR = Path(__file__).resolve().parent   # coding-agent/ — where .codegraph/ lives

mcp_registry = MCPToolRegistry()
mcp_registry.add_stdio(
    "codegraph",
    "codegraph", "serve", "--mcp", "--path", str(_AGENT_DIR),
).connect()

# ── Unified Diff helpers ──────────────────────────────────────────────────────

def _fuzzy_find(orig_lines: list[str], search_lines: list[str],
                threshold: float = 0.82) -> int:
    """
    Return 0-indexed start of the best fuzzy match for search_lines inside
    orig_lines, or -1 if similarity is below threshold.

    Comparison is done on stripped lines so minor indentation drift doesn't
    block the match — the actual replacement always uses the canonical content.
    """
    n = len(search_lines)
    if n == 0 or n > len(orig_lines):
        return -1
    s_stripped = [l.strip() for l in search_lines]
    best_score, best_idx = 0.0, -1
    for i in range(len(orig_lines) - n + 1):
        window = [l.strip() for l in orig_lines[i:i + n]]
        score  = SequenceMatcher(None, s_stripped, window).ratio()
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx if best_score >= threshold else -1


def _parse_diff_hunks(diff: str) -> list[dict]:
    """Parse a unified diff string into a list of hunk dicts."""
    hunks: list[dict] = []
    cur: Optional[dict] = None
    for line in diff.splitlines():
        m = re.match(r'@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@', line)
        if m:
            if cur is not None:
                hunks.append(cur)
            cur = {
                "orig_start": int(m.group(1)),
                "orig_count": int(m.group(2) if m.group(2) is not None else 1),
                "lines": [],
            }
        elif cur is not None and line and line[0] in (' ', '-', '+'):
            cur["lines"].append(line)
    if cur is not None:
        hunks.append(cur)
    return hunks


def _apply_unified_diff(original: str, diff: str) -> tuple[str, int]:
    """
    Apply a unified diff to original content.

    Each hunk's expected block (context + removes) is located via an exact
    position first, then a ±20-line fuzzy search if the line numbers are stale.
    Returns (new_content, hunks_applied).
    """
    result = original.splitlines(keepends=True)
    hunks  = _parse_diff_hunks(diff)
    offset = 0
    applied = 0

    for hunk in hunks:
        start      = hunk["orig_start"] - 1 + offset  # 0-indexed
        hunk_lines = hunk["lines"]

        # expected = lines we must find (context + removes)
        # replacement = lines we put back (context + adds)
        expected    = [l[1:] for l in hunk_lines if l[0] in (' ', '-')]
        replacement = [l[1:] for l in hunk_lines if l[0] in (' ', '+')]
        n = len(expected)

        # Normalise: ensure every replacement line ends with \n
        replacement = [l if l.endswith('\n') else l + '\n' for l in replacement]

        # Locate actual position (exact first, fuzzy fallback within ±20 lines)
        actual: Optional[int] = start if n == 0 else None
        if n > 0:
            window_exact = [l.rstrip('\n') for l in result[start:start + n]]
            expected_cmp = [l.rstrip('\n') for l in expected]
            if window_exact == expected_cmp:
                actual = start
            else:
                lo = max(0, start - 20)
                hi = min(len(result) - n + 1, start + 21)
                best_score, best_i = 0.0, start
                for i in range(lo, hi):
                    w = [l.strip() for l in result[i:i + n]]
                    s = SequenceMatcher(None,
                                       [l.strip() for l in expected], w).ratio()
                    if s > best_score:
                        best_score, best_i = s, i
                if best_score >= 0.75:
                    actual = best_i

        if actual is None:
            # Neither exact nor fuzzy match found — skip this hunk rather than
            # silently applying it at a wrong position and corrupting the file.
            continue

        result[actual:actual + n] = replacement
        offset  += len(replacement) - n
        applied += 1

    return "".join(result), applied


# ── File System & Shell Tools ─────────────────────────────────────────────────

def _safe(path_str: str) -> Optional[Path]:
    candidate = Path(path_str)
    p = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        p.relative_to(ROOT)
        return p
    except ValueError:
        return None


@tool
def read_file(path: str) -> str:
    """Read a file from the repo."""
    p = _safe(path)
    if not p: return "Error: path outside repo root"
    try:    return p.read_text("utf-8")
    except FileNotFoundError: return f"Error: '{path}' not found"
    except Exception as e:    return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Create or fully overwrite a file in the repo."""
    p = _safe(path)
    if not p: return "Error: path outside repo root"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")
    return f"Written '{path}' ({len(content)} chars)"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    Replace old_text with new_text in a file.

    Primary path: exact string match (fastest).
    Fallback: fuzzy line-by-line alignment via SequenceMatcher (threshold 0.82)
    so minor whitespace / indentation drift doesn't cause spurious failures.
    Reports the matched line number when fuzzy alignment is used.
    """
    p = _safe(path)
    if not p: return "Error: path outside repo root"
    try:    original = p.read_text("utf-8")
    except FileNotFoundError: return f"Error: '{path}' not found — use write_file to create it"

    # ── Fast path: exact match ────────────────────────────────────────────────
    if old_text in original:
        p.write_text(original.replace(old_text, new_text, 1), "utf-8")
        return f"Edited '{path}': exact match replaced ({len(old_text)}→{len(new_text)} chars)"

    # ── Fuzzy fallback: line-level SequenceMatcher alignment ─────────────────
    orig_lines   = original.splitlines(keepends=True)
    search_lines = old_text.splitlines(keepends=True)
    idx = _fuzzy_find(orig_lines, search_lines)
    if idx == -1:
        preview = original[:400] + ("…" if len(original) > 400 else "")
        return (f"Error: old_text not found (exact or fuzzy ≥0.82).\n"
                f"File preview:\n{preview}")

    replace_lines = new_text.splitlines(keepends=True)
    result = orig_lines[:idx] + replace_lines + orig_lines[idx + len(search_lines):]
    p.write_text("".join(result), "utf-8")
    return (f"Edited '{path}': fuzzy match at line {idx + 1}, "
            f"replaced {len(search_lines)}→{len(replace_lines)} lines")


@tool
def apply_diff(path: str, unified_diff: str) -> str:
    """
    Apply a unified diff (--- / +++ / @@ format) to a file.

    Accepts standard unified diff output (e.g. from `git diff` or `diff -u`).
    Stale line numbers are tolerated via ±20-line fuzzy context alignment.
    Example diff block:
        --- a/foo.py
        +++ b/foo.py
        @@ -10,4 +10,5 @@
         unchanged context
        -old line
        +new line
         unchanged context
    """
    p = _safe(path)
    if not p: return "Error: path outside repo root"
    try:    original = p.read_text("utf-8")
    except FileNotFoundError: return f"Error: '{path}' not found"

    try:
        new_content, n_hunks = _apply_unified_diff(original, unified_diff)
    except Exception as exc:
        return f"Error applying diff: {exc}"

    if n_hunks == 0:
        return "Error: no valid hunks found in diff"

    p.write_text(new_content, "utf-8")
    delta = len(new_content) - len(original)
    sign  = "+" if delta >= 0 else ""
    return f"Applied {n_hunks} hunk(s) to '{path}' ({sign}{delta} chars)"


@tool
def list_directory(path: str = ".") -> str:
    """List files/dirs at a repo path."""
    p = _safe(path)
    if not p: return "Error: path outside repo root"
    if not p.exists(): return f"Error: '{path}' not found"
    items = sorted(f"[{'dir' if i.is_dir() else 'file'}] {i.name}" for i in p.iterdir())
    return "\n".join(items) or "(empty)"


_SEARCH_EXCLUDE = {".venv", "venv", "node_modules", "__pycache__", ".git",
                   ".mem0store", ".vectorstore", ".codegraph", ".traces", ".idea"}

@tool
def search_workspace(pattern: str) -> str:
    """Glob-search the repo (e.g. '**/*.py', 'mem0/**/*.py')."""
    hits = sorted(
        str(p.relative_to(ROOT))
        for p in ROOT.glob(pattern)
        if not any(part in _SEARCH_EXCLUDE for part in p.parts)
    )
    if len(hits) > 500:
        return "\n".join(hits[:500]) + f"\n… ({len(hits) - 500} more truncated)"
    return "\n".join(hits) or "No matches"


async def _run_live_async(cmd: list[str], cwd: str, timeout: int = 30) -> str:
    """
    Async subprocess runner with real-time stdout streaming and cooperative
    cancellation.  Checks _cancel_ev after every output line so a mid-run
    user interrupt kills the subprocess promptly rather than waiting for it
    to finish naturally.
    """
    print(f"\n{'─'*52}\n$ {' '.join(cmd)}\n{'─'*52}", flush=True)
    lines: list[str] = []
    proc: Optional[asyncio.subprocess.Process] = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            # ── cooperative cancellation check ────────────────────────────
            if _cancel_ev is not None and _cancel_ev.is_set():
                proc.kill()
                lines.append("\n[Cancelled by user]\n")
                break
            line = raw.decode(errors="replace")
            print(line, end="", flush=True)
            lines.append(line)

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()

        print('─'*52, flush=True)
        output = "".join(lines)
        if proc.returncode and proc.returncode != 0:
            output += f"\n[exit code {proc.returncode}]"
        # Truncate huge outputs so they don't blow the LLM context window
        _MAX_OUT = 40_000
        if len(output) > _MAX_OUT:
            output = output[:_MAX_OUT] + f"\n… (output truncated, {len(output) - _MAX_OUT} chars omitted)"
        return output.strip() or "(no output)"

    except asyncio.CancelledError:
        if proc is not None:
            proc.kill()
        raise
    except Exception as exc:
        return f"Error: {exc}"


@tool
async def run_python(path: str, cli_args: str = "") -> str:
    """
    Run a Python file.  Output streams to terminal in real-time and is also
    returned to the agent.  Supports mid-run user interrupt via _cancel_ev.
    """
    p = _safe(path)
    if not p:             return "Error: path outside repo root"
    if not p.exists():    return f"Error: '{path}' not found"
    if p.suffix != ".py": return f"Error: '{path}' is not a .py file"
    import shlex
    cmd = [sys.executable, str(p)] + (shlex.split(cli_args) if cli_args.strip() else [])
    return await _run_live_async(cmd, cwd=str(p.parent))


@tool
async def run_shell(command: str) -> str:
    """
    Run a shell command in the repo root.  Output streams live to terminal.
    Uses shlex.split + shell=False to prevent command injection.
    Supports mid-run user interrupt via _cancel_ev.
    """
    import shlex
    cmd_list = shlex.split(command)
    return await _run_live_async(cmd_list, cwd=str(ROOT))


@tool
def save_memory(content: str) -> str:
    """Persist important information to long-term memory for future sessions."""
    _get_mem0().add(content, user_id=_get_current_user())
    return f"Saved to memory: {content}"


# ── RAG / Codebase Vector Search ─────────────────────────────────────────────
#
# Business scenarios this layer enables:
#
#  1. Large-codebase navigation — repo has 50k+ lines; agent would blow the
#     context window reading everything.  rag_search("payment processing logic")
#     returns the 6 most relevant snippets instead of 200 raw files.
#
#  2. Context-aware code generation — before writing a new endpoint the agent
#     calls rag_search("existing REST endpoint implementation") to retrieve
#     canonical patterns from the same codebase.
#
#  3. Private documentation Q&A — index internal markdown docs / ADRs / API
#     specs; agent answers "what's our rate-limit policy?" without hallucinating.
#
#  4. Legacy code understanding — index an unfamiliar repo, ask "where is
#     database connection error handling?" → returns exact file + line range.
#
#  5. Dependency / API discovery — index third-party library docs; agent finds
#     the right API without the developer needing to know it exists.
#
# Architecture:
#   ChromaDB (local persistent)  ←  SentenceTransformer all-MiniLM-L6-v2
#   Python-aware chunking (class/def boundaries, ~800 chars) with MD5 dedup
#   Metadata per chunk: file_path, start_line, end_line, symbol, lang

_SUPPORTED_EXTS: set[str] = {
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".sh",
}


def _chunk_python(text: str, max_chars: int = 900) -> list[tuple[str, int, int, str]]:
    """
    Split Python source into (chunk_text, start_line, end_line, symbol) tuples.

    Strategy:
      1. Identify top-level `def`/`class` boundaries via regex.
      2. Keep each definition as one chunk when ≤ max_chars.
      3. Definitions that exceed max_chars are sub-split at blank-line
         boundaries (inner methods / paragraphs) and given a numeric suffix.
      4. Non-code preamble (module docstring, imports) becomes chunk 0.
    """
    lines    = text.splitlines()
    total    = len(lines)
    boundary = re.compile(r'^(def |class )\S')

    # Collect start indices of top-level definitions
    starts = [i for i, l in enumerate(lines) if boundary.match(l)]
    starts.append(total)  # sentinel

    chunks: list[tuple[str, int, int, str]] = []

    def _flush(block_lines, s, e, sym):
        body = "\n".join(block_lines)
        if len(body) <= max_chars:
            chunks.append((body, s + 1, e, sym))
        else:
            # Sub-split at blank lines
            sub, sub_start = [], s
            for i, ln in enumerate(block_lines):
                if ln == "" and sub:
                    chunks.append(("\n".join(sub), sub_start + 1, s + i, f"{sym}[{len(chunks)}]"))
                    sub, sub_start = [], s + i + 1
                else:
                    sub.append(ln)
            if sub:
                chunks.append(("\n".join(sub), sub_start + 1, e, f"{sym}[{len(chunks)}]"))

    # Preamble block (imports, module docstring)
    if starts[0] > 0:
        _flush(lines[: starts[0]], 0, starts[0], "<module>")

    for idx in range(len(starts) - 1):
        s, e     = starts[idx], starts[idx + 1]
        sym      = lines[s].split("(")[0].split(":")[0].strip()
        _flush(lines[s:e], s, e, sym)

    return chunks


def _chunk_generic(text: str, max_chars: int = 900,
                   overlap: int = 100) -> list[tuple[str, int, int, str]]:
    """Fixed-size character chunking with line-count tracking for non-Python files."""
    lines   = text.splitlines()
    chunks  = []
    buf, buf_start, buf_len = [], 0, 0
    for i, line in enumerate(lines):
        ll = len(line) + 1
        if buf_len + ll > max_chars and buf:
            chunks.append(("\n".join(buf), buf_start + 1, i, "<chunk>"))
            # Overlap: keep last N chars worth of lines
            kept, kl = [], 0
            for l in reversed(buf):
                if kl + len(l) + 1 > overlap:
                    break
                kept.insert(0, l)
                kl += len(l) + 1
            buf, buf_start, buf_len = kept, i - len(kept), kl
        buf.append(line)
        buf_len += ll
    if buf:
        chunks.append(("\n".join(buf), buf_start + 1, len(lines), "<chunk>"))
    return chunks


class CodebaseRAG:
    """
    Persistent RAG layer backed by ChromaDB + SentenceTransformer embeddings.

    Indexing is incremental: each file's MD5 hash is stored as metadata;
    on re-index, only changed/new files are re-embedded, unchanged files
    are skipped.  This makes `rag_index` safe to call repeatedly (e.g. as
    a pre-turn hook) without performance degradation.
    """

    COLLECTION = "codebase_v1"
    _instance: Optional["CodebaseRAG"] = None

    def __init__(self, persist_dir: Path):
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._ef     = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        self._col    = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── indexing ───────────────────────────────────────────────────────────────

    def _file_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def index_path(
        self,
        path: Path,
        glob_patterns: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Walk `path` and index all matching files.
        Returns {"indexed": N, "skipped": N, "deleted": N}.
        """
        patterns = glob_patterns or ["**/*.py", "**/*.md"]
        candidates: list[Path] = []
        for pat in patterns:
            candidates.extend(path.glob(pat))
        candidates = [p for p in candidates if p.suffix in _SUPPORTED_EXTS and p.is_file()]

        stats = {"indexed": 0, "skipped": 0, "deleted": 0}

        # Fetch existing hashes to detect deletions and skip unchanged files
        existing: dict[str, str] = {}  # doc_id → content_hash
        try:
            res = self._col.get(include=["metadatas"])
            for meta in res["metadatas"]:
                if meta:
                    fid = f"{meta.get('file_path', '')}::{meta.get('content_hash', '')}"
                    existing[meta.get("file_path", "")] = meta.get("content_hash", "")
        except Exception:
            pass

        total = len(candidates)
        logger.info("RAG index: found %d files to scan (%d already in store)",
                    total, len(existing))

        indexed_paths: set[str] = set()
        for n_done, fpath in enumerate(candidates, 1):
            try:
                text = fpath.read_text("utf-8", errors="replace")
            except Exception:
                continue

            rel   = str(fpath.relative_to(ROOT))
            fhash = self._file_hash(text)
            indexed_paths.add(rel)

            if existing.get(rel) == fhash:
                stats["skipped"] += 1
                continue

            try:
                self._col.delete(where={"file_path": rel})
            except Exception:
                pass

            suffix = fpath.suffix
            raw_chunks = (
                _chunk_python(text) if suffix == ".py" else _chunk_generic(text)
            )

            ids, docs, metas = [], [], []
            for i, (body, s_line, e_line, sym) in enumerate(raw_chunks):
                if not body.strip():
                    continue
                doc_id = f"{rel}::{i}"
                ids.append(doc_id)
                docs.append(body)
                metas.append({
                    "file_path":    rel,
                    "start_line":   s_line,
                    "end_line":     e_line,
                    "symbol":       sym,
                    "lang":         suffix.lstrip("."),
                    "content_hash": fhash,
                })

            if ids:
                import contextlib
                with open(os.devnull, "w") as _devnull, \
                     contextlib.redirect_stderr(_devnull):
                    self._col.add(documents=docs, metadatas=metas, ids=ids)
            stats["indexed"] += 1
            logger.info("RAG index: [%d/%d] embedded %s (%d chunks)",
                        n_done, total, rel, len(ids))

        return stats

    # ── search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 6,
        file_filter: str = "",
    ) -> list[dict]:
        """
        Semantic similarity search.
        Returns list of {file_path, start_line, end_line, symbol, score, content}.

        file_filter is applied as a Python-side substring check (case-insensitive)
        after retrieval — more reliable than ChromaDB `$contains` across versions.
        We over-fetch (n * 4) to ensure enough results survive the filter.
        """
        fetch_n = min(n * 4 if file_filter else n, max(self._col.count(), 1))
        try:
            res = self._col.query(
                query_texts=[query],
                n_results=fetch_n,
            )
        except Exception as exc:
            return [{"error": str(exc)}]

        hits = []
        needle = file_filter.lower()
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            fp = meta.get("file_path", "")
            if needle and needle not in fp.lower():
                continue
            hits.append({
                "file_path":  fp,
                "start_line": meta.get("start_line", 0),
                "end_line":   meta.get("end_line",   0),
                "symbol":     meta.get("symbol",     ""),
                "score":      round(1 - dist, 4),
                "content":    doc,
            })
            if len(hits) >= n:
                break
        return hits

    def stats(self) -> dict:
        return {"total_chunks": self._col.count()}


_rag_lock = threading.Lock()

def _get_rag() -> CodebaseRAG:
    if CodebaseRAG._instance is None:
        with _rag_lock:
            if CodebaseRAG._instance is None:  # double-checked locking
                CodebaseRAG._instance = CodebaseRAG(ROOT / ".vectorstore")
    return CodebaseRAG._instance


# ── RAG tools ─────────────────────────────────────────────────────────────────

@tool
def rag_index(path: str = ".", patterns: str = "**/*.py **/*.md") -> str:
    """
    Index a directory into the vector store for semantic search.

    Skips unchanged files (MD5-based), so safe to call repeatedly.
    Use before rag_search on a new or recently-changed codebase.

    Args:
        path:     Directory to index (relative to repo root, default = root).
        patterns: Space-separated glob patterns, e.g. "**/*.py **/*.md **/*.ts"
    """
    p = _safe(path)
    if not p:          return "Error: path outside repo root"
    if not p.is_dir(): return f"Error: '{path}' is not a directory"

    pat_list = patterns.split()
    if CodebaseRAG._instance is None:
        logger.info("RAG: loading embedding model all-MiniLM-L6-v2 (first call, ~20s) ...")
    stats    = _get_rag().index_path(p, pat_list)
    total    = _get_rag().stats()["total_chunks"]
    return (
        f"Indexed '{path}': {stats['indexed']} files embedded, "
        f"{stats['skipped']} unchanged (skipped).  "
        f"Vector store total: {total} chunks."
    )


@tool
def rag_search(query: str, n_results: int = 6, file_filter: str = "") -> str:
    """
    Semantic search over the indexed codebase.

    Returns the most relevant code/doc chunks with file path and line numbers.
    Far more token-efficient than reading whole files when navigating a large repo.

    Args:
        query:       Natural-language description of what you're looking for.
        n_results:   Number of chunks to return (default 6, max 20).
        file_filter: Optional substring to restrict results to matching file paths
                     (e.g. "auth" to only search auth-related files).
    """
    if _get_rag().stats()["total_chunks"] == 0:
        return "Vector store is empty — call rag_index first."

    hits = _get_rag().search(query, n=min(n_results, 20), file_filter=file_filter)
    if not hits:
        return "No results found."
    if "error" in hits[0]:
        return f"Search error: {hits[0]['error']}"

    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        header  = f"[{i}] {h['file_path']}:{h['start_line']}-{h['end_line']}"
        symbol  = f"  ({h['symbol']})" if h['symbol'] not in ("<chunk>", "<module>") else ""
        score   = f"  score={h['score']:.3f}"
        divider = "─" * 60
        preview = h["content"][:600] + ("…" if len(h["content"]) > 600 else "")
        parts.append(f"{header}{symbol}{score}\n{divider}\n{preview}")

    return "\n\n".join(parts)


_BUILTIN_TOOLS = [
    read_file, write_file, edit_file, apply_diff,
    list_directory, search_workspace,
    run_python, run_shell, save_memory,
    rag_index, rag_search,
]
_TOOLS = _BUILTIN_TOOLS + mcp_registry.tools

# ── LLM ───────────────────────────────────────────────────────────────────────

_llm      = init_chat_model("anthropic:claude-sonnet-4-5").bind_tools(_TOOLS)
_tool_node = ToolNode(_TOOLS)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _tick(state: AgentState, key: str, elapsed: float) -> dict:
    obs = dict(state.get("observability") or {})
    obs.setdefault("node_timings", {})[key] = round(elapsed, 4)
    return obs

# ── Nodes ─────────────────────────────────────────────────────────────────────

def middleware_pre(state: AgentState) -> dict:
    _set_current_user(state.get("user_id", "default"))

    updates: dict[str, Any] = {}
    for mw in _MIDDLEWARE:
        try:
            updates = _deep_merge(updates, mw.before(state))
        except Exception as exc:
            logger.error("Middleware %s.before: %s", type(mw).__name__, exc)
            raise

    uid     = _get_current_user()
    results = _get_mem0().get_all(filters={"user_id": uid}, top_k=12)
    updates["memories"] = [r["memory"] for r in results.get("results", [])]
    return updates


def _trim_messages(messages: list, max_chars: int = 400_000) -> list:
    """Drop oldest messages when history would exceed the LLM context budget."""
    msgs = list(messages)
    total = sum(len(str(m.content)) for m in msgs)
    while len(msgs) > 2 and total > max_chars:
        dropped = msgs.pop(0)
        total -= len(str(dropped.content))
    # Never start with an orphan ToolMessage (needs a preceding AIMessage)
    while msgs and isinstance(msgs[0], ToolMessage):
        total -= len(str(msgs[0].content))
        msgs.pop(0)
    return msgs


def agent_node(state: AgentState) -> dict:
    t0       = time.time()
    trace_id = state.get("trace_id", "")
    span_id  = _tracer.start_span(trace_id, "agent", parent_id=state.get("root_span_id"))

    active_skills = (state.get("skill_ctx") or {}).get("active_skills", [])
    memories      = state.get("memories") or []

    sys_lines = [
        "You are a capable AI assistant with persistent memory, file tools, and skills.",
        "Always respond in the same language the user writes in.",
        f"Session: {state.get('session_id')} | User: {state.get('user_id')}",
        f"Repo root: {ROOT}",
        "Prefer edit_file for targeted changes; write_file only for new files.",
    ]
    if memories:
        sys_lines.append("\n## Long-term Memories")
        sys_lines.extend(f"- {m}" for m in memories)
    for skill in active_skills:
        if prompt := _SKILL_PROMPTS.get(skill):
            sys_lines.append(f"\n## {skill.title()} Skill\n{prompt}")

    sys_content = "\n".join(sys_lines)
    prompt_hash = hashlib.md5(sys_content.encode()).hexdigest()[:8]

    response = _llm.invoke(
        [SystemMessage(content=sys_content)] + _trim_messages(list(state["messages"]))
    )

    usage    = (response.response_metadata or {}).get("usage", {})
    in_tok   = usage.get("input_tokens",  0)
    out_tok  = usage.get("output_tokens", 0)

    _tracer.end_span(trace_id, span_id, attributes={
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "prompt_hash":   prompt_hash,
        "model":         "claude-sonnet-4-5",
        "tool_calls":    len(getattr(response, "tool_calls", []) or []),
    })

    obs = dict(state.get("observability") or {})
    obs.setdefault("node_timings", {})["agent"] = round(time.time() - t0, 4)
    obs["total_input_tokens"]  = obs.get("total_input_tokens",  0) + in_tok
    obs["total_output_tokens"] = obs.get("total_output_tokens", 0) + out_tok
    obs["model"]               = "claude-sonnet-4-5"

    requires_approval, approval_ctx = False, ""
    if not state.get("auto_approve"):
        for tc in getattr(response, "tool_calls", []):
            if tc["name"] in ("write_file", "edit_file"):
                requires_approval = True
                approval_ctx      = f"{tc['name']}(path='{tc['args'].get('path','?')}')"
                break

    return {
        "messages":          [response],
        "observability":     obs,
        "requires_approval": requires_approval,
        "approval_context":  approval_ctx,
    }


_APPROVE_ONE = frozenset([
    "yes","y","approve","ok","sure","go","do it",
    "是","好","允许","同意","确认","继续","可以","行","没问题",
])
_APPROVE_ALL = frozenset([
    "yes to all","all","approve all","auto","auto approve",
    "全部同意","全部允许","全部","都同意","都允许","批量同意","一键同意",
])


def human_review_node(state: AgentState) -> dict:
    answer   = interrupt({
        "prompt": (
            f"Approve: {state['approval_context']}?\n"
            "  yes      — this one\n"
            "  全部同意 — all remaining writes\n"
            "  no       — reject"
        ),
    })
    token    = str(answer).strip().lower()
    auto_all = token in _APPROVE_ALL
    approved = auto_all or token in _APPROVE_ONE

    logger.info("Review '%s' → %s%s", state["approval_context"],
                "APPROVED" if approved else "REJECTED",
                " (auto-approve ON)" if auto_all else "")

    if approved:
        return {"requires_approval": False, "auto_approve": auto_all}

    last_ai = state["messages"][-1]
    cancel  = [ToolMessage(content="Cancelled by user", tool_call_id=tc["id"])
               for tc in getattr(last_ai, "tool_calls", [])]
    return {
        "messages": cancel + [HumanMessage(
            content=f"[System] User rejected '{state['approval_context']}'. "
                    "Please respond without that file operation."
        )],
        "requires_approval": False,
    }


async def tools_node(state: AgentState) -> dict:
    t0     = time.time()
    span   = _tracer.start_span(state.get("trace_id",""), "tools",
                                parent_id=state.get("root_span_id"))
    # Must be async: run_shell / run_python are async tools and cannot be
    # invoked via the sync path (StructuredTool raises NotImplementedError).
    _set_current_user(state.get("user_id", "default"))
    result = await _tool_node.ainvoke(state)
    _tracer.end_span(state.get("trace_id",""), span,
                     attributes={"tool_count": len(result.get("messages", []))})
    return {**result, "observability": _tick(state, "tools", time.time() - t0)}


def middleware_post(state: AgentState) -> dict:
    for mw in reversed(_MIDDLEWARE):
        try:
            mw.after(state)
        except Exception as exc:
            logger.error("Middleware %s.after: %s", type(mw).__name__, exc)

    # Flush OTel trace to JSONL
    _tracer.flush(
        state.get("trace_id",     ""),
        state.get("session_id",   ""),
        state.get("user_id",      ""),
    )
    return {}

# ── Routing ───────────────────────────────────────────────────────────────────

def _route_agent(state: AgentState) -> str:
    if state.get("requires_approval"):
        return "human_review"
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "middleware_post"


def _route_after_review(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "agent"

# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("middleware_pre",  middleware_pre)
    g.add_node("agent",           agent_node)
    g.add_node("tools",           tools_node)
    g.add_node("human_review",    human_review_node)
    g.add_node("middleware_post", middleware_post)

    g.add_edge(START, "middleware_pre")
    g.add_edge("middleware_pre", "agent")
    g.add_conditional_edges("agent", _route_agent, {
        "tools":           "tools",
        "human_review":    "human_review",
        "middleware_post": "middleware_post",
    })
    g.add_edge("tools", "agent")
    g.add_conditional_edges("human_review", _route_after_review,
                            {"tools": "tools", "agent": "agent"})
    g.add_edge("middleware_post", END)
    return g


_checkpointer = MemorySaver()
graph         = _build_graph().compile(checkpointer=_checkpointer)

# ── Public Session API ────────────────────────────────────────────────────────

def new_session(user_id: str = "user") -> dict:
    sid = uuid.uuid4().hex[:8]
    return {"configurable": {"thread_id": sid}, "_user_id": user_id, "_sid": sid}


def _blank_state(message: str, session: dict) -> dict:
    return {
        "messages":          [HumanMessage(content=message)],
        "session_id":        session["_sid"],
        "user_id":           session["_user_id"],
        "memories":          [],
        "middleware_ctx":    {},
        "observability":     {"start_time": 0.0, "node_timings": {},
                              "errors": [], "total_input_tokens": 0, "total_output_tokens": 0},
        "skill_ctx":         {"active_skills": [], "results": {}},
        "requires_approval": False,
        "approval_context":  "",
        "auto_approve":      False,
        "trace_id":          "",
        "root_span_id":      "",
    }


def chat(message: str, session: dict) -> str:
    cfg     = session
    current = graph.get_state(cfg)
    if current.next:
        return resume(message, session)

    state_in = (
        {"messages": [HumanMessage(content=message)]}
        if current.values else _blank_state(message, session)
    )
    result  = graph.invoke(state_in, config=cfg)
    current = graph.get_state(cfg)
    if current.next:
        ctx = result.get("approval_context", "unknown")
        return (f"[INTERRUPTED] Awaiting approval for: {ctx}\n"
                "  yes / 全部同意 / no")
    return getattr(result["messages"][-1], "content", str(result["messages"][-1]))


def resume(answer: str, session: dict) -> str:
    result  = graph.invoke(Command(resume=answer), config=session)
    current = graph.get_state(session)
    if current.next:
        ctx = result.get("approval_context", "unknown")
        return (f"[INTERRUPTED] Awaiting approval for: {ctx}\n"
                "  yes / 全部同意 / no")
    return getattr(result["messages"][-1], "content", str(result["messages"][-1]))

# ── Streaming REPL helper ─────────────────────────────────────────────────────

def _print_stream(stream_input: Any, session: dict) -> Optional[str]:
    """Stream graph output token-by-token. Returns approval_context if interrupted."""
    cfg         = session
    tool_active = False

    print("Agent: ", end="", flush=True)
    for chunk, _meta in graph.stream(stream_input, config=cfg, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            text = ""
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                for blk in chunk.content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text += blk.get("text", "")
            if text:
                if tool_active:
                    print("\nAgent: ", end="", flush=True)
                    tool_active = False
                print(text, end="", flush=True)
            for tc in (getattr(chunk, "tool_call_chunks", None) or []):
                if tc.get("name"):
                    print(f"\n\n[→ {tc['name']}]", end="", flush=True)
        elif isinstance(chunk, ToolMessage):
            content   = str(chunk.content)
            preview   = content[:500] + ("…" if len(content) > 500 else "")
            tool_name = getattr(chunk, "name", None) or "tool"
            print(f"\n[← {tool_name}]\n{preview}", flush=True)
            tool_active = True

    print()
    state = graph.get_state(cfg)
    if state.next:
        return state.values.get("approval_context", "unknown")
    return None

# ── Async streaming helper ────────────────────────────────────────────────────

async def _aprint_stream(stream_input: Any, session: dict) -> Optional[str]:
    """
    Async version of _print_stream.  Drives graph.astream() and prints tokens
    as they arrive.  Returns approval_context string if an interrupt node fires,
    None on normal completion.
    """
    cfg         = session
    tool_active = False

    print("Agent: ", end="", flush=True)
    async for chunk, _meta in graph.astream(
        stream_input, config=cfg, stream_mode="messages"
    ):
        if isinstance(chunk, AIMessageChunk):
            text = ""
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                for blk in chunk.content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text += blk.get("text", "")
            if text:
                if tool_active:
                    print("\nAgent: ", end="", flush=True)
                    tool_active = False
                print(text, end="", flush=True)
            for tc in (getattr(chunk, "tool_call_chunks", None) or []):
                if tc.get("name"):
                    print(f"\n\n[→ {tc['name']}]", end="", flush=True)
        elif isinstance(chunk, ToolMessage):
            content   = str(chunk.content)
            preview   = content[:500] + ("…" if len(content) > 500 else "")
            tool_name = getattr(chunk, "name", None) or "tool"
            print(f"\n[← {tool_name}]\n{preview}", flush=True)
            tool_active = True

    print()
    state = graph.get_state(cfg)
    if state.next:
        return state.values.get("approval_context", "unknown")
    return None


# ── REPL ──────────────────────────────────────────────────────────────────────

def _print_report() -> None:
    reports = _observability_mw.all_reports()
    if not reports:
        print("[报告] 暂无数据\n")
        return
    print("\n[租户费用报告]")
    for r in reports:
        print(f"  user={r['user_id']:12s} turns={r['turns']:4d} "
              f"in={r['total_input_tokens']:7d} out={r['total_output_tokens']:7d} "
              f"cost=${r['total_cost_usd']:.5f} avg=${r['avg_cost_per_turn']:.5f}/turn")
    print()


async def async_main() -> None:
    """
    Full-duplex async REPL.

    Architecture
    ────────────
    A single background coroutine (_stdin_loop) reads lines from stdin into
    an asyncio.Queue.  The main loop pulls from that queue to start agent
    turns.  While a turn is executing, the main loop polls the queue every
    50 ms.  If the user types anything mid-turn:

      1. _cancel_ev is set → async shell tools kill their subprocess and
         return "[Cancelled by user]" to the LLM.
      2. The agent asyncio.Task is cancelled (CancelledError propagates up
         through graph.astream).
      3. The typed line is re-queued so it becomes the next agent input,
         letting the user redirect without losing their message.

    This gives true preemptive interrupt without busy-waiting threads.
    """
    global _cancel_ev
    _cancel_ev = asyncio.Event()

    # Replace asyncio's default executor with our daemon-thread variant so that
    # asyncio.to_thread() calls (stdin reader) don't block shutdown_default_executor()
    # when Ctrl-C is pressed while the thread is blocked on sys.stdin.readline().
    asyncio.get_running_loop().set_default_executor(
        _DaemonThreadPoolExecutor(thread_name_prefix="asyncio")
    )

    default_user = os.environ.get("AGENT_USER", "user")
    session = new_session(default_user)
    print("LangGraph Industrial Agent  [async full-duplex]")
    print("Commands : quit | new-session:<user_id> | report")
    print("Interrupt: type anything while agent is running")
    print("Approval : yes | 全部同意 | no")
    print("-" * 52)
    print(f"Session  : {session['_sid']} (user: {default_user})\n")

    # ── stdin reader ──────────────────────────────────────────────────────────
    input_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _stdin_loop() -> None:
        """Continuously forward stdin lines into input_queue (None = EOF)."""
        while True:
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
                await input_queue.put(line.rstrip("\n") if line else None)
                if not line:   # EOF
                    return
            except Exception:
                await input_queue.put(None)
                return

    stdin_task = asyncio.create_task(_stdin_loop())

    try:
        while True:
            # ── wait for the next user message ────────────────────────────
            print("You: ", end="", flush=True)
            raw = await input_queue.get()
            if raw is None:           # EOF
                break
            line = raw.strip()
            if not line:
                continue
            if line == "quit":
                break
            if line in ("report", "/report"):
                _print_report()
                continue
            if line.startswith("new-session:"):
                uid     = line.split(":", 1)[1].strip() or "user"
                session = new_session(uid)
                print(f"[New session {session['_sid']} for {uid}]\n")
                continue

            # ── build graph input ─────────────────────────────────────────
            _cancel_ev.clear()
            cfg     = session
            current = graph.get_state(cfg)
            stream_input: Any = (
                Command(resume=line) if current.next else
                ({"messages": [HumanMessage(content=line)]}
                 if current.values else _blank_state(line, session))
            )

            # ── launch agent turn as a cancellable task ───────────────────
            agent_task: asyncio.Task = asyncio.create_task(
                _aprint_stream(stream_input, session)
            )

            # ── concurrent interrupt monitor (50 ms polling) ──────────────
            interrupt_line: Optional[str] = None
            while not agent_task.done():
                await asyncio.sleep(0.05)
                try:
                    msg = input_queue.get_nowait()
                    if msg is None:           # EOF while running
                        agent_task.cancel()
                        break
                    # User typed mid-turn → preemptive cancel
                    print(f"\n\n[⚡ 中断 → '{msg}']", flush=True)
                    _cancel_ev.set()
                    agent_task.cancel()
                    interrupt_line = msg
                    break
                except asyncio.QueueEmpty:
                    pass

            # ── collect result ────────────────────────────────────────────
            interrupted_ctx: Optional[str] = None
            try:
                interrupted_ctx = await agent_task
            except asyncio.CancelledError:
                print("[Turn cancelled]\n")

            if interrupted_ctx:
                print(f"\n[需要审批: {interrupted_ctx}]")
                print("  yes / 全部同意 / no")

            # Re-queue the interrupt so it becomes the next agent input
            if interrupt_line:
                await input_queue.put(interrupt_line)

            print()

    finally:
        stdin_task.cancel()
        try:
            await stdin_task
        except (asyncio.CancelledError, Exception):
            pass
        # _DaemonThreadPoolExecutor.shutdown() is non-blocking; daemon threads
        # are killed on interpreter exit without blocking any atexit handlers.
        _extraction_pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    asyncio.run(async_main())
