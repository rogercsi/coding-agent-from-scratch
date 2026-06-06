# LangGraph Industrial-Grade Coding Agent

A production-ready AI coding assistant built on LangGraph with persistent memory,
full-duplex async REPL, RAG-based codebase search, MCP tool integration, and
OpenTelemetry observability.

## Features

- **Persistent long-term memory** — mem0 extracts and deduplicates facts from every conversation; recalled automatically on the next session
- **Codebase RAG** — ChromaDB + all-MiniLM-L6-v2 semantic search over your repo; indexes incrementally (MD5 dedup, only changed files re-embedded)
- **Full-duplex async REPL** — type to interrupt a running agent turn mid-stream; cancelled turn's message is re-queued so you never lose input
- **File & shell tools** — read/write/edit files, apply unified diffs, run Python/shell with live streaming output
- **MCP integration** — plug in any MCP server (stdio or HTTP) as LangChain tools with zero manual wrapper code; ships with [CodeGraph](https://codegraph.sh) for structural code intelligence
- **Human-in-the-loop approval** — `write_file` / `edit_file` require confirmation unless you grant auto-approve
- **OpenTelemetry tracing** — every turn emits a full span tree to `.traces/traces-YYYY-MM-DD.jsonl` (OTLP export optional); per-user token audit written to `.traces/audit-YYYY-MM-DD.jsonl`
- **Skill router** — intent-based prompt injection (file / code / analysis skills) without hardcoded routing logic

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | Tested on 3.12 |
| [uv](https://docs.astral.sh/uv/) | any | Recommended package manager |
| [codegraph CLI](https://codegraph.sh) | any | Structural code indexing; optional but recommended |
| Anthropic API key | — | `claude-sonnet-4-5` by default |

## Installation

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd assignment1-basics

# 2. Create a virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r coding-agent/requirements.txt

# 3. Configure environment (see Configuration below)
cp coding-agent/.env.example coding-agent/.env
# edit coding-agent/.env and set ANTHROPIC_API_KEY

# 4. (Optional) Index the repo with CodeGraph for structural code intelligence
cd coding-agent
codegraph init .
codegraph index .
```

## Configuration

Copy `.env.example` to `.env` and set the values:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional: use a different user identity (memories are scoped per user)
AGENT_USER=alice

# Optional: point to a custom Anthropic-compatible proxy
# ANTHROPIC_BASE_URL=https://your-proxy.example.com/anthropic
```

Then load the `.env` before running (or export the variables directly):

```bash
# Option A: use python-dotenv or direnv
# Option B: export manually
export $(cat coding-agent/.env | grep -v '^#' | xargs)

# Option C: pass inline
ANTHROPIC_API_KEY=sk-ant-... uv run python coding-agent/coding_agent.py
```

## Usage

```bash
uv run python coding-agent/coding_agent.py
```

### REPL commands

| Input | Effect |
|---|---|
| Any text | Send a message to the agent |
| `quit` | Exit the session |
| `new-session:<user_id>` | Start a new session under a different user (e.g. `new-session:bob`) |
| `report` | Print per-user token usage and cost summary |
| Type anything mid-turn | Interrupt the running agent turn; message is preserved and sent next |

### Approval flow

When the agent wants to edit or create a file it pauses for approval:

```
[需要审批: edit_file(path='src/main.py')]
  yes / 全部同意 / no
```

- `yes` / `y` — approve this one operation
- `全部同意` / `all` — approve all remaining file writes in this turn
- `no` — reject; agent is asked to respond without the file operation

## Architecture

```
stdin ──asyncio.Queue──► REPL loop
                              │
                    ┌─────────▼──────────┐
                    │   LangGraph graph   │
                    │                     │
                    │  middleware_pre      │  ← mem0 recall + skill router
                    │       │             │
                    │    agent_node        │  ← LLM (claude-sonnet-4-5)
                    │       │             │
                    │   ┌───▼────┐        │
                    │   │ tools  │        │  ← read/write/edit/RAG/shell/MCP
                    │   └───┬────┘        │
                    │       │             │
                    │  human_review?      │  ← interrupt() for file ops
                    │       │             │
                    │  middleware_post     │  ← mem0 extract + OTel flush
                    └─────────────────────┘

Memory layers
  Short-term : LangGraph MemorySaver (in-process, session-scoped)
  Long-term  : mem0 + ChromaDB (disk-persistent, user-scoped)

Tracing
  .traces/traces-YYYY-MM-DD.jsonl   per-turn span tree
  .traces/audit-YYYY-MM-DD.jsonl    per-turn token / cost audit
```

## Tools available to the agent

| Tool | Description |
|---|---|
| `read_file` | Read any file under the repo root |
| `write_file` | Create or overwrite a file (requires approval) |
| `edit_file` | Replace a text block in a file; fuzzy-matches minor whitespace drift (requires approval) |
| `apply_diff` | Apply a unified diff to a file; tolerates stale line numbers |
| `list_directory` | List files and directories |
| `search_workspace` | Glob-search the repo |
| `run_python` | Run a Python file with live streaming output |
| `run_shell` | Run a shell command at repo root (shlex-split, no injection) |
| `save_memory` | Explicitly persist a fact to long-term memory |
| `rag_index` | Index a directory into the vector store (incremental) |
| `rag_search` | Semantic search over the indexed codebase |
| `codegraph_*` | 8 structural code tools via CodeGraph MCP (search, callers, callees, impact, explore, …) |

## Extending

### Add a new tool

```python
from langchain_core.tools import tool

@tool
def my_tool(arg: str) -> str:
    """Description shown to the LLM."""
    ...

# Add to _BUILTIN_TOOLS list
_BUILTIN_TOOLS = [..., my_tool]
```

### Add an MCP server

```python
mcp_registry.add_stdio("myserver", "npx", "-y", "@my/mcp-server")
# or HTTP:
mcp_registry.add_http("myserver", "http://localhost:8000/mcp")
```

### Add a skill

```python
_SKILL_PROMPTS["myskill"] = "You are an expert at ..."

# Add trigger keywords to SkillRouterMiddleware._MAP:
_MAP = {
    ...,
    "myskill": ["keyword1", "keyword2"],
}
```

## Running tests

```bash
# Unit tests (pure logic, no network)
uv run pytest coding-agent/test_coding_agent.py -v

# End-to-end CodeGraph MCP tests (requires codegraph CLI + initialized index)
uv run pytest coding-agent/test_codegraph_e2e.py -v
```

## Data directories

These are created automatically and excluded from git:

| Path | Contents |
|---|---|
| `.mem0store/` | mem0 long-term memory (ChromaDB + SQLite) |
| `.vectorstore/` | RAG codebase index (ChromaDB) |
| `.traces/` | OTel JSONL traces and audit logs |

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✓ | — | Anthropic API key |
| `ANTHROPIC_BASE_URL` | | Anthropic default | Custom API endpoint / proxy |
| `AGENT_USER` | | `user` | Default user identity for memory scoping |
| `MEM0_TELEMETRY` | | `false` | Set to `true` to enable mem0 PostHog telemetry |
| `OTEL_CONSOLE` | | — | Set to any value to print OTel spans to stdout |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | | — | OTLP collector endpoint for span export |
