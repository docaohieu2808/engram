# engram

Memory traces for AI agents - Think like human.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-active-brightgreen)

Dual-memory brain combining episodic (vector) + semantic (graph) memory with LLM-powered reasoning. Exposes CLI, MCP server, and HTTP API for agent integration.

## Architecture

```
CLI / MCP / HTTP API
        │
  ┌─────┴──────┐
  │  Reasoning  │ ← LLM synthesis (Gemini)
  │   Engine    │
  ├──────┬──────┤
  │Episodic│Semantic│
  │(ChromaDB)│(SQLite+NetworkX)│
  └────────┴────────┘
       │              │
  Vector search   Knowledge graph
  (embeddings)    (entities + relations)
```

- **Episodic store**: ChromaDB vector DB, semantic similarity search over timestamped memories
- **Semantic graph**: SQLite-backed NetworkX graph, typed entities and relationships
- **Reasoning engine**: Combines both stores, feeds context to LLM for synthesis

## Install

```bash
# Install from source
pip install -e .

# Dev setup with extras
pip install -e ".[dev]"
```

Requires Python 3.11+. Set `GEMINI_API_KEY` for LLM reasoning and Gemini embeddings.

## Quick Start

```bash
# 1. Store a memory
engram remember "Deployed v2.1 to production at 14:00 - caused 503 spike"

# 2. Retrieve similar memories
engram recall "production incidents"

# 3. Reason across all memory
engram think "What deployment issues have we had?"

# 4. Add knowledge to graph
engram add node "PostgreSQL" --type Technology
engram add edge "Service:API" "Technology:PostgreSQL" --relation uses

# 5. Watch inbox for auto-ingest
engram watch --daemon
```

## CLI Reference

### Episodic Memory

```bash
engram remember <content> [--type fact|decision|preference|todo|error|context|workflow] [--priority 1-10] [--tags tag1,tag2] [--expires 2h|1d|7d]
engram recall <query> [--limit 5] [--type <type>] [--tags tag1,tag2] [--namespace <ns>]
```

Examples:
```bash
engram remember "Switched to Redis for session storage" --type decision --priority 7
engram remember "Fix auth bug by Monday" --type todo --tags urgent,auth --expires 3d
engram recall "database" --limit 10 --type error
engram recall "auth issues" --tags urgent --namespace work
```

### Semantic Graph

```bash
engram add node <name> --type <NodeType>
engram add edge <from_key> <to_key> --relation <relation>
engram remove node <key>
engram remove edge <from_key> <to_key> --relation <relation>
engram query [<keyword>] [--type <NodeType>] [--related-to <name>] [--format table|json]
```

Examples:
```bash
engram add node "API-Service" --type Service
engram add edge "Service:API-Service" "Technology:PostgreSQL" --relation uses
engram query --related-to "PostgreSQL"
engram query --type Service --format json
```

### Reasoning

```bash
engram think <question>
engram summarize [--count 20] [--save]
```

Examples:
```bash
engram think "Which services depend on PostgreSQL and have had recent errors?"
engram summarize --count 30 --save
```

### Maintenance

```bash
engram cleanup                          # Delete all expired memories
engram summarize [--count 20] [--save]  # Summarize recent N memories via LLM
```

### Ingest

```bash
engram ingest <file.json> [--dry-run]   # Ingest chat JSON: extract entities + remember context
engram migrate <export.json>            # Import from old agent-memory/neural-memory exports
```

### Schema

```bash
engram schema show                      # Show current schema (node types, relation types)
engram schema init                      # Initialize schema from built-in template
engram schema validate <file.yaml>      # Validate a custom schema file
```

### Config

```bash
engram config show                      # Print full config
engram config get <key>                 # Get single value (dot notation)
engram config set <key> <value>         # Set value (dot notation)
```

Examples:
```bash
engram config get llm.model
engram config set llm.model gemini/gemini-2.0-flash
engram config set hooks.on_remember http://localhost:9000/webhook
```

### System

```bash
engram status                           # Memory stats (episodic count, node/edge count)
engram dump                             # Export all memory data to JSON
engram watch [--daemon]                 # Watch inbox for chat files, auto-ingest on arrival
engram serve [--host 127.0.0.1] [--port 8765]  # Start HTTP API server
```

## Configuration

Config file: `~/.engram/config.yaml`

```yaml
episodic:
  provider: chromadb
  path: ~/.engram/episodic
  namespace: default               # Collection namespace (isolates memory sets)

embedding:
  provider: gemini
  model: gemini-embedding-001      # 3072 dimensions

semantic:
  provider: sqlite
  path: ~/.engram/semantic.db
  schema: devops                   # Built-in schema template

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}       # Expanded from environment

capture:
  enabled: true
  inbox: ~/.engram/inbox/
  poll_interval: 5                 # seconds

serve:
  host: 127.0.0.1
  port: 8765

hooks:
  on_remember: null                # POST {id, content, memory_type} after each remember()
  on_think: null                   # POST {question, answer} after each think()
```

Environment variables in `${VAR}` format are expanded at load time. No API key required for basic storage; reasoning and Gemini embeddings require `GEMINI_API_KEY`.

### Webhooks

Set `hooks.on_remember` or `hooks.on_think` to any HTTP URL. Engram will fire a POST request (JSON body) after each operation — fire-and-forget, never blocks the main operation.

```yaml
hooks:
  on_remember: http://localhost:9000/on-memory
  on_think: http://localhost:9000/on-think
```

Payloads:
- `on_remember`: `{"id": "...", "content": "...", "memory_type": "fact"}`
- `on_think`: `{"question": "...", "answer": "..."}`

### Namespaces

Use namespaces to isolate memory collections (e.g. per project, per user):

```bash
engram remember "prod DB is PostgreSQL 15" --namespace work
engram recall "database" --namespace work
```

Or set the default namespace in config:
```yaml
episodic:
  namespace: myproject
```

## MCP Setup

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/.venv/bin/engram-mcp",
      "env": {
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```

Available MCP tools:

| Tool | Description |
|------|-------------|
| `engram_remember` | Store memory with type, priority, tags, and optional namespace |
| `engram_recall` | Search episodic memories by similarity with optional tag/type/namespace filters |
| `engram_think` | Reason across episodic + semantic memory via LLM |
| `engram_summarize` | Summarize recent N memories into key insights via LLM |
| `engram_cleanup` | Delete all expired memories from the episodic store |
| `engram_status` | Show memory statistics |
| `engram_add_entity` | Add entity node to knowledge graph |
| `engram_add_relation` | Add relationship edge between entities |
| `engram_query_graph` | Query knowledge graph by keyword, type, or relatedness |
| `engram_ingest` | Dual ingest: extract entities + store memories from messages |

## HTTP API

Start server: `engram serve`  Default: `http://127.0.0.1:8765`

| Method | Endpoint | Body / Params | Description |
|--------|----------|---------------|-------------|
| `POST` | `/remember` | `{content, memory_type, priority, entities, tags}` | Store episodic memory |
| `POST` | `/think` | `{question}` | LLM reasoning over all memory |
| `GET` | `/recall` | `?query=&limit=5&offset=0&memory_type=&tags=` | Search episodic memories |
| `GET` | `/query` | `?keyword=&node_type=&related_to=&offset=0&limit=50` | Query semantic graph |
| `POST` | `/ingest` | `{messages: [...]}` | Dual ingest from message array |
| `POST` | `/cleanup` | — | Delete expired memories, returns `{deleted: N}` |
| `POST` | `/summarize` | `{count: 20, save: false}` | Summarize recent memories via LLM |
| `GET` | `/status` | — | Memory statistics |
| `GET` | `/health` | — | Liveness check |

Pagination on `/recall` and `/query`: use `offset` and `limit` query params. Response includes `total`, `offset`, `limit` fields.

## Embeddings

Two embedding modes depending on API key availability:

| Mode | Model | Dimensions | Requires |
|------|-------|-----------|---------|
| Gemini (default) | `gemini-embedding-001` | 3072 | `GEMINI_API_KEY` |
| Fallback | `all-MiniLM-L6-v2` (ChromaDB default) | 384 | nothing |

Embedding dimensions must remain consistent within a collection. If you switch embedding providers, reinitialize the episodic store or create a new collection.

## License

MIT
