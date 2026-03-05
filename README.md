# engram

**Memory traces for AI agents — Think like humans.**

[![PyPI](https://img.shields.io/pypi/v/engram-mem)](https://pypi.org/project/engram-mem/) ![Tests](https://img.shields.io/badge/tests-894%2B-brightgreen) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Dual-memory AI system combining **episodic (vector)** + **semantic (graph)** memory with LLM reasoning. Entity-gated ingestion ensures only meaningful data is stored. Enterprise-ready with multi-tenancy, auth, caching, observability, and Docker deployment.

Works with **any AI agent or IDE** — Claude Code, OpenClaw, Cursor, and any MCP-compatible client. Federates with external knowledge systems (mem0, LightRAG, Graphiti) via auto-discovery. Exposes **CLI**, **MCP** (stdio), **HTTP API** (`/api/v1/`), and **WebSocket** (`/ws`) interfaces.

```bash
pip install engram-mem
```

---

## Features

### Core Memory

- **Episodic Memory** — Qdrant vector store (embedded or server), semantic similarity search, Ebbinghaus decay, activation-based scoring, topic-key upsert
- **Semantic Graph** — NetworkX MultiDiGraph, typed entities and relationships, SQLite (default) or PostgreSQL backend, weighted edges
- **Reasoning Engine** — LLM synthesis (Gemini via litellm), dual-memory context fusion, constitution-guarded prompts
- **Recall Pipeline** — Query decision, temporal+pronoun entity resolution, parallel multi-source search, dedup, composite scoring
- **Entity-Gated Ingestion** — Only stores messages with extracted entities; skips noise (system prompts, trivial messages)
- **Auto Memory** — Detect and persist save-worthy messages automatically, poisoning guard for injection prevention
- **Meeting Ledger** — Structured meeting records with decisions, action items, attendees, topics
- **Feedback Loop** — Confidence scoring (+0.15/-0.2), importance adjustment, auto-delete on 3x negative feedback
- **Graph Visualization** — Interactive entity relationship explorer with dark theme, search, click-to-inspect (vis-network)

### Intelligence Layer

- **Temporal Resolution** — 28 Vietnamese+English date patterns resolve "hom nay/yesterday" to ISO dates before storing
- **Pronoun Resolution** — "anh ay/he/she" to named entity from graph context, LLM-based fallback
- **Fusion Formatter** — Group recall results by type `[preference]`/`[fact]`/`[lesson]` for structured LLM context
- **Memory Consolidation** — Jaccard clustering + LLM summarization reduces redundancy

### Multi-Agent & Federated Knowledge

- **Agent Support** — Claude Code, OpenClaw, Cursor, any MCP-compatible agent or IDE
- **Session Capture** — Real-time JSONL session watchers for OpenClaw + Claude Code (inotify/watchdog)
- **Federated Search** — Query mem0, LightRAG, Graphiti, custom REST/File/Postgres/MCP providers in parallel
- **Auto-Discovery** — Scans local ports, file paths, and MCP configs (`~/.claude/`, `~/.cursor/`) to find providers
- **Provider Adapters** — REST (with JWT auto-login), File (glob patterns), PostgreSQL (custom SQL), MCP (stdio)

### Enterprise

- **Multi-Surface** — CLI (Typer), MCP Server (stdio), HTTP API (FastAPI), WebSocket, Web UI
- **Authentication** — JWT + API keys with RBAC (ADMIN, AGENT, READER), optional, disabled by default
- **Multi-Tenancy** — Isolated per-tenant stores, contextvar propagation, row-level PostgreSQL isolation
- **Caching** — Redis-backed result caching with per-endpoint TTLs
- **Rate Limiting** — Sliding-window per-tenant limits, `fail_open` option
- **Audit Trail** — Structured before/after JSONL log for every episodic mutation
- **Resource Tiers** — 4-tier LLM degradation (FULL > STANDARD > BASIC > READONLY), 60s auto-recovery
- **Data Constitution** — 3-law LLM governance (namespace isolation, no fabrication, audit rights), SHA-256 tamper detection
- **Consolidation Scheduler** — Asyncio background tasks (cleanup daily, consolidate 6h, decay daily), tier-aware
- **Key Rotation** — Failover/round-robin for embedding API keys (GEMINI_API_KEY + GEMINI_API_KEY_FALLBACK)
- **Observability** — OpenTelemetry + JSONL audit logging (optional)
- **Deployment** — Docker Compose, Kubernetes-ready, health checks
- **Backup/Restore** — Memory snapshots, point-in-time recovery
- **Benchmark Suite** — p50/p95/p99 latency measurements for all endpoints

---

## Architecture

```mermaid
flowchart TD
    subgraph Agents["Agents & IDEs"]
        CC["Claude Code"]
        OC["OpenClaw"]
        CU["Cursor"]
        ANY["Any MCP Client"]
    end

    subgraph Interfaces
        CLI["CLI (Typer)"]
        MCP["MCP (stdio)"]
        HTTP["HTTP API /api/v1/"]
        WS["WebSocket /ws"]
    end

    CC & OC & CU & ANY --> MCP
    CLI & MCP & HTTP & WS --> Auth["Auth Middleware\n(JWT + RBAC, optional)"]
    Auth --> Tenant["TenantContext (ContextVar)"]
    Tenant --> Recall["Recall Pipeline\n(decision > resolve > search > feedback)"]
    Recall --> Episodic["EpisodicStore\n(Qdrant)"]
    Recall --> Semantic["SemanticGraph\n(NetworkX + SQLite/PG)"]
    Recall --> Fed["Federated Providers"]
    Episodic & Semantic --> Reasoning["Reasoning Engine\n(Gemini via litellm)"]
    Episodic --> Cache["Redis Cache (optional)"]
    WS --> EventBus["Event Bus\n(push events)"]

    subgraph Fed["Federated Knowledge"]
        M0["mem0"]
        LR["LightRAG"]
        GR["Graphiti"]
        REST["REST / File / PG / MCP"]
    end
```

---

## Quick Start

```bash
# Install from PyPI
pip install engram-mem

# Or from source
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram && pip install -e .

# Initialize config
engram init

# Set API key
export GEMINI_API_KEY="your-key"

# Start daemon (background HTTP server + watcher)
engram start

# Store a memory
engram remember "Deployed v2.1 to production at 14:00 - caused 503 spike"

# Search memories
engram recall "production incidents"

# Browse all data (episodic + semantic)
engram dump

# Reason across all memory
engram think "What deployment issues have we had?"
```

**Requirements:** Python 3.11+, `GEMINI_API_KEY` for LLM reasoning and embeddings. Basic storage works without it.

---

## Integrations

### Claude Code (MCP)

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "env": { "GEMINI_API_KEY": "your-key" }
    }
  }
}
```

### Cursor (MCP)

Add to Cursor's MCP settings — engram auto-discovers Cursor's config at `~/.cursor/settings.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "env": { "GEMINI_API_KEY": "your-key" }
    }
  }
}
```

### OpenClaw

Install the engram skill, then enable session watcher in `~/.engram/config.yaml`:

```yaml
capture:
  openclaw:
    enabled: true
    sessions_dir: ~/.openclaw/workspace/sessions
```

### Federated Knowledge Providers

Engram auto-discovers and federates with external memory systems. Supported providers:

| Provider | Type | Auto-Discovery |
|----------|------|----------------|
| **mem0** | REST | Port 8080, `/v1/memories` |
| **LightRAG** | REST | Port 9520, `/query` |
| **Graphiti** | REST | Port 8000, `/search` |
| **OpenClaw** | File | `~/.openclaw/workspace/memory/*.md` |
| **Custom REST** | REST | Manual config |
| **PostgreSQL** | SQL | Manual config |
| **MCP servers** | MCP | Scans `~/.claude/settings.json`, `~/.cursor/settings.json` |

```yaml
# Auto-discovery (enabled by default)
discovery:
  local: true
  hosts: ["10.10.0.2"]  # additional hosts to scan

# Or manual provider config
providers:
  - name: my-mem0
    type: rest
    url: http://localhost:8080
    search_endpoint: /v1/memories/search
    search_method: POST
    search_body: '{"query": "{query}", "limit": {limit}}'
    result_path: "results[].memory"
```

### HTTP API

```bash
# Start server
engram serve --port 8765

# Store memory
curl -X POST http://localhost:8765/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Deployed v1.0", "memory_type": "fact", "priority": 8}'

# Search
curl "http://localhost:8765/api/v1/recall?query=deployment&limit=5"

# Reason
curl -X POST http://localhost:8765/api/v1/think \
  -H "Content-Type: application/json" \
  -d '{"question": "What deployment issues have we had?"}'

# Meeting ledger
curl -X POST http://localhost:8765/api/v1/meeting-ledger \
  -H "Content-Type: application/json" \
  -d '{"title": "Sprint Review", "decisions": ["Ship v2"], "action_items": ["Update docs"]}'
```

---

## CLI Reference

### Memory Operations

```bash
# Store with options
engram remember <content> [--type fact|decision|preference|todo|error|context|workflow|meeting_ledger]
                          [--priority 1-10] [--tags tag1,tag2] [--expires 2h|1d|7d]
                          [--topic-key unique-key]

# Search
engram recall <query> [--limit 5] [--type <type>] [--tags tag1,tag2]
              [--resolve-entities] [--resolve-temporal]

# Smart query (auto-routes to recall or think)
engram ask <question>

# Reason across all memory
engram think <question>
engram summarize [--count 20] [--save]
```

### Semantic Graph

```bash
engram add node <name> --type <NodeType>
engram add edge <from_key> <to_key> --relation <relation>
engram remove node <key>
engram query [<keyword>] [--type <NodeType>] [--related-to <name>] [--format table|json]
engram autolink-orphans [--apply] [--min-co-mentions 3]
engram graph                          # Open interactive graph visualization
```

### Browse & Export

```bash
engram status                         # Summary counts
engram dump                           # Rich tables: all memories, nodes, edges
engram dump --format json             # Full JSON export
engram decay [--limit 20]             # Ebbinghaus decay report
```

### System

```bash
engram init                           # Initialize config
engram start                          # Start daemon (HTTP server + watcher)
engram stop                           # Stop daemon
engram serve [--host 0.0.0.0] [--port 8765]  # Foreground server
engram watch [--daemon]               # Watch inbox + OpenClaw sessions
engram health                         # Full system health check
engram resource-status                # Resource tier (FULL/STANDARD/BASIC/READONLY)
engram queue-status                   # Embedding queue status
engram scheduler-status               # Background task schedule
engram constitution-status            # 3-law governance + SHA-256 hash
```

### Maintenance

```bash
engram cleanup                        # Delete expired memories
engram consolidate [--limit 50]       # LLM-driven memory consolidation
engram ingest <file.json> [--dry-run] # Ingest chat JSON
engram backup                         # Export memory snapshot
engram restore <file>                 # Import snapshot
engram config show / get <key> / set <key> <value>
engram feedback <id> --positive|--negative
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `engram_remember` | Store memory with type, priority, tags, namespace |
| `engram_recall` | Search episodic memories (compact format by default) |
| `engram_think` | Reason across episodic + semantic memory via LLM |
| `engram_status` | Show memory statistics |
| `engram_get_memory` | Retrieve full memory content by ID or prefix |
| `engram_timeline` | Chronological context around a memory |
| `engram_add_entity` | Add entity node to knowledge graph |
| `engram_add_relation` | Add relationship edge between entities |
| `engram_query_graph` | Query knowledge graph |
| `engram_ingest` | Dual ingest: extract entities + store memories |
| `engram_meeting_ledger` | Record structured meeting (decisions, action items) |
| `engram_feedback` | Record positive/negative feedback on memories |
| `engram_auto_feedback` | Auto-detect feedback from conversation |
| `engram_cleanup` | Delete all expired memories |
| `engram_cleanup_dedup` | Deduplicate similar memories by cosine similarity |
| `engram_summarize` | Summarize recent N memories via LLM |
| `engram_session_start` | Begin new conversation session |
| `engram_session_end` | End active session with optional summary |
| `engram_session_summary` | Get summary of completed session |
| `engram_session_context` | Retrieve memories from active session |
| `engram_ask` | Smart query — auto-routes to recall or think |

---

## Configuration

**Config file:** `~/.engram/config.yaml` — Priority: CLI flags > env vars > YAML > defaults

```yaml
episodic:
  mode: embedded              # embedded (Qdrant in-process) or server
  path: ~/.engram/qdrant
  namespace: default

embedding:
  provider: gemini
  model: gemini-embedding-001
  key_strategy: failover      # failover or round-robin

semantic:
  provider: sqlite            # or postgresql
  path: ~/.engram/semantic.db

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}

serve:
  host: 127.0.0.1
  port: 8765

capture:
  openclaw:
    enabled: false
    sessions_dir: ~/.openclaw/workspace/sessions
  claude_code:
    enabled: false
    sessions_dir: ~/.claude/projects

auth:
  enabled: false
cache:
  enabled: false
  redis_url: redis://localhost:6379/0
rate_limit:
  enabled: false
audit:
  enabled: false
  path: ~/.engram/audit.jsonl
```

---

## API Reference

Start server: `engram serve [--host 0.0.0.0] [--port 8765]`

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness probe |
| `POST` | `/api/v1/remember` | Store episodic memory |
| `GET` | `/api/v1/recall` | Search memories (`?query=X&limit=5&offset=0`) |
| `POST` | `/api/v1/think` | LLM reasoning across episodic + semantic |
| `GET` | `/api/v1/query` | Graph search (`?keyword=X&node_type=Y&related_to=Z`) |
| `GET` | `/api/v1/memories` | List/filter memories with pagination |
| `GET` | `/api/v1/memories/export` | Export memories as JSON |
| `POST` | `/api/v1/meeting-ledger` | Record structured meeting |
| `POST` | `/api/v1/ingest` | Extract entities + store memories |
| `POST` | `/api/v1/feedback` | Record feedback on a memory |
| `GET` | `/api/v1/graph/data` | Graph data JSON for visualization |
| `GET` | `/graph` | Interactive graph visualization UI |
| `POST` | `/api/v1/cleanup` | Delete expired memories (admin) |
| `POST` | `/api/v1/cleanup/dedup` | Deduplicate memories (admin) |
| `POST` | `/api/v1/summarize` | LLM summary of recent memories (admin) |
| `GET` | `/api/v1/status` | Memory statistics |

---

## WebSocket API

Connect via `ws://host:8765/ws?token=JWT` (token optional when auth disabled).

**Commands:**

| Command | Payload |
|---------|---------|
| `remember` | `{"content": "...", "priority": 7}` |
| `recall` | `{"query": "...", "limit": 5}` |
| `think` | `{"question": "..."}` |
| `feedback` | `{"memory_id": "abc123", "feedback": "positive"}` |
| `query` | `{"keyword": "PostgreSQL"}` |
| `ingest` | `{"messages": [...]}` |
| `status` | `{}` |

**Push Events:** `memory_created`, `memory_updated`, `memory_deleted`, `feedback_recorded`

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | LLM + embeddings (primary key) |
| `GEMINI_API_KEY_FALLBACK` | Secondary key for key rotation |
| `ENGRAM_NAMESPACE` | Memory namespace isolation |
| `ENGRAM_AUTH_ENABLED` | Enable JWT auth |
| `ENGRAM_SEMANTIC_PROVIDER` | `sqlite` or `postgresql` |
| `ENGRAM_CACHE_ENABLED` | Enable Redis caching |
| `ENGRAM_AUDIT_ENABLED` | Enable audit logs |
| `ENGRAM_TELEMETRY_ENABLED` | Enable OpenTelemetry |

---

## Docker

```bash
# Quick start
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest

# Production with PostgreSQL + Redis
ENGRAM_AUTH_ENABLED=true \
ENGRAM_SEMANTIC_PROVIDER=postgresql \
ENGRAM_SEMANTIC_DSN=postgresql://user:pass@postgres:5432/engram \
ENGRAM_CACHE_ENABLED=true \
ENGRAM_CACHE_REDIS_URL=redis://redis:6379/0 \
docker compose up
```

---

## Testing

```bash
pytest tests/ -v                      # All tests
pytest tests/ --cov=src/engram        # With coverage
pytest tests/ -k "recall or feedback" # Specific suites
```

894+ tests, 61%+ code coverage, CI/CD via GitHub Actions.

---

## Documentation

- [Project Overview & PDR](docs/project-overview-pdr.md)
- [System Architecture](docs/system-architecture.md)
- [Code Standards](docs/code-standards.md)
- [Deployment Guide](docs/deployment-guide.md)
- [Codebase Summary](docs/codebase-summary.md)
- [Project Roadmap](docs/project-roadmap.md)
- [Changelog](docs/project-changelog.md)

---

## License

MIT — Copyright (c) Do Cao Hieu
