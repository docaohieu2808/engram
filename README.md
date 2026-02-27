# engram

**Memory traces for AI agents — Think like humans.**

![Version](https://img.shields.io/badge/version-0.4.0-blue) ![Tests](https://img.shields.io/badge/tests-894%2B-brightgreen) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Dual-memory AI system combining episodic (vector) + semantic (graph) memory with LLM reasoning. Enterprise-ready with multi-tenancy, auth, caching, observability, and Docker deployment. Exposes CLI, MCP, HTTP API (`/api/v1/`), and WebSocket API (`/ws`) for real-time agent communication.

---

## Features

### Core Memory

- **Episodic Memory** — ChromaDB vector store, semantic similarity search, Ebbinghaus decay, activation-based scoring, topic-key upsert
- **Semantic Graph** — NetworkX MultiDiGraph, typed entities and relationships, SQLite (default) or PostgreSQL backend, weighted edges
- **Reasoning Engine** — LLM synthesis (Gemini via litellm), dual-memory context fusion, constitution-guarded prompts
- **Recall Pipeline** — Query decision, temporal+pronoun entity resolution, parallel multi-source search, dedup, composite scoring
- **Auto Memory** — Detect and persist save-worthy messages automatically, poisoning guard for injection prevention
- **Feedback Loop** — Confidence scoring (+0.15/−0.2), importance adjustment, auto-delete on 3x negative feedback
- **Graph Visualization** — Interactive entity relationship explorer with dark theme, search, click-to-inspect (vis-network)

### Intelligence Layer

- **Temporal Resolution** — 28 Vietnamese+English date patterns resolve "hôm nay/yesterday" → ISO dates before storing
- **Pronoun Resolution** — "anh ấy/he/she" → named entity from graph context, LLM-based fallback
- **Fusion Formatter** — Group recall results by type `[preference]`/`[fact]`/`[lesson]` for structured LLM context
- **Memory Consolidation** — Jaccard clustering + LLM summarization reduces redundancy

### Enterprise

- **Multi-Surface** — CLI (Typer), MCP Server (stdio), HTTP API (FastAPI), WebSocket, Web UI
- **Authentication** — JWT + API keys with RBAC (ADMIN, AGENT, READER), optional, disabled by default
- **Multi-Tenancy** — Isolated per-tenant stores, contextvar propagation, row-level PostgreSQL isolation
- **Caching** — Redis-backed result caching with per-endpoint TTLs
- **Rate Limiting** — Sliding-window per-tenant limits, `fail_open` option
- **Audit Trail** — Structured before/after JSONL log for every episodic mutation
- **Resource Tiers** — 4-tier LLM degradation (FULL → STANDARD → BASIC → READONLY), 60s auto-recovery
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
    CLI["CLI (Typer)"]
    MCP["MCP (stdio)"]
    HTTP["HTTP API /api/v1/"]
    WS["WebSocket /ws"]

    CLI & MCP & HTTP & WS --> Auth["Auth Middleware\n(JWT + RBAC, optional)"]
    Auth --> Tenant["TenantContext (ContextVar)"]
    Tenant --> Recall["Recall Pipeline\n(decision → resolve → search → feedback)"]
    Recall --> Episodic["EpisodicStore\n(ChromaDB)"]
    Recall --> Semantic["SemanticGraph\n(NetworkX + SQLite/PG)"]
    Episodic & Semantic --> Reasoning["Reasoning Engine\n(Gemini via litellm)"]
    Episodic --> Cache["Redis Cache (optional)"]
    WS --> EventBus["Event Bus\n(push events)"]
```

---

## Quick Start

```bash
# Install from source
git clone https://github.com/docaohieu2808/engram.git
cd engram
pip install -e .

# Optional: dev dependencies
pip install -e ".[dev]"

# Optional: OpenTelemetry support
pip install -e ".[telemetry]"
```

**Requirements:** Python 3.11+

**Optional:** `GEMINI_API_KEY` for LLM reasoning and embeddings. Basic storage works without it.

### CLI Usage

```bash
export GEMINI_API_KEY="your-key"

# Store a memory
engram remember "Deployed v2.1 to production at 14:00 - caused 503 spike"

# Retrieve similar memories
engram recall "production incidents"

# Reason across all memory
engram think "What deployment issues have we had?"

# Add knowledge graph entities
engram add node "PostgreSQL" --type Technology
engram add edge "Service:API" "Technology:PostgreSQL" --relation uses

# Start HTTP API server
engram serve
```

### Python SDK

```python
from engram import EngramClient

async with EngramClient(namespace="my-agent") as client:
    # Auto-recalls relevant context before LLM call,
    # auto-extracts facts from response in the background
    response = await client.chat([
        {"role": "user", "content": "Deploy to production"}
    ])

# Explicit operations
await client.remember("No deploys on Fridays", memory_type="decision", priority=8)
results = await client.recall("deployment policy", limit=5)
answer = await client.think("What are our deployment rules?")
```

### HTTP API Usage

```bash
# Store memory
curl -X POST http://localhost:8765/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Deployed v1.0 to production", "memory_type": "fact", "priority": 8}'

# Search
curl "http://localhost:8765/api/v1/recall?query=deployment&limit=5"

# Reason
curl -X POST http://localhost:8765/api/v1/think \
  -H "Content-Type: application/json" \
  -d '{"question": "What deployment issues have we had?"}'
```

---

## Configuration

**Config file:** `~/.engram/config.yaml` — Priority: CLI flags > env vars > YAML > defaults

```yaml
episodic:
  provider: chromadb
  path: ~/.engram/episodic
  namespace: default

embedding:
  provider: gemini
  model: gemini-embedding-001
  # Key rotation: failover (default) or round-robin
  key_strategy: failover  # GEMINI_API_KEY + GEMINI_API_KEY_FALLBACK

semantic:
  provider: sqlite          # or postgresql
  path: ~/.engram/semantic.db
  # For PostgreSQL:
  # dsn: postgresql://user:pass@localhost/engram
  # pool_min: 5
  # pool_max: 20

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}

auth:
  enabled: false            # Set to true for production
  jwt_secret: "use-32+-chars-or-${ENV_VAR}"

cache:
  enabled: false            # Set to true with Redis
  redis_url: redis://localhost:6379/0

rate_limit:
  enabled: false
  requests_per_minute: 100
  fail_open: true           # Allow through on Redis failure

audit:
  enabled: false
  path: ~/.engram/audit.jsonl

telemetry:
  enabled: false            # Requires telemetry extra
  otlp_endpoint: http://localhost:4317

recall:
  enabled: true
  decision_skip_trivial: true
  entity_resolution_enabled: true
  parallel_search_enabled: true
  feedback_enabled: true
  auto_consolidate_threshold: 20
  retrieval_audit_enabled: true

ingestion:
  auto_memory_enabled: true
  guard_enabled: true

feedback:
  confidence_positive_delta: 0.15
  confidence_negative_delta: 0.2
  auto_delete_threshold: 3

resolution:
  temporal_enabled: true
  pronoun_enabled: true

fusion:
  formatter_enabled: true

graph:
  visualization_enabled: true
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | LLM + embeddings (primary key) |
| `GEMINI_API_KEY_FALLBACK` | Secondary key for key rotation |
| `GEMINI_KEY_STRATEGY` | `failover` (default) or `round-robin` |
| `ENGRAM_AUTH_ENABLED` | Enable auth (`true`/`false`) |
| `ENGRAM_AUTH_JWT_SECRET` | JWT signing key (32+ chars) |
| `ENGRAM_SEMANTIC_PROVIDER` | `sqlite` or `postgresql` |
| `ENGRAM_SEMANTIC_DSN` | PostgreSQL connection string |
| `ENGRAM_CACHE_ENABLED` | Enable Redis caching |
| `ENGRAM_CACHE_REDIS_URL` | Redis URL |
| `ENGRAM_RATE_LIMIT_ENABLED` | Enable rate limiting |
| `ENGRAM_RATE_LIMIT_REQUESTS_PER_MINUTE` | Default 60 |
| `ENGRAM_AUDIT_ENABLED` | Enable audit logs |
| `ENGRAM_TELEMETRY_ENABLED` | Enable OpenTelemetry |
| `ENGRAM_RECALL_ENABLED` | Enable recall pipeline |
| `ENGRAM_INGESTION_AUTO_MEMORY_ENABLED` | Detect save-worthy messages |
| `ENGRAM_INGESTION_GUARD_ENABLED` | Block prompt injection |
| `ENGRAM_FEEDBACK_ENABLED` | Enable feedback loop |
| `ENGRAM_RESOLUTION_TEMPORAL_ENABLED` | Temporal entity resolution |
| `ENGRAM_RESOLUTION_PRONOUN_ENABLED` | Pronoun resolution via LLM |
| `ENGRAM_FUSION_FORMATTER_ENABLED` | Format recall by type |
| `ENGRAM_GRAPH_VISUALIZATION_ENABLED` | Enable graph UI at /graph |

**Note:** `${VARIABLE}` syntax in YAML is expanded at load time.

---

## API Reference

Start server: `engram serve [--host 0.0.0.0] [--port 8765]`

All endpoints at `/api/v1/`; legacy routes redirect. Auth disabled by default.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Liveness check (always available) |
| `GET` | `/health/ready` | Readiness probe |
| `POST` | `/api/v1/remember` | Store episodic memory |
| `GET` | `/api/v1/recall` | Search memories (`?query=X&limit=5&offset=0`) |
| `POST` | `/api/v1/think` | LLM reasoning across episodic + semantic |
| `GET` | `/api/v1/query` | Graph search (`?keyword=X&node_type=Y&related_to=Z`) |
| `POST` | `/api/v1/ingest` | Extract entities + store memories from messages |
| `POST` | `/api/v1/feedback` | Record positive/negative feedback on a memory |
| `GET` | `/api/v1/graph/data` | Graph data JSON for visualization |
| `GET` | `/graph` | Interactive graph visualization UI |
| `POST` | `/api/v1/cleanup` | Delete expired memories (admin only) |
| `POST` | `/api/v1/summarize` | LLM synthesis of recent memories (admin only) |
| `POST` | `/api/v1/auth/token` | Issue JWT (admin_secret required) |
| `POST` | `/api/v1/backup` | Export all memory to JSON |
| `POST` | `/api/v1/restore` | Import backup snapshot |

**Responses:** All wrapped in `{data, meta}`. Errors: `{error: {code, message}, meta}`.

**Auth:** When enabled, use `Authorization: Bearer <JWT>` or `X-API-Key: <key>`.

---

## WebSocket API

Real-time bidirectional communication. Connect via `ws://host:8765/ws?token=JWT` (token optional when auth disabled).

**Commands (client to server):**

| Command | Payload |
|---------|---------|
| `remember` | `{"content": "...", "priority": 7}` |
| `recall` | `{"query": "...", "limit": 5}` |
| `think` | `{"question": "..."}` |
| `feedback` | `{"memory_id": "abc123", "feedback": "positive"}` |
| `query` | `{"keyword": "PostgreSQL"}` |
| `ingest` | `{"messages": [...]}` |
| `status` | `{}` |

**Example:**
```json
{"id": "corr-1", "type": "remember", "payload": {"content": "User prefers dark mode", "priority": 7}}
{"id": "corr-1", "type": "response", "status": "ok", "data": {"id": "mem-xyz"}}
```

**Push Events (server to all agents in same tenant):**
```json
{"type": "event", "event": "memory_created", "tenant_id": "default", "data": {"id": "mem-xyz"}}
```

Events: `memory_created`, `memory_updated`, `memory_deleted`, `feedback_recorded`

---

## CLI Reference

### Memory Operations

```bash
# Store with options
engram remember <content> [--type fact|decision|preference|todo|error|context|workflow]
                          [--priority 1-10] [--tags tag1,tag2] [--expires 2h|1d|7d]

# Search
engram recall <query> [--limit 5] [--type <type>] [--tags tag1,tag2] [--namespace <ns>]
              [--resolve-entities] [--resolve-temporal]

# Reason
engram think <question>
engram summarize [--count 20] [--save]
```

### Semantic Graph

```bash
engram add node <name> --type <NodeType>
engram add edge <from_key> <to_key> --relation <relation>
engram remove node <key>
engram query [<keyword>] [--type <NodeType>] [--related-to <name>] [--format table|json]
engram graph [--search <keyword>]       # Open interactive graph visualization
```

### Intelligence & Pipeline

```bash
engram resolve <query> [--context "..."]      # Entity/temporal resolution
engram feedback <id> [--positive|--negative]  # Record feedback on memory
engram audit [--last N]                        # View retrieval audit log
engram benchmark --questions file.json         # Run accuracy benchmarks
```

### System

```bash
engram status                          # Memory + graph stats
engram serve [--host 127.0.0.1] [--port 8765]
engram watch [--daemon]                # Watch inbox + start scheduler
engram resource-status                 # Show resource tier (FULL/STANDARD/BASIC/READONLY)
engram constitution-status             # Show 3-law governance + SHA-256 hash
engram scheduler-status                # Show background task schedule + state
```

### Maintenance

```bash
engram cleanup                         # Delete expired memories
engram ingest <file.json> [--dry-run]  # Ingest chat JSON
engram backup                          # Export memory snapshot
engram restore <file>                  # Import snapshot
engram config show / get <key> / set <key> <value>
```

---

## MCP Integration

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

**Available MCP Tools:**

| Tool | Description |
|------|-------------|
| `engram_remember` | Store memory with type, priority, tags, namespace |
| `engram_recall` | Search episodic memories (compact format by default) |
| `engram_resolve` | Entity/temporal resolution for query context |
| `engram_feedback` | Record positive/negative feedback on memories |
| `engram_think` | Reason across episodic + semantic memory via LLM |
| `engram_summarize` | Summarize recent N memories via LLM |
| `engram_cleanup` | Delete all expired memories |
| `engram_status` | Show memory statistics |
| `engram_get_memory` | Retrieve full memory content by ID or prefix |
| `engram_timeline` | Chronological context around a memory |
| `engram_add_entity` | Add entity node to knowledge graph |
| `engram_add_relation` | Add relationship edge between entities |
| `engram_query_graph` | Query knowledge graph |
| `engram_get_graph_data` | Retrieve graph data for visualization |
| `engram_ingest` | Dual ingest: extract entities + store memories |
| `engram_session_start` | Begin new conversation session |
| `engram_session_end` | End active session with optional summary |
| `engram_session_summary` | Get summary of completed session |
| `engram_session_context` | Retrieve memories from active session |

**Claude Code Hooks:** Engram ships Stop + SessionEnd hooks that capture conversation turns automatically — no SDK integration needed.

---

## Embeddings

Only one embedding model is supported:

| Mode | Model | Dimensions | Requires |
|------|-------|-----------|---------|
| Gemini (default) | `gemini-embedding-001` | 3072 | `GEMINI_API_KEY` |
| Fallback | `all-MiniLM-L6-v2` (ChromaDB default) | 384 | nothing |

Embedding dimensions must remain consistent within a collection. If you switch providers, reinitialize the episodic store or create a new collection.

**Key rotation:** Set `GEMINI_API_KEY_FALLBACK` and `GEMINI_KEY_STRATEGY=round-robin` to distribute quota across multiple API keys.

---

## Benchmarks

Run `python tests/benchmark_performance.py --host 127.0.0.1 --port 8765` against a running server.

Sample results (local, SQLite backend, no Redis):

| Operation | p50 | p95 | p99 | Notes |
|-----------|-----|-----|-----|-------|
| `GET /health` | 0.8ms | 1.2ms | 2.1ms | Always fast |
| `POST /remember` | 415ms | 680ms | 920ms | Embedding API bound |
| `GET /recall` | 1.3ms | 3.2ms | 5.8ms | Vector search |
| `POST /think` | 5.6s | 7.2s | 9.4s | LLM bound |

Flags: `--quick` for a faster subset, `--concurrency N` for load testing.

---

## Docker

### Quick Start

```bash
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest
```

### Production (PostgreSQL + Redis)

```bash
# Production env vars
ENGRAM_AUTH_ENABLED=true
ENGRAM_AUTH_JWT_SECRET=$(openssl rand -hex 32)
ENGRAM_SEMANTIC_PROVIDER=postgresql
ENGRAM_SEMANTIC_DSN=postgresql://user:pass@postgres:5432/engram
ENGRAM_CACHE_ENABLED=true
ENGRAM_CACHE_REDIS_URL=redis://redis:6379/0
ENGRAM_AUDIT_ENABLED=true
GEMINI_API_KEY=your-key
GEMINI_API_KEY_FALLBACK=your-fallback-key
GEMINI_KEY_STRATEGY=round-robin
```

See [deployment-guide.md](docs/deployment-guide.md) for Docker Compose with PostgreSQL, Redis, and OpenTelemetry.

---

## Test Coverage

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/engram --cov-report=html

# Specific suites
pytest tests/ -k "recall or resolution or feedback" -v
pytest tests/ -k "websocket" -v
```

- **894+ tests** across all modules
- **61%+ code coverage** (core features 80%+)
- **CI/CD:** GitHub Actions runs full suite on every PR and commit

---

## Documentation

- **[Project Overview & PDR](docs/project-overview-pdr.md)** — Features, requirements, config reference
- **[System Architecture](docs/system-architecture.md)** — Design, data flow, deployment patterns
- **[Code Standards](docs/code-standards.md)** — Conventions, patterns, best practices
- **[Deployment Guide](docs/deployment-guide.md)** — Docker, Kubernetes, environment variables, auth setup
- **[Codebase Summary](docs/codebase-summary.md)** — Module inventory, metrics
- **[Project Roadmap](docs/project-roadmap.md)** — Completed phases and future work
- **[Changelog](docs/project-changelog.md)** — Full version history

---

## Contributing

Pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Before submitting:**
- Run `ruff check src/` for linting
- Run `pytest tests/` — do not ignore failing tests
- Do not commit `.env` files or API keys

---

## License

MIT — Copyright (c) Do Cao Hieu
