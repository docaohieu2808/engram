# engram

**Memory traces for AI agents — Think like humans.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Tests](https://img.shields.io/badge/tests-726+-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green) ![Version](https://img.shields.io/badge/version-0.4.0-blue)

Dual-memory AI system combining episodic (vector) + semantic (graph) memory with LLM reasoning. Enterprise-ready with multi-tenancy, auth, caching, observability, and Docker deployment. Exposes CLI, MCP, and versioned HTTP API (/api/v1/).

## Features

- **Dual Memory** — Episodic (ChromaDB) + Semantic (PostgreSQL/SQLite) with LLM synthesis
- **Enhanced Recall Pipeline** — Query decision, temporal+pronoun resolution, parallel search fusion, learning feedback, auto-memory, poisoning guard
- **Temporal & Pronoun Resolution** — Vietnamese+English date patterns, LLM-based pronoun resolution before storing
- **Feedback Loop** — Confidence tracking, adaptive learning, auto-delete bad memories (3× negative)
- **Fusion Formatter** — Group recall results by type [preference]/[fact]/[lesson] for LLM context
- **Graph Visualization** — Interactive entity relationship explorer with dark theme, search, click-to-inspect
- **Multi-Tenancy** — Isolated per-tenant stores, row-level isolation in PostgreSQL
- **Authentication** — JWT + API keys with RBAC (ADMIN, AGENT, READER), optional
- **Caching** — Redis-backed result caching with per-endpoint TTLs
- **Rate Limiting** — Sliding-window per-tenant limits with burst allowance
- **Versioned API** — `/api/v1/` with backward-compat redirects, structured errors
- **Observability** — OpenTelemetry + JSONL audit logging (optional)
- **Deployment** — Docker Compose, Kubernetes-ready, health checks
- **Backup/Restore** — Memory snapshots, point-in-time recovery
- **Configuration** — YAML + env vars + CLI with type casting
- **Memory Audit Trail** — Traceable before/after log for every mutation (create, update, delete, cleanup)
- **Resource-Aware Retrieval** — 4-tier degradation (FULL→STANDARD→BASIC→READONLY) with auto-recovery
- **Data Constitution** — 3-law governance with SHA-256 tamper detection, injected into every LLM prompt
- **Consolidation Scheduler** — Asyncio-based background tasks (cleanup, consolidation, decay) with tier-aware execution

## Installation

```bash
# From source
git clone https://github.com/docaohieu2808/engram.git
cd engram
pip install -e .

# Dev setup with optional dependencies
pip install -e ".[dev]"

# With observability support (OpenTelemetry)
pip install -e ".[telemetry]"
```

**Requirements:** Python 3.11+

**Optional:** `GEMINI_API_KEY` for LLM reasoning and embeddings (basic storage works without it)

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

## Auto-Memory SDK

Drop `EngramClient` into any Python agent as a transparent LLM wrapper — it auto-recalls relevant memories before each call and auto-extracts facts/decisions after.

```python
from engram import EngramClient

async with EngramClient(namespace="my-agent") as client:
    # Automatic memory — no manual remember() calls needed
    response = await client.chat([
        {"role": "user", "content": "Deploy to production"}
    ])
    # engram auto-recalls relevant context before the LLM call
    # and auto-extracts facts/decisions from the response in the background
```

Explicit memory operations are also available:

```python
# Store a memory manually
await client.remember("Team decided: no deploys on Fridays", memory_type="decision", priority=8)

# Search memories by semantic similarity
results = await client.recall("deployment policy", limit=5)

# Reason across all memory
answer = await client.think("What are our deployment rules?")
```

Sync wrappers for non-async code:

```python
client = EngramClient(namespace="my-agent")
client.remember_sync("Redis chosen for session storage")
results = client.recall_sync("caching strategy")
```

### Claude Code Hooks

For Claude Code users, engram ships Stop + SessionEnd hooks that capture every conversation turn automatically — no SDK integration needed.

```bash
# hooks auto-register via ~/.claude/settings.json
# Stop hook: extracts facts from each response (async, non-blocking)
# SessionEnd hook: stores full session summary on /clear
```

## CLI Reference

### Recall Pipeline & Intelligence (v0.4.0)

```bash
engram resolve <query> [--context "..."]              # Entity/temporal resolution
engram feedback <id> [--positive|--negative]          # Record feedback on memory
engram audit [--last N]                               # View retrieval audit log
engram benchmark --questions file.json                # Run accuracy benchmarks
engram recall <query> [--resolve-entities] [--resolve-temporal]  # Enhanced recall
engram graph [--search <keyword>]                      # Interactive graph explorer
engram resource-status                                # Show LLM degradation tier
engram constitution-status                            # Show data constitution laws + hash
engram scheduler-status                               # Show scheduled tasks and state
```

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
engram migrate <export.json>            # Import from legacy memory exports
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
engram watch [--daemon]                 # Watch inbox for chat files, auto-ingest on arrival (starts scheduler)
engram serve [--host 127.0.0.1] [--port 8765]  # Start HTTP API server
engram resource-status                  # Show current resource tier and LLM call window stats
engram constitution-status              # Show constitution laws and SHA-256 hash verification
engram scheduler-status                 # Show scheduled tasks, last run times, and state
```

## Configuration

**Config file:** `~/.engram/config.yaml` (optional; defaults work for local development)

**Priority:** CLI flags > environment variables > YAML > defaults

### Example config.yaml
```yaml
episodic:
  provider: chromadb
  path: ~/.engram/episodic
  namespace: default

embedding:
  provider: gemini
  model: gemini-embedding-001

semantic:
  provider: postgresql  # or sqlite (default)
  dsn: postgresql://user:pass@localhost/engram
  pool_min: 5
  pool_max: 20

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}

auth:
  enabled: false  # Set to true for production
  jwt_secret: "use-32+-chars-or-${ENV_VAR}"

cache:
  enabled: false  # Set to true with Redis
  redis_url: redis://localhost:6379/0

rate_limit:
  enabled: false
  requests_per_minute: 100

audit:
  enabled: false
  path: ~/.engram/audit.jsonl

telemetry:
  enabled: false  # Requires telemetry extra
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

### Key Environment Variables
```bash
GEMINI_API_KEY                        # LLM + embeddings
ENGRAM_AUTH_ENABLED                   # Enable auth
ENGRAM_AUTH_JWT_SECRET                # JWT signing key (32+ chars)
ENGRAM_SEMANTIC_PROVIDER              # sqlite or postgresql
ENGRAM_SEMANTIC_DSN                   # PostgreSQL connection string
ENGRAM_CACHE_ENABLED                  # Enable Redis caching
ENGRAM_CACHE_REDIS_URL                # Redis URL
ENGRAM_RATE_LIMIT_ENABLED             # Enable rate limiting
ENGRAM_RATE_LIMIT_REQUESTS_PER_MINUTE # Default 60
ENGRAM_AUDIT_ENABLED                  # Enable audit logs
ENGRAM_TELEMETRY_ENABLED              # Enable OpenTelemetry
ENGRAM_RECALL_ENABLED                 # Enable recall pipeline (v0.3.1+)
ENGRAM_INGESTION_AUTO_MEMORY_ENABLED  # Detect save-worthy messages (v0.3.1+)
ENGRAM_INGESTION_GUARD_ENABLED        # Block prompt injection (v0.3.1+)
ENGRAM_FEEDBACK_ENABLED               # Enable feedback loop (v0.3.1+)
ENGRAM_RESOLUTION_TEMPORAL_ENABLED    # Temporal entity resolution (v0.4.0+)
ENGRAM_RESOLUTION_PRONOUN_ENABLED     # Pronoun resolution via LLM (v0.4.0+)
ENGRAM_FUSION_FORMATTER_ENABLED       # Format recall by type [preference]/[fact]/[lesson] (v0.4.0+)
ENGRAM_GRAPH_VISUALIZATION_ENABLED    # Enable graph UI at /graph (v0.4.0+)
```

**Note:** Environment variables override YAML; `${VARIABLE}` syntax in YAML is expanded at load time.

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
| `engram_recall` | Search episodic memories (compact format by default, v0.3.1+) |
| `engram_resolve` | Entity/temporal resolution for query context (v0.3.1+) |
| `engram_feedback` | Record positive/negative feedback on memories (v0.3.1+) |
| `engram_think` | Reason across episodic + semantic memory via LLM |
| `engram_summarize` | Summarize recent N memories into key insights via LLM |
| `engram_cleanup` | Delete all expired memories from the episodic store |
| `engram_status` | Show memory statistics |
| `engram_get_memory` | Retrieve full memory content by ID or prefix (v0.3.0+) |
| `engram_timeline` | Chronological context around a memory (v0.3.0+) |
| `engram_add_entity` | Add entity node to knowledge graph |
| `engram_add_relation` | Add relationship edge between entities |
| `engram_query_graph` | Query knowledge graph by keyword, type, or relatedness |
| `engram_ingest` | Dual ingest: extract entities + store memories from messages |
| `engram_session_start` | Begin new conversation session (v0.3.0+) |
| `engram_session_end` | End active session with optional summary (v0.3.0+) |
| `engram_session_summary` | Get summary of completed session (v0.3.0+) |
| `engram_session_context` | Retrieve memories from active session (v0.3.0+) |
| `engram_get_graph_data` | Retrieve full graph data for visualization (v0.4.0+) |

## HTTP API

Start server: `engram serve [--host 0.0.0.0] [--port 8765]`

**Endpoints** (all at `/api/v1/`; legacy routes redirect):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/remember` | Store episodic memory |
| `GET` | `/recall` | Search memories (pagination: `?offset=0&limit=10`) |
| `POST` | `/think` | LLM reasoning across episodic + semantic |
| `GET` | `/query` | Graph search (`?keyword=X&node_type=Y&related_to=Z`) |
| `POST` | `/ingest` | Extract entities + store memories from messages |
| `POST` | `/feedback` | Record feedback on memory (positive/negative, v0.4.0+) |
| `POST` | `/cleanup` | Delete expired memories (admin only) |
| `POST` | `/summarize` | LLM synthesis of recent memories (admin only) |
| `GET` | `/graph` | Interactive graph visualization (v0.4.0+) |
| `GET` | `/api/v1/graph/data` | Graph data JSON for visualization (v0.4.0+) |
| `POST` | `/auth/token` | Issue JWT (admin_secret required) |
| `GET` | `/health` | Liveness check (always available) |
| `POST` | `/backup` | Export all memory to JSON |
| `POST` | `/restore` | Import backup snapshot |

**Auth:** Disabled by default. Enable with `ENGRAM_AUTH_ENABLED=true` and `ENGRAM_AUTH_JWT_SECRET`. Use Bearer token or X-API-Key header.

**Responses:** All wrapped in `{data, meta}` with error format `{error: {code, message}, meta}`.

**Example:**
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

## Embeddings

Two embedding modes depending on API key availability:

| Mode | Model | Dimensions | Requires |
|------|-------|-----------|---------|
| Gemini (default) | `gemini-embedding-001` | 3072 | `GEMINI_API_KEY` |
| Fallback | `all-MiniLM-L6-v2` (ChromaDB default) | 384 | nothing |

Embedding dimensions must remain consistent within a collection. If you switch embedding providers, reinitialize the episodic store or create a new collection.

## Docker

### Quick Start
```bash
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest
```

### Production (PostgreSQL + Redis)
See [deployment-guide.md](docs/deployment-guide.md) for Docker Compose with PostgreSQL, Redis, and OpenTelemetry.

---

## Documentation

- **[Project Overview & PDR](docs/project-overview-pdr.md)** — Features, requirements, config reference
- **[System Architecture](docs/system-architecture.md)** — Design, data flow, deployment patterns
- **[Code Standards](docs/code-standards.md)** — Conventions, patterns, best practices
- **[Deployment Guide](docs/deployment-guide.md)** — Docker, Kubernetes, environment variables, auth setup
- **[Codebase Summary](docs/codebase-summary.md)** — Module inventory, metrics
- **[Project Roadmap](docs/project-roadmap.md)** — v0.3.0 → v1.0.0, completed phases, future work
- **[Changelog](docs/project-changelog.md)** — Full version history with added/changed/fixed entries

---

## Test Coverage

- **726+ tests** across all modules (181 new in v0.4.0)
- **75%+ code coverage** target
- **CI/CD:** GitHub Actions runs full test suite on every PR and commit
- **Load tests:** Marked as excluded from CI; run separately in staging
- **Recall pipeline tests:** Decision, resolution, search, feedback, auto-memory, guard, benchmarking
- **Brain feature tests:** Audit trail, resource tier, constitution, scheduler
- **Intelligence tests:** Temporal resolution, pronoun resolution, fusion formatter, graph visualization

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/engram --cov-report=html

# Run recall pipeline tests only
pytest tests/ -k "recall or resolution or feedback or decision" -v
```

---

## Advanced Features

### Intelligence Layer (v0.4.0)
✓ Temporal Resolution — Vietnamese+English date patterns resolve "hôm nay/yesterday" → ISO dates before storing
✓ Pronoun Resolution — "anh ấy/he/she" → named entity from graph context, LLM-based fallback
✓ Fusion Formatter — group recall results by type [preference]/[fact]/[lesson] for LLM context
✓ Graph Visualization — interactive entity relationship explorer, vis-network, dark theme, search, click-to-inspect
✓ Feedback Loop + Auto-delete — confidence ±0.15/0.2, importance ±1, auto-delete on 3× negative
✓ 726 tests, 181 new (v0.4.0)

### Brain Features (v0.3.2)
✓ Memory audit trail — before/after log for every mutation (create, update, delete, cleanup, batch)
✓ Resource-aware retrieval — 4-tier degradation (FULL→STANDARD→BASIC→READONLY), 60s auto-recovery
✓ Data constitution — 3 laws (namespace isolation, no fabrication, audit rights), SHA-256 tamper detection
✓ Consolidation scheduler — asyncio background tasks (cleanup daily, consolidate 6h, decay daily), tier-aware
✓ 545 tests, 39 new (v0.3.2)

### Recall Pipeline (v0.3.1)
✓ Query decision engine (skip trivial <10ms)
✓ Entity resolution (temporal + pronoun)
✓ Parallel search fusion (multi-source)
✓ Learning feedback loop (adaptive confidence)
✓ Auto-memory detection (save-worthy messages)
✓ Poisoning guard (injection prevention)
✓ Auto-consolidation (after N messages)
✓ Retrieval audit logging (JSONL)
✓ Benchmarking framework (accuracy by type)
✓ 506 tests, 159 new (v0.3.1)

### Enterprise Features (v0.2.0+)
✓ Multi-tenancy with contextvar isolation
✓ JWT + API key authentication (optional, backward compat)
✓ PostgreSQL semantic graph backend (SQLite default)
✓ Redis caching + rate limiting (optional)
✓ OpenTelemetry + JSONL audit logging (optional)
✓ Docker + GitHub Actions CI/CD
✓ Health checks + backup/restore
✓ API versioning (/api/v1/) with error codes

### Memory Features (v0.3.0+)
✓ Activation-based recall (composite scoring)
✓ Ebbinghaus decay modeling (retention scoring)
✓ Memory consolidation (Jaccard clustering)
✓ Session lifecycle management
✓ Git sync (compressed JSONL chunks)
✓ Terminal UI (interactive explorer)
✓ Progressive disclosure (compact recall)

---

## Support & Community

- **GitHub:** https://github.com/docaohieu2808/engram
- **Issues:** Report bugs and request features
- **Discussions:** Questions, architecture discussions, ideas
- **Contributing:** Pull requests welcome (see [CONTRIBUTING.md](CONTRIBUTING.md))

---

## License

MIT
