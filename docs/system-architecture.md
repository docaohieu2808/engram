# Engram System Architecture

## High-Level Overview

Engram is a dual-memory AI system that enables agents to reason like humans by combining:

1. **Episodic Memory** — Vector embeddings for semantic search (ChromaDB)
2. **Semantic Memory** — Knowledge graphs for entity relationships (PostgreSQL/SQLite + NetworkX MultiDiGraph)
3. **Reasoning Engine** — LLM synthesis connecting both stores (Gemini via litellm)
4. **Federation Layer** — External memory providers (REST, File, Postgres, MCP) via smart query router

All four interfaces (CLI, MCP, HTTP, WebSocket) share the same memory layers.

**Version:** 0.4.0 | **Tests:** 894+ | **LOC:** ~5000+

---

## System Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         Interfaces                             │
├──────────────────────┬─────────────────┬──────────────────────┬──────────────────────┤
│   CLI (Typer)        │  MCP (FastMCP)  │  HTTP API (FastAPI)  │  WebSocket API       │
│   Local tools        │  stdio-based    │  /api/v1/ endpoints  │  /ws?token=JWT       │
│                      │                 │                      │  bidirectional push  │
└──────────────────────┴────────┬────────┴──────────┬───────────┴──────────┬───────────┘
                                 │                   │
                    ┌────────────┴───────────────────┘
                    │
        ┌───────────▼─────────────┐
        │  Auth Middleware        │ ← JWT Bearer or X-API-Key
        │  (optional, default off)│   RBAC: ADMIN/AGENT/READER
        └───────────┬─────────────┘
                    │
        ┌───────────▼──────────────┐
        │  TenantContext (ContextVar)│
        │  tenant_id = ${JWT.tenant_id}│
        └───────────┬──────────────┘
                    │
    ┌───────────────┴────────────────────┐
    │   ReasoningEngine                  │ ← LLM Synthesis (Gemini)
    │   (think, summarize, ingest)       │
    ├─────────────────┬──────────────────┤
    │  EpisodicStore  │  SemanticGraph   │
    │  (ChromaDB)     │  (PG/SQLite+NX)  │
    └─────────────────┴──────────────────┘
         │                        │
         ├─ Chunking             ├─ Nodes (entities)
         ├─ Embeddings           ├─ Edges (relations)
         ├─ Vector search        └─ Query DSL
         │
    ┌────┴──────────────┐
    │ Redis Cache       │ ← Optional
    │ (recall_ttl,      │   Graceful fallback
    │  think_ttl,       │
    │  query_ttl)       │
    └────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │  Federation Layer (v0.2 — src/engram/providers/) │
    │                                                  │
    │  QueryRouter (classify_query)                    │
    │    → "internal" — skip external providers        │
    │    → "domain"   — fan-out to all active providers│
    │                                                  │
    │  ProviderRegistry (entry_points plugin system)   │
    │    ├─ RestAdapter   (Cognee, Mem0, LightRAG,     │
    │    │                 Graphiti)                   │
    │    ├─ FileAdapter   (OpenClaw memory files)      │
    │    ├─ PostgresAdapter (external PG tables)       │
    │    └─ McpAdapter   (MCP server subprocesses)     │
    │                                                  │
    │  AutoDiscovery (discover command)                │
    │    Tier 1: local port scan (8000,8080,9520,…)    │
    │    Tier 2: remote hosts                          │
    │    Tier 3: direct endpoints                      │
    │    Tier 4: MCP config files                      │
    └──────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ Rate Limiter     │ ← JWT-based (not header spoofing)
    │ (sliding window) │   Per-tenant via Redis
    └──────────────────┘

    ┌──────────────────────────┐
    │ Observability Stack      │ ← Optional
    │ - OpenTelemetry          │   Disabled by default
    │ - JSONL Audit logs       │
    │ - Structured Logging     │
    └──────────────────────────┘
```

---

## Data Flow Layers

### Layer 1: Episodic Memory (Vector Search)

**Path:** `src/engram/episodic/`

**Components:**
- **EpisodicStore** (`store.py`) — ChromaDB wrapper
  - One collection per tenant (namespace)
  - Auto-chunks content >1000 chars (800-char chunks with overlap)
  - Metadata: memory_type, priority, tags, expires, created_at, updated_at, access_count, last_accessed, decay_rate
  - **New (v0.3.0):** Activation-based recall with access_count + last_accessed tracking
  - **New (v0.3.0):** Ebbinghaus decay scoring for memory retention
- **Search** (`search.py`) — Embedding & similarity scoring + activation
  - Gemini embeddings (3072 dims) or fallback (384 dims)
  - Tag/type filtering on recall
  - **New (v0.3.0):** Composite scoring: similarity (0.5) + retention (0.2) + recency (0.15) + frequency (0.15)
  - **New (v0.3.0):** Batch updates to access_count + last_accessed on recall()
- **Models** — MemoryType enum (fact, decision, error, todo, preference, context, workflow)

**Operations:**
```python
# Store
remember(content, memory_type, priority, tags, expires) → id

# Retrieve
recall(query, limit, offset, memory_type, tags) → [Memory]

# Maintenance
cleanup() → int  # deleted count
```

**Storage:**
- ChromaDB path: `~/.engram/episodic/` (or `ENGRAM_EPISODIC_PATH`)
- Collections named: `engram_{tenant_id}` (e.g., engram_default, engram_acme)

---

### Layer 2: Semantic Memory (Knowledge Graph)

**Path:** `src/engram/semantic/`

**Components:**
- **Backend abstraction** (`backend.py`) — Plugin interface
  - SQLite implementation (`sqlite_backend.py`)
  - PostgreSQL implementation (`pg_backend.py`)
- **Graph** (`graph.py`) — NetworkX **MultiDiGraph** wrapper (in-memory query engine)
  - Graph type: `networkx.MultiDiGraph` — multiple directed edges per node pair (different relation types co-exist)
  - Nodes: typed entities with attributes
  - Edges: relationships with metadata, **weight: float (default 1.0), attributes: dict**
  - Operations: add/remove, query, relate, path finding
  - **New (v0.3.0):** Weighted edges support for scoring path relationships
- **Query DSL** (`query.py`) — Find nodes by keyword/type/relatedness with pagination
  - **New (v0.3.0):** Weight-aware scoring in path finding queries

**Schema:**
- **SQLite:** Single table schema
  ```sql
  CREATE TABLE nodes (id UUID, key TEXT UNIQUE, type TEXT, attributes JSON);
  CREATE TABLE edges (source_id UUID, target_id UUID, relation TEXT, metadata JSON);
  ```
- **PostgreSQL:** Same + tenant_id columns for multi-tenant isolation

**Operations:**
```python
# Build graph
add_node(name, type, attributes) → Node
add_edge(from_key, to_key, relation, metadata) → Edge

# Query
query(keyword, node_type, related_to, limit, offset) → [Node]
relate(entity_key) → [related_entities]
```

**Storage:**
- SQLite: `~/.engram/semantic.db` or per-tenant `~/.engram/semantic.{tenant_id}.db`
- PostgreSQL: Single database, `tenant_id` column filters rows

---

### Layer 3: Federation Layer (v0.2)

**Path:** `src/engram/providers/`

**Purpose:** Connect engram to external memory sources via pluggable adapters. Results are merged with internal recall before LLM synthesis.

**Components:**

- **MemoryProvider** (`base.py`) — Abstract base with built-in stats tracking and circuit breaker
  - `search(query, limit)` → `list[ProviderResult]`
  - `health()` → `bool`
  - `add(content, metadata)` → `str | None` (optional write support)
  - `tracked_search()` — wraps `search()` with latency + error counting; auto-disables after `max_consecutive_errors` (default 5)

- **Adapters** — Four built-in implementations:
  | Adapter | File | Purpose |
  |---------|------|---------|
  | RestAdapter | `rest_adapter.py` | HTTP POST/GET to Cognee, Mem0, LightRAG, Graphiti |
  | FileAdapter | `file_adapter.py` | Glob markdown/text files (e.g. OpenClaw workspace) |
  | PostgresAdapter | `postgres_adapter.py` | Parameterised SQL query against external tables |
  | McpAdapter | `mcp_adapter.py` | Spawn MCP server subprocess, call tool via stdio |

- **QueryRouter** (`router.py`) — Keyword-based classification before fan-out
  - `classify_query(query)` → `"internal"` or `"domain"`
  - Internal keywords (decisions, errors, todos) skip providers → fast path
  - Domain keywords (how-to, docs, setup) include all active providers
  - Queries >3 words default to `"domain"`
  - Supports Vietnamese keywords

- **ProviderRegistry** (`registry.py`) — Loads providers from config + `entry_points`
  - Built-in types: `rest`, `file`, `postgres`, `mcp`
  - Third-party: register via `entry_points(group="engram.providers")`
  - `get_active()` returns only enabled + non-auto-disabled providers

- **AutoDiscovery** (`discovery.py`) — Scans for running services
  - Known services: Cognee (8000), Mem0 (8080), LightRAG (9520), OpenClaw (file), Graphiti (8000)
  - SSRF protection: remote hosts validated against routable IPs only
  - MCP config scan: `~/.claude/settings.json`, `~/.cursor/settings.json`

**Config** (`~/.engram/config.yaml`): See [project-overview-pdr.md](./project-overview-pdr.md) for full provider config examples.

---

### Layers 3b–3f: Utility Layers (v0.3.0)

| Layer | Path | Purpose |
|-------|------|---------|
| 3b Consolidation | `consolidation/engine.py` | Jaccard clustering + LLM summarization → CONTEXT memories |
| 3c Privacy | `sanitize.py` | Strip `<private>...</private>` → `[REDACTED]` before ChromaDB insert |
| 3d Topic Upsert | `episodic/store.py` | `topic_key` param upserts existing memory by key; `revision_count` tracks updates |
| 3e Session | `session/store.py` | JSON-file session store; auto-injects `session_id` into memory metadata |
| 3f Git Sync | `sync/git_sync.py` | Compressed JSONL chunks to `.engram/`; manifest-based dedup; incremental export |

---

### Layer 3g: Recall Pipeline (v0.3.1)

**Path:** `src/engram/recall/`, `src/engram/ingestion/`, `src/engram/feedback/`, `src/engram/benchmark/`

**Purpose:** Enhanced query processing with intelligent decision-making, entity resolution, multi-source search, and adaptive learning.

**Components:**

- **Query Decision** (`decision.py`) — Trivial message detection
  - Regex patterns: "ok", "thanks", "hello", emoji
  - Skips vector search <10ms, returns empty
  - Prevents processing noise

- **Entity Resolver** (`entity_resolver.py`) — Context extraction
  - **Temporal:** Vietnamese+English date regex (no LLM), patterns: "hôm nay", "tuần trước", "yesterday", etc.
  - **Pronoun:** LLM-based resolution (gemini-flash) with fallback to direct matching
  - Methods: resolve_temporal(), resolve_pronoun(), resolve_text()

- **Parallel Search** (`parallel_search.py`) — Multi-source fusion
  - **Sources:** ChromaDB semantic search, entity graph keyword match, keyword fallback
  - **Fusion:** Parallel async queries, dedup by content hash, score ranking (0-1)
  - **SearchResult:** content, score, source, metadata, resolved_entities
  - Returns top-K after fusion

- **Feedback Loop** (`feedback/loop.py`) — Adaptive confidence
  - Track positive/negative feedback on memories
  - Adjust confidence: +0.15 (positive), -0.2 (negative)
  - Auto-delete: memory if negative_count >= 3 AND confidence < threshold
  - Improves result quality over time

- **Auto-Memory** (`capture/auto_memory.py`) — Selective ingestion
  - Detect save-worthy messages: "Save: " prefix, identity, preferences, decisions
  - Skip sensitive data: passwords, API keys, tokens, PII patterns
  - Automatically remember without user intervention

- **Poisoning Guard** (`ingestion/guard.py`) — Injection prevention
  - Block prompt injection patterns: "ignore instructions", "you are now", "forget", special tokens
  - Filter before storage, log attempts
  - Protects semantic graph integrity

- **Auto-Consolidate** (`consolidation/auto_trigger.py`) — Periodic cleanup
  - Trigger consolidation after N messages (default 20)
  - Runs async, doesn't block main flow
  - Reduces memory redundancy

- **Retrieval Audit** (`retrieval_audit_log.py`) — Query logging
  - JSONL append-only log: timestamp, query, results_count, source, latency_ms
  - Tracks recall patterns, debugging

- **Benchmarking** (`benchmark/runner.py`) — Accuracy measurement
  - Load question sets (JSON format)
  - Compare model answers vs. golden answers
  - Metrics: exact match, semantic similarity, F1 score
  - Report by question type (factual, reasoning, etc.)

**Data Flow:**
```
1. Query → QueryDecision (skip trivial?)
2. → EntityResolver (extract temporal/pronoun context)
3. → ParallelSearch (multi-source search + fusion)
4. → FeedbackLoop (retrieve + check confidence)
5. → AutoMemory (detect save-worthy → remember)
6. → PoisoningGuard (block injection attempts)
7. → AutoTrigger (consolidate if threshold reached)
8. → RetrievalAuditLog (log operation)
```

**Config:** RecallPipelineConfig, FeedbackConfig, IngestionConfig, ResolutionConfig, RetrievalAuditConfig

**New Models:** Entity, ResolvedText, SearchResult, FeedbackType, MemoryCandidate

**New EpisodicMemory Fields:** confidence (float, default 1.0), negative_count (int, default 0)

---

### Layers 3h–3k: Intelligence Layer (v0.4.0)

Four new modules providing enhanced entity resolution, adaptive learning, result formatting, and graph visualization. Wires 7 orphaned modules into main flow.

| Layer | Module | Purpose |
|-------|--------|---------|
| 3h | `recall/temporal_resolver.py` | 28 Vietnamese+English patterns resolve "hôm nay/yesterday" → ISO dates before storing |
| 3i | `recall/pronoun_resolver.py` | LLM-based "anh ấy/he/she" → named entity from graph context, fallback to direct match |
| 3j | `recall/fusion_formatter.py` | Group recall results by type [preference]/[fact]/[lesson] for LLM context |
| 3k | `static/graph.html` + API | Interactive entity relationship explorer at /graph, vis-network dark theme, search, click-to-inspect |

**Wired Modules (now active):**
- `ingestion/guard.py` — Prompt injection prevention (now required for security)
- `recall/decision.py` — Trivial message skip <10ms
- `providers/telemetry.py` — Latency tracking
- `episodic/fts_index.py` — Full-text search indexing
- `recall/parallel_search.py` — Multi-source fusion
- `capture/auto_memory.py` — Auto-detection of save-worthy messages
- `consolidation/auto_trigger.py` — Consolidation trigger after N messages

**Bug Fixes (v0.4.0):**
- FTS5 thread safety: Lock acquisition in parallel search
- OOM pagination: Limit results before aggregation
- Rate limiter race condition: Redis atomic increment with TTL

**New MCP Tool:** `engram_get_graph_data(search_keyword)` — retrieve filtered graph JSON for visualization

**New API Endpoints:**
- `POST /api/v1/feedback` — Record positive/negative feedback on memories
- `GET /api/v1/graph/data` — Graph data JSON for visualization
- `GET /graph` — Interactive graph HTML interface

**Config:** `resolution.temporal_enabled`, `resolution.pronoun_enabled`, `fusion.formatter_enabled`, `graph.visualization_enabled`

**CLI:** `engram graph` launches browser at localhost:8765/graph

---

### Layers 3l–3o: Brain Features (v0.3.2)

Four new modules providing operational reliability and LLM governance. See **[brain-features-architecture.md](./brain-features-architecture.md)** for full detail.

| Layer | Module | Purpose |
|-------|--------|---------|
| 3h | `audit.py` (extended) | Traceable before/after log for every episodic mutation |
| 3i | `resource_tier.py` | 4-tier LLM degradation (FULL→STANDARD→BASIC→READONLY), 60s auto-recovery |
| 3j | `constitution.py` | 3-law governance prefix injected into every LLM prompt, SHA-256 tamper detection |
| 3k | `scheduler.py` | Asyncio overlap-safe background tasks (cleanup daily, consolidate 6h, decay daily) |

CLI commands added: `engram resource-status`, `engram constitution-status`, `engram scheduler-status`

---

### Layers 3p–3q: TUI + MCP Progressive Disclosure (v0.3.0)

**TUI** (`src/engram/tui/`) — Interactive terminal interface via textual library.
- 4 screens: Dashboard, Search (live, <500ms), Recent, Sessions; vim navigation; `engram tui`; optional dep `engram[tui]`

**MCP Progressive Disclosure** (`src/engram/mcp/episodic_tools.py`) — Reduces MCP token usage.
- `engram_recall`: compact by default `{id, date, type, snippet}` (~50 tokens vs. 500)
- `engram_get_memory(id)`: full content on demand
- `engram_timeline(id, window_minutes)`: chronological context ±window

---

### Layer 4: Reasoning Engine

**Path:** `src/engram/reasoning/`

**Components:**
- **Engine** (`engine.py`) — Orchestrates dual-memory synthesis
  - Fetches relevant episodic memories (recall)
  - Traverses semantic graph for context
  - Builds prompt with both contexts
  - Calls LLM (Gemini via litellm)
  - Returns synthesized answer

**Operations:**
```python
# LLM reasoning
think(question, context_limit) → answer_str

# Batch synthesis
summarize(count, save) → insights_str

# Entity extraction + auto-ingest
ingest(messages) → {extracted_entities, stored_memories}
```

---

### Layer 5: HTTP API Server

**Path:** `src/engram/capture/server.py`

**Endpoints** (all at `/api/v1/` with optional `/` legacy redirects):

| Method | Endpoint | Purpose | Auth | Response |
|--------|----------|---------|------|----------|
| GET | /health | Liveness | No | `{status: ok}` |
| GET | /health/ready | Readiness probe | No | `{status, checks}` |
| POST | /api/v1/remember | Store episodic memory | No | `{id, created_at}` |
| GET | /api/v1/recall | Search memories | No | `{results, total, offset, limit}` |
| POST | /api/v1/think | LLM reasoning (federated) | No | `{answer}` |
| POST | /api/v1/ingest | Extract entities + store | No | `{extracted, stored}` |
| GET | /api/v1/query | Graph search | No | `{nodes, edges, total, offset, limit}` |
| GET | /api/v1/status | Memory + graph counts | No | `{episodic_count, graph_nodes, ...}` |
| POST | /api/v1/cleanup | Delete expired | Admin | `{deleted}` |
| POST | /api/v1/summarize | LLM synthesis | Admin | `{summary}` |
| POST | /api/v1/auth/token | Issue JWT | No (admin_secret in body) | `{token, expires_at}` |
| GET | /providers | List active providers + stats | Auth required | `{providers}` |

**Response envelope:** `{"data": {...}, "meta": {"request_id", "timestamp", "correlation_id"}}`

**Error envelope:** `{"error": {"code": "INVALID_REQUEST|UNAUTHORIZED|FORBIDDEN|NOT_FOUND|RATE_LIMITED|INTERNAL_ERROR", "message": "..."}, "meta": {...}}`

---

### Layer 5b: WebSocket API (v0.4.1)

**Path:** `src/engram/ws/`

**Purpose:** Bidirectional real-time communication for live memory push events to connected clients.

**Components:**
- **protocol.py** — WebSocket message protocol: command schema, event types, serialization
- **event_bus.py** — In-process pub/sub bus; broadcasts memory events (remember, delete, update) to all subscribers
- **connection_manager.py** — Manages active WebSocket connections with per-tenant isolation; handles connect/disconnect lifecycle
- **handler.py** — FastAPI WebSocket route handler; authenticates via JWT, dispatches commands, subscribes to event bus

**Connection:**
```
ws://localhost:8765/ws?token=<JWT>
```

**Supported Commands (client → server):**
- `recall` — Search memories
- `remember` — Store a memory
- `think` — LLM reasoning
- `query` — Graph query
- `ingest` — Entity extraction
- `feedback` — Record feedback
- `status` — Get memory counts

**Push Events (server → client):**
- `memory.created` — New memory stored
- `memory.deleted` — Memory removed
- `memory.updated` — Memory content/metadata changed
- `consolidation.completed` — Consolidation run finished
- `error` — Command error response

**Per-Tenant Isolation:** Connection manager partitions subscriptions by `tenant_id` from JWT; tenants only receive events for their own namespace.

---

### Layer 6: Multi-Tenancy

**Path:** `src/engram/tenant.py`

**TenantContext:** Global ContextVar holding current tenant_id (defaults to "default")

**StoreFactory:** Creates and caches per-tenant store instances
- One EpisodicStore per tenant (ChromaDB collection)
- One SemanticGraph per tenant (SQLite file or PG row filtering)
- LRU eviction: 100 graphs, 1000 episodic stores max
- Lazy initialization to support async event loops

**Tenant ID Propagation:**
1. **HTTP API:** Extracted from JWT `sub` claim or API key record
2. **MCP/CLI:** Set at startup from config namespace
3. **Default:** "default" for backward compatibility

---

### Layer 7: Authentication & Authorization

**Path:** `src/engram/auth.py`, `src/engram/auth_models.py`

**JWT Flow:**
1. POST /auth/token with `admin_secret` in header
2. Server signs JWT with config.auth.jwt_secret (HS256)
3. Client includes JWT in Authorization: Bearer header
4. Middleware verifies signature, extracts tenant_id + role
5. RBAC checked against endpoint requirements

**API Key Flow:**
1. CLI: `engram auth create-key <name> --role admin`
2. Key generated as 32-byte URL-safe string
3. Hash stored in ~/.engram/api_keys.json
4. Client includes in X-API-Key header
5. Middleware verifies hash, looks up role

**Roles:**
- **ADMIN** — Read + write all operations, cleanup, summarize
- **AGENT** — Write episodic/semantic, read all (typical usage)
- **READER** — Read-only, no modifications

**Public Paths** (never require auth):
- /health
- /api/v1/auth/token
- /{remember, recall, query, think, ingest} when auth.enabled = false (default)

---

### Layer 8: Caching & Rate Limiting

**Path:** `src/engram/cache.py`, `src/engram/rate_limiter.py`

**Cache:**
- Redis-backed (optional, disabled by default)
- Keys: `engram:{tenant_id}:{endpoint}:{hash(params)}`
- TTLs: recall (300s), think (900s), query (300s)
- Graceful fallback: system works without Redis

**Rate Limiter:**
- Sliding-window via Redis (optional, disabled by default)
- Per-tenant limits: config.rate_limit.requests_per_minute
- Burst allowance: config.rate_limit.burst
- Headers: X-RateLimit-Limit, -Remaining, -Reset; Retry-After
- JWT-based identity (not spoofable X-Forwarded-For header)
- **`fail_open`** option (default: true): if Redis is unavailable, requests are allowed through rather than rejected

---

### Layer 9: Observability

**Path:** `src/engram/logging_setup.py`, `src/engram/audit.py`

**Logging:**
- Format: "text" (default) or "json" (structured)
- Level: DEBUG, INFO, WARNING (default), ERROR, CRITICAL
- Correlation ID: ContextVar in all logs
- Third-party noise silenced (litellm, chromadb, httpx → WARNING)

**Audit Logging** (optional, disabled by default):
- JSONL format, one record per line
- Tracks: timestamp, tenant_id, user_id, action, resource, status, metadata
- Path: ~/.engram/audit.jsonl (configurable)

**OpenTelemetry** (optional, requires telemetry extra):
- Spans: remember, recall, think, query operations
- Traces: full request flow
- Exporters: OTLP protocol to collector
- Sample rate: config.telemetry.sample_rate (default 0.1 = 10%)

---

### Layer 10: Embedding Key Rotation

**Path:** `src/engram/episodic/embeddings.py`

**Purpose:** Distribute Gemini embedding API quota across multiple keys; failover on auth failure.

**Model:** `gemini-embedding-001` exclusively (3072 dimensions). Dimensions must remain consistent within a ChromaDB collection.

**Key Sources:**
- `GEMINI_API_KEY` — primary key (required)
- `GEMINI_API_KEY_FALLBACK` — secondary key (optional)

**Strategies** (set via `embedding.key_strategy` in config.yaml or `GEMINI_KEY_STRATEGY` env var):

| Strategy | Behavior |
|----------|----------|
| `failover` (default) | Use primary key; switch to fallback only on auth failure |
| `round-robin` | Rotate keys evenly across calls to spread quota usage |

**Error Handling:** If a key fails with `AuthenticationError`, the next key is tried. If all keys fail, a `RuntimeError` is raised.

**Config:**
```yaml
embedding:
  provider: gemini
  model: gemini-embedding-001
  key_strategy: failover  # or round-robin
```

---

### Layer 11: WebSocket Typed Payloads

**Path:** `src/engram/ws/protocol.py`

All WebSocket messages use JSON with the schema `{"id": "<correlation_id>", "type": "<command>", "payload": {...}}`.

**Command Payloads (client to server):**

| Command | Payload Schema |
|---------|----------------|
| `remember` | `{"content": str, "memory_type"?: str, "priority"?: int, "tags"?: [str]}` |
| `recall` | `{"query": str, "limit"?: int, "memory_type"?: str}` |
| `think` | `{"question": str}` |
| `feedback` | `{"memory_id": str, "feedback": "positive"\|"negative"}` |
| `query` | `{"keyword"?: str, "node_type"?: str, "related_to"?: str}` |
| `ingest` | `{"messages": [{"role": str, "content": str}]}` |
| `status` | `{}` |

**Response Schema (server to client):**
```json
{"id": "<correlation_id>", "type": "response", "status": "ok"|"error", "data": {...}}
```

**Push Event Schema (server to all agents in tenant):**
```json
{"type": "event", "event": "memory_created"|"memory_updated"|"memory_deleted"|"feedback_recorded", "tenant_id": "...", "data": {...}}
```

**Error Schema:**
```json
{"type": "error", "code": "UNKNOWN_COMMAND"|"UNAUTHORIZED"|"INTERNAL_ERROR", "message": "..."}
```

---

## Deployment Architecture

### Local Development
```bash
engram serve --host 127.0.0.1 --port 8765
```
- Single process, no external services required
- ChromaDB embedded, SQLite local
- Auth disabled by default

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8765
CMD ["engram", "serve", "--host", "0.0.0.0"]
```

Environment variables:
```bash
ENGRAM_SERVE_HOST=0.0.0.0
ENGRAM_SERVE_PORT=8765
ENGRAM_AUTH_ENABLED=true
ENGRAM_AUTH_JWT_SECRET=$(openssl rand -hex 32)
ENGRAM_SEMANTIC_PROVIDER=postgresql
ENGRAM_SEMANTIC_DSN=postgresql://user:pass@postgres:5432/engram
ENGRAM_CACHE_ENABLED=true
ENGRAM_CACHE_REDIS_URL=redis://redis:6379/0
ENGRAM_AUDIT_ENABLED=true
ENGRAM_TELEMETRY_ENABLED=true
ENGRAM_TELEMETRY_OTLP_ENDPOINT=http://otel-collector:4317
GEMINI_API_KEY=${GEMINI_API_KEY}
```

### Production Architecture (Recommended)
```
┌──────────────────────────────────────────┐
│         Load Balancer (nginx)             │ ← Rate limit at edge
└──────────────────┬───────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐      ┌───▼───┐      ┌──▼────┐
│engram-1│      │engram-2│      │engram-N│  ← Replicas (stateless)
└───┬───┘      └───┬───┘      └──┬────┘
    │              │              │
    └──────────────┼──────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
┌───▼───────────┐          ┌─────▼──────┐
│ PostgreSQL    │          │   Redis    │
│ (semantic)    │          │ (cache +   │
│              │          │  rate-limit)
└───────────────┘          └────────────┘

External:
- ChromaDB (embedded per pod, or shared persistent volume)
- Gemini API (external)
- OpenTelemetry Collector (optional)
```

---

## Configuration Hierarchy

**Priority order** (highest wins):
1. CLI flags (not yet implemented for serve, but available for CLI commands)
2. Environment variables (ENGRAM_*)
3. YAML config (~/.engram/config.yaml)
4. Built-in defaults (in Pydantic models)

Example:
```yaml
# ~/.engram/config.yaml
semantic:
  provider: sqlite
  path: ~/.engram/semantic.db

# Override with env var
export ENGRAM_SEMANTIC_PROVIDER=postgresql
export ENGRAM_SEMANTIC_DSN=postgresql://localhost/engram

# Result: provider=postgresql (env wins)
```

---

## Component Communication

All interfaces share the same request pipeline:

1. **CLI:** Parse args → load config → set TenantContext → StoreFactory → operation → Rich output
2. **MCP:** Receive tool call → load config → StoreFactory → execute → JSON result
3. **HTTP API:** Request → CorrelationIdMiddleware → AuthMiddleware (JWT/API key) → RateLimitMiddleware → handler → StoreFactory → execute → cache → `{data, meta}` response
4. **WebSocket:** `ws://.../ws?token=JWT` → authenticate → ConnectionManager.register → EventBus.subscribe → dispatch command → push events to tenant

---

## State & Persistence

| Component | Storage | Scope | Persistence |
|-----------|---------|-------|-------------|
| EpisodicStore | ChromaDB | Per tenant | Persistent (embedded DB) |
| SemanticGraph | SQLite/PostgreSQL | Per tenant | Persistent (file/DB) |
| Config | YAML file | Global | Persistent (~/.engram/config.yaml) |
| API keys | JSON file | Global | Persistent (~/.engram/api_keys.json) |
| TenantContext | ContextVar | Per request/task | Transient (memory only) |
| Cache | Redis | Per tenant | Transient (with TTL) |
| Audit logs | JSONL file | Global | Persistent (~/.engram/audit.jsonl) |
| Traces | OTLP exporter | Global | External (to collector) |

---

## Error Handling Strategy

1. **Input validation:** Pydantic validates all requests before business logic
2. **Structured errors:** All failures wrapped in ErrorCode + message
3. **Logging:** Errors logged with correlation_id for traceability
4. **User feedback:** HTTP 4xx/5xx responses with error code for client retry logic
5. **Graceful degradation:** Optional services (Redis, OTel) fail safely
6. **Audit trail:** Failures recorded in audit logs when enabled

---

## Security Boundaries

| Boundary | Control | Mechanism |
|----------|---------|-----------|
| Tenant isolation | Row-level (PG) or file-level (SQLite) | tenant_id column/filename |
| API access | JWT + API key verification | HMAC signature + hash lookup |
| Authorization | Role-based (RBAC) | Role enum + path-based rules (path normalization applied) |
| Content size | 10KB limit per memory | Pydantic Field(max_length=10000) |
| Secret storage | Hashed API keys only | SHA256 hashes in JSON |
| Audit trail | Immutable log | Append-only JSONL |
| Timing attacks | Constant-time comparison | `hmac.compare_digest` for key verification |
| JWT secret | Minimum length enforced | Startup validation rejects short secrets |
| SSRF (webhooks) | URL allowlist validation | Private/loopback IPs blocked in discovery + webhooks |
| SQL injection | Parameterised queries | PostgresAdapter uses `$1/$2` placeholders only |

---

## Performance Characteristics

| Operation | Typical Time | Bottleneck | Mitigation |
|-----------|--------------|-----------|-----------|
| remember() | <10ms | Embedding | Async + cache |
| recall(10 items) | <50ms | Vector search | Redis cache |
| think() | <2s | LLM call | Client-side timeout |
| query() | <50ms | Graph traversal | LRU cache |
| cleanup() | Variable | DB scan | Async background task |

---

## Future Architecture Evolution

See [project-roadmap.md](./project-roadmap.md) for full roadmap. Key upcoming architectural changes:
- Distributed semantic graph (multi-node, Raft consensus)
- Streaming LLM responses for large reasoning tasks
- Graph migrations (SQLite → PostgreSQL tooling)
- Observability dashboard UI

