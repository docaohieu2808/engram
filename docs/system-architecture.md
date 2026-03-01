# Engram System Architecture

## High-Level Overview

Engram is a dual-memory AI system that enables agents to reason like humans by combining:

1. **Episodic Memory** — Vector embeddings for semantic search (ChromaDB)
2. **Semantic Memory** — Knowledge graphs for entity relationships (PostgreSQL/SQLite + NetworkX MultiDiGraph)
3. **Reasoning Engine** — LLM synthesis connecting both stores (Gemini via litellm)
4. **Federation Layer** — External memory providers (REST, File, Postgres, MCP) via smart query router

All four interfaces (CLI, MCP, HTTP, WebSocket) share the same memory layers.

**Version:** 0.4.3 | **Tests:** 972+ | **LOC:** ~7100+

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

**Architecture (Phase 4, v0.4.1):** Pluggable backend protocol supports embedded ChromaDB + HTTP backends.

**Backend abstraction** (`backend.py`) — **NEW (v0.4.1)**
  - Protocol interface: `create()`, `remember()`, `recall()`, `delete()`, `cleanup()`, `count()`, `collection()`
  - Implementations:
    - `chromadb_backend.py` — Local ChromaDB (default)
    - `chromadb_http_backend.py` — Remote ChromaDB server (HTTP REST)
    - `_legacy_collection_backend.py` — Backward-compat shim
  - Config: `episodic.mode` (embedded/http), `episodic.host`, `episodic.port`

**Store refactored (Phase 2, v0.4.1)** — Modularized from 1,170 LOC to 111 LOC shell + 6 mixins
  - `store.py` (111 LOC) — Shell coordinator
  - `episodic_crud.py` (180 LOC) — _EpisodicCrudMixin (remember, get, update, delete)
  - `episodic_search.py` (150 LOC) — _EpisodicSearchMixin (recall, search_similar, filtering)
  - `episodic_maintenance.py` (120 LOC) — _EpisodicMaintenanceMixin (cleanup, decay)
  - `batch_operations.py` (100 LOC) — _BatchMixin (bulk ops)
  - `fts_sync.py` (80 LOC) — Async FTS helpers

**Embeddings** (`embeddings.py`) — Multi-key rotation for Gemini API
  - Model: `gemini-embedding-001` exclusively (3072 dimensions)
  - Key strategies: failover (default) or round-robin
  - Sources: `GEMINI_API_KEY` + `GEMINI_API_KEY_FALLBACK` (optional)

**Search** (`search.py`) — Similarity scoring + activation-based recall
  - Tag/type filtering on recall
  - Composite scoring: similarity (0.5) + retention (0.2) + recency (0.15) + frequency (0.15)
  - Batch updates to access_count + last_accessed on recall()

**Models** — MemoryType enum (fact, decision, error, todo, preference, context, workflow)

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

**Lazy Loading (Phase 5, v0.4.1)** — **NEW**
  - Indexed SQL queries skip full NetworkX load for large graphs
  - Backend pagination: limit/offset on DB queries reduce in-memory footprint
  - Graph `query()` and `relate()` now fetch only relevant nodes from DB before loading into memory
  - Performance: 10,000+ node graphs respond in <100ms vs. 5s+ when loading full graph

**Backend abstraction** (`backend.py`) — Plugin interface
  - SQLite implementation (`sqlite_backend.py`)
  - PostgreSQL implementation (`pg_backend.py`)
  - Methods: `create()`, `load_graph()`, `save_graph()`, `add_node()`, `add_edge()`, `find_related()`, `query()`, `close()`

**Graph** (`graph.py`) — NetworkX **MultiDiGraph** wrapper (in-memory query engine)
  - Graph type: `networkx.MultiDiGraph` — multiple directed edges per node pair (different relation types co-exist)
  - Nodes: typed entities with attributes
  - Edges: relationships with metadata, **weight: float (default 1.0), attributes: dict**
  - Operations: add/remove, query, relate, path finding
  - Weighted edges support for scoring path relationships

**Query DSL** (`query.py`) — Find nodes by keyword/type/relatedness with pagination
  - DB-backed queries with limit/offset
  - Weight-aware scoring in path finding
  - Optional full-graph load when needed (default: lazy fetch)

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

**Path:** `src/engram/capture/`

**Architecture (Phase 3, v0.4.1):** Refactored from 1,099 LOC to modular router structure.

**Server Shell** (`server.py`, 275 LOC) — **REFACTORED (v0.4.1)**
  - Factory pattern for app creation
  - Middleware registration (correlation ID, rate limiting)
  - Router mounting from submodules
  - Legacy endpoint redirects for backward compatibility

**Middleware** (`middleware.py`) — **NEW (v0.4.1)**
  - CorrelationIdMiddleware — X-Correlation-ID header propagation
  - RateLimitMiddleware — Per-tenant sliding-window limiting

**Router Modules** (`routers/`) — **NEW (v0.4.1)**
  - `memory_routes.py` — POST /api/v1/remember, GET /api/v1/recall
  - `memories_crud.py` — PUT/DELETE /api/v1/memories/{id}, GET /api/v1/memories
  - `graph_routes.py` — GET /api/v1/query, GET /api/v1/graph/data, GET /graph (HTML)
  - `admin_routes.py` — POST /api/v1/cleanup, POST /api/v1/summarize, POST /api/v1/feedback

**Server Helpers** (`server_helpers.py`) — **NEW (v0.4.1)**
  - Shared route validation logic, response builders

**Endpoints** (all at `/api/v1/` with optional `/` legacy redirects):

| Method | Endpoint | Purpose | Auth | Response |
|--------|----------|---------|------|----------|
| GET | /health | Liveness | No | `{status: ok}` |
| GET | /health/ready | Readiness probe | No | `{status, checks}` |
| POST | /api/v1/remember | Store episodic memory | No | `{id, created_at}` |
| GET | /api/v1/recall | Search memories | No | `{results, total, offset, limit}` |
| GET | /api/v1/memories/{id} | Get single memory | No | `{id, content, metadata}` |
| PUT | /api/v1/memories/{id} | Update memory metadata | No | `{id, updated_at}` |
| DELETE | /api/v1/memories/{id} | Delete memory | No | `{deleted: true}` |
| POST | /api/v1/think | LLM reasoning (federated) | No | `{answer}` |
| POST | /api/v1/ingest | Extract entities + store | No | `{extracted, stored}` |
| GET | /api/v1/query | Graph search | No | `{nodes, edges, total, offset, limit}` |
| GET | /api/v1/graph/data | Graph JSON (filtered) | No | `{nodes, edges}` |
| GET | /graph | Interactive graph UI | No | `{HTML}` |
| GET | /api/v1/status | Memory + graph counts | No | `{episodic_count, graph_nodes, ...}` |
| POST | /api/v1/feedback | Record memory feedback | No | `{confidence, importance}` |
| POST | /api/v1/cleanup | Delete expired | Admin | `{deleted}` |
| POST | /api/v1/summarize | LLM synthesis | Admin | `{summary}` |
| POST | /api/v1/auth/token | Issue JWT | No | `{token, expires_at}` |
| GET | /providers | List active providers + stats | Auth | `{providers}` |

**Response envelope:** `{"data": {...}, "meta": {"request_id", "timestamp", "correlation_id"}}`

**Error envelope:** `{"error": {"code": "INVALID_REQUEST|UNAUTHORIZED|FORBIDDEN|NOT_FOUND|RATE_LIMITED|INTERNAL_ERROR", "message": "..."}, "meta": {...}}`

---

### Layer 5b: WebSocket API (v0.4.1) & Event Bus (Phase 6)

**Path:** `src/engram/ws/`

**Purpose:** Bidirectional real-time communication for live memory push events to connected clients.

**Event Bus (Phase 6, v0.4.1)** — **NEW**
  - **In-process:** `event_bus.py` (80 LOC) — Per-tenant pub/sub channels
  - **Redis adapter:** `redis_event_bus.py` (150 LOC) — **NEW** Distributed event bus for multi-instance deployments
    - Config: `event_bus.enabled`, `event_bus.backend` (memory/redis), `event_bus.redis_url`
    - Async FTS writes via event bus (non-blocking search index updates)
    - Decouples episodic mutations from WebSocket delivery
  - Lifecycle: `subscribe()`, `unsubscribe()`, `broadcast()`, `close()`

**WebSocket Components:**
- **protocol.py** (60 LOC) — Message schema: {type, payload, request_id}
- **connection_manager.py** (100 LOC) — Per-tenant connection registry; cleanup on client drop
- **handler.py** (120 LOC) — FastAPI route `/ws?token=JWT`; command dispatch; event subscription

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

### Layer 5c: HTTP Client SDK (Phase 7, v0.4.1)

**Path:** `src/engram/http_client.py` (200 LOC) — **NEW**

Standalone Python SDK for connecting to Engram HTTP API from external applications.

**Features:**
- Async HTTP client (httpx) with automatic JWT token management
- Support for all major operations: remember, recall, think, query, feedback
- Built-in request retry logic + error handling
- Type hints + Pydantic response models
- Works with multi-tenant instances (tenant_id in constructor)
- Context manager support for clean resource cleanup

**Usage:**
```python
from engram import EngramHttpClient

async with EngramHttpClient("http://localhost:8765", tenant_id="acme") as client:
    # Store a memory
    memory_id = await client.remember("My context", memory_type="fact", priority=5)

    # Search memories
    results = await client.recall("What happened yesterday?", limit=5)

    # Reason about memories
    answer = await client.think("Summarize our progress")

    # Record feedback
    confidence = await client.feedback(memory_id, feedback_type="positive")
```

**Configuration:**
- API base URL (required)
- API key or JWT token (optional, anonymous by default)
- Custom headers (X-Correlation-ID, etc.)
- Request timeout, retry strategy, connection pooling

### Layer 5d: Plugin Entry Points (Phase 7, v0.4.1)

**Path:** `pyproject.toml` — **EXTENDED**

Extensible plugin architecture via Python entry points:

**Entry point groups:**
- `engram.episodic_backends` — Custom episodic store implementations
- `engram.semantic_backends` — Custom semantic graph backends
- `engram.providers` — External memory provider adapters
- `engram.cli_commands` — Third-party CLI commands
- `engram.mcp_tools` — Custom MCP tools

**Plugin registration:**
```toml
[project.entry-points."engram.providers"]
my_provider = "my_package.providers:MyProviderAdapter"

[project.entry-points."engram.cli_commands"]
my_command = "my_package.cli:my_command_group"
```

**Discovery:**
- Auto-loaded at startup via `importlib.metadata.entry_points()`
- Plugins must conform to abstract interface (Protocol)
- Health checks + graceful failure if plugin unavailable
- Config-driven enable/disable per plugin

---

### Layer 5e: Configurable Everything (v0.4.2)

**Path:** `src/engram/config.py` (extended), `src/engram/capture/routers/config_routes.py` (new), `src/engram/static/components/settings.js` (new)

**Purpose:** Eliminate all hardcoded tuning parameters; expose via YAML config + WebUI live editor for zero-restart updates.

**New Config Classes:**
- **ExtractionConfig** — llm_model (inherit from llm.model), temperature (0.1), max_retries (3), retry_delay_seconds (1.0), chunk_size (50), user_msg_max_len (2000), assistant_msg_max_len (3000)
- **RecallConfig (extended)** — search_limit (15), entity_search_limit (10), provider_search_limit (5), entity_graph_depth (2), entity_boost_score (0.55), semantic_edge_score (0.5), entity_co_mention_score (0.4), keyword_exact_match_score (0.6), fuzzy_match_score (0.3), fusion_similarity_weight (0.6), fusion_retention_weight (0.4), entity_resolution_context_window (10), entity_resolution_max_len (3000), fusion_entry_max_chars (200), format_for_llm_max_chars (2000), federated_search_timeout (10.0)
- **SchedulerConfig** — consolidate_interval_seconds (21600), cleanup_interval_seconds (86400), decay_report_interval_seconds (86400), tick_interval_seconds (60), task_timeout_seconds (300), decay_access_multiplier (0.1)
- **HealthConfig, CacheConfig, HooksConfig, RetrievalAuditConfig** — Secondary params (health.check_llm_model, cache.max_graph_cache_size, hooks.webhook_timeout_seconds, etc.)

**API Endpoints (new):**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/config` | Read current config.yaml as JSON |
| PUT | `/api/v1/config` | Merge config changes, validate, persist to config.yaml |

**WebUI Settings Tab (new):**
- Grouped sections: LLM, Embedding, Recall, Extraction, Scheduler, Health, Cache
- Form controls: text, number, select, multi-select, boolean toggle
- Live validation: type checking, range limits, dependency checks
- "Restart Required" badge for server-level changes (llm.model, episodic.mode, auth.enabled)
- "Save" button persists changes to ~/.engram/config.yaml

**Model Selector UI (new):**
- LLM section dropdown: Gemini, Claude, OpenAI, Custom
- API key input per provider (masked password field)
- Test button: sends probe query to verify model connectivity
- Auto-set `llm.disable_thinking` flag based on model family (Claude Sonnet → true, Gemini → false, etc.)

**Think Flag Unification:**
- All hardcoded `thinking={"type":"disabled"}` removed
- 4 modules now read `cfg.llm.disable_thinking`:
  - `recall/entity_resolver.py:100`
  - `health/components.py:98`
  - `memory_extractor.py:83`
  - `consolidation/engine.py:162`

**YAML Config Example:**
```yaml
llm:
  model: gemini-2.5-flash
  disable_thinking: false
  provider: gemini

extraction:
  llm_model: null  # inherit from llm.model
  temperature: 0.1
  max_retries: 3

recall:
  search_limit: 15
  entity_boost_score: 0.55
  fusion_similarity_weight: 0.6

scheduler:
  consolidate_interval_seconds: 21600
  cleanup_interval_seconds: 86400
```

**Env Var Overlay:**
```bash
ENGRAM_LLM_MODEL=claude-opus-4-6 engram serve
ENGRAM_EXTRACTION_TEMPERATURE=0.5 engram think "query"
```

---

### Layer 12: Setup Wizard (v0.4.3)

**Path:** `src/engram/setup/`, `src/engram/cli/setup_cmd.py`

**Purpose:** Interactive CLI wizard to auto-detect installed AI agents/IDEs and configure engram shared memory integration.

**Components:**

- **AgentConnector** (`setup/connectors/base.py`) — Abstract base class
  - `detect()` → `DetectionResult` (installed, version, config_path)
  - `configure(dry_run)` → `ConfigureResult` (success, files_modified, backup_path)
  - `verify()` → `bool`

- **McpJsonConnector** (`setup/connectors/mcp_json_mixin.py`) — Shared MCP config mixin
  - JSON merge strategy: read existing → merge engram entry → write back
  - Absolute path resolution for `engram-mcp` binary (venv-safe)
  - Backup original config before modification
  - Idempotent: skip if already configured

- **Connectors** (9 total, sorted by tier):
  | Tier | Connector | Config Method |
  |------|-----------|--------------|
  | 1 | Claude Code | MCP JSON (`~/.claude/mcp.json`) |
  | 1 | OpenClaw | Skill file (`~/.openclaw/workspace/skills/engram/SKILL.md`) |
  | 1 | Cursor | MCP JSON (`~/.cursor/mcp.json`) |
  | 1 | Windsurf | MCP JSON (`~/.codeium/windsurf/mcp_config.json`) |
  | 2 | Cline | MCP JSON (VSCode extension settings) |
  | 2 | Aider | YAML config (`~/.aider.conf.yml`) |
  | 2 | Zed | JSON settings (`~/.config/zed/settings.json`) |
  | 2 | Void | MCP JSON (`~/.void/mcp.json`) |
  | 3 | Antigravity | Proxy config (`~/.antigravity/config.json`) |

- **Detector** (`setup/detector.py`) — `scan_agents()` iterates registry, returns (connector, result) pairs
- **Wizard UI** (`setup/wizard.py`) — Rich panels + questionary checkbox prompts
- **Verifier** (`setup/verifier.py`) — Post-config verification + server status check
- **Federation** (`setup/federation/`) — Mem0, Cognee, Zep provider detection stubs

**CLI Flags:**
- `engram setup` — Interactive mode (questionary checkbox)
- `engram setup --dry-run` — Preview changes without writing
- `engram setup --non-interactive` — Configure all detected agents
- `engram setup --status` — Show connection status + verification

**Non-TTY Detection:** Auto-fallback to `--non-interactive` when stdin is not a terminal (CI, SSH pipes).

---

See [deployment-operations.md](./deployment-operations.md) for deployment architecture, configuration, component communication, state persistence, error handling, security, and performance details.