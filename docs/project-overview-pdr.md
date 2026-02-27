# Engram: Project Overview & Product Development Requirements

## What is Engram?

Engram is a dual-memory AI agent system that thinks like humans. It combines:

- **Episodic Memory** — Vector database (ChromaDB) for semantic similarity search over timestamped memories
- **Semantic Memory** — Knowledge graph (PostgreSQL/SQLite + NetworkX) for typed entities and relationships
- **Reasoning Engine** — LLM synthesis (Gemini) that connects episodic + semantic memory to answer questions
- **Federation Layer** — External provider adapters (REST, File, Postgres, MCP) extending recall across services

Exposes four interfaces: **CLI** (Typer), **MCP** (Claude integration), **HTTP API** (FastAPI), **WebSocket API** (real-time bidirectional).

**Version:** 0.4.0 | **Status:** Enterprise-ready + Advanced Recall + Consolidation + TUI + Recall Pipeline + Brain Features + Intelligence Layer + WebSocket API + Benchmark Suite | **Tests:** 894+ | **License:** MIT

---

## Core Features

### Memory Storage
- **Episodic:** Remember facts, decisions, errors, todos with type, priority, tags, expiry
- **Semantic:** Build knowledge graphs—entities, relationships, attributes
- **Dual-mode reasoning:** LLM connects both stores to synthesize answers

### Interfaces
1. **CLI** — `engram remember`, `engram recall`, `engram think`, etc.
2. **MCP** — Tools for Claude Code and other MCP clients
3. **HTTP API** — REST endpoints at `/api/v1/` with structured errors, pagination
4. **WebSocket API** — Bidirectional real-time at `/ws?token=JWT`; 7 commands + push events; per-tenant isolation via event bus

### Enterprise Features
- **Multi-Tenancy** — Per-tenant isolation via tenant_id
- **Authentication** — JWT + API keys (RBAC: ADMIN/AGENT/READER)
- **Caching** — Redis optional, intelligent TTLs
- **Rate Limiting** — Sliding-window per-tenant limits
- **Observability** — OpenTelemetry + JSONL audit logs
- **Deployment** — Docker-ready, GitHub Actions CI/CD
- **Health Checks** — Liveness + readiness probes
- **Backup/Restore** — Snapshot and restore memory
- **Config** — YAML + env var expansion + CLI overlay

### Federation (v0.2)
- **Provider adapters:** REST, File, Postgres, MCP — connect to external memory services
- **Auto-discovery:** Finds Cognee, Mem0, LightRAG, OpenClaw, Graphiti automatically
- **Smart routing:** Keyword classification routes queries to internal or federated providers
- **Circuit breaker:** Providers auto-disable after consecutive errors; re-enable manually
- **Plugin system:** Third-party adapters via `entry_points(group="engram.providers")`

### Recall Pipeline (v0.3.1)
- **Query Decision:** Trivial message skip (ok, thanks, emoji)
- **Entity Resolution:** Temporal (Vietnamese+English regex) + pronoun resolution (LLM with fallback)
- **Parallel Search:** Multi-source (ChromaDB semantic + entity graph + keyword fallback) with fusion
- **Learning Pipeline:** Feedback loop (±0.15/0.2 confidence), auto-delete after 3× negative + low confidence
- **Auto-Memory Detection:** Detect save-worthy messages (Save: prefix, identity, preferences, decisions)
- **Poisoning Guard:** Block prompt injection, special tokens, instruction overrides
- **Auto-Consolidate:** Trigger consolidation after N messages (default 20)
- **Retrieval Audit:** JSONL log for all recall operations
- **Benchmarking:** Run question sets, measure accuracy by type

### Brain Features (v0.3.2)
- **Memory Audit Trail:** Structured before/after log for every mutation — MODIFICATION_TYPES: memory_create, memory_delete, memory_update, metadata_update, config_change, batch_create, cleanup_expired
- **Resource-Aware Retrieval:** ResourceMonitor with sliding window tracks LLM call success/failure; 4 tiers (FULL, STANDARD, BASIC, READONLY); BASIC returns raw results without synthesis; auto-recovery after 60s cooldown
- **Data Constitution:** 3 laws (namespace isolation, no fabrication, audit rights); auto-creates ~/.engram/constitution.md on first load; SHA-256 tamper detection; compact prefix injected into every LLM prompt
- **Consolidation Scheduler:** Asyncio recursive setTimeout pattern (overlap-safe); 3 default tasks (cleanup_expired daily, consolidate_memories 6h, decay_report daily); respects resource tier; state persisted to ~/.engram/scheduler_state.json; starts automatically with `engram watch`

### Embedding & Key Rotation
- **Model:** `gemini-embedding-001` exclusively (3072 dimensions)
- **Key Rotation:** `GEMINI_API_KEY` + `GEMINI_API_KEY_FALLBACK` with configurable strategy
- **Strategies:** `failover` (default — use primary, switch on failure) or `round-robin` (rotate evenly to spread quota)
- **Config:** `embedding.key_strategy` in config.yaml or `GEMINI_KEY_STRATEGY` env var

### Benchmark Suite
- **Runner:** `tests/benchmark_performance.py` — measures p50/p95/p99 latency per endpoint
- **Operations:** health, remember, recall, think across configurable concurrency levels
- **Usage:** `python tests/benchmark_performance.py [--quick] [--concurrency N] [--host H] [--port P]`
- **Sample results:** health 0.8ms p50, remember 415ms p50 (embedding API bound), recall 1.3ms p50, think 5.6s p50 (LLM bound)

### Extensibility
- **Pluggable semantic backend:** SQLite (default) or PostgreSQL
- **Schema templates:** Built-in schemas (devops, marketing, personal) or custom
- **Webhooks:** Fire-and-forget on memory operations (SSRF-protected)
- **Async:** Full async/await support for scale

---

## Architecture

```
┌─────────────────────────────────────────┐
│  CLI / MCP / HTTP API (/api/v1/)        │
└────────────────────┬────────────────────┘
                     │
        ┌────────────┴────────────┐
        │  Auth Middleware        │ ← JWT + API key
        │  (enabled = false)      │   RBAC enforcement
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  TenantContext          │ ← tenant_id per request
        └────────────┬────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  Reasoning Engine             │ ← LLM (Gemini)
        ├────────┬─────────────────────┤
        │Episodic│ Semantic            │
        │(ChromaDB)│ (PG/SQLite+NX)    │
        └────────┴─────────────────────┘
            │              │
        Vector Search   Knowledge Graph
            │              │
        Redis Cache   asyncpg pool
```

---

## Product Development Requirements (PDR)

### Functional Requirements

#### FR1: Episodic Memory
- **FR1.1** Store memories with content, type, priority, tags, expiry
- **FR1.2** Search by semantic similarity with optional filters (type, tags, namespace)
- **FR1.3** Retrieve by ID, pagination support
- **FR1.4** Auto-cleanup of expired memories
- **FR1.5** Auto-chunking for large content (>1000 chars)
- **Acceptance:** remember() + recall() work with Gemini + fallback embeddings

#### FR2: Semantic Memory
- **FR2.1** Create typed entities (nodes) with attributes
- **FR2.2** Create relationships (edges) between entities
- **FR2.3** Query graph by keyword, entity type, or relatedness
- **FR2.4** Find related nodes within distance
- **Acceptance:** add_node(), add_edge(), query() support SQLite and PostgreSQL

#### FR3: Reasoning
- **FR3.1** Combine episodic + semantic context in single query
- **FR3.2** LLM synthesis to answer questions
- **FR3.3** Summarize recent memories into key insights
- **FR3.4** Entity extraction from unstructured text + auto-ingest
- **Acceptance:** think() returns LLM response; summarize() produces insights

#### FR4: Interfaces
- **FR4.1** CLI commands for all operations
- **FR4.2** MCP tools for Claude integration
- **FR4.3** HTTP API at /api/v1/ with structured errors
- **FR4.4** Pagination on list endpoints
- **FR4.5** Correlation IDs across all requests
- **Acceptance:** All endpoints documented, tested, work without auth

#### FR5: Multi-Tenancy
- **FR5.1** Isolate memories per tenant_id
- **FR5.2** Separate ChromaDB collections per tenant
- **FR5.3** Separate SQLite files per tenant (or PG row-level isolation)
- **FR5.4** Tenant context propagated via JWT or contextvar
- **Acceptance:** Different tenants cannot access each other's data

#### FR6: Authentication & Authorization
- **FR6.1** JWT encode/decode with HS256
- **FR6.2** API key generation + verification (SHA256 hashes)
- **FR6.3** Three roles: ADMIN (write all), AGENT (write episodic/semantic), READER (read-only)
- **FR6.4** /auth/token endpoint to issue tokens
- **FR6.5** Authentication optional (disabled by default)
- **Acceptance:** When enabled, requests without valid Bearer or X-API-Key are rejected

#### FR7: Configuration
- **FR7.1** YAML-based config at ~/.engram/config.yaml
- **FR7.2** Environment variable expansion (${VAR} syntax)
- **FR7.3** ENGRAM_* env var overlay for Docker/CI
- **FR7.4** CLI config get/set with dot notation
- **FR7.5** Type casting (int, bool, str) based on Pydantic fields
- **Acceptance:** Config can be managed via YAML, env vars, or CLI

#### FR8: Caching & Rate Limiting
- **FR8.1** Optional Redis caching for recall/think/query results
- **FR8.2** Per-endpoint TTLs (recall 300s, think 900s, query 300s)
- **FR8.3** Sliding-window rate limiting per tenant
- **FR8.4** Rate limit headers (X-RateLimit-*)
- **FR8.5** Graceful fallback when Redis unavailable
- **Acceptance:** Cache can be toggled on/off; system works without Redis

#### FR9: Observability
- **FR9.1** Structured logging (JSON or text format)
- **FR9.2** Correlation IDs across requests
- **FR9.3** OpenTelemetry instrumentation (optional)
- **FR9.4** JSONL audit logging for compliance
- **FR9.5** Health check endpoints (/health, /health/ready)
- **Acceptance:** Logs are structured; telemetry is optional; health checks reliable

#### FR10: Backup & Disaster Recovery
- **FR10.1** Export all episodic memories to JSON
- **FR10.2** Export semantic graph to JSON
- **FR10.3** Import backup files
- **FR10.4** Per-tenant backup support
- **Acceptance:** Backup/restore roundtrip preserves data integrity

#### FR11: Federation (v0.2)
- **FR11.1** Connect to external memory services via REST, File, Postgres, or MCP adapters
- **FR11.2** Auto-discover services: Cognee, Mem0, LightRAG, OpenClaw, Graphiti
- **FR11.3** Keyword-based routing — internal queries skip external providers (fast path)
- **FR11.4** Parallel fan-out with per-provider timeout (3s default)
- **FR11.5** Circuit breaker: auto-disable provider after N consecutive errors
- **FR11.6** Third-party providers via Python entry_points plugin system
- **FR11.7** /think endpoint merges federated results with internal context
- **FR11.8** GET /providers endpoint returns active provider list + stats (auth required)
- **Acceptance:** Providers registered in config are queried on domain queries; circuit breaker stops failing providers; `engram discover` adds services to config

#### FR12: Security Hardening (v0.2)
- **FR12.1** SSRF prevention in webhooks and auto-discovery
- **FR12.2** SQL injection prevention in PostgresAdapter (parameterised queries)
- **FR12.3** Rate limit identity based on JWT claims (not X-Forwarded-For)
- **FR12.4** Timing-safe comparison for API key verification
- **FR12.5** RBAC path normalization to prevent bypass
- **FR12.6** JWT secret minimum length validation at startup
- **Acceptance:** All six controls verified via unit tests

#### FR13: Ebbinghaus Decay (v0.3.0)
- **FR13.1** Retention score formula: `e^(-decay_rate * days / (1 + 0.1 * access_count))`
- **FR13.2** EpisodicMemory tracks: access_count, last_accessed, decay_rate
- **FR13.3** Config options: episodic.decay_rate, episodic.decay_enabled
- **FR13.4** Env vars: ENGRAM_EPISODIC_DECAY_RATE, ENGRAM_EPISODIC_DECAY_ENABLED
- **FR13.5** CLI: `engram decay --limit N` shows retention report
- **FR13.6** Soft scoring (retention used in composite scoring), hard TTL (expires_at) unchanged
- **Acceptance:** Retention scores calculated correctly; access_count increments on recall

#### FR14: Typed Relationships with Weight (v0.3.0)
- **FR14.1** SemanticEdge now includes: weight: float (default 1.0), attributes: dict
- **FR14.2** SQLite + PostgreSQL backends migrated non-destructively
- **FR14.3** Schema validation is warn-only (never blocks)
- **FR14.4** Weight used in path scoring for weighted graph queries
- **FR14.5** Existing data gets default values on migration
- **Acceptance:** Weighted edges queryable; queries return weight in results

#### FR15: Activation-Based Recall (v0.3.0)
- **FR15.1** Composite score: `similarity*0.5 + retention*0.2 + recency*0.15 + frequency*0.15`
- **FR15.2** Components: similarity (ChromaDB cosine), retention (Ebbinghaus), recency (days), frequency (access_count)
- **FR15.3** ScoringConfig: similarity_weight, retention_weight, recency_weight, frequency_weight
- **FR15.4** Env vars: ENGRAM_SCORING_SIMILARITY_WEIGHT, ENGRAM_SCORING_RETENTION_WEIGHT, etc.
- **FR15.5** search() batch-updates access_count + last_accessed on each recall
- **Acceptance:** recall() returns results ranked by composite score; activation tracking works

#### FR16: Memory Consolidation (v0.3.0)
- **FR16.1** Jaccard similarity clustering of entity/tag sets
- **FR16.2** LLM summarization of each cluster → stored as CONTEXT memory
- **FR16.3** New package: src/engram/consolidation/engine.py
- **FR16.4** ConsolidationEngine: cluster() → summarize() → store()
- **FR16.5** EpisodicMemory fields: consolidation_group, consolidated_into
- **FR16.6** Config: ConsolidationConfig (enabled, min_cluster_size, similarity_threshold)
- **FR16.7** CLI: `engram consolidate --limit N`
- **Acceptance:** consolidate() reduces redundancy; summaries are coherent

#### FR17: OpenClaw Realtime Watcher (v0.3.0)
- **FR17.1** Watchdog/inotify-based watcher for ~/.openclaw/agents/main/sessions/*.jsonl
- **FR17.2** Per-file byte position tracking; parses JSONL format
- **FR17.3** Captures user/assistant messages only (skips toolCall/toolResult/session/custom/error)
- **FR17.4** Cleans tags like [message_id: ...]
- **FR17.5** Integrated into `engram watch --daemon` (runs parallel with inbox watcher)
- **FR17.6** Config: capture.openclaw.enabled, capture.openclaw.sessions_dir
- **FR17.7** Systemd user service for auto-start on boot
- **Acceptance:** New OpenClaw sessions detected in realtime; messages ingested cleanly

#### FR18: Privacy Tag Stripping (v0.3.0)
- **FR18.1** `<private>...</private>` tags stripped to `[REDACTED]` before storage
- **FR18.2** Applied via `sanitize_content()` regex in all remember/remember_batch calls
- **FR18.3** Covers episodic_tools.py and CLI episodic.py commands
- **FR18.4** Config: `sanitize.enabled` boolean (default true)
- **Acceptance:** Private content never stored, audit log shows redaction

#### FR19: Topic Key Upsert (v0.3.0)
- **FR19.1** `topic_key` optional param on `remember()` — same key updates existing memory
- **FR19.2** `revision_count` tracks update count per memory
- **FR19.3** ChromaDB `where` filter lookup for existing topic_key before insert
- **FR19.4** Falls back to new insert if topic_key not found
- **FR19.5** Files: models.py, store.py, episodic_tools.py, cli/episodic.py
- **Acceptance:** `remember(..., topic_key="same")` twice updates single memory with revision_count=2

#### FR20: Progressive Disclosure (MCP) (v0.3.0)
- **FR20.1** `engram_recall` returns compact format by default (id prefix, date, type, snippet)
- **FR20.2** `compact: bool = True` param for backward compatibility with full content
- **FR20.3** New `engram_get_memory(id)` — retrieve full content by ID or 8-char prefix
- **FR20.4** New `engram_timeline(id, window_minutes)` — chronological context around memory
- **FR20.5** Reduces token usage for recall in MCP context
- **Acceptance:** Compact recall returns <200 chars per item; get_memory returns full content; timeline shows ±window_minutes

#### FR21: Session Lifecycle (v0.3.0)
- **FR21.1** New package: `src/engram/session/store.py` — JSON-file backed SessionStore
- **FR21.2** Four new MCP tools: `engram_session_start`, `engram_session_end`, `engram_session_summary`, `engram_session_context`
- **FR21.3** Auto-injects `session_id` into memory metadata from active session
- **FR21.4** Config: `session.sessions_dir` (default ~/.engram/sessions)
- **FR21.5** CLI: `engram session-start`, `engram session-end`
- **FR21.6** Session metadata: id, start_time, end_time, summary, tags
- **Acceptance:** Memories created during session tagged with session_id; sessions queryable by date/summary

#### FR22: Git Sync (v0.3.0)
- **FR22.1** New package: `src/engram/sync/git_sync.py` — export memories to `.engram/` in git repo
- **FR22.2** Compressed JSONL chunks; manifest-based tracking (no re-export/re-import)
- **FR22.3** CLI: `engram sync` (export), `engram sync --import` (re-import), `engram sync --status` (check)
- **FR22.4** Chunk size: 10KB default; incremental updates on each sync
- **FR22.5** Manifest: `.engram/manifest.json` tracks exported memory IDs, last sync timestamp
- **Acceptance:** Memories exported as compressed chunks; git tracks history; manifest prevents duplicates

#### FR23: TUI (Terminal UI) (v0.3.0)
- **FR23.1** New package: `src/engram/tui/` with textual library
- **FR23.2** Screens: Dashboard (stats), Search (query), Recent (timeline), Sessions (active/archived)
- **FR23.3** Vim keys (h/j/k/l), tab navigation (d/s/r/e), drill-down on row select
- **FR23.4** Optional dependency: `pip install engram[tui]`
- **FR23.5** CLI: `engram tui` launches full terminal interface
- **FR23.6** Memory detail view: full content, metadata, related memories
- **Acceptance:** TUI loads within 1s; search returns results in <500ms; drill-down shows full memory

#### FR24: Recall Pipeline (v0.3.1)
- **FR24.1** Query Decision: Skip trivial messages (ok, thanks, hello, emoji) via regex
- **FR24.2** Entity Resolution: Temporal (Vietnamese+English regex, no LLM) + Pronoun (gemini-flash with fallback)
- **FR24.3** Parallel Search: Multi-source (ChromaDB semantic + entity graph + keyword fallback) with fusion
- **FR24.4** Fusion: Dedup by content hash, score ranking
- **FR24.5** Feedback Loop: Detect positive/negative feedback, adjust confidence ±0.15/0.2, auto-delete after 3× negative + low confidence
- **FR24.6** Auto-Memory: Detect save-worthy messages (Save: prefix, identity, preferences, decisions), skip sensitive data
- **FR24.7** Poisoning Guard: Block prompt injection (ignore instructions, you are now, special tokens)
- **FR24.8** Auto-Consolidate: Trigger consolidation after N messages (default 20)
- **FR24.9** Retrieval Audit: JSONL audit log for all recall operations
- **FR24.10** Benchmarking: Run question sets, measure accuracy by type
- **FR24.11** New CLI commands: `engram resolve`, `engram feedback`, `engram audit`, `engram benchmark`, `engram recall --resolve-entities --resolve-temporal`
- **FR24.12** New Config sections: ResolutionConfig, RecallPipelineConfig, FeedbackConfig, IngestionConfig, RetrievalAuditConfig
- **FR24.13** New Models: Entity, ResolvedText, SearchResult, FeedbackType, MemoryCandidate
- **FR24.14** EpisodicMemory new fields: confidence (float, default 1.0), negative_count (int, default 0)
- **Acceptance:** Trivial queries skip <10ms; entity resolution <500ms; parallel search <2s; feedback loop tracks accuracy

#### FR25: Memory Audit Trail (v0.3.2)
- **FR25.1** `log_modification()` records action, before/after values, mod_type, reversible flag, timestamp
- **FR25.2** `read_recent(n)` retrieves last N audit entries
- **FR25.3** Wired into: remember(), delete(), update_metadata(), _update_topic(), cleanup_expired()
- **FR25.4** MODIFICATION_TYPES: memory_create, memory_delete, memory_update, metadata_update, config_change, batch_create, cleanup_expired
- **Acceptance:** Every episodic mutation produces a traceable audit entry with before/after diff

#### FR26: Resource-Aware Retrieval (v0.3.2)
- **FR26.1** ResourceMonitor tracks LLM call success/failure with configurable sliding window
- **FR26.2** 4 tiers: FULL (all features), STANDARD (reduced), BASIC (no synthesis), READONLY (no LLM calls)
- **FR26.3** think() and summarize() check tier before issuing LLM calls
- **FR26.4** BASIC tier returns raw recall results without LLM synthesis (degraded but functional)
- **FR26.5** Auto-recovers to higher tier after 60s cooldown without failures
- **FR26.6** CLI: `engram resource-status` shows current tier and window stats
- **Acceptance:** System degrades gracefully on LLM failures; tier auto-recovers

#### FR27: Data Constitution (v0.3.2)
- **FR27.1** 3 laws enforced: namespace isolation, no fabrication, audit rights
- **FR27.2** Auto-creates ~/.engram/constitution.md on first load
- **FR27.3** SHA-256 hash verification detects tampering
- **FR27.4** Compact prefix injected into reasoning engine and summarize prompts
- **FR27.5** CLI: `engram constitution-status` shows laws + hash verification result
- **Acceptance:** LLM prompts always include constitution prefix; tampered constitution file detected

#### FR28: Consolidation Scheduler (v0.3.2)
- **FR28.1** Asyncio-based recursive setTimeout pattern prevents task overlap
- **FR28.2** 3 default tasks: cleanup_expired (daily), consolidate_memories (every 6h, requires LLM), decay_report (daily)
- **FR28.3** Respects resource tier — skips LLM-dependent tasks on BASIC tier
- **FR28.4** State persisted to ~/.engram/scheduler_state.json (last run, next run, task list)
- **FR28.5** Starts automatically when `engram watch` runs
- **FR28.6** CLI: `engram scheduler-status` shows all tasks with last/next run times
- **Acceptance:** Tasks run on schedule; no overlap; skipped tasks logged; state survives restart

#### FR29: Temporal & Pronoun Resolution (v0.4.0)
- **FR29.1** Temporal Resolver (`src/engram/recall/temporal_resolver.py`): 28 Vietnamese+English patterns
- **FR29.2** Patterns: "hôm nay/hôm qua", "tuần trước/tuần tới", "yesterday/tomorrow", "last week", ISO date → ISO date resolution
- **FR29.3** Wired into store.remember() before episodic insert; adds resolved_dates to memory metadata
- **FR29.4** Pronoun Resolver (`src/engram/recall/pronoun_resolver.py`): LLM-based entity mapping with fallback
- **FR29.5** Resolves: "anh ấy", "he/she/they", "it" → named entity from SemanticGraph context
- **FR29.6** Wired into engram_recall() pipeline; skip if graph empty or no pronouns detected
- **Acceptance:** Relative dates resolve correctly; pronouns match entities in graph; fallback to direct matching

#### FR30: Feedback Loop + Auto-Adjust (v0.4.0)
- **FR30.1** `feedback/auto_adjust.py` tracks positive/negative feedback on memories
- **FR30.2** Confidence adjustment: +0.15 (positive), -0.2 (negative), starts at 1.0 per memory
- **FR30.3** Importance adjustment: +1 (positive), -1 (negative) on memory metadata
- **FR30.4** Auto-delete: If negative_count >= 3 AND confidence < 0.5 → delete from episodic store
- **FR30.5** New POST /api/v1/feedback endpoint with {memory_id, feedback_type: "positive"|"negative"}
- **FR30.6** New MCP tool `engram_feedback(id, feedback_type)` records feedback + returns updated confidence
- **Acceptance:** Feedback persists; confidence/importance update; auto-delete triggers correctly

#### FR31: Fusion Formatter (v0.4.0)
- **FR31.1** `src/engram/recall/fusion_formatter.py` groups recall results by memory type
- **FR31.2** Groups: [preference], [fact], [lesson], [decision], [todo], [error], [workflow], [context]
- **FR31.3** Each group sorted by score descending; compact format by default (id, date, snippet)
- **FR31.4** Wired into engram_recall() after parallel search; optional toggle via config fusion.formatter_enabled
- **FR31.5** LLM reasoning engine uses formatted results with type hints for better context
- **Acceptance:** Results grouped by type; formatting improves LLM reasoning quality; optional config toggle

#### FR32: Graph Visualization UI (v0.4.0)
- **FR32.1** `static/graph.html`: vis-network library, dark theme, interactive entity relationship explorer
- **FR32.2** Features: drag-to-move nodes, click-to-inspect entity details, search by name, zoom/pan
- **FR32.3** Endpoints: GET /graph (HTML page), GET /api/v1/graph/data (JSON nodes/edges for frontend)
- **FR32.4** Node colors by entity type; edge labels show relationship types; physics simulation for layout
- **FR32.5** CLI: `engram graph` launches browser at localhost:8765/graph
- **FR32.6** MCP tool: `engram_get_graph_data(search_keyword)` returns filtered graph JSON
- **Acceptance:** Graph renders in <500ms; interactive exploration works; search filters correctly

#### FR33: WebSocket API (v0.4.1)
- **FR33.1** Route: `GET /ws?token=<JWT>` — JWT verified on WebSocket upgrade; reject 401 on invalid/missing
- **FR33.2** New package `src/engram/ws/`: protocol.py, event_bus.py, connection_manager.py, handler.py
- **FR33.3** **7 commands (client → server):** recall, remember, think, query, ingest, feedback, status
- **FR33.4** **Push events (server → client):** memory.created, memory.deleted, memory.updated, consolidation.completed, error
- **FR33.5** Per-tenant isolation: connection_manager partitions subscriptions by tenant_id; clients only receive their namespace events
- **FR33.6** event_bus.py decouples EpisodicStore mutations from WebSocket delivery; episodic store fires events post-mutation
- **FR33.7** 71 WebSocket tests + 33 P0 gap tests = 104 new tests (total: 893)
- **Acceptance:** Clients receive push events within 100ms of mutation; unauthenticated connections rejected; per-tenant isolation verified

---

### Non-Functional Requirements

#### NFR1: Performance
- **Search:** recall() completes in <100ms (p99) for 10k memories
- **Graph:** query() completes in <50ms for typical graph sizes
- **Reasoning:** think() in <2s (LLM time included)
- **Throughput:** HTTP API handles 100+ req/s with rate limiting
- **Cache hit rate:** >80% for repeated queries

#### NFR2: Scalability
- **Multi-tenant:** Support 1000+ tenants with LRU eviction
- **Memory limit:** Graph cache max 100 instances (configurable)
- **Connection pooling:** asyncpg min 5, max 20 per process
- **Test coverage:** 75%+ code coverage; 545+ tests

#### NFR3: Reliability
- **Availability:** 99.9% uptime target (health checks every 30s)
- **Error handling:** All errors wrapped in structured ErrorCode enum
- **Backward compat:** Existing CLI/MCP tools work unchanged
- **Graceful degradation:** System functional without optional services (Redis, OTel)

#### NFR4: Security
- **Auth:** JWT (HS256) + API keys (SHA256)
- **Content limit:** 10KB default max per memory
- **Audit:** Compliance logging available
- **Secrets:** API keys stored as hashes only
- **RBAC:** Three-tier role enforcement with path normalization
- **SSRF:** Private/loopback IPs blocked in webhooks + discovery
- **Timing:** `hmac.compare_digest` for constant-time key verification
- **JWT secret:** Minimum length enforced at startup

#### NFR5: Usability
- **Documentation:** README + 6 doc files covering setup, API, deployment
- **CLI:** Rich formatting, helpful error messages
- **Config:** Sensible defaults; minimal required setup
- **Error messages:** Include error codes for troubleshooting

#### NFR6: Maintainability
- **Code organization:** 15+ focused modules
- **Type hints:** Full Pydantic models for requests/responses
- **Logging:** Configurable levels, third-party noise silenced
- **Testing:** Unit + integration tests, GitHub Actions CI/CD

---

## Success Criteria

### For v0.2.0 (Completed)
- ✓ 13 phases delivered (10 enterprise + federation + security + bug fixes)
- ✓ 345 tests passing
- ✓ PostgreSQL backend option working
- ✓ Multi-tenant support production-ready
- ✓ Auth optional but enforced correctly
- ✓ Docker image building
- ✓ All 32 bug fixes resolved
- ✓ CI/CD passing on every commit
- ✓ Federation layer (REST/File/Postgres/MCP adapters)
- ✓ Auto-discovery for 5 known services
- ✓ Security hardening (SSRF, SQL injection, timing, RBAC, JWT)

### For v0.4.0 (Current)
- ✓ Temporal Resolution: 28 Vietnamese+English patterns
- ✓ Pronoun Resolution: LLM-based entity mapping
- ✓ Fusion Formatter: group results by type [preference]/[fact]/[lesson]
- ✓ Graph Visualization: interactive vis-network explorer at /graph
- ✓ Feedback Loop + Auto-delete wired end-to-end
- ✓ Embedding key rotation (failover + round-robin)
- ✓ Performance benchmark suite (p50/p95/p99)
- ✓ WebSocket API: 7 commands + push events, per-tenant isolation

### For Future Releases
- Distributed semantic graph (cluster mode)
- Streaming LLM responses for large reasoning tasks
- Graph migrations (SQLite → PostgreSQL)
- Advanced RBAC (resource-level permissions)
- Observability dashboard UI

---

## Configuration Reference

| Config Key | Default | Type | Purpose |
|---|---|---|---|
| episodic.provider | chromadb | str | Vector DB provider |
| episodic.path | ~/.engram/episodic | str | ChromaDB path |
| episodic.namespace | default | str | Collection name (tenant) |
| semantic.provider | sqlite | str | Graph backend (sqlite or postgresql) |
| semantic.path | ~/.engram/semantic.db | str | SQLite path (ignored for PG) |
| semantic.dsn | (env var) | str | PostgreSQL connection string |
| auth.enabled | false | bool | Enable JWT/API key auth |
| auth.jwt_secret | (empty) | str | JWT signing key (min 32 chars when enabled) |
| cache.enabled | false | bool | Enable Redis caching |
| cache.redis_url | redis://localhost:6379/0 | str | Redis connection |
| rate_limit.enabled | false | bool | Enable sliding-window rate limits |
| rate_limit.requests_per_minute | 60 | int | Requests allowed per minute |
| llm.provider | gemini | str | LLM provider (currently Gemini only) |
| llm.model | gemini/gemini-2.0-flash | str | Model name |
| audit.enabled | false | bool | Enable audit logging |
| telemetry.enabled | false | bool | Enable OpenTelemetry |
| providers | [] | list | Federation provider entries (name, type, url/path, enabled) |
| discovery.local | true | bool | Scan localhost ports for known services |
| discovery.hosts | [] | list | Additional remote hosts to scan |
| discovery.endpoints | [] | list | Direct endpoint URLs to probe |
| recall.enabled | true | bool | Enable recall pipeline |
| recall.decision_skip_trivial | true | bool | Skip trivial queries (ok, thanks, emoji) |
| recall.entity_resolution_enabled | true | bool | Enable entity/temporal resolution |
| recall.parallel_search_enabled | true | bool | Enable multi-source search fusion |
| recall.feedback_enabled | true | bool | Enable feedback loop tracking |
| recall.auto_consolidate_threshold | 20 | int | Messages before auto-consolidate |
| recall.retrieval_audit_enabled | true | bool | Enable JSONL audit logging |
| ingestion.auto_memory_enabled | true | bool | Detect save-worthy messages |
| ingestion.guard_enabled | true | bool | Block prompt injection |
| feedback.confidence_positive_delta | 0.15 | float | Confidence boost on positive feedback |
| feedback.confidence_negative_delta | 0.2 | float | Confidence penalty on negative feedback |
| feedback.auto_delete_threshold | 3 | int | Negatives before auto-delete |
| resolution.temporal_enabled | true | bool | Temporal entity resolution |
| resolution.pronoun_enabled | true | bool | Pronoun resolution via LLM |
| embedding.model | gemini-embedding-001 | str | Embedding model (only gemini-embedding-001 supported) |
| embedding.key_strategy | failover | str | Key rotation strategy: `failover` or `round-robin` |
| fusion.formatter_enabled | true | bool | Group recall results by type |
| graph.visualization_enabled | true | bool | Enable interactive graph UI at /graph |
| rate_limit.fail_open | true | bool | Allow requests through on Redis failure |

---

## Dependencies & Versions

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11+ | |
| typer | 0.9.0+ | CLI |
| fastapi | 0.100.0+ | HTTP API |
| chromadb | 0.4.0+ | Episodic store |
| networkx | 3.0+ | Graph in-memory |
| pydantic | 2.0+ | Validation |
| litellm | 1.0.0+ | LLM provider abstraction |
| asyncpg | 0.29.0+ | PostgreSQL driver (optional) |
| redis | 5.0.0+ | Caching/rate limiting (optional) |
| opentelemetry-* | 1.20+ | Observability (optional, telemetry extra) |

---

## Installation & Quick Start

```bash
# Install from PyPI (when available)
pip install engram

# Or from source with dev extras
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram
pip install -e ".[dev]"

# Set API key
export GEMINI_API_KEY="your-key"

# Try CLI
engram remember "Deployed v1.0 to prod"
engram recall "deployment"
engram think "What deployments have we done?"

# Or start HTTP API
engram serve
# Then: curl http://localhost:8765/health
```

---

## Support & Community

- **GitHub:** https://github.com/docaohieu2808/Engram-Mem
- **Issues:** Bug reports and feature requests
- **Discussions:** Community support and ideas
- **Documentation:** ./docs/ directory in repository

---

## Roadmap & Future Work

- v0.3: Advanced query DSL (Cypher-like), GraphQL endpoint, streaming responses
- v0.4: Multi-node clustering, distributed semantic graph, Raft consensus
- v0.5: Observability dashboard UI, custom embedding models
- v1.0: Production release, marketplace, enterprise SLA

---

## Change Log

**v0.4.0** (2026-02-27) — Intelligence Layer + Graph Visualization + Benchmark Suite
- Temporal Resolution: 28 Vietnamese+English patterns resolve "hôm nay/yesterday" → ISO dates before storing
- Pronoun Resolution: "anh ấy/he/she" → named entity from graph context, LLM-based fallback
- Feedback Loop + Auto-adjust: confidence ±0.15/0.2, importance ±1, auto-delete on 3× negative
- Fusion Formatter: group recall results by type [preference]/[fact]/[lesson] for LLM context
- Graph Visualization: interactive entity relationship explorer at /graph, vis-network dark theme, search, click-to-inspect
- WebSocket API: bidirectional `/ws?token=JWT`; 7 commands + push events; per-tenant isolation via event_bus
- Embedding key rotation: failover/round-robin strategy, GEMINI_API_KEY_FALLBACK support
- Performance benchmark suite: p50/p95/p99 (tests/benchmark_performance.py)
- 7 orphaned modules wired: guard, decision, telemetry, temporal search, parallel search, auto memory, auto consolidation
- 3 critical bug fixes: FTS5 thread safety, OOM pagination, rate limiter race condition (fail_open added)
- New CLI: `engram graph`
- New MCP tools: `engram_get_graph_data`, enhanced `engram_feedback`
- New API: POST /api/v1/feedback, GET /api/v1/graph/data, GET /graph, GET /ws
- 894+ tests

**v0.3.2** (2026-02-25) — Brain Features
- Memory Audit Trail: traceable before/after log for every episodic mutation
- Resource-Aware Retrieval: 4-tier degradation (FULL→STANDARD→BASIC→READONLY) with auto-recovery
- Data Constitution: 3-law governance injected into every LLM prompt, SHA-256 tamper detection
- Consolidation Scheduler: asyncio background tasks (cleanup daily, consolidate 6h, decay daily), tier-aware
- New CLI: `engram resource-status`, `engram constitution-status`, `engram scheduler-status`
- 545+ tests (39 new)

**v0.3.1** (2026-02-25) — Recall Pipeline Upgrade
- Query Decision: Skip trivial messages (ok, thanks, hello, emoji) via regex
- Entity Resolution: Temporal (Vietnamese+English) + pronoun resolution (LLM with fallback)
- Parallel Search: Multi-source fusion (ChromaDB semantic + entity graph + keyword fallback)
- Learning Pipeline: Feedback loop (±0.15/0.2 confidence), auto-delete after 3× negative
- Auto-Memory: Detect save-worthy messages, skip sensitive data
- Poisoning Guard: Block prompt injection (ignore instructions, special tokens)
- Auto-Consolidate: Trigger after N messages (default 20)
- Retrieval Audit: JSONL audit log for all recall operations
- Benchmarking: Run question sets, measure accuracy by type
- New CLI: `engram resolve`, `engram feedback`, `engram audit`, `engram benchmark`
- 506+ tests (up from 380)

**v0.3.0** (2026-02-25) — Activation-Based Recall + Consolidation + TUI
- Privacy tag stripping: `<private>...</private>` → `[REDACTED]` before storage
- Topic key upsert: same topic_key updates existing memory with revision_count tracking
- Progressive disclosure: compact recall + engram_get_memory + engram_timeline MCP tools
- Session lifecycle: engram_session_start/end/summary/context with auto session_id tagging
- Git sync: incremental compressed JSONL chunks to .engram/ with manifest tracking
- TUI: interactive terminal interface (textual library) with Search/Recent/Sessions screens
- 380+ tests (up from 345)

**v0.2.0** (2026-02-25) — Enterprise + Federation + Security
- Federation layer: REST/File/Postgres/MCP provider adapters
- Auto-discovery for Cognee, Mem0, LightRAG, OpenClaw, Graphiti
- Smart query router (keyword-based, internal vs. domain)
- Circuit breaker per provider; entry_points plugin system
- CLI: `engram discover`, `engram providers list/test/stats/add`
- Security hardening: SSRF, SQL injection, timing attacks, RBAC, JWT secret validation
- 11 bug fixes: federated think, UTC datetime, graph O(V), node cache, McpAdapter cleanup
- 345 tests (up from 270)

**v0.2.0** (2026-02-24) — Enterprise Upgrade Complete
- Config + Logging foundation, PostgreSQL semantic graph backend
- JWT + API key auth with RBAC, multi-tenancy support
- Redis caching + rate limiting, API versioning (/api/v1/)
- OpenTelemetry + JSONL audit, Docker + GitHub Actions
- Health checks + backup/restore, test expansion (270 tests)
- 21 bug fixes (3 critical, 7 high, 11 medium)

**v0.1.0** (2025-XX-XX) — Initial Release
- Dual-memory architecture, CLI, MCP, HTTP API
- ChromaDB + SQLite graph, Gemini LLM integration

