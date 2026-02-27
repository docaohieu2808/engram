# Engram Codebase Summary

## Overview
Engram v0.4.0 is a dual-memory AI agent system with intelligent recall pipeline, brain-level safety features, automated maintenance, interactive graph visualization, real-time WebSocket API, and embedding key rotation. Combines episodic (vector) and semantic (NetworkX MultiDiGraph) memory, LLM reasoning, activation/consolidation, session lifecycle, terminal UI, advanced query processing, audit trail, resource-aware degradation, data constitution, background scheduler, temporal/pronoun resolution, feedback loop, fusion formatting, bidirectional WebSocket push, and performance benchmarking. ~5000+ LoC, 894+ tests, Python 3.11+.

**Top files:** `capture/server.py` (API), `episodic/store.py` (vector store), `episodic/embeddings.py` (key rotation), `config.py` (configuration), `constitution.py` (LLM governance), `resource_tier.py` (degradation), `scheduler.py` (background tasks), `audit.py` (mutation log), `recall/temporal_resolver.py` (date resolution), `recall/pronoun_resolver.py` (entity mapping), `recall/fusion_formatter.py` (result grouping), `feedback/auto_adjust.py` (confidence tracking), `ws/handler.py` (WebSocket), `tests/benchmark_performance.py` (perf benchmarks).

---

## Directory Structure

### `src/engram/` — Core modules

#### `config.py` (280 LOC)
YAML-based configuration system with env var expansion and overlay.
- **Config models:** EpisodicConfig, SemanticConfig, AuthConfig, TelemetryConfig, CacheConfig, RateLimitConfig, AuditConfig
- **Env var expansion:** `${VAR}` syntax in YAML, per-field type casting
- **Env var overlay:** ENGRAM_* variables override YAML (e.g., ENGRAM_LLM_MODEL)
- **Helpers:** load_config(), save_config(), get/set via dot notation (e.g., "llm.model")

#### `auth.py` (200 LOC)
JWT and API key authentication for HTTP API. **Disabled by default** for backward compat.
- **JWT:** create_jwt(), verify_jwt() with HS256 algorithm
- **API Keys:** generated as SHA256 hashes, stored in `~/.engram/api_keys.json`
- **RBAC:** Three roles — ADMIN (read+write), AGENT (write episodic/semantic), READER (read-only)
- **FastAPI dependency:** get_auth_context() — checks Bearer token or X-API-Key header
- **Public paths:** /health, /api/v1/auth/token never require auth

#### `auth_models.py` (50 LOC)
Pydantic models for auth: TokenPayload, APIKeyRecord, AuthContext, Role enum.

#### `tenant.py` (165 LOC)
Tenant isolation via contextvars. One tenant per request/CLI session.
- **TenantContext:** get/set tenant_id globally; defaults to "default"
- **StoreFactory:** Creates and caches per-tenant EpisodicStore and SemanticGraph
  - LRU eviction: 100 graph max; 1000 episodic stores
  - SQLite: separate .{tenant_id}.db files
  - PostgreSQL: share connection pool, tenant_id column isolates data
- **Validation:** tenant_id must match [a-zA-Z0-9_-]{1,64}

#### `errors.py` (50 LOC)
Structured error codes and ErrorResponse schema.
- **ErrorCode enum:** INVALID_REQUEST, UNAUTHORIZED, FORBIDDEN, NOT_FOUND, RATE_LIMITED, INTERNAL_ERROR, etc.
- **EngramError:** Custom exception with error_code + message
- **ErrorResponse:** JSON response format with code + message

#### `logging_setup.py` (100 LOC)
Logging configuration with JSON and text formats, correlation IDs.
- **Formats:** "text" (default), "json" (structured)
- **Correlation ID:** ContextVar propagated through all logs
- **Third-party silence:** litellm, chromadb, httpx set to WARNING

#### `cache.py` (150 LOC)
Redis-backed caching for API results (optional, disabled by default).
- **TTLs:** recall_ttl (300s), think_ttl (900s), query_ttl (300s)
- **Keys:** engram:{tenant_id}:{endpoint}:{hash(params)}
- **Graceful fallback:** System works without Redis

#### `rate_limiter.py` (100 LOC)
Sliding-window rate limiting via Redis (optional).
- **Config:** requests_per_minute, burst size
- **Per-tenant:** Limits keyed by tenant_id or IP
- **Headers:** X-RateLimit-Limit, -Remaining, -Reset; Retry-After

#### `audit.py` (100 LOC — extended v0.3.2)
Audit logging for compliance (optional, disabled by default) + memory mutation trail.
- **Backend:** File-based JSONL (one record per line)
- **Records:** {timestamp, tenant_id, user_id, action, resource, status, metadata}
- **Path:** ~/.engram/audit.jsonl (configurable)
- **New (v0.3.2):** `log_modification(mod_type, before, after, reversible)` — per-mutation trail
- **New (v0.3.2):** `read_recent(n)` — retrieve last N audit entries
- **MODIFICATION_TYPES:** memory_create, memory_delete, memory_update, metadata_update, config_change, batch_create, cleanup_expired

#### `resource_tier.py` (120 LOC) — **NEW (v0.3.2)**
Resource-aware LLM call management with 4-tier degradation.
- **ResourceMonitor:** Sliding-window success/failure tracker for LLM calls
- **Tiers:** FULL → STANDARD → BASIC → READONLY
- **BASIC behavior:** Returns raw recall results without LLM synthesis
- **Auto-recovery:** Promotes tier after 60s cooldown without failures
- **Integration:** think() and summarize() in reasoning engine check tier before LLM calls
- **CLI:** `engram resource-status`

#### `constitution.py` (100 LOC) — **NEW (v0.3.2)**
Data governance via constitutional constraints injected into LLM prompts.
- **3 Laws:** namespace isolation, no fabrication, audit rights
- **Auto-creation:** ~/.engram/constitution.md generated on first load
- **Tamper detection:** SHA-256 hash stored alongside constitution file
- **Prompt injection:** Compact prefix prepended to all reasoning + summarize calls
- **CLI:** `engram constitution-status`

#### `scheduler.py` (150 LOC) — **NEW (v0.3.2)**
Asyncio-based background task scheduler for periodic maintenance.
- **Pattern:** Recursive setTimeout (each task schedules its own next run after completion)
- **Default tasks:** cleanup_expired (daily), consolidate_memories (6h, LLM), decay_report (daily)
- **Tier awareness:** Skips LLM-dependent tasks on BASIC tier
- **Persistence:** ~/.engram/scheduler_state.json tracks last run, next run, task history
- **Auto-start:** Launched by `engram watch`
- **CLI:** `engram scheduler-status`

#### `backup.py` (150 LOC)
Backup and restore utilities for memory snapshots.
- **Functions:** backup_episodic(), restore_episodic(), backup_graph(), restore_graph()
- **Format:** JSON dumps of memory + graph structure
- **CLI:** engram backup / engram restore commands

#### `health.py` (80 LOC)
Health checks for liveness and readiness probes.
- **Checks:** episodic store, semantic graph, Redis (if enabled), PostgreSQL (if enabled)
- **Endpoints:** /health (liveness), /health/ready (readiness)

---

### `src/engram/episodic/` — Vector memory (ChromaDB)

#### `store.py` (420 LOC)
ChromaDB vector database wrapper.
- **Collections:** One per namespace (tenant_id)
- **Chunking:** Auto-splits long content (>1000 chars) into 800-char chunks with overlap
- **Metadata:** memory_type, priority, tags, expires, created_at, updated_at, access_count, last_accessed, decay_rate, **topic_key, revision_count, session_id** (v0.3.0)
- **Methods:** remember(), recall(), cleanup(), get(), count()
- **Hooks:** Fires on_remember webhook after insert (fire-and-forget)
- **Fallback embedding:** Gemini (default, 3072 dims) or all-MiniLM-L6-v2 (384 dims)
- **New (v0.3.0):** Topic key upsert — same topic_key updates existing memory with revision_count
- **New (v0.3.0):** Session_id auto-injection from active SessionStore
- **New (v0.3.0):** Batch updates to access_count + last_accessed on recall()
- **New (v0.3.2):** Audit trail wired into remember(), delete(), update_metadata(), _update_topic(), cleanup_expired()

#### `embeddings.py` (93 LOC) — **NEW (v0.4.0)**
Embedding helpers with multi-key rotation for Gemini API.
- **Model:** `gemini-embedding-001` exclusively (3072 dimensions)
- **Key sources:** `GEMINI_API_KEY` (primary) + `GEMINI_API_KEY_FALLBACK` (optional secondary)
- **Strategies:** `failover` (default — primary first, fallback on auth error) or `round-robin` (rotate evenly)
- **Config:** `embedding.key_strategy` in config.yaml or `GEMINI_KEY_STRATEGY` env var
- **Error handling:** Tries each key in order; raises `RuntimeError` if all keys fail
- **`EMBEDDING_DIM`:** Module-level constant = 3072

#### `search.py` (180 LOC)
Search utilities: embedding generation, similarity scoring, activation-based recall.
- **search_similar():** Vector search with optional tag/type filters
- **chunk_text():** Split text with configurable chunk size + overlap
- **New (v0.3.0):** Composite scoring: similarity (0.5) + retention (0.2) + recency (0.15) + frequency (0.15)
- **New (v0.3.0):** ScoringConfig with configurable weights

---

### `src/engram/semantic/` — Knowledge graph (SQLite/PostgreSQL + NetworkX)

#### `backend.py` (80 LOC)
Abstract backend interface — supports SQLite and PostgreSQL.
- **Methods:** create_graph(), load_graph(), save_graph(), close()

#### `sqlite_backend.py` (220 LOC)
SQLite semantic graph backend.
- **Schema:** Nodes (id, key, type, attributes_json), Edges (source_id, target_id, relation, metadata_json, **weight, attributes**) (v0.3.0)
- **File:** ~/.engram/semantic.db or tenant-specific .{tenant_id}.db
- **Operations:** Add/remove nodes & edges, find related, query
- **New (v0.3.0):** Non-destructive migration adds weight/attributes columns with defaults

#### `pg_backend.py` (220 LOC)
PostgreSQL semantic graph backend (Phase 2, enterprise upgrade).
- **Schema:** Same tables, adds tenant_id column for multi-tenant isolation, **weight + attributes** (v0.3.0)
- **Connection pool:** asyncpg, configurable pool_min (5) and pool_max (20)
- **Async:** Full async/await support via asyncpg
- **New (v0.3.0):** Non-destructive migration maintains backward compatibility

#### `graph.py` (280 LOC)
NetworkX wrapper for in-memory graph operations.
- **Nodes:** Entities with type and attributes
- **Edges:** Relationships with type metadata, **weight: float (default 1.0), attributes: dict** (v0.3.0)
- **Queries:** Related entities, path finding, subgraph extraction
- **Methods:** add_node(), add_edge(), query(), relate(), remove_*()
- **New (v0.3.0):** Weight-aware scoring in path queries

#### `query.py` (140 LOC)
Graph query DSL and utilities.
- **query():** Find nodes by keyword/type/related_to with pagination
- **Filtering:** By node type, relation type, connection distance
- **Results:** Nodes, edges, connectivity stats
- **New (v0.3.0):** Weight-aware result scoring

---

### `src/engram/recall/` — Intelligent Query Processing (v0.3.1)

#### `decision.py` (50 LOC) — **NEW (v0.3.1)**
Query decision engine for trivial message detection.
- Regex patterns: "ok", "thanks", "hello", emoji
- Returns empty result <10ms without vector search
- Prevents processing noise

#### `entity_resolver.py` (180 LOC) — **NEW (v0.3.1)**
Entity and temporal resolution for context extraction.
- **Temporal:** Vietnamese+English date regex (no LLM cost)
- **Pronoun:** LLM-based (gemini-flash) with fallback to direct match
- Methods: resolve_temporal(), resolve_pronoun(), resolve_text()

#### `parallel_search.py` (200 LOC) — **NEW (v0.3.1)**
Multi-source search with fusion and deduplication.
- **Sources:** ChromaDB semantic, entity graph keyword, keyword fallback
- **Fusion:** Parallel async, dedup by content hash, score ranking
- **SearchResult:** content, score (0-1), source, metadata, resolved_entities
- Top-K selection after merging

#### `temporal_resolver.py` (150 LOC) — **NEW (v0.4.0)**
Vietnamese+English temporal pattern resolution.
- 28 patterns: "hôm nay/hôm qua", "tuần trước/tuần tới", "yesterday/tomorrow", "last week", etc.
- Regex-based (no LLM cost); returns ISO date or None
- Wired into store.remember() before episodic insert
- Metadata: resolved_dates (list of {pattern, resolved_date})
- Config: resolution.temporal_enabled (default true)

#### `pronoun_resolver.py` (120 LOC) — **NEW (v0.4.0)**
LLM-based pronoun resolution with fallback.
- Resolves: "anh ấy", "he/she/they", "it" → named entity from SemanticGraph
- LLM call (gemini-flash) with entity context; fallback to direct keyword matching
- Methods: resolve_pronouns(text, graph_entities) → {pronoun: resolved_entity, confidence}
- Wired into engram_recall() and engram_think()
- Config: resolution.pronoun_enabled (default true)

#### `fusion_formatter.py` (140 LOC) — **NEW (v0.4.0)**
Group recall results by memory type for structured LLM context.
- Groups: [preference], [fact], [lesson], [decision], [todo], [error], [workflow], [context]
- Each group sorted by score descending; compact format (id, date, snippet)
- Methods: format_results(search_results) → {grouped_results: {type: [results]}}
- Wired into engram_recall() after parallel_search + fusion
- LLM reasoning uses formatted structure with type hints
- Config: fusion.formatter_enabled (default true)

---

### `src/engram/feedback/` — Adaptive Learning (v0.3.1+)

#### `loop.py` (120 LOC) — **NEW (v0.3.1)**
Feedback loop for tracking and improving recall accuracy.
- Track positive/negative feedback on memories
- Adjust confidence: +0.15 (positive), -0.2 (negative)
- Auto-delete: if negative_count >= 3 AND confidence < threshold
- Methods: record_feedback(), get_confidence(), should_auto_delete()

#### `auto_adjust.py` (150 LOC) — **NEW (v0.4.0)**
Enhanced feedback system with confidence and importance tracking.
- Confidence adjustment: +0.15 (positive), -0.2 (negative), range [0.0, 1.0]
- Importance adjustment: +1 (positive), -1 (negative) on memory.priority
- Auto-delete: if negative_count >= 3 AND confidence < 0.5
- Methods: record_feedback(memory_id, type) → updated_confidence
- Integration: POST /api/v1/feedback endpoint, MCP tool `engram_feedback`
- Persistence: Updates stored directly on EpisodicMemory records
- Config: feedback.confidence_positive_delta (0.15), confidence_negative_delta (0.2), auto_delete_threshold (3)

---

### `src/engram/ingestion/` — Advanced Ingestion (v0.3.1)

#### `auto_memory.py` (130 LOC) — **NEW (v0.3.1)**
Auto-detection of save-worthy messages.
- Patterns: "Save: " prefix, identity (I am, my name), preferences (like, prefer), decisions (decided)
- Sensitive data skip: passwords, API keys, tokens, PII regex
- Auto-remember without user intervention
- Config: IngestionConfig (enabled, sensitive_patterns)

#### `guard.py` (100 LOC) — **NEW (v0.3.1)**
Prompt injection prevention.
- Block patterns: "ignore instructions", "you are now", "forget", special tokens
- Filter before storage, log attempts
- Protects semantic graph integrity
- Config: GuardConfig (enabled, blocked_patterns)

---

### `src/engram/consolidation/auto_trigger.py` — Auto-Consolidate (v0.3.1)

#### `auto_trigger.py` (80 LOC) — **NEW (v0.3.1)**
Auto-trigger consolidation after message threshold.
- Trigger after N messages (default 20, configurable)
- Async, non-blocking
- Reduces memory redundancy over time
- Methods: should_consolidate(), trigger_if_needed()

---

### `src/engram/retrieval_audit_log.py` — Query Logging (v0.3.1)

#### `retrieval_audit_log.py` (100 LOC) — **NEW (v0.3.1)**
JSONL append-only audit logging for all recall operations.
- Tracks: timestamp, query, results_count, source, latency_ms, resolved_entities
- Enables debugging and pattern analysis
- Path: ~/.engram/retrieval_audit.jsonl (configurable)
- Methods: log_recall(), log_resolution(), log_feedback()

---

### `src/engram/benchmark/` — Accuracy Measurement (v0.3.1)

#### `runner.py` (150 LOC) — **NEW (v0.3.1)**
Benchmark framework for measuring recall accuracy.
- Load question sets (JSON: {question, golden_answer, type})
- Compare model answers vs. golden answers
- Metrics: exact match, semantic similarity (cosine), F1 score
- Report by question type (factual, reasoning, etc.)
- Methods: run_benchmark(), evaluate_answer(), generate_report()

---

### `src/engram/providers/` — Federation layer (v0.2)

#### `base.py` (120 LOC)
Abstract base class and data models for all external memory providers.
- **ProviderResult:** `{content, score, source, metadata}`
- **ProviderStats:** per-provider `{query_count, hit_count, error_count, consecutive_errors, total_latency_ms}`
- **MemoryProvider:** ABC with `search()`, `health()`, `add()`, `tracked_search()`, circuit breaker (auto-disable after `max_consecutive_errors`)

#### `registry.py` (155 LOC)
Loads and manages all provider instances.
- **ProviderRegistry:** `load_from_config()`, `register()`, `get_active()`, `close_all()`
- Third-party adapters via `entry_points(group="engram.providers")`
- Lazy import of built-in adapters

#### `router.py` (115 LOC)
Keyword-based query classification and fan-out.
- **classify_query():** returns `"internal"` (skip providers) or `"domain"` (fan-out)
- **federated_search():** parallel async queries with `asyncio.gather`, per-provider timeout (3s default), dedup + score sort
- Supports English + Vietnamese keywords

#### `discovery.py` (270 LOC)
Auto-discover external memory services.
- **discover():** Four tiers — port scan, remote hosts, direct endpoints, MCP config files
- **Known services:** Cognee (8000), Mem0 (8080), LightRAG (9520), OpenClaw (file), Graphiti (8000)
- **SSRF protection:** `_is_safe_discovery_host()` blocks private/loopback IPs on remote hosts
- **MCP config scan:** Parses `~/.claude/settings.json`, `~/.cursor/settings.json` for memory-related MCP servers

#### `rest_adapter.py` (130 LOC)
HTTP REST adapter for Cognee, Mem0, LightRAG, Graphiti.
- Configurable `search_endpoint`, `search_method`, `search_body` template, `result_path`
- `result_path` supports dot-bracket notation (e.g. `data[].text`)
- SSRF protection on webhook URLs

#### `file_adapter.py`
Reads local markdown/text files matching a glob pattern.
- Used for OpenClaw workspace memory (`~/.openclaw/workspace/memory/*.md`)
- Returns file content as ProviderResult with filename as source

#### `postgres_adapter.py`
Queries an external PostgreSQL table with parameterised SQL.
- `search_query` must use `$1` placeholder for query text
- SQL injection safe by design

#### `mcp_adapter.py`
Spawns an MCP server subprocess and calls a tool via stdio.
- Safe cleanup on partial init failure (prevents resource leaks)
- `tool_name` defaults to `search_memory`

---

### `src/engram/reasoning/` — LLM synthesis

#### `engine.py` (200 LOC — extended v0.3.2)
Combines episodic + semantic memory, feeds to LLM for synthesis.
- **Providers:** Gemini (default) via litellm
- **Operations:** think() (Q&A), summarize() (key insights), ingest() (extract entities + remember)
- **Prompts:** Calibrated for dual-memory reasoning
- **Caching:** Optional Redis caching for expensive LLM calls
- **New (v0.3.2):** ResourceMonitor tier check before LLM calls; BASIC tier returns raw recall
- **New (v0.3.2):** Constitution prefix injected into think() and summarize() prompts

---

### `src/engram/consolidation/` — Memory Consolidation (v0.3.0)

#### `engine.py` (200 LOC) — **NEW (v0.3.0)**
Consolidates redundant episodic memories via Jaccard clustering + LLM summarization.
- **ConsolidationEngine:** Clusters memories by entity/tag set similarity
  - `consolidate(limit, similarity_threshold)` → groups + summarizes clusters
  - Stores results as CONTEXT memory type with consolidation_group tracking
- **Clustering:** Jaccard similarity on entity/tag sets
- **LLM Synthesis:** Summarizes each cluster into single insight
- **Config:** ConsolidationConfig (enabled, min_cluster_size, similarity_threshold)
- **CLI:** `engram consolidate --limit N` shows consolidation report

---

### `src/engram/sanitize.py` — Content Sanitization (v0.3.0)

#### `sanitize.py` (80 LOC) — **NEW (v0.3.0)**
Strips sensitive content from episodic memories before storage.
- **SanitizeConfig:** enabled (bool, default true)
- **sanitize_content():** Regex-based removal of `<private>...</private>` tags → `[REDACTED]`
- **Applied in:** episodic_tools.py, CLI episodic.py, before ChromaDB storage
- **Use case:** Prevent accidental storage of passwords, API keys, personal info

---

### `src/engram/session/` — Session Lifecycle Management (v0.3.0)

#### `store.py` (150 LOC) — **NEW (v0.3.0)**
JSON-file backed session store for tracking active sessions.
- **SessionStore:** Manages session creation, state, summaries
- **Session model:** id (uuid), start_time, end_time, summary, tags, metadata
- **Methods:** start_session(), end_session(), get_active(), get_summary(), update_metadata()
- **Storage:** ~/.engram/sessions/ (JSON files, one per session)
- **Integration:** Auto-injects session_id into memory metadata during active session
- **MCP tools:** engram_session_start, engram_session_end, engram_session_summary, engram_session_context
- **CLI:** engram session-start, engram session-end

---

### `src/engram/sync/` — Git Repository Sync (v0.3.0)

#### `git_sync.py` (120 LOC) — **NEW (v0.3.0)**
Exports episodic memories to git repo as compressed JSONL chunks.
- **GitSync:** Manages incremental export/import of memories
- **Chunk format:** Compressed JSONL (10KB chunks default), stored in .engram/ directory
- **Manifest tracking:** .engram/manifest.json tracks exported IDs + timestamps
- **Methods:** export(), import(), get_status()
- **Features:** Deduplication (manifest prevents re-export), incremental updates
- **CLI:** engram sync, engram sync --import, engram sync --status
- **Use case:** Version control for memory archives, backup to git

---

### `src/engram/tui/` — Terminal User Interface (v0.3.0)

#### TUI Package (150 LOC total) — **NEW (v0.3.0)**
Interactive terminal interface via textual library.
- **Screens:** Dashboard (memory stats), Search (query with live results), Recent (timeline), Sessions (active/archived)
- **Navigation:** Vim keys (h/j/k/l), tab keys (d/s/r/e), row select for drill-down
- **Features:** Live search with debouncing, memory detail view, session context
- **Performance:** Loads in <1s, search results <500ms
- **Dependencies:** textual (optional via engram[tui] install)
- **CLI:** engram tui launches full interface
- **Files:** dashboard.py (home screen), search.py, recent.py, sessions.py, app.py (main)

---

### `static/graph.html` & Graph Visualization API (v0.4.0)

#### `graph.html` (400 LOC) — **NEW (v0.4.0)**
Interactive entity relationship explorer using vis-network library.
- **Frontend:** vis-network (force-directed graph), dark theme, responsive layout
- **Features:** Drag-to-move nodes, click-to-inspect entity details, search by name, zoom/pan
- **Node styling:** Colors by entity type, size by connection count
- **Edge labels:** Show relationship types (uses, depends_on, etc.)
- **Performance:** Renders graphs with 1000+ nodes in <500ms
- **Search:** Client-side filtering by entity name with real-time update
- **Data source:** GET /api/v1/graph/data endpoint (JSON nodes/edges)
- **Accessibility:** Keyboard navigation, context-aware tooltips

#### `server.py` routes (NEW in v0.4.0)
- `GET /graph` — Serves graph.html with node/edge data embedded
- `GET /api/v1/graph/data` — Returns JSON {nodes: [...], edges: [...]} for frontend
- Query params: `?search=keyword` filters entities by name pattern
- Response format: `{data: {nodes: [...], edges: [...]}, meta: {...}}`

#### `mcp/semantic_tools.py` (UPDATED v0.4.0)
- New tool: `engram_get_graph_data(search_keyword)` — retrieve filtered graph JSON
- Useful for Claude integration to visualize entity relationships
- Returns: nodes (id, label, type, size) + edges (from, to, relation, weight)

#### CLI Integration
- `engram graph [--search keyword]` — Launches browser at localhost:8765/graph
- Opens graph.html with optional search filter pre-applied

---

### `src/engram/ws/` — WebSocket API (v0.4.1)

#### `protocol.py` (60 LOC) — **NEW (v0.4.1)**
WebSocket message protocol definitions.
- Command schema: {type, payload, request_id}
- Event types: memory.created, memory.deleted, memory.updated, consolidation.completed, error
- JSON serialization/deserialization helpers

#### `event_bus.py` (80 LOC) — **NEW (v0.4.1)**
In-process pub/sub event bus for memory change propagation.
- Per-tenant subscriber channels
- Async broadcast to all connections in a namespace
- Decouples episodic/semantic mutations from WebSocket delivery

#### `connection_manager.py` (100 LOC) — **NEW (v0.4.1)**
Active WebSocket connection lifecycle management.
- Per-tenant connection registry
- Connect/disconnect with cleanup on client drop
- Broadcast helpers for targeted or all-tenant delivery

#### `handler.py` (120 LOC) — **NEW (v0.4.1)**
FastAPI WebSocket route handler.
- Route: `GET /ws?token=JWT`
- Authenticates client via JWT on upgrade; rejects invalid/missing tokens
- Dispatches 7 commands to existing memory operations
- Subscribes connection to event_bus for push events

---

### `src/engram/capture/` — External agent integration

#### `server.py` (650 LOC) — HTTP API
FastAPI endpoints for all operations. Multi-tenant, versioned, fully authenticated.
- **Routes:**
  - `POST /api/v1/remember` — Store episodic memory
  - `GET /api/v1/recall` — Search episodic (with pagination)
  - `POST /api/v1/think` — LLM reasoning
  - `POST /api/v1/ingest` — Extract entities + store memories
  - `GET /api/v1/query` — Graph query
  - `POST /api/v1/feedback` — Record positive/negative feedback (NEW v0.4.0)
  - `POST /api/v1/cleanup` — Delete expired
  - `POST /api/v1/summarize` — LLM synthesis
  - `POST /api/v1/auth/token` — Issue JWT
  - `GET /api/v1/health` — Liveness
  - `GET /api/v1/graph/data` — Graph data JSON (NEW v0.4.0)
  - `GET /graph` — Interactive graph HTML (NEW v0.4.0)
  - `POST /api/v1/backup` — Snapshot memory
  - `POST /api/v1/restore` — Load backup
- **Middleware:** CorrelationIdMiddleware (X-Correlation-ID), RateLimitMiddleware (per-tenant sliding-window)
- **Versioning:** /api/v1/ prefix; legacy /remember routes redirect
- **Response format:** Structured ErrorResponse on failures
- **Pagination:** offset + limit on /recall and /query
- **NEW (v0.4.0):** Feedback endpoint accepts {memory_id, feedback_type} and returns {confidence, importance}

#### `extractor.py` (150 LOC)
Entity extraction from unstructured text (LLM-powered).
- **Extract:** Entities with types, relationships
- **Ingest:** Auto-create graph nodes, emit memory records
- **Schema validation:** Against configured schema (devops/marketing/personal)

#### `watcher.py` (100 LOC)
File system watcher for inbox auto-ingest.
- **Polls** ~/.engram/inbox/ every N seconds for new .json files
- **Processes:** Message arrays, extracts entities, stores memories
- **Daemon:** Runs via engram watch --daemon

#### `openclaw_watcher.py` (150 LOC) — **NEW (v0.3.0)**
Watchdog/inotify-based realtime watcher for OpenClaw session streams.
- **Target:** `~/.openclaw/agents/main/sessions/*.jsonl`
- **Parser:** Reads JSONL format, tracks per-file byte positions
- **Filters:** Captures user/assistant messages only (skips toolCall/toolResult/session/custom/error)
- **Cleaner:** Removes message ID tags like `[message_id: ...]`
- **Integration:** Runs parallel with inbox watcher in `engram watch --daemon`
- **Config:** `capture.openclaw.enabled`, `capture.openclaw.sessions_dir`
- **Systemd:** Auto-start service available for boot integration

---

### `src/engram/cli/` — Command-line interface (Typer)

#### `episodic.py` (150 LOC)
- `engram remember` — Store memory with type/priority/tags
- `engram recall` — Search memories by similarity
- `engram cleanup` — Delete expired

#### `semantic.py` (150 LOC)
- `engram add node <name> --type <Type>` — Create entity
- `engram add edge <from> <to> --relation <rel>` — Create relationship
- `engram remove node/edge` — Delete
- `engram query [<keyword>] [--type] [--related-to]` — Search graph

#### `reasoning.py` (120 LOC)
- `engram think <question>` — LLM reasoning
- `engram summarize [--count 20] [--save]` — Key insights

#### `auth_cmd.py` (100 LOC)
- `engram auth create-key <name> [--role admin|agent|reader]` — Generate API key
- `engram auth revoke <name>` — Deactivate key
- `engram auth list` — Show all keys (hashes only)

#### `config_cmd.py` (80 LOC)
- `engram config show` — Print full config
- `engram config get <key>` — Get value (dot notation)
- `engram config set <key> <value>` — Set value

#### `backup_cmd.py` (100 LOC)
- `engram backup` — Export all memory to backup.json
- `engram restore <file>` — Import backup

#### `migrate_cmd.py` (80 LOC)
- `engram migrate <export.json>` — Import from legacy memory exports

#### `providers_cmd.py` (215 LOC)
- `engram discover [--add-host <host>]` — Scan for memory services + prompt to add to config
- `engram providers list` — Show all configured providers with status
- `engram providers test <name>` — Health check + sample search for a provider
- `engram providers stats` — Query/hit/error counts per provider
- `engram providers add --name <n> --url <u> [--type <t>]` — Add provider by URL or path

#### `system.py` (100 LOC — extended v0.3.2)
- `engram status` — Memory counts + graph stats
- `engram dump` — Export JSON
- `engram serve [--host] [--port]` — Start HTTP API
- `engram resource-status` — Show current resource tier and LLM call window stats
- `engram constitution-status` — Show constitution laws and SHA-256 hash verification
- `engram scheduler-status` — Show scheduled tasks with last/next run times

---

### `src/engram/mcp/` — Model Context Protocol (MCP)

#### `server.py` (150 LOC)
MCP protocol server (stdio-based for Claude integration).
- **Tools:** engram_remember, engram_recall, engram_think, engram_add_entity, engram_add_relation, engram_query_graph, engram_ingest, engram_summarize, engram_cleanup, engram_status

#### `episodic_tools.py`, `semantic_tools.py`, `reasoning_tools.py`
Tool implementations for MCP.

#### `session_tools.py` (80 LOC) — **NEW (v0.3.0)**
MCP tools for session lifecycle management.
- **engram_session_start:** Begin new session, returns session_id
- **engram_session_end:** End active session, optionally save summary
- **engram_session_summary:** Get summary of completed session
- **engram_session_context:** Retrieve memories from active session by time window

---

### `src/engram/schema/` — Schema management

#### `builtin/devops.yaml`, `marketing.yaml`, `personal.yaml`
Pre-defined node types and relation types for different domains.

#### `loader.py` (80 LOC)
Load and validate schemas; enforce type constraints on graph operations.

---

### `.github/workflows/` — CI/CD

#### `ci.yml` (50 LOC)
- Python 3.11+, runs pytest (270 tests)
- Linting: ruff check
- Coverage: Targets 75%+
- Triggers: PR to main, push to main

#### `release.yml` (30 LOC)
- Tag-triggered: bumps version, publishes to PyPI

---

### `tests/benchmark_performance.py` — Performance Benchmark (v0.4.0)

#### `benchmark_performance.py` (250 LOC) — **NEW (v0.4.0)**
HTTP server latency benchmark measuring p50/p95/p99 per endpoint.
- **Operations:** health, remember, recall, think
- **Metrics:** p50/p95/p99/mean latency, error count, total requests
- **Flags:** `--quick` (fast subset), `--concurrency N`, `--host`, `--port`
- **Usage:** `python tests/benchmark_performance.py --host 127.0.0.1 --port 8765`
- **Dependencies:** `aiohttp` (async HTTP client)

---

### `tests/` — Comprehensive test suite (894+ tests)

#### `test_*.py`
- `test_config.py` — Config loading, env var expansion, overlay
- `test_auth.py` — JWT/API key generation, verification, RBAC
- `test_episodic_store.py` — Remember, recall, cleanup
- `test_semantic_graph.py` — Add/remove nodes/edges, queries
- `test_reasoning_engine.py` — LLM synthesis mocking
- `test_http_server.py` — All HTTP endpoints, auth, pagination, error handling
- `test_cache.py` — Redis caching (skip if Redis unavailable)
- `test_rate_limiter.py` — Rate limit enforcement
- `test_cli_commands.py` — All CLI commands
- `test_tenant_isolation.py` — Multi-tenant store factory
- `test_health_checks.py` — Liveness + readiness
- `test_backup_restore.py` — Memory snapshots
- `test_audit.py` — Audit logging

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Python version | 3.11+ |
| Total LOC | ~5000+ |
| Test count | 894+ |
| Modules | 42+ |
| Endpoints | 14 (HTTP) + 1 (WebSocket /ws) |
| CLI commands | 35+ |
| MCP tools | 19 (see README) |
| Config fields | 75+ |
| Supported roles | 3 (ADMIN, AGENT, READER) |
| Max tenants cached | 100 (graphs), 1000 (episodic) |
| Provider adapter types | 4 (rest, file, postgres, mcp) |
| Known auto-discoverable services | 5 (Cognee, Mem0, LightRAG, OpenClaw, Graphiti) |
| TUI screens | 4 (Dashboard, Search, Recent, Sessions) |
| Session storage | JSON files (~/.engram/sessions) |
| Recall pipeline components | 8 (decision, resolver, search, feedback, auto-memory, guard, auto-trigger, audit) |
| Embedding model | gemini-embedding-001 (3072d) only |
| Key rotation strategies | 2 (failover, round-robin) |
| WS commands | 7 (remember, recall, think, feedback, query, ingest, status) |
| Benchmark operations | 4 (health, remember, recall, think) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| typer | 0.9.0+ | CLI framework |
| rich | 13.0+ | Terminal output |
| chromadb | 0.4.0+ | Vector DB (episodic) |
| networkx | 3.0+ | Graph operations (in-memory) |
| pydantic | 2.0+ | Config + request validation |
| litellm | 1.0.0+ | LLM provider abstraction (Gemini) |
| fastapi | 0.100.0+ | HTTP API framework |
| uvicorn | 0.23.0+ | ASGI server |
| pyjwt | 2.8.0+ | JWT encode/decode |
| pyyaml | 6.0+ | Config file parsing |
| mcp | 1.0.0+ | Model Context Protocol |
| asyncpg | 0.29.0+ | PostgreSQL async driver (optional, Phase 2) |
| redis | 5.0.0+ | Cache + rate limiting (optional, Phase 5) |
| pytest | 7.0+ | Testing (dev) |
| opentelemetry-* | 1.20+ | Observability (optional, Phase 7, via telemetry extra) |

---

## Features (v0.3.2 — Brain Features)

**4 brain-only features (v0.3.2):**

1. Memory Audit Trail — Traceable before/after log wired into all EpisodicStore mutations
2. Resource-Aware Retrieval — 4-tier degradation with auto-recovery (ResourceMonitor)
3. Data Constitution — SHA-256 governance with prompt injection (3 laws)
4. Consolidation Scheduler — Asyncio overlap-safe background tasks with tier awareness

**545+ tests:** 39 new tests for brain features.

---

## Features (v0.3.1 — Recall Pipeline)

**4-phase delivery (v0.3.1):**

1. Core Recall Pipeline — Query decision, entity resolution, parallel search fusion
2. Learning Pipeline — Feedback loop, auto-memory detection, poisoning guard
3. Optimization — Auto-consolidation, retrieval audit log, benchmarking
4. CLI — Enhanced recall command, feedback tracking, audit log viewer, benchmark runner

**506+ tests:** 159 new tests for recall pipeline components.

---

## Features (v0.3.0)

**11-phase delivery (v0.3.0):**

1. Ebbinghaus Decay — Retention scoring via access history
2. Typed Relationships + Weight — SemanticEdge weights for path scoring
3. Activation-Based Recall — Composite scoring (similarity/retention/recency/frequency)
4. Memory Consolidation — Jaccard clustering + LLM summarization
5. OpenClaw Watcher — Watchdog/inotify for JSONL sessions
6. Privacy Tag Stripping — `<private>...</private>` → `[REDACTED]`
7. Topic Key Upsert — Same topic_key updates existing memory with revision_count
8. Progressive Disclosure — Compact recall, get_memory, timeline MCP tools
9. Session Lifecycle — Session start/end/summary/context with auto session_id
10. Git Sync — Compressed JSONL chunks to .engram/ with manifest tracking
11. TUI — Interactive terminal interface (textual library)

---

## Features (v0.2.0)

**13-phase delivery (v0.2.0):**

1. Configuration Foundation — YAML + env vars, structured logging
2. PostgreSQL Backend — Pluggable semantic graph (SQLite default)
3. Authentication — JWT + API keys, RBAC (disabled by default)
4. Multi-Tenancy — Per-tenant stores, contextvar isolation
5. Caching & Rate Limiting — Redis optional, sliding-window limits
6. API Versioning — /api/v1/ with backward-compat redirects
7. Observability — OpenTelemetry + JSONL audit logging
8. Docker & CI/CD — GitHub Actions, automated testing + release
9. Health Checks — Liveness + readiness probes, backup/restore
10. Test Expansion — 270 → 345 tests
11. Federation System — REST/File/Postgres/MCP adapters, auto-discovery, circuit breaker
12. Security Hardening — SSRF, SQL injection, timing attack, RBAC normalization, JWT validation
13. Bug Fixes (11) — federated think, UTC datetime, McpAdapter cleanup, graph O(V), node cache

