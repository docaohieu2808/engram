# Engram Project Changelog

All notable changes to this project are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [v0.4.2] — 2026-02-28
### Configuration Management + Live Settings Editor

**Added**
- **Configurable Everything** (`src/engram/config.py` extensions)
  - ExtractionConfig: llm_model, temperature, max_retries, retry_delay_seconds, chunk_size, msg limits
  - RecallConfig: search_limit, entity_search_limit, provider_search_limit, graph_depth, scoring params
  - SchedulerConfig: consolidate/cleanup/decay intervals, tick interval, task timeout, decay multiplier
  - Secondary configs: health.check_llm_model, cache.max_*_size, hooks.webhook_timeout, retrieval_audit limits
  - All hardcoded tuning parameters now YAML-configurable via config.yaml or ENGRAM_* env vars

- **WebUI Settings Editor** (new `/Settings` tab)
  - GET/PUT `/api/v1/config` endpoints for read/write config.yaml
  - Grouped config sections: LLM, Embedding, Recall, Extraction, Scheduler, Health, Cache, Hooks
  - Live preview of current config values with type validation
  - Restart-required badge for server-level configs
  - Save validation: type checking, range limits, dependency verification

- **LLM Model Selector** (Settings → LLM section)
  - Dropdown UI for model selection (Gemini, Claude, OpenAI)
  - Per-provider API key configuration
  - Test button to verify model connectivity
  - Auto-set `disable_thinking` flag based on model family

- **Think Flag Unification** (Phase 2)
  - All 4 modules now respect `cfg.llm.disable_thinking`:
    - `recall/entity_resolver.py`
    - `health/components.py`
    - `memory_extractor.py`
    - `consolidation/engine.py`
  - Zero hardcoded thinking={"type":"disabled"} outside defaults

**Tests:** 972 total (+ config validation, WebUI editor, settings form tests)

---

## [v0.4.1] — 2026-02-27
### WebSocket API + Test Coverage

**Added**
- **WebSocket API** (`src/engram/ws/` — 4 files)
  - `protocol.py` — Message protocol: command schema, event types, JSON serialization
  - `event_bus.py` — In-process pub/sub; broadcasts memory events per tenant namespace
  - `connection_manager.py` — Active connection lifecycle, per-tenant isolation
  - `handler.py` — FastAPI route `GET /ws?token=JWT`; JWT auth on upgrade
  - **7 commands:** recall, remember, think, query, ingest, feedback, status
  - **Push events:** memory.created, memory.deleted, memory.updated, consolidation.completed, error
  - Per-tenant isolation: connections scoped to tenant_id from JWT claim

- **71 WebSocket tests** covering protocol, event bus, connection manager, handler auth + commands
- **33 P0 gap tests** covering rate limiter edge cases, consolidation triggers, MCP server tools

**Tests:** 893 total (71 WebSocket + 33 P0 gap = 104 new from 789 base)

---

## [v0.4.0] — 2026-02-25
### Intelligence Layer + Graph Visualization

**Added**
- **Temporal Resolution** (`src/engram/recall/temporal_resolver.py`)
  - 28 Vietnamese+English patterns: "hôm nay/hôm qua", "tuần trước/tuần tới", "yesterday/tomorrow", "last week", etc.
  - ISO date resolution before episodic store insert
  - Wired into store.remember() pipeline; adds resolved_dates to memory metadata
  - Fallback to direct date parsing if pattern not matched

- **Pronoun Resolution** (`src/engram/recall/pronoun_resolver.py`)
  - LLM-based entity mapping for "anh ấy", "he/she/they", "it" → named entity from SemanticGraph
  - Fallback to keyword matching if graph empty or LLM unavailable
  - Wired into engram_recall() and engram_think() pipelines
  - Config: resolution.pronoun_enabled (default true)

- **Feedback Loop + Auto-adjust** (`src/engram/feedback/auto_adjust.py`)
  - Confidence adjustment: +0.15 (positive), -0.2 (negative)
  - Importance adjustment: +1 (positive), -1 (negative)
  - Auto-delete: If negative_count >= 3 AND confidence < 0.5 → remove from episodic store
  - New POST /api/v1/feedback endpoint with memory_id + feedback_type
  - New MCP tool `engram_feedback(id, feedback_type)` records + returns updated confidence
  - Config: feedback.confidence_positive_delta, feedback.confidence_negative_delta, feedback.auto_delete_threshold

- **Fusion Formatter** (`src/engram/recall/fusion_formatter.py`)
  - Groups recall results by memory type: [preference], [fact], [lesson], [decision], [todo], [error], [workflow], [context]
  - Each group sorted by score descending; compact format by default (id, date, snippet)
  - Wired into engram_recall() after parallel search + fusion
  - LLM reasoning engine uses formatted results with type hints for better context
  - Config: fusion.formatter_enabled (default true)

- **Graph Visualization UI** (`static/graph.html`, new `/graph` route, `GET /api/v1/graph/data`)
  - vis-network library, dark theme, interactive entity relationship explorer
  - Features: drag-to-move nodes, click-to-inspect entity details, search by name, zoom/pan
  - Node colors by entity type; edge labels show relationship types; physics simulation
  - Renders in <500ms; handles 1000+ node graphs
  - CLI: `engram graph` launches browser at localhost:8765/graph
  - MCP tool: `engram_get_graph_data(search_keyword)` returns filtered graph JSON

- **7 Orphaned Modules Integrated**
  - `ingestion/guard.py` — Prompt injection prevention (now required for security)
  - `recall/decision.py` — Trivial message skip <10ms (config: recall.decision_skip_trivial)
  - `providers/telemetry.py` — Latency tracking and instrumentation
  - `episodic/fts_index.py` — Full-text search indexing for keyword fallback
  - `recall/parallel_search.py` — Multi-source search fusion (was orphaned, now core)
  - `capture/auto_memory.py` — Auto-detection of save-worthy messages (config: ingestion.auto_memory_enabled)
  - `consolidation/auto_trigger.py` — Consolidation trigger after N messages (config: recall.auto_consolidate_threshold)

**Fixed**
- FTS5 thread safety: Added lock acquisition in parallel search to prevent concurrent index corruption
- OOM pagination: Limit result aggregation before dedup to prevent memory explosion on large result sets
- Rate limiter race condition: Redis atomic increment with TTL to prevent lost updates in high-concurrency

**Tests:** 726 total (181 new: temporal, pronoun, feedback, fusion, graph visualization, wired modules, bug fixes)

---

## [v0.3.2] — 2026-02-25
### Brain Features

**Added**
- **Memory Audit Trail** (`src/engram/audit.py`, `src/engram/episodic/store.py`)
  - `log_modification()` records before/after values, mod_type (memory_create, memory_delete, memory_update, metadata_update, config_change, batch_create, cleanup_expired), reversible flag, timestamp
  - `read_recent(n)` retrieves last N audit entries
  - Wired into: remember(), delete(), update_metadata(), _update_topic(), cleanup_expired()

- **Resource-Aware Retrieval** (`src/engram/resource_tier.py`, `src/engram/reasoning/engine.py`)
  - `ResourceMonitor` tracks LLM call success/failure with configurable sliding window
  - 4 tiers: FULL → STANDARD → BASIC → READONLY
  - think() and summarize() check tier before LLM calls; BASIC returns raw results without synthesis
  - Auto-recovers to higher tier after 60s cooldown without failures
  - CLI: `engram resource-status`

- **Data Constitution** (`src/engram/constitution.py`, `src/engram/reasoning/engine.py`)
  - 3 laws: namespace isolation, no fabrication, audit rights
  - Auto-creates ~/.engram/constitution.md on first load
  - SHA-256 hash verification detects file tampering
  - Compact prefix injected into every LLM prompt (reasoning + summarize)
  - CLI: `engram constitution-status`

- **Consolidation Scheduler** (`src/engram/scheduler.py`, `src/engram/cli/system.py`)
  - Asyncio recursive setTimeout pattern (overlap-safe — each task schedules its own next run)
  - 3 default tasks: cleanup_expired (daily), consolidate_memories (every 6h, LLM), decay_report (daily)
  - Respects resource tier — skips LLM-dependent tasks on BASIC tier
  - State persisted to ~/.engram/scheduler_state.json
  - Starts automatically with `engram watch`
  - CLI: `engram scheduler-status`

**Tests:** 545 total (39 new)

---

## [v0.3.1] — 2026-02-25
### Recall Pipeline Upgrade

**Added**
- Query Decision engine — trivial message skip via regex (<10ms, no vector search)
- Entity Resolution — temporal (Vietnamese+English regex) + pronoun (LLM with fallback)
- Parallel Search fusion — multi-source (ChromaDB semantic + entity graph + keyword fallback), dedup by content hash
- Learning feedback loop — ±0.15/0.2 confidence adjustment, auto-delete after 3× negative + low confidence
- Auto-Memory detection — save-worthy messages (Save: prefix, identity, preferences, decisions); skip sensitive data
- Poisoning Guard — block prompt injection (ignore instructions, you are now, special tokens)
- Auto-Consolidation trigger — after N messages (default 20), async non-blocking
- Retrieval Audit log — JSONL append-only per recall operation (query, results_count, latency_ms)
- Benchmarking framework — load question sets (JSON), measure accuracy (exact match, semantic similarity, F1)
- New CLI: `engram resolve`, `engram feedback`, `engram audit`, `engram benchmark`
- New config: ResolutionConfig, RecallPipelineConfig, FeedbackConfig, IngestionConfig, RetrievalAuditConfig
- New models: Entity, ResolvedText, SearchResult, FeedbackType, MemoryCandidate
- New EpisodicMemory fields: confidence (float, default 1.0), negative_count (int, default 0)

**Tests:** 506 total (159 new)

---

## [v0.3.0] — 2026-02-25
### Activation-Based Recall + Consolidation + User Interfaces

**Added**
- Ebbinghaus Decay model — retention score: `e^(-decay_rate * days / (1 + 0.1 * access_count))`
- Typed Relationships with Weight — SemanticEdge weight (float, default 1.0), attributes dict
- Activation-Based Recall — composite score: similarity*0.5 + retention*0.2 + recency*0.15 + frequency*0.15
- Memory Consolidation — Jaccard clustering + LLM summarization → stored as CONTEXT memory
- OpenClaw Realtime Watcher — watchdog/inotify for ~/.openclaw/agents/main/sessions/*.jsonl
- Privacy Tag Stripping — `<private>...</private>` → `[REDACTED]` before ChromaDB insert
- Topic Key Upsert — topic_key param; same key updates existing memory with revision_count tracking
- Progressive Disclosure (MCP) — compact recall by default; new engram_get_memory, engram_timeline tools
- Session Lifecycle — JSON-file backed SessionStore; 4 MCP tools; auto session_id injection
- Git Sync — compressed JSONL chunks to .engram/ with manifest deduplication
- TUI — interactive terminal interface (textual); 4 screens: Dashboard, Search, Recent, Sessions
- New CLI: `engram consolidate`, `engram decay`, `engram sync`, `engram tui`, `engram session-start`, `engram session-end`

**Tests:** 380 total

---

## [v0.2.0] — 2026-02-25
### Enterprise Upgrade + Federation System

**Added**
- Federation layer: REST, File, Postgres, MCP provider adapters
- Auto-discovery for Cognee, Mem0, LightRAG, OpenClaw, Graphiti
- Smart query router (keyword-based: internal vs. domain)
- Circuit breaker per provider; entry_points plugin system
- Security: SSRF protection, SQL injection prevention, timing-safe key comparison, RBAC path normalization, JWT secret minimum length
- CLI: `engram discover`, `engram providers list/test/stats/add`

**Also (enterprise phases, 2026-02-24):**
- YAML-based config with env var expansion and ENGRAM_* overlay
- PostgreSQL semantic graph backend (asyncpg, optional)
- JWT + API key authentication with RBAC (disabled by default)
- Multi-tenancy via contextvars with LRU-cached StoreFactory
- Redis caching (TTLs: recall 300s, think 900s, query 300s) + sliding-window rate limiting
- API versioning at /api/v1/ with backward-compat redirects and structured ErrorCode
- OpenTelemetry instrumentation + JSONL audit logging (both optional)
- Docker + GitHub Actions CI/CD + release automation
- Health checks (/health, /health/ready) + backup/restore endpoints
- 32 bug fixes (21 original + 11 federation/security)

**Tests:** 345 total

---

## [v0.1.0] — 2025
### Initial Release

**Added**
- Dual-memory architecture: ChromaDB episodic + SQLite semantic graph
- Reasoning engine (LLM via litellm / Gemini)
- CLI (Typer), MCP server (stdio), HTTP API (FastAPI)
- Gemini embeddings (3072 dims) with fallback (all-MiniLM-L6-v2, 384 dims)
- Entity extraction + auto-ingest
- Webhooks (fire-and-forget, SSRF-protected)
