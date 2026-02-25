# Engram Project Changelog

All notable changes to this project are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

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
