# Changelog

All notable changes to engram. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [v0.5.25] — Current

Latest release. See [GitHub releases](https://github.com/docaohieu2808/Engram-Mem/releases) for the full release history.

---

## [v0.4.3] — 2026-03-01

### Interactive Setup Wizard

**Added**

- `engram setup` command — interactive wizard to auto-detect and configure AI agents/IDEs
  - Auto-detects: Claude Code, OpenClaw, Cursor, Windsurf, Cline, Aider, Zed, Void, Antigravity
  - `--dry-run` flag previews changes without writing
  - `--non-interactive` flag for CI/headless environments
  - `--status` flag shows current connection status

- Connector architecture (`src/engram/setup/connectors/`)
  - `AgentConnector` ABC with `detect()`, `configure()`, `verify()` interface
  - `McpJsonConnector` mixin for MCP JSON config merge
  - Connector registry with auto-discovery, sorted by tier (1-3)

- MCP path resolution fix — resolves `engram-mcp` to absolute path when installed in a venv

- Federation provider stubs for Mem0, Cognee, Zep detection

---

## [v0.4.2] — 2026-02-28

### Configuration Management + Live Settings Editor

**Added**

- Configurable everything — all hardcoded tuning parameters now YAML-configurable:
  - `ExtractionConfig`: llm_model, temperature, max_retries, chunk_size
  - `RecallConfig`: search_limit, entity_search_limit, graph_depth, scoring params
  - `SchedulerConfig`: consolidate/cleanup/decay intervals, task timeout

- WebUI Settings Editor (new `/Settings` tab in web UI)
  - `GET/PUT /api/v1/config` endpoints for read/write config.yaml
  - Grouped config sections: LLM, Embedding, Recall, Extraction, Scheduler, Health, Cache
  - Live preview with type validation and restart-required badges

---

## [v0.4.1] — 2026-02-25

### Memory Classification + Actionable Memories

**Added**

- `capture/memory_classifier.py` — heuristic classifier (regex patterns, no LLM cost)
  - Classifies: todo, decision, preference, error, workflow, lesson, fact
  - Supports English and Vietnamese patterns
- Integrated into `do_ingest()` and `do_ingest_messages()` pipelines

---

## [v0.4.0] — 2026-02-20

### Entity-First Ingestion

**Changed**

- Switched to `/api/v1/ingest` endpoint for entity-first ingestion
- `EntityExtractor` now gates storage — only messages with extracted entities are stored
- Dual storage: episodic (Qdrant) + semantic graph updated atomically on each ingest

---

## [v0.3.2] — 2026-02-15

### Brain Features + Intelligence Layer

**Added**

- Temporal resolution — 28 Vietnamese+English date patterns
- Pronoun resolution — graph-context + LLM fallback
- Fusion formatter — `[type]`-grouped recall output
- Memory consolidation — Jaccard clustering + LLM summarization
- Meeting ledger — structured meeting records
- Feedback loop — confidence scoring (+0.15/-0.2), auto-delete on 3x negative
- Graph visualization — interactive vis-network UI with dark theme
- Data constitution — 3-law LLM governance, SHA-256 tamper detection

---

## [v0.3.1] — 2026-02-10

### Recall Pipeline

**Added**

- Full recall pipeline: query decision → temporal resolution → pronoun resolution → parallel search → dedup → fusion
- Auto-memory detection — save-worthy message detection without explicit `engram remember`
- Poisoning guard — blocks prompt injection from being stored
- Retrieval audit — JSONL log for all recall operations
- Auto-consolidate trigger after N messages

---

## [v0.2.0] — 2026-02-01

### Federation Layer

**Added**

- Provider adapters: REST, File, PostgreSQL, MCP
- Auto-discovery: local port scan + remote hosts + MCP config files
- Smart query router: `internal` vs `domain` classification
- Circuit breaker: auto-disable on consecutive errors
- Plugin system: third-party adapters via `entry_points`

---

## [v0.1.0] — 2026-01-15

### Initial Release

**Added**

- Dual-memory system: EpisodicStore (Qdrant) + SemanticGraph (NetworkX + SQLite/PG)
- Reasoning Engine (Gemini via litellm)
- CLI (Typer), MCP server (FastMCP), HTTP API (FastAPI), WebSocket API
- Multi-tenancy, JWT auth, Redis caching, rate limiting
- Audit logging, OpenTelemetry, Docker deployment
- Backup/restore, health checks, benchmark suite
