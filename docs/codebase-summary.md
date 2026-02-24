# Engram Codebase Summary

## Overview
Engram v0.2.0 is a dual-memory AI agent system combining episodic (vector) and semantic (graph) memory with LLM reasoning, extended by a federation layer for connecting external memory sources. ~2500 LoC, 345 tests, Python 3.11+.

**Top files:** `capture/server.py` (API), `episodic/store.py` (vector store), `config.py` (configuration), `providers/` (federation layer).

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

#### `audit.py` (100 LOC)
Audit logging for compliance (optional, disabled by default).
- **Backend:** File-based JSONL (one record per line)
- **Records:** {timestamp, tenant_id, user_id, action, resource, status, metadata}
- **Path:** ~/.engram/audit.jsonl (configurable)

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

#### `store.py` (380 LOC)
ChromaDB vector database wrapper.
- **Collections:** One per namespace (tenant_id)
- **Chunking:** Auto-splits long content (>1000 chars) into 800-char chunks with overlap
- **Metadata:** memory_type, priority, tags, expires, created_at, updated_at
- **Methods:** remember(), recall(), cleanup(), get(), count()
- **Hooks:** Fires on_remember webhook after insert (fire-and-forget)
- **Fallback embedding:** Gemini (default, 3072 dims) or all-MiniLM-L6-v2 (384 dims)

#### `search.py` (120 LOC)
Search utilities: embedding generation, similarity scoring, filtering.
- **search_similar():** Vector search with optional tag/type filters
- **chunk_text():** Split text with configurable chunk size + overlap

---

### `src/engram/semantic/` — Knowledge graph (SQLite/PostgreSQL + NetworkX)

#### `backend.py` (80 LOC)
Abstract backend interface — supports SQLite and PostgreSQL.
- **Methods:** create_graph(), load_graph(), save_graph(), close()

#### `sqlite_backend.py` (200 LOC)
SQLite semantic graph backend.
- **Schema:** Nodes (id, key, type, attributes_json), Edges (source_id, target_id, relation, metadata_json)
- **File:** ~/.engram/semantic.db or tenant-specific .{tenant_id}.db
- **Operations:** Add/remove nodes & edges, find related, query

#### `pg_backend.py` (200 LOC)
PostgreSQL semantic graph backend (Phase 2, enterprise upgrade).
- **Schema:** Same tables, adds tenant_id column for multi-tenant isolation
- **Connection pool:** asyncpg, configurable pool_min (5) and pool_max (20)
- **Async:** Full async/await support via asyncpg

#### `graph.py` (250 LOC)
NetworkX wrapper for in-memory graph operations.
- **Nodes:** Entities with type and attributes
- **Edges:** Relationships with type metadata
- **Queries:** Related entities, path finding, subgraph extraction
- **Methods:** add_node(), add_edge(), query(), relate(), remove_*()

#### `query.py` (120 LOC)
Graph query DSL and utilities.
- **query():** Find nodes by keyword/type/related_to with pagination
- **Filtering:** By node type, relation type, connection distance
- **Results:** Nodes, edges, connectivity stats

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

#### `engine.py` (200 LOC)
Combines episodic + semantic memory, feeds to LLM for synthesis.
- **Providers:** Gemini (default) via litellm
- **Operations:** think() (Q&A), summarize() (key insights), ingest() (extract entities + remember)
- **Prompts:** Calibrated for dual-memory reasoning
- **Caching:** Optional Redis caching for expensive LLM calls

---

### `src/engram/capture/` — External agent integration

#### `server.py` (590 LOC) — HTTP API
FastAPI endpoints for all operations. Multi-tenant, versioned, fully authenticated.
- **Routes:**
  - `POST /api/v1/remember` — Store episodic memory
  - `GET /api/v1/recall` — Search episodic (with pagination)
  - `POST /api/v1/think` — LLM reasoning
  - `POST /api/v1/ingest` — Extract entities + store memories
  - `GET /api/v1/query` — Graph query
  - `POST /api/v1/cleanup` — Delete expired
  - `POST /api/v1/summarize` — LLM synthesis
  - `POST /api/v1/auth/token` — Issue JWT
  - `GET /api/v1/health` — Liveness
  - `POST /api/v1/backup` — Snapshot memory
  - `POST /api/v1/restore` — Load backup
- **Middleware:** CorrelationIdMiddleware (X-Correlation-ID), RateLimitMiddleware (per-tenant sliding-window)
- **Versioning:** /api/v1/ prefix; legacy /remember routes redirect
- **Response format:** Structured ErrorResponse on failures
- **Pagination:** offset + limit on /recall and /query

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
- `engram migrate <export.json>` — Import from old agent-memory exports

#### `providers_cmd.py` (215 LOC)
- `engram discover [--add-host <host>]` — Scan for memory services + prompt to add to config
- `engram providers list` — Show all configured providers with status
- `engram providers test <name>` — Health check + sample search for a provider
- `engram providers stats` — Query/hit/error counts per provider
- `engram providers add --name <n> --url <u> [--type <t>]` — Add provider by URL or path

#### `system.py` (100 LOC)
- `engram status` — Memory counts + graph stats
- `engram dump` — Export JSON
- `engram serve [--host] [--port]` — Start HTTP API

---

### `src/engram/mcp/` — Model Context Protocol (MCP)

#### `server.py` (150 LOC)
MCP protocol server (stdio-based for Claude integration).
- **Tools:** engram_remember, engram_recall, engram_think, engram_add_entity, engram_add_relation, engram_query_graph, engram_ingest, engram_summarize, engram_cleanup, engram_status

#### `episodic_tools.py`, `semantic_tools.py`, `reasoning_tools.py`
Tool implementations for MCP.

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

### `tests/` — Comprehensive test suite (270+ tests)

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
| Total LOC | ~2500 |
| Test count | 345 |
| Modules | 20+ |
| Endpoints | 13 (HTTP) |
| CLI commands | 25+ |
| MCP tools | 8 |
| Config fields | 50+ |
| Supported roles | 3 (ADMIN, AGENT, READER) |
| Max tenants cached | 100 (graphs), 1000 (episodic) |
| Provider adapter types | 4 (rest, file, postgres, mcp) |
| Known auto-discoverable services | 5 (Cognee, Mem0, LightRAG, OpenClaw, Graphiti) |

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

## Features (v0.2.0)

**13-phase delivery:**

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
11. **Federation System** — REST/File/Postgres/MCP adapters, auto-discovery, circuit breaker
12. **Security Hardening** — SSRF, SQL injection, timing attack, RBAC normalization, JWT validation
13. **Bug Fixes (11)** — federated think, UTC datetime, McpAdapter cleanup, graph O(V), node cache

**Bug fixes:** 32 total. Active: 31 episodic memories, OpenClaw file provider connected.

