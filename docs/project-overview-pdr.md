# Engram: Project Overview & Product Development Requirements

## What is Engram?

Engram is a dual-memory AI agent system that thinks like humans. It combines:

- **Episodic Memory** — Vector database (ChromaDB) for semantic similarity search over timestamped memories
- **Semantic Memory** — Knowledge graph (PostgreSQL/SQLite + NetworkX) for typed entities and relationships
- **Reasoning Engine** — LLM synthesis (Gemini) that connects episodic + semantic memory to answer questions
- **Federation Layer** — External provider adapters (REST, File, Postgres, MCP) extending recall across services

Exposes three interfaces: **CLI** (Typer), **MCP** (Claude integration), **HTTP API** (FastAPI).

**Version:** 0.2.0 | **Status:** Enterprise-ready + Federation | **Tests:** 345 | **License:** MIT

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
- **Test coverage:** 75%+ code coverage; 345 tests

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

### For Future Releases
- Distributed semantic graph (cluster mode)
- Streaming LLM responses for large reasoning tasks
- Graph migrations (SQLite → PostgreSQL)
- Custom embedding models
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
git clone https://github.com/engram/engram.git
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

- **GitHub:** https://github.com/engram/engram
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

