# Engram System Architecture

## High-Level Overview

Engram is a dual-memory AI system that enables agents to reason like humans by combining:

1. **Episodic Memory** — Vector embeddings for semantic search (ChromaDB)
2. **Semantic Memory** — Knowledge graphs for entity relationships (PostgreSQL/SQLite + NetworkX)
3. **Reasoning Engine** — LLM synthesis connecting both stores (Gemini via litellm)
4. **Federation Layer** — External memory providers (REST, File, Postgres, MCP) via smart query router

All three interfaces (CLI, MCP, HTTP) share the same memory layers.

---

## System Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         Interfaces                             │
├──────────────────────┬─────────────────┬──────────────────────┤
│   CLI (Typer)        │  MCP (FastMCP)  │  HTTP API (FastAPI)  │
│   Local tools        │  stdio-based    │  /api/v1/ endpoints  │
└──────────────────────┴────────┬────────┴──────────┬────────────┘
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
- **Graph** (`graph.py`) — NetworkX wrapper (in-memory query engine)
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

**Config** (`~/.engram/config.yaml`):
```yaml
providers:
  - name: openclaw
    type: file
    path: ~/.openclaw/workspace/memory/
    pattern: "*.md"
    enabled: true

  - name: mem0
    type: rest
    url: http://localhost:8080
    search_endpoint: /v1/memories/search
    search_method: POST
    search_body: '{"query": "{query}", "limit": {limit}}'
    result_path: results[].memory
    enabled: true

discovery:
  local: true          # scan localhost ports on startup
  hosts: []            # additional remote hosts
  endpoints: []        # direct endpoint URLs to probe
```

---

### Layer 3b: Memory Consolidation (v0.3.0)

**Path:** `src/engram/consolidation/`

**Purpose:** Automatically cluster and summarize related episodic memories via LLM synthesis, reducing redundancy and improving recall efficiency.

**Components:**
- **ConsolidationEngine** (`engine.py`) — Jaccard-based clustering + LLM summarization
  - `consolidate(limit, similarity_threshold)` → clusters + summaries stored as CONTEXT memory
  - Tracks consolidation_group, consolidated_into fields on EpisodicMemory
  - LLM summarizes semantically similar memory clusters
- **Config:** ConsolidationConfig (enabled, min_cluster_size, similarity_threshold)

**Storage:** Consolidated memories stored as CONTEXT type in episodic store with metadata linking to cluster members.

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

**Request/Response Structure:**

All responses follow this format:
```json
{
  "data": {...},
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-24T10:00:00Z",
    "correlation_id": "from-header"
  }
}
```

Errors:
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Content must be at least 1 character",
    "details": {...}
  },
  "meta": {
    "request_id": "uuid",
    "correlation_id": "..."
  }
}
```

**Error Codes** (`src/engram/errors.py`):
- INVALID_REQUEST — Validation failed
- UNAUTHORIZED — Auth missing/invalid
- FORBIDDEN — Auth valid but insufficient role
- NOT_FOUND — Resource not found
- RATE_LIMITED — Too many requests
- INTERNAL_ERROR — Server error

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

### CLI → Memory
```
CLI (Typer)
  ↓
[Parse args]
  ↓
[Load config]
  ↓
[Set TenantContext from namespace]
  ↓
[Create or retrieve stores via StoreFactory]
  ↓
[Call remember/recall/add_node/think]
  ↓
[Print results via Rich]
```

### MCP → Memory
```
MCP Server (FastMCP/stdio)
  ↓
[Receive tool call]
  ↓
[Load config]
  ↓
[Create or retrieve stores]
  ↓
[Execute tool implementation]
  ↓
[Return JSON result]
```

### HTTP API → Memory
```
Request
  ↓
[CorrelationIdMiddleware - set X-Correlation-ID]
  ↓
[AuthMiddleware - verify JWT/API key, set TenantContext]
  ↓
[RateLimitMiddleware - check limits]
  ↓
[Route handler]
  ↓
[StoreFactory.get_episodic(tenant_id)]
  ↓
[StoreFactory.get_graph(tenant_id)]
  ↓
[Execute operation]
  ↓
[Cache result if enabled]
  ↓
[Return response + metadata]
```

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

- **v0.3:** Advanced query DSL (Cypher-like), GraphQL endpoint, streaming responses
- **v0.4:** Distributed semantic graph, multi-node clustering, Raft consensus
- **v0.5:** Observability dashboard UI, custom embedding models, advanced RBAC
- **v1.0:** Production release, marketplace, enterprise SLA, compliance certifications

