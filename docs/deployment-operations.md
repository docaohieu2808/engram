# Engram Deployment & Operations

Operational guide covering deployment architectures, configuration management, component communication, state persistence, error handling strategies, security boundaries, and performance characteristics.

---

## Deployment Architecture

### Local Development
```bash
engram serve --host 127.0.0.1 --port 8765
```
- Single process, no external services required
- ChromaDB embedded, SQLite local
- Auth disabled by default
- Ideal for prototyping and testing

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8765
CMD ["engram", "serve", "--host", "0.0.0.0"]
```

**Environment variables:**
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

**Characteristics:**
- **Stateless API pods** — No session state; all data in shared databases
- **Load balancer** — Distributes traffic, optional rate limiting at edge
- **PostgreSQL** — Semantic graph + session storage; multi-tenant isolation via tenant_id
- **Redis** — Cache (recall, think, query results), rate limiting, event bus (distributed)
- **ChromaDB** — Episodic vector store; can be embedded per pod or shared volume

---

## Configuration Hierarchy

**Priority order** (highest wins):
1. CLI flags (not yet implemented for serve, but available for CLI commands)
2. Environment variables (`ENGRAM_*`)
3. YAML config (`~/.engram/config.yaml`)
4. Built-in defaults (in Pydantic models)

**Example:**
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

### Environment Variable Expansion

All YAML fields support `${VAR}` syntax for secrets:

```yaml
# config.yaml
auth:
  jwt_secret: ${JWT_SECRET}
llm:
  api_key: ${GEMINI_API_KEY}
```

Values are expanded at load time; missing vars fail at startup (no fallback).

### Episodic Backend Configuration

**Embedded ChromaDB (default):**
```yaml
episodic:
  mode: embedded
  path: ~/.engram/episodic/
```

**HTTP ChromaDB server:**
```yaml
episodic:
  mode: http
  host: localhost
  port: 8000
```

---

## Component Communication

All interfaces share the same request pipeline:

1. **CLI:** Parse args → load config → set TenantContext → StoreFactory → operation → Rich output
2. **MCP:** Receive tool call → load config → StoreFactory → execute → JSON result
3. **HTTP API:** Request → CorrelationIdMiddleware → AuthMiddleware (JWT/API key) → RateLimitMiddleware → handler → StoreFactory → execute → cache → `{data, meta}` response
4. **WebSocket:** `ws://.../ws?token=JWT` → authenticate → ConnectionManager.register → EventBus.subscribe → dispatch command → push events to tenant

### Request Pipeline Diagram
```
┌─────────────────┐
│ Incoming Request │
└────────┬────────┘
         │
    ┌────▼─────────────────────────┐
    │ CorrelationIdMiddleware       │ ← Assigns X-Correlation-ID
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ AuthMiddleware                │ ← Verify JWT/API key
    │ (JWT verify/API key lookup)   │
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ RateLimitMiddleware           │ ← Check per-tenant quota
    │ (sliding window, per tenant)  │
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ TenantContext.set()           │ ← Set tenant_id from JWT
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ Handler/Router                │ ← Execute operation
    │ (remember/recall/think/etc)   │
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ StoreFactory.get_episodic()   │ ← Retrieve per-tenant store
    │ StoreFactory.get_semantic()   │   (cached, LRU eviction)
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ Business Logic                │ ← Episodic/semantic operations
    │ (backend protocol calls)      │
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ Cache Layer (optional)        │ ← Redis cache if enabled
    │ (recall_ttl, think_ttl, etc)  │
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │ Response Envelope             │ ← {data, meta, error?}
    │ (JSON with request_id)        │
    └────┬──────────────────────────┘
         │
  ┌──────▼──────────┐
  │ HTTP 200/4xx/5xx │
  └─────────────────┘
```

---

## State & Persistence

| Component | Storage | Scope | Persistence |
|-----------|---------|-------|-------------|
| EpisodicStore | ChromaDB | Per tenant | Persistent (embedded DB or HTTP) |
| SemanticGraph | SQLite/PostgreSQL | Per tenant | Persistent (file/DB) |
| Config | YAML file | Global | Persistent (~/.engram/config.yaml) |
| API keys | JSON file | Global | Persistent (~/.engram/api_keys.json) |
| TenantContext | ContextVar | Per request/task | Transient (memory only) |
| Cache | Redis | Per tenant | Transient (with TTL) |
| Audit logs | JSONL file | Global | Persistent (~/.engram/audit.jsonl) |
| Traces | OTLP exporter | Global | External (to collector) |
| Sessions | JSON files | Per session | Persistent (~/.engram/sessions/) |
| Event Bus | Memory/Redis | Per tenant | Transient (subscriptions only) |

---

## Error Handling Strategy

1. **Input validation:** Pydantic validates all requests before business logic
2. **Structured errors:** All failures wrapped in ErrorCode + message
3. **Logging:** Errors logged with correlation_id for traceability
4. **User feedback:** HTTP 4xx/5xx responses with error code for client retry logic
5. **Graceful degradation:** Optional services (Redis, OTel) fail safely
6. **Audit trail:** Failures recorded in audit logs when enabled

### Error Codes

| Code | Status | Meaning | Retry |
|------|--------|---------|-------|
| INVALID_REQUEST | 400 | Bad input (validation failed) | No |
| UNAUTHORIZED | 401 | Missing/invalid JWT | No |
| FORBIDDEN | 403 | Valid auth but insufficient permissions | No |
| NOT_FOUND | 404 | Memory/entity not found | No |
| RATE_LIMITED | 429 | Quota exceeded | Yes (Retry-After header) |
| INTERNAL_ERROR | 500 | Server error | Yes (with backoff) |

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
| query() | <50ms | Graph traversal (lazy load) | Indexed SQL queries |
| cleanup() | Variable | DB scan | Async background task |

### Optimization Tips

1. **Caching:** Enable Redis for high-throughput scenarios (recall, think endpoints)
2. **Rate limiting:** Set `requests_per_minute` per tenant; burst allowance for spiky loads
3. **Event bus:** Use Redis event bus instead of in-process for multi-pod deployments
4. **Semantic graph:** Lazy loading indexes SQL queries; avoid full graph loads when possible
5. **Episodic backend:** Embedded ChromaDB for <100K memories; HTTP backend for shared deployments

---

## Monitoring & Observability

### Health Checks

**Liveness probe:** `GET /health`
- Returns `{status: ok}` if service can respond
- Use for restart policy (no content deps)

**Readiness probe:** `GET /health/ready`
- Returns `{status, checks: {episodic: ok, semantic: ok, redis: ok}}`
- Use for load balancer drain (waits for dependencies)

**Example Kubernetes probe:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8765
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8765
  initialDelaySeconds: 10
  periodSeconds: 5
```

### Audit Logging

Enable with `ENGRAM_AUDIT_ENABLED=true`. Records:
- Every episodic mutation (remember, delete, update)
- Every semantic operation (add node, add edge, delete)
- Auth attempts (success/failure)
- Cleanup operations
- Config changes

Location: `~/.engram/audit.jsonl` (one record per line, ISO timestamps)

### OpenTelemetry (Optional)

Enable with `ENGRAM_TELEMETRY_ENABLED=true`. Exports:
- Spans for remember, recall, think, query operations
- Full request traces (middleware → handler → operation)
- Latency metrics (p50, p95, p99)
- Exporter: OTLP protocol to collector (default: localhost:4317)

Sample rate: `ENGRAM_TELEMETRY_SAMPLE_RATE` (default 0.1 = 10%)

---

## Scaling Considerations

### Horizontal Scaling (Multi-Pod)

1. **Shared databases:** PostgreSQL + Redis (not local SQLite)
2. **Episodic backend:** HTTP ChromaDB server (not embedded)
3. **Event bus:** Redis adapter (not in-process)
4. **Session storage:** PostgreSQL (not local JSON files)
5. **Load balancer:** nginx/HAProxy for connection distribution
6. **No pod affinity:** All pods are stateless

### Vertical Scaling (Single Pod)

1. **Increase resources:** CPU (CRUD parallelism), RAM (graph LRU cache)
2. **Cache tuning:** Increase Redis memory for hot memories
3. **Batch operations:** Use bulk inserts where possible
4. **Background tasks:** Adjust scheduler concurrency (default 4)

---

## Future Architecture Evolution

See [project-roadmap.md](./project-roadmap.md) for full roadmap. Key upcoming changes:
- Distributed semantic graph (multi-node, Raft consensus)
- Streaming LLM responses for large reasoning tasks
- Graph migrations (SQLite → PostgreSQL tooling)
- Observability dashboard UI
- Multi-region replication
