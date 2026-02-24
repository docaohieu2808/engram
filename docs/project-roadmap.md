# Engram Project Roadmap

## Version History

### v0.2.0 (Completed — 2026-02-25)
**Enterprise Upgrade + Federation System**

Transformed engram from prototype to enterprise-grade system with 10 independent phases, then extended with federation and security hardening:

**Phase 11: Federation System (Provider Adapters)** ✓
- Provider adapter framework: REST, File, Postgres, MCP (`src/engram/providers/`)
- Smart query router with keyword-based classification (internal vs. domain queries)
- Auto-discovery: Cognee, Mem0, LightRAG, OpenClaw, Graphiti
- Provider registry with entry_points plugin system (third-party adapters)
- Per-provider stats tracking + circuit breaker (auto-disable on consecutive errors)
- CLI commands: `engram discover`, `engram providers list/test/stats/add`
- Config: `providers` and `discovery` sections in config.yaml
- `/think` endpoint now uses federated providers for context enrichment
- GET /providers endpoint (auth required)

**Phase 12: Security Hardening** ✓
- SSRF protection in webhooks and discovery (private/loopback IPs blocked)
- SQL injection prevention in PostgresAdapter (parameterised queries)
- JWT-based rate limiting (not header spoofing via X-Forwarded-For)
- Timing attack mitigation using `hmac.compare_digest`
- RBAC path normalization to prevent bypass
- JWT secret minimum length validation at startup

**Phase 13: Bug Fixes (11 issues)** ✓
- `/api/v1/think` wires active providers into ReasoningEngine
- UTC-aware datetime for memory expiry comparisons
- McpAdapter safe cleanup on partial init failure
- Graph traversal optimized from O(E) to O(V)
- Engine node cache invalidation after ingest

**Current stats:** 31 episodic memories in production, OpenClaw file provider active, 345 tests passing.

---

### v0.2.0 enterprise phases (completed 2026-02-24):

**Phase 1: Config + Logging Foundation** ✓
- YAML-based configuration with env var expansion
- Layered config hierarchy: YAML < env vars < CLI
- Structured logging (text + JSON formats)
- Correlation ID propagation across all operations
- Third-party logger silencing

**Phase 2: PostgreSQL Semantic Graph** ✓
- Pluggable backend interface (abstract base)
- PostgreSQL implementation with asyncpg connection pooling
- SQLite remains default for single-node deployments
- Tenant_id column for multi-tenant row-level isolation
- Automatic schema creation on first connection
- Migration utilities for SQLite → PostgreSQL

**Phase 3: Authentication (JWT + API Keys)** ✓
- JWT encode/decode (HS256)
- API key generation + SHA256 hashing
- RBAC: Three roles (ADMIN, AGENT, READER)
- Backward compat: Auth disabled by default
- Admin bootstrap mechanism for /auth/token

**Phase 4: Multi-Tenancy** ✓
- TenantContext via contextvars (thread-safe)
- StoreFactory with LRU caching (100 graphs, 1000 episodic stores)
- Per-tenant ChromaDB collections (engram_{tenant_id})
- Per-tenant semantic files (SQLite) or row filtering (PostgreSQL)
- Tenant validation: alphanumeric + hyphens/underscores, max 64 chars

**Phase 5: Redis Caching + Rate Limiting** ✓
- Optional Redis caching for recall/think/query results
- Per-endpoint TTLs: recall (300s), think (900s), query (300s)
- Sliding-window rate limiter per tenant
- Burst allowance support
- Headers: X-RateLimit-Limit, -Remaining, -Reset
- Graceful fallback when Redis unavailable

**Phase 6: API Versioning + Error Codes** ✓
- Structured endpoints at /api/v1/
- Legacy endpoint redirects for backward compat
- ErrorCode enum: INVALID_REQUEST, UNAUTHORIZED, FORBIDDEN, NOT_FOUND, RATE_LIMITED, INTERNAL_ERROR, etc.
- Consistent ErrorResponse JSON format
- Request/response metadata: request_id, correlation_id, timestamp

**Phase 7: Observability (OTel + Audit)** ✓
- OpenTelemetry instrumentation (optional, disabled by default)
- JSONL audit logging: timestamp, tenant_id, user_id, action, resource, status, metadata
- Audit backend abstraction (file-based for v0.2.0)
- Configurable sample rates and telemetry endpoint
- Service name in trace metadata

**Phase 8: Docker + CI/CD** ✓
- Multi-stage Dockerfile with production optimizations
- Docker Compose for development (single container)
- Docker Compose for production (engram + PostgreSQL + Redis + OpenTelemetry Collector)
- GitHub Actions CI: pytest (270 tests), ruff linting, coverage reporting
- GitHub Actions CD: auto-publish to PyPI on version tags
- Health check support for Kubernetes probes

**Phase 9: Health Checks + Backup/Restore** ✓
- /health endpoint for liveness checks
- /health/ready endpoint for readiness probes
- Dependency health checks: episodic store, semantic graph, Redis (if enabled), PostgreSQL (if enabled)
- backup() CLI command + /api/v1/backup endpoint
- restore() CLI command + /api/v1/restore endpoint
- Per-tenant backup support
- JSON format for easy inspection

**Phase 10: Testing Expansion** ✓
- 270 tests (up from 153)
- Coverage target: 75%+
- Unit tests: config, auth, episodic, semantic, reasoning
- Integration tests: HTTP endpoints, multi-tenant isolation
- End-to-end tests: full workflow (remember → recall → think)
- Load tests (marked as excluded from CI)
- Fixtures for mocking external services

**Bug Fixes: 21 Total** ✓
- 3 Critical
  - Reasoning accuracy: improved prompt design to infer conclusions
  - Recall consistency: searches both episodic and semantic graph
  - Vietnamese diacritics support in embeddings
- 7 High
  - Auth middleware request routing
  - Redis cache key collision
  - Semantic graph node update atomicity
  - Multi-tenant namespace isolation
  - Rate limiter reset calculation
  - Error message clarity
  - Health check timeout handling
- 11 Medium/Low
  - Minor UI improvements
  - Logging adjustments
  - Documentation updates

---

## v0.3.0 (Planned — Q2 2026)
**Advanced Querying & Performance**

### Features
- **Cypher-like DSL** for semantic graph queries (beyond simple keyword search)
  - Path finding (find shortest path between entities)
  - Pattern matching (multi-hop relationships)
  - Aggregations (count, sum, avg on attributes)
- **Query optimization:** Index creation suggestions, explain plans
- **Batch operations:** Bulk remember, bulk ingest, bulk add_nodes
- **Streaming responses:** WebSocket support for long-running operations
- **Query caching:** Smarter cache invalidation strategies

### Performance
- **Benchmarking framework:** Public performance metrics
- **GraphQL endpoint** (alternative to REST) for flexible queries
- **Connection pooling improvements:** Adaptive pool sizing

### Quality
- 300+ tests
- Performance regression detection in CI
- Load testing in production-like environment

---

## v0.4.0 (Planned — Q3 2026)
**Distributed & Multi-Node**

### Architecture
- **Graph replication:** Multi-node semantic backend (quorum-based)
- **Episodic store clustering:** Distributed ChromaDB coordination
- **Leader election:** Automatic failover for primary node
- **Consensus protocol:** Raft-based for semantic graph

### Features
- **Cross-node transactions:** Distributed remember + add_node
- **Sharding:** Tenant-aware partitioning across nodes
- **Disaster recovery:** Point-in-time restore from replicas
- **Read replicas:** Read-only nodes for scaling reads

### Operations
- **Cluster formation:** CLI tools for adding/removing nodes
- **Health monitoring:** Per-node and cluster-wide dashboards
- **Automatic rebalancing:** When nodes join/leave

---

## v0.5.0 (Planned — Q4 2026)
**Observability Dashboard + Advanced Features**

### Dashboard UI
- **Memory timeline:** Visual timeline of recent memories
- **Knowledge graph visualization:** Interactive node/edge explorer
- **Tenant analytics:** Memory counts, recall patterns, reasoning latency
- **Health dashboard:** System status, performance metrics
- **Audit viewer:** Search and filter audit logs

### Features
- **Custom embedding models:** Support for non-Gemini embeddings (OpenAI, Anthropic, local)
- **Plugin system:** Custom extractors, transformers, validators
- **Advanced RBAC:** Resource-level permissions (e.g., read only tenant-X memories)
- **Scheduled tasks:** Cleanup/summarization on cron schedule
- **Webhooks at scale:** Kafka/RabbitMQ integration for high-volume events

### LLM Enhancements
- **Multi-model support:** Route queries to different LLMs (fast vs. accurate)
- **Context windows:** Adaptive prompt sizing based on model limits
- **Fine-tuning:** Lightweight model adaptation for domain-specific reasoning

---

## v1.0.0 (Planned — Q1 2027)
**Production-Ready Release**

### Stability
- 99.99% uptime SLA
- Chaos engineering: Resilience under failures
- Long-term storage: Multi-year data retention testing
- Migration tools: Easy upgrades between major versions

### Features
- **Marketplace:** Pre-built schemas, extractors, reasoning prompts
- **Enterprise support:** SLA guarantees, priority support channel
- **Compliance:** GDPR, CCPA, SOC 2 certifications
- **Data residency:** Regional deployment options

### Documentation
- Comprehensive API reference (OpenAPI/Swagger)
- Video tutorials for common workflows
- Migration guides from competitive systems
- Performance tuning guides for large deployments

---

## Completed Phases (v0.2.0)

| Phase | Status | Tests |
|-------|--------|-------|
| 1. Config + Logging | ✓ Complete | 20+ |
| 2. PostgreSQL Graph | ✓ Complete | 25+ |
| 3. Authentication | ✓ Complete | 30+ |
| 4. Multi-Tenancy | ✓ Complete | 20+ |
| 5. Caching + Rate Limit | ✓ Complete | 25+ |
| 6. API Versioning | ✓ Complete | 15+ |
| 7. Observability | ✓ Complete | 20+ |
| 8. Docker + CI/CD | ✓ Complete | 10+ |
| 9. Health + Backup | ✓ Complete | 20+ |
| 10. Testing Expansion | ✓ Complete | 270+ |
| 11. Federation System | ✓ Complete | included in 345 |
| 12. Security Hardening | ✓ Complete | included in 345 |
| 13. Bug Fixes (11) | ✓ Complete | — |

**Total Tests:** 345 | **Bug Fixes:** 32 (21 original + 11 v0.2 phase 3)

---

## Known Limitations & Future Improvements

### Current Limitations (v0.2.0)
- Single-node deployment only (multi-node in v0.4)
- No built-in query DSL (Cypher-like support in v0.3)
- ChromaDB doesn't support distributed mode
- Streaming responses not supported
- No native dashboard UI (v0.5)
- Provider stats are in-memory only (reset on restart)

### Roadmap Priorities
1. **Query DSL** (Q2 2026) — Power users need complex graph queries
2. **Multi-node** (Q3 2026) — Enterprise reliability requirements
3. **Dashboard** (Q4 2026) — Operational visibility
4. **Marketplace** (Q1 2027) — Community contributions + ecosystem

### Community Feedback Wanted
- What workflows are most common for your use case?
- Which LLM providers should we support?
- Preferred query syntax (SQL-like, GraphQL, Cypher)?
- What observability metrics matter most?

---

## Success Metrics

### v0.2.0 (Completed)
- [x] 13 phases complete (10 enterprise + federation + security + bug fixes)
- [x] 345 tests passing
- [x] PostgreSQL backend option working
- [x] Multi-tenant support production-ready
- [x] Auth optional but enforced correctly
- [x] Docker image building and running
- [x] All 32 bug fixes resolved
- [x] CI/CD passing on every commit
- [x] Federation system: REST/File/Postgres/MCP adapters
- [x] Auto-discovery for Cognee, Mem0, LightRAG, OpenClaw, Graphiti
- [x] Security hardening: SSRF, SQL injection, timing attacks, RBAC path normalization
- [x] /api/v1/think uses federated providers

### v0.3.0 (Target)
- [ ] Advanced graph queries (path finding, patterns)
- [ ] Query performance <100ms (p99)
- [ ] 300+ tests, 80%+ coverage
- [ ] GraphQL endpoint operational
- [ ] Streaming WebSocket support

### v0.4.0 (Target)
- [ ] Multi-node cluster operational (3+ nodes)
- [ ] Automatic failover tested
- [ ] 350+ tests covering distribution
- [ ] Data consistency verified under failures

### v1.0.0 (Target)
- [ ] 400+ tests, 85%+ coverage
- [ ] 99.99% uptime in production
- [ ] Marketplace with 10+ community contributions
- [ ] Enterprise customer deployments

---

## Deprecation Policy

### Backward Compatibility Commitment
- Breaking changes only at major versions (v1.0, v2.0, etc.)
- Minimum 6 months deprecation notice before removal
- Legacy endpoints maintained with redirects
- Migration guides provided with each deprecation

### Example: API Versioning
```
v0.2.0: /remember → /api/v1/remember (redirect)
v1.0.0: /api/v1/remember (canonical)
v2.0.0: /api/v2/remember (new interface, v1 deprecated)
v3.0.0: /api/v2/remember removed (after 6mo notice in v2.0)
```

---

## Release Schedule

| Version | Target Date | Focus | Status |
|---------|-------------|-------|--------|
| v0.2.0 | 2026-02-25 | Enterprise + Federation + Security | ✓ Released |
| v0.3.0 | 2026-06-30 | Advanced Queries | Planned |
| v0.4.0 | 2026-09-30 | Multi-Node Distribution | Planned |
| v0.5.0 | 2026-12-31 | Dashboard + Features | Planned |
| v1.0.0 | 2027-03-31 | Production Release | Planned |

---

## Contributing

### How to Help
- **Bug reports:** Use GitHub Issues with reproduction steps
- **Feature requests:** Open GitHub Discussions first
- **Code contributions:** Fork → branch → PR (see CONTRIBUTING.md)
- **Documentation:** Corrections and improvements welcome
- **Testing:** Additional test cases for edge cases
- **Translations:** Localize docs to other languages

### Development Setup
```bash
git clone https://github.com/engram/engram.git
cd engram
pip install -e ".[dev]"
pytest tests/ -v
```

### Release Process
1. Update version in pyproject.toml
2. Update CHANGELOG.md with user-facing changes
3. Create release PR, get 2 approvals
4. Tag commit with v{version}
5. GitHub Actions auto-publishes to PyPI

---

## Support & Communication

### Channels
- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Q&A, architecture discussions, ideas
- **Twitter/X:** Release announcements, blog posts
- **Email:** support@engram.io (enterprise)

### Response Times
- **Critical bugs:** <4 hours
- **High priority:** <24 hours
- **Discussions:** Best effort, typically <48 hours

---

## Funding & Sustainability

### Current Status
- Open source (MIT license)
- Community maintained
- Accepting sponsorships

### How You Can Help
- Star the repo on GitHub
- Share with colleagues
- Contribute code or docs
- Report bugs with details

### Future Monetization (Optional)
- Engram Cloud: Managed hosting
- Enterprise support: Priority SLA, training, consulting
- Marketplace: Revenue share with plugin creators
- (All open source code remains free)

