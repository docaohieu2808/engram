# Engram Project Roadmap

## Version History

### v0.2.0 (Completed — 2026-02-24)
**Enterprise Upgrade Complete**

Transformed engram from prototype to enterprise-grade system with 10 independent phases:

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

| Phase | Status | Completion | Tests | Effort |
|-------|--------|------------|-------|--------|
| 1. Config + Logging | ✓ Complete | 100% | 20+ | 4h |
| 2. PostgreSQL Graph | ✓ Complete | 100% | 25+ | 6h |
| 3. Authentication | ✓ Complete | 100% | 30+ | 5h |
| 4. Multi-Tenancy | ✓ Complete | 100% | 20+ | 5h |
| 5. Caching + Rate Limit | ✓ Complete | 100% | 25+ | 4h |
| 6. API Versioning | ✓ Complete | 100% | 15+ | 3h |
| 7. Observability | ✓ Complete | 100% | 20+ | 4h |
| 8. Docker + CI/CD | ✓ Complete | 100% | 10+ | 3h |
| 9. Health + Backup | ✓ Complete | 100% | 20+ | 3h |
| 10. Testing Expansion | ✓ Complete | 100% | 270+ | 3h |

**Total Effort:** 40 hours | **Total Tests:** 270 | **Bug Fixes:** 21

---

## Known Limitations & Future Improvements

### Current Limitations (v0.2.0)
- Single-node deployment only (multi-node in Phase 11)
- No built-in query DSL (Cypher-like support in Phase 12)
- ChromaDB doesn't support distributed mode
- Streaming responses not supported
- No native dashboard UI (built-in Phase 13)

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
- [x] 10 phases independently deployable
- [x] 270+ tests passing (target: 75%+ coverage)
- [x] PostgreSQL backend option working
- [x] Multi-tenant support production-ready
- [x] Auth optional but enforced correctly
- [x] Docker image building and running
- [x] All 21 bug fixes resolved
- [x] CI/CD passing on every commit

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
| v0.2.0 | 2026-02-24 | Enterprise Foundation | ✓ Released |
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

