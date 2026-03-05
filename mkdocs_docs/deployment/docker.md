# Docker Deployment

## Quick Start

```bash
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest
```

Engram starts with embedded Qdrant and SQLite — no external dependencies needed.

## Docker Compose (Production)

Full production stack with PostgreSQL and Redis:

```bash
ENGRAM_AUTH_ENABLED=true \
GEMINI_API_KEY="your-key" \
ENGRAM_SEMANTIC_PROVIDER=postgresql \
ENGRAM_SEMANTIC_DSN=postgresql://engram:password@postgres:5432/engram \
ENGRAM_CACHE_ENABLED=true \
ENGRAM_CACHE_REDIS_URL=redis://redis:6379/0 \
docker compose up
```

Or with an `.env` file:

```bash
# .env
GEMINI_API_KEY=your-key
ENGRAM_AUTH_ENABLED=true
ENGRAM_JWT_SECRET=change-this-secret
ENGRAM_SEMANTIC_PROVIDER=postgresql
ENGRAM_SEMANTIC_DSN=postgresql://engram:password@postgres:5432/engram
ENGRAM_CACHE_ENABLED=true
ENGRAM_CACHE_REDIS_URL=redis://redis:6379/0
ENGRAM_AUDIT_ENABLED=true
ENGRAM_RATE_LIMIT_ENABLED=true
```

```bash
docker compose --env-file .env up -d
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | LLM + embeddings (required) |
| `GEMINI_API_KEY_FALLBACK` | Failover key |
| `ENGRAM_AUTH_ENABLED` | Enable JWT auth |
| `ENGRAM_JWT_SECRET` | JWT signing secret (required if auth enabled) |
| `ENGRAM_SEMANTIC_PROVIDER` | `sqlite` or `postgresql` |
| `ENGRAM_SEMANTIC_DSN` | PostgreSQL DSN |
| `ENGRAM_CACHE_ENABLED` | Enable Redis caching |
| `ENGRAM_CACHE_REDIS_URL` | Redis URL |
| `ENGRAM_AUDIT_ENABLED` | Enable JSONL audit log |
| `ENGRAM_RATE_LIMIT_ENABLED` | Enable rate limiting |
| `ENGRAM_NAMESPACE` | Default namespace |

## Persisting Data

Mount volumes to persist storage outside the container:

```bash
docker run \
  -e GEMINI_API_KEY="your-key" \
  -p 8765:8765 \
  -v engram-qdrant:/root/.engram/qdrant \
  -v engram-db:/root/.engram \
  engram:latest
```

## Health Checks

The Dockerfile includes a built-in health check:

```bash
docker inspect --format='{{.State.Health.Status}}' <container-id>
```

Or test directly:

```bash
curl http://localhost:8765/health
curl http://localhost:8765/health/ready
```

## Scaling Considerations

- **Episodic store**: Switch to `episodic.mode: server` with a dedicated Qdrant instance for horizontal scaling
- **Semantic graph**: Use PostgreSQL backend with connection pooling (PgBouncer)
- **Cache**: Redis cluster or Redis Sentinel for high availability
- **Load balancing**: Engram is stateless at the HTTP layer (state is in Qdrant/PG/Redis), so multiple instances can run behind a load balancer

## Development with Docker Compose

```bash
docker compose -f docker-compose.dev.yml up
```

The dev compose file mounts the source directory for live reloading during development.
