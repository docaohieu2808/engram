# Environment Variables

All configuration can be set via environment variables. They take priority over `~/.engram/config.yaml` values.

## Core

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Primary key for LLM reasoning and embeddings | — |
| `GEMINI_API_KEY_FALLBACK` | Secondary key for failover/round-robin rotation | — |
| `ENGRAM_NAMESPACE` | Memory namespace for tenant isolation | `default` |
| `ENGRAM_CONFIG` | Path to config YAML file | `~/.engram/config.yaml` |

## Server

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_HOST` | HTTP server bind address | `127.0.0.1` |
| `ENGRAM_PORT` | HTTP server port | `8765` |

## Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_EPISODIC_MODE` | `embedded` or `server` | `embedded` |
| `ENGRAM_EPISODIC_PATH` | Qdrant data directory | `~/.engram/qdrant` |
| `ENGRAM_SEMANTIC_PROVIDER` | `sqlite` or `postgresql` | `sqlite` |
| `ENGRAM_SEMANTIC_PATH` | SQLite database path | `~/.engram/semantic.db` |
| `ENGRAM_SEMANTIC_DSN` | PostgreSQL DSN | — |

## Authentication

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_AUTH_ENABLED` | Enable JWT + API key auth | `false` |
| `ENGRAM_JWT_SECRET` | JWT signing secret | — |

## Caching

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_CACHE_ENABLED` | Enable Redis caching | `false` |
| `ENGRAM_CACHE_REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |

## Observability

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_AUDIT_ENABLED` | Enable JSONL audit log | `false` |
| `ENGRAM_AUDIT_PATH` | Audit log file path | `~/.engram/audit.jsonl` |
| `ENGRAM_TELEMETRY_ENABLED` | Enable OpenTelemetry | `false` |

## Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_RATE_LIMIT_ENABLED` | Enable per-tenant rate limiting | `false` |

## Usage Example

```bash
# Minimal setup
export GEMINI_API_KEY="your-key"
engram start

# Production setup with PostgreSQL + Redis
export GEMINI_API_KEY="your-key"
export GEMINI_API_KEY_FALLBACK="backup-key"
export ENGRAM_NAMESPACE="production"
export ENGRAM_AUTH_ENABLED="true"
export ENGRAM_JWT_SECRET="your-secret"
export ENGRAM_SEMANTIC_PROVIDER="postgresql"
export ENGRAM_SEMANTIC_DSN="postgresql://user:pass@localhost:5432/engram"
export ENGRAM_CACHE_ENABLED="true"
export ENGRAM_CACHE_REDIS_URL="redis://localhost:6379/0"
export ENGRAM_AUDIT_ENABLED="true"
export ENGRAM_RATE_LIMIT_ENABLED="true"
engram serve --host 0.0.0.0 --port 8765
```

## Docker

Pass environment variables to the container:

```bash
docker run \
  -e GEMINI_API_KEY="your-key" \
  -e ENGRAM_SEMANTIC_PROVIDER="postgresql" \
  -e ENGRAM_SEMANTIC_DSN="postgresql://user:pass@postgres:5432/engram" \
  -p 8765:8765 \
  engram:latest
```

Or use an env file:

```bash
docker run --env-file .env -p 8765:8765 engram:latest
```
