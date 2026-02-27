# Engram Deployment Guide

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- pip or conda
- GEMINI_API_KEY (for reasoning; optional for basic storage)

### Installation
```bash
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram
pip install -e ".[dev]"
export GEMINI_API_KEY="your-key-here"
```

### Start Server
```bash
engram serve
# Server runs at http://127.0.0.1:8765
curl http://127.0.0.1:8765/health
```

### Test with CLI
```bash
engram remember "First memory" --type fact --priority 5
engram recall "memory"
engram think "What have I remembered?"
```

---

## Docker Deployment

### Basic Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

EXPOSE 8765

CMD ["engram", "serve", "--host", "0.0.0.0", "--port", "8765"]
```

### Build & Run
```bash
docker build -t engram:latest .

docker run \
  -e GEMINI_API_KEY="your-key" \
  -e ENGRAM_SERVE_HOST=0.0.0.0 \
  -e ENGRAM_SERVE_PORT=8765 \
  -v engram-data:/root/.engram \
  -p 8765:8765 \
  engram:latest
```

### Docker Compose (Development)
```yaml
version: "3.8"

services:
  engram:
    build: .
    ports:
      - "8765:8765"
    environment:
      GEMINI_API_KEY: "${GEMINI_API_KEY}"
      ENGRAM_SERVE_HOST: "0.0.0.0"
      ENGRAM_SEMANTIC_PROVIDER: "sqlite"
      LOG_LEVEL: "INFO"
    volumes:
      - engram-data:/root/.engram
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  engram-data:
```

---

## Production Deployment (PostgreSQL + Redis)

### Docker Compose (Production)
```yaml
version: "3.8"

services:
  engram-api:
    build: .
    ports:
      - "8765:8765"
    depends_on:
      - postgres
      - redis
    environment:
      GEMINI_API_KEY: "${GEMINI_API_KEY}"
      ENGRAM_SERVE_HOST: "0.0.0.0"
      ENGRAM_SERVE_PORT: "8765"
      # PostgreSQL backend
      ENGRAM_SEMANTIC_PROVIDER: "postgresql"
      ENGRAM_SEMANTIC_DSN: "postgresql://engram:password@postgres:5432/engram"
      ENGRAM_SEMANTIC_POOL_MIN: "5"
      ENGRAM_SEMANTIC_POOL_MAX: "20"
      # Redis caching
      ENGRAM_CACHE_ENABLED: "true"
      ENGRAM_CACHE_REDIS_URL: "redis://redis:6379/0"
      ENGRAM_CACHE_RECALL_TTL: "300"
      ENGRAM_CACHE_THINK_TTL: "900"
      ENGRAM_CACHE_QUERY_TTL: "300"
      # Rate limiting
      ENGRAM_RATE_LIMIT_ENABLED: "true"
      ENGRAM_RATE_LIMIT_REDIS_URL: "redis://redis:6379/1"
      ENGRAM_RATE_LIMIT_REQUESTS_PER_MINUTE: "100"
      # Authentication
      ENGRAM_AUTH_ENABLED: "true"
      ENGRAM_AUTH_JWT_SECRET: "${JWT_SECRET}"
      ENGRAM_AUTH_JWT_EXPIRY_HOURS: "24"
      ENGRAM_AUTH_ADMIN_SECRET: "${ADMIN_SECRET}"
      # Audit logging
      ENGRAM_AUDIT_ENABLED: "true"
      ENGRAM_AUDIT_PATH: "/root/.engram/audit.jsonl"
      # Observability
      ENGRAM_TELEMETRY_ENABLED: "true"
      ENGRAM_TELEMETRY_OTLP_ENDPOINT: "http://otel-collector:4317"
      ENGRAM_TELEMETRY_SAMPLE_RATE: "0.1"
      LOG_LEVEL: "INFO"
      LOG_FORMAT: "json"
    volumes:
      - engram-audit:/root/.engram
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: "engram"
      POSTGRES_USER: "engram"
      POSTGRES_PASSWORD: "${DB_PASSWORD}"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U engram"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  otel-collector:
    image: otel/opentelemetry-collector:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  engram-audit:
```

### PostgreSQL Setup
```bash
# Initialize schema (runs automatically on first connection)
psql -U engram -d engram << EOF
CREATE TABLE IF NOT EXISTS nodes (
  id UUID PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  type TEXT NOT NULL,
  attributes JSONB,
  tenant_id TEXT DEFAULT 'default',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS edges (
  source_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  target_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  metadata JSONB,
  tenant_id TEXT DEFAULT 'default',
  created_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (source_id, target_id, relation)
);

CREATE INDEX idx_nodes_tenant ON nodes(tenant_id);
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_edges_tenant ON edges(tenant_id);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
EOF
```

---

## Environment Variables Reference

### Core Settings
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_SERVE_HOST | 127.0.0.1 | Bind address |
| ENGRAM_SERVE_PORT | 8765 | Listen port |
| LOG_LEVEL | WARNING | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| LOG_FORMAT | text | Logging format (text or json) |

### API Keys
| Variable | Purpose | Required |
|----------|---------|----------|
| GEMINI_API_KEY | LLM reasoning + embeddings | For think/summarize; optional for storage-only |

### Episodic Memory (ChromaDB)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_EPISODIC_PROVIDER | chromadb | Vector DB provider |
| ENGRAM_EPISODIC_PATH | ~/.engram/episodic | ChromaDB storage path |
| ENGRAM_EPISODIC_NAMESPACE | default | Collection name |

### Semantic Memory
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_SEMANTIC_PROVIDER | sqlite | Backend (sqlite or postgresql) |
| ENGRAM_SEMANTIC_PATH | ~/.engram/semantic.db | SQLite path (ignored if postgresql) |
| ENGRAM_SEMANTIC_DSN | ${ENGRAM_SEMANTIC_DSN} | PostgreSQL connection string |
| ENGRAM_SEMANTIC_POOL_MIN | 5 | Min connection pool size (PG only) |
| ENGRAM_SEMANTIC_POOL_MAX | 20 | Max connection pool size (PG only) |

### Embeddings
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_EMBEDDING_PROVIDER | gemini | Embedding provider |
| ENGRAM_EMBEDDING_MODEL | gemini-embedding-001 | Model name |

### Authentication
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_AUTH_ENABLED | false | Enable JWT + API key auth |
| ENGRAM_AUTH_JWT_SECRET | (empty) | JWT signing key (32+ chars recommended) |
| ENGRAM_AUTH_JWT_EXPIRY_HOURS | 24 | Token lifetime |
| ENGRAM_AUTH_ADMIN_SECRET | (empty) | Admin bootstrap secret for /auth/token |

### Caching (Optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_CACHE_ENABLED | false | Enable Redis caching |
| ENGRAM_CACHE_REDIS_URL | redis://localhost:6379/0 | Redis connection |
| ENGRAM_CACHE_RECALL_TTL | 300 | Recall result TTL (seconds) |
| ENGRAM_CACHE_THINK_TTL | 900 | Think result TTL (seconds) |
| ENGRAM_CACHE_QUERY_TTL | 300 | Graph query TTL (seconds) |

### Rate Limiting (Optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_RATE_LIMIT_ENABLED | false | Enable rate limiting |
| ENGRAM_RATE_LIMIT_REDIS_URL | redis://localhost:6379/0 | Redis connection |
| ENGRAM_RATE_LIMIT_REQUESTS_PER_MINUTE | 60 | Request limit |
| ENGRAM_RATE_LIMIT_BURST | 10 | Burst allowance |

### Audit Logging (Optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_AUDIT_ENABLED | false | Enable JSONL audit logging |
| ENGRAM_AUDIT_BACKEND | file | Backend (file only currently) |
| ENGRAM_AUDIT_PATH | ~/.engram/audit.jsonl | Audit log path |

### Observability (Optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_TELEMETRY_ENABLED | false | Enable OpenTelemetry |
| ENGRAM_TELEMETRY_OTLP_ENDPOINT | (empty) | Collector endpoint |
| ENGRAM_TELEMETRY_SAMPLE_RATE | 0.1 | Trace sample rate (0-1) |
| ENGRAM_TELEMETRY_SERVICE_NAME | engram | Service name in traces |

### Recall Pipeline (v0.3.1)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_RECALL_ENABLED | true | Enable recall pipeline |
| ENGRAM_RECALL_DECISION_SKIP_TRIVIAL | true | Skip trivial queries |
| ENGRAM_RECALL_ENTITY_RESOLUTION_ENABLED | true | Enable entity resolution |
| ENGRAM_RECALL_PARALLEL_SEARCH_ENABLED | true | Enable multi-source search |
| ENGRAM_RECALL_FEEDBACK_ENABLED | true | Enable feedback loop |
| ENGRAM_RECALL_AUTO_CONSOLIDATE_THRESHOLD | 20 | Messages before consolidate |
| ENGRAM_RECALL_RETRIEVAL_AUDIT_ENABLED | true | Enable retrieval audit log |

### Ingestion Guard (v0.3.1)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_INGESTION_AUTO_MEMORY_ENABLED | true | Detect save-worthy messages |
| ENGRAM_INGESTION_GUARD_ENABLED | true | Block prompt injection |

### Feedback Loop (v0.3.1)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_FEEDBACK_CONFIDENCE_POSITIVE_DELTA | 0.15 | Positive feedback boost |
| ENGRAM_FEEDBACK_CONFIDENCE_NEGATIVE_DELTA | 0.2 | Negative feedback penalty |
| ENGRAM_FEEDBACK_AUTO_DELETE_THRESHOLD | 3 | Negatives before auto-delete |

### Entity Resolution (v0.3.1)
| Variable | Default | Purpose |
|----------|---------|---------|
| ENGRAM_RESOLUTION_TEMPORAL_ENABLED | true | Temporal resolution (no LLM) |
| ENGRAM_RESOLUTION_PRONOUN_ENABLED | true | Pronoun resolution (LLM) |

---

## Configuration File

Store YAML at `~/.engram/config.yaml` or custom path:

```yaml
episodic:
  provider: chromadb
  path: ~/.engram/episodic
  namespace: default

embedding:
  provider: gemini
  model: gemini-embedding-001

semantic:
  provider: postgresql
  dsn: postgresql://engram:password@localhost/engram
  pool_min: 5
  pool_max: 20

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}

serve:
  host: 0.0.0.0
  port: 8765

auth:
  enabled: true
  jwt_secret: "use-at-least-32-chars-or-env-var"
  jwt_expiry_hours: 24
  admin_secret: "bootstrap-secret"

cache:
  enabled: true
  redis_url: redis://redis:6379/0
  recall_ttl: 300
  think_ttl: 900
  query_ttl: 300

rate_limit:
  enabled: true
  redis_url: redis://redis:6379/1
  requests_per_minute: 100
  burst: 10

audit:
  enabled: true
  backend: file
  path: ~/.engram/audit.jsonl

telemetry:
  enabled: true
  otlp_endpoint: http://otel-collector:4317
  sample_rate: 0.1
  service_name: engram

logging:
  format: json
  level: INFO
```

---

## Authentication Setup

### Generate JWT Secret
```bash
openssl rand -hex 32
# Output: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1
```

### Create Admin Bootstrap Token
```bash
# 1. Set admin_secret in config or env
export ENGRAM_AUTH_ADMIN_SECRET="bootstrap-secret-123"

# 2. Request token via HTTP
curl -X POST http://localhost:8765/api/v1/auth/token \
  -H "Authorization: Bearer bootstrap-secret-123" \
  -H "Content-Type: application/json" \
  -d '{
    "sub": "user@example.com",
    "role": "admin",
    "tenant_id": "default"
  }'

# Response:
# {
#   "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "expires_at": "2026-02-25T10:00:00Z"
# }
```

### Generate API Keys
```bash
# Create key
engram auth create-key "my-agent" --role agent

# Output:
# Key: u7dA3-p9kL2mB5x8vN4q1wY6z3cJ0sT5
# Name: my-agent
# Role: agent
# (Save this somewhere safe — it's only shown once!)

# Use in requests
curl http://localhost:8765/api/v1/remember \
  -H "X-API-Key: u7dA3-p9kL2mB5x8vN4q1wY6z3cJ0sT5" \
  -H "Content-Type: application/json" \
  -d '{"content": "test", "memory_type": "fact"}'

# List keys
engram auth list

# Revoke key
engram auth revoke "my-agent"
```

---

## Health Checks

### Liveness Probe
```bash
curl http://localhost:8765/health
# 200: {status: ok}
# 503: Dependencies down
```

### Readiness Probe
```bash
curl http://localhost:8765/health/ready
# 200: All systems operational
# 503: One or more systems down
```

### Kubernetes Probe Example
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8765
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8765
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## Backup & Restore

### Export All Memory
```bash
# Via CLI
engram backup > backup-2026-02-24.json

# Via HTTP
curl http://localhost:8765/api/v1/backup \
  -H "X-API-Key: your-key" > backup.json
```

### Import Backup
```bash
# Via CLI
engram restore backup-2026-02-24.json

# Via HTTP
curl -X POST http://localhost:8765/api/v1/restore \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d @backup.json
```

---

## Monitoring & Observability

### OpenTelemetry Collector (docker-compose)
```yaml
otel-collector:
  image: otel/opentelemetry-collector:latest
  command: ["--config=/etc/otel-config.yaml"]
  volumes:
    - ./otel-collector-config.yaml:/etc/otel-config.yaml
  ports:
    - "4317:4317"  # gRPC receiver
```

### Audit Logs
```bash
# View recent audits
tail -f ~/.engram/audit.jsonl | jq .

# Parse with jq
cat ~/.engram/audit.jsonl | \
  jq 'select(.action=="remember") | {timestamp, user_id, tenant_id}'
```

### Logs
```bash
# Text format
engram serve --host 0.0.0.0 2>&1 | grep ERROR

# JSON format
ENGRAM_LOG_FORMAT=json engram serve --host 0.0.0.0 | \
  jq 'select(.level=="ERROR")'
```

---

## Scaling Considerations

### Horizontal Scaling (Multiple API Instances)
```yaml
# Behind load balancer
engram-api-1:
  image: engram:latest
  environment:
    - ENGRAM_SEMANTIC_DSN=postgresql://...
    - ENGRAM_CACHE_REDIS_URL=redis://shared-redis:6379/0

engram-api-2:
  image: engram:latest
  environment:
    - ENGRAM_SEMANTIC_DSN=postgresql://...
    - ENGRAM_CACHE_REDIS_URL=redis://shared-redis:6379/0

# nginx upstream
upstream engram {
  server engram-api-1:8765;
  server engram-api-2:8765;
  server engram-api-3:8765;
}
```

### Multi-Tenancy Limits
- Graph cache: max 100 tenants (LRU eviction)
- Episodic stores: max 1000 (with LRU eviction)
- Increase via code modification or use PostgreSQL backend for better isolation

### Performance Tuning
| Setting | Default | Adjust For |
|---------|---------|-----------|
| ENGRAM_SEMANTIC_POOL_MAX | 20 | High concurrency: 30-50 |
| ENGRAM_CACHE_RECALL_TTL | 300 | High hit rate: 600-900 |
| ENGRAM_RATE_LIMIT_REQUESTS_PER_MINUTE | 60 | Throughput: 100-500 |
| ENGRAM_TELEMETRY_SAMPLE_RATE | 0.1 | Detailed debugging: 1.0 |

---

## Troubleshooting

### Server won't start
```bash
# Check logs
engram serve 2>&1 | head -20

# Check config
engram config show

# Check dependencies
curl http://localhost:5432  # PostgreSQL
redis-cli ping              # Redis
```

### Slow recalls
```bash
# Enable caching
ENGRAM_CACHE_ENABLED=true engram serve

# Check Redis connection
redis-cli INFO stats

# Monitor query time (via logs)
ENGRAM_LOG_LEVEL=DEBUG engram serve
```

### Auth failures
```bash
# Verify JWT secret
openssl enc -d -base64 <<< $(echo -n "secret" | openssl enc -base64)

# Test token
curl -X POST http://localhost:8765/api/v1/auth/token \
  -H "Authorization: Bearer ${ADMIN_SECRET}" \
  -H "Content-Type: application/json" \
  -d '{"sub": "test"}'

# Check API key
engram auth list
```

### Memory leaks
```bash
# Monitor graph cache
engram status | grep "Graphs cached"

# Restart to clear (or wait for LRU eviction)
systemctl restart engram
```

---

## CI/CD Integration

### GitHub Actions
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: [v*]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t engram:${{ github.sha }} .

      - name: Push to registry
        run: |
          docker tag engram:${{ github.sha }} engram:latest
          docker push engram:latest

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/engram \
            engram=engram:${{ github.sha }}
          kubectl rollout status deployment/engram
```

### Pre-deployment Checks
```bash
#!/bin/bash
set -e

# Run tests
pytest tests/ -v

# Run linting
ruff check src/

# Build Docker image
docker build -t engram:test .

# Test health check
docker run -d --name engram-test engram:test
sleep 2
docker exec engram-test curl http://localhost:8765/health
docker stop engram-test

echo "All checks passed!"
```

