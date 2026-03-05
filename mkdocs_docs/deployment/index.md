# Deployment

Engram can be deployed as a local daemon, a Docker container, or a production service with PostgreSQL and Redis.

## Deployment Options

| Option | Use Case | Complexity |
|--------|----------|------------|
| Local daemon | Single developer, local AI agents | Minimal |
| Docker (single) | Isolated environment, easy setup | Low |
| Docker Compose | Production with PostgreSQL + Redis | Medium |
| Kubernetes | Multi-tenant, high availability | High |

## Local Daemon

The simplest deployment — runs engram as a background process on your machine:

```bash
pip install engram-mem
engram init
export GEMINI_API_KEY="your-key"
engram start
```

The daemon starts:
- HTTP API server at `http://127.0.0.1:8765`
- Session watcher (if capture is enabled)
- Background scheduler (cleanup, consolidation, decay)

```bash
engram stop     # Stop daemon
engram health   # Check status
```

## Docker

See [Docker deployment](docker.md) for full instructions.

Quick start:

```bash
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest
```

## Health Checks

All deployment targets should monitor these endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness — returns 200 if process is alive |
| `GET /health/ready` | Readiness — checks episodic, semantic, and LLM subsystems |

Kubernetes probe example:

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
  initialDelaySeconds: 15
  periodSeconds: 10
```

## Resource Tier Monitoring

```bash
engram resource-status
```

| Tier | Meaning |
|------|---------|
| `FULL` | All features available |
| `STANDARD` | No LLM (API unavailable) |
| `BASIC` | Keyword search only |
| `READONLY` | Read-only mode |

The system auto-recovers to `FULL` within 60 seconds of the LLM API becoming available again.

## Backup and Restore

```bash
# Export snapshot
engram backup
# Creates: ~/.engram/backup-YYYYMMDD-HHMMSS.json

# Restore
engram restore ~/.engram/backup-20260306-013000.json
```

Snapshots include all episodic memories and semantic graph data. Schedule regular backups in production via cron or the Docker Compose setup.

## Observability

Enable OpenTelemetry and audit logging for production:

```yaml
audit:
  enabled: true
  path: /var/log/engram/audit.jsonl

telemetry:
  enabled: true
  endpoint: http://otel-collector:4317
```

Or via environment variables:

```bash
export ENGRAM_AUDIT_ENABLED=true
export ENGRAM_TELEMETRY_ENABLED=true
```
