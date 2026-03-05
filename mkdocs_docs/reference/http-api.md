# HTTP API Reference

Start the server:

```bash
engram serve [--host 0.0.0.0] [--port 8765]
# or as part of the full daemon:
engram start
```

Base URL: `http://localhost:8765`

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness probe |
| `POST` | `/api/v1/remember` | Store episodic memory |
| `GET` | `/api/v1/recall` | Search memories |
| `POST` | `/api/v1/think` | LLM reasoning across episodic + semantic |
| `GET` | `/api/v1/query` | Semantic graph search |
| `GET` | `/api/v1/memories` | List/filter memories with pagination |
| `GET` | `/api/v1/memories/export` | Export memories as JSON |
| `POST` | `/api/v1/meeting-ledger` | Record structured meeting |
| `POST` | `/api/v1/ingest` | Extract entities + store memories |
| `POST` | `/api/v1/feedback` | Record feedback on a memory |
| `GET` | `/api/v1/graph/data` | Graph data JSON for visualization |
| `GET` | `/graph` | Interactive graph visualization UI |
| `POST` | `/api/v1/cleanup` | Delete expired memories (admin) |
| `POST` | `/api/v1/cleanup/dedup` | Deduplicate memories (admin) |
| `POST` | `/api/v1/summarize` | LLM summary of recent memories (admin) |
| `GET` | `/api/v1/status` | Memory statistics |
| `GET` | `/api/v1/config` | Read current config |
| `PUT` | `/api/v1/config` | Write config (live update) |

## Examples

### Store a Memory

```bash
curl -X POST http://localhost:8765/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Deployed v1.0 to production",
    "memory_type": "fact",
    "priority": 8,
    "tags": ["deployment", "production"]
  }'
```

### Search Memories

```bash
curl "http://localhost:8765/api/v1/recall?query=deployment&limit=5"

# With type filter
curl "http://localhost:8765/api/v1/recall?query=deployment&type=fact&limit=10"
```

### LLM Reasoning

```bash
curl -X POST http://localhost:8765/api/v1/think \
  -H "Content-Type: application/json" \
  -d '{"question": "What deployment issues have we had?"}'
```

### Record a Meeting

```bash
curl -X POST http://localhost:8765/api/v1/meeting-ledger \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Sprint Review",
    "decisions": ["Ship v2 on Friday"],
    "action_items": ["Update docs", "Fix login bug"],
    "attendees": ["Alice", "Bob"],
    "topics": ["Release planning"]
  }'
```

### Ingest Chat Messages

```bash
curl -X POST http://localhost:8765/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "We decided to use PostgreSQL for the backend"},
      {"role": "assistant", "content": "Got it, I will remember that PostgreSQL is the backend database"}
    ]
  }'
```

### Record Feedback

```bash
curl -X POST http://localhost:8765/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"memory_id": "abc123", "feedback": "positive"}'
```

### List Memories with Pagination

```bash
curl "http://localhost:8765/api/v1/memories?limit=20&offset=0&type=decision"
```

## Authentication

When `auth.enabled: true`, include a JWT Bearer token or API key:

```bash
curl -H "Authorization: Bearer <jwt-token>" http://localhost:8765/api/v1/recall?query=test

# Or API key
curl -H "X-API-Key: <api-key>" http://localhost:8765/api/v1/recall?query=test
```

## Health Checks

```bash
# Liveness
curl http://localhost:8765/health

# Readiness (checks all subsystems)
curl http://localhost:8765/health/ready
```

Response format:

```json
{
  "status": "healthy",
  "episodic": "ok",
  "semantic": "ok",
  "llm": "ok",
  "cache": "disabled"
}
```
