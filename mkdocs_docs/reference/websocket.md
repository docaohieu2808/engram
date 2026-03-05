# WebSocket API

The WebSocket API provides real-time bidirectional communication with engram.

**Endpoint:** `ws://host:8765/ws`

**Authentication:** `ws://host:8765/ws?token=<JWT>` (token optional when auth disabled)

## Connection

```javascript
const ws = new WebSocket("ws://localhost:8765/ws");

ws.onopen = () => {
  ws.send(JSON.stringify({ command: "status", payload: {} }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

## Commands

All commands follow the format:

```json
{
  "command": "<command-name>",
  "payload": { ... }
}
```

### `remember`

Store a memory.

```json
{
  "command": "remember",
  "payload": {
    "content": "Switched to PostgreSQL backend",
    "memory_type": "decision",
    "priority": 9
  }
}
```

### `recall`

Search episodic memories.

```json
{
  "command": "recall",
  "payload": {
    "query": "database decisions",
    "limit": 5
  }
}
```

### `think`

LLM reasoning across episodic + semantic memory.

```json
{
  "command": "think",
  "payload": {
    "question": "What architecture decisions have we made?"
  }
}
```

### `feedback`

Record positive or negative feedback on a memory.

```json
{
  "command": "feedback",
  "payload": {
    "memory_id": "abc123",
    "feedback": "positive"
  }
}
```

### `query`

Search the semantic knowledge graph.

```json
{
  "command": "query",
  "payload": {
    "keyword": "PostgreSQL"
  }
}
```

### `ingest`

Extract entities from messages and store memories.

```json
{
  "command": "ingest",
  "payload": {
    "messages": [
      { "role": "user", "content": "We decided to use Redis for caching" }
    ]
  }
}
```

### `status`

Get memory statistics.

```json
{
  "command": "status",
  "payload": {}
}
```

## Push Events

Engram pushes events to all connected clients when memory state changes:

| Event | Trigger |
|-------|---------|
| `memory_created` | New memory stored |
| `memory_updated` | Memory modified (feedback, consolidation) |
| `memory_deleted` | Memory deleted (expired, negative feedback threshold) |
| `feedback_recorded` | Feedback saved |

Push event format:

```json
{
  "event": "memory_created",
  "data": {
    "memory_id": "abc123",
    "content": "...",
    "memory_type": "fact",
    "created_at": "2026-03-06T01:30:00Z"
  }
}
```

## Per-Tenant Isolation

The WebSocket event bus is isolated per tenant. Clients only receive push events for their own tenant's memories. Tenant context is derived from the JWT token when auth is enabled.

## Python Example

```python
import asyncio
import websockets
import json

async def main():
    async with websockets.connect("ws://localhost:8765/ws") as ws:
        # Store a memory
        await ws.send(json.dumps({
            "command": "remember",
            "payload": {"content": "Test memory", "priority": 5}
        }))
        response = await ws.recv()
        print(json.loads(response))

asyncio.run(main())
```
