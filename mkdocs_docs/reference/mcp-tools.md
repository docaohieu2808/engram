# MCP Tools

All tools are available when engram is connected as an MCP server (`engram-mcp`).

## Tool Reference

| Tool | Description |
|------|-------------|
| `engram_remember` | Store memory with type, priority, tags, namespace |
| `engram_recall` | Search episodic memories (compact format by default) |
| `engram_think` | Reason across episodic + semantic memory via LLM |
| `engram_status` | Show memory statistics |
| `engram_get_memory` | Retrieve full memory content by ID or prefix |
| `engram_timeline` | Chronological context around a memory |
| `engram_add_entity` | Add entity node to knowledge graph |
| `engram_add_relation` | Add relationship edge between entities |
| `engram_query_graph` | Query knowledge graph |
| `engram_ingest` | Dual ingest: extract entities + store memories |
| `engram_meeting_ledger` | Record structured meeting (decisions, action items) |
| `engram_feedback` | Record positive/negative feedback on memories |
| `engram_auto_feedback` | Auto-detect feedback from conversation context |
| `engram_cleanup` | Delete all expired memories |
| `engram_cleanup_dedup` | Deduplicate similar memories by cosine similarity |
| `engram_summarize` | Summarize recent N memories via LLM |
| `engram_session_start` | Begin new conversation session |
| `engram_session_end` | End active session with optional summary |
| `engram_session_summary` | Get summary of completed session |
| `engram_session_context` | Retrieve memories from active session |
| `engram_ask` | Smart query — auto-routes to recall or think |

## Tool Details

### `engram_remember`

Store a memory with optional metadata.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | string | The memory content |
| `memory_type` | string | fact, decision, preference, todo, error, context, workflow, meeting_ledger |
| `priority` | int | 1-10, default 5 |
| `tags` | list[str] | Optional tags for filtering |
| `namespace` | string | Tenant namespace |
| `expires` | string | Expiry: `2h`, `1d`, `7d` |
| `topic_key` | string | Unique key for upsert (replaces existing) |

### `engram_recall`

Search episodic memories by semantic similarity.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query |
| `limit` | int | Max results, default 5 |
| `memory_type` | string | Optional type filter |
| `tags` | list[str] | Optional tag filter |
| `resolve_entities` | bool | Resolve pronoun/entity references |
| `resolve_temporal` | bool | Resolve date/time expressions |

### `engram_think`

Run LLM reasoning across all memory (episodic + semantic + federated).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | string | The question to reason about |

### `engram_ingest`

Extract entities from messages and store as memories. This is the primary tool for bulk ingestion — it runs entity extraction first and only stores messages that contain meaningful entities (entity-gated).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | list | List of `{role, content}` message objects |

### `engram_meeting_ledger`

Record a structured meeting with decisions and action items.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | string | Meeting title |
| `decisions` | list[str] | Decisions made |
| `action_items` | list[str] | Tasks to follow up |
| `attendees` | list[str] | Participants |
| `topics` | list[str] | Topics discussed |

### `engram_feedback`

Record feedback to adjust memory confidence scores.

- Positive: +0.15 confidence
- Negative: -0.2 confidence
- 3x negative + low confidence: auto-delete

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | string | Memory ID |
| `feedback` | string | `positive` or `negative` |

## Session Tools

Session tools track context within a conversation:

```
engram_session_start  →  engram_session_context (during)  →  engram_session_end
                                                              engram_session_summary
```

Sessions allow the agent to retrieve only the memories from the current conversation, separate from the full memory store.

## Recommended Usage Pattern

```python
# Start of session
engram_session_start()
engram_recall("relevant context for current task")

# During session
engram_remember("important decision: use Redis for caching")
engram_add_entity("Redis", type="Technology")

# End of session
engram_session_end(summary="Implemented Redis caching layer")
```
