# CLI Reference

All commands are available via the `engram` binary after installation.

## Memory Operations

```bash
# Store a memory
engram remember <content> \
  [--type fact|decision|preference|todo|error|context|workflow|meeting_ledger] \
  [--priority 1-10] \
  [--tags tag1,tag2] \
  [--expires 2h|1d|7d] \
  [--topic-key unique-key]

# Search episodic memories
engram recall <query> \
  [--limit 5] \
  [--type <type>] \
  [--tags tag1,tag2] \
  [--resolve-entities] \
  [--resolve-temporal]

# Smart query — auto-routes to recall or think
engram ask <question>

# Reason across all memory (episodic + semantic)
engram think <question>

# Summarize recent N memories via LLM
engram summarize [--count 20] [--save]
```

## Semantic Graph

```bash
# Add a node (entity)
engram add node <name> --type <NodeType>

# Add an edge (relationship)
engram add edge <from_key> <to_key> --relation <relation>

# Remove a node
engram remove node <key>

# Query the graph
engram query [<keyword>] \
  [--type <NodeType>] \
  [--related-to <name>] \
  [--format table|json]

# Auto-link orphaned nodes
engram autolink-orphans [--apply] [--min-co-mentions 3]

# Open interactive graph visualization in browser
engram graph
```

## Browse & Export

```bash
# Summary counts (memories, nodes, edges)
engram status

# Rich tables: all memories, nodes, edges
engram dump

# Full JSON export
engram dump --format json

# Ebbinghaus decay report
engram decay [--limit 20]
```

## System

```bash
# Initialize config at ~/.engram/config.yaml
engram init

# Start daemon (HTTP server + session watcher)
engram start

# Stop daemon
engram stop

# Start foreground HTTP server
engram serve [--host 0.0.0.0] [--port 8765]

# Watch inbox + OpenClaw sessions
engram watch [--daemon]

# Full system health check
engram health

# Resource tier status (FULL/STANDARD/BASIC/READONLY)
engram resource-status

# Embedding queue status
engram queue-status

# Background task schedule
engram scheduler-status

# 3-law governance + SHA-256 hash
engram constitution-status
```

## Maintenance

```bash
# Delete expired memories
engram cleanup

# LLM-driven memory consolidation (cluster + summarize)
engram consolidate [--limit 50]

# Ingest chat JSON (entity extraction + memory storage)
engram ingest <file.json> [--dry-run]

# Export memory snapshot
engram backup

# Import snapshot
engram restore <file>

# Config management
engram config show
engram config get <key>
engram config set <key> <value>

# Record feedback
engram feedback <id> --positive
engram feedback <id> --negative
```

## Setup & Integration

```bash
# Interactive agent setup wizard
engram setup

# Preview without writing
engram setup --dry-run

# Configure all detected agents automatically (CI/headless)
engram setup --non-interactive

# Show current connection status
engram setup --status

# Discover federated providers
engram discover
```

## Memory Types

| Type | Description |
|------|-------------|
| `fact` | General fact or information (default) |
| `decision` | A decision made |
| `preference` | User or agent preference |
| `todo` | Task or action item |
| `error` | Error or bug encountered |
| `context` | Contextual information |
| `workflow` | Process or workflow step |
| `meeting_ledger` | Structured meeting record |
