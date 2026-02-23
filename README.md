# engram

Memory traces for AI agents - Think like human.

Dual-memory brain combining episodic (vector) + semantic (graph) memory with realtime auto-capture and human-like reasoning.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Remember something
engram remember "Fixed the database connection timeout"

# Recall memories
engram recall "database issues"

# Combined reasoning
engram think "What database issues happened recently?"

# Auto-capture from chat files
engram watch --daemon

# HTTP server for agent integration
engram serve
```
