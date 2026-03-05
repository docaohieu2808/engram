# Getting Started

Get engram running in under 5 minutes.

## Prerequisites

- Python 3.11+
- `GEMINI_API_KEY` — for LLM reasoning and embeddings ([get one free](https://aistudio.google.com/))
- Basic storage (episodic + semantic graph) works without the API key

## Install

```bash
pip install engram-mem
```

Or from source:

```bash
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram && pip install -e .
```

## Initialize

```bash
engram init
export GEMINI_API_KEY="your-key"
```

This creates `~/.engram/config.yaml` with sensible defaults.

## First Commands

```bash
# Start the background daemon (HTTP server + session watcher)
engram start

# Store your first memory
engram remember "Deployed v2.1 to production at 14:00 - caused 503 spike"

# Search memories
engram recall "production incidents"

# Reason across all memory
engram think "What deployment issues have we had?"

# Check system status
engram status
```

## Connect Your AI Agent

=== "Claude Code"

    Add to `~/.claude.json`:

    ```json
    {
      "mcpServers": {
        "engram": {
          "command": "engram-mcp",
          "env": { "GEMINI_API_KEY": "your-key" }
        }
      }
    }
    ```

=== "Cursor"

    Add to Cursor's MCP settings:

    ```json
    {
      "mcpServers": {
        "engram": {
          "command": "engram-mcp",
          "env": { "GEMINI_API_KEY": "your-key" }
        }
      }
    }
    ```

=== "Auto Setup Wizard"

    ```bash
    engram setup
    ```

    The wizard auto-detects installed agents (Claude Code, Cursor, OpenClaw, Windsurf, etc.) and configures them interactively.

## Next Steps

- [Detailed Installation](installation.md) — venv, Docker, source install
- [Configuration Reference](configuration.md) — all config options
- [CLI Reference](../reference/cli.md) — full command list
- [Integrations](../integrations/index.md) — connect your AI agent
