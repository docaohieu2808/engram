# Goose (Block)

[Goose](https://github.com/block/goose) is Block's open-source AI agent with built-in MCP support. It runs as a CLI or desktop app.

## Setup

```bash
# Install Goose
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash

# Add engram extension to ~/.config/goose/config.yaml
```

Add the following to your `~/.config/goose/config.yaml` under `extensions:`:

```yaml
extensions:
  engram:
    enabled: true
    type: stdio
    name: engram
    description: AI agent brain with dual memory (episodic + semantic)
    display_name: Engram Memory
    bundled: false
    cmd: engram-mcp
    args: []
    env_keys:
      GEMINI_API_KEY: your-key
```

## Usage

Start a Goose session — engram tools are available automatically:

```bash
goose session
```

Goose will discover engram's 21 MCP tools and use them when relevant.

## Available Tools

All 21 engram MCP tools are available. See [MCP Tools Reference](../reference/mcp-tools.md) for the full list.

## Troubleshooting

```bash
# Check Goose config
goose info

# Test engram-mcp binary directly
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | engram-mcp
```
