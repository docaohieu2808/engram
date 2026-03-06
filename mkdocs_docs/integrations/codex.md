# Codex CLI (OpenAI)

[Codex CLI](https://github.com/openai/codex) is OpenAI's coding agent that runs locally. It supports MCP servers as extensions.

## Setup

```bash
# Install Codex CLI
npm i -g @openai/codex

# Add engram as MCP server
codex mcp add engram \
  --env GEMINI_API_KEY="your-key" \
  -- engram-mcp

# Verify
codex mcp list
```

This creates an entry in `~/.codex/config.toml`:

```toml
[mcp_servers.engram]
command = "engram-mcp"

[mcp_servers.engram.env]
GEMINI_API_KEY = "your-key"
```

## Usage

Once configured, Codex can use engram's 21 MCP tools during sessions. Start a session:

```bash
codex "recall what we discussed about deployment"
```

Codex will automatically discover and use engram tools like `engram_recall`, `engram_remember`, and `engram_think` when relevant to your queries.

## Available Tools

All 21 engram MCP tools are available. See [MCP Tools Reference](../reference/mcp-tools.md) for the full list.

## Troubleshooting

**MCP server not connecting:**

```bash
# Check MCP server status
codex mcp list

# Test engram-mcp directly
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | engram-mcp

# Remove and re-add
codex mcp remove engram
codex mcp add engram --env GEMINI_API_KEY="your-key" -- engram-mcp
```
