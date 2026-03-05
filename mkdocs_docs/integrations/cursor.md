# Cursor

Engram integrates with Cursor via the MCP (Model Context Protocol) stdio transport.

## Setup

Add to Cursor's MCP settings (typically `~/.cursor/settings.json` or via the Cursor UI under Settings > MCP):

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

## Auto Setup

```bash
engram setup
```

The wizard auto-discovers Cursor's config at `~/.cursor/settings.json` and patches it automatically.

## Auto-Discovery

Engram also scans `~/.cursor/settings.json` during federation auto-discovery to find MCP providers you have configured in Cursor.

## Verify

```bash
engram setup --status
```

## Namespace Isolation

Use per-project namespaces to keep memories scoped to each workspace:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "env": {
        "GEMINI_API_KEY": "your-key",
        "ENGRAM_NAMESPACE": "my-project"
      }
    }
  }
}
```

## Available MCP Tools

All [MCP tools](../reference/mcp-tools.md) are available once connected — `engram_remember`, `engram_recall`, `engram_think`, and more.
