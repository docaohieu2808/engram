# Integrations

Engram works with any MCP-compatible AI agent or IDE. It also federates with external memory systems via auto-discovery.

## Supported Agents

| Agent | Protocol | Setup |
|-------|----------|-------|
| Claude Code | MCP (stdio) | [Guide](claude-code.md) |
| Codex CLI (OpenAI) | MCP (stdio) | [Guide](codex.md) |
| Cursor | MCP (stdio) | [Guide](cursor.md) |
| Goose (Block) | MCP (stdio) | [Guide](goose.md) |
| OpenClaw | MCP + Session Capture | [Guide](openclaw.md) |
| Windsurf | MCP (stdio) | `engram setup` |
| Cline | MCP (stdio) | `engram setup` |
| Any MCP client | MCP (stdio) | `command: engram-mcp` |

## Auto Setup

The fastest way to connect any supported agent:

```bash
engram setup
```

The wizard auto-detects installed agents and configures them interactively. Use flags for CI/headless environments:

```bash
engram setup --non-interactive   # Configure all detected agents
engram setup --dry-run           # Preview without writing
engram setup --status            # Show current connection status
```

## Federated Knowledge Providers

Engram can query external memory systems in parallel alongside its own stores. See [Federated Providers](federated-providers.md).

## MCP Entry Point

All agents use the same MCP server binary:

```bash
engram-mcp
```

This runs engram as an MCP stdio server, exposing all [MCP tools](../reference/mcp-tools.md) to the connected agent.
