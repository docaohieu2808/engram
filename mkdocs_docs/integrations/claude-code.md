# Claude Code

Engram integrates with Claude Code via the MCP (Model Context Protocol) stdio transport.

## Setup

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

If `engram-mcp` is not on your `PATH` (e.g., installed in a venv), use the absolute path:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/venv/bin/engram-mcp",
      "env": { "GEMINI_API_KEY": "your-key" }
    }
  }
}
```

## Auto Setup

```bash
engram setup
```

The wizard detects Claude Code and patches `~/.claude.json` automatically, resolving the correct `engram-mcp` path.

## Verify

```bash
engram setup --status
```

Or in a Claude Code session, check available tools — you should see `engram_remember`, `engram_recall`, `engram_think`, etc.

## Session Capture

Engram can watch Claude Code session files and auto-ingest conversation context:

```yaml
# ~/.engram/config.yaml
capture:
  claude_code:
    enabled: true
    sessions_dir: ~/.claude/projects
```

Start the watcher:

```bash
engram watch --daemon
```

## Available MCP Tools

Once connected, Claude Code has access to all engram MCP tools. See the [MCP Tools reference](../reference/mcp-tools.md) for the full list.

## Recommended Workflow

Add this to your `CLAUDE.md` or system prompt:

```markdown
## Memory (Engram)
At the start of every session, use `engram_recall` to load relevant context.
When you learn important information, use `engram_remember` to store it.
```

This ensures continuity across sessions — engram acts as persistent memory for Claude Code.

## Namespace Isolation

Use separate namespaces per project to keep memories scoped:

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
