# Gemini CLI (Google)

[Gemini CLI](https://github.com/google-gemini/gemini-cli) is Google's AI coding agent with native MCP support.

## Setup

```bash
# Install Gemini CLI
npm i -g @google/gemini-cli

# Add engram as MCP server
gemini mcp add engram engram-mcp \
  --scope user \
  -e "GEMINI_API_KEY=your-key" \
  --trust \
  --description "AI agent brain with dual memory"

# Verify connection
gemini mcp list
```

## Usage

Start a session — engram tools are available automatically:

```bash
gemini
```

Ask Gemini to use engram:

```
> recall what we discussed about deployment
> remember "switched to PostgreSQL for semantic graph"
> think "what are the recurring issues in our infrastructure?"
```

## Available Tools

All 21 engram MCP tools are available. See [MCP Tools Reference](../reference/mcp-tools.md) for the full list.

## Troubleshooting

```bash
# Check server status
gemini mcp list

# Re-add if needed
gemini mcp remove engram
gemini mcp add engram engram-mcp --scope user -e "GEMINI_API_KEY=your-key" --trust
```
