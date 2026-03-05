# Installation

## Requirements

- Python 3.11 or higher
- pip 23+
- `GEMINI_API_KEY` (optional — required for LLM reasoning and embeddings)

## From PyPI

```bash
pip install engram-mem
```

Installs the `engram` CLI and `engram-mcp` MCP server entry point.

## In a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install engram-mem
```

## From Source

```bash
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram
pip install -e .
```

## Docker

```bash
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest
```

See [Docker deployment](../deployment/docker.md) for production setup with PostgreSQL and Redis.

## Verify Installation

```bash
engram --version
engram health
```

## Post-Install Setup

```bash
# Initialize config file at ~/.engram/config.yaml
engram init

# Set API key
export GEMINI_API_KEY="your-key"

# Or add to ~/.engram/config.yaml:
# llm:
#   api_key: your-key
```

## Auto-Setup Wizard

```bash
engram setup
```

Detects and configures all installed AI agents (Claude Code, Cursor, OpenClaw, Windsurf, Cline, Aider, Zed). Use `--dry-run` to preview without writing, `--status` to check current connections.

## Upgrading

```bash
pip install --upgrade engram-mem
```

Check the [Changelog](../development/changelog.md) for breaking changes before upgrading.
