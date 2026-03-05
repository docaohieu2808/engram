# Development Guide

## Setup

```bash
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/engram/
├── config.py              # Config system (YAML + env vars + CLI)
├── auth.py                # JWT + API key auth
├── tenant.py              # Multi-tenancy via ContextVar
├── models.py              # Shared Pydantic models
├── errors.py              # Error codes + response models
├── cache.py               # Redis caching
├── rate_limiter.py        # Sliding-window rate limiting
├── audit.py               # JSONL audit logging
├── backup.py              # Backup/restore
├── health.py              # Health checks
├── episodic/              # Qdrant vector store
├── semantic/              # Knowledge graph (NetworkX + SQLite/PG)
├── reasoning/             # LLM synthesis engine
├── capture/               # HTTP server + entity extractor + watcher
├── providers/             # Federation layer + adapters
├── recall/                # Recall pipeline
├── cli/                   # Typer CLI commands
├── mcp/                   # MCP server (FastMCP)
└── setup/                 # Agent setup wizard + connectors
```

## Code Standards

- Python 3.11+ type annotations throughout
- Pydantic models for all request/response schemas
- Each module under 200 lines — split when exceeded
- `kebab-case` file names (e.g., `sqlite-backend.py`, `rate-limiter.py`)
- Try/except error handling with structured error responses
- No global state — use dependency injection and contextvars

See the [project code standards](https://github.com/docaohieu2808/Engram-Mem/blob/main/docs/code-standards.md) for detailed conventions.

## Running Locally

```bash
# Start the HTTP server in dev mode
engram serve --host 127.0.0.1 --port 8765

# Or the full daemon
engram start

# Run the MCP server (for testing MCP integration)
engram-mcp
```

## Running Tests

See [Testing](testing.md) for the full testing guide.

```bash
pytest tests/ -v
pytest tests/ --cov=src/engram
```

## Making Changes

1. Create a feature branch
2. Implement changes following [code standards](https://github.com/docaohieu2808/Engram-Mem/blob/main/docs/code-standards.md)
3. Run `pytest tests/ -v` — all tests must pass
4. Run `ruff check src/` for linting
5. Submit a pull request

## Contributing

Contributions are welcome. Please:

- Follow existing patterns — read the code before writing new code
- Keep modules focused and under 200 lines
- Write tests for new functionality
- Update this documentation if you change public interfaces

## License

MIT — Copyright (c) Do Cao Hieu
