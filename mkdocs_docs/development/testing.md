# Testing

Engram has 894+ tests with 61%+ code coverage, run via GitHub Actions CI on every push.

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/engram --cov-report=term-missing

# Specific test suites
pytest tests/ -k "recall or feedback"
pytest tests/ -k "episodic"
pytest tests/ -k "semantic"
pytest tests/ -k "api"
```

## Test Structure

```
tests/
├── test_episodic.py         # EpisodicStore: CRUD, search, decay, feedback
├── test_semantic.py         # SemanticGraph: nodes, edges, query DSL
├── test_reasoning.py        # ReasoningEngine: think, summarize
├── test_recall_pipeline.py  # Full recall pipeline stages
├── test_api.py              # HTTP API endpoint tests (FastAPI TestClient)
├── test_websocket.py        # WebSocket command + push event tests
├── test_mcp.py              # MCP tool tests
├── test_auth.py             # JWT + API key auth
├── test_federation.py       # Provider adapters + query router
├── test_ingestion.py        # Entity extraction + entity gate
├── test_classifier.py       # Memory type classifier
├── test_capture.py          # Session watcher
└── test_cli.py              # CLI command tests (Typer runner)
```

## Benchmarks

```bash
# Run the benchmark suite
pytest tests/ -k "benchmark" -v

# Or via CLI
engram benchmark run
```

Benchmarks measure p50/p95/p99 latency for:
- `recall` (vector search)
- `think` (LLM reasoning)
- `ingest` (entity extraction + storage)
- `api` (all HTTP endpoints)

## CI/CD

Tests run automatically on GitHub Actions on every push and pull request to `main`. The workflow runs:

1. `ruff check` — linting
2. `pytest tests/ --cov=src/engram` — full test suite with coverage
3. Coverage report uploaded to CI summary

## Writing Tests

Follow the existing patterns:

```python
import pytest
from engram.episodic.store import EpisodicStore
from engram.config import EngramConfig

@pytest.fixture
def store(tmp_path):
    config = EngramConfig(episodic={"mode": "embedded", "path": str(tmp_path)})
    return EpisodicStore(config)

def test_remember_and_recall(store):
    memory_id = store.remember("PostgreSQL is our database", memory_type="fact")
    results = store.recall("database", limit=5)
    assert any(r.id == memory_id for r in results)
```

Key conventions:
- Use `tmp_path` fixture for isolated storage in each test
- Test both happy path and error scenarios
- Do not use mocks for storage — test against real (embedded) backends
- Keep individual test functions focused and under 50 lines
