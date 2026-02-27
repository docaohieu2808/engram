# Engram Code Standards & Conventions

## Overview

Engram follows Python 3.11+ best practices with emphasis on clarity, type safety, and testability. This document defines coding conventions, patterns, and architectural principles used throughout the project.

---

## File Organization

### Module Naming
- **kebab-case** for file names with descriptive purpose
- Examples: `auth_models.py`, `sqlite_backend.py`, `rate_limiter.py`
- CLI commands: `episodic.py`, `semantic.py`, `reasoning.py`, `auth_cmd.py`

### Directory Structure
```
src/engram/
├── __init__.py              # Top-level exports
├── config.py                # Config system
├── auth.py                  # Auth logic
├── auth_models.py           # Auth Pydantic models
├── tenant.py                # Multi-tenancy
├── errors.py                # Error codes + response models
├── logging_setup.py         # Logging configuration
├── cache.py                 # Redis caching
├── rate_limiter.py          # Rate limiting
├── audit.py                 # Audit logging
├── backup.py                # Backup/restore
├── health.py                # Health checks
├── models.py                # Shared Pydantic models (MemoryType, etc.)
├── episodic/
│   ├── __init__.py
│   ├── store.py             # ChromaDB wrapper
│   └── search.py            # Embedding & search
├── semantic/
│   ├── __init__.py
│   ├── backend.py           # Abstract interface
│   ├── sqlite_backend.py    # SQLite implementation
│   ├── pg_backend.py        # PostgreSQL implementation
│   ├── graph.py             # NetworkX wrapper
│   └── query.py             # Query DSL
├── reasoning/
│   ├── __init__.py
│   └── engine.py            # LLM synthesis
├── capture/
│   ├── __init__.py
│   ├── server.py            # FastAPI HTTP API
│   ├── extractor.py         # Entity extraction
│   └── watcher.py           # File system watcher
├── cli/
│   ├── __init__.py
│   ├── episodic.py          # remember, recall, cleanup
│   ├── semantic.py          # add/remove nodes/edges, query
│   ├── reasoning.py         # think, summarize
│   ├── auth_cmd.py          # auth commands
│   ├── config_cmd.py        # config management
│   ├── backup_cmd.py        # backup/restore
│   ├── migrate_cmd.py       # migration utilities
│   └── system.py            # status, dump, serve
├── mcp/
│   ├── __init__.py
│   ├── server.py            # MCP protocol
│   ├── episodic_tools.py    # Episodic MCP tools
│   ├── semantic_tools.py    # Semantic MCP tools
│   └── reasoning_tools.py   # Reasoning MCP tools
└── schema/
    ├── __init__.py
    ├── loader.py            # Schema loading
    └── builtin/
        ├── devops.yaml
        ├── marketing.yaml
        └── personal.yaml
```

---

## Type Hints & Validation

### Always Use Type Hints
```python
# Good
def remember(
    content: str,
    memory_type: MemoryType,
    priority: int,
    tags: list[str] | None = None,
) -> str:  # Returns memory ID
    pass

# Bad
def remember(content, memory_type, priority, tags=None):
    pass
```

### Use Pydantic for Validation
```python
from pydantic import BaseModel, Field

class RememberRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    memory_type: MemoryType = MemoryType.FACT
    priority: int = Field(default=5, ge=1, le=10)
    tags: list[str] = []
```

### Avoid `Any` Type
```python
# Good
from typing import Optional

def get_config_value(config: Config, key_path: str) -> Any:
    """OK here because config structure is dynamic."""
    pass

# Bad — too generic
def process(data: Any) -> Any:
    pass
```

---

## Async/Await Patterns

### Always Mark Async Functions Explicitly
```python
# Good
async def get_graph(self, tenant_id: str) -> SemanticGraph:
    async with await self._get_lock():
        return self._graphs[tenant_id]

# Use asyncio.run() at entry point only
if __name__ == "__main__":
    asyncio.run(main())
```

### Avoid Blocking Calls in Async Context
```python
# Good
from asyncpg import create_pool
pool = await create_pool("postgresql://...")

# Bad
import time
time.sleep(1)  # Never in async!
```

### Handle Async Context Variables
```python
from contextvars import ContextVar

_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="default")

class TenantContext:
    @staticmethod
    def set(tid: str) -> None:
        _tenant_id.set(tid)

    @staticmethod
    def get() -> str:
        return _tenant_id.get()
```

---

## Error Handling

### Use Structured Errors
```python
from engram.errors import EngramError, ErrorCode

# Good
if len(content) < 1:
    raise EngramError(ErrorCode.INVALID_REQUEST, "Content required")

# Bad
raise ValueError("Content required")
```

### HTTP Exception Mapping
```python
from fastapi import HTTPException

try:
    # business logic
except EngramError as e:
    if e.error_code == ErrorCode.FORBIDDEN:
        raise HTTPException(status_code=403, detail=e.message)
    elif e.error_code == ErrorCode.UNAUTHORIZED:
        raise HTTPException(status_code=401, detail=e.message)
```

### Always Log Errors with Context
```python
import logging

logger = logging.getLogger("engram")

try:
    recall(query)
except Exception as e:
    logger.error("Recall failed", exc_info=True, extra={
        "query": query,
        "correlation_id": correlation_id.get(),
        "tenant_id": TenantContext.get(),
    })
    raise
```

---

## Configuration Management

### Access Config via Module-Level Cache
```python
# Good — cached at startup
_config: Config | None = None

def init_config(config: Config) -> None:
    global _config
    _config = config

def _get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config

# Bad — reloads every request
def get_auth_context(request: Request) -> AuthContext:
    config = load_config()  # Don't do this!
    ...
```

### Use Env Var Expansion, Not os.environ Directly
```python
# Good — transparent and testable
class LLMConfig(BaseModel):
    api_key: str = "${GEMINI_API_KEY}"

config = load_config()  # Expands ${...}

# Bad
import os
api_key = os.environ.get("GEMINI_API_KEY", "default")
```

---

## Logging Standards

### Use Module Logger
```python
import logging

logger = logging.getLogger("engram")  # or specific module
```

### Include Context in Log Records
```python
from engram.logging_setup import correlation_id

logger.info("Remembering memory", extra={
    "correlation_id": correlation_id.get(),
    "tenant_id": TenantContext.get(),
    "content_len": len(content),
})
```

### Respect Log Levels
- **DEBUG:** Detailed tracing (disabled in production)
- **INFO:** User-facing events (memory stored, API called)
- **WARNING:** Potential issues (Redis unavailable, cache miss)
- **ERROR:** Failures requiring action (query failed, auth invalid)
- **CRITICAL:** System-level failures (database unreachable)

### Silence Third-Party Noise
```python
# In logging_setup.py
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
```

---

## Testing Conventions

### Test File Organization
```
tests/
├── test_config.py           # Config loading, env var expansion
├── test_auth.py             # JWT, API keys, RBAC
├── test_episodic_store.py   # Remember, recall, cleanup
├── test_semantic_graph.py   # Nodes, edges, queries
├── test_reasoning_engine.py # LLM synthesis (mocked)
├── test_http_server.py      # All HTTP endpoints
├── test_cache.py            # Redis caching
├── test_rate_limiter.py     # Rate limiting
├── test_cli_commands.py     # CLI commands
├── test_tenant_isolation.py # Multi-tenant isolation
├── test_health_checks.py    # Health endpoints
├── test_backup_restore.py   # Snapshot/restore
└── test_audit.py            # Audit logging
```

### Use Pytest Fixtures for Setup
```python
import pytest

@pytest.fixture
def config() -> Config:
    """Provides test config."""
    return Config(
        episodic=EpisodicConfig(namespace="test"),
        semantic=SemanticConfig(path=":memory:"),
    )

@pytest.fixture
async def episodic_store(config: Config) -> EpisodicStore:
    """Provides test episodic store."""
    store = EpisodicStore(config.episodic, config.embedding, namespace="test")
    yield store
    # Cleanup
```

### Mock External Services
```python
from unittest.mock import patch, AsyncMock

@patch("engram.reasoning.engine.litellm.acompletion")
async def test_think(mock_llm: AsyncMock) -> None:
    mock_llm.return_value = {"choices": [{"message": {"content": "answer"}}]}
    engine = ReasoningEngine(config)
    result = await engine.think("question")
    assert "answer" in result
```

### Test Both Happy Path and Error Cases
```python
async def test_remember_success(episodic_store: EpisodicStore) -> None:
    """Happy path: valid content stored."""
    mem_id = await episodic_store.remember("test", MemoryType.FACT)
    assert mem_id is not None

async def test_remember_empty_content(episodic_store: EpisodicStore) -> None:
    """Error path: empty content rejected."""
    with pytest.raises(EngramError) as exc:
        await episodic_store.remember("", MemoryType.FACT)
    assert exc.value.error_code == ErrorCode.INVALID_REQUEST
```

### Mark Integration Tests
```python
@pytest.mark.integration
async def test_http_server_end_to_end() -> None:
    """Full flow: HTTP → auth → episodic → reasoning."""
    async with AsyncClient(app=create_app()) as client:
        res = await client.post("/remember", json={"content": "test"})
        assert res.status_code == 200
```

---

## FastAPI Patterns

### Use Dependency Injection
```python
from fastapi import Depends, HTTPException

async def get_auth_context(request: Request) -> AuthContext:
    """Middleware: extract and validate auth."""
    ...

@app.post("/api/v1/remember")
async def remember_endpoint(
    req: RememberRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> MemoryResponse:
    """Auth context automatically injected."""
    TenantContext.set(auth.tenant_id)
    ...
```

### Use Structured Responses
```python
from pydantic import BaseModel

class MemoryResponse(BaseModel):
    data: dict
    meta: dict = Field(default_factory=dict)

@app.post("/api/v1/remember")
async def remember_endpoint(...) -> MemoryResponse:
    mem_id = await store.remember(...)
    return MemoryResponse(
        data={"id": mem_id},
        meta={"request_id": request.headers.get("X-Request-ID")},
    )
```

### Exception Handlers
```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(EngramError)
async def engram_error_handler(request: Request, exc: EngramError) -> JSONResponse:
    logger.error(f"Engram error: {exc.error_code}", extra={"error": str(exc)})
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": exc.error_code.value,
                "message": exc.message,
            },
            "meta": {
                "correlation_id": correlation_id.get(),
            },
        },
    )
```

---

## Code Quality Guidelines

### Line Length
- **Target:** 100 characters (configured in pyproject.toml via ruff)
- **Exception:** Strings, URLs, very long names

### Imports
```python
# Order: stdlib, third-party, local (blank lines between groups)
import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from engram.config import Config
from engram.errors import EngramError
```

### Naming Conventions
| Thing | Convention | Example |
|-------|-----------|---------|
| Modules | snake_case | `auth_models.py` |
| Classes | PascalCase | `EpisodicStore` |
| Functions | snake_case | `get_config_value()` |
| Constants | UPPER_SNAKE_CASE | `_MAX_GRAPH_CACHE = 100` |
| Private | Leading underscore | `_get_config()` |
| Type aliases | PascalCase | `MemoryID = str` |

### Docstrings
```python
def remember(
    content: str,
    memory_type: MemoryType,
    priority: int,
    tags: list[str] | None = None,
) -> str:
    """Store episodic memory.

    Args:
        content: Memory text (required, max 10KB)
        memory_type: Category (fact, decision, error, etc.)
        priority: 1-10 importance level
        tags: Optional labels for grouping

    Returns:
        Memory ID (UUID string)

    Raises:
        EngramError: If content is empty or too large
    """
    if not content or len(content) > 10_000:
        raise EngramError(ErrorCode.INVALID_REQUEST, "Content invalid")
    ...
```

### Comments
```python
# Good — explains why, not what
def cleanup(self) -> int:
    """Delete expired memories.

    Fires async to avoid blocking caller.
    """
    # Remove only if expiry timestamp is in past
    return self._delete_expired()

# Bad — restates code
deleted_count = 0  # Create counter
for memory in memories:  # Loop through
    if memory.expires < now:  # Check if expired
        self.delete(memory)  # Delete
        deleted_count += 1  # Increment
```

---

## Performance Considerations

### Connection Pooling
```python
# Good — reuse connections
pool = await asyncpg.create_pool(dsn, min_size=5, max_size=20)

# Bad — creates new connection per query
async with asyncpg.connect(dsn) as conn:
    await conn.execute(...)
```

### Caching Strategy
```python
# Cache expensive operations, not simple ones
@cache.get_or_set("episodic:recall", ttl=300)
async def recall(query: str) -> list:
    return await store.recall(query)  # Expensive vector search

# Don't cache trivial lookups
config = load_config()  # Already cached at module level
```

### Batch Operations
```python
# Good — batch inserts
async with graph.batch_insert() as batch:
    for node in nodes:
        await batch.add_node(node)

# Bad — one at a time
for node in nodes:
    await graph.add_node(node)
```

---

## Security Practices

### Never Log Secrets
```python
# Good
logger.info("Auth configured", extra={"jwt_enabled": config.auth.enabled})

# Bad
logger.debug(f"JWT secret: {config.auth.jwt_secret}")
```

### Validate Input Early
```python
# Good — validate at boundaries
class RememberRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)

# Bad — validate deep in business logic
def remember(content: str) -> str:
    if len(content) < 1:
        ...
```

### Use Hashing for Sensitive Data
```python
from hashlib import sha256

# Good
key_hash = sha256(api_key.encode()).hexdigest()
stored_hash = api_key_record.key_hash

# Bad
stored_key = api_key  # Never store plaintext
```

---

## Backward Compatibility

### Deprecate Gracefully
```python
# Old endpoint still works, redirects to new
@app.get("/remember")
async def remember_legacy(...) -> RedirectResponse:
    """Deprecated. Use POST /api/v1/remember instead."""
    logger.warning("Legacy endpoint used", extra={"path": "/remember"})
    return RedirectResponse("/api/v1/remember")
```

### Version APIs
```python
# Good — versioned
v1 = APIRouter(prefix="/api/v1")

@v1.post("/remember")
async def remember_v1(...) -> MemoryResponse:
    ...

# Future
v2 = APIRouter(prefix="/api/v2")
@v2.post("/remember")
async def remember_v2(...) -> MemoryResponseV2:
    ...
```

---

---

## Embedding Model Policy

### Supported Model

Only `gemini-embedding-001` (3072 dimensions) is supported. Do NOT use other embedding models.

```python
# Good — use the module constant
from engram.episodic.embeddings import EMBEDDING_DIM  # = 3072

# Bad — hardcode dimension
embedding_dim = 768  # Wrong model assumed
```

Rationale: ChromaDB collections lock to a fixed embedding dimension at creation time. Mixing models corrupts collections and requires full re-indexing.

### Fallback Behaviour

When `GEMINI_API_KEY` is not set, ChromaDB falls back to `all-MiniLM-L6-v2` (384 dims). This fallback exists for offline/testing scenarios only. Never configure production with mismatched dimensions.

### Key Rotation Configuration

Use `src/engram/episodic/embeddings.py` — never call `litellm.embedding()` directly from other modules.

```yaml
# config.yaml — key rotation settings
embedding:
  provider: gemini
  model: gemini-embedding-001
  key_strategy: failover   # or round-robin
```

```bash
# Environment variables for key rotation
export GEMINI_API_KEY="primary-key"
export GEMINI_API_KEY_FALLBACK="secondary-key"
export GEMINI_KEY_STRATEGY="round-robin"   # overrides config.yaml
```

**Strategies:**
- `failover` (default): always try primary key first; use fallback only on `AuthenticationError`
- `round-robin`: rotate keys evenly using a process-level counter; use to spread quota

**When to use round-robin:** high-throughput scenarios where a single API key would hit rate limits under sustained embedding load.

---

## Review Checklist

Before commit, verify:
- [ ] Type hints on all functions
- [ ] Docstrings on public APIs
- [ ] Error handling with EngramError
- [ ] Logging with context (correlation_id, tenant_id)
- [ ] Tests covering happy path + errors
- [ ] No hardcoded secrets
- [ ] Async functions for I/O-bound work
- [ ] Pydantic validation at boundaries
- [ ] Line length <100 chars (ruff check)
- [ ] Imports organized (stdlib, third-party, local)
- [ ] Embedding calls go through `embeddings.py`, not direct litellm calls
- [ ] Embedding model is `gemini-embedding-001` only

