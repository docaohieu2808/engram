# Plugin Authoring Guide

## Overview

Engram's provider system lets you plug in external memory sources (databases, APIs, files, etc.)
by implementing `MemoryProvider` and registering via Python package entry-points.

## Implement MemoryProvider ABC

```python
# my_package/my_provider.py
from engram.providers.base import MemoryProvider, ProviderResult

class MyProvider(MemoryProvider):
    def __init__(self, name: str, url: str, **kwargs):
        super().__init__(name=name, provider_type="my-provider", **kwargs)
        self.url = url

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        # Fetch results from your backend
        # Return empty list on error (fail-open)
        try:
            results = await _fetch_from_backend(self.url, query, limit)
            return [ProviderResult(content=r["text"], score=r["score"]) for r in results]
        except Exception:
            return []

    async def health(self) -> bool:
        try:
            resp = await _ping(self.url)
            return resp.ok
        except Exception:
            return False
```

Required methods:
- `search(query, limit)` — semantic search, return `list[ProviderResult]`
- `health()` — reachability check, return `bool`

Optional override:
- `add(content, metadata)` — write support, return ID string or `None`

`ProviderResult` fields: `content` (str), `score` (float 0-1), `source` (str), `metadata` (dict).

## Register via pyproject.toml

```toml
[project.entry-points."engram.providers"]
my-provider = "my_package.my_provider:MyProvider"
```

The entry-point name (`my-provider`) is the `type` string used in Engram config:

```yaml
# ~/.engram/config.yaml
providers:
  - name: my-source
    type: my-provider        # matches entry-point name
    enabled: true
    url: "https://my-backend/search"
```

## Reference: FileAdapter

`engram.providers.file_adapter:FileAdapter` searches markdown/text files
in a local directory. Entry-point name: `file`.

```yaml
providers:
  - name: obsidian-vault
    type: file
    path: ~/Documents/Obsidian
    pattern: "*.md"
```

## Built-in Provider Types

| type       | class                                    | description              |
|------------|------------------------------------------|--------------------------|
| `file`     | `engram.providers.file_adapter:FileAdapter`     | Local markdown/text files |
| `rest`     | `engram.providers.rest_adapter:RestAdapter`     | Generic REST API          |
| `postgres` | `engram.providers.postgres_adapter:PostgresAdapter` | PostgreSQL FTS        |
| `mcp`      | `engram.providers.mcp_adapter:McpAdapter`       | MCP tool bridge           |
