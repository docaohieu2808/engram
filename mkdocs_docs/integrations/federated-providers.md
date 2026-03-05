# Federated Providers

Engram federates with external memory and knowledge systems, querying them in parallel alongside its own stores during recall.

## Supported Providers

| Provider | Type | Auto-Discovery |
|----------|------|----------------|
| **mem0** | REST | Port 8080, `/v1/memories` |
| **LightRAG** | REST | Port 9520, `/query` |
| **Graphiti** | REST | Port 8000, `/search` |
| **OpenClaw** | File | `~/.openclaw/workspace/memory/*.md` |
| **Custom REST** | REST | Manual config |
| **PostgreSQL** | SQL | Manual config |
| **MCP servers** | MCP | Scans `~/.claude/settings.json`, `~/.cursor/settings.json` |

## Auto-Discovery

Engram scans for providers automatically when `discovery.local: true` (default):

```mermaid
flowchart TD
    AD["Auto-Discovery"] --> T1["Tier 1: Local port scan\n(8000, 8080, 9520, ...)"]
    AD --> T2["Tier 2: Remote hosts\n(config: discovery.hosts)"]
    AD --> T3["Tier 3: Direct endpoints\n(known provider paths)"]
    AD --> T4["Tier 4: MCP config files\n(~/.claude/, ~/.cursor/)"]
    T1 & T2 & T3 & T4 --> Reg["Provider Registry"]
```

Run discovery manually:

```bash
engram discover
```

## Manual Provider Config

```yaml
# ~/.engram/config.yaml

# Enable auto-discovery
discovery:
  local: true
  hosts: ["10.10.0.2"]   # Additional hosts to scan

# Or define providers explicitly
providers:
  - name: my-mem0
    type: rest
    url: http://localhost:8080
    search_endpoint: /v1/memories/search
    search_method: POST
    search_body: '{"query": "{query}", "limit": {limit}}'
    result_path: "results[].memory"

  - name: my-postgres
    type: postgresql
    dsn: postgresql://user:pass@localhost:5432/knowledge
    query: "SELECT content FROM memories WHERE content ILIKE '%{query}%' LIMIT {limit}"

  - name: my-files
    type: file
    glob: "~/notes/**/*.md"
```

## Query Routing

The smart query router classifies each query before fanning out:

| Classification | Behavior |
|---------------|----------|
| `internal` | Queries only engram's own episodic + semantic stores |
| `domain` | Fan-out to all active federated providers in parallel |

## Provider Adapters

| Adapter | Description |
|---------|-------------|
| `RestAdapter` | HTTP REST with optional JWT auto-login |
| `FileAdapter` | Glob patterns over local markdown/text files |
| `PostgresAdapter` | Custom SQL queries against external PG tables |
| `McpAdapter` | Spawns MCP server subprocesses (stdio) |

## Circuit Breaker

Providers auto-disable after consecutive errors and can be re-enabled manually:

```bash
engram providers list        # Show provider status
engram providers enable mem0 # Re-enable a disabled provider
```

## Plugin System

Third-party adapters can be installed via Python entry points:

```toml
# pyproject.toml
[project.entry-points."engram.providers"]
my-adapter = "my_package.adapter:MyAdapter"
```

The provider registry loads all registered adapters automatically.
