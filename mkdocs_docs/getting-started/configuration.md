# Configuration

**Config file:** `~/.engram/config.yaml`

**Priority order:** CLI flags > environment variables > YAML > built-in defaults

## Full Reference

```yaml
episodic:
  mode: embedded              # embedded (Qdrant in-process) or server
  path: ~/.engram/qdrant
  namespace: default

embedding:
  provider: gemini
  model: gemini-embedding-001
  key_strategy: failover      # failover or round-robin

semantic:
  provider: sqlite            # or postgresql
  path: ~/.engram/semantic.db

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}  # env var interpolation supported

serve:
  host: 127.0.0.1
  port: 8765

capture:
  openclaw:
    enabled: false
    sessions_dir: ~/.openclaw/workspace/sessions
  claude_code:
    enabled: false
    sessions_dir: ~/.claude/projects

auth:
  enabled: false              # Enable JWT + RBAC

cache:
  enabled: false
  redis_url: redis://localhost:6379/0

rate_limit:
  enabled: false

audit:
  enabled: false
  path: ~/.engram/audit.jsonl

discovery:
  local: true
  hosts: []                   # Additional hosts to scan for federated providers
```

## Key Settings Explained

### Episodic Mode

| Mode | Description | When to Use |
|------|-------------|-------------|
| `embedded` | Qdrant runs in-process | Development, single-node |
| `server` | External Qdrant server | Production, shared storage |

### Embedding Key Strategy

| Strategy | Description |
|----------|-------------|
| `failover` | Use primary key, fall back to `GEMINI_API_KEY_FALLBACK` on failure |
| `round-robin` | Rotate between keys for load balancing |

### Semantic Backend

| Provider | Description |
|----------|-------------|
| `sqlite` | Default, zero-config, file-based |
| `postgresql` | Production-grade, multi-tenant row isolation |

## CLI Config Commands

```bash
# Show all current settings
engram config show

# Get a specific value
engram config get llm.model

# Set a value
engram config set serve.port 9000

# Live settings editor (Web UI)
# Navigate to http://localhost:8765 > Settings tab
```

## Environment Variables

All config values can be set via environment variables. See the [Environment Variables reference](../reference/environment-variables.md) for the full list.

Common overrides:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_API_KEY_FALLBACK="backup-key"
export ENGRAM_NAMESPACE="my-project"
export ENGRAM_AUTH_ENABLED="true"
export ENGRAM_SEMANTIC_PROVIDER="postgresql"
export ENGRAM_SEMANTIC_DSN="postgresql://user:pass@localhost:5432/engram"
```
