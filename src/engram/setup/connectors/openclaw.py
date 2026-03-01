"""Connector for OpenClaw AI agent — detects and installs engram SKILL.md."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import AgentConnector, ConfigureResult, DetectionResult

_OPENCLAW_DIR = Path.home() / ".openclaw"
_SKILL_DIR = _OPENCLAW_DIR / "workspace" / "skills" / "engram"
_SKILL_PATH = _SKILL_DIR / "SKILL.md"

# SKILL.md content matching OpenClaw's skill frontmatter format
_SKILL_CONTENT = """\
---
name: engram
description: Cross-agent shared memory system. Remember, recall, and reason across sessions. Shared with Claude Code.
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "requires": { "bins": ["engram"], "env": [] }
      }
  }
---

# Engram — Cross-agent Memory

Shared memory across all AI agents (Claude Code, OpenClaw). Use engram to remember, recall, and reason across sessions.

## Commands

### Remember something
```bash
engram remember "information to store" --tags tag1,tag2
```

### Recall memories
```bash
engram recall "search query" --top-k 5
```

### Think (combined reasoning across memories)
```bash
engram think "question or topic"
```

### Smart query (auto-routes to recall or think)
```bash
engram ask "any question about past context"
```

### Ingest context (extract entities + remember)
```bash
engram ingest "path/to/file.md"
```

## MCP Server

Engram exposes an MCP server for tool-based access:

```
engram-mcp
```

## Tools (via MCP)
- **remember** — Store a memory with content and optional tags
- **recall** — Search memories by query with semantic + keyword matching
- **think** — Reason over memories to answer a question
- **forget** — Remove a specific memory by ID
- **ask** — Smart query that auto-routes to recall or think

## When to use

- **Always recall** at the start of a new task to check for relevant past context
- **Always remember** important decisions, outcomes, and lessons learned
- **Use think** for complex questions that need reasoning across multiple memories
- **Use ask** when unsure whether to recall or think — it auto-routes

## Configuration

Server: `engram serve` (default port 8765)
Config: `~/.engram/config.yaml`

## Important

- Engram is shared with Claude Code — memories stored here are visible to both agents
- Use meaningful tags for better retrieval
- Keep memories concise but information-rich
"""


class OpenClawConnector(AgentConnector):
    """Connector for OpenClaw AI agent — installs engram as a skill."""

    name = "openclaw"
    display_name = "OpenClaw"
    tier = 1

    def detect(self) -> DetectionResult:
        """Detect OpenClaw via binary or ~/.openclaw/ directory."""
        binary = shutil.which("openclaw")
        dir_exists = _OPENCLAW_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'openclaw' binary nor ~/.openclaw/ found",
            )

        version = _get_openclaw_version(binary)
        details = binary or str(_OPENCLAW_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_SKILL_PATH,
            details=details,
        )

    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Create ~/.openclaw/workspace/skills/engram/SKILL.md."""
        # Already configured — idempotent
        if _SKILL_PATH.exists() and _SKILL_PATH.read_text(encoding="utf-8") == _SKILL_CONTENT:
            return ConfigureResult(
                success=True,
                message=f"Engram skill already installed at {_SKILL_PATH}",
                files_modified=[],
                backup_path=None,
            )

        backup = None
        if not dry_run:
            # Backup existing SKILL.md if present
            if _SKILL_PATH.exists():
                backup = _SKILL_PATH.with_suffix(".bak")
                shutil.copy2(_SKILL_PATH, backup)

            _SKILL_DIR.mkdir(parents=True, exist_ok=True)
            _SKILL_PATH.write_text(_SKILL_CONTENT, encoding="utf-8")

        action = "Would write" if dry_run else "Wrote"
        return ConfigureResult(
            success=True,
            message=f"{action} engram skill to {_SKILL_PATH}",
            files_modified=[] if dry_run else [_SKILL_PATH],
            backup_path=backup,
        )

    def verify(self) -> bool:
        """Return True if engram SKILL.md exists in OpenClaw skills directory."""
        return _SKILL_PATH.exists()


def _get_openclaw_version(binary: str | None) -> str | None:
    """Try to retrieve OpenClaw version string via CLI."""
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout or result.stderr).strip()
        return output or None
    except (OSError, subprocess.TimeoutExpired):
        return None


# Register in global connector registry
_ALL_CONNECTORS.append(OpenClawConnector)
