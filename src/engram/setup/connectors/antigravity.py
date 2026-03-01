"""Connector for Antigravity AI proxy — detection only, documents manual setup steps."""

from __future__ import annotations

import shutil
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import AgentConnector, ConfigureResult, DetectionResult

_ANTIGRAVITY_DIR = Path.home() / ".antigravity"

# Manual setup guide written to ~/.engram/ for user reference
_ENGRAM_DIR = Path.home() / ".engram"
_SETUP_GUIDE_PATH = _ENGRAM_DIR / "antigravity-setup.md"

_SETUP_GUIDE_CONTENT = """\
# Engram + Antigravity Integration Guide

Antigravity is an AI proxy that routes requests between different model providers.
Because Antigravity's proxy config varies per deployment, automatic configuration
is not supported. Follow the steps below to integrate engram manually.

## Option 1: MCP via Antigravity proxy

If your Antigravity deployment supports MCP server proxying, add the following
to your Antigravity route config (exact format depends on your version):

```json
{
  "routes": {
    "engram-memory": {
      "type": "mcp",
      "command": "engram-mcp",
      "args": []
    }
  }
}
```

## Option 2: Use engram CLI directly

Engram works independently of Antigravity. Use it alongside any AI agent:

```bash
# Remember information
engram remember "key insight" --tags project,session

# Recall context before starting a task
engram recall "relevant topic"

# Reason across memories
engram think "question about past work"
```

## Option 3: Enable engram MCP server

Start the engram MCP server and point your MCP client to it:

```bash
engram serve   # starts on port 8765 by default
```

Then configure your Antigravity proxy to forward MCP requests to:
  http://127.0.0.1:8765

## Configuration

- Engram config: `~/.engram/config.yaml`
- Engram MCP binary: `engram-mcp`
- Server default: `http://127.0.0.1:8765`

## Support

For more information visit: https://github.com/your-org/engram
"""


class AntigravityConnector(AgentConnector):
    """Connector for Antigravity AI proxy — detection + manual setup guide only."""

    name = "antigravity"
    display_name = "Antigravity"
    tier = 2

    def detect(self) -> DetectionResult:
        """Detect Antigravity via binary or ~/.antigravity/ directory."""
        binary = shutil.which("antigravity")
        dir_exists = _ANTIGRAVITY_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'antigravity' binary nor ~/.antigravity/ found",
            )

        details = binary or str(_ANTIGRAVITY_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=None,
            config_path=_SETUP_GUIDE_PATH,
            details=details,
        )

    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Write manual setup guide — automatic proxy config not supported."""
        if not dry_run:
            _ENGRAM_DIR.mkdir(parents=True, exist_ok=True)
            _SETUP_GUIDE_PATH.write_text(_SETUP_GUIDE_CONTENT, encoding="utf-8")

        action = "Would write" if dry_run else "Wrote"
        return ConfigureResult(
            success=True,
            message=(
                f"{action} manual setup guide to {_SETUP_GUIDE_PATH}. "
                "Antigravity requires manual proxy config — see guide for details."
            ),
            files_modified=[] if dry_run else [_SETUP_GUIDE_PATH],
            backup_path=None,
        )

    def verify(self) -> bool:
        """Return True if the setup guide has been written."""
        return _SETUP_GUIDE_PATH.exists()


# Register in global connector registry
_ALL_CONNECTORS.append(AntigravityConnector)
