"""Connector for Aider (AI pair programmer CLI) — detects and configures engram instructions."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import AgentConnector, ConfigureResult, DetectionResult

# Aider reads ~/.aider.conf.yml for persistent CLI options
_AIDER_CONF_YML = Path.home() / ".aider.conf.yml"
_AIDER_CONF_JSON = Path.home() / ".aider.conf.json"

# Instructions file path — written by this connector and referenced by aider
_ENGRAM_INSTRUCTIONS_DIR = Path.home() / ".engram"
_ENGRAM_AIDER_INSTRUCTIONS = _ENGRAM_INSTRUCTIONS_DIR / "aider-instructions.md"

_INSTRUCTIONS_CONTENT = """\
# Engram Memory — Aider Integration

Engram provides persistent cross-session memory for AI agents. Use these commands
during your aider sessions to store and retrieve context.

## Quick Reference

### Store a memory
```bash
engram remember "description of what happened or was decided" --tags aider,project
```

### Retrieve relevant memories
```bash
engram recall "what are the relevant past decisions about this?"
```

### Reason across memories
```bash
engram think "what is the current state of the authentication system?"
```

### Smart query (auto-routes)
```bash
engram ask "any question about past context"
```

## Recommended Workflow

1. **Start of session**: Run `engram recall "project context"` to load relevant memories
2. **After key decisions**: Run `engram remember "we decided to use X because Y"`
3. **End of session**: Run `engram remember "session summary: completed X, next steps Y"`

## Configuration

- Server: `engram serve` (port 8765 by default)
- Config: `~/.engram/config.yaml`
- MCP server binary: `engram-mcp`

## Tags

Use tags to organize memories by project, topic, or agent:
```bash
engram remember "content" --tags aider,backend,auth
```
"""


class AiderConnector(AgentConnector):
    """Connector for Aider CLI — creates engram instructions and adds to aider config."""

    name = "aider"
    display_name = "Aider"
    tier = 2

    def detect(self) -> DetectionResult:
        """Detect Aider via binary or config file."""
        binary = shutil.which("aider")
        conf_exists = _AIDER_CONF_YML.exists() or _AIDER_CONF_JSON.exists()

        if not binary and not conf_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'aider' binary nor ~/.aider.conf.yml found",
            )

        config_path = _AIDER_CONF_YML if _AIDER_CONF_YML.exists() else _AIDER_CONF_JSON
        version = _get_aider_version(binary)
        details = binary or str(config_path)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=config_path if conf_exists else None,
            details=details,
        )

    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Write engram instructions file and add read reference to aider config."""
        files_modified: list[Path] = []

        # Step 1: Write engram aider instructions file
        instructions_existed = _ENGRAM_AIDER_INSTRUCTIONS.exists()
        if not dry_run:
            _ENGRAM_INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
            _ENGRAM_AIDER_INSTRUCTIONS.write_text(_INSTRUCTIONS_CONTENT, encoding="utf-8")
            files_modified.append(_ENGRAM_AIDER_INSTRUCTIONS)

        # Step 2: Add read reference to ~/.aider.conf.yml
        conf_result = _add_read_to_aider_conf(dry_run)
        if conf_result:
            files_modified.append(conf_result)

        action = "Would write" if dry_run else "Wrote"
        msg = (
            f"{action} engram instructions to {_ENGRAM_AIDER_INSTRUCTIONS} "
            f"and registered in ~/.aider.conf.yml"
        )
        return ConfigureResult(
            success=True,
            message=msg,
            files_modified=files_modified,
            backup_path=None,
        )

    def verify(self) -> bool:
        """Return True if engram aider instructions file exists."""
        return _ENGRAM_AIDER_INSTRUCTIONS.exists()


def _get_aider_version(binary: str | None) -> str | None:
    """Try to retrieve Aider version string via CLI."""
    if not binary:
        return None
    import subprocess
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


def _add_read_to_aider_conf(dry_run: bool) -> Path | None:
    """Add engram instructions path to aider's read list in ~/.aider.conf.yml.

    Returns the config path if it was modified, else None.
    """
    instructions_str = str(_ENGRAM_AIDER_INSTRUCTIONS)

    # Load existing config or start fresh
    existing: dict = {}
    if _AIDER_CONF_YML.exists():
        try:
            with open(_AIDER_CONF_YML) as f:
                existing = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            existing = {}

    # Check if already configured
    read_list: list = existing.get("read", []) or []
    if instructions_str in read_list:
        return None  # Already present — idempotent

    if dry_run:
        return None

    read_list.append(instructions_str)
    existing["read"] = read_list

    _AIDER_CONF_YML.parent.mkdir(parents=True, exist_ok=True)
    with open(_AIDER_CONF_YML, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    return _AIDER_CONF_YML


# Register in global connector registry
_ALL_CONNECTORS.append(AiderConnector)
