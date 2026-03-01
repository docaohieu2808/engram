"""Shared MCP JSON config merge logic for agents using mcpServers format.

Used by Claude Code, Cursor, and Windsurf connectors.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from engram.setup.connectors.base import AgentConnector, ConfigureResult

# Engram MCP server entry to inject into any mcpServers config
ENGRAM_MCP_ENTRY: dict = {
    "command": "engram-mcp",
    "args": [],
    "env": {},
}


def _read_mcp_config(path: Path) -> dict:
    """Read existing MCP JSON config or return empty structure."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable file — start fresh but preserve original via backup
            return {}
    return {}


def _merge_engram_entry(config: dict) -> dict:
    """Add engram entry to mcpServers in config dict. Idempotent."""
    mcp_servers = config.setdefault("mcpServers", {})
    mcp_servers["engram"] = ENGRAM_MCP_ENTRY
    return config


def _backup_config(path: Path) -> Path | None:
    """Create a .bak copy of the config file before modification. Returns backup path."""
    if not path.exists():
        return None
    backup = path.with_suffix(".bak")
    shutil.copy2(path, backup)
    return backup


def _write_mcp_config(path: Path, config: dict, dry_run: bool) -> None:
    """Write merged config back to disk (creates parent dirs if needed)."""
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _verify_mcp_config(path: Path) -> bool:
    """Return True if engram entry exists in mcpServers of config file."""
    config = _read_mcp_config(path)
    return "engram" in config.get("mcpServers", {})


class McpJsonConnector(AgentConnector):
    """Base class for agents that store MCP servers in a JSON mcpServers config.

    Subclasses must set:
        config_path: Path  — absolute path to the agent's MCP JSON config file
    """

    config_path: Path

    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Merge engram MCP entry into agent's mcpServers JSON config."""
        path = self.config_path
        existing = _read_mcp_config(path)

        # Already configured — idempotent
        if "engram" in existing.get("mcpServers", {}):
            return ConfigureResult(
                success=True,
                message=f"Engram already configured in {path}",
                files_modified=[],
                backup_path=None,
            )

        backup = None
        if not dry_run:
            backup = _backup_config(path)

        merged = _merge_engram_entry(existing)
        _write_mcp_config(path, merged, dry_run)

        action = "Would write" if dry_run else "Wrote"
        return ConfigureResult(
            success=True,
            message=f"{action} engram MCP entry to {path}",
            files_modified=[] if dry_run else [path],
            backup_path=backup,
        )

    def verify(self) -> bool:
        """Return True if engram entry exists in agent's MCP config."""
        return _verify_mcp_config(self.config_path)
