"""Connector for Cursor IDE — detects and configures MCP server."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

_CURSOR_DIR = Path.home() / ".cursor"
_MCP_CONFIG_PATH = _CURSOR_DIR / "mcp.json"


class CursorConnector(McpJsonConnector):
    """Connector for Cursor AI-powered IDE."""

    name = "cursor"
    display_name = "Cursor"
    tier = 1
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Cursor via binary or ~/.cursor/ directory."""
        binary = shutil.which("cursor")
        dir_exists = _CURSOR_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'cursor' binary nor ~/.cursor/ found",
            )

        version = _get_cursor_version(binary)
        details = binary or str(_CURSOR_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_MCP_CONFIG_PATH,
            details=details,
        )


def _get_cursor_version(binary: str | None) -> str | None:
    """Try to retrieve Cursor version string via CLI."""
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
_ALL_CONNECTORS.append(CursorConnector)
