"""Connector for Claude Code (Anthropic) — detects and configures MCP server."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

# Claude Code stores global MCP config here
_CLAUDE_DIR = Path.home() / ".claude"
_MCP_CONFIG_PATH = _CLAUDE_DIR / "mcp.json"


class ClaudeCodeConnector(McpJsonConnector):
    """Connector for Anthropic Claude Code IDE integration."""

    name = "claude-code"
    display_name = "Claude Code"
    tier = 1
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Claude Code via binary or ~/.claude/ directory."""
        binary = shutil.which("claude")
        dir_exists = _CLAUDE_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'claude' binary nor ~/.claude/ found",
            )

        version = _get_claude_version(binary)
        details = binary or str(_CLAUDE_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_MCP_CONFIG_PATH,
            details=details,
        )


def _get_claude_version(binary: str | None) -> str | None:
    """Try to retrieve Claude Code version string via CLI."""
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
_ALL_CONNECTORS.append(ClaudeCodeConnector)
