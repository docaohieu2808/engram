"""Connector for OpenAI Codex CLI — detects and configures MCP server."""

from __future__ import annotations

import shutil
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

# Codex CLI stores MCP config in ~/.codex/mcp.json
_CODEX_DIR = Path.home() / ".codex"
_MCP_CONFIG_PATH = _CODEX_DIR / "mcp.json"


class CodexConnector(McpJsonConnector):
    """Connector for OpenAI Codex CLI agent.

    Codex CLI uses ~/.codex/mcp.json with standard mcpServers format.
    """

    name = "codex"
    display_name = "OpenAI Codex"
    tier = 1
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Codex CLI via binary or ~/.codex/ directory."""
        binary = shutil.which("codex")
        dir_exists = _CODEX_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'codex' binary nor ~/.codex/ found",
            )

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=None,
            config_path=_MCP_CONFIG_PATH,
            details=binary or str(_CODEX_DIR),
        )


# Register in global connector registry
_ALL_CONNECTORS.append(CodexConnector)
