"""Connector for Windsurf (Codeium) IDE — detects and configures MCP server."""

from __future__ import annotations

from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

_CODEIUM_DIR = Path.home() / ".codeium"
_WINDSURF_DIR = _CODEIUM_DIR / "windsurf"
_MCP_CONFIG_PATH = _WINDSURF_DIR / "mcp_config.json"


class WindsurfConnector(McpJsonConnector):
    """Connector for Windsurf (by Codeium) AI-powered IDE."""

    name = "windsurf"
    display_name = "Windsurf"
    tier = 1
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Windsurf via ~/.codeium/windsurf/ or ~/.codeium/ directory."""
        windsurf_exists = _WINDSURF_DIR.exists()
        codeium_exists = _CODEIUM_DIR.exists()

        if not windsurf_exists and not codeium_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither ~/.codeium/windsurf/ nor ~/.codeium/ found",
            )

        # Prefer windsurf-specific dir; fall back to codeium root
        details = str(_WINDSURF_DIR if windsurf_exists else _CODEIUM_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=None,  # No CLI version command available
            config_path=_MCP_CONFIG_PATH,
            details=details,
        )


# Register in global connector registry
_ALL_CONNECTORS.append(WindsurfConnector)
