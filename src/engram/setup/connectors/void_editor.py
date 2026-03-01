"""Connector for Void editor — VSCode-compatible, detects and configures MCP server."""

from __future__ import annotations

import shutil
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

# Void uses a VSCode-compatible config layout under ~/.void/
_VOID_DIR = Path.home() / ".void"
_MCP_CONFIG_PATH = _VOID_DIR / "mcp.json"


class VoidEditorConnector(McpJsonConnector):
    """Connector for Void editor (VSCode-compatible fork with AI features)."""

    name = "void"
    display_name = "Void"
    tier = 2
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Void via binary or ~/.void/ directory."""
        binary = shutil.which("void")
        dir_exists = _VOID_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'void' binary nor ~/.void/ found",
            )

        version = _get_void_version(binary)
        details = binary or str(_VOID_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_MCP_CONFIG_PATH,
            details=details,
        )


def _get_void_version(binary: str | None) -> str | None:
    """Try to retrieve Void version string via CLI."""
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


# Register in global connector registry
_ALL_CONNECTORS.append(VoidEditorConnector)
