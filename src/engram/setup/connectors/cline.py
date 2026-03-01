"""Connector for Cline (VSCode extension) — detects and configures MCP server."""

from __future__ import annotations

import shutil
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

# Cline stores MCP settings in ~/.cline/ or VSCode extension user data dir
_CLINE_DIR = Path.home() / ".cline"
_MCP_CONFIG_PATH = _CLINE_DIR / "mcp_settings.json"

# VSCode extensions directory — look for any cline extension folder
_VSCODE_EXTENSIONS_DIR = Path.home() / ".vscode" / "extensions"


def _find_cline_extension() -> Path | None:
    """Return path to cline extension dir if found in ~/.vscode/extensions/."""
    if not _VSCODE_EXTENSIONS_DIR.exists():
        return None
    for entry in _VSCODE_EXTENSIONS_DIR.iterdir():
        if entry.is_dir() and "cline" in entry.name.lower():
            return entry
    return None


class ClineConnector(McpJsonConnector):
    """Connector for Cline VSCode extension (saoudrizwan.claude-dev)."""

    name = "cline"
    display_name = "Cline"
    tier = 2
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Cline via ~/.cline/ directory or VSCode extensions."""
        dir_exists = _CLINE_DIR.exists()
        extension_path = _find_cline_extension()

        if not dir_exists and not extension_path:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither ~/.cline/ nor VSCode cline extension found",
            )

        details = str(_CLINE_DIR) if dir_exists else str(extension_path)
        # Extract version from extension dir name (e.g. saoudrizwan.claude-dev-3.2.1)
        version: str | None = None
        if extension_path:
            parts = extension_path.name.rsplit("-", 1)
            if len(parts) == 2 and parts[1][0].isdigit():
                version = parts[1]

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_MCP_CONFIG_PATH,
            details=details,
        )


# Register in global connector registry
_ALL_CONNECTORS.append(ClineConnector)
