"""Connector for GitHub Copilot (VS Code) — detects and configures MCP server."""

from __future__ import annotations

import shutil
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

# VS Code stores MCP config in ~/.vscode/mcp.json (global) or .vscode/mcp.json (workspace)
_VSCODE_DIR = Path.home() / ".vscode"
_MCP_CONFIG_PATH = _VSCODE_DIR / "mcp.json"


class GitHubCopilotConnector(McpJsonConnector):
    """Connector for GitHub Copilot in VS Code.

    GitHub Copilot uses VS Code's MCP config at ~/.vscode/mcp.json
    with standard mcpServers format.
    """

    name = "github-copilot"
    display_name = "GitHub Copilot"
    tier = 1
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect VS Code + GitHub Copilot extension."""
        binary = shutil.which("code")
        dir_exists = _VSCODE_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'code' binary nor ~/.vscode/ found",
            )

        # Check if Copilot extension is installed
        copilot_installed = _has_copilot_extension(binary)

        if not copilot_installed:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="VS Code found but GitHub Copilot extension not detected",
            )

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=None,
            config_path=_MCP_CONFIG_PATH,
            details=binary or str(_VSCODE_DIR),
        )


def _has_copilot_extension(binary: str | None) -> bool:
    """Check if GitHub Copilot extension is installed in VS Code."""
    # Check extensions directory for copilot
    extensions_dir = Path.home() / ".vscode" / "extensions"
    if extensions_dir.exists():
        for ext in extensions_dir.iterdir():
            if ext.name.startswith("github.copilot-"):
                return True
    # Fallback: try CLI
    if binary:
        import subprocess
        try:
            result = subprocess.run(
                [binary, "--list-extensions"],
                capture_output=True, text=True, timeout=10,
            )
            return "github.copilot" in (result.stdout or "").lower()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False


# Register in global connector registry
_ALL_CONNECTORS.append(GitHubCopilotConnector)
