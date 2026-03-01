"""Connector for Zed editor — detects and configures context_servers MCP entry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import AgentConnector, ConfigureResult, DetectionResult

_ZED_CONFIG_DIR = Path.home() / ".config" / "zed"
_ZED_SETTINGS_PATH = _ZED_CONFIG_DIR / "settings.json"

# Zed uses context_servers (not mcpServers) format
_ENGRAM_CONTEXT_SERVER_ENTRY: dict = {
    "command": {"path": "engram-mcp", "args": []},
}


def _read_zed_settings(path: Path) -> dict:
    """Read existing Zed settings JSON or return empty dict."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_zed_settings(path: Path, config: dict) -> None:
    """Write settings JSON to disk, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


class ZedConnector(AgentConnector):
    """Connector for Zed editor — uses context_servers format for MCP."""

    name = "zed"
    display_name = "Zed"
    tier = 2

    def detect(self) -> DetectionResult:
        """Detect Zed via binary or ~/.config/zed/ directory."""
        binary = shutil.which("zed")
        dir_exists = _ZED_CONFIG_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'zed' binary nor ~/.config/zed/ found",
            )

        version = _get_zed_version(binary)
        details = binary or str(_ZED_CONFIG_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_ZED_SETTINGS_PATH,
            details=details,
        )

    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Merge engram context_servers entry into Zed settings.json."""
        existing = _read_zed_settings(_ZED_SETTINGS_PATH)

        # Idempotent — already configured
        if "engram" in existing.get("context_servers", {}):
            return ConfigureResult(
                success=True,
                message=f"Engram already configured in {_ZED_SETTINGS_PATH}",
                files_modified=[],
                backup_path=None,
            )

        backup: Path | None = None
        if not dry_run and _ZED_SETTINGS_PATH.exists():
            backup = _ZED_SETTINGS_PATH.with_suffix(".bak")
            import shutil as _shutil
            _shutil.copy2(_ZED_SETTINGS_PATH, backup)

        context_servers = existing.setdefault("context_servers", {})
        context_servers["engram"] = _ENGRAM_CONTEXT_SERVER_ENTRY

        if not dry_run:
            _write_zed_settings(_ZED_SETTINGS_PATH, existing)

        action = "Would write" if dry_run else "Wrote"
        return ConfigureResult(
            success=True,
            message=f"{action} engram context_servers entry to {_ZED_SETTINGS_PATH}",
            files_modified=[] if dry_run else [_ZED_SETTINGS_PATH],
            backup_path=backup,
        )

    def verify(self) -> bool:
        """Return True if engram entry exists in Zed context_servers config."""
        settings = _read_zed_settings(_ZED_SETTINGS_PATH)
        return "engram" in settings.get("context_servers", {})


def _get_zed_version(binary: str | None) -> str | None:
    """Try to retrieve Zed version string via CLI."""
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
_ALL_CONNECTORS.append(ZedConnector)
