"""Verification and server management utilities for the engram setup wizard."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.setup.connectors.base import AgentConnector

_DEFAULT_PORT = 8765
_DEFAULT_HOST = "127.0.0.1"


def check_server_running(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> bool:
    """Return True if the engram server is accepting connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def start_server_background(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> bool:
    """Start `engram serve` in background. Returns True if server comes up within 3s."""
    try:
        subprocess.Popen(
            ["engram", "serve", "--host", host, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent process group
        )
    except OSError:
        return False

    # Brief health check — poll up to 3 seconds
    for _ in range(6):
        time.sleep(0.5)
        if check_server_running(host, port):
            return True
    return False


def verify_all(
    connectors: list[AgentConnector],
) -> list[tuple[str, bool, str]]:
    """Verify each configured connector.

    Returns list of (display_name, success, message) tuples.
    """
    results: list[tuple[str, bool, str]] = []
    for connector in connectors:
        try:
            ok = connector.verify()
            if ok:
                msg = "Configuration verified"
            else:
                msg = "Verification failed — config may be missing or invalid"
        except Exception as exc:  # noqa: BLE001
            ok = False
            msg = f"Verification error: {exc}"
        results.append((connector.display_name, ok, msg))
    return results


# Agents that require a restart after MCP config changes
_RESTART_REQUIRED: set[str] = {
    "claude-code",
    "cursor",
    "windsurf",
    "cline",
    "void",
    "zed",
}


def get_restart_hints(connectors: list[AgentConnector]) -> list[str]:
    """Return human-readable restart hints for agents that need it after config."""
    hints: list[str] = []
    for connector in connectors:
        if connector.name in _RESTART_REQUIRED:
            hints.append(f"Restart {connector.display_name} to load the engram MCP server")
    return hints
