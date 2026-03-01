"""Federation provider for Zep — session-based memory service."""

from __future__ import annotations

import os
import socket

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_ZEP_DEFAULT_PORT = 8000
_ZEP_DEFAULT_URL = f"http://localhost:{_ZEP_DEFAULT_PORT}"


def _check_local_zep() -> bool:
    """Return True if something is listening on localhost:8000 (likely Zep)."""
    try:
        with socket.create_connection(("localhost", _ZEP_DEFAULT_PORT), timeout=1):
            return True
    except (OSError, ConnectionRefusedError):
        return False


class ZepProvider(FederationProvider):
    """Connects engram to Zep session memory as a federated recall source."""

    name = "zep"
    display_name = "Zep"

    def detect(self) -> bool:
        """Return True if ZEP_API_URL env var is set or local Zep server detected."""
        return bool(os.environ.get("ZEP_API_URL")) or _check_local_zep()

    def prompt_config(self) -> ProviderEntry | None:
        """Prompt for Zep server URL and optional API key, return a REST ProviderEntry."""
        default_url = os.environ.get("ZEP_API_URL", _ZEP_DEFAULT_URL)
        url = questionary.text(
            "Zep server URL:",
            default=default_url,
        ).ask()
        if not url:
            return None

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        # Optional API key — Zep Cloud requires one, local Zep typically does not
        use_key = questionary.confirm(
            "Does this Zep instance require an API key?",
            default=bool(os.environ.get("ZEP_API_KEY")),
        ).ask()

        headers: dict[str, str] = {}
        if use_key:
            # Always use env var reference — never store raw secret in config
            headers["Authorization"] = "Api-Key ${ZEP_API_KEY}"

        return ProviderEntry(
            name="zep",
            type="rest",
            enabled=True,
            url=url,
            search_endpoint="/api/v2/memories/search",
            search_method="POST",
            search_body='{"text": "{query}", "limit": 5}',
            result_path="results",
            headers=headers,
            timeout_seconds=5.0,
        )
