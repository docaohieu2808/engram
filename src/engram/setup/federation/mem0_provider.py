"""Federation provider for Mem0 — REST-based external memory service."""

from __future__ import annotations

import os
from pathlib import Path

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_MEM0_DIR = Path.home() / ".mem0"
_MEM0_API_URL = "https://api.mem0.ai/v1"


class Mem0Provider(FederationProvider):
    """Connects engram to Mem0 REST API as a federated recall source."""

    name = "mem0"
    display_name = "Mem0"

    def detect(self) -> bool:
        """Return True if MEM0_API_KEY env var is set or ~/.mem0/ exists."""
        return bool(os.environ.get("MEM0_API_KEY")) or _MEM0_DIR.exists()

    def prompt_config(self) -> ProviderEntry | None:
        """Prompt for Mem0 API key and return a REST ProviderEntry."""
        # Prefer env var reference — never store raw key in config
        env_key = os.environ.get("MEM0_API_KEY", "")
        if env_key:
            key_ref = "${MEM0_API_KEY}"
        else:
            key_ref = questionary.password(
                "Mem0 API key (stored as ${MEM0_API_KEY} reference — set env var to use):",
                default="",
            ).ask()
            if not key_ref:
                return None
            # Always use env var reference pattern for security
            key_ref = "${MEM0_API_KEY}"

        return ProviderEntry(
            name="mem0",
            type="rest",
            enabled=True,
            url=_MEM0_API_URL,
            search_endpoint="/memories/search",
            search_method="POST",
            search_body='{"query": "{query}", "limit": 5}',
            result_path="results",
            headers={"Authorization": f"Token {key_ref}"},
            timeout_seconds=5.0,
        )
