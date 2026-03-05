"""Federation provider for ChromaDB — open-source embedding database."""

from __future__ import annotations

import os
import socket

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_CHROMA_DEFAULT_PORT = 8000
_CHROMA_DEFAULT_URL = f"http://localhost:{_CHROMA_DEFAULT_PORT}"


class ChromaProvider(FederationProvider):
    """Connects engram to ChromaDB server as a federated recall source."""

    name = "chroma"
    display_name = "ChromaDB"

    def detect(self) -> bool:
        if os.environ.get("CHROMA_URL"):
            return True
        try:
            with socket.create_connection(("localhost", _CHROMA_DEFAULT_PORT), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def prompt_config(self) -> ProviderEntry | None:
        url = questionary.text(
            "ChromaDB server URL:",
            default=os.environ.get("CHROMA_URL", _CHROMA_DEFAULT_URL),
        ).ask()
        if not url:
            return None

        collection = questionary.text(
            "ChromaDB collection name:",
            default=os.environ.get("CHROMA_COLLECTION", "memories"),
        ).ask()
        if not collection:
            return None

        return ProviderEntry(
            name="chroma",
            type="rest",
            enabled=True,
            url=url,
            search_endpoint=f"/api/v2/collections/{collection}/query",
            search_method="POST",
            search_body='{"query_texts": ["{query}"], "n_results": 5}',
            result_path="documents",
            headers={},
            timeout_seconds=5.0,
        )
