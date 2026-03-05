"""Federation provider for Weaviate — vector database with GraphQL API."""

from __future__ import annotations

import os
import socket

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_WEAVIATE_DEFAULT_PORT = 8080
_WEAVIATE_DEFAULT_URL = f"http://localhost:{_WEAVIATE_DEFAULT_PORT}"


class WeaviateProvider(FederationProvider):
    """Connects engram to Weaviate as a federated recall source."""

    name = "weaviate"
    display_name = "Weaviate"

    def detect(self) -> bool:
        if os.environ.get("WEAVIATE_URL") or os.environ.get("WEAVIATE_API_KEY"):
            return True
        try:
            with socket.create_connection(("localhost", _WEAVIATE_DEFAULT_PORT), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def prompt_config(self) -> ProviderEntry | None:
        url = questionary.text(
            "Weaviate URL:",
            default=os.environ.get("WEAVIATE_URL", _WEAVIATE_DEFAULT_URL),
        ).ask()
        if not url:
            return None

        collection = questionary.text(
            "Weaviate collection name:",
            default=os.environ.get("WEAVIATE_COLLECTION", "Memory"),
        ).ask()
        if not collection:
            return None

        headers: dict[str, str] = {}
        if os.environ.get("WEAVIATE_API_KEY") or questionary.confirm(
            "Does this Weaviate instance require an API key?", default=False
        ).ask():
            headers["Authorization"] = "Bearer ${WEAVIATE_API_KEY}"

        return ProviderEntry(
            name="weaviate",
            type="rest",
            enabled=True,
            url=url,
            search_endpoint=f"/v1/objects/{collection}/search",
            search_method="POST",
            search_body='{"query": "{query}", "limit": 5}',
            result_path="objects",
            headers=headers,
            timeout_seconds=5.0,
        )
