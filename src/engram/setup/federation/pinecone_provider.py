"""Federation provider for Pinecone — managed vector database."""

from __future__ import annotations

import os

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider


class PineconeProvider(FederationProvider):
    """Connects engram to Pinecone as a federated recall source."""

    name = "pinecone"
    display_name = "Pinecone"

    def detect(self) -> bool:
        return bool(os.environ.get("PINECONE_API_KEY"))

    def prompt_config(self) -> ProviderEntry | None:
        host = questionary.text(
            "Pinecone index host (e.g. my-index-abc123.svc.pinecone.io):",
            default=os.environ.get("PINECONE_INDEX_HOST", ""),
        ).ask()
        if not host:
            return None

        if not host.startswith(("http://", "https://")):
            host = "https://" + host

        return ProviderEntry(
            name="pinecone",
            type="rest",
            enabled=True,
            url=host,
            search_endpoint="/query",
            search_method="POST",
            search_body='{"topK": 5, "includeMetadata": true}',
            result_path="matches",
            headers={"Api-Key": "${PINECONE_API_KEY}"},
            timeout_seconds=5.0,
        )
