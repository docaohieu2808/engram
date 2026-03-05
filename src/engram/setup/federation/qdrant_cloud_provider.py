"""Federation provider for Qdrant Cloud — managed vector search service."""

from __future__ import annotations

import os

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider


class QdrantCloudProvider(FederationProvider):
    """Connects engram to Qdrant Cloud as a federated recall source."""

    name = "qdrant-cloud"
    display_name = "Qdrant Cloud"

    def detect(self) -> bool:
        return bool(os.environ.get("QDRANT_CLOUD_URL") and os.environ.get("QDRANT_CLOUD_API_KEY"))

    def prompt_config(self) -> ProviderEntry | None:
        url = questionary.text(
            "Qdrant Cloud cluster URL (e.g. https://abc-123.us-east.aws.cloud.qdrant.io):",
            default=os.environ.get("QDRANT_CLOUD_URL", ""),
        ).ask()
        if not url:
            return None

        collection = questionary.text(
            "Qdrant collection name:",
            default=os.environ.get("QDRANT_CLOUD_COLLECTION", "memories"),
        ).ask()
        if not collection:
            return None

        return ProviderEntry(
            name="qdrant-cloud",
            type="rest",
            enabled=True,
            url=url,
            search_endpoint=f"/collections/{collection}/points/query",
            search_method="POST",
            search_body='{"query": "{query}", "limit": 5, "with_payload": true}',
            result_path="result.points",
            headers={"api-key": "${QDRANT_CLOUD_API_KEY}"},
            timeout_seconds=5.0,
        )
