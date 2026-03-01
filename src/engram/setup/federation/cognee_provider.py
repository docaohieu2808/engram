"""Federation provider for Cognee — graph-based knowledge memory service."""

from __future__ import annotations

import os

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_COGNEE_DEFAULT_URL = "https://api.cognee.ai"


class CogneeProvider(FederationProvider):
    """Connects engram to Cognee graph memory API as a federated recall source."""

    name = "cognee"
    display_name = "Cognee"

    def detect(self) -> bool:
        """Return True if COGNEE_API_KEY env var is set."""
        return bool(os.environ.get("COGNEE_API_KEY"))

    def prompt_config(self) -> ProviderEntry | None:
        """Prompt for Cognee endpoint and API key, return a REST ProviderEntry."""
        endpoint = questionary.text(
            "Cognee API endpoint URL:",
            default=os.environ.get("COGNEE_API_URL", _COGNEE_DEFAULT_URL),
        ).ask()
        if not endpoint:
            return None

        # Validate minimal URL format
        if not endpoint.startswith(("http://", "https://")):
            endpoint = "https://" + endpoint

        # Always use env var reference for security — never store raw key
        key_ref = "${COGNEE_API_KEY}"

        return ProviderEntry(
            name="cognee",
            type="rest",
            enabled=True,
            url=endpoint,
            search_endpoint="/api/v1/search",
            search_method="POST",
            search_body='{"query": "{query}", "searchType": "INSIGHTS"}',
            result_path="results",
            headers={"Authorization": f"Bearer {key_ref}"},
            timeout_seconds=8.0,
        )
