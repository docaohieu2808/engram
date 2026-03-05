"""Federation provider for LlamaCloud — hosted RAG/index service by LlamaIndex."""

from __future__ import annotations

import os

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_LLAMACLOUD_DEFAULT_URL = "https://api.cloud.llamaindex.ai"


class LlamaCloudProvider(FederationProvider):
    """Connects engram to LlamaCloud as a federated recall source."""

    name = "llamacloud"
    display_name = "LlamaCloud (LlamaIndex)"

    def detect(self) -> bool:
        return bool(os.environ.get("LLAMA_CLOUD_API_KEY"))

    def prompt_config(self) -> ProviderEntry | None:
        url = questionary.text(
            "LlamaCloud API URL:",
            default=os.environ.get("LLAMA_CLOUD_API_URL", _LLAMACLOUD_DEFAULT_URL),
        ).ask()
        if not url:
            return None

        index_id = questionary.text(
            "LlamaCloud index/pipeline ID:",
            default=os.environ.get("LLAMA_CLOUD_INDEX_ID", ""),
        ).ask()
        if not index_id:
            return None

        return ProviderEntry(
            name="llamacloud",
            type="rest",
            enabled=True,
            url=url,
            search_endpoint=f"/api/v1/pipelines/{index_id}/retrieve",
            search_method="POST",
            search_body='{"query": "{query}", "dense_similarity_top_k": 5}',
            result_path="retrieval_nodes",
            headers={"Authorization": "Bearer ${LLAMA_CLOUD_API_KEY}"},
            timeout_seconds=10.0,
        )
