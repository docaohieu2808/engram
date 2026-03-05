"""Federation provider for Milvus/Zilliz — distributed vector database."""

from __future__ import annotations

import os
import socket

import questionary

from engram.config import ProviderEntry
from engram.setup.federation.base import FederationProvider

_MILVUS_DEFAULT_PORT = 19530
_ZILLIZ_CLOUD_URL = "https://controller.api.cloud.zilliz.com"


class MilvusProvider(FederationProvider):
    """Connects engram to Milvus or Zilliz Cloud as a federated recall source."""

    name = "milvus"
    display_name = "Milvus / Zilliz"

    def detect(self) -> bool:
        if os.environ.get("ZILLIZ_CLOUD_URL") or os.environ.get("MILVUS_URL"):
            return True
        try:
            with socket.create_connection(("localhost", _MILVUS_DEFAULT_PORT), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def prompt_config(self) -> ProviderEntry | None:
        is_cloud = questionary.confirm(
            "Are you using Zilliz Cloud (managed Milvus)?", default=False
        ).ask()

        if is_cloud:
            url = questionary.text(
                "Zilliz Cloud endpoint URL:",
                default=os.environ.get("ZILLIZ_CLOUD_URL", _ZILLIZ_CLOUD_URL),
            ).ask()
            headers = {"Authorization": "Bearer ${ZILLIZ_API_KEY}"}
        else:
            url = questionary.text(
                "Milvus REST URL:",
                default=os.environ.get("MILVUS_URL", "http://localhost:19530"),
            ).ask()
            headers = {}

        if not url:
            return None

        collection = questionary.text(
            "Collection name:",
            default=os.environ.get("MILVUS_COLLECTION", "memories"),
        ).ask()
        if not collection:
            return None

        return ProviderEntry(
            name="milvus",
            type="rest",
            enabled=True,
            url=url,
            search_endpoint="/v2/vectordb/entities/search",
            search_method="POST",
            search_body='{"collectionName": "' + collection + '", "data": ["{query}"], "limit": 5, "outputFields": ["content", "metadata"]}',
            result_path="data",
            headers=headers,
            timeout_seconds=5.0,
        )
