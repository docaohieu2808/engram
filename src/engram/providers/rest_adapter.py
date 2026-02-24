"""REST API adapter for external memory services (Cognee, Mem0, LightRAG, etc.)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import aiohttp

from engram.providers.base import MemoryProvider, ProviderResult

logger = logging.getLogger("engram.providers.rest")


def _extract_by_path(data: Any, path: str) -> list[str]:
    """Extract values from nested data using dot-bracket path (e.g. 'data[].text').

    Supports:
      - 'key' — direct dict access
      - 'key[]' — iterate over list
      - 'key[].sub' — iterate + access sub-key
    """
    if not path:
        if isinstance(data, str):
            return [data]
        if isinstance(data, list):
            return [str(item) for item in data]
        return [str(data)]

    parts = re.split(r"\.(?![^\[]*\])", path)  # split on dots not inside brackets
    current: list[Any] = [data]

    for part in parts:
        next_items: list[Any] = []
        is_array = part.endswith("[]")
        key = part.rstrip("[]") if is_array else part

        for item in current:
            if key and isinstance(item, dict):
                val = item.get(key)
            elif key and isinstance(item, list):
                # Try numeric index
                try:
                    val = item[int(key)]
                except (ValueError, IndexError):
                    val = None
            else:
                val = item

            if val is None:
                continue

            if is_array and isinstance(val, list):
                next_items.extend(val)
            else:
                next_items.append(val)

        current = next_items

    return [str(item) for item in current if item is not None]


class RestAdapter(MemoryProvider):
    """Connects to any REST API that supports search-style queries."""

    def __init__(
        self,
        name: str,
        url: str,
        search_endpoint: str,
        search_method: str = "POST",
        search_body: str = "",
        result_path: str = "",
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 3.0,
        **kwargs: Any,
    ):
        super().__init__(name=name, provider_type="rest", **kwargs)
        self.url = url.rstrip("/")
        self.search_endpoint = search_endpoint
        self.search_method = search_method.upper()
        self.search_body = search_body
        self.result_path = result_path
        self.headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        search_url = f"{self.url}{self.search_endpoint}"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            if self.search_method == "GET":
                params = {"query": query, "limit": str(limit)}
                async with session.get(search_url, params=params, headers=self.headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            else:
                # Substitute placeholders; escape query for JSON safety
                safe_query = json.dumps(query)[1:-1]  # strip outer quotes, keeps escapes
                body_str = self.search_body.replace("{query}", safe_query).replace("{limit}", str(limit))
                body = json.loads(body_str) if body_str else {"query": query, "limit": limit}
                async with session.post(search_url, json=body, headers=self.headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

        if self.debug:
            logger.debug("[%s] response: %s", self.name, json.dumps(data, default=str)[:500])

        # Extract results using result_path
        texts = _extract_by_path(data, self.result_path)
        results = []
        for i, text in enumerate(texts[:limit]):
            results.append(ProviderResult(
                content=text,
                score=1.0 - (i * 0.1),  # rank-based score
                source=self.name,
            ))
        return results

    async def health(self) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.url}/health", headers=self.headers) as resp:
                    return resp.status < 500
        except Exception:
            return False
