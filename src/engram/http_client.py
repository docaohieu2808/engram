"""EngramHttpClient — async httpx-based client wrapping the Engram REST API.

Usage:
    async with EngramHttpClient("http://localhost:8000", api_key="secret") as client:
        mem_id = await client.remember("Visited Paris today")
        results = await client.recall("Paris trip")
        answer = await client.think("Where have I been recently?")
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("engram.http_client")


class EngramHttpClient:
    """Async HTTP client for Engram REST API.

    All methods are fail-open: HTTP errors are caught, logged as warnings,
    and empty/None values returned so callers never see exceptions.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        tenant_id: str = "default",
        timeout: float = 30.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {base_url!r}")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._tenant_id = tenant_id
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    #  Context manager                                                     #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "EngramHttpClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._build_headers(),
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    #  Core API methods                                                    #
    # ------------------------------------------------------------------ #

    async def remember(
        self,
        content: str,
        memory_type: str = "fact",
        priority: int = 2,
        tags: list[str] | None = None,
        expires: str | None = None,
    ) -> str | None:
        """Store a memory. Returns memory ID on success, None on failure."""
        payload: dict[str, Any] = {
            "content": content,
            "memory_type": memory_type,
            "priority": priority,
            "tags": tags or [],
        }
        if expires:
            payload["expires_at"] = expires
        try:
            resp = await self._post("/remember", payload)
            return resp.get("id")
        except httpx.HTTPError as exc:
            logger.warning("EngramHttpClient.remember failed: %s", exc)
            return None

    async def recall(
        self,
        query: str,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Search memories by semantic similarity. Returns list of result dicts."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if memory_type:
            params["memory_type"] = memory_type
        try:
            resp = await self._get("/recall", params)
            return resp.get("results", [])
        except httpx.HTTPError as exc:
            logger.warning("EngramHttpClient.recall failed: %s", exc)
            return []

    async def think(self, question: str) -> str:
        """Reason over memories to answer a question. Returns empty string on failure."""
        try:
            resp = await self._post("/think", {"question": question})
            return resp.get("answer", "")
        except httpx.HTTPError as exc:
            logger.warning("EngramHttpClient.think failed: %s", exc)
            return ""

    async def feedback(self, memory_id: str, feedback_type: str) -> dict:
        """Submit feedback (positive/negative) for a memory. Returns result dict."""
        try:
            return await self._post("/feedback", {"memory_id": memory_id, "feedback": feedback_type})
        except httpx.HTTPError as exc:
            logger.warning("EngramHttpClient.feedback failed: %s", exc)
            return {}

    async def graph_query(
        self,
        keyword: str | None = None,
        node_type: str | None = None,
        related_to: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Query semantic graph. Returns list of node dicts."""
        params: dict[str, Any] = {"limit": limit}
        if keyword:
            params["keyword"] = keyword
        if node_type:
            params["node_type"] = node_type
        if related_to:
            params["related_to"] = related_to
        try:
            resp = await self._get("/query", params)
            return resp.get("results", [])
        except httpx.HTTPError as exc:
            logger.warning("EngramHttpClient.graph_query failed: %s", exc)
            return []

    async def health(self) -> bool:
        """Check server health. Returns True if reachable and healthy."""
        try:
            client = self._require_client()
            resp = await client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("EngramHttpClient.health failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_headers(self) -> dict[str, str]:
        """Build request headers — never includes raw api_key in logs."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._tenant_id:
            headers["X-Tenant-ID"] = self._tenant_id
        return headers

    def _require_client(self) -> httpx.AsyncClient:
        """Return the active httpx client or raise RuntimeError."""
        if self._client is None:
            raise RuntimeError(
                "EngramHttpClient must be used as an async context manager "
                "(`async with EngramHttpClient(...) as client:`)"
            )
        return self._client

    async def _post(self, path: str, body: dict) -> dict:
        """POST request. Raises httpx.HTTPError on non-2xx."""
        client = self._require_client()
        resp = await client.post(path, json=body)
        resp.raise_for_status()
        return resp.json()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """GET request. Raises httpx.HTTPError on non-2xx."""
        client = self._require_client()
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()
