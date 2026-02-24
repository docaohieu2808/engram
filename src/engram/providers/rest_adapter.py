"""REST API adapter for external memory services (Cognee, Mem0, LightRAG, etc.)."""

from __future__ import annotations

import json
import logging
import re
import time
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
        auth_login_endpoint: str = "",
        auth_username: str = "",
        auth_password: str = "",
        **kwargs: Any,
    ):
        # M7 fix: validate URL scheme to prevent file://, ftp://, etc.
        url_lower = url.lower()
        if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
            raise ValueError(
                f"REST adapter '{name}': URL must use http:// or https:// scheme, got: {url!r}"
            )
        super().__init__(name=name, provider_type="rest", **kwargs)
        self.url = url.rstrip("/")
        self.search_endpoint = search_endpoint
        self.search_method = search_method.upper()
        self.search_body = search_body
        self.result_path = result_path
        self.headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        # JWT auto-login
        self._auth_login_endpoint = auth_login_endpoint
        self._auth_username = auth_username
        self._auth_password = auth_password
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def _auth_enabled(self) -> bool:
        return bool(self._auth_login_endpoint and self._auth_username)

    async def _ensure_token(self) -> None:
        """Login and cache JWT token. Refresh 5 min before expiry."""
        if not self._auth_enabled:
            return
        if self._token and time.time() < self._token_expires_at - 300:
            return

        login_url = f"{self.url}{self._auth_login_endpoint}"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                login_url,
                data={
                    "username": self._auth_username,
                    "password": self._auth_password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        self._token = data.get("access_token", "")
        # Decode expiry from JWT payload (base64 middle segment)
        self._token_expires_at = self._parse_jwt_expiry(self._token)
        self.headers["Authorization"] = f"Bearer {self._token}"
        logger.info("[%s] JWT token acquired, expires at %s", self.name, self._token_expires_at)

    @staticmethod
    def _parse_jwt_expiry(token: str) -> float:
        """Extract exp claim from JWT without external libs."""
        import base64

        try:
            payload = token.split(".")[1]
            # Add padding
            payload += "=" * (4 - len(payload) % 4)
            decoded = json.loads(base64.urlsafe_b64decode(payload))
            return float(decoded.get("exp", 0))
        except Exception:
            # Fallback: assume 1 hour from now
            return time.time() + 3600

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        await self._ensure_token()
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
            await self._ensure_token()
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.url}/health", headers=self.headers) as resp:
                    return resp.status < 500
        except Exception:
            return False
