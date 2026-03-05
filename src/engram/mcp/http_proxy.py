"""HTTP proxy for MCP server when embedded Qdrant is locked by HTTP server.

Forwards episodic and semantic operations to the running engram HTTP API,
allowing MCP (Claude Code) and HTTP server (OpenClaw) to coexist.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("engram")


class HttpEpisodicProxy:
    """Proxy EpisodicStore operations through engram HTTP API."""

    def __init__(self, cfg):
        self._base = f"http://{cfg.serve.host}:{cfg.serve.port}/api/v1"
        self._namespace = cfg.episodic.namespace or "default"

    async def remember(self, content: str, **kwargs) -> str:
        import httpx
        payload: dict[str, Any] = {"content": content}
        if kwargs.get("tags"):
            payload["tags"] = kwargs["tags"]
        mt = kwargs.get("memory_type")
        if mt:
            payload["type"] = mt.value if hasattr(mt, "value") else str(mt)
        if kwargs.get("priority"):
            payload["priority"] = kwargs["priority"]
        if kwargs.get("topic_key"):
            payload["topic_key"] = kwargs["topic_key"]
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._base}/remember", json=payload)
            resp.raise_for_status()
            return resp.json().get("id", "")

    async def search(self, query: str, limit: int = 5, **kwargs) -> list:
        import httpx
        params: dict[str, Any] = {"query": query, "top_k": limit}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base}/recall", params=params)
            resp.raise_for_status()
            data = resp.json()
        memories = []
        for m in data.get("results", data.get("memories", [])):
            mem = SimpleNamespace(
                id=m.get("id", ""),
                content=m.get("content", ""),
                memory_type=m.get("type", "fact"),
                priority=m.get("priority", 5),
                tags=m.get("tags", []),
                timestamp=m.get("timestamp", ""),
                access_count=m.get("access_count", 0),
                score=m.get("score", 0.0),
                decay_rate=m.get("decay_rate", 0.1),
            )
            memories.append(mem)
        return memories

    async def get_by_id(self, memory_id: str) -> Any:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base}/memories/{memory_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            m = resp.json()
            return SimpleNamespace(**m) if isinstance(m, dict) else m

    async def delete(self, memory_id: str) -> bool:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(f"{self._base}/memories/{memory_id}")
            return resp.status_code == 200

    async def stats(self) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base}/status")
            return resp.json().get("episodic", {"count": 0})

    async def get_recent(self, n: int = 10) -> list:
        return await self.search("", limit=n)

    async def cleanup_expired(self) -> int:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._base}/cleanup")
            return resp.json().get("deleted", 0)


class HttpGraphProxy:
    """Proxy SemanticGraph operations through engram HTTP API."""

    def __init__(self, cfg):
        self._base = f"http://{cfg.serve.host}:{cfg.serve.port}/api/v1"

    async def query(self, query: str, **kwargs) -> list:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self._base}/graph/query", params={"q": query})
            if resp.status_code != 200:
                return []
            return [SimpleNamespace(**n) for n in resp.json().get("nodes", [])]

    async def add_node(self, name: str, node_type: str, **kwargs) -> None:
        import httpx
        payload = {"name": name, "type": node_type, **kwargs}
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{self._base}/graph/nodes", json=payload)

    async def add_edge(self, from_node: str, to_node: str, relation: str, **kwargs) -> None:
        import httpx
        payload = {"from_node": from_node, "to_node": to_node, "relation": relation, **kwargs}
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{self._base}/graph/edges", json=payload)

    async def stats(self) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base}/status")
            return resp.json().get("semantic", {"node_count": 0, "edge_count": 0})

    async def dump(self) -> dict:
        return await self.stats()

    async def get_related(self, names: list[str]) -> dict:
        return {n: {"edges": []} for n in names}


class HttpEngineProxy:
    """Proxy ReasoningEngine operations through engram HTTP API."""

    def __init__(self, cfg):
        self._base = f"http://{cfg.serve.host}:{cfg.serve.port}/api/v1"

    async def think(self, query: str, **kwargs) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self._base}/think", json={"query": query})
            resp.raise_for_status()
            return resp.json()

    async def summarize(self, n: int = 20, save: bool = False) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self._base}/summarize", json={"count": n, "save": save})
            resp.raise_for_status()
            return resp.json().get("summary", "")
