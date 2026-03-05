"""HTTP proxy for MCP server when embedded Qdrant is locked by HTTP server.

Forwards episodic and semantic operations to the running engram HTTP API,
allowing MCP (Claude Code) and HTTP server (OpenClaw) to coexist.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("engram")


def _build_memory_namespace(m: dict, ts) -> SimpleNamespace:
    """Build a SimpleNamespace with proper types for MCP episodic_tools."""
    from engram.models import MemoryType
    raw_type = m.get("memory_type", m.get("type", "fact"))
    try:
        mt = MemoryType(raw_type)
    except ValueError:
        mt = MemoryType.FACT
    return SimpleNamespace(
        id=m.get("id", ""),
        content=m.get("content", ""),
        memory_type=mt,
        priority=m.get("priority", 5),
        tags=m.get("tags", []),
        entities=m.get("entities", []),
        timestamp=ts,
        access_count=m.get("access_count", 0),
        score=m.get("score", 0.0),
        decay_rate=m.get("decay_rate", 0.1),
    )


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
        from datetime import datetime, timezone
        params: dict[str, Any] = {"query": query, "limit": limit}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base}/recall", params=params)
            resp.raise_for_status()
            data = resp.json()
        memories = []
        for m in data.get("results", data.get("memories", [])):
            # Parse timestamp string to datetime object
            ts = m.get("timestamp", "")
            if isinstance(ts, str) and ts:
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
            elif not ts:
                ts = datetime.now(timezone.utc)
            mem = _build_memory_namespace(m, ts)
            memories.append(mem)
        return memories

    async def get_by_id(self, memory_id: str) -> Any:
        import httpx
        from datetime import datetime, timezone
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base}/memories/{memory_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            m = resp.json()
            if not isinstance(m, dict):
                return m
            ts = m.get("timestamp", "")
            if isinstance(ts, str) and ts:
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
            return _build_memory_namespace(m, ts)

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

    async def get_recent(self, n: int = 10, **kwargs) -> list:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base}/memories", params={"limit": min(n, 100), "offset": 0})
            data = resp.json()
        results = []
        for m in data.get("memories", []):
            ts = m.get("timestamp", "")
            if isinstance(ts, str) and ts:
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
            elif not ts:
                ts = datetime.now(timezone.utc)
            results.append(_build_memory_namespace(m, ts))
        return results

    async def cleanup_expired(self) -> int:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._base}/cleanup")
            return resp.json().get("deleted", 0)

    async def cleanup_dedup(self, threshold: float = 0.85, dry_run: bool = False) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base}/cleanup/dedup",
                json={"threshold": threshold, "dry_run": dry_run},
            )
            resp.raise_for_status()
            return resp.json()


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

    async def add_nodes_batch(self, nodes) -> None:
        for n in nodes:
            name = n.name if hasattr(n, "name") else n.get("name", "")
            ntype = n.type if hasattr(n, "type") else n.get("type", "")
            await self.add_node(name, ntype)

    async def add_edge(self, from_node: str, to_node: str, relation: str, **kwargs) -> None:
        import httpx
        payload = {"from_node": from_node, "to_node": to_node, "relation": relation, **kwargs}
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{self._base}/graph/edges", json=payload)

    async def add_edges_batch(self, edges) -> None:
        for e in edges:
            fn = e.from_node if hasattr(e, "from_node") else e.get("from_node", "")
            tn = e.to_node if hasattr(e, "to_node") else e.get("to_node", "")
            rel = e.relation if hasattr(e, "relation") else e.get("relation", "")
            await self.add_edge(fn, tn, rel)

    async def stats(self) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base}/status")
            return resp.json().get("semantic", {"node_count": 0, "edge_count": 0})

    async def get_nodes(self, type: str | None = None) -> list:
        import httpx
        params = {"keyword": "*", "limit": 500}
        if type:
            params["node_type"] = type
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self._base}/query", params=params)
            if resp.status_code != 200:
                return []
            return [SimpleNamespace(**n) for n in resp.json().get("results", [])]

    async def get_edges(self, node_key: str | None = None) -> list:
        # HTTP API doesn't expose edges listing directly; return empty
        return []

    async def dump(self) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self._base}/graph/data")
            if resp.status_code != 200:
                return {"nodes": [], "edges": []}
            data = resp.json()
        return {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
        }

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
