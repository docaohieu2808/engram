"""Health check utilities for engram components."""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field


@dataclass
class ComponentHealth:
    name: str
    status: str  # "healthy" or "unhealthy"
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)
    error: str = ""


async def check_chromadb(store) -> ComponentHealth:
    """Check ChromaDB / episodic store health."""
    start = time.monotonic()
    try:
        stats = await store.stats()
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("chromadb", "healthy", ms, {"count": stats["count"]})
    except Exception as exc:
        return ComponentHealth("chromadb", "unhealthy", error=str(exc))


async def check_semantic(graph) -> ComponentHealth:
    """Check semantic graph health."""
    start = time.monotonic()
    try:
        stats = await graph.stats()
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("semantic", "healthy", ms, {"nodes": stats["node_count"]})
    except Exception as exc:
        return ComponentHealth("semantic", "unhealthy", error=str(exc))


async def check_disk(path: str = "~/.engram") -> ComponentHealth:
    """Check available disk space at engram data path."""
    import os
    p = os.path.expanduser(path)
    try:
        usage = shutil.disk_usage(p)
    except FileNotFoundError:
        # Path may not exist yet; fall back to home directory
        usage = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = usage.free / (1024 ** 3)
    status = "healthy" if free_gb > 1 else "unhealthy"
    return ComponentHealth("disk", status, details={"free_gb": round(free_gb, 1)})


async def deep_check(episodic, graph) -> dict:
    """Run all component checks and return aggregated status."""
    results = await asyncio.gather(
        check_chromadb(episodic),
        check_semantic(graph),
        check_disk(),
        return_exceptions=True,
    )

    components: dict = {}
    overall = "healthy"

    for r in results:
        if isinstance(r, Exception):
            continue
        entry: dict = {"status": r.status, "latency_ms": round(r.latency_ms, 1)}
        if r.details:
            entry.update(r.details)
        if r.error:
            entry["error"] = r.error
        components[r.name] = entry
        if r.status == "unhealthy":
            overall = "unhealthy"

    return {"status": overall, "components": components}
