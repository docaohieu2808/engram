"""Individual component health checks for engram subsystems."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field

logger = logging.getLogger("engram")


@dataclass
class ComponentHealth:
    name: str
    status: str  # "healthy", "unhealthy", or "degraded"
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
    p = os.path.expanduser(path)
    try:
        usage = shutil.disk_usage(p)
    except FileNotFoundError:
        usage = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = usage.free / (1024 ** 3)
    status = "healthy" if free_gb > 1 else "unhealthy"
    return ComponentHealth("disk", status, details={"free_gb": round(free_gb, 1)})


def check_api_keys() -> ComponentHealth:
    """Check Gemini API key availability (no network call)."""
    primary = os.environ.get("GEMINI_API_KEY", "")
    fallback = os.environ.get("GEMINI_API_KEY_FALLBACK", "")
    keys_found = sum(1 for k in [primary, fallback] if k)
    if keys_found == 0:
        return ComponentHealth("api_keys", "unhealthy", error="No GEMINI_API_KEY set")
    details = {"primary": bool(primary), "fallback": bool(fallback), "count": keys_found}
    strategy = os.environ.get("GEMINI_KEY_STRATEGY", "failover")
    details["strategy"] = strategy
    return ComponentHealth("api_keys", "healthy", details=details)


async def check_fts5(db_path: str = "~/.engram/fts_index.db") -> ComponentHealth:
    """Check FTS5 full-text search index health."""
    start = time.monotonic()
    try:
        resolved = os.path.expanduser(db_path)
        if not os.path.exists(resolved):
            return ComponentHealth("fts5", "degraded", details={"exists": False})
        import sqlite3
        conn = sqlite3.connect(resolved)
        row = conn.execute("SELECT count(*) FROM memories_fts").fetchone()
        conn.close()
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("fts5", "healthy", ms, {"indexed": row[0]})
    except Exception as exc:
        return ComponentHealth("fts5", "unhealthy", error=str(exc))


async def check_llm(model: str = "", disable_thinking: bool = False) -> ComponentHealth:
    """Check LLM connectivity with a minimal test call."""
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        return ComponentHealth("llm", "degraded", error="No API key — skipped")
    start = time.monotonic()
    try:
        import litellm
        kwargs = dict(
            model=model or "gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        if disable_thinking:
            kwargs["thinking"] = {"type": "disabled"}
        litellm.completion(**kwargs)
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("llm", "healthy", ms, {"model": model or "gemini-2.5-flash"})
    except Exception as exc:
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("llm", "unhealthy", ms, error=str(exc))


async def check_embedding() -> ComponentHealth:
    """Check embedding API with a single-text test."""
    if not os.environ.get("GEMINI_API_KEY"):
        return ComponentHealth("embedding", "degraded", error="No API key — skipped")
    start = time.monotonic()
    try:
        import litellm
        resp = litellm.embedding(
            model="gemini/gemini-embedding-001",
            input=["health check"],
            api_key=os.environ["GEMINI_API_KEY"],
        )
        ms = (time.monotonic() - start) * 1000
        dim = len(resp.data[0]["embedding"]) if resp.data else 0
        return ComponentHealth("embedding", "healthy", ms, {"dimensions": dim})
    except Exception as exc:
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("embedding", "unhealthy", ms, error=str(exc))


async def check_watcher() -> ComponentHealth:
    """Check watcher daemon status via PID file or process scan."""
    # Method 1: PID file (daemon mode with --daemon flag)
    pid_path = os.path.expanduser("~/.engram/watcher.pid")
    if os.path.exists(pid_path):
        try:
            with open(pid_path) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return ComponentHealth("watcher", "healthy", details={"running": True, "pid": pid})
        except (ProcessLookupError, ValueError):
            pass  # Fall through to process scan
        except PermissionError:
            return ComponentHealth("watcher", "healthy", details={"running": True, "pid": pid})

    # Method 2: Process scan (systemd or non-daemon mode)
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "engram watch"], capture_output=True, text=True, timeout=3,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p]
        if pids:
            return ComponentHealth("watcher", "healthy", details={"running": True, "pid": pids[0]})
    except Exception:
        pass

    return ComponentHealth("watcher", "degraded", details={"running": False})


async def check_redis(redis_url: str = "redis://localhost:6379/0") -> ComponentHealth:
    """Ping Redis server. Returns degraded (skip) if redis package unavailable."""
    start = time.monotonic()
    try:
        import redis as redis_lib
        r = redis_lib.from_url(redis_url, socket_connect_timeout=2)
        r.ping()
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("redis", "healthy", ms, {"url": redis_url})
    except ImportError:
        return ComponentHealth("redis", "degraded", error="redis package not installed")
    except Exception as exc:
        ms = (time.monotonic() - start) * 1000
        return ComponentHealth("redis", "unhealthy", ms, error=str(exc))


def check_constitution() -> ComponentHealth:
    """Verify constitution file exists and compute its hash."""
    try:
        from engram.constitution import load_constitution, compute_constitution_hash, get_constitution_path
        path = get_constitution_path()
        content = load_constitution()
        hash_val = compute_constitution_hash(content)
        exists = path.exists()
        return ComponentHealth(
            "constitution", "healthy",
            details={"exists": exists, "hash": hash_val, "laws": 3},
        )
    except Exception as exc:
        return ComponentHealth("constitution", "unhealthy", error=str(exc))


def check_pending_queue() -> ComponentHealth:
    """Check pending queue status — degraded if items are waiting for retry."""
    try:
        from engram.episodic.pending_queue import get_pending_queue
        count = get_pending_queue().count()
        if count == 0:
            return ComponentHealth("pending_queue", "healthy", details={"count": 0})
        return ComponentHealth(
            "pending_queue", "degraded",
            details={"count": count},
            error=f"{count} items waiting for embedding retry",
        )
    except Exception as exc:
        return ComponentHealth("pending_queue", "unhealthy", error=str(exc))


def check_resource_tier() -> ComponentHealth:
    """Report current resource tier from the global resource monitor."""
    try:
        from engram.resource_tier import get_resource_monitor
        monitor = get_resource_monitor()
        s = monitor.status()
        tier = s["tier"]
        style = {"full": "healthy", "standard": "healthy", "basic": "degraded", "readonly": "unhealthy"}
        status = style.get(tier, "degraded")
        return ComponentHealth(
            "resource_tier", status,
            details={
                "tier": tier,
                "recent_failures": s["recent_failures"],
                "forced": s["forced"],
            },
        )
    except Exception as exc:
        return ComponentHealth("resource_tier", "unhealthy", error=str(exc))
