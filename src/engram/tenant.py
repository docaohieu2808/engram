"""Tenant isolation for engram — ContextVar-based tenant propagation and per-tenant store factory.

Tenant ID flows:
  HTTP API  → auth token → TenantContext.set(tenant_id) per request
  MCP/CLI   → config namespace → TenantContext.set(namespace) at startup
  Default   → "default" (backward compat, single-tenant mode)
"""

from __future__ import annotations

import re
import asyncio
import logging
from collections import OrderedDict
from contextvars import ContextVar
from typing import TYPE_CHECKING

# Top-level imports required so patch("engram.tenant.EpisodicStore") works in tests
from engram.episodic.store import EpisodicStore
from engram.semantic import create_graph

if TYPE_CHECKING:
    from engram.config import Config
    from engram.semantic.graph import SemanticGraph

logger = logging.getLogger("engram")

# Tenant ID contextvar — default "default" for backward compatibility
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="default")

# Tenant ID validation: alphanumeric + hyphens/underscores, max 64 chars
_TENANT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_tenant_id(tenant_id: str) -> str:
    """Validate and return tenant_id. Raises ValueError on invalid format."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id {tenant_id!r}: must match [a-zA-Z0-9_-]{{1,64}}"
        )
    return tenant_id


class TenantContext:
    """Thread-safe tenant ID propagation via contextvars."""

    @staticmethod
    def get() -> str:
        """Return current tenant ID (defaults to 'default')."""
        return _tenant_id.get()

    @staticmethod
    def set(tid: str) -> None:
        """Set tenant ID for the current async task context."""
        validate_tenant_id(tid)
        _tenant_id.set(tid)


class StoreFactory:
    """Create and cache tenant-scoped store instances.

    EpisodicStore: one per tenant (ChromaDB collection engram_{tenant_id}).
    SemanticGraph: one per tenant (SQLite file per tenant, or same PG DB).
    Max 100 tenant graphs cached in memory (LRU eviction).
    """

    _MAX_GRAPH_CACHE = 100
    _MAX_EPISODIC_CACHE = 1000

    def __init__(self, config: "Config") -> None:
        self._config = config
        # LRU caches: OrderedDict preserves insertion order for eviction
        self._episodic_stores: OrderedDict[str, "EpisodicStore"] = OrderedDict()
        self._graphs: OrderedDict[str, "SemanticGraph"] = OrderedDict()
        self._lock: asyncio.Lock | None = None  # M7: lazy init to avoid event loop issues

    def get_episodic(self, tenant_id: str | None = None) -> "EpisodicStore":
        """Return EpisodicStore for the given tenant (defaults to TenantContext.get()).

        Creates a new store if one doesn't exist for this tenant.
        EpisodicStore uses ChromaDB collection engram_{tenant_id} for isolation.
        """
        tid = validate_tenant_id(tenant_id or TenantContext.get())

        if tid in self._episodic_stores:
            # Move to end (most recently used)
            self._episodic_stores.move_to_end(tid)
            return self._episodic_stores[tid]

        store = EpisodicStore(
            self._config.episodic,
            self._config.embedding,
            namespace=tid,
            on_remember_hook=self._config.hooks.on_remember,
            guard_enabled=self._config.ingestion.poisoning_guard,
        )
        self._episodic_stores[tid] = store

        # LRU eviction: remove oldest when over limit
        if len(self._episodic_stores) > self._MAX_EPISODIC_CACHE:
            oldest_tid = next(iter(self._episodic_stores))
            del self._episodic_stores[oldest_tid]
            logger.info("Evicted episodic store for tenant %s (LRU cache full)", oldest_tid)

        return store

    async def _get_lock(self) -> asyncio.Lock:
        """Lazy lock creation to avoid asyncio.Lock() created outside event loop (M7)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_graph(self, tenant_id: str | None = None) -> "SemanticGraph":
        """Return SemanticGraph for the given tenant (defaults to TenantContext.get()).

        For SQLite: separate DB file per tenant at {base_path}.{tenant_id}.db
        For PostgreSQL: same pool, tenant_id column isolates rows (future).
        LRU eviction when cache exceeds 100 entries.
        """
        from engram.config import SemanticConfig

        tid = validate_tenant_id(tenant_id or TenantContext.get())

        async with await self._get_lock():
            if tid in self._graphs:
                self._graphs.move_to_end(tid)
                return self._graphs[tid]

            # Build tenant-scoped semantic config
            cfg = self._config.semantic
            if cfg.provider == "postgresql":
                # PG: share the same pool (tenant isolation via app-level namespacing)
                tenant_cfg = cfg
            else:
                # SQLite: separate file per tenant (strong isolation)
                import os
                base_path = os.path.expanduser(cfg.path)
                # Strip .db suffix if present, add tenant suffix
                if base_path.endswith(".db"):
                    tenant_path = base_path[:-3] + f".{tid}.db"
                else:
                    tenant_path = f"{base_path}.{tid}.db"
                tenant_cfg = SemanticConfig(
                    provider=cfg.provider,
                    path=tenant_path,
                    **{"schema": cfg.schema_name},
                )

            graph = create_graph(tenant_cfg)
            self._graphs[tid] = graph

            # LRU eviction: close and remove oldest when over limit
            if len(self._graphs) > self._MAX_GRAPH_CACHE:
                oldest_tid, oldest_graph = next(iter(self._graphs.items()))
                del self._graphs[oldest_tid]
                try:
                    await oldest_graph.close()
                except Exception as exc:
                    logger.debug("tenant: error closing evicted graph for %s: %s", oldest_tid, exc)
                logger.info("Evicted semantic graph for tenant %s (LRU cache full)", oldest_tid)

            return graph

    async def close_all(self) -> None:
        """Close all cached graph backends (call on server shutdown)."""
        async with await self._get_lock():
            for graph in self._graphs.values():
                try:
                    await graph.close()
                except Exception as exc:
                    logger.debug("tenant: error closing graph on shutdown: %s", exc)
            self._graphs.clear()
            self._episodic_stores.clear()
