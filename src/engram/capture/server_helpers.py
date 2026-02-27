"""Shared helpers and dependency resolvers for engram HTTP server routes."""

from __future__ import annotations

from typing import Any

from engram.auth_models import AuthContext, Role
from engram.config import Config
from engram.errors import EngramError, ErrorCode
from engram.episodic.store import EpisodicStore
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph
from engram.tenant import StoreFactory, TenantContext


def resolve_episodic(state: Any, auth: AuthContext) -> EpisodicStore:
    """Return tenant-scoped or legacy episodic store from app.state."""
    store_factory: StoreFactory | None = getattr(state, "store_factory", None)
    episodic: EpisodicStore | None = getattr(state, "episodic", None)
    if store_factory is not None:
        TenantContext.set(auth.tenant_id)
        return store_factory.get_episodic(auth.tenant_id)
    if episodic is None:
        raise EngramError(ErrorCode.INTERNAL, "Episodic store not configured")
    return episodic


async def resolve_graph(state: Any, auth: AuthContext) -> SemanticGraph:
    """Return tenant-scoped or legacy semantic graph from app.state."""
    store_factory: StoreFactory | None = getattr(state, "store_factory", None)
    graph: SemanticGraph | None = getattr(state, "graph", None)
    if store_factory is not None:
        TenantContext.set(auth.tenant_id)
        return await store_factory.get_graph(auth.tenant_id)
    if graph is None:
        raise EngramError(ErrorCode.INTERNAL, "Semantic graph not configured")
    return graph


def resolve_engine(state: Any, auth: AuthContext, ep: EpisodicStore, gr: SemanticGraph) -> ReasoningEngine:
    """Return reasoning engine wired to tenant stores. Uses cached config (H2)."""
    engine: ReasoningEngine | None = getattr(state, "engine", None)
    cfg: Config = state.cfg
    if engine is not None:
        return engine
    return ReasoningEngine(
        ep, gr,
        model=cfg.llm.model,
        on_think_hook=cfg.hooks.on_think,
        recall_config=cfg.recall_pipeline,
        scoring_config=cfg.scoring,
    )


def require_admin(auth: AuthContext) -> None:
    """Raise 403 if caller is not ADMIN. Only enforced when auth is enabled.

    Note: get_auth_context already handles auth disabled → default ADMIN role,
    so checking role here is safe for both enabled and disabled modes.
    When auth is disabled, default role is ADMIN so this never raises.
    """
    if auth.role != Role.ADMIN:
        raise EngramError(ErrorCode.FORBIDDEN, "Admin role required")


def serialize_memory(m: Any) -> dict[str, Any]:
    """Serialize an EpisodicMemory to a JSON-safe dict."""
    mt = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
    return {
        "id": m.id,
        "content": m.content,
        "memory_type": mt,
        "priority": m.priority,
        "confidence": m.confidence,
        "negative_count": getattr(m, "negative_count", 0),
        "tags": m.tags if isinstance(m.tags, list) else ([t.strip() for t in m.tags.split(",")] if m.tags else []),
        "entities": m.entities if isinstance(m.entities, list) else ([e.strip() for e in m.entities.split(",")] if m.entities else []),
        "access_count": getattr(m, "access_count", 0),
        "decay_rate": getattr(m, "decay_rate", 0.1),
        "timestamp": m.timestamp.isoformat() if hasattr(m.timestamp, "isoformat") else str(m.timestamp),
        "expires_at": m.expires_at.isoformat() if m.expires_at and hasattr(m.expires_at, "isoformat") else str(m.expires_at) if m.expires_at else None,
        "topic_key": getattr(m, "topic_key", None),
        "revision_count": getattr(m, "revision_count", 0),
        "consolidation_group": getattr(m, "consolidation_group", None),
        "consolidated_into": getattr(m, "consolidated_into", None),
        "source": getattr(m, "source", "") or getattr(m, "metadata", {}).get("source", ""),
        "metadata": getattr(m, "metadata", {}),
    }
