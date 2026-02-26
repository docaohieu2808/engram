"""WebSocket endpoint — auth, lifecycle, command dispatch, event broadcasting."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from engram.auth import verify_jwt
from engram.auth_models import AuthContext, Role, TokenPayload
from engram.config import Config
from engram.episodic.store import EpisodicStore
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph
from engram.tenant import StoreFactory, TenantContext
from engram.ws.connection_manager import manager
from engram.ws.event_bus import event_bus
from engram.ws.protocol import WSCommand, WSError, WSEvent, WSResponse

logger = logging.getLogger("engram.ws")

_COMMANDS = {"remember", "recall", "think", "feedback", "query", "ingest", "status"}
_WRITE_COMMANDS = {"remember", "feedback", "ingest"}


def register_ws_routes(
    app: FastAPI,
    store_factory: StoreFactory | None,
    episodic: EpisodicStore | None,
    graph: SemanticGraph | None,
    engine: ReasoningEngine | None,
    ingest_fn: Any,
    cache: Any,
    config: Config,
) -> None:
    """Register /ws WebSocket endpoint on the FastAPI app."""

    def _resolve_ep(auth: AuthContext) -> EpisodicStore:
        if store_factory is not None:
            TenantContext.set(auth.tenant_id)
            return store_factory.get_episodic(auth.tenant_id)
        if episodic is None:
            raise RuntimeError("Episodic store not configured")
        return episodic

    async def _resolve_gr(auth: AuthContext) -> SemanticGraph:
        if store_factory is not None:
            TenantContext.set(auth.tenant_id)
            return await store_factory.get_graph(auth.tenant_id)
        if graph is None:
            raise RuntimeError("Semantic graph not configured")
        return graph

    def _resolve_eng(auth: AuthContext, ep: EpisodicStore, gr: SemanticGraph) -> ReasoningEngine:
        if engine is not None:
            return engine
        return ReasoningEngine(
            ep, gr, model=config.llm.model,
            on_think_hook=config.hooks.on_think,
            recall_config=config.recall_pipeline,
            scoring_config=config.scoring,
        )

    # Wire event bus to broadcast manager
    async def _broadcast_handler(tenant_id: str, event: str, data: dict[str, Any]) -> None:
        ws_event = WSEvent(event=event, tenant_id=tenant_id, data=data)
        sender = data.pop("_sender", "")
        await manager.broadcast(tenant_id, ws_event, exclude_sub=sender)

    event_bus.subscribe(_broadcast_handler)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, token: str = Query(default="")):
        # --- Authenticate ---
        auth, sub = _authenticate(token, config)
        if auth is None:
            await ws.close(code=4001, reason="Authentication failed")
            return

        await manager.connect(ws, auth.tenant_id, sub)
        try:
            while True:
                raw = await ws.receive_json()
                # Parse command
                try:
                    cmd = WSCommand.model_validate(raw)
                except Exception:
                    await ws.send_json(WSError(message="Malformed command").model_dump())
                    continue
                # Validate command type
                if cmd.type not in _COMMANDS:
                    await ws.send_json(WSError(
                        id=cmd.id, code="UNKNOWN_COMMAND",
                        message=f"Unknown command: {cmd.type}",
                    ).model_dump())
                    continue
                # RBAC: READER cannot use write commands
                if auth.role == Role.READER and cmd.type in _WRITE_COMMANDS:
                    await ws.send_json(WSError(
                        id=cmd.id, code="FORBIDDEN",
                        message="Reader role cannot perform write operations",
                    ).model_dump())
                    continue
                # Dispatch
                try:
                    result = await _dispatch(
                        cmd, auth, sub, _resolve_ep, _resolve_gr, _resolve_eng, ingest_fn,
                    )
                    await ws.send_json(WSResponse(id=cmd.id, data=result).model_dump())
                except Exception as exc:
                    logger.exception("ws command %s failed", cmd.type)
                    await ws.send_json(WSError(
                        id=cmd.id, code="INTERNAL_ERROR", message=str(exc),
                    ).model_dump())
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(ws, auth.tenant_id, sub)


def _authenticate(token: str, config: Config) -> tuple[AuthContext | None, str]:
    """Validate JWT from query param. Returns (AuthContext, subject) or (None, '')."""
    if not config.auth.enabled:
        return AuthContext(), "anonymous"
    if not token:
        return None, ""
    payload: TokenPayload | None = verify_jwt(token, config.auth.jwt_secret)
    if payload is None:
        return None, ""
    return AuthContext(tenant_id=payload.tenant_id, role=payload.role), payload.sub


async def _dispatch(
    cmd: WSCommand,
    auth: AuthContext,
    sub: str,
    resolve_ep: Any,
    resolve_gr: Any,
    resolve_eng: Any,
    ingest_fn: Any,
) -> dict[str, Any]:
    """Route command to appropriate store operation and return result dict."""
    p = cmd.payload

    if cmd.type == "remember":
        ep = resolve_ep(auth)
        mem_id = await ep.remember(
            p["content"],
            memory_type=p.get("memory_type", "fact"),
            priority=p.get("priority", 5),
            entities=p.get("entities", []),
            tags=p.get("tags", []),
        )
        await event_bus.emit(auth.tenant_id, "memory_created", {
            "id": mem_id, "content": p["content"], "_sender": sub,
        })
        return {"id": mem_id}

    if cmd.type == "recall":
        ep = resolve_ep(auth)
        results = await ep.search(p.get("query", ""), limit=min(p.get("limit", 5), 100))
        return {"results": [_safe_dump(r) for r in results]}

    if cmd.type == "think":
        ep = resolve_ep(auth)
        gr = await resolve_gr(auth)
        eng = resolve_eng(auth, ep, gr)
        result = await eng.think(p["question"])
        return {"answer": result.get("answer", ""), "degraded": result.get("degraded", False)}

    if cmd.type == "feedback":
        from engram.feedback.auto_adjust import adjust_memory
        ep = resolve_ep(auth)
        result = await adjust_memory(ep, p["memory_id"], p["feedback"])
        await event_bus.emit(auth.tenant_id, "feedback_recorded", {
            **result, "_sender": sub,
        })
        return result

    if cmd.type == "query":
        gr = await resolve_gr(auth)
        nodes = await gr.query(p.get("keyword", ""))
        items = nodes if isinstance(nodes, list) else [nodes]
        return {"results": [_safe_dump(n) for n in items[:50]]}

    if cmd.type == "ingest":
        if ingest_fn is None:
            raise RuntimeError("Ingest not configured")
        result = await ingest_fn(p["messages"])
        return {"result": _safe_dump(result)}

    if cmd.type == "status":
        ep = resolve_ep(auth)
        return {"episodic": await ep.stats()}

    return {}


def _safe_dump(obj: Any) -> Any:
    """Convert object to JSON-safe dict, handling Pydantic models and datetimes."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return str(obj)
