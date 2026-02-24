"""API v1 route handlers for engram HTTP server."""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi import HTTPException

from engram.auth import create_jwt, get_auth_context
from engram.auth_models import AuthContext, Role, TokenPayload
from engram.capture.server_models import (
    IngestRequest,
    RememberRequest,
    SummarizeRequest,
    ThinkRequest,
    TokenRequest,
)
from engram.config import load_config
from engram.errors import EngramError, ErrorCode
from engram.episodic.store import EpisodicStore
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph

v1_router = APIRouter(prefix="/api/v1")


def _require_admin(auth: AuthContext) -> None:
    """Raise FORBIDDEN if caller is not ADMIN."""
    if auth.role != Role.ADMIN:
        raise EngramError(ErrorCode.FORBIDDEN, "Admin role required")


def build_v1_router(
    episodic: EpisodicStore,
    graph: SemanticGraph,
    engine: ReasoningEngine,
    ingest_fn: Any = None,
) -> APIRouter:
    """Return a configured v1 APIRouter bound to the given stores."""
    router = APIRouter(prefix="/api/v1")

    @router.post("/auth/token")
    async def auth_token(req: TokenRequest):
        """Issue a JWT token. Caller must supply the configured jwt_secret."""
        config = load_config()
        if not config.auth.enabled:
            raise EngramError(ErrorCode.NOT_FOUND, "Auth not enabled", status_code=404)
        if req.jwt_secret != config.auth.jwt_secret:
            raise EngramError(ErrorCode.AUTH_INVALID, "Invalid secret")
        try:
            role = Role(req.role)
        except ValueError:
            raise EngramError(
                ErrorCode.VALIDATION_ERROR,
                f"Invalid role: {req.role}",
                details={"field": "role", "value": req.role},
            )
        expiry = int(time.time()) + config.auth.jwt_expiry_hours * 3600
        payload = TokenPayload(sub=req.sub, role=role, tenant_id=req.tenant_id, exp=expiry)
        token = create_jwt(payload, config.auth.jwt_secret)
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": config.auth.jwt_expiry_hours * 3600,
        }

    @router.post("/ingest")
    async def ingest(req: IngestRequest, auth: AuthContext = Depends(get_auth_context)):
        if ingest_fn:
            result = await ingest_fn(req.messages)
            return {"status": "ok", "result": result.model_dump()}
        return {"status": "error", "message": "Ingest function not configured"}

    @router.post("/remember")
    async def remember(req: RememberRequest, auth: AuthContext = Depends(get_auth_context)):
        mem_id = await episodic.remember(
            req.content,
            memory_type=req.memory_type,
            priority=req.priority,
            entities=req.entities,
            tags=req.tags,
        )
        return {"status": "ok", "id": mem_id}

    @router.post("/think")
    async def think(req: ThinkRequest, auth: AuthContext = Depends(get_auth_context)):
        answer = await engine.think(req.question)
        return {"status": "ok", "answer": answer}

    @router.get("/recall")
    async def recall(
        query: str,
        limit: int = 5,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[str] = None,
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Search episodic memories with optional filters and pagination."""
        filters = {"memory_type": memory_type} if memory_type else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        results = await episodic.search(query, limit=limit + offset, filters=filters, tags=tag_list)
        paginated = results[offset : offset + limit]

        graph_nodes = await graph.query(query)
        graph_results = []
        for node in graph_nodes[:3]:
            related = await graph.get_related([node.name])
            edges = related.get(node.name, {}).get("edges", [])
            graph_results.append({
                "node": node.model_dump(),
                "edges": [e.model_dump() for e in edges[:5]],
            })

        return {
            "status": "ok",
            "results": [r.model_dump() for r in paginated],
            "graph_results": graph_results,
            "total": len(results),
            "offset": offset,
            "limit": limit,
        }

    @router.get("/query")
    async def query(
        keyword: str,
        node_type: Optional[str] = None,
        related_to: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Query semantic graph by keyword, type, or relatedness."""
        if related_to:
            related = await graph.get_related([related_to], depth=2)
            nodes = []
            for data in related.values():
                if isinstance(data, dict):
                    nodes.extend(data.get("nodes", []))
        else:
            nodes = await graph.query(keyword, type=node_type)

        paginated = nodes[offset : offset + limit]
        return {
            "status": "ok",
            "results": [n.model_dump() for n in paginated],
            "total": len(nodes),
            "offset": offset,
            "limit": limit,
        }

    @router.post("/cleanup")
    async def cleanup(auth: AuthContext = Depends(get_auth_context)):
        """Delete all expired memories from episodic store."""
        _require_admin(auth)
        deleted = await episodic.cleanup_expired()
        return {"status": "ok", "deleted": deleted}

    @router.post("/summarize")
    async def summarize(req: SummarizeRequest, auth: AuthContext = Depends(get_auth_context)):
        """Summarize recent N memories into key insights using LLM."""
        _require_admin(auth)
        summary = await engine.summarize(n=req.count, save=req.save)
        return {"status": "ok", "summary": summary}

    @router.get("/status")
    async def status(auth: AuthContext = Depends(get_auth_context)):
        ep_stats = await episodic.stats()
        sem_stats = await graph.stats()
        return {"episodic": ep_stats, "semantic": sem_stats}

    return router
