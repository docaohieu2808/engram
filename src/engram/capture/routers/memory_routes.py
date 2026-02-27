"""Routes: POST /remember, GET /recall, POST /think, POST /ingest,
GET /query, POST /feedback, POST /cleanup, POST /cleanup/dedup, POST /summarize, GET /status.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from engram.auth import get_auth_context
from engram.auth_models import AuthContext
from engram.cache import EngramCache
from engram.config import Config
from engram.errors import EngramError, ErrorCode
from engram.models import MemoryType
from engram.capture.server_helpers import (
    require_admin,
    resolve_engine,
    resolve_episodic,
    resolve_graph,
    serialize_memory,
)

router = APIRouter()


# --- Request models ---


class RememberRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    memory_type: MemoryType = MemoryType.FACT
    priority: int = Field(default=5, ge=1, le=10)
    entities: list[str] = []
    tags: list[str] = []


class ThinkRequest(BaseModel):
    question: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    messages: list[dict[str, Any]]


class QueryRequest(BaseModel):
    keyword: str = Field(..., min_length=1)
    node_type: Optional[str] = None


class SummarizeRequest(BaseModel):
    count: int = Field(default=20, ge=1, le=1000)
    save: bool = False


class FeedbackRequest(BaseModel):
    """M1: Typed model — avoids raw request.json()."""
    memory_id: str = Field(..., min_length=1)
    feedback: str = Field(..., pattern="^(positive|negative)$")


# --- Route handlers ---


@router.post("/remember")
async def remember(req: RememberRequest, request: Request, auth: AuthContext = Depends(get_auth_context)):
    state = request.app.state
    ep = resolve_episodic(state, auth)
    mem_id = await ep.remember(
        req.content, memory_type=req.memory_type, priority=req.priority,
        entities=req.entities, tags=req.tags,
    )
    cache: EngramCache | None = getattr(state, "cache", None)
    if cache is not None:
        await cache.invalidate(auth.tenant_id, "recall")
        await cache.invalidate(auth.tenant_id, "think")
    return {"status": "ok", "id": mem_id}


@router.post("/think")
async def think(req: ThinkRequest, request: Request, auth: AuthContext = Depends(get_auth_context)):
    state = request.app.state
    cfg: Config = state.cfg
    cache: EngramCache | None = getattr(state, "cache", None)
    ep = resolve_episodic(state, auth)
    gr = await resolve_graph(state, auth)
    eng = resolve_engine(state, auth, ep, gr)
    # I1 fix: pass active providers to engine for federated search
    provider_registry = getattr(state, "provider_registry", None)
    if provider_registry is not None:
        eng._providers = provider_registry.get_active()

    if cache is not None:
        cached = await cache.get(auth.tenant_id, "think", {"q": req.question})
        if cached is not None:
            return cached
        think_result = await eng.think(req.question)
        result = {"status": "ok", "answer": think_result["answer"], "degraded": think_result["degraded"]}
        await cache.set(auth.tenant_id, "think", {"q": req.question}, result, ttl=cfg.cache.think_ttl)
        return result

    think_result = await eng.think(req.question)
    return {"status": "ok", "answer": think_result["answer"], "degraded": think_result["degraded"]}


@router.post("/ingest")
async def ingest(req: IngestRequest, request: Request, auth: AuthContext = Depends(get_auth_context)):
    state = request.app.state
    ingest_fn = getattr(state, "ingest_fn", None)
    cache: EngramCache | None = getattr(state, "cache", None)
    engine = getattr(state, "engine", None)
    if ingest_fn:
        result = await ingest_fn(req.messages)
        if cache is not None:
            await cache.invalidate(auth.tenant_id, "recall")
        if engine is not None:
            engine.invalidate_cache()
        return {"status": "ok", "result": result.model_dump()}
    raise HTTPException(status_code=501, detail="Ingest function not configured")


@router.get("/recall")
async def recall(
    query: str,
    limit: int = 5,
    offset: int = 0,
    memory_type: Optional[str] = None,
    tags: Optional[str] = None,
    include_graph: bool = True,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """Search episodic memories with optional filters and pagination."""
    from engram.recall.decision import should_skip_recall

    if should_skip_recall(query):
        return {"status": "ok", "results": [], "graph_results": [], "total": 0, "offset": offset, "limit": limit}

    state = request.app.state
    cfg: Config = state.cfg
    cache: EngramCache | None = getattr(state, "cache", None)

    if cache is not None:
        cache_params = {"q": query, "limit": limit, "offset": offset, "mt": memory_type, "tags": tags, "ig": include_graph}
        cached = await cache.get(auth.tenant_id, "recall", cache_params)
        if cached is not None:
            return cached

    ep = resolve_episodic(state, auth)
    limit = min(limit, 100)
    offset = min(offset, 500)
    filters = {"memory_type": memory_type} if memory_type else None
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    results = await ep.search(query, limit=limit + offset, filters=filters, tags=tag_list)
    paginated = results[offset:offset + limit]

    graph_results = []
    if include_graph:
        gr = await resolve_graph(state, auth)
        graph_nodes = await gr.query(query)
        for node in graph_nodes[:3]:
            related = await gr.get_related([node.name])
            edges = related.get(node.name, {}).get("edges", [])
            graph_results.append({
                "node": node.model_dump(),
                "edges": [e.model_dump() for e in edges[:5]],
            })

    result = {
        "status": "ok",
        "results": [r.model_dump() for r in paginated],
        "graph_results": graph_results,
        "total": len(results),
        "offset": offset,
        "limit": limit,
    }
    if cache is not None:
        cache_params = {"q": query, "limit": limit, "offset": offset, "mt": memory_type, "tags": tags, "ig": include_graph}
        await cache.set(auth.tenant_id, "recall", cache_params, result, ttl=cfg.cache.recall_ttl)
    return result


@router.get("/query")
async def query(
    keyword: str,
    node_type: Optional[str] = None,
    related_to: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """Query semantic graph by keyword, type, or relatedness."""
    state = request.app.state
    cfg: Config = state.cfg
    cache: EngramCache | None = getattr(state, "cache", None)

    if cache is not None:
        cache_params = {"kw": keyword, "nt": node_type, "rt": related_to, "offset": offset, "limit": limit}
        cached = await cache.get(auth.tenant_id, "query", cache_params)
        if cached is not None:
            return cached

    gr = await resolve_graph(state, auth)
    if related_to:
        related = await gr.get_related([related_to], depth=2)
        nodes = []
        for data in related.values():
            if isinstance(data, dict):
                nodes.extend(data.get("nodes", []))
    else:
        nodes = await gr.query(keyword, type=node_type)

    paginated = nodes[offset:offset + limit]
    result = {
        "status": "ok",
        "results": [n.model_dump() for n in paginated],
        "total": len(nodes),
        "offset": offset,
        "limit": limit,
    }
    if cache is not None:
        cache_params = {"kw": keyword, "nt": node_type, "rt": related_to, "offset": offset, "limit": limit}
        await cache.set(auth.tenant_id, "query", cache_params, result, ttl=cfg.cache.query_ttl)
    return result


@router.post("/feedback")
async def feedback_endpoint(body: FeedbackRequest, request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Provide feedback on a memory to adjust confidence/importance."""
    from engram.feedback.auto_adjust import adjust_memory

    state = request.app.state
    ep = resolve_episodic(state, auth)
    result = await adjust_memory(ep, body.memory_id, body.feedback)
    return {"status": "ok", **result}


@router.post("/cleanup")
async def cleanup(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Delete all expired memories from episodic store."""
    require_admin(auth)
    state = request.app.state
    ep = resolve_episodic(state, auth)
    deleted = await ep.cleanup_expired()
    return {"status": "ok", "deleted": deleted}


@router.post("/cleanup/dedup")
async def cleanup_dedup(
    threshold: float = 0.85,
    dry_run: bool = False,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """Retroactively deduplicate episodic memories by cosine similarity."""
    require_admin(auth)
    state = request.app.state
    ep = resolve_episodic(state, auth)
    result = await ep.cleanup_dedup(threshold=threshold, dry_run=dry_run)
    return {"status": "ok", **result}


@router.post("/summarize")
async def summarize(req: SummarizeRequest, request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Summarize recent N memories into key insights using LLM."""
    require_admin(auth)
    state = request.app.state
    ep = resolve_episodic(state, auth)
    gr = await resolve_graph(state, auth)
    eng = resolve_engine(state, auth, ep, gr)
    summary = await eng.summarize(n=req.count, save=req.save)
    return {"status": "ok", "summary": summary}


@router.get("/status")
async def status(request: Request, auth: AuthContext = Depends(get_auth_context)):
    state = request.app.state
    ep = resolve_episodic(state, auth)
    gr = await resolve_graph(state, auth)
    ep_stats = await ep.stats()
    sem_stats = await gr.stats()
    return {"episodic": ep_stats, "semantic": sem_stats}
