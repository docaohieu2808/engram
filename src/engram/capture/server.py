"""HTTP webhook server for external agent integration."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from engram.auth import get_auth_context
from engram.auth_models import AuthContext, Role, TokenPayload
from engram.config import Config, load_config
from engram.errors import EngramError, ErrorCode, ErrorResponse
from engram.episodic.store import EpisodicStore
from engram.logging_setup import correlation_id
from engram.models import MemoryType
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph
from engram.tenant import StoreFactory, TenantContext


# --- Request/Response Models ---


class IngestRequest(BaseModel):
    messages: list[dict[str, Any]]


class RememberRequest(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.FACT
    priority: int = 5
    entities: list[str] = []
    tags: list[str] = []


class ThinkRequest(BaseModel):
    question: str


class QueryRequest(BaseModel):
    keyword: str
    node_type: Optional[str] = None  # renamed from `type` to avoid builtin shadow


class SummarizeRequest(BaseModel):
    count: int = 20
    save: bool = False


class TokenRequest(BaseModel):
    sub: str
    role: str = "agent"
    tenant_id: str = "default"
    jwt_secret: str  # caller must provide secret to obtain token


# --- Middleware ---


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Read or generate X-Correlation-ID, set contextvar, echo in response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        correlation_id.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


# --- App Factory ---


_LEGACY_GET_ROUTES = ["/recall", "/query", "/status"]
_LEGACY_POST_ROUTES = ["/remember", "/think", "/cleanup", "/summarize", "/ingest", "/auth/token"]


def create_app(
    episodic: EpisodicStore | None = None,
    graph: SemanticGraph | None = None,
    engine: ReasoningEngine | None = None,
    ingest_fn: Any = None,
    store_factory: StoreFactory | None = None,
) -> FastAPI:
    """Create FastAPI app wired to memory stores.

    Accepts either legacy single-tenant stores (episodic, graph, engine) or a
    StoreFactory for multi-tenant mode. When store_factory is provided, stores
    are resolved per-request from the authenticated tenant_id.
    """
    from fastapi import APIRouter
    app = FastAPI(title="engram", description="Memory traces for AI agents")
    v1 = APIRouter(prefix="/api/v1")

    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Exception handlers ---

    @app.exception_handler(EngramError)
    async def engram_error_handler(request: Request, exc: EngramError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse.from_engram_error(exc).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        err = EngramError(
            ErrorCode.VALIDATION_ERROR,
            "Request validation failed",
            details={"errors": exc.errors()},
        )
        return JSONResponse(
            status_code=err.status_code,
            content=ErrorResponse.from_engram_error(err).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=ErrorResponse.internal().model_dump())

    def _resolve_episodic(auth: AuthContext) -> EpisodicStore:
        """Return tenant-scoped or legacy episodic store."""
        if store_factory is not None:
            TenantContext.set(auth.tenant_id)
            return store_factory.get_episodic(auth.tenant_id)
        assert episodic is not None
        return episodic

    async def _resolve_graph(auth: AuthContext) -> SemanticGraph:
        """Return tenant-scoped or legacy semantic graph."""
        if store_factory is not None:
            TenantContext.set(auth.tenant_id)
            return await store_factory.get_graph(auth.tenant_id)
        assert graph is not None
        return graph

    def _resolve_engine(auth: AuthContext, ep: EpisodicStore, gr: SemanticGraph) -> ReasoningEngine:
        """Return reasoning engine wired to tenant stores."""
        if engine is not None:
            return engine
        # Build a per-request engine when using StoreFactory
        from engram.config import load_config
        cfg = load_config()
        return ReasoningEngine(ep, gr, model=cfg.llm.model, on_think_hook=cfg.hooks.on_think)

    # --- Root-level public route ---

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # --- Legacy 301 redirects ---

    for _path in _LEGACY_GET_ROUTES:
        _target = f"/api/v1{_path}"

        def _make_get_redirect(target: str):
            async def _redirect(request: Request):
                qs = request.url.query
                url = f"{target}?{qs}" if qs else target
                return RedirectResponse(url=url, status_code=301)
            return _redirect

        app.add_api_route(_path, _make_get_redirect(_target), methods=["GET"])

    for _path in _LEGACY_POST_ROUTES:
        _target = f"/api/v1{_path}"

        def _make_post_redirect(target: str):
            async def _redirect(request: Request):
                return RedirectResponse(url=target, status_code=301)
            return _redirect

        app.add_api_route(_path, _make_post_redirect(_target), methods=["POST"])

    # --- API v1 routes ---

    @v1.post("/auth/token")
    async def auth_token(req: TokenRequest):
        """Issue a JWT token. Caller must supply the configured jwt_secret."""
        from engram.auth import create_jwt

        config = load_config()
        if not config.auth.enabled:
            raise HTTPException(status_code=404, detail="Auth not enabled")
        if req.jwt_secret != config.auth.jwt_secret:
            raise HTTPException(status_code=401, detail="Invalid secret")
        try:
            role = Role(req.role)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid role: {req.role}")
        expiry = int(time.time()) + config.auth.jwt_expiry_hours * 3600
        payload = TokenPayload(sub=req.sub, role=role, tenant_id=req.tenant_id, exp=expiry)
        token = create_jwt(payload, config.auth.jwt_secret)
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": config.auth.jwt_expiry_hours * 3600,
        }

    @v1.post("/ingest")
    async def ingest(req: IngestRequest, auth: AuthContext = Depends(get_auth_context)):
        if ingest_fn:
            result = await ingest_fn(req.messages)
            return {"status": "ok", "result": result.model_dump()}
        return {"status": "error", "message": "Ingest function not configured"}

    @v1.post("/remember")
    async def remember(req: RememberRequest, auth: AuthContext = Depends(get_auth_context)):
        ep = _resolve_episodic(auth)
        mem_id = await ep.remember(
            req.content, memory_type=req.memory_type, priority=req.priority,
            entities=req.entities, tags=req.tags,
        )
        return {"status": "ok", "id": mem_id}

    @v1.post("/think")
    async def think(req: ThinkRequest, auth: AuthContext = Depends(get_auth_context)):
        ep = _resolve_episodic(auth)
        gr = await _resolve_graph(auth)
        eng = _resolve_engine(auth, ep, gr)
        answer = await eng.think(req.question)
        return {"status": "ok", "answer": answer}

    @v1.get("/recall")
    async def recall(
        query: str,
        limit: int = 5,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[str] = None,  # comma-separated tag list
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Search episodic memories with optional filters and pagination."""
        ep = _resolve_episodic(auth)
        gr = await _resolve_graph(auth)
        filters = {"memory_type": memory_type} if memory_type else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        results = await ep.search(query, limit=limit + offset, filters=filters, tags=tag_list)
        paginated = results[offset:offset + limit]

        # Also search semantic graph for matching entities
        graph_nodes = await gr.query(query)
        graph_results = []
        for node in graph_nodes[:3]:
            related = await gr.get_related([node.name])
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

    @v1.get("/query")
    async def query(
        keyword: str,
        node_type: Optional[str] = None,
        related_to: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Query semantic graph by keyword, type, or relatedness."""
        gr = await _resolve_graph(auth)
        if related_to:
            related = await gr.get_related([related_to], depth=2)
            nodes = []
            for data in related.values():
                if isinstance(data, dict):
                    nodes.extend(data.get("nodes", []))
        else:
            nodes = await gr.query(keyword, type=node_type)

        paginated = nodes[offset:offset + limit]
        return {
            "status": "ok",
            "results": [n.model_dump() for n in paginated],
            "total": len(nodes),
            "offset": offset,
            "limit": limit,
        }

    @v1.post("/cleanup")
    async def cleanup(auth: AuthContext = Depends(get_auth_context)):
        """Delete all expired memories from episodic store."""
        _require_admin(auth)
        ep = _resolve_episodic(auth)
        deleted = await ep.cleanup_expired()
        return {"status": "ok", "deleted": deleted}

    @v1.post("/summarize")
    async def summarize(req: SummarizeRequest, auth: AuthContext = Depends(get_auth_context)):
        """Summarize recent N memories into key insights using LLM."""
        _require_admin(auth)
        ep = _resolve_episodic(auth)
        gr = await _resolve_graph(auth)
        eng = _resolve_engine(auth, ep, gr)
        summary = await eng.summarize(n=req.count, save=req.save)
        return {"status": "ok", "summary": summary}

    @v1.get("/status")
    async def status(auth: AuthContext = Depends(get_auth_context)):
        ep = _resolve_episodic(auth)
        gr = await _resolve_graph(auth)
        ep_stats = await ep.stats()
        sem_stats = await gr.stats()
        return {"episodic": ep_stats, "semantic": sem_stats}

    app.include_router(v1)
    return app


def _require_admin(auth: AuthContext) -> None:
    """Raise 403 if caller is not ADMIN. Only enforced when auth is enabled.

    Note: get_auth_context already handles auth disabled → default ADMIN role,
    so checking role here is safe for both enabled and disabled modes.
    When auth is disabled, default role is ADMIN so this never raises.
    """
    if auth.role != Role.ADMIN:
        raise EngramError(ErrorCode.FORBIDDEN, "Admin role required")


def run_server(
    episodic: EpisodicStore | None = None,
    graph: SemanticGraph | None = None,
    engine: ReasoningEngine | None = None,
    config: Config | None = None,
    ingest_fn: Any = None,
    store_factory: StoreFactory | None = None,
) -> None:
    """Run HTTP server. Accepts legacy single-tenant stores or a StoreFactory."""
    if config is None:
        config = load_config()
    app = create_app(episodic, graph, engine, ingest_fn, store_factory)
    uvicorn.run(app, host=config.serve.host, port=config.serve.port)
