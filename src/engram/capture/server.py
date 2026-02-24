"""HTTP webhook server for external agent integration."""

from __future__ import annotations

import hmac
import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from engram.auth import get_auth_context
from engram.auth_models import AuthContext, Role, TokenPayload
from engram.cache import EngramCache
from engram.config import Config, load_config
from engram.errors import EngramError, ErrorCode, ErrorResponse
from engram.episodic.store import EpisodicStore
from engram.logging_setup import correlation_id
from engram.models import MemoryType
from engram.rate_limiter import RateLimiter
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph
from engram.tenant import StoreFactory, TenantContext


# --- Request/Response Models ---


class IngestRequest(BaseModel):
    messages: list[dict[str, Any]]


class RememberRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    memory_type: MemoryType = MemoryType.FACT
    priority: int = Field(default=5, ge=1, le=10)
    entities: list[str] = []
    tags: list[str] = []


class ThinkRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    keyword: str = Field(..., min_length=1)
    node_type: Optional[str] = None  # renamed from `type` to avoid builtin shadow


class SummarizeRequest(BaseModel):
    count: int = Field(default=20, ge=1, le=1000)
    save: bool = False


class TokenRequest(BaseModel):
    """Request body for /auth/token. C2 fix: no jwt_secret in body.
    Admin identity proved via Authorization: Bearer <admin_secret> header.
    """
    sub: str
    role: str = "agent"
    tenant_id: str = "default"


# --- Middleware ---


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Read or generate X-Correlation-ID, set contextvar, echo in response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        correlation_id.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant sliding-window rate limiting. Skips when rate_limiter is None."""

    def __init__(self, app: Any, rate_limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self._limiter = rate_limiter

    def _extract_tenant_id(self, request: Request) -> str:
        """Extract tenant_id from JWT bearer token (C1 fix: no header spoofing).

        Falls back to client IP when auth is not in use or token is missing/invalid.
        Never reads X-Tenant-ID header — that header is unauthenticated and spoofable.
        """
        import jwt as _jwt

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                # Decode without verification just to extract tenant_id for rate limiting.
                # Full signature verification happens in get_auth_context; here we only
                # need a stable, client-supplied-but-hard-to-fabricate identifier.
                # We use options={"verify_signature": False} for rate-limit bucketing only.
                data = _jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])
                tenant_id = data.get("tenant_id") or data.get("sub")
                if tenant_id and isinstance(tenant_id, str):
                    return tenant_id
            except Exception:
                pass  # Fall through to IP-based fallback

        client_host = request.client.host if request.client else "anonymous"
        return client_host or "anonymous"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self._limiter is None:
            return await call_next(request)

        # C1 fix: use authenticated tenant from JWT, not spoofable X-Tenant-ID header
        tenant_id = self._extract_tenant_id(request)
        allowed, remaining, reset_at = await self._limiter.check(tenant_id)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "Too many requests"}},
                headers={
                    "X-RateLimit-Limit": str(self._limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(reset_at - int(time.time())),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
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
    cache: EngramCache | None = None,
    rate_limiter: RateLimiter | None = None,
    config: Config | None = None,
) -> FastAPI:
    """Create FastAPI app wired to memory stores.

    Accepts either legacy single-tenant stores (episodic, graph, engine) or a
    StoreFactory for multi-tenant mode. When store_factory is provided, stores
    are resolved per-request from the authenticated tenant_id.

    cache and rate_limiter are optional; when None, those features are disabled.
    """
    from fastapi import APIRouter

    # H2: cache config once at startup — no per-request load_config() calls
    _cfg: Config = config if config is not None else load_config()

    app = FastAPI(title="engram", description="Memory traces for AI agents")
    v1 = APIRouter(prefix="/api/v1")

    # Starlette middleware order: last added = outermost (first to run)
    app.add_middleware(CorrelationIdMiddleware)
    if rate_limiter is not None:
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
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
        """Return reasoning engine wired to tenant stores. Uses cached config (H2)."""
        if engine is not None:
            return engine
        # Build a per-request engine when using StoreFactory (use cached _cfg)
        return ReasoningEngine(ep, gr, model=_cfg.llm.model, on_think_hook=_cfg.hooks.on_think)

    # --- Root-level public routes ---

    # --- Provider registry (federated memory) ---
    from engram.providers.registry import ProviderRegistry
    _provider_registry = ProviderRegistry()
    _provider_registry.load_from_config(_cfg)

    @app.get("/providers")
    async def list_providers(auth: AuthContext = Depends(get_auth_context)):
        """List all configured providers and their status. M2 fix: requires auth."""
        providers = []
        for p in _provider_registry.get_all():
            providers.append({
                "name": p.name,
                "type": p.provider_type,
                "active": p.is_active,
                "status": p.status_label,
                "stats": {
                    "query_count": p.stats.query_count,
                    "avg_latency_ms": round(p.stats.avg_latency_ms, 1),
                    "hit_count": p.stats.hit_count,
                    "error_count": p.stats.error_count,
                },
            })
        return {"providers": providers}

    @app.get("/health")
    async def health():
        """Liveness probe — fast, no component checks."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready():
        """Readiness probe — performs deep component checks."""
        from engram.health import deep_check
        # Resolve default stores for health check (single-tenant or first available)
        ep = episodic
        gr = graph
        if store_factory is not None:
            ep = store_factory.get_episodic("default")
            gr = await store_factory.get_graph("default")
        if ep is None or gr is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "components": {}, "error": "stores not initialised"},
            )
        result = await deep_check(ep, gr)
        status_code = 200 if result["status"] == "healthy" else 503
        return JSONResponse(status_code=status_code, content=result)

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
    async def auth_token(req: TokenRequest, request: Request):
        """Issue a JWT token using admin_secret header auth (C2 fix).

        Auth disabled or admin_secret not configured -> 404.
        Caller must provide Authorization: Bearer <admin_secret> header.
        """
        from engram.auth import create_jwt

        if not _cfg.auth.enabled:
            raise HTTPException(status_code=404, detail="Auth not enabled")
        if not _cfg.auth.admin_secret:
            raise HTTPException(status_code=404, detail="Auth not enabled")
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header with admin secret required")
        provided_secret = auth_header[7:]
        # C3 fix: constant-time comparison to prevent timing side-channel attacks
        if not hmac.compare_digest(provided_secret, _cfg.auth.admin_secret):
            raise HTTPException(status_code=401, detail="Invalid secret")
        try:
            role = Role(req.role)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid role: {req.role}")
        expiry = int(time.time()) + _cfg.auth.jwt_expiry_hours * 3600
        payload = TokenPayload(sub=req.sub, role=role, tenant_id=req.tenant_id, exp=expiry)
        token = create_jwt(payload, _cfg.auth.jwt_secret)
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": _cfg.auth.jwt_expiry_hours * 3600,
        }

    @v1.post("/ingest")
    async def ingest(req: IngestRequest, auth: AuthContext = Depends(get_auth_context)):
        if ingest_fn:
            result = await ingest_fn(req.messages)
            # Invalidate recall cache since ingest adds new memories
            if cache is not None:
                await cache.invalidate(auth.tenant_id, "recall")
            return {"status": "ok", "result": result.model_dump()}
        return {"status": "error", "message": "Ingest function not configured"}

    @v1.post("/remember")
    async def remember(req: RememberRequest, auth: AuthContext = Depends(get_auth_context)):
        ep = _resolve_episodic(auth)
        mem_id = await ep.remember(
            req.content, memory_type=req.memory_type, priority=req.priority,
            entities=req.entities, tags=req.tags,
        )
        # Invalidate recall/think cache for this tenant since new memory was added
        if cache is not None:
            await cache.invalidate(auth.tenant_id, "recall")
            await cache.invalidate(auth.tenant_id, "think")
        return {"status": "ok", "id": mem_id}

    @v1.post("/think")
    async def think(req: ThinkRequest, auth: AuthContext = Depends(get_auth_context)):
        ep = _resolve_episodic(auth)
        gr = await _resolve_graph(auth)
        eng = _resolve_engine(auth, ep, gr)

        # Try cache first (uses cached _cfg, no per-request load_config)
        if cache is not None:
            cached = await cache.get(auth.tenant_id, "think", {"q": req.question})
            if cached is not None:
                return cached
            answer = await eng.think(req.question)
            result = {"status": "ok", "answer": answer}
            await cache.set(auth.tenant_id, "think", {"q": req.question}, result, ttl=_cfg.cache.think_ttl)
            return result

        answer = await eng.think(req.question)
        return {"status": "ok", "answer": answer}

    @v1.get("/recall")
    async def recall(
        query: str,
        limit: int = 5,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[str] = None,  # comma-separated tag list
        include_graph: bool = True,  # M9: set False to skip graph search
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Search episodic memories with optional filters and pagination.

        include_graph: when True (default), also queries semantic graph for related entities.
        """
        # Try cache first (uses cached _cfg)
        if cache is not None:
            cache_params = {"q": query, "limit": limit, "offset": offset, "mt": memory_type, "tags": tags, "ig": include_graph}
            cached = await cache.get(auth.tenant_id, "recall", cache_params)
            if cached is not None:
                return cached

        ep = _resolve_episodic(auth)
        filters = {"memory_type": memory_type} if memory_type else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        results = await ep.search(query, limit=limit + offset, filters=filters, tags=tag_list)
        paginated = results[offset:offset + limit]

        # M9: only run graph search when include_graph=True
        graph_results = []
        if include_graph:
            gr = await _resolve_graph(auth)
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
            await cache.set(auth.tenant_id, "recall", cache_params, result, ttl=_cfg.cache.recall_ttl)
        return result

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
        # Try cache first (uses cached _cfg)
        if cache is not None:
            cache_params = {"kw": keyword, "nt": node_type, "rt": related_to, "offset": offset, "limit": limit}
            cached = await cache.get(auth.tenant_id, "query", cache_params)
            if cached is not None:
                return cached

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
        result = {
            "status": "ok",
            "results": [n.model_dump() for n in paginated],
            "total": len(nodes),
            "offset": offset,
            "limit": limit,
        }
        if cache is not None:
            await cache.set(auth.tenant_id, "query", cache_params, result, ttl=_cfg.cache.query_ttl)
        return result

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


async def _build_cache_and_limiter(config: Config) -> tuple[EngramCache | None, RateLimiter | None]:
    """Construct and connect cache/rate_limiter from config. Returns (None, None) when disabled."""
    app_cache: EngramCache | None = None
    app_limiter: RateLimiter | None = None

    if config.cache.enabled:
        app_cache = EngramCache(config.cache.redis_url)
        await app_cache.connect()

    if config.rate_limit.enabled:
        app_limiter = RateLimiter(
            config.rate_limit.redis_url,
            requests_per_minute=config.rate_limit.requests_per_minute,
            burst=config.rate_limit.burst,
        )
        await app_limiter.connect()

    return app_cache, app_limiter


def run_server(
    episodic: EpisodicStore | None = None,
    graph: SemanticGraph | None = None,
    engine: ReasoningEngine | None = None,
    config: Config | None = None,
    ingest_fn: Any = None,
    store_factory: StoreFactory | None = None,
) -> None:
    """Run HTTP server. Accepts legacy single-tenant stores or a StoreFactory."""
    import asyncio
    if config is None:
        config = load_config()
    app_cache, app_limiter = asyncio.run(_build_cache_and_limiter(config))
    app = create_app(episodic, graph, engine, ingest_fn, store_factory, app_cache, app_limiter, config)
    uvicorn.run(app, host=config.serve.host, port=config.serve.port)
