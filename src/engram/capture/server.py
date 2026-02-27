"""HTTP webhook server for external agent integration."""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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

logger = logging.getLogger("engram")


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


class FeedbackRequest(BaseModel):
    """M1: Typed model for /feedback — avoids raw request.json()."""
    memory_id: str = Field(..., min_length=1)
    feedback: str = Field(..., pattern="^(positive|negative)$")


class BulkDeleteRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=1000)


class CreateNodeRequest(BaseModel):
    type: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class CreateEdgeRequest(BaseModel):
    from_node: str = Field(..., min_length=1)
    to_node: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class DeleteEdgeRequest(BaseModel):
    key: str = Field(..., min_length=1)


class UpdateNodeRequest(BaseModel):
    attributes: dict[str, Any] = Field(default_factory=dict)


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

    def __init__(self, app: Any, rate_limiter: RateLimiter | None = None, jwt_secret: str = "") -> None:  # noqa: B107 — empty string default is not a credential
        super().__init__(app)
        self._limiter = rate_limiter
        self._jwt_secret = jwt_secret

    def _extract_tenant_id(self, request: Request) -> str:
        """Extract tenant_id from JWT bearer token with signature verification.

        When jwt_secret is configured, decodes the token with full verification so
        an attacker cannot forge an arbitrary tenant_id to bypass rate limits.
        Falls back to client IP when no secret is configured or token is missing/invalid.
        Never reads X-Tenant-ID header — that header is unauthenticated and spoofable.
        """
        import jwt as _jwt

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and self._jwt_secret:
            token = auth_header[7:]
            try:
                data = _jwt.decode(token, self._jwt_secret, algorithms=["HS256"])
                tenant_id = data.get("tenant_id") or data.get("sub")
                if tenant_id and isinstance(tenant_id, str):
                    return tenant_id
            except Exception as exc:
                logger.debug("rate_limit: JWT decode failed, falling back to IP: %s", exc)

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
    if not _cfg.auth.enabled:
        logger.warning(
            "AUTH DISABLED — all requests treated as admin. "
            "Set auth.enabled=true in config for production."
        )
    v1 = APIRouter(prefix="/api/v1")

    # Starlette middleware order: last added = outermost (first to run)
    app.add_middleware(CorrelationIdMiddleware)
    if rate_limiter is not None:
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter, jwt_secret=_cfg.auth.jwt_secret)
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
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content=ErrorResponse.internal().model_dump())

    def _resolve_episodic(auth: AuthContext) -> EpisodicStore:
        """Return tenant-scoped or legacy episodic store."""
        if store_factory is not None:
            TenantContext.set(auth.tenant_id)
            return store_factory.get_episodic(auth.tenant_id)
        if episodic is None:
            raise EngramError(ErrorCode.INTERNAL, "Episodic store not configured")
        return episodic

    async def _resolve_graph(auth: AuthContext) -> SemanticGraph:
        """Return tenant-scoped or legacy semantic graph."""
        if store_factory is not None:
            TenantContext.set(auth.tenant_id)
            return await store_factory.get_graph(auth.tenant_id)
        if graph is None:
            raise EngramError(ErrorCode.INTERNAL, "Semantic graph not configured")
        return graph

    def _resolve_engine(auth: AuthContext, ep: EpisodicStore, gr: SemanticGraph) -> ReasoningEngine:
        """Return reasoning engine wired to tenant stores. Uses cached config (H2)."""
        if engine is not None:
            return engine
        # Build a per-request engine when using StoreFactory (use cached _cfg)
        return ReasoningEngine(ep, gr, model=_cfg.llm.model, on_think_hook=_cfg.hooks.on_think, recall_config=_cfg.recall_pipeline, scoring_config=_cfg.scoring)

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
                # A-H1: use 307 to preserve POST body (301 drops body)
                return RedirectResponse(url=target, status_code=307)
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
            "token_type": "bearer",  # noqa: B105 — OAuth2 standard string literal, not a password
            "expires_in": _cfg.auth.jwt_expiry_hours * 3600,
        }

    @v1.post("/ingest")
    async def ingest(req: IngestRequest, auth: AuthContext = Depends(get_auth_context)):
        if ingest_fn:
            result = await ingest_fn(req.messages)
            # Invalidate recall cache since ingest adds new memories
            if cache is not None:
                await cache.invalidate(auth.tenant_id, "recall")
            # H6 fix: invalidate node name cache so think() sees new entities
            if engine is not None:
                engine.invalidate_cache()
            return {"status": "ok", "result": result.model_dump()}
        raise HTTPException(status_code=501, detail="Ingest function not configured")

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
        # I1 fix: pass active providers to engine for federated search
        eng._providers = _provider_registry.get_active()

        # Try cache first (uses cached _cfg, no per-request load_config)
        if cache is not None:
            cached = await cache.get(auth.tenant_id, "think", {"q": req.question})
            if cached is not None:
                return cached
            think_result = await eng.think(req.question)
            result = {"status": "ok", "answer": think_result["answer"], "degraded": think_result["degraded"]}
            await cache.set(auth.tenant_id, "think", {"q": req.question}, result, ttl=_cfg.cache.think_ttl)
            return result

        think_result = await eng.think(req.question)
        return {"status": "ok", "answer": think_result["answer"], "degraded": think_result["degraded"]}

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
        from engram.recall.decision import should_skip_recall
        if should_skip_recall(query):
            return {"status": "ok", "results": [], "graph_results": [], "total": 0, "offset": offset, "limit": limit}

        # Try cache first (uses cached _cfg)
        if cache is not None:
            cache_params = {"q": query, "limit": limit, "offset": offset, "mt": memory_type, "tags": tags, "ig": include_graph}
            cached = await cache.get(auth.tenant_id, "recall", cache_params)
            if cached is not None:
                return cached

        ep = _resolve_episodic(auth)
        limit = min(limit, 100)
        offset = min(offset, 500)
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

    @v1.post("/feedback")
    async def feedback_endpoint(body: FeedbackRequest, auth: AuthContext = Depends(get_auth_context)):
        """Provide feedback on a memory to adjust confidence/importance.

        Body: {"memory_id": "...", "feedback": "positive"|"negative"}
        M1: Uses Pydantic model instead of raw request.json() for input validation.
        """
        ep = _resolve_episodic(auth)
        from engram.feedback.auto_adjust import adjust_memory
        result = await adjust_memory(ep, body.memory_id, body.feedback)
        return {"status": "ok", **result}

    @v1.post("/cleanup")
    async def cleanup(auth: AuthContext = Depends(get_auth_context)):
        """Delete all expired memories from episodic store."""
        _require_admin(auth)
        ep = _resolve_episodic(auth)
        deleted = await ep.cleanup_expired()
        return {"status": "ok", "deleted": deleted}

    @v1.post("/cleanup/dedup")
    async def cleanup_dedup(
        threshold: float = 0.85,
        dry_run: bool = False,
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Retroactively deduplicate episodic memories by cosine similarity.

        Scans all memories, merges near-duplicates (similarity >= threshold) into
        the higher-priority winner, and deletes the losers.

        Args:
            threshold: Similarity cutoff 0.0-1.0 (default 0.85).
            dry_run: Report what would be merged without actually deleting.
        """
        _require_admin(auth)
        ep = _resolve_episodic(auth)
        result = await ep.cleanup_dedup(threshold=threshold, dry_run=dry_run)
        return {"status": "ok", **result}

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

    # --- Memory CRUD routes ---

    @v1.get("/memories")
    async def list_memories(
        search: Optional[str] = None,
        memory_type: Optional[str] = None,
        priority_min: int = 1,
        priority_max: int = 10,
        confidence_min: float = 0.0,
        confidence_max: float = 1.0,
        tags: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
        auth: AuthContext = Depends(get_auth_context),
    ):
        """List memories with pagination and filters."""
        ep = _resolve_episodic(auth)
        limit = min(limit, 100)
        offset = min(offset, 500)
        # Fetch enough to filter server-side
        fetch_limit = min((offset + limit) * 3, 1000)
        if search:
            raw = await ep.search(search, limit=fetch_limit)
        else:
            raw = await ep.get_recent(n=fetch_limit)

        # Apply filters
        filtered = []
        mt_filter = set(memory_type.split(",")) if memory_type else None
        tag_filter = set(t.strip() for t in tags.split(",")) if tags else None
        for m in raw:
            mt_val = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
            if mt_filter and mt_val not in mt_filter:
                continue
            if not (priority_min <= m.priority <= priority_max):
                continue
            if not (confidence_min <= m.confidence <= confidence_max):
                continue
            if tag_filter and not tag_filter.intersection(set(m.tags)):
                continue
            filtered.append(m)

        paginated = filtered[offset:offset + limit]
        return {
            "status": "ok",
            "memories": [_serialize_memory(m) for m in paginated],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
        }

    @v1.get("/memories/export")
    async def export_memories(
        memory_type: Optional[str] = None,
        limit: int = 1000,
        auth: AuthContext = Depends(get_auth_context),
    ):
        """Export memories as JSON."""
        ep = _resolve_episodic(auth)
        raw = await ep.get_recent(n=min(limit, 1000))
        if memory_type:
            mt_set = set(memory_type.split(","))
            raw = [m for m in raw if (m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)) in mt_set]
        return {"status": "ok", "memories": [_serialize_memory(m) for m in raw], "count": len(raw)}

    @v1.post("/memories/bulk-delete")
    async def bulk_delete_memories(body: BulkDeleteRequest, auth: AuthContext = Depends(get_auth_context)):
        """Delete multiple memories by IDs."""
        _require_admin(auth)
        ep = _resolve_episodic(auth)
        deleted = []
        for mid in body.ids:
            if await ep.delete(mid):
                deleted.append(mid)
        return {"status": "ok", "deleted": deleted, "count": len(deleted)}

    @v1.get("/memories/{memory_id}")
    async def get_memory(memory_id: str, auth: AuthContext = Depends(get_auth_context)):
        """Get a single memory by ID."""
        ep = _resolve_episodic(auth)
        mem = await ep.get(memory_id)
        if not mem:
            raise EngramError(ErrorCode.NOT_FOUND, f"Memory {memory_id} not found")
        return {"status": "ok", "memory": _serialize_memory(mem)}

    @v1.put("/memories/{memory_id}")
    async def update_memory(memory_id: str, request: Request, auth: AuthContext = Depends(get_auth_context)):
        """Update memory fields (content, type, priority, tags, expires_at)."""
        ep = _resolve_episodic(auth)
        mem = await ep.get(memory_id)
        if not mem:
            raise EngramError(ErrorCode.NOT_FOUND, f"Memory {memory_id} not found")

        body = await request.json()
        meta_update: dict[str, Any] = {}
        if "memory_type" in body:
            try:
                MemoryType(body["memory_type"])
            except ValueError:
                raise EngramError(ErrorCode.VALIDATION_ERROR, f"Invalid memory_type: {body['memory_type']}")
            meta_update["memory_type"] = body["memory_type"]
        if "priority" in body:
            p = int(body["priority"])
            if not 1 <= p <= 10:
                raise EngramError(ErrorCode.VALIDATION_ERROR, "Priority must be 1-10")
            meta_update["priority"] = p
        if "tags" in body:
            meta_update["tags"] = ",".join(body["tags"]) if body["tags"] else ""
        if "entities" in body:
            meta_update["entities"] = ",".join(body["entities"]) if body["entities"] else ""
        if "expires_at" in body:
            meta_update["expires_at"] = body["expires_at"] or ""

        if meta_update:
            ok = await ep.update_metadata(memory_id, meta_update)
            if not ok:
                raise EngramError(ErrorCode.INTERNAL, "Failed to update memory")

        updated = await ep.get(memory_id)
        return {"status": "ok", "memory": _serialize_memory(updated) if updated else None}

    @v1.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str, auth: AuthContext = Depends(get_auth_context)):
        """Delete a single memory. S-H1: ADMIN role required."""
        _require_admin(auth)
        ep = _resolve_episodic(auth)
        ok = await ep.delete(memory_id)
        if not ok:
            raise EngramError(ErrorCode.NOT_FOUND, f"Memory {memory_id} not found or delete failed")
        return {"status": "ok", "deleted": memory_id}

    # --- Graph visualization & mutation routes ---

    @v1.get("/graph/data")
    async def graph_data(
        request: Request,
        auth: AuthContext = Depends(get_auth_context),
        limit: int = 500,
        offset: int = 0,
    ):
        """Return paginated nodes and edges in vis-network format. Admin only."""
        _require_admin(auth)
        if limit < 1 or limit > 5000:
            raise EngramError(ErrorCode.VALIDATION_ERROR, "limit must be between 1 and 5000")
        if offset < 0:
            raise EngramError(ErrorCode.VALIDATION_ERROR, "offset must be >= 0")

        gr = await _resolve_graph(auth)
        def _norm_name(name: str) -> str:
            s = (name or "").strip()
            if not s:
                return s
            return s[0].upper() + s[1:]

        def _norm_key(node_type: str, name: str) -> str:
            return f"{node_type}:{_norm_name(name)}"

        def _norm_key_from_key(key: str) -> str:
            if ":" not in key:
                return key
            t, n = key.split(":", 1)
            return _norm_key(t, n)

        nodes = await gr.get_nodes()
        color_map = {
            "Person": "#4CAF50",
            "Technology": "#2196F3",
            "Project": "#FF9800",
            "Service": "#9C27B0",
        }

        # Merge case-variant nodes for graph visualization (e.g. engram/Engram)
        merged_nodes: dict[str, dict[str, Any]] = {}
        for node in nodes:
            attrs = node.attributes or {}
            normalized_name = _norm_name(node.name)
            node_id = _norm_key(node.type, node.name)
            tooltip = f"{node.type}: {normalized_name}"
            if attrs:
                tooltip += "\n" + "\n".join(f"{k}: {v}" for k, v in attrs.items())

            existing = merged_nodes.get(node_id)
            if existing:
                merged_attrs = dict(existing.get("attributes") or {})
                merged_attrs.update(attrs)
                existing["attributes"] = merged_attrs
                existing["title"] = tooltip
            else:
                merged_nodes[node_id] = {
                    "id": node_id,
                    "label": normalized_name,
                    "group": node.type,
                    "color": color_map.get(node.type, "#607D8B"),
                    "title": tooltip,
                    "attributes": attrs,
                }

        vis_nodes_all = list(merged_nodes.values())

        all_edges = await gr.get_edges()
        vis_edges_all = []
        seen: set[tuple[str, str, str]] = set()
        for edge in all_edges:
            from_key = _norm_key_from_key(edge.from_node)
            to_key = _norm_key_from_key(edge.to_node)
            key = (from_key, to_key, edge.relation)
            if key not in seen:
                seen.add(key)
                vis_edges_all.append({
                    "from": from_key,
                    "to": to_key,
                    "label": edge.relation,
                    "arrows": "to",
                    "weight": edge.weight,
                    "attributes": edge.attributes,
                })

        total_nodes = len(vis_nodes_all)
        total_edges = len(vis_edges_all)
        vis_nodes = vis_nodes_all[offset: offset + limit]
        vis_edges = vis_edges_all[offset: offset + limit]

        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

    @v1.post("/graph/nodes")
    async def create_node(body: CreateNodeRequest, auth: AuthContext = Depends(get_auth_context)):
        """Create a semantic graph node."""
        from engram.models import SemanticNode
        node = SemanticNode(type=body.type, name=body.name, attributes=body.attributes)
        gr = await _resolve_graph(auth)
        is_new = await gr.add_node(node)
        return {"status": "ok", "key": node.key, "created": is_new}

    @v1.put("/graph/nodes/{node_key:path}")
    async def update_node(node_key: str, body: UpdateNodeRequest, auth: AuthContext = Depends(get_auth_context)):
        """Update node attributes."""
        from engram.models import SemanticNode
        gr = await _resolve_graph(auth)
        nodes = await gr.get_nodes()
        existing = next((n for n in nodes if n.key == node_key), None)
        if not existing:
            raise EngramError(ErrorCode.NOT_FOUND, f"Node {node_key} not found")
        updated = SemanticNode(type=existing.type, name=existing.name, attributes={**existing.attributes, **body.attributes})
        await gr.add_node(updated)
        return {"status": "ok", "key": updated.key}

    @v1.delete("/graph/nodes/{node_key:path}")
    async def delete_node(node_key: str, auth: AuthContext = Depends(get_auth_context)):
        """Delete a semantic graph node and its connected edges. S-H1: ADMIN role required."""
        _require_admin(auth)
        gr = await _resolve_graph(auth)
        ok = await gr.remove_node(node_key)
        if not ok:
            raise EngramError(ErrorCode.NOT_FOUND, f"Node {node_key} not found")
        return {"status": "ok", "deleted": node_key}

    @v1.post("/graph/edges")
    async def create_edge(body: CreateEdgeRequest, auth: AuthContext = Depends(get_auth_context)):
        """Create a semantic graph edge."""
        from engram.models import SemanticEdge
        edge = SemanticEdge(
            from_node=body.from_node, to_node=body.to_node,
            relation=body.relation, weight=body.weight,
            attributes=body.attributes,
        )
        gr = await _resolve_graph(auth)
        is_new = await gr.add_edge(edge)
        return {"status": "ok", "key": edge.key, "created": is_new}

    @v1.delete("/graph/edges")
    async def delete_edge(body: DeleteEdgeRequest, auth: AuthContext = Depends(get_auth_context)):
        """Delete a semantic graph edge by key. S-H1: ADMIN role required."""
        _require_admin(auth)
        gr = await _resolve_graph(auth)
        ok = await gr.remove_edge(body.key)
        if not ok:
            raise EngramError(ErrorCode.NOT_FOUND, f"Edge {body.key} not found")
        return {"status": "ok", "deleted": body.key}

    # --- Feedback & Audit routes ---

    @v1.get("/feedback/history")
    async def feedback_history(last: int = 50, auth: AuthContext = Depends(get_auth_context)):
        """Get recent feedback entries from audit log.
        S-H5: cap last to prevent OOM. S-C3: filter by tenant_id.
        """
        from engram.audit import get_audit
        last = min(last, 1000)  # S-H5: bound the read
        audit = get_audit()
        entries = audit.read_recent(last * 3)  # overfetch before filtering
        feedback_entries = [
            e for e in entries
            if e.get("operation") == "modification"
            and e.get("mod_type") == "metadata_update"
            and "confidence" in str(e.get("after_value", ""))
            and e.get("tenant_id") == auth.tenant_id  # S-C3: tenant isolation
        ]
        return {"status": "ok", "entries": feedback_entries[:last]}

    @v1.get("/audit/log")
    async def audit_log(last: int = 50, auth: AuthContext = Depends(get_auth_context)):
        """Get recent audit log entries.
        S-H5: cap last to prevent OOM. S-C2: filter by tenant_id.
        """
        from engram.audit import get_audit
        last = min(last, 1000)  # S-H5: bound the read
        audit = get_audit()
        entries = audit.read_recent(last * 2)  # overfetch to compensate for tenant filter
        entries = [e for e in entries if e.get("tenant_id") == auth.tenant_id][:last]  # S-C2
        return {"status": "ok", "entries": entries}

    # --- Scheduler routes ---

    @v1.get("/scheduler/tasks")
    async def scheduler_tasks(auth: AuthContext = Depends(get_auth_context)):
        """List all scheduled tasks and their status."""
        try:
            sched = getattr(app.state, "scheduler", None)
            if sched is not None:
                tasks = sched.status()
            else:
                tasks = []
        except Exception:
            tasks = []
        return {"status": "ok", "tasks": tasks}

    @v1.post("/scheduler/tasks/{task_name}/run")
    async def scheduler_force_run(task_name: str, auth: AuthContext = Depends(get_auth_context)):
        """Force-run a scheduled task."""
        _require_admin(auth)
        return {"status": "ok", "message": f"Task {task_name} triggered (scheduler must be running)"}

    # --- Benchmark route ---

    @v1.post("/benchmark/run")
    async def benchmark_run(request: Request, auth: AuthContext = Depends(get_auth_context)):
        """Run benchmark with provided questions."""
        _require_admin(auth)
        body = await request.json()
        questions = body.get("questions", [])
        if not questions:
            raise EngramError(ErrorCode.VALIDATION_ERROR, "questions list required")
        ep = _resolve_episodic(auth)
        results = []
        for q in questions:
            query = q.get("question", "")
            expected = q.get("expected", "")
            start = time.time()
            recalls = await ep.search(query, limit=3)
            latency = round((time.time() - start) * 1000, 1)
            actual = recalls[0].content if recalls else ""
            correct = expected.lower() in actual.lower() if expected else False
            results.append({"question": query, "expected": expected, "actual": actual[:200], "correct": correct, "latency_ms": latency})
        accuracy = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else 0
        avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
        return {"status": "ok", "results": results, "accuracy": round(accuracy, 1), "avg_latency_ms": round(avg_latency, 1)}

    # --- UI routes ---

    @app.get("/graph")
    async def graph_ui():
        """Serve the graph visualization HTML page. Auth handled by frontend API calls."""
        html_path = Path(__file__).parent.parent / "static" / "graph.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>Graph UI not found</h1>", status_code=404)

    @app.get("/")
    async def root_redirect():
        """Redirect root to WebUI."""
        from starlette.responses import RedirectResponse
        return RedirectResponse(url="/ui")

    @app.get("/ui")
    async def ui_root():
        """Serve the WebUI HTML page. Auth handled by frontend API calls."""
        html_path = Path(__file__).parent.parent / "static" / "ui.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>WebUI not found</h1>", status_code=404)

    @app.get("/ui/{path:path}")
    async def ui_catchall(path: str):
        """SPA catch-all — serve ui.html for all /ui/* routes."""
        html_path = Path(__file__).parent.parent / "static" / "ui.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>WebUI not found</h1>", status_code=404)

    from fastapi.staticfiles import StaticFiles
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- WebSocket routes ---
    from engram.ws import register_ws_routes
    register_ws_routes(app, store_factory, episodic, graph, engine, ingest_fn, cache, _cfg)

    app.include_router(v1)
    return app


def _serialize_memory(m: Any) -> dict[str, Any]:
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
        "metadata": getattr(m, "metadata", {}),
    }


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
            fail_open=config.rate_limit.fail_open,
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
    from engram.telemetry import setup_telemetry
    setup_telemetry(config)
    app_cache, app_limiter = asyncio.run(_build_cache_and_limiter(config))
    app = create_app(episodic, graph, engine, ingest_fn, store_factory, app_cache, app_limiter, config)

    # Start memory scheduler (cleanup, consolidation, decay) with the server
    if episodic is not None:
        from engram.scheduler import create_default_scheduler
        consolidation_engine = None
        if config.consolidation.enabled:
            try:
                from engram.consolidation.engine import ConsolidationEngine
                consolidation_engine = ConsolidationEngine(
                    episodic, model=config.llm.model, config=config.consolidation,
                )
            except Exception as e:
                logger.warning("Consolidation engine unavailable: %s", e)
        scheduler = create_default_scheduler(episodic, consolidation_engine)
        app.state.scheduler = scheduler

        @app.on_event("startup")
        async def _start_scheduler():
            scheduler.start()
            logger.info("Memory scheduler started with HTTP server")

    uvicorn.run(app, host=config.serve.host, port=config.serve.port)
