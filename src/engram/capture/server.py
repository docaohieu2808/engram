"""HTTP webhook server for external agent integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from engram.cache import EngramCache
from engram.config import Config, load_config
from engram.errors import EngramError, ErrorCode, ErrorResponse
from engram.episodic.store import EpisodicStore
from engram.rate_limiter import RateLimiter
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph
from engram.tenant import StoreFactory

# Re-export so callers using `engram.capture.server._serialize_memory` etc. keep working
from engram.capture.server_helpers import serialize_memory as _serialize_memory  # noqa: F401
from engram.capture.server_helpers import require_admin as _require_admin  # noqa: F401
from engram.capture.middleware import CorrelationIdMiddleware, RateLimitMiddleware  # noqa: F401
from engram.capture.routers.memory_routes import router as _memory_router
from engram.capture.routers.memories_crud import router as _memories_crud_router
from engram.capture.routers.graph_routes import router as _graph_router
from engram.capture.routers.admin_routes import router as _admin_router

logger = logging.getLogger("engram")

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
    """Create FastAPI app wired to memory stores."""
    # H2: cache config once at startup — no per-request load_config() calls
    _cfg: Config = config if config is not None else load_config()

    app = FastAPI(title="engram", description="Memory traces for AI agents")

    if not _cfg.auth.enabled:
        logger.warning(
            "AUTH DISABLED — all requests treated as admin. "
            "Set auth.enabled=true in config for production."
        )
        if _cfg.serve.host == "0.0.0.0":
            if os.environ.get("ENGRAM_ALLOW_INSECURE") != "1":
                raise RuntimeError(
                    "Refusing to start: auth disabled on 0.0.0.0 (network-exposed). "
                    "Set auth.enabled=true or ENGRAM_ALLOW_INSECURE=1 to override."
                )

    # --- Store app-wide dependencies on state so routers can access them ---
    app.state.episodic = episodic
    app.state.graph = graph
    app.state.engine = engine
    app.state.ingest_fn = ingest_fn
    app.state.store_factory = store_factory
    app.state.cache = cache
    app.state.cfg = _cfg

    # Provider registry (federated memory)
    from engram.providers.registry import ProviderRegistry
    _provider_registry = ProviderRegistry()
    _provider_registry.load_from_config(_cfg)
    app.state.provider_registry = _provider_registry

    # --- Middleware (last added = outermost / first to run) ---
    app.add_middleware(CorrelationIdMiddleware)
    if rate_limiter is not None:
        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=rate_limiter,
            jwt_secret=_cfg.auth.jwt_secret,
        )
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

    # --- Legacy 301/307 redirects ---
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
                return RedirectResponse(url=target, status_code=307)
            return _redirect

        app.add_api_route(_path, _make_post_redirect(_target), methods=["POST"])

    # --- Health probes (no auth required, outside /api/v1) ---

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready():
        from engram.health import deep_check
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

    # --- UI routes ---

    @app.get("/graph")
    async def graph_ui():
        html_path = Path(__file__).parent.parent / "static" / "graph.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>Graph UI not found</h1>", status_code=404)

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/ui")

    @app.get("/ui")
    async def ui_root():
        html_path = Path(__file__).parent.parent / "static" / "ui.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>WebUI not found</h1>", status_code=404)

    @app.get("/ui/{path:path}")
    async def ui_catchall(path: str):
        html_path = Path(__file__).parent.parent / "static" / "ui.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>WebUI not found</h1>", status_code=404)

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- API v1 routers ---
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(_memory_router)
    v1.include_router(_memories_crud_router)
    v1.include_router(_graph_router)
    v1.include_router(_admin_router)
    app.include_router(v1)

    # --- WebSocket routes ---
    from engram.ws import register_ws_routes
    from engram.ws.event_bus import event_bus as _default_event_bus, make_event_bus
    _ws_event_bus = make_event_bus(_cfg.event_bus)
    # Replace module singleton if a Redis bus was created so emit() calls route correctly
    import engram.ws.event_bus as _eb_module
    _eb_module.event_bus = _ws_event_bus
    register_ws_routes(app, store_factory, episodic, graph, engine, ingest_fn, cache, _cfg)

    # Start/stop Redis bus alongside the app lifecycle
    if _ws_event_bus is not _default_event_bus and hasattr(_ws_event_bus, "start"):
        @app.on_event("startup")
        async def _start_event_bus():
            await _ws_event_bus.start()
            logger.info("RedisEventBus started")

        @app.on_event("shutdown")
        async def _stop_event_bus():
            await _ws_event_bus.close()
            logger.info("RedisEventBus stopped")

    return app


async def _build_cache_and_limiter(config: Config) -> tuple[EngramCache | None, RateLimiter | None]:
    """Construct and connect cache/rate_limiter from config."""
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

    if episodic is not None:
        from engram.scheduler import create_default_scheduler
        consolidation_engine = None
        if config.consolidation.enabled:
            try:
                from engram.consolidation.engine import ConsolidationEngine
                consolidation_engine = ConsolidationEngine(
                    episodic, model=config.llm.model, config=config.consolidation,
                    disable_thinking=config.llm.disable_thinking,
                )
            except Exception as e:
                logger.warning("Consolidation engine unavailable: %s", e)
        scheduler = create_default_scheduler(episodic, consolidation_engine, config=config.scheduler)
        app.state.scheduler = scheduler

        @app.on_event("startup")
        async def _start_scheduler():
            scheduler.start()
            logger.info("Memory scheduler started with HTTP server")

    uvicorn.run(app, host=config.serve.host, port=config.serve.port)
