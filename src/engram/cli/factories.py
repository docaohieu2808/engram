"""Lazy store/engine factories shared across CLI commands.

All factories cache their instances via nonlocal variables in the ``make_factories``
closure so each CLI invocation only constructs each store once.

When embedded mode is configured and the server daemon is running, factories
will fail due to Qdrant file lock. In that case callers should use HTTP client.
"""

from __future__ import annotations


def _server_is_running() -> bool:
    """Check if engram background server is running."""
    try:
        from engram.cli.daemon_cmd import _is_running
        running, _ = _is_running()
        return running
    except Exception:
        return False


def make_factories(get_config, get_namespace=None):
    """Return a dict of lazy factory callables bound to ``get_config``.

    Returned keys: get_episodic, get_graph, get_providers, get_engine, get_extractor.
    Each callable is idempotent — repeated calls return the cached instance.
    """

    def _resolve_namespace():
        return get_namespace() if get_namespace else None

    _cached_episodic = None
    _cached_graph = None

    _http_proxy = False

    def get_episodic():
        nonlocal _cached_episodic, _cached_graph, _http_proxy
        if _cached_episodic is None:
            from engram.episodic.store import EpisodicStore
            cfg = get_config()
            try:
                _cached_episodic = EpisodicStore(
                    cfg.episodic, cfg.embedding,
                    namespace=_resolve_namespace(),
                    on_remember_hook=cfg.hooks.on_remember,
                    guard_enabled=cfg.ingestion.poisoning_guard,
                )
            except RuntimeError as e:
                if "already accessed" in str(e):
                    from engram.mcp.http_proxy import HttpEpisodicProxy, HttpGraphProxy
                    _cached_episodic = HttpEpisodicProxy(cfg)
                    _cached_graph = HttpGraphProxy(cfg)
                    _http_proxy = True
                else:
                    raise
        return _cached_episodic

    def get_graph():
        nonlocal _cached_graph
        if _cached_graph is None:
            from engram.semantic import create_graph
            cfg = get_config()
            try:
                _cached_graph = create_graph(cfg.semantic)
            except Exception:
                if _http_proxy:
                    from engram.mcp.http_proxy import HttpGraphProxy
                    _cached_graph = HttpGraphProxy(get_config())
                else:
                    raise
        return _cached_graph

    def get_providers():
        from engram.providers.registry import ProviderRegistry
        cfg = get_config()
        registry = ProviderRegistry()
        registry.load_from_config(cfg)
        return registry.get_active()

    def get_engine():
        if _http_proxy:
            from engram.mcp.http_proxy import HttpEngineProxy
            return HttpEngineProxy(get_config())
        from engram.reasoning.engine import ReasoningEngine
        cfg = get_config()
        return ReasoningEngine(
            get_episodic(), get_graph(),
            model=cfg.llm.model,
            on_think_hook=cfg.hooks.on_think,
            providers=get_providers(),
            recall_config=cfg.recall_pipeline,
            disable_thinking=cfg.llm.disable_thinking,
        )

    def get_extractor():
        from engram.capture.extractor import EntityExtractor
        from engram.schema.loader import load_schema
        cfg = get_config()
        schema = load_schema(cfg.semantic.schema_name)
        return EntityExtractor(
            model=cfg.extraction.llm_model or cfg.llm.model,
            schema=schema,
            disable_thinking=cfg.llm.disable_thinking,
            chunk_size=cfg.extraction.chunk_size,
            max_retries=cfg.extraction.max_retries,
            retry_delay_seconds=cfg.extraction.retry_delay_seconds,
            temperature=cfg.extraction.temperature,
        )

    return {
        "get_episodic": get_episodic,
        "get_graph": get_graph,
        "get_providers": get_providers,
        "get_engine": get_engine,
        "get_extractor": get_extractor,
    }
