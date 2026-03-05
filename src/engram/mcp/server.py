"""MCP server exposing engram memory tools to Claude Code and other MCP clients."""

from __future__ import annotations

from typing import Any

from mcp.server import FastMCP

from engram.config import load_config
from engram.episodic.store import EpisodicStore
from engram.session.store import SessionStore
from engram.providers.base import MemoryProvider
from engram.providers.registry import ProviderRegistry
from engram.reasoning.engine import ReasoningEngine
from engram.semantic import create_graph
from engram.semantic.graph import SemanticGraph
from engram.tenant import TenantContext

# Create MCP server
mcp = FastMCP("engram", instructions="AI agent brain with dual memory (episodic + semantic)")

# Lazy-initialized shared instances
_instances: dict[str, Any] = {}


def _get_config():
    if "config" not in _instances:
        _instances["config"] = load_config()
    return _instances["config"]


def _get_episodic() -> EpisodicStore:
    if "episodic" not in _instances:
        cfg = _get_config()
        namespace = cfg.episodic.namespace or "default"
        TenantContext.set(namespace)
        try:
            _instances["episodic"] = EpisodicStore(
                cfg.episodic, cfg.embedding,
                namespace=namespace,
                on_remember_hook=cfg.hooks.on_remember,
            )
        except RuntimeError as e:
            if "already accessed" in str(e):
                # Embedded Qdrant locked by HTTP server — proxy through HTTP API
                import logging
                logging.getLogger("engram").info("MCP: embedded Qdrant locked, proxying via HTTP")
                from engram.mcp.http_proxy import HttpEpisodicProxy, HttpGraphProxy
                _instances["episodic"] = HttpEpisodicProxy(cfg)
                _instances["graph"] = HttpGraphProxy(cfg)
                _instances["_http_proxy"] = True
            else:
                raise
    return _instances["episodic"]


def _get_graph() -> SemanticGraph:
    if "graph" not in _instances:
        cfg = _get_config()
        try:
            _instances["graph"] = create_graph(cfg.semantic)
        except Exception:
            if _instances.get("_http_proxy"):
                from engram.mcp.http_proxy import HttpGraphProxy
                _instances["graph"] = HttpGraphProxy(cfg)
            else:
                raise
    return _instances["graph"]


def _get_providers() -> list[MemoryProvider]:
    if "providers" not in _instances:
        cfg = _get_config()
        registry = ProviderRegistry()
        registry.load_from_config(cfg)
        _instances["providers"] = registry
    return _instances["providers"].get_active()


def _get_session_store() -> SessionStore:
    if "session_store" not in _instances:
        cfg = _get_config()
        _instances["session_store"] = SessionStore(cfg.session.sessions_dir)
    return _instances["session_store"]


def _get_engine() -> ReasoningEngine:
    if "engine" not in _instances:
        if _instances.get("_http_proxy"):
            from engram.mcp.http_proxy import HttpEngineProxy
            cfg = _get_config()
            _instances["engine"] = HttpEngineProxy(cfg)
        else:
            cfg = _get_config()
            _instances["engine"] = ReasoningEngine(
                _get_episodic(), _get_graph(), model=cfg.llm.model,
                on_think_hook=cfg.hooks.on_think, providers=_get_providers(),
                recall_config=cfg.recall_pipeline,
                disable_thinking=cfg.llm.disable_thinking,
            )
    return _instances["engine"]


# Register tools from sub-modules
from engram.mcp import episodic_tools as _ep  # noqa: E402
from engram.mcp import semantic_tools as _sem  # noqa: E402
from engram.mcp import reasoning_tools as _rea  # noqa: E402

_ep.register(mcp, _get_episodic, _get_graph, _get_config, get_providers=_get_providers)
_sem.register(mcp, _get_graph)
_rea.register(mcp, _get_engine, _get_episodic, _get_graph, get_providers=_get_providers)

from engram.mcp import session_tools as _sess  # noqa: E402
_sess.register(mcp, _get_session_store, _get_episodic)


def main():
    """Run the MCP server via stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
