"""MCP server exposing engram memory tools to Claude Code and other MCP clients."""

from __future__ import annotations

from typing import Any

from mcp.server import FastMCP

from engram.config import load_config
from engram.episodic.store import EpisodicStore
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph

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
        _instances["episodic"] = EpisodicStore(
            cfg.episodic, cfg.embedding, on_remember_hook=cfg.hooks.on_remember
        )
    return _instances["episodic"]


def _get_graph() -> SemanticGraph:
    if "graph" not in _instances:
        cfg = _get_config()
        _instances["graph"] = SemanticGraph(cfg.semantic)
    return _instances["graph"]


def _get_engine() -> ReasoningEngine:
    if "engine" not in _instances:
        cfg = _get_config()
        _instances["engine"] = ReasoningEngine(
            _get_episodic(), _get_graph(), model=cfg.llm.model, on_think_hook=cfg.hooks.on_think
        )
    return _instances["engine"]


# Register tools from sub-modules
from engram.mcp import episodic_tools as _ep  # noqa: E402
from engram.mcp import semantic_tools as _sem  # noqa: E402
from engram.mcp import reasoning_tools as _rea  # noqa: E402

_ep.register(mcp, _get_episodic, _get_graph, _get_config)
_sem.register(mcp, _get_graph)
_rea.register(mcp, _get_engine, _get_episodic, _get_graph)


def main():
    """Run the MCP server via stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
