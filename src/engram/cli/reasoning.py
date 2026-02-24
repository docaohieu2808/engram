"""CLI commands for reasoning: think."""

from __future__ import annotations

import typer
from rich.console import Console

from engram.utils import run_async

console = Console()


def register(app: typer.Typer, get_config) -> None:
    """Register reasoning commands on the main Typer app."""

    def _get_engine():
        from engram.episodic.store import EpisodicStore
        from engram.reasoning.engine import ReasoningEngine
        from engram.semantic import create_graph
        cfg = get_config()
        episodic = EpisodicStore(
            cfg.episodic, cfg.embedding, on_remember_hook=cfg.hooks.on_remember
        )
        graph = create_graph(cfg.semantic)
        return ReasoningEngine(episodic, graph, model=cfg.llm.model, on_think_hook=cfg.hooks.on_think)

    @app.command()
    def think(question: str = typer.Argument(..., help="Question to reason about")):
        """Combined reasoning across both memories."""
        engine = _get_engine()
        answer = run_async(engine.think(question))
        console.print(answer)
