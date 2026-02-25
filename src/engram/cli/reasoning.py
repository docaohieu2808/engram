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
        from engram.providers.registry import ProviderRegistry
        from engram.reasoning.engine import ReasoningEngine
        from engram.semantic import create_graph
        cfg = get_config()
        episodic = EpisodicStore(
            cfg.episodic, cfg.embedding, on_remember_hook=cfg.hooks.on_remember
        )
        graph = create_graph(cfg.semantic)
        registry = ProviderRegistry()
        registry.load_from_config(cfg)
        providers = registry.get_active()
        return ReasoningEngine(
            episodic, graph, model=cfg.llm.model,
            on_think_hook=cfg.hooks.on_think, providers=providers,
        )

    @app.command()
    def ask(query: str = typer.Argument(..., help="Any question or search query")):
        """Smart query — auto-routes to recall or think based on intent.

        Why/how/explain questions → think (LLM reasoning).
        Simple lookups → recall (vector search).
        """
        from engram.providers.router import classify_intent
        intent = classify_intent(query)
        if intent == "think":
            console.print(f"[dim]intent: think[/dim]")
            engine = _get_engine()
            answer = run_async(engine.think(query))
            console.print(answer)
        else:
            console.print(f"[dim]intent: recall[/dim]")
            from engram.cli.episodic import _get_episodic, _get_semantic
            from engram.providers.registry import ProviderRegistry
            from engram.providers.router import federated_search
            cfg = get_config()
            store = _get_episodic(get_config)
            results = run_async(store.search(query, limit=5))
            # Federated
            registry = ProviderRegistry()
            registry.load_from_config(cfg)
            providers = registry.get_active()
            if providers:
                for r in run_async(federated_search(query, providers, limit=3)):
                    console.print(f"[magenta]\\[{r.source}][/magenta] {r.content[:300]}")
            if not results:
                console.print("[dim]No memories found.[/dim]")
                return
            for mem in results:
                ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                console.print(f"[cyan][{ts}][/cyan] ({mem.memory_type.value}) {mem.content}")

    @app.command()
    def think(question: str = typer.Argument(..., help="Question to reason about")):
        """Combined reasoning across both memories."""
        engine = _get_engine()
        answer = run_async(engine.think(question))
        console.print(answer)
