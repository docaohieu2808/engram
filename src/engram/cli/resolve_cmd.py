"""CLI command for entity and temporal resolution."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from engram.utils import run_async

console = Console()


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register the resolve command on the main Typer app."""

    @app.command()
    def resolve(
        query: str = typer.Argument(..., help="Query text to resolve"),
        context: Optional[str] = typer.Option(
            None, "--context", "-c",
            help="Context string (simulates a single prior message)",
        ),
        no_temporal: bool = typer.Option(
            False, "--no-temporal", help="Skip temporal reference resolution"
        ),
        no_pronouns: bool = typer.Option(
            False, "--no-pronouns", help="Skip pronoun resolution"
        ),
    ):
        """Resolve pronouns and temporal references in a query."""
        from engram.recall.entity_resolver import resolve as do_resolve

        context_messages: list[dict] = []
        if context:
            context_messages = [{"role": "user", "content": context}]

        result = run_async(
            do_resolve(
                query,
                context=context_messages,
                resolve_temporal_refs=not no_temporal,
                resolve_pronoun_refs=not no_pronouns,
            )
        )

        console.print(f"[bold]Original:[/bold]  {result.original}")
        console.print(f"[bold]Resolved:[/bold]  {result.resolved}")

        if result.entities:
            entity_names = ", ".join(e.name for e in result.entities)
            console.print(f"[bold]Entities:[/bold]  {entity_names}")

        if result.temporal_refs:
            for original, replacement in result.temporal_refs.items():
                console.print(f"[bold]Temporal:[/bold]  '{original}' → {replacement}")

        if result.resolved == result.original:
            console.print("[dim]No changes made by resolver.[/dim]")
