"""CLI command for memory feedback (positive/negative)."""

from __future__ import annotations

import typer
from rich.console import Console

from engram.utils import run_async

console = Console()


def _get_episodic(get_config, namespace: str | None = None):
    from engram.episodic.store import EpisodicStore
    cfg = get_config()
    return EpisodicStore(cfg.episodic, cfg.embedding, namespace=namespace)


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register the feedback command on the main Typer app."""

    def _resolve_namespace():
        return get_namespace() if get_namespace else None

    @app.command()
    def feedback(
        memory_id: str = typer.Argument(..., help="Memory ID to give feedback on"),
        positive: bool = typer.Option(False, "--positive", help="Mark memory as correct"),
        negative: bool = typer.Option(False, "--negative", help="Mark memory as incorrect"),
    ):
        """Give positive or negative feedback on a memory."""
        from engram.feedback.loop import FeedbackProcessor
        from engram.models import FeedbackType

        if positive and negative:
            console.print("[red]Error:[/red] Use either --positive or --negative, not both.")
            raise typer.Exit(1)

        if not positive and not negative:
            console.print("[red]Error:[/red] Specify --positive or --negative.")
            raise typer.Exit(1)

        feedback_type = FeedbackType.POSITIVE if positive else FeedbackType.NEGATIVE

        store = _get_episodic(get_config, _resolve_namespace())
        cfg = get_config()
        processor = FeedbackProcessor(store, cfg.feedback)

        result = run_async(processor.apply_feedback(memory_id, feedback_type))

        if result.get("error") == "memory_not_found":
            console.print(f"[red]Memory not found:[/red] {memory_id}")
            raise typer.Exit(1)

        action = result.get("action", "unknown")

        if action == "deleted":
            console.print(
                f"[yellow]Memory auto-deleted[/yellow] (id={memory_id[:8]}...) "
                "— too many negative signals."
            )
        else:
            label = "[green]positive[/green]" if positive else "[red]negative[/red]"
            console.print(
                f"Feedback {label} applied to memory {memory_id[:8]}... — "
                f"confidence={result.get('confidence', 0):.2f}, "
                f"priority={result.get('priority', 0)}, "
                f"negative_count={result.get('negative_count', 0)}"
            )
