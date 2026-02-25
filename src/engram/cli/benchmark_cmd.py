"""CLI command for running memory recall benchmarks."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from engram.utils import run_async

console = Console()


def _get_episodic(get_config, namespace: str | None = None):
    from engram.episodic.store import EpisodicStore
    cfg = get_config()
    return EpisodicStore(cfg.episodic, cfg.embedding, namespace=namespace)


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register the benchmark command on the main Typer app."""

    def _resolve_namespace():
        return get_namespace() if get_namespace else None

    @app.command()
    def benchmark(
        questions: str = typer.Option(
            ..., "--questions", "-q", help="Path to benchmark questions JSON file"
        ),
    ):
        """Run memory recall benchmark against a questions file."""
        from engram.benchmark.runner import BenchmarkRunner

        store = _get_episodic(get_config, _resolve_namespace())

        try:
            runner = BenchmarkRunner(store, questions)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        console.print(f"[dim]Running benchmark with questions from {questions}...[/dim]")
        result = run_async(runner.run())

        # Summary
        accuracy_pct = result.accuracy * 100
        color = "green" if accuracy_pct >= 70 else ("yellow" if accuracy_pct >= 40 else "red")
        console.print(
            f"\n[bold]Accuracy:[/bold] [{color}]{accuracy_pct:.1f}%[/{color}] "
            f"({result.correct}/{result.total})"
        )

        # Breakdown by type
        if result.by_type:
            table = Table(title="Results by Type")
            table.add_column("Type", style="cyan")
            table.add_column("Correct", justify="right", style="green")
            table.add_column("Total", justify="right")
            table.add_column("Accuracy", justify="right", style="yellow")

            for q_type, counts in sorted(result.by_type.items()):
                total = counts["total"]
                correct = counts["correct"]
                acc = (correct / total * 100) if total > 0 else 0.0
                table.add_row(q_type, str(correct), str(total), f"{acc:.1f}%")

            console.print(table)
