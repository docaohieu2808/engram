"""CLI commands for backup and restore of engram data."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from engram.utils import run_async

console = Console()


def register(app: typer.Typer, get_config) -> None:
    """Register backup/restore commands on the main Typer app."""

    def _get_stores():
        from engram.episodic.store import EpisodicStore
        from engram.semantic import create_graph

        cfg = get_config()
        episodic = EpisodicStore(cfg.episodic, cfg.embedding)
        graph = create_graph(cfg.semantic)
        return episodic, graph

    @app.command()
    def backup(
        output: Path = typer.Option(
            ..., "--output", "-o", help="Output path for the .tar.gz backup archive"
        ),
    ):
        """Export all episodic and semantic memories to a tar.gz archive."""
        from engram.backup import backup as do_backup

        episodic, graph = _get_stores()
        output_str = str(output)
        if not output_str.endswith(".tar.gz"):
            output_str += ".tar.gz"

        console.print(f"[bold]Backing up to:[/bold] {output_str}")
        try:
            manifest = run_async(do_backup(episodic, graph, output_str))
        except Exception as exc:
            console.print(f"[red]Backup failed:[/red] {exc}")
            raise typer.Exit(1)

        console.print(
            f"[green]Backup complete.[/green] "
            f"episodic={manifest['episodic_count']}, "
            f"semantic_nodes={manifest['semantic_nodes']}, "
            f"timestamp={manifest['timestamp']}"
        )

    @app.command()
    def restore(
        archive: Path = typer.Argument(..., help="Path to a .tar.gz backup archive"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    ):
        """Import episodic and semantic memories from a tar.gz backup archive."""
        from engram.backup import restore as do_restore

        if not archive.exists():
            console.print(f"[red]Archive not found:[/red] {archive}")
            raise typer.Exit(1)

        if not yes:
            confirmed = typer.confirm(
                f"Restore from {archive}? This will add data to the current store."
            )
            if not confirmed:
                console.print("[yellow]Restore cancelled.[/yellow]")
                raise typer.Exit(0)

        episodic, graph = _get_stores()
        console.print(f"[bold]Restoring from:[/bold] {archive}")
        try:
            result = run_async(do_restore(episodic, graph, str(archive)))
        except Exception as exc:
            console.print(f"[red]Restore failed:[/red] {exc}")
            raise typer.Exit(1)

        console.print(
            f"[green]Restore complete.[/green] "
            f"episodic={result['episodic_restored']}, "
            f"nodes={result['semantic_nodes_restored']}, "
            f"edges={result['semantic_edges_restored']}"
        )
