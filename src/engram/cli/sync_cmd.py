"""CLI commands for git-friendly memory sync."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from engram.utils import run_async

console = Console()


def register(app: typer.Typer, get_config) -> None:
    """Register sync commands on the main Typer app."""

    def _get_episodic():
        from engram.episodic.store import EpisodicStore
        cfg = get_config()
        return EpisodicStore(cfg.episodic, cfg.embedding)

    @app.command()
    def sync(
        import_: bool = typer.Option(False, "--import", help="Import chunks from .engram/ instead of exporting"),
        status: bool = typer.Option(False, "--status", help="Show sync status"),
        dir: Optional[str] = typer.Option(None, "--dir", help="Sync directory (default: .engram/ in git root)"),
    ):
        """Sync memories to/from .engram/ for git-friendly sharing across machines."""
        from engram.sync.git_sync import export_memories, import_memories, sync_status as do_status

        if status:
            info = do_status(sync_dir=dir)
            if "error" in info:
                console.print(f"[red]{info['error']}[/red]")
            else:
                console.print(f"Sync dir: {info['sync_dir']}")
                console.print(f"Synced memories: {info['synced_ids_count']}")
                console.print(f"Export chunks: {info['chunks']}")
                console.print(f"Imported chunks: {info['imported_chunks']}")
                console.print(f"Last sync: {info['last_sync'] or 'never'}")
            return

        store = _get_episodic()
        if import_:
            result = run_async(import_memories(store, sync_dir=dir))
            if result["imported"] == 0:
                console.print("[dim]Nothing new to import.[/dim]")
            else:
                console.print(f"[green]Imported[/green] {result['imported']} memories from {result['chunks_processed']} chunks")
        else:
            result = run_async(export_memories(store, sync_dir=dir))
            if result["new_count"] == 0:
                console.print("[dim]Nothing new to sync.[/dim]")
            else:
                console.print(f"[green]Synced[/green] {result['new_count']} memories → {result['chunk_file']}")
