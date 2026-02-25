"""CLI command for retrieval audit log inspection."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register the audit command on the main Typer app."""

    @app.command()
    def audit(
        last: int = typer.Option(100, "--last", "-n", help="Number of recent entries to show"),
    ):
        """Show recent retrieval audit log entries."""
        from engram.retrieval_audit_log import RetrievalAuditLog

        cfg = get_config()
        audit_log = RetrievalAuditLog(cfg.retrieval_audit)

        if not audit_log.enabled:
            console.print("[yellow]Retrieval audit logging is disabled.[/yellow]")
            console.print("Enable with: [bold]engram config set retrieval_audit.enabled true[/bold]")
            return

        entries = audit_log.read_recent(last)

        if not entries:
            console.print("[dim]No audit log entries found.[/dim]")
            return

        table = Table(title=f"Retrieval Audit Log (last {len(entries)} entries)")
        table.add_column("Timestamp", style="cyan", no_wrap=True)
        table.add_column("Query", style="white", max_width=50)
        table.add_column("Results", justify="right", style="green")
        table.add_column("Top Score", justify="right", style="yellow")
        table.add_column("Latency (ms)", justify="right", style="magenta")
        table.add_column("Source", style="dim")

        for entry in entries:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            query = entry.get("query", "")
            results_count = str(entry.get("results_count", 0))
            top_score = f"{entry.get('top_score', 0.0):.4f}"
            latency_ms = str(entry.get("latency_ms", 0))
            source = entry.get("source", "")

            table.add_row(ts, query, results_count, top_score, latency_ms, source)

        console.print(table)
