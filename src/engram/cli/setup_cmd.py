"""CLI command: engram setup — interactive wizard to connect AI agents."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def register(app: typer.Typer, get_config) -> None:  # noqa: ANN001
    """Register the `setup` command on the main Typer app."""

    @app.command()
    def setup(
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview changes without writing any files.",
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Skip prompts; configure all detected agents automatically.",
        ),
        status: bool = typer.Option(
            False,
            "--status",
            help="Show current agent connection status without making changes.",
        ),
    ):
        """Interactive wizard to connect AI agents to engram shared memory."""
        if status:
            from engram.setup.wizard import run_status
            run_status(get_config)
        else:
            from engram.setup.wizard import run_wizard
            run_wizard(get_config, dry_run=dry_run, non_interactive=non_interactive)
