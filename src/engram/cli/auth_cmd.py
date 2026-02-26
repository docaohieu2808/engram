"""CLI commands for API key management: create, list, revoke."""

from __future__ import annotations

from typing import Callable

import typer
from rich.console import Console
from rich.table import Table

from engram.auth import create_api_key, list_api_keys, revoke_api_key
from engram.auth_models import Role
from engram.config import Config

console = Console()


def register(auth_app: typer.Typer, get_config: Callable[[], Config]) -> None:
    """Register auth subcommands onto auth_app typer group."""

    @auth_app.command("create-key")
    def create_key(
        name: str = typer.Argument(..., help="Name/label for the API key"),
        role: str = typer.Option("agent", "--role", "-r", help="Role: admin | agent | reader"),
        tenant_id: str = typer.Option("default", "--tenant", "-t", help="Tenant ID"),
    ):
        """Generate a new API key and print it. The key is shown only once."""
        try:
            role_enum = Role(role)
        except ValueError:
            console.print(f"[red]Invalid role '{role}'. Must be: admin, agent, reader[/red]")
            raise typer.Exit(1)

        key, record = create_api_key(name, role_enum, tenant_id)
        console.print(f"\n[green]API key created for '{name}'[/green]")
        console.print(f"  Role:      {record.role.value}")
        console.print(f"  Tenant:    {record.tenant_id}")
        console.print("\n[bold yellow]Key (shown once — save it now):[/bold yellow]")
        console.print(f"  {key}\n")

    @auth_app.command("list-keys")
    def list_keys():
        """List all API keys (name, role, tenant, active status)."""
        records = list_api_keys()
        if not records:
            console.print("[dim]No API keys found.[/dim]")
            return

        table = Table(title="API Keys", show_lines=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Tenant", style="blue")
        table.add_column("Active", style="green")
        table.add_column("Key Hash (prefix)", style="dim")

        for rec in records:
            active_str = "yes" if rec.active else "[red]no[/red]"
            table.add_row(
                rec.name,
                rec.role.value,
                rec.tenant_id,
                active_str,
                rec.key_hash[:12] + "...",
            )

        console.print(table)

    @auth_app.command("revoke-key")
    def revoke_key(
        name: str = typer.Argument(..., help="Name of the API key to revoke"),
    ):
        """Deactivate an API key by name."""
        if revoke_api_key(name):
            console.print(f"[green]API key '{name}' has been revoked.[/green]")
        else:
            console.print(f"[yellow]No active API key found with name '{name}'.[/yellow]")
            raise typer.Exit(1)
