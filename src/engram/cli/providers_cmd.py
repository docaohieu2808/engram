"""CLI commands for provider management: discover, list, test, stats, add."""

from __future__ import annotations

import asyncio
from typing import Callable

import typer
from rich.console import Console
from rich.table import Table

from engram.config import Config, ProviderEntry, save_config

console = Console()


def register(app: typer.Typer, providers_app: typer.Typer, get_config: Callable[[], Config]) -> None:
    """Register provider CLI commands."""

    @app.command()
    def discover(
        add_host: str = typer.Option(None, "--add-host", help="Add a remote host for scanning"),
    ):
        """Auto-discover external memory services (local + remote)."""
        cfg = get_config()

        if add_host:
            if add_host not in cfg.discovery.hosts:
                cfg.discovery.hosts.append(add_host)
                save_config(cfg)
                console.print(f"[green]Added {add_host} to discovery hosts[/green]")
            else:
                console.print(f"[yellow]{add_host} already in discovery hosts[/yellow]")
            return

        from engram.providers.discovery import discover as run_discover

        console.print("[bold]Scanning for memory services...[/bold]\n")
        found = asyncio.run(run_discover(cfg.discovery))

        if not found:
            console.print("[yellow]No services found.[/yellow]")
            console.print("Tip: Add remote hosts with --add-host <ip/domain>")
            return

        # Show what was found
        table = Table(title="Discovered Services")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Location")
        table.add_column("Status")

        for entry in found:
            location = entry.url or entry.path or entry.command
            status = "[green]ready[/green]" if entry.enabled else "[yellow]confirm needed[/yellow]"
            table.add_row(entry.name, entry.type, location, status)

        console.print(table)

        # Ask user to add
        if typer.confirm("\nAdd discovered services to config?"):
            existing_names = {p.name for p in cfg.providers}
            added = 0
            for entry in found:
                if entry.name not in existing_names:
                    cfg.providers.append(entry)
                    added += 1
            save_config(cfg)
            console.print(f"[green]Added {added} providers to config[/green]")

    @providers_app.command("list")
    def list_providers():
        """List all configured providers and their status."""
        cfg = get_config()
        from engram.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry.load_from_config(cfg)

        table = Table(title="Memory Providers")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Location")
        table.add_column("Status")

        for provider in registry.get_all():
            location = ""
            # Find matching config entry for location info
            for entry in cfg.providers:
                if entry.name == provider.name:
                    location = entry.url or entry.path or entry.command
                    break
            status = f"[green]✓ {provider.status_label}[/green]" if provider.is_active else f"[red]✗ {provider.status_label}[/red]"
            table.add_row(provider.name, provider.provider_type, location, status)

        if not registry.get_all():
            console.print("[yellow]No providers configured. Run 'engram discover' to find services.[/yellow]")
        else:
            console.print(table)

    @providers_app.command("test")
    def test_provider(name: str = typer.Argument(help="Provider name to test")):
        """Test connectivity and search for a specific provider."""
        cfg = get_config()
        from engram.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry.load_from_config(cfg)
        provider = registry.get(name)

        if not provider:
            console.print(f"[red]Provider '{name}' not found[/red]")
            raise typer.Exit(1)

        async def _test():
            console.print(f"[bold]Testing {name}...[/bold]")

            # Health check
            healthy = await provider.health()
            if healthy:
                console.print("  [green]✓ Health check passed[/green]")
            else:
                console.print("  [red]✗ Health check failed[/red]")
                return

            # Test search
            test_query = "test memory search"
            results = await provider.search(test_query, limit=3)
            console.print(f"  [green]✓ Search returned {len(results)} results[/green]")
            for r in results[:3]:
                console.print(f"    [{r.source}] {r.content[:100]}")

        asyncio.run(_test())

    @providers_app.command("stats")
    def provider_stats():
        """Show query statistics for all providers."""
        cfg = get_config()
        from engram.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry.load_from_config(cfg)

        table = Table(title="Provider Statistics")
        table.add_column("Provider", style="cyan")
        table.add_column("Queries", justify="right")
        table.add_column("Avg(ms)", justify="right")
        table.add_column("Hits", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Status")

        for provider in registry.get_all():
            s = provider.stats
            status = "[green]✓ active[/green]" if provider.is_active else f"[red]✗ {provider.status_label}[/red]"
            table.add_row(
                provider.name,
                str(s.query_count),
                f"{s.avg_latency_ms:.0f}",
                str(s.hit_count),
                str(s.error_count),
                status,
            )

        console.print(table)

    @providers_app.command("add")
    def add_provider(
        name: str = typer.Option(..., "--name", help="Provider name"),
        url: str = typer.Option(None, "--url", help="Provider URL"),
        path: str = typer.Option(None, "--path", help="File path for file adapter"),
        provider_type: str = typer.Option(None, "--type", help="Provider type (rest, file, postgres, mcp)"),
    ):
        """Add a provider directly by URL or path."""
        cfg = get_config()

        if url:
            # Auto-detect service type
            async def _detect():
                from engram.providers.discovery import _detect_service_type, _build_entry_from_service
                svc = await _detect_service_type(url)
                if svc:
                    entry = _build_entry_from_service(svc, url=url)
                    entry.name = name
                    return entry
                return None

            entry = asyncio.run(_detect())
            if entry:
                cfg.providers.append(entry)
                save_config(cfg)
                console.print(f"[green]✓ Detected {entry.type} service, added '{name}'[/green]")
                return

            if not provider_type:
                console.print("[yellow]Could not auto-detect service type. Use --type to specify.[/yellow]")
                raise typer.Exit(1)

            # Manual entry
            entry = ProviderEntry(name=name, type=provider_type, url=url, enabled=True)
            cfg.providers.append(entry)
            save_config(cfg)
            console.print(f"[green]Added provider '{name}' (type={provider_type})[/green]")

        elif path:
            entry = ProviderEntry(name=name, type="file", path=path, enabled=True)
            cfg.providers.append(entry)
            save_config(cfg)
            console.print(f"[green]Added file provider '{name}' at {path}[/green]")

        else:
            console.print("[red]Must specify --url or --path[/red]")
            raise typer.Exit(1)
