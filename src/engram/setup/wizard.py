"""Interactive setup wizard UI for engram — rich display + questionary prompts."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    from engram.setup.connectors.base import AgentConnector, ConfigureResult, DetectionResult
    from engram.setup.federation.base import FederationProvider

console = Console()


def run_wizard(get_config, dry_run: bool, non_interactive: bool) -> None:  # noqa: ANN001
    """Orchestrate the full interactive setup wizard flow.

    Args:
        get_config:       Callable returning the engram Config object (from CLI closure).
        dry_run:          Preview changes without writing any files.
        non_interactive:  Auto-select all detected agents, skip prompts.
    """
    from engram.setup.detector import scan_agents
    from engram.setup.federation import get_all_providers
    from engram.setup.verifier import (
        check_server_running,
        get_restart_hints,
        start_server_background,
        verify_all,
    )

    # Fall back to non-interactive when stdin is not a terminal (e.g. piped/CI)
    if not sys.stdin.isatty() and not non_interactive:
        console.print(
            "[yellow]Non-TTY environment detected — switching to --non-interactive mode.[/yellow]"
        )
        non_interactive = True

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold cyan]Engram Setup Wizard[/bold cyan]\n"
            "[dim]Connect your AI agents to shared memory[/dim]",
            expand=False,
        )
    )

    # ── Scan phase ────────────────────────────────────────────────────────────
    pairs: list[tuple[AgentConnector, DetectionResult]] = []
    with console.status("[bold green]Scanning for AI agents...[/bold green]", spinner="dots"):
        pairs = scan_agents()

    if not pairs:
        console.print(
            "[dim]No connectors registered yet. "
            "Agent connectors will be added in subsequent phases.[/dim]\n"
        )
        return

    _display_scan_results(pairs)

    # ── Agent selection ───────────────────────────────────────────────────────
    selected_connectors = _select_agents(pairs, non_interactive)

    if not selected_connectors:
        console.print("[yellow]No agents selected. Nothing to configure.[/yellow]\n")
        return

    # ── Federation provider selection (optional) ──────────────────────────────
    federation_entries = []
    if not non_interactive:
        all_providers = get_all_providers()
        federation_entries = _select_and_configure_providers(all_providers)

    # ── Confirmation ──────────────────────────────────────────────────────────
    if not non_interactive:
        agent_names = ", ".join(c.display_name for c in selected_connectors)
        confirmed = questionary.confirm(
            f"Configure {len(selected_connectors)} agent(s): {agent_names}?",
            default=True,
        ).ask()
        if not confirmed:
            console.print("[dim]Aborted.[/dim]\n")
            return

    # ── Configure phase ───────────────────────────────────────────────────────
    if dry_run:
        console.print("\n[yellow]Dry-run mode — no files will be written.[/yellow]")

    cfg_results = _configure_selected(selected_connectors, dry_run)

    # ── Write federation provider entries to config ───────────────────────────
    if federation_entries and not dry_run:
        _write_federation_entries(get_config, federation_entries)

    # ── Verification phase ────────────────────────────────────────────────────
    verify_results: list[tuple[str, bool, str]] = []
    if not dry_run:
        with console.status("[bold green]Verifying configuration...[/bold green]", spinner="dots"):
            verify_results = verify_all(selected_connectors)

    # ── Server management ─────────────────────────────────────────────────────
    server_running = check_server_running()
    server_started = False
    if not dry_run and not server_running and not non_interactive:
        server_started = _offer_start_server()
        if server_started:
            server_running = True

    # ── Summary panel ─────────────────────────────────────────────────────────
    restart_hints = get_restart_hints(selected_connectors)
    _display_summary(
        cfg_results=cfg_results,
        verify_results=verify_results,
        federation_count=len(federation_entries),
        server_running=server_running,
        server_started=server_started,
        restart_hints=restart_hints,
        dry_run=dry_run,
    )


def run_status(get_config) -> None:  # noqa: ANN001
    """Display current engram connection status for all registered agents."""
    from engram.setup.detector import scan_agents
    from engram.setup.verifier import check_server_running, verify_all

    console.print()
    console.print(
        Panel(
            "[bold cyan]Engram Status[/bold cyan]\n"
            "[dim]Current agent connection status[/dim]",
            expand=False,
        )
    )

    with console.status("[bold green]Scanning agents...[/bold green]", spinner="dots"):
        pairs = scan_agents()

    if not pairs:
        console.print("[dim]No connectors registered.[/dim]\n")
        return

    _display_scan_results(pairs)

    # Verify installed agents
    installed_connectors = [c for c, r in pairs if r.installed]
    if installed_connectors:
        verify_results = verify_all(installed_connectors)
        _display_verify_results(verify_results)

    # Server status
    server_running = check_server_running()
    status_label = "[green]Running[/green]" if server_running else "[red]Not running[/red]"
    console.print(f"\n  Engram server: {status_label} (port 8765)")
    if not server_running:
        console.print("  [dim]Start with: engram serve[/dim]")
    console.print()


# ── Display helpers ────────────────────────────────────────────────────────────


def _display_scan_results(
    results: list[tuple[AgentConnector, DetectionResult]],
) -> None:
    """Render detection results as a rich table."""
    table = Table(title="Detected Agents", show_lines=True, min_width=60)
    table.add_column("Status", justify="center", width=7)
    table.add_column("Agent", style="bold")
    table.add_column("Version", width=12)
    table.add_column("Details")

    for _connector, result in results:
        if result.installed:
            status = "[green]✓[/green]"
        else:
            status = "[red]✗[/red]"

        table.add_row(
            status,
            result.name,
            result.version or "[dim]-[/dim]",
            result.details or "[dim]-[/dim]",
        )

    console.print(table)


def _display_verify_results(verify_results: list[tuple[str, bool, str]]) -> None:
    """Render verification results as a rich table."""
    table = Table(title="Verification", show_lines=True, min_width=60)
    table.add_column("Status", justify="center", width=7)
    table.add_column("Agent", style="bold")
    table.add_column("Result")

    for name, ok, msg in verify_results:
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(status, name, msg)

    console.print(table)


def _select_agents(
    results: list[tuple[AgentConnector, DetectionResult]],
    non_interactive: bool,
) -> list[AgentConnector]:
    """Return connectors the user wants to configure.

    In non-interactive mode, all detected (installed) connectors are returned.
    """
    installed_pairs = [(c, r) for c, r in results if r.installed]

    if not installed_pairs:
        console.print("[dim]No installed agents found to configure.[/dim]")
        return []

    if non_interactive:
        return [c for c, _ in installed_pairs]

    choices = [
        questionary.Choice(
            title=r.name + (f" ({r.version})" if r.version else ""),
            value=c,
            checked=True,  # pre-check installed agents
        )
        for c, r in installed_pairs
    ]

    selected = questionary.checkbox(
        "Select agents to connect (space to toggle, enter to confirm):",
        choices=choices,
    ).ask()

    # questionary returns None if the user aborts (Ctrl-C)
    return selected if selected is not None else []


def _select_and_configure_providers(
    providers: list[FederationProvider],
) -> list:
    """Let user pick federation providers and collect their ProviderEntry configs."""
    from engram.config import ProviderEntry

    detected = [p for p in providers if p.detect()]
    choices = [
        questionary.Choice(
            title=p.display_name + (" [detected]" if p in detected else ""),
            value=p,
            checked=p in detected,
        )
        for p in providers
    ]

    selected = questionary.checkbox(
        "Connect external memory providers? (optional — press enter to skip):",
        choices=choices,
    ).ask()

    if not selected:
        return []

    entries: list[ProviderEntry] = []
    for provider in selected:
        console.print(f"\n[bold]Configure {provider.display_name}[/bold]")
        entry = provider.prompt_config()
        if entry is not None:
            entries.append(entry)

    return entries


def _configure_selected(
    connectors: list[AgentConnector],
    dry_run: bool,
) -> list[ConfigureResult]:
    """Run configure() on each connector, showing a rich progress bar."""
    from engram.setup.connectors.base import ConfigureResult as CR

    results: list[CR] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Configuring agents...", total=len(connectors))
        for connector in connectors:
            progress.update(task, description=f"Configuring [bold]{connector.display_name}[/bold]...")
            try:
                result = connector.configure(dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001
                result = CR(success=False, message=f"Error: {exc}")
            results.append(result)
            progress.advance(task)

    return results


def _write_federation_entries(get_config, entries: list) -> None:
    """Append federation ProviderEntry items to config.yaml (preserving existing)."""
    from engram.config import get_config_path, load_config, save_config

    config = load_config()
    existing_names = {p.name for p in config.providers}
    added = 0
    for entry in entries:
        if entry.name not in existing_names:
            config.providers.append(entry)
            added += 1

    if added:
        save_config(config)
        console.print(f"  [green]✓[/green] Added {added} federation provider(s) to config.yaml")


def _offer_start_server() -> bool:
    """Prompt user to start engram server; return True if started successfully."""
    from engram.setup.verifier import start_server_background

    start = questionary.confirm(
        "Engram server is not running. Start it now?",
        default=True,
    ).ask()

    if not start:
        return False

    with console.status("[bold green]Starting engram server...[/bold green]", spinner="dots"):
        ok = start_server_background()

    if ok:
        console.print("  [green]✓[/green] Engram server started on port 8765")
    else:
        console.print("  [yellow]![/yellow] Could not confirm server started — run 'engram serve' manually")

    return ok


def _display_summary(
    cfg_results: list[ConfigureResult],
    verify_results: list[tuple[str, bool, str]],
    federation_count: int,
    server_running: bool,
    server_started: bool,
    restart_hints: list[str],
    dry_run: bool,
) -> None:
    """Render a final summary panel."""
    lines: list[str] = []

    # Configuration results
    for result in cfg_results:
        icon = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        lines.append(f"  {icon} {result.message}")

    # Verification results (only when not dry-run)
    if verify_results:
        lines.append("")
        lines.append("  [dim]Verification:[/dim]")
        for name, ok, msg in verify_results:
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            lines.append(f"  {icon} {name}: {msg}")

    # Federation
    if federation_count:
        lines.append("")
        lines.append(f"  [cyan]·[/cyan] {federation_count} federation provider(s) added to config")

    # Server status
    lines.append("")
    if server_started:
        lines.append("  [green]✓[/green] Engram server started (port 8765)")
    elif server_running:
        lines.append("  [green]✓[/green] Engram server already running (port 8765)")
    else:
        lines.append("  [yellow]![/yellow] Engram server not running — start with: engram serve")

    # Restart hints
    if restart_hints:
        lines.append("")
        lines.append("  [dim]Next steps:[/dim]")
        for hint in restart_hints:
            lines.append(f"  [cyan]·[/cyan] {hint}")

    if dry_run:
        lines.append("")
        lines.append("  [yellow]Dry-run: no files were written.[/yellow]")

    success_count = sum(1 for r in cfg_results if r.success)
    all_ok = success_count == len(cfg_results)

    footer = (
        "\n[bold green]Done! All agents now share memory via Engram.[/bold green]"
        if all_ok
        else f"\n[yellow]{success_count}/{len(cfg_results)} agents configured successfully.[/yellow]"
    )

    body = "\n".join(lines) + footer
    title = "[bold green]Setup Complete[/bold green]" if all_ok else "[bold yellow]Setup Finished[/bold yellow]"

    console.print()
    console.print(Panel(body, title=title, expand=False))
    console.print()
