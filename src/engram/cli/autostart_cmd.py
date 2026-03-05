"""CLI command: engram autostart — install systemd user services for auto-start on boot."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console

console = Console()

_SERVICE_TEMPLATE = """\
[Unit]
Description=Engram Memory Daemon
After=network.target

[Service]
Type=simple
ExecStart={engram_bin} serve
{env_lines}Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

_WATCHER_TEMPLATE = """\
[Unit]
Description=Engram Session Watcher
After=engram.service
Requires=engram.service

[Service]
Type=simple
ExecStart={engram_bin} watch
{env_lines}Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def _read_env_vars() -> str:
    """Read API keys from ~/.engram/.env and format as systemd Environment= lines."""
    env_file = Path.home() / ".engram" / ".env"
    lines = []
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                lines.append(f"Environment={line}\n")
    return "".join(lines)


def _install_systemd(dry_run: bool = False) -> bool:
    """Create and enable systemd user services. Returns True on success."""
    engram_bin = shutil.which("engram")
    if not engram_bin:
        console.print("[red]engram binary not found in PATH[/red]")
        return False

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)

    env_lines = _read_env_vars()
    services = {
        "engram.service": _SERVICE_TEMPLATE.format(engram_bin=engram_bin, env_lines=env_lines),
        "engram-watcher.service": _WATCHER_TEMPLATE.format(engram_bin=engram_bin, env_lines=env_lines),
    }

    for name, content in services.items():
        path = service_dir / name
        if dry_run:
            console.print(f"  [dim]Would write:[/dim] {path}")
        else:
            path.write_text(content)
            console.print(f"  [green]\u2713[/green] Created {path}")

    if dry_run:
        console.print("  [dim]Would enable and start services[/dim]")
        return True

    # Reload systemd
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

    # Enable and start
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "engram.service", "engram-watcher.service"],
        check=True,
    )
    console.print("  [green]\u2713[/green] Services enabled and started")

    # Enable linger so services survive logout
    user = os.environ.get("USER", "")
    if user:
        subprocess.run(["loginctl", "enable-linger", user], check=False)
        console.print(f"  [green]\u2713[/green] Linger enabled for {user}")

    return True


def _uninstall_systemd() -> bool:
    """Stop, disable, and remove systemd user services."""
    service_dir = Path.home() / ".config" / "systemd" / "user"
    names = ["engram-watcher.service", "engram.service"]

    # Stop and disable
    subprocess.run(
        ["systemctl", "--user", "disable", "--now"] + names,
        check=False,
    )
    console.print("  [green]\u2713[/green] Services stopped and disabled")

    # Remove files
    for name in names:
        path = service_dir / name
        if path.exists():
            path.unlink()
            console.print(f"  [green]\u2713[/green] Removed {path}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return True


def _check_status() -> None:
    """Show current autostart status."""
    service_dir = Path.home() / ".config" / "systemd" / "user"
    for name in ["engram.service", "engram-watcher.service"]:
        path = service_dir / name
        if not path.exists():
            console.print(f"  [dim]{name}:[/dim] not installed")
            continue
        result = subprocess.run(
            ["systemctl", "--user", "is-active", name],
            capture_output=True, text=True,
        )
        state = result.stdout.strip()
        color = "green" if state == "active" else "red"
        console.print(f"  [{color}]{name}:[/{color}] {state}")


def register(app: typer.Typer) -> None:
    """Register the `autostart` command on the main Typer app."""

    @app.command()
    def autostart(
        uninstall: bool = typer.Option(False, "--uninstall", help="Remove autostart services."),
        status: bool = typer.Option(False, "--status", help="Show autostart status."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Preview without making changes."),
    ):
        """Install systemd user services so engram starts automatically on boot."""
        if platform.system() != "Linux":
            console.print("[yellow]Autostart currently only supports Linux (systemd).[/yellow]")
            raise typer.Exit(1)

        if status:
            _check_status()
        elif uninstall:
            _uninstall_systemd()
        else:
            _install_systemd(dry_run=dry_run)
