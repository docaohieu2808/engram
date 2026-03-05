"""CLI commands: engram start / stop / logs — daemon management for serve."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console

logger = logging.getLogger("engram")
console = Console()

_ENGRAM_DIR = Path.home() / ".engram"
_PID_FILE = _ENGRAM_DIR / "serve.pid"
_LOG_FILE = _ENGRAM_DIR / "serve.log"


def _is_running() -> tuple[bool, int | None]:
    """Check if serve daemon is running. Returns (running, pid)."""
    if not _PID_FILE.exists():
        return False, None
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        _PID_FILE.unlink(missing_ok=True)
        return False, None
    try:
        os.kill(pid, 0)
    except OSError:
        _PID_FILE.unlink(missing_ok=True)
        return False, None
    # Verify PID belongs to engram (Linux /proc)
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        if "engram" not in cmdline:
            _PID_FILE.unlink(missing_ok=True)
            return False, None
    except OSError:
        pass
    return True, pid


def _cleanup_pid() -> None:
    """Remove PID file on exit if it belongs to this process."""
    try:
        if _PID_FILE.exists():
            pid = int(_PID_FILE.read_text().strip())
            if pid == os.getpid():
                _PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _daemonize() -> None:
    """Fork to background daemon process."""
    pid = os.fork()
    if pid > 0:
        # Parent — write child PID and exit
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(pid))
        console.print(f"  [green]engram started[/green] (PID={pid})")
        console.print(f"  Logs: [cyan]{_LOG_FILE}[/cyan]")
        sys.exit(0)

    # Child — detach and redirect output to log file
    os.setsid()
    log = open(_LOG_FILE, "a")
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    log.close()
    atexit.register(_cleanup_pid)


def register(app: typer.Typer, get_config) -> None:
    """Register start/stop/logs commands."""

    @app.command()
    def start(
        port: int | None = typer.Option(None, "--port", "-p"),
        host: str | None = typer.Option(None, "--host"),
        foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground (no daemon)"),
    ):
        """Start engram server in background (daemon mode)."""
        running, pid = _is_running()
        if running:
            console.print(f"[yellow]Already running[/yellow] (PID={pid})")
            return

        # Load env and config
        from engram.cli.system import _load_env_file
        _load_env_file()
        from engram.config import apply_llm_api_key
        cfg = get_config()
        apply_llm_api_key(cfg)
        if port:
            cfg.serve.port = port
        if host:
            cfg.serve.host = host

        if not foreground:
            _daemonize()

        # Now in child (daemon) or foreground — start server
        from engram.capture.server import run_server
        from engram.telemetry import setup_telemetry
        from engram.cli.ingest import do_ingest_messages
        from engram.cli.factories import make_factories

        setup_telemetry(cfg)
        _factories = make_factories(get_config, lambda: None)
        _get_episodic = _factories["get_episodic"]
        _get_graph = _factories["get_graph"]
        _get_engine = _factories["get_engine"]
        _get_extractor = _factories["get_extractor"]

        async def ingest_messages(messages, source: str = ""):
            return await do_ingest_messages(messages, _get_extractor, _get_graph, _get_episodic, source=source)

        run_server(_get_episodic(), _get_graph(), _get_engine(), cfg, ingest_messages)

    @app.command()
    def stop():
        """Stop the engram background server."""
        running, pid = _is_running()
        if not running:
            console.print("[yellow]Not running.[/yellow]")
            return
        try:
            os.kill(pid, signal.SIGTERM)
            _PID_FILE.unlink(missing_ok=True)
            console.print(f"[green]Stopped[/green] (PID={pid})")
        except OSError as e:
            console.print(f"[red]Failed to stop:[/red] {e}")
            _PID_FILE.unlink(missing_ok=True)

    @app.command()
    def logs(
        lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
        follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    ):
        """Show engram server logs."""
        if not _LOG_FILE.exists():
            console.print("[dim]No log file found.[/dim]")
            return
        if follow:
            os.execlp("tail", "tail", "-f", str(_LOG_FILE))
        else:
            import subprocess
            subprocess.run(["tail", "-n", str(lines), str(_LOG_FILE)])
