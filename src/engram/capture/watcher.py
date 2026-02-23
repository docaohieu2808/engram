"""File watcher for auto-capturing chat files from inbox directory."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

PID_FILE = Path.home() / ".engram" / "watcher.pid"


class InboxWatcher:
    """Watch inbox directory for new chat files and auto-ingest them."""

    def __init__(
        self,
        inbox_dir: str,
        ingest_fn: Callable[[list[dict[str, Any]]], Coroutine],
        poll_interval: int = 5,
    ):
        self._inbox = Path(os.path.expanduser(inbox_dir))
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._processed_dir = Path.home() / ".engram" / "processed"
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._ingest_fn = ingest_fn
        self._poll_interval = poll_interval
        self._running = False

    async def start(self) -> None:
        """Start polling loop. Recovers orphaned .processing files first."""
        self._running = True
        self._recover_orphaned_files()
        print(f"[engram] Watching {self._inbox} (poll={self._poll_interval}s)")

        while self._running:
            await self._process_inbox()
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False

    def _recover_orphaned_files(self) -> None:
        """Recover .processing files orphaned by previous crashes (>1 hour old)."""
        for pf in self._inbox.glob("*.processing"):
            try:
                age = datetime.now().timestamp() - pf.stat().st_mtime
                if age > 3600:  # 1 hour
                    # Restore original extension from stem (e.g. chat.jsonl.processing → chat.jsonl)
                    original = pf.with_suffix("")
                    if not original.suffix:
                        original = pf.with_suffix(".json")
                    pf.rename(original)
                    print(f"[engram] Recovered orphaned {pf.name} → {original.name}")
            except Exception as e:
                print(f"[engram] Failed to recover {pf.name}: {e}")

    async def _process_inbox(self) -> None:
        """Process all JSON and JSONL files in inbox."""
        for pattern in ("*.json", "*.jsonl"):
            for path in sorted(self._inbox.glob(pattern)):
                if path.name.startswith("."):
                    continue
                await self._process_file(path)

    async def _process_file(self, path: Path) -> None:
        """Atomically claim and process a single file."""
        is_jsonl = path.suffix == ".jsonl"
        # Atomic claim - rename to .processing
        processing = path.with_suffix(".processing")
        try:
            path.rename(processing)
        except FileNotFoundError:
            return  # Another process claimed it

        try:
            messages = self._load_chat_file(processing, is_jsonl=is_jsonl)
            if messages:
                await self._ingest_fn(messages)
                print(f"[engram] Ingested {path.name} ({len(messages)} messages)")

            # Move to processed with timestamp prefix
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = self._processed_dir / f"{ts}_{path.name}"
            shutil.move(str(processing), str(dest))

        except Exception as e:
            print(f"[engram] Error processing {path.name}: {e}")
            # Move back to inbox for retry
            try:
                processing.rename(path)
            except Exception:
                pass

    @staticmethod
    def _load_chat_file(path: Path, is_jsonl: bool | None = None) -> list[dict[str, Any]]:
        """Load messages from chat JSON or JSONL file.

        Args:
            is_jsonl: Explicit format hint. Auto-detects from extension if None.
        """
        if is_jsonl is None:
            is_jsonl = path.suffix == ".jsonl" or (path.suffix == ".processing" and ".jsonl" in path.stem)
        if is_jsonl:
            messages = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return messages

        # Standard JSON: {"messages": [...]} or [...]
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "messages" in data:
            return data["messages"]
        return []


def daemonize() -> None:
    """Fork to background daemon process."""
    pid = os.fork()
    if pid > 0:
        # Parent - write PID and exit
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(pid))
        print(f"[engram] Watcher daemon started (PID={pid})")
        sys.exit(0)

    # Child - redirect stdout/stderr to log
    os.setsid()
    log_path = Path.home() / ".engram" / "watcher.log"
    log = open(log_path, "a")
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())


def is_daemon_running() -> bool:
    """Check if watcher daemon is running."""
    if not PID_FILE.exists():
        return False
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        PID_FILE.unlink(missing_ok=True)
        return False


def stop_daemon() -> bool:
    """Stop the watcher daemon."""
    if not PID_FILE.exists():
        return False
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        return True
    except OSError:
        PID_FILE.unlink(missing_ok=True)
        return False
