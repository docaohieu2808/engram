"""File watcher for auto-capturing chat files from inbox directory."""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import signal
import shutil
import sys
from datetime import datetime, timezone, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger("engram")

PID_FILE = Path.home() / ".engram" / "watcher.pid"
_MAX_RETRIES = 3

# Graceful shutdown flag set by SIGTERM handler
_shutdown = False


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Set shutdown flag on SIGTERM — watcher will exit after current file."""
    global _shutdown
    _shutdown = True


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
        self._failed_dir = Path.home() / ".engram" / "failed"
        self._failed_dir.mkdir(parents=True, exist_ok=True)
        self._ingest_fn = ingest_fn
        self._poll_interval = poll_interval
        self._running = False
        self._retry_counts: dict[str, int] = {}

    async def start(self) -> None:
        """Start polling loop. Recovers orphaned .processing files first."""
        global _shutdown
        _shutdown = False
        signal.signal(signal.SIGTERM, _handle_sigterm)
        self._running = True
        self._recover_orphaned_files()
        logger.info("Watching %s (poll=%ss)", self._inbox, self._poll_interval)

        try:
            while self._running and not _shutdown:
                await self._process_inbox()
                await asyncio.sleep(self._poll_interval)
        finally:
            _cleanup_pid()

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False

    def _recover_orphaned_files(self) -> None:
        """Recover .processing files orphaned by previous crashes (>1 hour old)."""
        for pf in self._inbox.glob("*.processing"):
            try:
                age = datetime.now(timezone.utc).timestamp() - pf.stat().st_mtime
                if age > 3600:  # 1 hour
                    # Recover original extension from stem (e.g. "file.jsonl.processing" → "file.jsonl")
                    stem = pf.stem  # "file.jsonl" when suffix is ".processing"
                    if "." in stem:
                        original = pf.with_name(stem)  # keeps the original extension
                    else:
                        original = pf.with_suffix(".json")
                    pf.rename(original)
                    logger.info("Recovered orphaned %s -> %s", pf.name, original.name)
            except Exception as e:
                logger.warning("Failed to recover %s: %s", pf.name, e)

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
        processing = path.with_suffix(".processing")
        try:
            path.rename(processing)
        except FileNotFoundError:
            return  # Another process claimed it

        file_key = path.name
        try:
            messages = self._load_chat_file(processing, is_jsonl=is_jsonl)
            if messages:
                await self._ingest_fn(messages)
                logger.info("Ingested %s (%d messages)", path.name, len(messages))

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            dest = self._processed_dir / f"{ts}_{path.name}"
            shutil.move(str(processing), str(dest))
            # Reset retry count on success
            self._retry_counts.pop(file_key, None)

        except Exception as e:
            logger.error("Error processing %s: %s", path.name, e)
            retries = self._retry_counts.get(file_key, 0) + 1
            self._retry_counts[file_key] = retries

            if retries >= _MAX_RETRIES:
                # Move to failed directory after max retries exhausted
                logger.warning(
                    "%s failed %d times, moving to failed/", path.name, retries
                )
                try:
                    dest = self._failed_dir / path.name
                    shutil.move(str(processing), str(dest))
                except Exception:
                    pass
                self._retry_counts.pop(file_key, None)
            else:
                # Restore to inbox for retry
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


def _cleanup_pid() -> None:
    """Remove PID file if it belongs to current process."""
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            if pid == os.getpid():
                PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def daemonize() -> None:
    """Fork to background daemon process."""
    pid = os.fork()
    if pid > 0:
        # Parent - write PID and exit
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(pid))
        logger.info("Watcher daemon started (PID=%d)", pid)
        sys.exit(0)

    # Child - redirect stdout/stderr to log, close inherited fds
    os.setsid()
    log_path = Path.home() / ".engram" / "watcher.log"
    log = open(log_path, "a")
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    log.close()

    # Register PID cleanup for child process
    atexit.register(_cleanup_pid)


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
