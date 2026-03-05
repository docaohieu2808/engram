"""Generic JSONL session watcher using inotify/watchdog.

Watches directories for *.jsonl session files (OpenClaw, Claude Code, etc).
Parses user/assistant messages and ingests them into engram episodic memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine

from watchdog.events import FileModifiedEvent, FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("engram")

# Regex to strip common system tags from messages
_TAG_PATTERN = re.compile(r"<[^>]+>.*?</[^>]+>", re.DOTALL)
_SYSTEM_TAG_PATTERN = re.compile(r"\[message_id:\s*\d+\]")

# Match Telegram metadata blocks with or without ```json``` backticks
_METADATA_BLOCK_PATTERN = re.compile(
    r"(?:Conversation info|Sender)\s*\(untrusted metadata\):\s*(?:```json\s*)?\{[^}]*\}\s*(?:```)?",
    re.DOTALL,
)

# Patterns to skip entirely — OpenClaw system noise
_SKIP_PATTERNS = [
    re.compile(r"^HEARTBEAT_OK$"),
    re.compile(r"^NO_REPLY$"),
    re.compile(r"Read HEARTBEAT\.md if it exists"),
    re.compile(r"HEARTBEAT_OK.*Current time:", re.DOTALL),
]

# Roles we capture — skip toolCall, toolResult, session, thinking_level_change, etc.
_CAPTURE_ROLES = {"user", "assistant"}


def _extract_text(content: list[dict[str, Any]]) -> str:
    """Extract plain text from content array.

    Handles: text blocks, thinking blocks (assistant reasoning),
    and toolCall args with message/text/content fields.
    """
    parts = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
        elif btype == "thinking":
            text = block.get("thinking", "").strip()
            if text:
                parts.append(text)
        elif btype == "toolCall":
            # Capture message-sending tool calls (send_message, etc.)
            tc = block.get("toolCall", {})
            args = tc.get("args", {})
            for key in ("message", "text", "content"):
                val = args.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                    break
    return "\n".join(parts)


def _should_skip(text: str) -> bool:
    """Return True if message is OpenClaw system noise that should not be stored."""
    for pat in _SKIP_PATTERNS:
        if pat.search(text):
            return True
    return False


def _clean_tags(text: str) -> str:
    """Remove system tags, message IDs, Telegram metadata, and reply markers from captured text."""
    text = _TAG_PATTERN.sub("", text)
    text = _SYSTEM_TAG_PATTERN.sub("", text)
    text = _METADATA_BLOCK_PATTERN.sub("", text)
    # Strip OpenClaw reply markers
    text = re.sub(r"\[\[reply_to_current\]\]\s*", "", text)
    text = re.sub(r"\[\[reply_to:\d+\]\]\s*", "", text)
    return text.strip()


def parse_session_line(line: str) -> dict[str, Any] | None:
    """Parse a single JSONL line from a session file.

    Works with both OpenClaw and Claude Code session formats.
    Returns {"role": "user"|"assistant", "content": "text"} or None.
    """
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None

    # OpenClaw uses type="message", Claude Code uses type="user"/"assistant"
    line_type = data.get("type")
    if line_type not in ("message", "user", "assistant"):
        return None

    msg = data.get("message", {})
    role = msg.get("role")
    if role not in _CAPTURE_ROLES:
        return None

    content = msg.get("content")
    if not content:
        return None

    # Content can be list of blocks or plain string
    if isinstance(content, list):
        text = _extract_text(content)
    elif isinstance(content, str):
        text = content.strip()
    else:
        return None

    if not text:
        return None

    # Skip OpenClaw system noise (HEARTBEAT, NO_REPLY)
    if _should_skip(text):
        return None

    text = _clean_tags(text)
    if not text:
        return None

    # Skip error-only messages
    if msg.get("stopReason") == "error":
        return None

    return {"role": role, "content": text}


# Keep backward compat alias
parse_openclaw_line = parse_session_line


class _SessionFileHandler(FileSystemEventHandler):
    """Watchdog handler that detects new lines in .jsonl session files."""

    def __init__(
        self,
        ingest_fn: Callable[[list[dict[str, Any]]], Coroutine],
        loop: asyncio.AbstractEventLoop,
        label: str = "session",
        positions_file: Path | None = None,
    ):
        super().__init__()
        self._ingest_fn = ingest_fn
        self._loop = loop
        self._label = label
        self._positions_file = positions_file
        # Track file read positions: path -> byte offset
        self._positions: dict[str, int] = self._load_positions()
        # Per-file lock to prevent duplicate reads from rapid inotify events
        self._file_locks: dict[str, threading.Lock] = {}

    def _load_positions(self) -> dict[str, int]:
        """Load persisted positions from disk."""
        if not self._positions_file or not self._positions_file.exists():
            return {}
        try:
            return json.loads(self._positions_file.read_text())
        except Exception:
            return {}

    def _save_positions(self) -> None:
        """Persist positions to disk for crash recovery."""
        if not self._positions_file:
            return
        try:
            self._positions_file.parent.mkdir(parents=True, exist_ok=True)
            self._positions_file.write_text(json.dumps(self._positions))
        except Exception:
            pass

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        if not event.src_path.endswith(".jsonl"):
            return
        self._process_new_lines(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        if not event.src_path.endswith(".jsonl"):
            return
        self._positions[event.src_path] = 0
        self._process_new_lines(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", "")
        if not dest.endswith(".jsonl"):
            return
        if event.src_path in self._positions:
            self._positions[dest] = self._positions.pop(event.src_path)
        else:
            self._positions.setdefault(dest, 0)
        self._process_new_lines(dest)

    def _process_new_lines(self, path: str) -> None:
        """Read new lines from file since last known position, parse and ingest."""
        # Per-file lock prevents duplicate reads from rapid inotify events
        lock = self._file_locks.setdefault(path, threading.Lock())
        if not lock.acquire(blocking=False):
            return  # Another call is already processing this file
        try:
            file_size = os.path.getsize(path)
            last_pos = self._positions.get(path, 0)

            if file_size < last_pos:
                last_pos = 0
            if file_size <= last_pos:
                return

            messages = []
            with open(path, "r", encoding="utf-8") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parsed = parse_session_line(line)
                    if parsed:
                        messages.append(parsed)
                new_pos = f.tell()

            self._positions[path] = new_pos
            self._save_positions()

            if messages:
                logger.info(
                    "%s: captured %d messages from %s",
                    self._label, len(messages), Path(path).name,
                )
                print(f"{self._label}: captured {len(messages)} messages from {Path(path).name}", flush=True)
                fut = asyncio.run_coroutine_threadsafe(
                    self._ingest_fn(messages, source=self._label), self._loop
                )
                def _done(f):
                    try:
                        f.result()
                        print(f"{self._label}: ingested OK", flush=True)
                    except Exception as ex:
                        print(f"{self._label}: INGEST FAILED for {Path(path).name}: {ex}", flush=True)
                        logger.warning("%s ingest failed for %s: %s", self._label, Path(path).name, ex)
                fut.add_done_callback(_done)

        except Exception as e:
            logger.warning("%s watcher error on %s: %s", self._label, path, e)
        finally:
            lock.release()


class SessionWatcher:
    """Watch a directory for JSONL session files and auto-ingest messages.

    Uses watchdog (inotify on Linux) for < 1s latency detection.
    Tracks per-file read position to only process new lines.
    Supports recursive watching (for Claude Code's nested project dirs).
    """

    def __init__(
        self,
        sessions_dir: str,
        ingest_fn: Callable[[list[dict[str, Any]]], Coroutine],
        label: str = "session",
        recursive: bool = False,
        state_dir: str | None = None,
    ):
        self._sessions_dir = Path(os.path.expanduser(sessions_dir))
        self._ingest_fn = ingest_fn
        self._label = label
        self._recursive = recursive
        self._observer: Observer | None = None
        # Persist positions to survive restarts
        if state_dir:
            self._positions_file = Path(os.path.expanduser(state_dir)) / f"watcher-{label}-positions.json"
        else:
            self._positions_file = Path(os.path.expanduser("~/.engram")) / f"watcher-{label}-positions.json"

    async def start(self) -> None:
        """Start watching sessions directory."""
        if not self._sessions_dir.exists():
            logger.warning(
                "%s sessions dir not found: %s — watcher disabled",
                self._label, self._sessions_dir,
            )
            return

        loop = asyncio.get_running_loop()
        handler = _SessionFileHandler(
            self._ingest_fn, loop, self._label,
            positions_file=self._positions_file,
        )

        # Use persisted positions; for NEW files not yet tracked, start at end
        glob_pattern = "**/*.jsonl" if self._recursive else "*.jsonl"
        for jsonl in self._sessions_dir.glob(glob_pattern):
            path_str = str(jsonl)
            if path_str not in handler._positions:
                handler._positions[path_str] = jsonl.stat().st_size

        self._observer = Observer()
        self._observer.schedule(handler, str(self._sessions_dir), recursive=self._recursive)
        self._observer.daemon = True
        self._observer.start()

        logger.info("%s watcher started: %s (recursive=%s)", self._label, self._sessions_dir, self._recursive)

        # Periodic fallback scans to handle missed FS events (debounced: 30s interval)
        try:
            while self._observer.is_alive():
                for jsonl in self._sessions_dir.glob(glob_pattern):
                    handler._process_new_lines(str(jsonl))
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.stop()

    def stop(self) -> None:
        """Stop the watchdog observer."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=5)
            logger.info("%s watcher stopped", self._label)


# Backward compat alias
OpenClawWatcher = SessionWatcher
