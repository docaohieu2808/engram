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
from pathlib import Path
from typing import Any, Callable, Coroutine

from watchdog.events import FileModifiedEvent, FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("engram")

# Regex to strip common system tags from messages
_TAG_PATTERN = re.compile(r"<[^>]+>.*?</[^>]+>", re.DOTALL)
_SYSTEM_TAG_PATTERN = re.compile(r"\[message_id:\s*\d+\]")

# Roles we capture — skip toolCall, toolResult, session, thinking_level_change, etc.
_CAPTURE_ROLES = {"user", "assistant"}


def _extract_text(content: list[dict[str, Any]]) -> str:
    """Extract plain text from content array, skipping non-text blocks."""
    parts = []
    for block in content:
        if block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _clean_tags(text: str) -> str:
    """Remove system tags and message IDs from captured text."""
    text = _TAG_PATTERN.sub("", text)
    text = _SYSTEM_TAG_PATTERN.sub("", text)
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

    # Both OpenClaw and Claude Code use type="message"
    if data.get("type") != "message":
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
    ):
        super().__init__()
        self._ingest_fn = ingest_fn
        self._loop = loop
        self._label = label
        # Track file read positions: path -> byte offset
        self._positions: dict[str, int] = {}

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

            if messages:
                logger.info(
                    "%s: captured %d messages from %s",
                    self._label, len(messages), Path(path).name,
                )
                fut = asyncio.run_coroutine_threadsafe(
                    self._ingest_fn(messages), self._loop
                )
                def _done(f):
                    try:
                        f.result()
                    except Exception as ex:
                        logger.warning("%s ingest failed for %s: %s", self._label, Path(path).name, ex)
                fut.add_done_callback(_done)

        except Exception as e:
            logger.warning("%s watcher error on %s: %s", self._label, path, e)


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
    ):
        self._sessions_dir = Path(os.path.expanduser(sessions_dir))
        self._ingest_fn = ingest_fn
        self._label = label
        self._recursive = recursive
        self._observer: Observer | None = None

    async def start(self) -> None:
        """Start watching sessions directory."""
        if not self._sessions_dir.exists():
            logger.warning(
                "%s sessions dir not found: %s — watcher disabled",
                self._label, self._sessions_dir,
            )
            return

        loop = asyncio.get_running_loop()
        handler = _SessionFileHandler(self._ingest_fn, loop, self._label)

        # Initialize positions to end of existing files (don't re-ingest history)
        glob_pattern = "**/*.jsonl" if self._recursive else "*.jsonl"
        for jsonl in self._sessions_dir.glob(glob_pattern):
            handler._positions[str(jsonl)] = jsonl.stat().st_size

        self._observer = Observer()
        self._observer.schedule(handler, str(self._sessions_dir), recursive=self._recursive)
        self._observer.daemon = True
        self._observer.start()

        logger.info("%s watcher started: %s (recursive=%s)", self._label, self._sessions_dir, self._recursive)

        # Periodic fallback scans to handle missed FS events
        try:
            while self._observer.is_alive():
                for jsonl in self._sessions_dir.glob(glob_pattern):
                    handler._process_new_lines(str(jsonl))
                await asyncio.sleep(1)
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
