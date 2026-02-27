"""Fail-open pending queue for embedding failures.

When embedding fails (e.g. API key expired), messages are enqueued to
~/.engram/pending_queue.jsonl and retried by the scheduler.

JSONL format per line:
  {"content": "...", "timestamp": "ISO8601", "metadata": {...}, "enqueued_at": "ISO8601"}
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("engram")

_DEFAULT_QUEUE_PATH = "~/.engram/pending_queue.jsonl"


class PendingQueue:
    """Thread-safe JSONL file queue for failed embedding retries."""

    def __init__(self, path: str = _DEFAULT_QUEUE_PATH) -> None:
        self._path = Path(path).expanduser()
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def enqueue(
        self,
        content: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one entry to the JSONL queue. Best-effort: logs on failure."""
        entry = {
            "content": content,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as exc:
            logger.error("PendingQueue: failed to write entry: %s", exc)

    def count(self) -> int:
        """Return number of pending entries."""
        if not self._path.exists():
            return 0
        try:
            with self._lock:
                with self._path.open("r", encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    async def drain(self, store: Any) -> tuple[int, int]:
        """Try to remember() each queued item. Returns (success_count, failed_count).

        Successfully stored items are removed. Failed items remain for next drain.
        """
        if not self._path.exists():
            return 0, 0

        with self._lock:
            try:
                raw_lines = self._path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                logger.warning("PendingQueue.drain: cannot read queue: %s", exc)
                return 0, 0

        entries = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("PendingQueue.drain: skipping malformed line: %s", exc)

        if not entries:
            self._clear()
            return 0, 0

        success, failed_entries = 0, []
        for entry in entries:
            content = entry.get("content", "")
            ts_str = entry.get("timestamp")
            metadata = entry.get("metadata") or {}
            timestamp = None
            if ts_str:
                try:
                    timestamp = datetime.fromisoformat(ts_str)
                except ValueError:
                    pass
            try:
                await store.remember(content=content, timestamp=timestamp, metadata=metadata or None)
                success += 1
            except Exception as exc:
                logger.debug("PendingQueue.drain: retry failed for %r: %s", content[:40], exc)
                failed_entries.append(entry)

        # Rewrite queue with only failed entries
        with self._lock:
            try:
                if failed_entries:
                    lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in failed_entries) + "\n"
                    self._path.write_text(lines, encoding="utf-8")
                else:
                    self._path.unlink(missing_ok=True)
            except Exception as exc:
                logger.error("PendingQueue.drain: failed to rewrite queue: %s", exc)

        return success, len(failed_entries)

    def _clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass


# Module-level singleton (lazy init)
_queue: PendingQueue | None = None


def get_pending_queue() -> PendingQueue:
    global _queue
    if _queue is None:
        _queue = PendingQueue()
    return _queue
