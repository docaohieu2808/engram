"""Retrieval audit log — JSONL logger for memory retrieval operations.

Logs query, results count, top score, latency, and source for each recall.
Append-only JSONL format for easy analysis.

Rotation: files are rotated at _MAX_BYTES (default 10 MB), keeping up to
_MAX_BACKUPS numbered rotations (.1, .2, .3).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engram.config import RetrievalAuditConfig

logger = logging.getLogger("engram")

# Log rotation defaults (M17)
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_MAX_BACKUPS = 3                # .1 .2 .3 kept


def _rotate(path: Path) -> None:
    """Rotate log files: .3 dropped, .2 → .3, .1 → .2, base → .1."""
    # Drop oldest backup
    oldest = Path(f"{path}.{_MAX_BACKUPS}")
    if oldest.exists():
        oldest.unlink(missing_ok=True)
    # Shift existing backups
    for i in range(_MAX_BACKUPS - 1, 0, -1):
        src = Path(f"{path}.{i}")
        dst = Path(f"{path}.{i + 1}")
        if src.exists():
            src.rename(dst)
    # Rotate active log to .1
    if path.exists():
        path.rename(Path(f"{path}.1"))


class RetrievalAuditLog:
    """Log all memory retrievals to JSONL file for debugging and analysis."""

    def __init__(self, config: RetrievalAuditConfig | None = None):
        self._config = config or RetrievalAuditConfig()
        self._path = Path(os.path.expanduser(self._config.path))

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self,
        query: str,
        results_count: int,
        top_score: float,
        latency_ms: int,
        source: str,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a retrieval log entry as JSONL. Rotates file when size exceeds limit."""
        if not self._config.enabled:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "results_count": results_count,
            "top_score": round(top_score, 4),
            "latency_ms": latency_ms,
            "source": source,
        }
        if results:
            # Store only id, score, snippet for each result
            entry["results"] = [
                {"id": r.get("id", ""), "score": round(r.get("score", 0), 4),
                 "snippet": r.get("content", "")[:100]}
                for r in results[:10]
            ]

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Rotate before write if file exceeds max size (M17)
            if self._path.exists() and self._path.stat().st_size >= _MAX_BYTES:
                _rotate(self._path)
            with open(self._path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to write retrieval audit log: %s", e)

    def read_recent(self, n: int = 100) -> list[dict[str, Any]]:
        """Read the most recent N log entries without loading the entire file (M17)."""
        if not self._path.exists():
            return []
        try:
            # Tail-read: seek backwards to find last N lines efficiently
            entries: list[str] = []
            with open(self._path, "rb") as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                if file_size == 0:
                    return []
                # Read in chunks from end until we have enough lines
                chunk_size = 8192
                pos = file_size
                buf = b""
                while pos > 0 and len(entries) <= n:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    buf = f.read(read_size) + buf
                    lines = buf.split(b"\n")
                    # Keep the incomplete first line in buf for next iteration
                    buf = lines[0]
                    # Collect complete lines from end (skip empty)
                    entries = [l.decode("utf-8", errors="replace") for l in lines[1:] if l.strip()] + entries
                    if len(entries) >= n:
                        break
                # Handle remaining buffer (first partial line)
                if buf.strip():
                    entries = [buf.decode("utf-8", errors="replace")] + entries
            recent = entries[-n:] if len(entries) > n else entries
            return [json.loads(line) for line in recent if line.strip()]
        except Exception as e:
            logger.debug("Failed to read retrieval audit log: %s", e)
            return []
