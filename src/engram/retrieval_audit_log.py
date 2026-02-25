"""Retrieval audit log — JSONL logger for memory retrieval operations.

Logs query, results count, top score, latency, and source for each recall.
Append-only JSONL format for easy analysis.
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
        """Append a retrieval log entry as JSONL."""
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
            with open(self._path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to write retrieval audit log: %s", e)

    def read_recent(self, n: int = 100) -> list[dict[str, Any]]:
        """Read the most recent N log entries."""
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text().strip().split("\n")
            recent = lines[-n:] if len(lines) > n else lines
            return [json.loads(line) for line in recent if line.strip()]
        except Exception as e:
            logger.debug("Failed to read retrieval audit log: %s", e)
            return []
