"""SQLite-backed embedding queue for graceful degradation on embedding API failure.

Memories enqueued here when Gemini API fails; retried by background scheduler.
Schema: id, content, metadata_json, created_at, retry_count, last_error, status.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("engram")

_DEFAULT_DB_PATH = "~/.engram/embedding_queue.db"
_MAX_RETRIES = 5
_BATCH_SIZE = 50

_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS embedding_queue ("
    "id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', "
    "created_at TEXT NOT NULL, retry_count INTEGER NOT NULL DEFAULT 0, "
    "last_error TEXT, status TEXT NOT NULL DEFAULT 'pending')"
)
_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_eq_status ON embedding_queue(status)"


class EmbeddingQueue:
    """Thread-safe SQLite queue for failed embedding retries."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path).expanduser()
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_IDX)
                conn.commit()
            finally:
                conn.close()

    def enqueue(self, memory_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory to the embedding queue (idempotent on memory_id)."""
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO embedding_queue (id, content, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (memory_id, content, meta_json, now),
                )
                conn.commit()
            except Exception as exc:
                logger.error("EmbeddingQueue.enqueue failed: %s", exc)
            finally:
                conn.close()

    def dequeue_batch(self, limit: int = _BATCH_SIZE) -> list[dict[str, Any]]:
        """Return oldest pending items for processing. Items stay as 'pending' until marked."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, content, metadata_json, retry_count FROM embedding_queue "
                    "WHERE status = 'pending' AND retry_count < ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (_MAX_RETRIES, limit),
                ).fetchall()
                return [
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "metadata": json.loads(r["metadata_json"] or "{}"),
                        "retry_count": r["retry_count"],
                    }
                    for r in rows
                ]
            finally:
                conn.close()

    def mark_done(self, memory_id: str) -> None:
        """Remove entry from queue after successful embedding."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE embedding_queue SET status = 'done' WHERE id = ?", (memory_id,)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_failed(self, memory_id: str, error: str) -> None:
        """Increment retry_count and store error. After MAX_RETRIES, status → 'failed'."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE embedding_queue "
                    "SET retry_count = retry_count + 1, last_error = ?, "
                    "    status = CASE WHEN retry_count + 1 >= ? THEN 'failed' ELSE 'pending' END "
                    "WHERE id = ?",
                    (error[:500], _MAX_RETRIES, memory_id),
                )
                conn.commit()
            finally:
                conn.close()

    def pending_count(self) -> int:
        """Return number of entries with status='pending'."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM embedding_queue WHERE status = 'pending'"
                ).fetchone()
                return row[0] if row else 0
            finally:
                conn.close()

    def queue_status(self) -> dict[str, Any]:
        """Return summary stats for CLI/MCP display."""
        with self._lock:
            conn = self._connect()
            try:
                counts = dict(conn.execute("SELECT status, COUNT(*) FROM embedding_queue GROUP BY status").fetchall())
                oldest = conn.execute(
                    "SELECT created_at, last_error FROM embedding_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
                return {
                    "pending": counts.get("pending", 0), "done": counts.get("done", 0),
                    "failed": counts.get("failed", 0),
                    "oldest_pending": oldest["created_at"] if oldest else None,
                    "last_error": oldest["last_error"] if oldest else None,
                }
            finally:
                conn.close()


async def process_embedding_queue(store: Any, queue: "EmbeddingQueue | None" = None) -> dict[str, Any]:
    """Drain pending items: embed → ChromaDB add → mark_done. Called by scheduler."""
    import asyncio, sys
    if queue is None:
        queue = get_embedding_queue()
    batch = queue.dequeue_batch()
    if not batch:
        return {"processed": 0, "failed": 0, "skipped": True}

    def _get_embeddings(*args, **kwargs):
        return sys.modules["engram.episodic.store"]._get_embeddings(*args, **kwargs)

    processed = failed = 0
    for item in batch:
        mid, content, metadata = item["id"], item["content"], item["metadata"]
        try:
            await store._ensure_backend()
            await store._detect_embedding_dim()
            embeddings = await asyncio.to_thread(
                _get_embeddings, store._embed_model, [content], store._embedding_dim
            )
            await store._backend.add(id=mid, embedding=embeddings[0], content=content, metadata=metadata)
            queue.mark_done(mid)
            processed += 1
            logger.info("EmbeddingQueue: processed %s", mid[:8])
        except Exception as exc:
            queue.mark_failed(mid, str(exc))
            failed += 1
            logger.warning("EmbeddingQueue: retry failed for %s: %s", mid[:8], exc)

    return {"processed": processed, "failed": failed}


# Module-level singleton (lazy init)
_queue: EmbeddingQueue | None = None


def get_embedding_queue() -> EmbeddingQueue:
    """Return the global EmbeddingQueue singleton."""
    global _queue
    if _queue is None:
        _queue = EmbeddingQueue()
    return _queue
