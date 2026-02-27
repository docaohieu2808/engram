"""Maintenance mixin for EpisodicStore: cleanup_expired, cleanup_dedup, reconcile_stores, stats.

Depends on: episodic_builder helpers, fts_sync helpers.
All methods rely on self._* attributes resolved at runtime via MRO.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from engram.episodic.episodic_builder import _safe_json_list
from engram.episodic.fts_sync import fts_delete, fts_get_all_ids, fts_insert_batch

logger = logging.getLogger("engram")


class _EpisodicMaintenanceMixin:
    """Mixin providing cleanup_expired(), cleanup_dedup(), reconcile_stores(), stats()."""

    async def stats(self) -> dict[str, Any]:
        """Return collection statistics including embedding dimension."""
        await self._ensure_backend()
        count = await self._backend.count()
        await self._detect_embedding_dim()
        result: dict[str, Any] = {
            "count": count,
            "collection": self.COLLECTION_NAME,
            "namespace": self._namespace,
        }
        if self._embedding_dim is not None:
            result["embedding_dim"] = self._embedding_dim
        return result

    async def reconcile_stores(self) -> dict[str, int]:
        """Sync FTS5 index with ChromaDB (source of truth). Returns stats dict.

        D-C3: Periodic reconciliation to fix drift between ChromaDB and FTS5
        since writes to both stores are not atomic.
        """
        if not self._fts:
            return {"orphaned_removed": 0, "missing_added": 0}
        await self._ensure_backend()
        # Fetch all IDs from both stores
        all_chroma = await self._backend.get_many(include=[])
        chroma_ids = set(all_chroma["ids"])
        fts_ids = await fts_get_all_ids(self._fts)
        # Remove FTS entries not present in ChromaDB
        orphaned_fts = fts_ids - chroma_ids
        for oid in orphaned_fts:
            await fts_delete(self._fts, oid)
        # Add FTS entries for ChromaDB IDs missing from FTS
        missing_fts = chroma_ids - fts_ids
        if missing_fts:
            batch = await self._backend.get_many(
                ids=list(missing_fts), include=["documents", "metadatas"]
            )
            entries: list[tuple[str, str, str]] = []
            for mid, doc, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
                mt_str = meta.get("memory_type", "fact") if meta else "fact"
                entries.append((mid, doc or "", mt_str))
            if entries:
                await fts_insert_batch(self._fts, entries)
        return {"orphaned_removed": len(orphaned_fts), "missing_added": len(missing_fts)}

    async def cleanup_expired(self) -> int:
        """Delete all expired memories. Returns count deleted.

        Processes in paginated chunks of 1000 to avoid OOM on large stores.
        """
        _PAGE = 1000
        try:
            await self._ensure_backend()
            total = await self._backend.count()
            if total == 0:
                return 0
        except Exception as e:
            raise RuntimeError(f"Cleanup failed: {e}") from e

        now = datetime.now(timezone.utc)
        expired_ids: list[str] = []

        # Paginate through all memories in chunks
        for offset in range(0, total, _PAGE):
            try:
                result = await self._backend.get_many(
                    include=["metadatas"], limit=_PAGE, offset=offset,
                )
            except Exception:
                break
            for i, mem_id in enumerate(result["ids"]):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                raw_expires = meta.get("expires_at")
                if raw_expires:
                    try:
                        expires_dt = datetime.fromisoformat(raw_expires)
                        if expires_dt.tzinfo is None:
                            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                        if expires_dt < now:
                            expired_ids.append(mem_id)
                    except (ValueError, TypeError):
                        pass

        if expired_ids:
            # Delete in batches too
            for i in range(0, len(expired_ids), _PAGE):
                batch = expired_ids[i:i + _PAGE]
                await self._backend.delete(ids=batch)
                # Sync FTS5 index
                if self._fts:
                    for mid in batch:
                        await fts_delete(self._fts, mid)
            if self._audit:
                self._audit.log_modification(
                    tenant_id=self._namespace, actor="system",
                    mod_type="cleanup_expired", resource_id="",
                    after_value={"deleted_count": len(expired_ids), "ids": expired_ids[:20]},
                    reversible=False,
                    description=f"Cleaned up {len(expired_ids)} expired memories",
                )

        return len(expired_ids)

    async def cleanup_dedup(
        self,
        threshold: float = 0.85,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Retroactively deduplicate existing memories by cosine similarity.

        Paginates through ALL memories, finds near-duplicates above threshold,
        merges entities/tags into the winner (higher priority), and deletes losers.

        Args:
            threshold: Cosine similarity cutoff (0.0-1.0, default 0.85).
            batch_size: Number of memories to load per page.
            dry_run: When True, report what WOULD be merged without any writes.

        Returns:
            {"merged": X, "deleted": Y, "remaining": Z, "dry_run": bool}
        """
        try:
            await self._ensure_backend()
            total = await self._backend.count()
            if total == 0:
                return {"merged": 0, "deleted": 0, "remaining": 0, "dry_run": dry_run}
        except Exception as e:
            raise RuntimeError(f"cleanup_dedup failed: {e}") from e

        processed: set[str] = set()
        to_delete: list[str] = []
        merged_count = 0

        # Paginate through all memories to fetch IDs + embeddings
        for offset in range(0, total, batch_size):
            try:
                page = await self._backend.get_many(
                    include=["embeddings", "metadatas", "documents"],
                    limit=batch_size,
                    offset=offset,
                )
            except Exception as e:
                logger.warning("cleanup_dedup: failed to fetch page offset=%d: %s", offset, e)
                break

            page_ids = page.get("ids", [])
            _raw_emb = page.get("embeddings")
            page_embeddings = _raw_emb if _raw_emb is not None else []
            _raw_meta = page.get("metadatas")
            page_metadatas = _raw_meta if _raw_meta is not None else []

            for i, mem_id in enumerate(page_ids):
                if mem_id in processed:
                    continue

                embedding = page_embeddings[i] if i < len(page_embeddings) else None
                if embedding is None or (hasattr(embedding, '__len__') and len(embedding) == 0):
                    processed.add(mem_id)
                    continue

                # Query top-N similar memories for this embedding
                try:
                    n_query = min(20, total)  # Use cached total instead of per-iteration count()
                    sim_result = await self._backend.query(
                        query_embeddings=[embedding],
                        n_results=n_query,
                        include=["metadatas", "distances"],
                    )
                except Exception as e:
                    logger.debug("cleanup_dedup: query failed for %s: %s", mem_id[:8], e)
                    processed.add(mem_id)
                    continue

                if not sim_result["ids"] or not sim_result["ids"][0]:
                    processed.add(mem_id)
                    continue

                winner_meta = page_metadatas[i] if i < len(page_metadatas) else {}
                winner_entities = set(_safe_json_list(winner_meta.get("entities", "[]")))
                winner_tags = set(_safe_json_list(winner_meta.get("tags", "[]")))
                winner_priority = int(winner_meta.get("priority", 5) or 5)
                dups_for_this: list[str] = []

                for j, candidate_id in enumerate(sim_result["ids"][0]):
                    if candidate_id == mem_id or candidate_id in processed:
                        continue
                    distance = sim_result["distances"][0][j]
                    similarity = 1.0 - (distance / 2.0)
                    if similarity < threshold:
                        continue

                    # This candidate is a near-duplicate of mem_id — absorb it
                    cand_meta = sim_result["metadatas"][0][j] if sim_result.get("metadatas") else {}
                    cand_entities = set(_safe_json_list(cand_meta.get("entities", "[]")))
                    cand_tags = set(_safe_json_list(cand_meta.get("tags", "[]")))
                    cand_priority = int(cand_meta.get("priority", 5) or 5)

                    winner_entities |= cand_entities
                    winner_tags |= cand_tags
                    winner_priority = max(winner_priority, cand_priority)
                    dups_for_this.append(candidate_id)
                    processed.add(candidate_id)

                if dups_for_this:
                    merged_count += 1
                    to_delete.extend(dups_for_this)
                    if not dry_run:
                        try:
                            await self._backend.update(
                                ids=[mem_id],
                                metadatas=[{
                                    "entities": json.dumps(sorted(winner_entities)),
                                    "tags": json.dumps(sorted(winner_tags)),
                                    "priority": winner_priority,
                                }],
                            )
                        except Exception as e:
                            logger.warning(
                                "cleanup_dedup: update winner %s failed: %s", mem_id[:8], e
                            )

                    logger.debug(
                        "cleanup_dedup: %s absorbs %d duplicate(s) (sim>=%.2f)",
                        mem_id[:8], len(dups_for_this), threshold,
                    )

                processed.add(mem_id)

            processed_total = len(processed)
            if processed_total % 100 == 0 or offset + batch_size >= total:
                logger.info(
                    "cleanup_dedup: processed %d/%d memories, %d duplicates found so far",
                    processed_total, total, len(to_delete),
                )

        # Delete all losers
        deleted_count = 0
        if to_delete and not dry_run:
            _BATCH = 500
            for i in range(0, len(to_delete), _BATCH):
                batch = to_delete[i:i + _BATCH]
                try:
                    await self._backend.delete(ids=batch)
                    deleted_count += len(batch)
                except Exception as e:
                    logger.warning("cleanup_dedup: delete batch failed: %s", e)
                # Remove from FTS index
                for dup_id in batch:
                    await fts_delete(self._fts, dup_id)
        elif dry_run:
            deleted_count = len(to_delete)

        remaining = await self._backend.count()

        if not dry_run and self._audit and to_delete:
            self._audit.log_modification(
                tenant_id=self._namespace, actor="system",
                mod_type="cleanup_dedup", resource_id="",
                after_value={"merged_groups": merged_count, "deleted": deleted_count},
                reversible=False,
                description=f"Retroactive dedup: merged {merged_count} groups, deleted {deleted_count} duplicates",
            )

        logger.info(
            "cleanup_dedup complete: merged=%d deleted=%d remaining=%d dry_run=%s",
            merged_count, deleted_count, remaining, dry_run,
        )
        return {"merged": merged_count, "deleted": deleted_count, "remaining": remaining, "dry_run": dry_run}
