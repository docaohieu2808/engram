"""Tests for FTS5 full-text search index.

Covers: FtsIndex CRUD, search, Vietnamese text, EpisodicStore integration,
and ParallelSearcher integration.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.episodic.fts_index import FtsIndex, FtsResult
from engram.models import EpisodicMemory, MemoryType, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fts(tmp_path: Path) -> FtsIndex:
    """Create a FtsIndex backed by a temp DB file."""
    return FtsIndex(db_path=str(tmp_path / "test_fts.db"))


def _make_episodic_memory(content: str, mem_id: str = "mem-1", priority: int = 5) -> EpisodicMemory:
    from datetime import datetime, timezone
    return EpisodicMemory(
        id=mem_id,
        content=content,
        memory_type=MemoryType.FACT,
        priority=priority,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# FtsIndex unit tests
# ---------------------------------------------------------------------------

class TestFtsIndexCRUD:
    def test_insert_and_search(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "PostgreSQL is the database", "fact")
        results = fts.search("PostgreSQL")
        assert len(results) == 1
        assert results[0].id == "id-1"
        assert results[0].memory_type == "fact"

    def test_insert_updates_existing(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "original content", "fact")
        fts.insert("id-1", "updated content", "decision")
        results = fts.search("updated")
        assert len(results) == 1
        assert results[0].id == "id-1"
        # Old content should not appear
        old_results = fts.search("original")
        assert len(old_results) == 0

    def test_delete_removes_entry(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "something to delete", "fact")
        fts.delete("id-1")
        results = fts.search("delete")
        assert len(results) == 0

    def test_delete_nonexistent_is_noop(self, tmp_path):
        fts = make_fts(tmp_path)
        # Should not raise
        fts.delete("nonexistent-id")

    def test_insert_batch(self, tmp_path):
        fts = make_fts(tmp_path)
        entries = [
            ("id-1", "Redis for caching", "fact"),
            ("id-2", "Kafka for streaming", "fact"),
            ("id-3", "PostgreSQL for storage", "decision"),
        ]
        fts.insert_batch(entries)
        assert len(fts.search("Redis")) == 1
        assert len(fts.search("Kafka")) == 1
        assert len(fts.search("PostgreSQL")) == 1

    def test_insert_batch_empty_is_noop(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert_batch([])  # Should not raise


class TestFtsIndexSearch:
    def test_exact_keyword_match(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "The quick brown fox jumps over the lazy dog", "fact")
        results = fts.search("fox")
        assert len(results) == 1

    def test_multi_keyword_match(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "deploy to production server", "fact")
        fts.insert("id-2", "local development environment", "fact")
        fts.insert("id-3", "production database config", "fact")
        results = fts.search("production")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"id-1", "id-3"}

    def test_no_match_returns_empty(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "unrelated content here", "fact")
        results = fts.search("zebra")
        assert results == []

    def test_empty_query_returns_empty(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "some content", "fact")
        assert fts.search("") == []
        assert fts.search("   ") == []

    def test_limit_respected(self, tmp_path):
        fts = make_fts(tmp_path)
        for i in range(10):
            fts.insert(f"id-{i}", f"keyword content item {i}", "fact")
        results = fts.search("keyword", limit=3)
        assert len(results) <= 3

    def test_invalid_fts_query_returns_empty(self, tmp_path):
        """Lone FTS operators like 'AND' alone should not raise — return empty."""
        fts = make_fts(tmp_path)
        fts.insert("id-1", "some content", "fact")
        # FTS5 treats bare 'AND' as operator — should not crash
        results = fts.search("AND")
        assert isinstance(results, list)

    def test_snippet_included_in_result(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "The database uses PostgreSQL version 15", "fact")
        results = fts.search("PostgreSQL")
        assert len(results) == 1
        assert isinstance(results[0].snippet, str)
        assert len(results[0].snippet) > 0

    def test_vietnamese_text_search(self, tmp_path):
        """FTS5 unicode61 tokenizer should handle Vietnamese characters."""
        fts = make_fts(tmp_path)
        fts.insert("id-vn-1", "Hệ thống sử dụng cơ sở dữ liệu PostgreSQL", "fact")
        fts.insert("id-vn-2", "Triển khai lên môi trường production", "fact")
        results = fts.search("PostgreSQL")
        assert len(results) == 1
        assert results[0].id == "id-vn-1"

    def test_fts_result_namedtuple_fields(self, tmp_path):
        fts = make_fts(tmp_path)
        fts.insert("id-1", "test content for field check", "decision")
        results = fts.search("test")
        assert len(results) == 1
        r = results[0]
        assert r.id == "id-1"
        assert r.memory_type == "decision"
        assert isinstance(r.snippet, str)


# ---------------------------------------------------------------------------
# EpisodicStore integration tests (mocked ChromaDB)
# ---------------------------------------------------------------------------

class TestEpisodicStoreFtsIntegration:
    """Integration tests for FTS5 methods in EpisodicStore using real FtsIndex."""

    def _make_store(self, tmp_path: Path):
        """Build a minimal EpisodicStore with mocked ChromaDB and real FtsIndex."""
        from engram.config import EpisodicConfig, EmbeddingConfig
        from engram.episodic.store import EpisodicStore

        config = EpisodicConfig(path=str(tmp_path / "chroma"))
        embed_config = EmbeddingConfig()

        store = EpisodicStore.__new__(EpisodicStore)
        # Minimal attribute setup without calling __init__ (avoids chromadb I/O)
        store._namespace = "test"
        store.COLLECTION_NAME = "engram_test"
        store._collection = None
        store._embedding_dim = None
        store._on_remember_hook = None
        store._audit = None
        store._decay_enabled = False
        store._default_decay_rate = 0.1
        store._scoring = __import__("engram.config", fromlist=["ScoringConfig"]).ScoringConfig()
        store._embed_model = "gemini/gemini-embedding-001"
        store._client = MagicMock()
        store._fts = FtsIndex(db_path=str(tmp_path / "fts.db"))
        return store

    @pytest.mark.asyncio
    async def test_search_fulltext_returns_memories(self, tmp_path):
        store = self._make_store(tmp_path)
        # Pre-populate FTS index
        store._fts.insert("mem-abc", "PostgreSQL deployment on production", "fact")

        # Mock ChromaDB collection.get
        from datetime import datetime, timezone
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["mem-abc"],
            "documents": ["PostgreSQL deployment on production"],
            "metadatas": [{"memory_type": "fact", "priority": 5,
                           "timestamp": datetime.now(timezone.utc).isoformat(),
                           "entities": "[]", "tags": "[]",
                           "access_count": 0, "decay_rate": 0.1}],
        }
        store._collection = mock_collection

        results = await store.search_fulltext("PostgreSQL")
        assert len(results) == 1
        assert results[0].id == "mem-abc"
        assert "PostgreSQL" in results[0].content

    @pytest.mark.asyncio
    async def test_search_fulltext_no_match(self, tmp_path):
        store = self._make_store(tmp_path)
        store._collection = MagicMock()
        results = await store.search_fulltext("nonexistent_term_xyz")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_fulltext_chromadb_error_returns_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        store._fts.insert("mem-1", "some keyword content", "fact")

        mock_collection = MagicMock()
        mock_collection.get.side_effect = RuntimeError("ChromaDB error")
        store._collection = mock_collection

        # Should not raise, returns empty list
        results = await store.search_fulltext("keyword")
        assert results == []


# ---------------------------------------------------------------------------
# ParallelSearcher FTS integration tests
# ---------------------------------------------------------------------------

class TestParallelSearcherFts:
    def _make_searcher_with_fts(self, fts_results: list[EpisodicMemory]):
        from engram.recall.parallel_search import ParallelSearcher
        from engram.config import RecallPipelineConfig

        episodic = AsyncMock()
        episodic.search = AsyncMock(return_value=[])
        episodic.search_fulltext = AsyncMock(return_value=fts_results)

        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[])

        config = RecallPipelineConfig(fallback_threshold=0.0)
        return ParallelSearcher(episodic, semantic, config)

    @pytest.mark.asyncio
    async def test_fts_results_included_in_fusion(self):
        from engram.recall.parallel_search import ParallelSearcher

        mem = _make_episodic_memory("exact keyword match result", mem_id="fts-1")
        searcher = self._make_searcher_with_fts([mem])

        results = await searcher.search("keyword")
        sources = {r.source for r in results}
        assert "fts" in sources

    @pytest.mark.asyncio
    async def test_fts_source_label(self):
        mem = _make_episodic_memory("unique fts content", mem_id="fts-2")
        searcher = self._make_searcher_with_fts([mem])

        results = await searcher.search("unique")
        fts_results = [r for r in results if r.source == "fts"]
        assert len(fts_results) == 1
        assert fts_results[0].content == "unique fts content"

    @pytest.mark.asyncio
    async def test_fts_failure_does_not_crash_search(self):
        from engram.recall.parallel_search import ParallelSearcher
        from engram.config import RecallPipelineConfig

        episodic = AsyncMock()
        episodic.search = AsyncMock(return_value=[])
        episodic.search_fulltext = AsyncMock(side_effect=Exception("FTS error"))

        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[])

        searcher = ParallelSearcher(episodic, semantic, RecallPipelineConfig(fallback_threshold=0.0))
        # Should not raise
        results = await searcher.search("test")
        assert isinstance(results, list)
