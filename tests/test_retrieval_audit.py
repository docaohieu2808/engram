"""Tests for retrieval audit log."""

import json
import pytest
from pathlib import Path

from engram.retrieval_audit_log import RetrievalAuditLog
from engram.config import RetrievalAuditConfig


class TestRetrievalAuditLog:

    def _make_log(self, tmp_path: Path, enabled: bool = True) -> RetrievalAuditLog:
        config = RetrievalAuditConfig(enabled=enabled, path=str(tmp_path / "audit.jsonl"))
        return RetrievalAuditLog(config)

    def test_log_writes_jsonl(self, tmp_path):
        audit = self._make_log(tmp_path)
        audit.log("test query", results_count=5, top_score=0.89, latency_ms=150, source="semantic")

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["query"] == "test query"
        assert entry["results_count"] == 5
        assert entry["top_score"] == 0.89
        assert entry["latency_ms"] == 150
        assert entry["source"] == "semantic"
        assert "timestamp" in entry

    def test_log_disabled_does_nothing(self, tmp_path):
        audit = self._make_log(tmp_path, enabled=False)
        audit.log("query", results_count=1, top_score=0.5, latency_ms=10, source="test")
        assert not (tmp_path / "audit.jsonl").exists()

    def test_log_with_results(self, tmp_path):
        audit = self._make_log(tmp_path)
        results = [{"id": "mem-1", "score": 0.9, "content": "hello world"}]
        audit.log("query", results_count=1, top_score=0.9, latency_ms=50, source="semantic", results=results)

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert len(entry["results"]) == 1
        assert entry["results"][0]["id"] == "mem-1"

    def test_log_appends(self, tmp_path):
        audit = self._make_log(tmp_path)
        audit.log("q1", results_count=1, top_score=0.5, latency_ms=10, source="a")
        audit.log("q2", results_count=2, top_score=0.7, latency_ms=20, source="b")

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_read_recent(self, tmp_path):
        audit = self._make_log(tmp_path)
        for i in range(5):
            audit.log(f"q{i}", results_count=i, top_score=0.1 * i, latency_ms=10, source="test")

        entries = audit.read_recent(3)
        assert len(entries) == 3
        assert entries[-1]["query"] == "q4"

    def test_read_recent_empty_file(self, tmp_path):
        audit = self._make_log(tmp_path)
        entries = audit.read_recent()
        assert entries == []

    def test_enabled_property(self, tmp_path):
        audit = self._make_log(tmp_path, enabled=True)
        assert audit.enabled is True

    def test_path_property(self, tmp_path):
        audit = self._make_log(tmp_path)
        assert audit.path == tmp_path / "audit.jsonl"
