"""Tests for benchmark runner."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

from engram.benchmark.runner import BenchmarkRunner, BenchmarkResult
from engram.models import EpisodicMemory, MemoryType


def _make_memory(content: str) -> EpisodicMemory:
    return EpisodicMemory(
        id="mem-1", content=content, memory_type=MemoryType.FACT,
        timestamp=datetime.now(timezone.utc),
    )


class TestBenchmarkResult:

    def test_accuracy_zero_total(self):
        r = BenchmarkResult()
        assert r.accuracy == 0.0

    def test_accuracy_calculation(self):
        r = BenchmarkResult()
        r.total = 10
        r.correct = 8
        assert r.accuracy == 0.8

    def test_to_dict(self):
        r = BenchmarkResult()
        r.total = 5
        r.correct = 3
        d = r.to_dict()
        assert d["total"] == 5
        assert d["accuracy"] == 0.6


class TestBenchmarkRunner:

    def _write_questions(self, tmp_path: Path, questions: list) -> Path:
        p = tmp_path / "questions.json"
        p.write_text(json.dumps(questions))
        return p

    @pytest.mark.asyncio
    async def test_run_all_correct(self, tmp_path):
        questions = [
            {"question": "What database?", "expected_answer_contains": ["PostgreSQL"], "type": "factual"},
            {"question": "What cache?", "expected_answer_contains": ["Redis"], "type": "factual"},
        ]
        qfile = self._write_questions(tmp_path, questions)

        store = AsyncMock()
        store.search = AsyncMock(side_effect=[
            [_make_memory("We use PostgreSQL for storage")],
            [_make_memory("Redis is used for caching")],
        ])

        runner = BenchmarkRunner(store, qfile)
        result = await runner.run()
        assert result.total == 2
        assert result.correct == 2
        assert result.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_run_partial_correct(self, tmp_path):
        questions = [
            {"question": "What DB?", "expected_answer_contains": ["PostgreSQL"], "type": "factual"},
            {"question": "Who?", "expected_answer_contains": ["Trâm"], "type": "pronoun"},
        ]
        qfile = self._write_questions(tmp_path, questions)

        store = AsyncMock()
        store.search = AsyncMock(side_effect=[
            [_make_memory("We use PostgreSQL")],
            [_make_memory("Unknown person")],  # doesn't contain "Trâm"
        ])

        runner = BenchmarkRunner(store, qfile)
        result = await runner.run()
        assert result.total == 2
        assert result.correct == 1
        assert result.by_type["factual"]["correct"] == 1
        assert result.by_type["pronoun"]["correct"] == 0

    @pytest.mark.asyncio
    async def test_run_no_results(self, tmp_path):
        questions = [{"question": "test?", "expected_answer_contains": ["x"], "type": "general"}]
        qfile = self._write_questions(tmp_path, questions)
        store = AsyncMock()
        store.search = AsyncMock(return_value=[])

        runner = BenchmarkRunner(store, qfile)
        result = await runner.run()
        assert result.correct == 0

    def test_file_not_found(self, tmp_path):
        store = AsyncMock()
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(store, tmp_path / "nonexistent.json")

    def test_invalid_format(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"not": "array"}')
        store = AsyncMock()
        with pytest.raises(ValueError):
            BenchmarkRunner(store, p)

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, tmp_path):
        questions = [{"question": "test", "expected_answer_contains": ["postgresql"], "type": "general"}]
        qfile = self._write_questions(tmp_path, questions)
        store = AsyncMock()
        store.search = AsyncMock(return_value=[_make_memory("PostgreSQL is great")])

        runner = BenchmarkRunner(store, qfile)
        result = await runner.run()
        assert result.correct == 1
