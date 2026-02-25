"""Benchmark runner — evaluate memory recall quality against test questions.

Loads a JSON question file, runs each question through the recall pipeline,
and checks if expected answer fragments appear in results.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from engram.episodic.store import EpisodicStore

logger = logging.getLogger("engram")


class BenchmarkResult:
    """Holds benchmark run results."""

    def __init__(self) -> None:
        self.total: int = 0
        self.correct: int = 0
        self.by_type: dict[str, dict[str, int]] = {}
        self.details: list[dict[str, Any]] = []

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "by_type": self.by_type,
        }


class BenchmarkRunner:
    """Run memory quality benchmark against a question set.

    Questions format (JSON array):
    [
        {
            "question": "Lần cuối tôi gặp Trâm là khi nào?",
            "expected_answer_contains": ["2026-02-04", "Q7"],
            "type": "temporal"
        }
    ]
    """

    def __init__(self, store: "EpisodicStore", questions_file: str | Path):
        self._store = store
        self._questions = self._load_questions(questions_file)

    @staticmethod
    def _load_questions(path: str | Path) -> list[dict[str, Any]]:
        """Load questions from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark questions file not found: {p}")
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Questions file must contain a JSON array")
        return data

    async def run(self) -> BenchmarkResult:
        """Run all benchmark questions and return results."""
        result = BenchmarkResult()

        for q in self._questions:
            question = q.get("question", "")
            expected = q.get("expected_answer_contains", [])
            q_type = q.get("type", "general")

            # Initialize type tracking
            if q_type not in result.by_type:
                result.by_type[q_type] = {"total": 0, "correct": 0}

            result.total += 1
            result.by_type[q_type]["total"] += 1

            # Search for the question
            start = time.monotonic()
            memories = await self._store.search(question, limit=10)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Check if any result contains all expected fragments
            all_content = " ".join(m.content for m in memories)
            is_correct = all(
                fragment.lower() in all_content.lower()
                for fragment in expected
            ) if expected else False

            if is_correct:
                result.correct += 1
                result.by_type[q_type]["correct"] += 1

            result.details.append({
                "question": question,
                "type": q_type,
                "correct": is_correct,
                "latency_ms": elapsed_ms,
                "results_count": len(memories),
            })

        return result
