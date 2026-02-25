"""Tests for fusion_formatter — memory type grouping and LLM context formatting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.recall.fusion_formatter import format_for_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(content: str, memory_type: str) -> SimpleNamespace:
    """Build a minimal result object with .content and .memory_type."""
    return SimpleNamespace(content=content, memory_type=memory_type)


# ---------------------------------------------------------------------------
# Basic formatting
# ---------------------------------------------------------------------------


def test_single_preference_formatted():
    results = [_result("User prefers dark mode", "preference")]
    out = format_for_llm(results)
    assert out == "[preference] User prefers dark mode"


def test_single_fact_formatted():
    results = [_result("User lives in HCMC", "fact")]
    out = format_for_llm(results)
    assert out == "[fact] User lives in HCMC"


def test_unknown_type_becomes_memory():
    results = [_result("Something random", "random_type")]
    out = format_for_llm(results)
    assert out == "[memory] Something random"


def test_empty_results_returns_empty_string():
    assert format_for_llm([]) == ""


# ---------------------------------------------------------------------------
# Type priority ordering
# ---------------------------------------------------------------------------


def test_mixed_types_sorted_by_priority():
    results = [
        _result("Don't deploy on Friday", "lesson"),
        _result("User lives in HCMC", "fact"),
        _result("User prefers dark mode", "preference"),
    ]
    out = format_for_llm(results)
    lines = out.splitlines()
    assert lines[0].startswith("[preference]")
    assert lines[1].startswith("[fact]")
    assert lines[2].startswith("[lesson]")


def test_decision_comes_after_fact():
    results = [
        _result("Decided to use PostgreSQL", "decision"),
        _result("Python version is 3.12", "fact"),
    ]
    out = format_for_llm(results)
    lines = out.splitlines()
    assert lines[0].startswith("[fact]")
    assert lines[1].startswith("[decision]")


def test_observation_comes_after_context():
    results = [
        _result("Observed high latency", "observation"),
        _result("Session context here", "context"),
    ]
    out = format_for_llm(results)
    lines = out.splitlines()
    assert lines[0].startswith("[context]")
    assert lines[1].startswith("[observation]")


# ---------------------------------------------------------------------------
# Content truncation
# ---------------------------------------------------------------------------


def test_long_content_truncated_at_200_chars():
    long_content = "x" * 300
    results = [_result(long_content, "fact")]
    out = format_for_llm(results)
    # Content part after "[fact] " should be <= 200 chars (truncated with "...")
    content_part = out[len("[fact] "):]
    assert len(content_part) <= 200
    assert content_part.endswith("...")


def test_short_content_not_truncated():
    results = [_result("Short content", "fact")]
    out = format_for_llm(results)
    assert "Short content" in out
    assert "..." not in out


# ---------------------------------------------------------------------------
# max_chars budget
# ---------------------------------------------------------------------------


def test_max_chars_stops_adding_entries():
    # Each entry is ~30 chars; set max_chars = 50 to cut off after first
    results = [
        _result("First preference entry here", "preference"),
        _result("Second preference entry here", "preference"),
        _result("Third preference entry here", "preference"),
    ]
    out = format_for_llm(results, max_chars=50)
    lines = out.splitlines()
    assert len(lines) < 3


def test_max_chars_default_allows_many_entries():
    results = [_result(f"Memory number {i}", "fact") for i in range(20)]
    out = format_for_llm(results, max_chars=2000)
    # Should include multiple entries
    assert out.count("[fact]") > 1


# ---------------------------------------------------------------------------
# Multiple entries per type
# ---------------------------------------------------------------------------


def test_multiple_entries_same_type():
    results = [
        _result("User prefers dark mode", "preference"),
        _result("User prefers vim keybindings", "preference"),
    ]
    out = format_for_llm(results)
    assert out.count("[preference]") == 2


def test_mixed_types_all_appear():
    results = [
        _result("A preference", "preference"),
        _result("A fact", "fact"),
        _result("A lesson", "lesson"),
        _result("A decision", "decision"),
        _result("A context", "context"),
        _result("An observation", "observation"),
    ]
    out = format_for_llm(results, max_chars=5000)
    for t in ("preference", "fact", "lesson", "decision", "context", "observation"):
        assert f"[{t}]" in out
