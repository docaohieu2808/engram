"""Tests for MemoryExtractor — LLM-powered episodic memory extraction.

Tests extraction prompt formatting, JSON response parsing, priority filtering,
dedup logic, and fail-open behavior on LLM errors.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.memory_extractor import MemoryExtractor, EXTRACTION_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str):
    """Build a minimal litellm-style response object."""
    msg = SimpleNamespace(content=content, role="assistant")
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def extractor():
    return MemoryExtractor(model="test/model")


# ---------------------------------------------------------------------------
# Extraction prompt formatting
# ---------------------------------------------------------------------------

def test_extraction_prompt_contains_placeholders():
    """EXTRACTION_PROMPT has {user_msg} and {assistant_msg} placeholders."""
    assert "{user_msg}" in EXTRACTION_PROMPT
    assert "{assistant_msg}" in EXTRACTION_PROMPT


def test_extraction_prompt_format():
    """EXTRACTION_PROMPT formats correctly with user/assistant content."""
    prompt = EXTRACTION_PROMPT.format(user_msg="hello", assistant_msg="world")
    assert "hello" in prompt
    assert "world" in prompt


# ---------------------------------------------------------------------------
# extract() — LLM call and response handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_valid_items(extractor):
    """extract() returns parsed items when LLM returns valid JSON array."""
    items_json = json.dumps([
        {"content": "Decision to use PostgreSQL", "type": "decision", "priority": 8},
        {"content": "Deploy on Fridays is forbidden", "type": "preference", "priority": 7},
    ])
    fake_response = _make_llm_response(items_json)

    with patch("engram.memory_extractor.litellm.acompletion", new_callable=AsyncMock, return_value=fake_response):
        result = await extractor.extract("We chose PostgreSQL", "Good choice, robust DB")

    assert len(result) == 2
    types = {r["type"] for r in result}
    assert "decision" in types
    assert "preference" in types


@pytest.mark.asyncio
async def test_extract_returns_empty_on_llm_error(extractor):
    """extract() returns empty list when LLM call raises (fail-open)."""
    with patch("engram.memory_extractor.litellm.acompletion", side_effect=RuntimeError("API down")):
        result = await extractor.extract("user msg", "assistant msg")

    assert result == []


@pytest.mark.asyncio
async def test_extract_caps_input_length(extractor):
    """extract() caps user_msg at 2000 chars and assistant_msg at 3000 chars."""
    long_user = "x" * 5000
    long_assistant = "y" * 6000
    captured_prompt = []

    async def capture(model, messages, **kwargs):
        captured_prompt.append(messages[0]["content"])
        return _make_llm_response("[]")

    with patch("engram.memory_extractor.litellm.acompletion", side_effect=capture):
        await extractor.extract(long_user, long_assistant)

    assert captured_prompt
    prompt_text = captured_prompt[0]
    # user_msg capped at 2000 x's
    assert "x" * 2000 in prompt_text
    assert "x" * 2001 not in prompt_text
    # assistant_msg capped at 3000 y's
    assert "y" * 3000 in prompt_text
    assert "y" * 3001 not in prompt_text


# ---------------------------------------------------------------------------
# _parse() — JSON response parsing
# ---------------------------------------------------------------------------

def test_parse_clean_json_array(extractor):
    """_parse() handles clean JSON array with valid items."""
    content = json.dumps([
        {"content": "Use Python 3.12", "type": "decision", "priority": 7},
    ])
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Use Python 3.12"
    assert result[0]["type"] == "decision"
    assert result[0]["priority"] == 7


def test_parse_strips_markdown_fences(extractor):
    """_parse() strips ```json ... ``` fences before parsing."""
    content = '```json\n[{"content": "Redis for caching", "type": "decision", "priority": 8}]\n```'
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Redis for caching"


def test_parse_filters_low_priority_items(extractor):
    """_parse() drops items with priority < 4."""
    content = json.dumps([
        {"content": "Important decision", "type": "decision", "priority": 7},
        {"content": "Trivial note", "type": "fact", "priority": 2},
        {"content": "Another noise", "type": "fact", "priority": 1},
    ])
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Important decision"


def test_parse_filters_exactly_priority_four(extractor):
    """_parse() keeps items with priority == 4 (threshold is < 4)."""
    content = json.dumps([
        {"content": "Borderline item", "type": "fact", "priority": 4},
    ])
    result = extractor._parse(content)
    assert len(result) == 1


def test_parse_normalizes_unknown_type(extractor):
    """_parse() coerces unknown type to 'fact'."""
    content = json.dumps([
        {"content": "Some item", "type": "random_type", "priority": 5},
    ])
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["type"] == "fact"


def test_parse_handles_wrapped_object_items_key(extractor):
    """_parse() unwraps { 'items': [...] } wrapper format."""
    content = json.dumps({"items": [
        {"content": "Fact A", "type": "fact", "priority": 6},
    ]})
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Fact A"


def test_parse_handles_wrapped_object_memories_key(extractor):
    """_parse() unwraps { 'memories': [...] } wrapper format."""
    content = json.dumps({"memories": [
        {"content": "Memory B", "type": "decision", "priority": 8},
    ]})
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Memory B"


def test_parse_handles_single_object_wrapped_in_array(extractor):
    """_parse() wraps single dict into array."""
    content = json.dumps({"content": "Single fact", "type": "fact", "priority": 5})
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Single fact"


def test_parse_returns_empty_on_invalid_json(extractor):
    """_parse() returns empty list for invalid JSON."""
    result = extractor._parse("this is not json {{{{")
    assert result == []


def test_parse_returns_empty_on_empty_string(extractor):
    """_parse() returns empty list for empty string."""
    result = extractor._parse("")
    assert result == []


def test_parse_skips_items_with_empty_content(extractor):
    """_parse() skips items where content is empty or missing."""
    content = json.dumps([
        {"content": "", "type": "fact", "priority": 6},
        {"content": "   ", "type": "fact", "priority": 7},
        {"type": "decision", "priority": 8},  # no content key
    ])
    result = extractor._parse(content)
    assert result == []


def test_parse_handles_invalid_priority_gracefully(extractor):
    """_parse() defaults to priority 5 when priority value is non-numeric."""
    content = json.dumps([
        {"content": "Some item", "type": "fact", "priority": "high"},
    ])
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["priority"] == 5


def test_parse_all_valid_types(extractor):
    """_parse() accepts all valid type values without normalization."""
    valid_types = ["fact", "decision", "preference", "error", "todo", "workflow"]
    items_json = json.dumps([
        {"content": f"Item for {t}", "type": t, "priority": 5}
        for t in valid_types
    ])
    result = extractor._parse(items_json)
    assert len(result) == len(valid_types)
    result_types = {r["type"] for r in result}
    assert result_types == set(valid_types)


def test_parse_skips_non_dict_items(extractor):
    """_parse() skips non-dict elements in the array."""
    content = json.dumps([
        "just a string",
        42,
        None,
        {"content": "Valid item", "type": "fact", "priority": 6},
    ])
    result = extractor._parse(content)
    assert len(result) == 1
    assert result[0]["content"] == "Valid item"


def test_parse_non_list_top_level_returns_empty(extractor):
    """_parse() returns empty for a JSON value that is not array or object."""
    result = extractor._parse('"just a string"')
    assert result == []


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

def test_default_model_is_gemini_flash():
    """MemoryExtractor defaults to gemini-2.0-flash when no model specified."""
    ext = MemoryExtractor()
    assert "gemini" in ext._model
    assert "flash" in ext._model


def test_custom_model_is_used():
    """MemoryExtractor uses the model passed to constructor."""
    ext = MemoryExtractor(model="openai/gpt-4o-mini")
    assert ext._model == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_extract_passes_model_to_litellm(extractor):
    """extract() passes the configured model to litellm.acompletion."""
    called_with = {}

    async def capture(model, messages, **kwargs):
        called_with["model"] = model
        return _make_llm_response("[]")

    with patch("engram.memory_extractor.litellm.acompletion", side_effect=capture):
        await extractor.extract("user", "assistant")

    assert called_with["model"] == "test/model"
