"""Tests for EntityExtractor LLM-based entity/edge parsing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.capture.extractor import EntityExtractor
from engram.models import SchemaDefinition, NodeDef, EdgeDef


def _make_llm_response(payload: dict):
    msg = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def extractor(sample_schema):
    return EntityExtractor(model="test/model", schema=sample_schema)


async def test_extract_entities(extractor):
    """extract_entities parses nodes and edges from LLM JSON response."""
    payload = {
        "nodes": [
            {"type": "Service", "name": "api-gateway", "attributes": {}},
            {"type": "Team", "name": "platform", "attributes": {}},
        ],
        "edges": [
            {"from_node": "Team:platform", "to_node": "Service:api-gateway", "relation": "owns"},
        ],
    }
    messages = [{"role": "user", "content": "platform team owns api-gateway"}]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_llm_response(payload))):
        result = await extractor.extract_entities(messages)

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    node_names = {n.name for n in result.nodes}
    assert "api-gateway" in node_names
    assert "platform" in node_names


async def test_parse_response_invalid_json(extractor):
    """_parse_response returns empty ExtractionResult for invalid JSON."""
    result = extractor._parse_response("this is not json {{")
    assert result.nodes == []
    assert result.edges == []


def test_chunk_messages(extractor):
    """_chunk_messages splits large message lists with overlap=2."""
    messages = [{"role": "user", "content": str(i)} for i in range(10)]
    # chunk_size=5: first chunk [0..4], next starts at 3 (5-2 overlap)
    chunks = extractor._chunk_messages(messages, chunk_size=5)
    assert len(chunks) > 1
    # Verify overlap: last 2 of chunk N == first 2 of chunk N+1
    assert chunks[0][-2:] == chunks[1][:2]


def test_chunk_messages_no_split(extractor):
    """_chunk_messages returns single chunk when messages fit in chunk_size."""
    messages = [{"role": "user", "content": "hi"}] * 3
    chunks = extractor._chunk_messages(messages, chunk_size=10)
    assert len(chunks) == 1
