"""Tests for schema loader: builtin loading, error handling, prompt generation."""

from __future__ import annotations

import pytest

from engram.schema.loader import load_schema, schema_to_prompt


def test_load_builtin_devops():
    """Loading 'devops' builtin schema succeeds with nodes and edges."""
    schema = load_schema("devops")
    assert len(schema.nodes) > 0
    assert len(schema.edges) > 0


def test_load_not_found():
    """Loading unknown schema raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_schema("nonexistent-schema-xyz")


def test_schema_to_prompt(sample_schema):
    """schema_to_prompt output contains all node and edge names."""
    prompt = schema_to_prompt(sample_schema)
    assert "Service" in prompt
    assert "Team" in prompt
    assert "owns" in prompt
