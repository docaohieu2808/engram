"""Tests for episodic search helpers: temporal and filtered queries."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from engram.episodic.search import filtered_search, temporal_search
from engram.models import EpisodicMemory, MemoryType


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    return store


async def test_temporal_search(mock_store):
    """temporal_search passes timestamp filters to store.search."""
    await temporal_search(mock_store, "deploy", start_date="2024-01-01", end_date="2024-12-31")
    mock_store.search.assert_called_once()
    _, kwargs = mock_store.search.call_args
    assert kwargs["filters"] is not None
    assert "$and" in kwargs["filters"]


async def test_filtered_search_by_type(mock_store):
    """filtered_search passes memory_type filter to store.search."""
    await filtered_search(mock_store, "error logs", memory_type="error")
    mock_store.search.assert_called_once()
    _, kwargs = mock_store.search.call_args
    assert kwargs["filters"] == {"memory_type": {"$eq": "error"}}
