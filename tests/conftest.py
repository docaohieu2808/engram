"""Shared pytest fixtures for engram test suite."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engram.config import EmbeddingConfig, EpisodicConfig, SemanticConfig
from engram.episodic.store import EpisodicStore
from engram.models import EdgeDef, NodeDef, SchemaDefinition
from engram.semantic.graph import SemanticGraph

# Fixed 384-dim vector for deterministic embedding tests
_FIXED_EMBEDDING = [0.1] * 384


def _mock_embeddings(_model: str, texts: list[str]) -> list[list[float]]:
    return [_FIXED_EMBEDDING for _ in texts]


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Temporary directory for config files."""
    return tmp_path / "config"


@pytest.fixture
def mock_embeddings():
    """Patch _get_embeddings to return fixed 384-dim vectors."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings) as m:
        yield m


@pytest.fixture
def episodic_store(tmp_path, mock_embeddings):
    """EpisodicStore backed by tmp ChromaDB path with mocked embeddings."""
    config = EpisodicConfig(path=str(tmp_path / "episodic"))
    embed_config = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    return EpisodicStore(config=config, embedding_config=embed_config)


@pytest.fixture
def semantic_graph(tmp_path):
    """SemanticGraph backed by tmp SQLite path."""
    config = SemanticConfig(path=str(tmp_path / "semantic.db"))
    return SemanticGraph(config=config)


@pytest.fixture
def sample_schema():
    """SchemaDefinition with 2 node types and 1 edge type."""
    return SchemaDefinition(
        nodes=[
            NodeDef(name="Service", description="A software service"),
            NodeDef(name="Team", description="An engineering team"),
        ],
        edges=[
            EdgeDef(name="owns", from_types=["Team"], to_types=["Service"]),
        ],
    )
