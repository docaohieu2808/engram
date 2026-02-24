"""Tests for CLI commands using Typer's CliRunner with mocked stores."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from engram.cli import app
from engram.models import EpisodicMemory, MemoryType, SemanticNode, SemanticEdge

runner = CliRunner()


# --- Helpers ---

def _make_memory(content: str = "test fact") -> EpisodicMemory:
    return EpisodicMemory(id="abc123def456", content=content, memory_type=MemoryType.FACT)


def _make_node(type: str = "Service", name: str = "api") -> SemanticNode:
    return SemanticNode(type=type, name=name)


def _make_edge(from_node: str = "Team:devops", to_node: str = "Service:api", relation: str = "owns") -> SemanticEdge:
    return SemanticEdge(from_node=from_node, to_node=to_node, relation=relation)


# --- Fixtures ---

@pytest.fixture
def mock_episodic_store():
    store = AsyncMock()
    store.remember = AsyncMock(return_value="abc123def456")
    store.search = AsyncMock(return_value=[_make_memory()])
    store.cleanup_expired = AsyncMock(return_value=2)
    store.stats = AsyncMock(return_value={"count": 5})
    return store


@pytest.fixture
def mock_graph():
    graph = AsyncMock()
    graph.query = AsyncMock(return_value=[_make_node()])
    graph.get_nodes = AsyncMock(return_value=[_make_node()])
    graph.get_related = AsyncMock(return_value={})
    graph.stats = AsyncMock(return_value={"node_count": 3, "edge_count": 1, "node_types": {"Service": 3}})
    graph.add_node = AsyncMock(return_value=True)
    graph.add_edge = AsyncMock(return_value=True)
    graph.remove_node = AsyncMock(return_value=True)
    graph.remove_edge = AsyncMock(return_value=True)
    return graph


@pytest.fixture
def mock_engine():
    engine = AsyncMock()
    engine.think = AsyncMock(return_value="42 is the answer")
    engine.summarize = AsyncMock(return_value="Summary of recent memories")
    return engine


# --- remember tests ---

def test_remember_basic(mock_episodic_store, mock_graph):
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["remember", "Hello world"])
    assert result.exit_code == 0
    assert "Remembered" in result.output
    mock_episodic_store.remember.assert_called_once()


def test_remember_with_type(mock_episodic_store, mock_graph):
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["remember", "Deploy to prod", "--type", "decision"])
    assert result.exit_code == 0
    assert "decision" in result.output


def test_remember_with_tags(mock_episodic_store, mock_graph):
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["remember", "Tagged memory", "--tags", "deploy,prod"])
    assert result.exit_code == 0
    assert "tags=" in result.output or "Remembered" in result.output


def test_remember_with_expires(mock_episodic_store, mock_graph):
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["remember", "Temporary fact", "--expires", "7d"])
    assert result.exit_code == 0
    assert "expires=" in result.output or "Remembered" in result.output


def test_remember_invalid_type(mock_episodic_store, mock_graph):
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["remember", "Bad type memory", "--type", "nonexistent_type"])
    assert result.exit_code != 0


# --- recall tests ---

def test_recall_basic(mock_episodic_store, mock_graph):
    mock_graph.query = AsyncMock(return_value=[])
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["recall", "test query"])
    assert result.exit_code == 0
    assert "test fact" in result.output


def test_recall_with_limit(mock_episodic_store, mock_graph):
    mock_graph.query = AsyncMock(return_value=[])
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["recall", "test query", "--limit", "3"])
    assert result.exit_code == 0
    call_kwargs = mock_episodic_store.search.call_args
    assert call_kwargs.kwargs.get("limit") == 3 or call_kwargs.args[1] == 3


def test_recall_with_type_filter(mock_episodic_store, mock_graph):
    mock_graph.query = AsyncMock(return_value=[])
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["recall", "test", "--type", "fact"])
    assert result.exit_code == 0


def test_recall_with_tags(mock_episodic_store, mock_graph):
    mock_graph.query = AsyncMock(return_value=[])
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["recall", "test", "--tags", "deploy"])
    assert result.exit_code == 0


def test_recall_no_results(mock_episodic_store, mock_graph):
    mock_episodic_store.search = AsyncMock(return_value=[])
    mock_graph.query = AsyncMock(return_value=[])
    with patch("engram.cli.episodic._get_episodic", return_value=mock_episodic_store), \
         patch("engram.cli.episodic._get_semantic", return_value=mock_graph):
        result = runner.invoke(app, ["recall", "nothing matches"])
    assert result.exit_code == 0
    assert "No memories found" in result.output


# --- think tests ---

def test_think_basic(mock_engine):
    with patch("engram.reasoning.engine.ReasoningEngine.think", new=AsyncMock(return_value="42 is the answer")), \
         patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.semantic.create_graph", return_value=MagicMock()), \
         patch("engram.cli.reasoning.run_async", return_value="42 is the answer"):
        result = runner.invoke(app, ["think", "What is life?"])
    assert result.exit_code == 0
    assert "42 is the answer" in result.output


# --- add node / add edge tests ---

def test_add_node(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["add", "node", "my-service", "--type", "Service"])
    assert result.exit_code == 0
    assert "Added" in result.output or "Updated" in result.output
    mock_graph.add_node.assert_called_once()


def test_add_node_already_exists(mock_graph):
    mock_graph.add_node = AsyncMock(return_value=False)
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["add", "node", "existing", "--type", "Service"])
    assert result.exit_code == 0
    assert "Updated" in result.output


def test_add_edge(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["add", "edge", "Team:devops", "Service:api", "--relation", "owns"])
    assert result.exit_code == 0
    assert "Added" in result.output or "Updated" in result.output
    mock_graph.add_edge.assert_called_once()


# --- remove node / remove edge tests ---

def test_remove_node_found(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["remove", "node", "Service:api"])
    assert result.exit_code == 0
    assert "Removed" in result.output


def test_remove_node_not_found(mock_graph):
    mock_graph.remove_node = AsyncMock(return_value=False)
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["remove", "node", "Service:missing"])
    assert result.exit_code == 0
    assert "Not found" in result.output


def test_remove_edge_found(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["remove", "edge", "Team:devops|owns|Service:api"])
    assert result.exit_code == 0
    assert "Removed" in result.output


def test_remove_edge_not_found(mock_graph):
    mock_graph.remove_edge = AsyncMock(return_value=False)
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["remove", "edge", "missing-key"])
    assert result.exit_code == 0
    assert "Not found" in result.output


# --- query tests ---

def test_query_by_keyword(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["query", "api"])
    assert result.exit_code == 0
    mock_graph.query.assert_called_once()


def test_query_by_type(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["query", "--type", "Service"])
    assert result.exit_code == 0


def test_query_no_keyword(mock_graph):
    with patch("engram.cli.semantic._get_graph", return_value=mock_graph):
        result = runner.invoke(app, ["query"])
    assert result.exit_code == 0
    mock_graph.get_nodes.assert_called_once()


# --- status tests ---
# _get_episodic/_get_graph are closures in system.register(); patch at class level.

def test_status_command(mock_episodic_store, mock_graph):
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.episodic.store.EpisodicStore.stats", mock_episodic_store.stats), \
         patch("engram.semantic.create_graph", return_value=mock_graph):
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Episodic Memory" in result.output
    assert "Semantic Memory" in result.output


def test_status_shows_counts(mock_episodic_store, mock_graph):
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.episodic.store.EpisodicStore.stats", mock_episodic_store.stats), \
         patch("engram.semantic.create_graph", return_value=mock_graph):
        result = runner.invoke(app, ["status"])
    assert "5" in result.output   # episodic count
    assert "3" in result.output   # node count


# --- cleanup tests ---

def test_cleanup_expired(mock_episodic_store):
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.episodic.store.EpisodicStore.cleanup_expired", mock_episodic_store.cleanup_expired):
        result = runner.invoke(app, ["cleanup"])
    assert result.exit_code == 0
    assert "2" in result.output or "Cleaned" in result.output


def test_cleanup_none_expired(mock_episodic_store):
    mock_episodic_store.cleanup_expired = AsyncMock(return_value=0)
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.episodic.store.EpisodicStore.cleanup_expired", mock_episodic_store.cleanup_expired):
        result = runner.invoke(app, ["cleanup"])
    assert result.exit_code == 0
    assert "No expired" in result.output


# --- summarize tests ---
# _get_engine is a closure; patch ReasoningEngine.summarize + run_async shortcut.

def test_summarize_basic():
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.semantic.create_graph", return_value=MagicMock()), \
         patch("engram.cli.system.run_async", return_value="Summary of recent memories"):
        result = runner.invoke(app, ["summarize"])
    assert result.exit_code == 0
    assert "Summary of recent memories" in result.output


def test_summarize_with_count():
    captured = {}

    def fake_run_async(coro):
        # Extract n from the coroutine's frame locals if possible; just return text
        captured["called"] = True
        return "Summary of recent memories"

    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.semantic.create_graph", return_value=MagicMock()), \
         patch("engram.cli.system.run_async", side_effect=fake_run_async):
        result = runner.invoke(app, ["summarize", "--count", "10"])
    assert result.exit_code == 0
    assert captured.get("called")


# --- backup / restore tests ---
# _get_stores is a closure in backup_cmd.register(); patch at module level (do_backup/do_restore).

def test_backup_command(tmp_path):
    output = str(tmp_path / "test_backup")
    manifest = {
        "episodic_count": 5,
        "semantic_nodes": 3,
        "timestamp": "2025-01-01T00:00:00",
        "version": "0.2.0",
    }
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.semantic.create_graph", return_value=MagicMock()), \
         patch("engram.backup.backup", new=AsyncMock(return_value=manifest)):
        result = runner.invoke(app, ["backup", "--output", output])
    assert result.exit_code == 0
    assert "Backup complete" in result.output


def test_restore_command(tmp_path):
    archive = tmp_path / "backup.tar.gz"
    archive.write_bytes(b"fake")
    restore_result = {
        "episodic_restored": 5,
        "semantic_nodes_restored": 3,
        "semantic_edges_restored": 1,
    }
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.semantic.create_graph", return_value=MagicMock()), \
         patch("engram.backup.restore", new=AsyncMock(return_value=restore_result)):
        result = runner.invoke(app, ["restore", str(archive), "--yes"])
    assert result.exit_code == 0
    assert "Restore complete" in result.output


def test_restore_archive_not_found(tmp_path):
    missing = str(tmp_path / "nonexistent.tar.gz")
    with patch("engram.episodic.store.EpisodicStore.__init__", return_value=None), \
         patch("engram.semantic.create_graph", return_value=MagicMock()):
        result = runner.invoke(app, ["restore", missing, "--yes"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "Archive not found" in result.output
