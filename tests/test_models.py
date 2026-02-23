"""Tests for core pydantic data models."""

from __future__ import annotations

from engram.models import (
    EdgeDef,
    EpisodicMemory,
    MemoryType,
    NodeDef,
    Priority,
    SchemaDefinition,
    SemanticEdge,
    SemanticNode,
)


def test_episodic_memory_defaults():
    """EpisodicMemory defaults to FACT type and NORMAL priority."""
    mem = EpisodicMemory(content="server rebooted")
    assert mem.memory_type == MemoryType.FACT
    assert mem.priority == Priority.NORMAL
    assert mem.entities == []
    assert mem.metadata == {}


def test_semantic_node_key():
    """SemanticNode.key computed as 'type:name'."""
    node = SemanticNode(type="Service", name="api-gateway")
    assert node.key == "Service:api-gateway"


def test_semantic_edge_key():
    """SemanticEdge.key computed as 'from--relation-->to'."""
    edge = SemanticEdge(
        from_node="Team:platform",
        to_node="Service:api-gateway",
        relation="owns",
    )
    assert edge.key == "Team:platform--owns-->Service:api-gateway"


def test_schema_definition():
    """SchemaDefinition holds NodeDef and EdgeDef lists correctly."""
    schema = SchemaDefinition(
        nodes=[
            NodeDef(name="Service", description="A service"),
            NodeDef(name="Team", description="An engineering team"),
        ],
        edges=[
            EdgeDef(name="owns", from_types=["Team"], to_types=["Service"]),
        ],
    )
    assert len(schema.nodes) == 2
    assert len(schema.edges) == 1
    assert schema.nodes[0].name == "Service"
    assert schema.edges[0].from_types == ["Team"]
