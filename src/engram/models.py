"""Core data models for engram dual-memory system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator


# --- Enums ---


class MemoryType(str, Enum):
    """Classification of episodic memories."""

    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    TODO = "todo"
    ERROR = "error"
    CONTEXT = "context"
    WORKFLOW = "workflow"


class Priority(int, Enum):
    """Memory priority levels (1=lowest, 10=critical)."""

    LOWEST = 1
    LOW = 3
    NORMAL = 5
    HIGH = 7
    CRITICAL = 10


# --- Episodic Memory (Vector DB) ---


class EpisodicMemory(BaseModel):
    """A single episodic memory stored in vector DB."""

    id: str = ""
    content: str
    memory_type: MemoryType = MemoryType.FACT
    priority: int = Priority.NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    # Ebbinghaus decay fields
    access_count: int = 0
    last_accessed: datetime | None = None
    decay_rate: float = 0.1
    # Consolidation fields
    consolidation_group: str | None = None
    consolidated_into: str | None = None

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"Priority must be 1-10, got {v}")
        return v


# --- Semantic Memory (Graph DB) ---


class AttributeDef(BaseModel):
    """Schema definition for a node attribute."""

    name: str
    type: str = "string"
    required: bool = False


class NodeDef(BaseModel):
    """Schema definition for a node type."""

    name: str
    description: str = ""
    attributes: list[AttributeDef] = Field(default_factory=list)


class EdgeDef(BaseModel):
    """Schema definition for an edge type."""

    name: str
    from_types: list[str] = Field(default_factory=list)
    to_types: list[str] = Field(default_factory=list)


class SchemaDefinition(BaseModel):
    """Complete schema for semantic memory."""

    nodes: list[NodeDef] = Field(default_factory=list)
    edges: list[EdgeDef] = Field(default_factory=list)
    extraction_hints: dict[str, Any] = Field(default_factory=dict)


class SemanticNode(BaseModel):
    """A node in the semantic graph."""

    type: str
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def key(self) -> str:
        return f"{self.type}:{self.name}"


class SemanticEdge(BaseModel):
    """An edge in the semantic graph."""

    from_node: str  # node key (type:name)
    to_node: str
    relation: str
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def key(self) -> str:
        return f"{self.from_node}--{self.relation}-->{self.to_node}"


class ExtractionResult(BaseModel):
    """Result of LLM entity extraction."""

    nodes: list[SemanticNode] = Field(default_factory=list)
    edges: list[SemanticEdge] = Field(default_factory=list)


class IngestResult(BaseModel):
    """Result of dual ingest operation."""

    episodic_count: int = 0
    semantic_nodes: int = 0
    semantic_edges: int = 0
