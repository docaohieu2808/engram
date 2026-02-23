"""Load and validate YAML schemas for semantic memory extraction."""

from __future__ import annotations

from pathlib import Path

import yaml

from engram.models import EdgeDef, NodeDef, SchemaDefinition

BUILTIN_DIR = Path(__file__).parent / "builtin"


def load_schema(name_or_path: str) -> SchemaDefinition:
    """Load schema by builtin name or file path."""
    # Try builtin first
    builtin = BUILTIN_DIR / f"{name_or_path}.yaml"
    if builtin.exists():
        return _load_from_file(builtin)

    # Try as file path
    path = Path(name_or_path)
    if path.exists():
        return _load_from_file(path)

    raise FileNotFoundError(f"Schema not found: {name_or_path}")


def _load_from_file(path: Path) -> SchemaDefinition:
    """Parse YAML schema file into SchemaDefinition."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    nodes = [NodeDef(**n) for n in raw.get("nodes", [])]
    edges = [EdgeDef(**e) for e in raw.get("edges", [])]
    hints = raw.get("extraction_hints", {})

    return SchemaDefinition(nodes=nodes, edges=edges, extraction_hints=hints)


def validate_schema(path: str) -> list[str]:
    """Validate schema file, return list of errors (empty = valid)."""
    schema = load_schema(path)
    errors: list[str] = []
    node_names = {n.name for n in schema.nodes}

    for edge in schema.edges:
        for ft in edge.from_types:
            if ft not in node_names:
                errors.append(f"Edge '{edge.name}' references unknown from_type '{ft}'")
        for tt in edge.to_types:
            if tt not in node_names:
                errors.append(f"Edge '{edge.name}' references unknown to_type '{tt}'")

    return errors


def schema_to_prompt(schema: SchemaDefinition) -> str:
    """Format schema as human-readable text for LLM extraction prompt."""
    lines = ["## Entity Schema\n", "### Node Types"]
    for n in schema.nodes:
        attrs = ""
        if n.attributes:
            attr_strs = [f"{a.name} ({a.type})" for a in n.attributes]
            attrs = f" — attributes: {', '.join(attr_strs)}"
        lines.append(f"- **{n.name}**: {n.description}{attrs}")

    lines.append("\n### Relationship Types")
    for e in schema.edges:
        froms = ", ".join(e.from_types) or "any"
        tos = ", ".join(e.to_types) or "any"
        lines.append(f"- **{e.name}**: {froms} → {tos}")

    if schema.extraction_hints:
        lines.append("\n### Extraction Hints")
        for key, val in schema.extraction_hints.items():
            if isinstance(val, list):
                lines.append(f"- {key}: {', '.join(str(v) for v in val)}")
            else:
                lines.append(f"- {key}: {val}")

    return "\n".join(lines)
