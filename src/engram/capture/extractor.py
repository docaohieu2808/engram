"""LLM-powered entity extraction from chat messages."""

from __future__ import annotations

import json
import re
from typing import Any

import litellm
litellm.suppress_debug_info = True

from engram.models import ExtractionResult, SchemaDefinition, SemanticEdge, SemanticNode
from engram.schema.loader import schema_to_prompt

# Extraction prompt template
EXTRACTION_PROMPT = """You are an entity extraction system. Extract entities and relationships from the conversation below.

{schema}

## Instructions
- Extract ONLY entities and relationships that match the schema above
- Return valid JSON with "nodes" and "edges" arrays
- Each node: {{"type": "NodeType", "name": "EntityName", "attributes": {{}}}}
- Each edge: {{"from_node": "Type:Name", "to_node": "Type:Name", "relation": "relation_name"}}
- Be precise - only extract clearly stated facts
- Deduplicate entities by name

## Conversation
{messages}

## Output (JSON only, no markdown)
"""


class EntityExtractor:
    """Extract semantic entities from chat messages using LLM."""

    def __init__(self, model: str, schema: SchemaDefinition):
        self._model = model
        self._schema = schema
        self._schema_prompt = schema_to_prompt(schema)

    async def extract_entities(self, messages: list[dict[str, Any]]) -> ExtractionResult:
        """Extract entities from a list of chat messages."""
        all_nodes: dict[str, SemanticNode] = {}
        all_edges: dict[str, SemanticEdge] = {}

        for chunk in self._chunk_messages(messages, chunk_size=50):
            result = await self._extract_chunk(chunk)
            for node in result.nodes:
                all_nodes[node.key] = node
            for edge in result.edges:
                all_edges[edge.key] = edge

        return ExtractionResult(
            nodes=list(all_nodes.values()),
            edges=list(all_edges.values()),
        )

    async def extract_from_text(self, text: str) -> ExtractionResult:
        """Extract entities from plain text."""
        messages = [{"role": "user", "content": text}]
        return await self.extract_entities(messages)

    async def _extract_chunk(self, messages: list[dict]) -> ExtractionResult:
        """Run LLM extraction on a chunk of messages."""
        formatted = "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in messages
        )
        prompt = EXTRACTION_PROMPT.format(schema=self._schema_prompt, messages=formatted)

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return self._parse_response(content)
        except Exception as e:
            # Log but don't crash - return empty result
            print(f"[engram] Extraction error: {e}")
            return ExtractionResult()

    def _parse_response(self, content: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        # Strip markdown code blocks if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ExtractionResult()

        nodes = []
        for n in data.get("nodes", []):
            if "type" in n and "name" in n:
                nodes.append(SemanticNode(
                    type=n["type"], name=n["name"],
                    attributes=n.get("attributes", {}),
                ))

        edges = []
        for e in data.get("edges", []):
            if "from_node" in e and "to_node" in e and "relation" in e:
                edges.append(SemanticEdge(
                    from_node=e["from_node"], to_node=e["to_node"],
                    relation=e["relation"],
                ))

        return ExtractionResult(nodes=nodes, edges=edges)

    @staticmethod
    def _chunk_messages(messages: list[dict], chunk_size: int = 50) -> list[list[dict]]:
        """Split messages into overlapping chunks."""
        if len(messages) <= chunk_size:
            return [messages]

        chunks = []
        overlap = 2
        i = 0
        while i < len(messages):
            end = min(i + chunk_size, len(messages))
            chunks.append(messages[i:end])
            if end == len(messages):
                break
            i = end - overlap
        return chunks
