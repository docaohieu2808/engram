"""LLM-powered entity extraction from chat messages."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import litellm

from engram.capture.extraction_filters import should_extract
from engram.models import ExtractionResult, SchemaDefinition, SemanticEdge, SemanticNode
from engram.schema.loader import schema_to_prompt

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

# Extraction prompt template
EXTRACTION_PROMPT = """You are an entity extraction system. Extract entities and relationships from the conversation below.

{schema}

## Instructions
- Extract entities and relationships that match the schema above
- Return valid JSON with "nodes" and "edges" arrays
- Each node: {{"type": "NodeType", "name": "EntityName", "attributes": {{}}}}
- Each edge: {{"from_node": "Type:Name", "to_node": "Type:Name", "relation": "relation_name"}}
- Be precise - only extract clearly stated facts
- Deduplicate entities by name (case-insensitive: "Docker compose" = "Docker Compose")
- Use proper casing for entity names: preserve acronyms (API, HTTP, SQL), camelCase (GitHub, OpenClaw), technical names (HAProxy, RabbitMQ, PostgreSQL)
- **CRITICAL**: Every node MUST have at least one edge connecting it to another node. Never create orphan nodes. If you cannot find a relationship for an entity, do NOT include it.
- Common relation types: uses, runs_on, part_of, located_at, works_with, knows, manages, depends_on, created_by, connects_to

## What NOT to extract
- Do NOT extract AI agent names (Claude Code, OpenClaw, Fullstack-developer, assistant) as Person — they are Service/Tool nodes if relevant at all
- Do NOT extract email addresses as Person nodes
- Do NOT extract every technology mentioned — only extract tools/services the USER actually uses or manages
- Do NOT extract file paths, command names, or code identifiers as entities
- Do NOT extract usernames, handles, or account names as Person nodes (e.g. Docaohieu2808 is a username, Admin@docaohieu.com is an email — NOT separate people)
- Merge user aliases into one canonical name: Hiếu = Hieudc = Docaohieu2808 = "Ông Hiếu" = Admin@docaohieu.com → always use "Hiếu"

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
        all_nodes: dict[str, SemanticNode] = {}  # case-insensitive key → node
        all_edges: dict[str, SemanticEdge] = {}

        for chunk in self._chunk_messages(messages, chunk_size=50):
            result = await self._extract_chunk(chunk)
            for node in result.nodes:
                # Case-insensitive dedup: keep first occurrence's casing
                ci_key = node.key.lower()
                if ci_key not in all_nodes:
                    all_nodes[ci_key] = node
            for edge in result.edges:
                ci_key = edge.key.lower()
                if ci_key not in all_edges:
                    all_edges[ci_key] = edge

        return ExtractionResult(
            nodes=list(all_nodes.values()),
            edges=list(all_edges.values()),
        )

    async def extract_from_text(self, text: str) -> ExtractionResult:
        """Extract entities from plain text."""
        messages = [{"role": "user", "content": text}]
        return await self.extract_entities(messages)

    @staticmethod
    def filter_entities_for_content(
        content: str,
        all_entity_names: list[str],
        context_messages: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Content-first entity assignment with context-only validation.

        Rules:
        1) Candidate must appear in the current content (never inject from context).
        2) Context is only used to validate/boost candidates and canonical casing.
        3) Case-insensitive dedupe in final output.
        """
        if not content:
            return []

        content_lower = content.lower()

        # Step 1: candidates from current content only (word/phrase boundary check)
        content_candidates: list[str] = []
        for name in all_entity_names:
            n = (name or "").strip()
            if not n:
                continue
            # Avoid pure substring pollution; require rough boundary semantics.
            if re.search(rf"(?<!\\w){re.escape(n.lower())}(?!\\w)", content_lower):
                content_candidates.append(n)

        if not content_candidates:
            return []

        # Step 2: optional context validation (no adding new entities)
        validated: list[str] = []
        context_text = ""
        if context_messages:
            context_text = "\n".join(str(m.get("content", "")) for m in context_messages).lower()

        for cand in content_candidates:
            c_low = cand.lower()
            if not context_text:
                validated.append(cand)
                continue

            # Candidate is valid if it appears in nearby context text
            # OR if it's already a known extracted schema entity (same list source).
            if c_low in context_text or any(c_low == (e or "").strip().lower() for e in all_entity_names):
                validated.append(cand)

        # Step 3: case-insensitive dedupe, prefer title/canonical-looking form
        out: list[str] = []
        seen: dict[str, str] = {}
        for e in validated:
            key = e.casefold()
            if key not in seen:
                seen[key] = e
            elif seen[key].islower() and not e.islower():
                seen[key] = e

        out = list(seen.values())
        return out

    async def _extract_chunk(self, messages: list[dict]) -> ExtractionResult:
        """Run LLM extraction on a chunk of messages, with up to 2 retries on transient errors."""
        # Skip chunk entirely if ALL messages are junk content
        extractable = [m for m in messages if should_extract(m.get("content", ""))]
        if not extractable:
            logger.debug("Skipping extraction chunk — all %d messages filtered as junk", len(messages))
            return ExtractionResult()

        formatted = "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in messages
        )
        prompt = EXTRACTION_PROMPT.format(schema=self._schema_prompt, messages=formatted)

        last_exc: Exception | None = None
        for attempt in range(3):  # 1 initial + 2 retries
            try:
                response = await litellm.acompletion(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    thinking={"type": "disabled"},
                )
                content = response.choices[0].message.content
                return self._parse_response(content)
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in ("connection", "rate", "timeout", "503", "429"))
                if is_transient and attempt < 2:
                    logger.warning("Extraction transient error (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(1)
                    continue
                break

        logger.error("Extraction error: %s", last_exc)
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
