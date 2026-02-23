"""Combined reasoning engine - query both episodic and semantic memory."""

from __future__ import annotations

from typing import Any

import litellm
litellm.suppress_debug_info = True

from engram.episodic.store import EpisodicStore
from engram.models import EpisodicMemory, SemanticNode
from engram.semantic.graph import SemanticGraph

REASONING_PROMPT = """You are a memory reasoning assistant. Based on the retrieved memories below, answer the user's question.

## Episodic Memories (experiences, events)
{episodic_context}

## Semantic Knowledge (entities, relationships)
{semantic_context}

## Question
{question}

## Instructions
- Synthesize information from both memory sources
- Be specific - cite dates, names, and details from memories
- If memories contradict, note the conflict
- If no relevant memories found, say so honestly
- Keep answer concise and direct
"""


class ReasoningEngine:
    """Combined query engine across episodic + semantic memory."""

    def __init__(
        self,
        episodic: EpisodicStore,
        graph: SemanticGraph,
        model: str,
    ):
        self._episodic = episodic
        self._graph = graph
        self._model = model

    async def think(self, question: str) -> str:
        """Answer a question by combining episodic and semantic memory."""
        # 1. Vector search for relevant episodic memories
        episodic_results = await self._episodic.search(question, limit=5)

        # 2. Find entity hints in question by matching against known nodes
        entity_hints = await self._extract_entity_hints(question)

        # 3. Graph traversal for those entities
        semantic_results: dict[str, Any] = {}
        for entity in entity_hints:
            related = await self._graph.get_related([entity], depth=2)
            if related:
                semantic_results[entity] = related

        # 4. If we have results, use LLM to synthesize
        if episodic_results or semantic_results:
            return await self._synthesize(question, episodic_results, semantic_results)

        return "No relevant memories found for this question."

    async def _extract_entity_hints(self, question: str) -> list[str]:
        """Match words in question against known graph node names."""
        all_nodes = await self._graph.get_nodes()
        node_names = {n.name.lower(): n.name for n in all_nodes}

        # Simple word matching against known entities
        words = question.lower().split()
        found: list[str] = []

        for name_lower, name in node_names.items():
            # Check if entity name appears in question (case-insensitive)
            if name_lower in question.lower():
                found.append(name)
                continue
            # Check individual words
            for word in words:
                if len(word) > 2 and word in name_lower:
                    found.append(name)
                    break

        return found

    async def _synthesize(
        self,
        question: str,
        episodic: list[EpisodicMemory],
        semantic: dict[str, Any],
    ) -> str:
        """Use LLM to reason over combined memory results."""
        # Format episodic context
        episodic_lines = []
        for mem in episodic:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            episodic_lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}")
        episodic_ctx = "\n".join(episodic_lines) if episodic_lines else "No episodic memories found."

        # Format semantic context
        semantic_lines = []
        for entity, data in semantic.items():
            semantic_lines.append(f"\n### {entity}")
            if isinstance(data, dict):
                for key, items in data.items():
                    if isinstance(items, dict):
                        for node_name, related in items.items():
                            if "nodes" in related:
                                for n in related["nodes"]:
                                    if isinstance(n, SemanticNode):
                                        semantic_lines.append(f"  - {n.key}")
                            if "edges" in related:
                                for e in related["edges"]:
                                    semantic_lines.append(f"  - {e}")
        semantic_ctx = "\n".join(semantic_lines) if semantic_lines else "No semantic knowledge found."

        prompt = REASONING_PROMPT.format(
            episodic_context=episodic_ctx,
            semantic_context=semantic_ctx,
            question=question,
        )

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: return raw results without LLM synthesis
            return f"Memory results (LLM synthesis failed: {e}):\n\n{episodic_ctx}\n\n{semantic_ctx}"
