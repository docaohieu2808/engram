"""Combined reasoning engine - query both episodic and semantic memory."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

import litellm
litellm.suppress_debug_info = True

from engram.episodic.store import EpisodicStore
from engram.models import EpisodicMemory, SemanticEdge, SemanticNode
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
            # get_related returns {entity: {nodes, edges}} — merge directly
            semantic_results.update(related)

        # 4. If we have results, use LLM to synthesize
        if episodic_results or semantic_results:
            return await self._synthesize(question, episodic_results, semantic_results)

        return "No relevant memories found for this question."

    async def _extract_entity_hints(self, question: str) -> list[str]:
        """Match words in question against known graph node names.

        Supports Vietnamese diacritics: 'tram' matches 'Trâm'.
        """
        all_nodes = await self._graph.get_nodes()
        node_names = {n.name.lower(): n.name for n in all_nodes}

        q_lower = question.lower()
        q_stripped = self._strip(q_lower)
        words = q_lower.split()
        words_stripped = [self._strip(w) for w in words]
        found: list[str] = []

        for name_lower, name in node_names.items():
            name_stripped = self._strip(name_lower)
            # Check if entity name appears in question (exact or stripped diacritics)
            if name_lower in q_lower or name_stripped in q_stripped:
                found.append(name)
                continue
            # Check individual words
            for word, word_s in zip(words, words_stripped):
                if len(word) > 2 and (word in name_lower or word_s in name_stripped):
                    found.append(name)
                    break

        return found

    @staticmethod
    def _strip(text: str) -> str:
        """Remove diacritics for fuzzy matching."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

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

        # Format semantic context — include attributes for LLM synthesis
        semantic_lines = []
        for entity, data in semantic.items():
            semantic_lines.append(f"\n### {entity}")
            if isinstance(data, dict):
                # get_related returns {"nodes": [...], "edges": [...]}
                for n in data.get("nodes", []):
                    if isinstance(n, SemanticNode):
                        attrs = json.dumps(n.attributes, ensure_ascii=False) if n.attributes else ""
                        semantic_lines.append(f"  - [{n.type}] {n.name}{' | ' + attrs if attrs else ''}")
                for e in data.get("edges", []):
                    if isinstance(e, SemanticEdge):
                        semantic_lines.append(f"  - {e.from_node} --{e.relation}--> {e.to_node}")
                    else:
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
