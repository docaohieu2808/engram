"""Combined reasoning engine - query both episodic and semantic memory."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import litellm

from engram.constitution import get_constitution_prompt_prefix
from engram.episodic.store import EpisodicStore
from engram.hooks import fire_hook
from engram.models import EpisodicMemory, SemanticEdge, SemanticNode
from engram.providers.base import MemoryProvider
from engram.providers.router import federated_search
from engram.resource_tier import ResourceTier, get_resource_monitor
from engram.semantic.graph import SemanticGraph
from engram.utils import strip_diacritics

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

SUMMARIZE_PROMPT = """You are a memory assistant. Summarize the following memories into key insights.
Be concise. Group related items. Highlight patterns, decisions, and important facts.

## Memories
{memories}

## Instructions
- Extract the most important insights
- Group related memories together
- Keep the summary under 200 words
- Use bullet points
"""

REASONING_PROMPT = """You are a memory reasoning assistant. Based on the retrieved memories below, answer the user's question.

## Episodic Memories (experiences, events)
{episodic_context}

## Semantic Knowledge (entities, relationships)
{semantic_context}

## External Knowledge (from connected providers)
{provider_context}

## Question
{question}

## Instructions
- THINK LIKE A HUMAN: read between the lines, infer hidden motivations, unspoken intentions
- Don't just summarize — DEDUCE. If someone does X despite Y, ask WHY and connect the dots
- Look for patterns: repeated behavior, contradictions between words and actions, emotional subtext
- Answer the REAL question, not the surface question. What is the user truly asking?
- Be specific - cite dates, names, and details from memories
- If memories contradict, note the conflict and reason about what it reveals
- If no relevant memories found, say so honestly
- Keep answer concise and direct — one strong insight beats five weak summaries
"""


class ReasoningEngine:
    """Combined query engine across episodic + semantic memory."""

    def __init__(
        self,
        episodic: EpisodicStore,
        graph: SemanticGraph,
        model: str,
        on_think_hook: str | None = None,
        providers: list[MemoryProvider] | None = None,
        recall_config=None,
    ):
        self._episodic = episodic
        self._graph = graph
        self._model = model
        self._on_think_hook = on_think_hook
        self._providers = providers or []
        # Cache of graph node names to avoid full scan on every think() call
        self._node_names_cache: list[str] | None = None
        self._recall_config = recall_config
        self._parallel_search = getattr(recall_config, "parallel_search", False) if recall_config else False

    def invalidate_cache(self) -> None:
        """Invalidate entity hints cache (call when graph mutates)."""
        self._node_names_cache = None

    async def summarize(self, n: int = 20, save: bool = False) -> str:
        """Summarize recent N memories into key insights using LLM.

        Args:
            n: Number of recent memories to include.
            save: If True, store the summary as a new memory with type=context.
        """
        from engram.models import MemoryType

        memories = await self._episodic.get_recent(n)
        if not memories:
            return "No memories to summarize."

        memory_lines = []
        for mem in memories:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            memory_lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}")
        memories_text = "\n".join(memory_lines)

        # Inject constitution as immutable prefix
        constitution_prefix = get_constitution_prompt_prefix()
        prompt = constitution_prefix + SUMMARIZE_PROMPT.format(memories=memories_text)

        monitor = get_resource_monitor()
        if not monitor.can_use_llm():
            # Degraded mode: return raw memory list without LLM summarization
            return f"[Resource tier: {monitor.get_tier().value} — LLM unavailable]\n\n" + memories_text

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            monitor.record_success()
            summary = response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("rate", "429")):
                monitor.record_failure("rate_limit")
            elif any(k in err_str for k in ("quota", "billing")):
                monitor.record_failure("quota")
            else:
                monitor.record_failure("transient")
            logger.error("Summarize LLM call failed: %s", e)
            return f"Summarization failed: {e}"

        if save and summary:
            await self._episodic.remember(
                summary,
                memory_type=MemoryType.CONTEXT,
                priority=6,
                tags=["summary"],
            )

        return summary

    async def think(self, question: str) -> str:
        """Answer a question by combining episodic, semantic, and federated memory."""
        # 1. Vector search for relevant episodic memories
        if self._parallel_search and self._episodic and self._graph:
            from engram.recall.parallel_search import ParallelSearcher
            searcher = ParallelSearcher(self._episodic, self._graph, self._recall_config)
            search_results = await searcher.search(question, limit=5)
            # Format parallel search results for the synthesis prompt
            episodic_context = "\n".join(
                f"[{r.source}] (score={r.score:.2f}) {r.content}" for r in search_results
            )
            # Use SearchResult list as episodic_results proxy for downstream synthesis
            episodic_results = search_results
            _use_parallel = True
        else:
            episodic_results = await self._episodic.search(question, limit=5)
            episodic_context = None
            _use_parallel = False

        # 2. Find entity hints in question by matching against known nodes
        entity_hints = await self._extract_entity_hints(question)

        # 3. Graph traversal for those entities
        semantic_results: dict[str, Any] = {}
        for entity in entity_hints:
            related = await self._graph.get_related([entity], depth=2)
            semantic_results.update(related)

        # 4. Federated search across external providers
        provider_results = await federated_search(
            question, self._providers, limit=5, timeout_seconds=10.0,
        )

        # 5. If we have results, use LLM to synthesize (resource-aware)
        monitor = get_resource_monitor()
        if episodic_results or semantic_results or provider_results:
            if monitor.can_use_llm():
                answer = await self._synthesize(
                    question, episodic_results, semantic_results, provider_results,
                    episodic_context_override=episodic_context if _use_parallel else None,
                )
            else:
                # BASIC tier: return raw results without LLM synthesis
                tier = monitor.get_tier()
                logger.info("Resource tier %s — skipping LLM synthesis", tier.value)
                if _use_parallel:
                    answer = episodic_context or "No relevant memories found for this question."
                else:
                    answer = self._format_raw_results(episodic_results, semantic_results)
        else:
            answer = "No relevant memories found for this question."

        fire_hook(self._on_think_hook, {"question": question, "answer": answer})
        return answer

    async def _extract_entity_hints(self, question: str) -> list[str]:
        """Match words in question against known graph node names.

        Supports Vietnamese diacritics: 'tram' matches 'Trâm'.
        Uses cached node names to avoid full graph scan on every call.
        """
        if self._node_names_cache is None:
            all_nodes = await self._graph.get_nodes()
            self._node_names_cache = [n.name for n in all_nodes]
        node_names = {n.lower(): n for n in self._node_names_cache}

        q_lower = question.lower()
        q_stripped = strip_diacritics(q_lower)
        words = q_lower.split()
        words_stripped = [strip_diacritics(w) for w in words]
        found: list[str] = []

        for name_lower, name in node_names.items():
            name_stripped = strip_diacritics(name_lower)
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

    def _format_raw_results(
        self,
        episodic: list[EpisodicMemory],
        semantic: dict[str, Any],
    ) -> str:
        """Format raw memory results without LLM synthesis (degraded mode)."""
        lines = ["[Resource-aware mode: returning raw memories without LLM synthesis]\n"]
        if episodic:
            lines.append("## Episodic Memories")
            for mem in episodic:
                ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                lines.append(f"- [{ts}] ({mem.memory_type.value}) {mem.content}")
        if semantic:
            lines.append("\n## Semantic Knowledge")
            for entity, data in semantic.items():
                lines.append(f"- {entity}")
                if isinstance(data, dict):
                    for n in data.get("nodes", []):
                        if isinstance(n, SemanticNode):
                            lines.append(f"  - [{n.type}] {n.name}")
        return "\n".join(lines)

    async def _synthesize(
        self,
        question: str,
        episodic: list[EpisodicMemory],
        semantic: dict[str, Any],
        provider_results: list | None = None,
        episodic_context_override: str | None = None,
    ) -> str:
        """Use LLM to reason over combined memory results."""
        from engram.providers.base import ProviderResult

        # Format episodic context — use pre-built parallel search context if provided
        if episodic_context_override is not None:
            episodic_ctx = episodic_context_override or "No episodic memories found."
        else:
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

        # Format provider context
        provider_lines = []
        for r in (provider_results or []):
            if isinstance(r, ProviderResult):
                provider_lines.append(f"[{r.source}] {r.content}")
        provider_ctx = "\n".join(provider_lines) if provider_lines else "No external knowledge found."

        # Inject constitution as immutable prefix (Law I, II, III)
        constitution_prefix = get_constitution_prompt_prefix()
        prompt = constitution_prefix + REASONING_PROMPT.format(
            episodic_context=episodic_ctx,
            semantic_context=semantic_ctx,
            provider_context=provider_ctx,
            question=question,
        )

        monitor = get_resource_monitor()
        last_exc: Exception | None = None
        for attempt in range(3):  # 1 initial + 2 retries
            try:
                response = await litellm.acompletion(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                monitor.record_success()
                return response.choices[0].message.content
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                # Classify error for resource monitor
                if any(k in err_str for k in ("rate", "429")):
                    monitor.record_failure("rate_limit")
                elif any(k in err_str for k in ("quota", "billing", "402")):
                    monitor.record_failure("quota")
                else:
                    monitor.record_failure("transient")
                is_transient = any(k in err_str for k in ("connection", "rate", "timeout", "503", "429"))
                if is_transient and attempt < 2:
                    logger.warning("Synthesis transient error (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(1)
                    continue
                break

        # Fallback: return raw results without LLM synthesis
        logger.error("LLM synthesis failed: %s", last_exc)
        return f"Memory results (LLM synthesis failed: {last_exc}):\n\n{episodic_ctx}\n\n{semantic_ctx}"
