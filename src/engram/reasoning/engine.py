"""Combined reasoning engine - query both episodic and semantic memory."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import litellm

from engram.constitution import get_constitution_prompt_prefix
from engram.llm_utils import _llm_call_with_fallback as _shared_llm_call, is_anthropic_model
from engram.episodic.store import EpisodicStore
from engram.hooks import fire_hook
from engram.models import EpisodicMemory, SemanticNode
from engram.providers.base import MemoryProvider
from engram.resource_tier import get_resource_monitor
from engram.semantic.graph import SemanticGraph
from engram.config import RecallConfig, LLMConfig
from engram.reasoning.synthesizer import SUMMARIZE_PROMPT, synthesize
from engram.reasoning.retriever import retrieve

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")


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
        scoring_config=None,
        disable_thinking: bool = False,
        recall: RecallConfig | None = None,
    ):
        self._episodic = episodic
        self._graph = graph
        self._model = model
        self._disable_thinking = disable_thinking
        self._on_think_hook = on_think_hook
        self._providers: list[MemoryProvider] = providers or []
        self._node_names_cache: list[str] | None = None
        self._personal_entity_names: list[str] | None = None
        self._recall_config = recall_config
        self._scoring_config = scoring_config
        self._recall = recall or RecallConfig()
        self._parallel_search = getattr(recall_config, "parallel_search", False) if recall_config else False
        self._default_model = LLMConfig().model

    def set_providers(self, providers: list[MemoryProvider]) -> None:
        """Replace active providers for federated search."""
        self._providers = providers

    async def _llm_call_with_fallback(self, kwargs: dict) -> Any:
        return await _shared_llm_call(kwargs, primary_model=kwargs.get("model", self._model))

    def invalidate_cache(self) -> None:
        """Invalidate entity hints cache (call when graph mutates)."""
        self._node_names_cache = None
        self._personal_entity_names = None

    async def summarize(self, n: int = 20, save: bool = False) -> str:
        """Summarize recent N memories into key insights using LLM."""
        from engram.models import MemoryType

        memories = await self._episodic.get_recent(n)
        if not memories:
            return "No memories to summarize."

        memory_lines = []
        for mem in memories:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            memory_lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}")
        memories_text = "\n".join(memory_lines)

        constitution_prefix = get_constitution_prompt_prefix()
        now = datetime.now(timezone.utc).astimezone()
        prompt = constitution_prefix + SUMMARIZE_PROMPT.format(
            current_datetime=now.strftime("%Y-%m-%d %H:%M (%A, %Z)"),
            memories=memories_text,
        )

        monitor = get_resource_monitor()
        if not monitor.can_use_llm():
            return f"[Resource tier: {monitor.get_tier().value} — LLM unavailable]\n\n" + memories_text

        try:
            kwargs = dict(model=self._model, messages=[{"role": "user", "content": prompt}], temperature=0.3)
            if is_anthropic_model(self._model):
                if self._disable_thinking:
                    kwargs["thinking"] = {"type": "disabled"}
                else:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": 5000}
                    kwargs.pop("temperature", None)
            response = await self._llm_call_with_fallback(kwargs)
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

    async def think(self, question: str, mode: str | None = None) -> dict:
        """Answer a question by combining episodic, semantic, and federated memory.

        Args:
            question: The user's question.
            mode: Optional override — "research" skips memory, only uses providers + LLM.

        Returns a dict with keys:
            answer (str): The synthesized or raw answer text.
            degraded (bool): True when LLM was skipped due to resource tier.
        """
        _search_limit = self._recall.search_limit

        # 0. Classify question type
        from engram.recall.question_classifier import classify_question, QuestionType
        _PERSONAL_ENTITY_TYPES = {"person", "project", "environment"}
        if self._node_names_cache is None and self._graph:
            try:
                all_nodes = await self._graph.get_nodes()
                self._node_names_cache = [n.name for n in all_nodes]
                self._personal_entity_names = [
                    n.name for n in all_nodes
                    if n.type.lower() in _PERSONAL_ENTITY_TYPES
                ]
            except Exception:
                pass
        personal_entities = getattr(self, "_personal_entity_names", None)
        q_type = classify_question(question, known_entities=personal_entities)
        if mode == "research":
            skip_memory = True
            q_type = QuestionType.GENERAL
        else:
            skip_memory = q_type == QuestionType.GENERAL
        logger.info("think(): mode=%s, q_type=%s, skip_memory=%s", mode or "auto", q_type.value, skip_memory)
        if skip_memory:
            logger.info("think(): skipping episodic/semantic retrieval")
        elif q_type == QuestionType.MIXED:
            _search_limit = 3

        # 0b. Resolve temporal references
        search_question = question
        if not skip_memory:
            try:
                from engram.recall.temporal_resolver import resolve_temporal
                resolved_q, _resolved_date = resolve_temporal(question)
                if _resolved_date:
                    search_question = resolved_q
            except Exception as exc:
                logger.debug("engine: temporal resolution failed, using original query: %s", exc)

        # 1-4. Retrieve memories, entities, and provider results
        ret = await retrieve(
            question=question,
            search_question=search_question,
            episodic=self._episodic,
            graph=self._graph,
            providers=self._providers,
            recall=self._recall,
            recall_config=self._recall_config,
            scoring_config=self._scoring_config,
            node_names_cache=self._node_names_cache,
            skip_memory=skip_memory,
            q_type=q_type,
            search_limit=_search_limit,
            parallel_search=self._parallel_search,
        )
        episodic_results = ret.episodic_results
        semantic_results = ret.semantic_results
        provider_results = ret.provider_results
        episodic_context = ret.episodic_context
        _use_parallel = ret.use_parallel

        # 5. Synthesize answer (resource-aware)
        monitor = get_resource_monitor()
        has_results = bool(episodic_results or semantic_results or provider_results)
        should_synthesize = has_results or skip_memory
        if should_synthesize:
            if monitor.can_use_llm():
                result = await synthesize(
                    question=question,
                    episodic=episodic_results,
                    semantic=semantic_results,
                    model=self._model,
                    disable_thinking=self._disable_thinking,
                    provider_results=provider_results,
                    episodic_context_override=episodic_context if _use_parallel else None,
                    q_type=q_type,
                )
            else:
                tier = monitor.get_tier()
                logger.warning("think() degraded: resource tier %s — skipping LLM synthesis", tier.value)
                if skip_memory:
                    result = {"answer": "Resource tier too low for general knowledge questions.", "degraded": True}
                elif _use_parallel:
                    result = {
                        "answer": episodic_context or "No relevant memories found for this question.",
                        "degraded": True,
                    }
                else:
                    result = self._format_raw_results(episodic_results, semantic_results)
        else:
            result = {"answer": "No relevant memories found for this question.", "degraded": False}

        # 6. Source attribution
        sources: list[dict] = []
        if episodic_results:
            origin_counts: Counter = Counter()
            for r in episodic_results:
                origin = getattr(r, "origin", "") or ""
                if not origin and hasattr(r, "metadata") and isinstance(r.metadata, dict):
                    origin = r.metadata.get("source", "")
                origin_counts[origin or "unknown"] += 1
            sources.append({"type": "episodic", "count": len(episodic_results), "origins": dict(origin_counts)})
        if semantic_results:
            sources.append({"type": "semantic", "count": len(semantic_results)})
        if provider_results:
            provider_counts = Counter(r.source for r in provider_results)
            for name, count in provider_counts.items():
                sources.append({"type": "provider", "name": name, "count": count})
        if not sources:
            sources.append({"type": "llm", "name": "trained_knowledge"})
        result["sources"] = sources
        result["question_type"] = q_type.value if q_type else ""

        # 7. Recall boost
        await self._boost_recalled_memories(episodic_results)

        fire_hook(self._on_think_hook, {"question": question, "answer": result["answer"]})
        return result

    async def _boost_recalled_memories(self, results) -> None:
        """Increment importance for memories recalled during Think (spaced repetition)."""
        if not results or not self._episodic:
            return
        _BOOST = 0.1
        _MAX_IMPORTANCE = 10
        try:
            for r in results:
                mid = getattr(r, "id", None)
                if not mid:
                    continue
                current = getattr(r, "importance", None) or 5
                new_imp = min(current + _BOOST, _MAX_IMPORTANCE)
                if new_imp != current:
                    await self._episodic.update_metadata(mid, {"importance": new_imp})
            logger.debug("recall boost: bumped %d memories by +%.1f", len(results), _BOOST)
        except Exception as exc:
            logger.debug("recall boost failed (non-critical): %s", exc)

    def _format_raw_results(self, episodic: list[EpisodicMemory], semantic: dict[str, Any]) -> dict:
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
        return {"answer": "\n".join(lines), "degraded": True}
