"""Memory retrieval pipeline — episodic, semantic, and federated search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from engram.episodic.store import EpisodicStore
from engram.models import EpisodicMemory, SearchResult
from engram.providers.base import MemoryProvider
from engram.providers.router import federated_search
from engram.semantic.graph import SemanticGraph
from engram.config import RecallConfig
from engram.utils import strip_diacritics

logger = logging.getLogger("engram")


@dataclass
class RetrievalResult:
    episodic_results: list
    semantic_results: dict[str, Any]
    provider_results: list
    episodic_context: str | None
    use_parallel: bool


async def retrieve(
    question: str,
    search_question: str,
    episodic: EpisodicStore,
    graph: SemanticGraph,
    providers: list[MemoryProvider],
    recall: RecallConfig,
    recall_config: Any,
    scoring_config: Any,
    node_names_cache: list[str] | None,
    skip_memory: bool,
    q_type: Any,
    search_limit: int,
    parallel_search: bool,
) -> RetrievalResult:
    """Run full retrieval pipeline: episodic + entity boost + semantic + federated."""
    from engram.recall.question_classifier import QuestionType

    # 1. Vector search for relevant episodic memories
    episodic_results: list = []
    episodic_context: str | None = None
    use_parallel = False

    if skip_memory:
        pass
    elif parallel_search and episodic and graph:
        from engram.recall.parallel_search import ParallelSearcher
        from engram.recall.fusion_formatter import format_for_llm
        searcher = ParallelSearcher(episodic, graph, recall_config, scoring_config, recall=recall)
        search_results = await searcher.search(search_question, limit=search_limit)
        _max_chars = 3000 if q_type == QuestionType.MIXED else 6000
        episodic_context = format_for_llm(search_results, max_chars=_max_chars) or "\n".join(
            f"[{r.source}] (score={r.score:.2f}) {r.content}" for r in search_results
        )
        episodic_results = search_results
        use_parallel = True
    else:
        episodic_results = await episodic.search(search_question, limit=search_limit)

    # 2. Entity hints from question
    entity_hints: list[str] = []
    if not skip_memory:
        entity_hints = await _extract_entity_hints(question, graph, node_names_cache)

    # 2b. Entity-boosted search
    if entity_hints and episodic and not skip_memory:
        entity_memories = await _search_by_entities(entity_hints, episodic, limit=recall.entity_search_limit)
        if use_parallel:
            from engram.recall.fusion_formatter import format_for_llm
            existing_contents = {r.content for r in episodic_results}
            for mem in entity_memories:
                if mem.content not in existing_contents:
                    episodic_results.append(SearchResult(
                        id=mem.id,
                        content=mem.content,
                        score=recall.entity_boost_score,
                        source="entity_boost",
                        memory_type=mem.memory_type.value if hasattr(mem.memory_type, "value") else str(mem.memory_type),
                        importance=mem.priority,
                        timestamp=mem.timestamp,
                        metadata=mem.metadata,
                    ))
                    existing_contents.add(mem.content)
            episodic_context = format_for_llm(episodic_results, max_chars=6000) or episodic_context
        else:
            existing_ids = {m.id for m in episodic_results}
            for mem in entity_memories:
                if mem.id not in existing_ids:
                    episodic_results.append(mem)
                    existing_ids.add(mem.id)

    # 3. Graph traversal for entity hints
    semantic_results: dict[str, Any] = {}
    for entity in entity_hints:
        related = await graph.get_related([entity], depth=recall.entity_graph_depth)
        semantic_results.update(related)

    # 4. Federated search
    provider_results = await federated_search(
        question, providers,
        limit=recall.provider_search_limit,
        timeout_seconds=recall.federated_search_timeout,
        force_federation=skip_memory,
    )

    # 4b. Filter parallel results below minimum relevance score
    if use_parallel and episodic_results:
        from engram.recall.fusion_formatter import format_for_llm
        from engram.recall.question_classifier import QuestionType
        _min_score = recall.min_relevance_score
        if q_type == QuestionType.MIXED:
            _min_score = max(_min_score, 0.6)
        _before = len(episodic_results)
        episodic_results = [r for r in episodic_results if r.score >= _min_score]
        _filtered = _before - len(episodic_results)
        if _filtered:
            logger.debug("retrieve(): filtered %d low-relevance results (score < %.2f)", _filtered, _min_score)
        episodic_context = format_for_llm(episodic_results, max_chars=6000) or "\n".join(
            f"[{r.source}] (score={r.score:.2f}) {r.content}" for r in episodic_results
        ) if episodic_results else None

    return RetrievalResult(
        episodic_results=episodic_results,
        semantic_results=semantic_results,
        provider_results=provider_results,
        episodic_context=episodic_context,
        use_parallel=use_parallel,
    )


async def _extract_entity_hints(
    question: str,
    graph: SemanticGraph,
    node_names_cache: list[str] | None,
) -> list[str]:
    """Match words in question against known graph node names."""
    if node_names_cache is None:
        all_nodes = await graph.get_nodes()
        node_names_cache = [n.name for n in all_nodes]
    node_names = {n.lower(): n for n in node_names_cache}

    q_lower = question.lower()
    q_stripped = strip_diacritics(q_lower)
    words = q_lower.split()
    words_stripped = [strip_diacritics(w) for w in words]
    found: list[str] = []

    for name_lower, name in node_names.items():
        name_stripped = strip_diacritics(name_lower)
        if name_lower in q_lower or name_stripped in q_stripped:
            found.append(name)
            continue
        for word, word_s in zip(words, words_stripped):
            if len(word) > 2 and (word == name_lower or word_s == name_stripped):
                found.append(name)
                break

    return found


async def _search_by_entities(
    entity_names: list[str],
    episodic: EpisodicStore,
    limit: int = 10,
) -> list[EpisodicMemory]:
    """Search episodic memories by entity name via vector similarity."""
    results: list[EpisodicMemory] = []
    seen_ids: set[str] = set()
    for name in entity_names[:3]:
        try:
            hits = await episodic.search(name, limit=limit)
            for mem in hits:
                if mem.id not in seen_ids:
                    results.append(mem)
                    seen_ids.add(mem.id)
        except Exception as e:
            logger.debug("Entity search failed for %s: %s", name, e)
    return results[:limit]
