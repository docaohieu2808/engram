"""Memory consolidation engine — clusters related memories and LLM-summarizes them.

Clusters by entity/tag Jaccard overlap, then LLM-summarizes each cluster into a
CONTEXT memory linking back to originals via consolidation_group.
"""

from __future__ import annotations

import logging
import uuid

import litellm

from engram.config import ConsolidationConfig
from engram.episodic.store import EpisodicStore
from engram.models import EpisodicMemory, MemoryType
from engram.sanitize import sanitize_llm_input

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

_CONSOLIDATION_PROMPT = """You are a memory consolidation system. Given a cluster of related memories,
create a concise summary that captures the key facts, decisions, and patterns.

Memories:
{memories}

Rules:
- Summarize in 1-3 sentences
- Preserve key facts, names, dates, decisions
- Remove redundancy
- Use present tense for ongoing facts, past tense for events
- Output ONLY the summary text, nothing else"""


class ConsolidationEngine:
    """Clusters related episodic memories and LLM-summarizes each cluster."""

    def __init__(
        self,
        episodic: EpisodicStore,
        model: str,
        config: ConsolidationConfig | None = None,
    ):
        self._episodic = episodic
        self._model = model
        self._config = config or ConsolidationConfig()

    async def consolidate(self, limit: int = 50) -> list[str]:
        """Fetch recent unconsolidated memories, cluster, summarize, store.

        Returns list of new consolidated memory IDs.
        """
        memories = await self._episodic.get_recent(n=limit)
        unconsolidated = [m for m in memories if not m.consolidated_into]

        if len(unconsolidated) < self._config.min_cluster_size:
            logger.info("Not enough unconsolidated memories to consolidate (%d)", len(unconsolidated))
            return []

        clusters = self._cluster_by_overlap(unconsolidated)
        new_ids: list[str] = []

        for cluster in clusters:
            if len(cluster) < self._config.min_cluster_size:
                continue
            try:
                summary = await self._summarize_cluster(cluster)
                group_id = str(uuid.uuid4())

                # Store summary as new CONTEXT memory
                all_entities = list({e for m in cluster for e in m.entities})
                all_tags = list({t for m in cluster for t in m.tags})
                all_tags.append("consolidated")

                new_id = await self._episodic.remember(
                    summary,
                    memory_type=MemoryType.CONTEXT,
                    priority=6,
                    entities=all_entities,
                    tags=all_tags,
                    metadata={"consolidation_group": group_id},
                )

                # Mark originals as consolidated
                for mem in cluster:
                    await self._episodic.update_metadata(mem.id, {
                        "consolidated_into": new_id,
                        "consolidation_group": group_id,
                    })

                new_ids.append(new_id)
                logger.info(
                    "Consolidated %d memories into %s (group=%s)",
                    len(cluster), new_id, group_id[:8],
                )
            except Exception as e:
                logger.warning("Consolidation failed for cluster: %s", e)

        return new_ids

    def _cluster_by_overlap(
        self, memories: list[EpisodicMemory],
    ) -> list[list[EpisodicMemory]]:
        """Cluster memories by Jaccard similarity of entity+tag sets.

        Uses union-find to group memories with overlap above threshold.
        """
        n = min(len(memories), 200)  # Cap to avoid O(n²) explosion
        memories = memories[:n]
        # Build feature sets (entities + tags)
        features = [set(m.entities) | set(m.tags) for m in memories]

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        threshold = self._config.similarity_threshold
        for i in range(n):
            if not features[i]:
                continue
            for j in range(i + 1, n):
                if not features[j]:
                    continue
                intersection = len(features[i] & features[j])
                union_size = len(features[i] | features[j])
                if union_size > 0 and intersection / union_size >= threshold:
                    union(i, j)

        # Group by root
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return [[memories[i] for i in indices] for indices in groups.values()]

    async def _summarize_cluster(self, cluster: list[EpisodicMemory]) -> str:
        """LLM-summarize a cluster of related memories."""
        # I-C3: sanitize memory content (user-originated) before LLM interpolation
        memory_texts = "\n".join(
            f"- [{m.memory_type.value}] {sanitize_llm_input(m.content, max_len=500)}"
            for m in cluster
        )
        prompt = _CONSOLIDATION_PROMPT.format(memories=memory_texts)

        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            thinking={"type": "disabled"},
        )
        return response.choices[0].message.content.strip()
