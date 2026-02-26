"""EngramClient — transparent LLM wrapper with auto-memory.

Drop-in replacement for direct litellm calls. Automatically recalls relevant
memories before each LLM call and extracts new memories from responses.

Usage:
    client = EngramClient(namespace="my-agent")
    response = await client.chat([{"role": "user", "content": "What did we decide?"}])

    # Or sync:
    response = client.chat_sync([{"role": "user", "content": "What did we decide?"}])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

from engram.config import load_config
from engram.models import MemoryType
from engram.utils import run_async

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

# Max chars of injected memory context to avoid bloating the prompt
_MAX_MEMORY_CONTEXT_CHARS = 2000


class EngramClient:
    """Universal auto-memory client for any AI agent.

    Wraps litellm calls with pre-call recall and post-call extraction.
    All memory operations are fail-open — LLM calls always succeed even if
    memory ops fail.
    """

    def __init__(
        self,
        namespace: str | None = None,
        auto_recall: bool = True,
        auto_extract: bool = True,
        model: str | None = None,
        extract_model: str | None = None,
    ) -> None:
        self._config = load_config()
        self._model = model or self._config.llm.model
        self._namespace = namespace
        self._auto_recall = auto_recall
        self._auto_extract = auto_extract
        self._extract_model = extract_model  # defaults inside MemoryExtractor

        # Lazy-init stores — created on first use
        self._episodic = None
        self._graph = None
        self._engine = None
        self._extractor = None  # MemoryExtractor, lazy-init
        # A-C1: track fire-and-forget background tasks to cancel on close()
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ #
    #  Store lifecycle                                                     #
    # ------------------------------------------------------------------ #

    async def _ensure_stores(self) -> None:
        """Lazy-init episodic store and semantic graph from config."""
        if self._episodic is None:
            from engram.episodic.store import EpisodicStore
            self._episodic = EpisodicStore(
                config=self._config.episodic,
                embedding_config=self._config.embedding,
                namespace=self._namespace,
            )
        if self._graph is None:
            from engram.semantic import create_graph
            self._graph = create_graph(
                self._config.semantic,
                tenant_id=self._namespace or "default",
            )

    async def _ensure_engine(self) -> None:
        """Lazy-init ReasoningEngine (requires stores)."""
        await self._ensure_stores()
        if self._engine is None:
            from engram.reasoning.engine import ReasoningEngine
            self._engine = ReasoningEngine(
                episodic=self._episodic,
                graph=self._graph,
                model=self._model,
            )

    def _ensure_extractor(self):
        """Lazy-init MemoryExtractor."""
        if self._extractor is None:
            from engram.memory_extractor import MemoryExtractor
            self._extractor = MemoryExtractor(model=self._extract_model)
        return self._extractor

    # ------------------------------------------------------------------ #
    #  Core async API                                                      #
    # ------------------------------------------------------------------ #

    async def chat(self, messages: list[dict[str, Any]], model: str | None = None, **kwargs) -> Any:
        """Transparent LLM wrapper with auto-memory.

        1. Pre-call: recall relevant memories from last user msg
        2. Inject memories as system context into messages
        3. Forward to LLM via litellm
        4. Post-call: fire-and-forget extraction task
        5. Return LLM response unchanged

        Memory errors never break the LLM call (fail-open).
        """
        model = model or self._model
        augmented_messages = list(messages)  # copy — don't mutate caller's list

        # Pre-call: inject memory context
        if self._auto_recall:
            try:
                await self._ensure_stores()
                memories = await self._recall_for_context(messages)
                if memories:
                    augmented_messages = self._inject_memories(messages, memories)
            except Exception as e:
                logger.warning("EngramClient: pre-call recall failed (continuing): %s", e)

        # LLM call — always runs
        response = await litellm.acompletion(model=model, messages=augmented_messages, **kwargs)

        # Post-call: fire-and-forget extraction
        if self._auto_extract:
            try:
                user_msg = _extract_last_user_content(messages)
                assistant_msg = response.choices[0].message.content or ""
                task = asyncio.create_task(self._extract_and_store(user_msg, assistant_msg))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
            except Exception as e:
                logger.warning("EngramClient: post-call task creation failed: %s", e)

        return response

    async def remember(
        self,
        content: str,
        memory_type: str = "fact",
        priority: int = 5,
        tags: list[str] | None = None,
    ) -> str:
        """Explicitly store a memory. Returns memory ID."""
        await self._ensure_stores()
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            mem_type = MemoryType.FACT
        return await self._episodic.remember(
            content=content,
            memory_type=mem_type,
            priority=priority,
            tags=tags or [],
        )

    async def recall(self, query: str, limit: int = 5) -> list[Any]:
        """Search episodic memories by semantic similarity."""
        await self._ensure_stores()
        return await self._episodic.search(query, limit=limit)

    async def think(self, question: str) -> str:
        """Reason over memories to answer a question."""
        await self._ensure_engine()
        return await self._engine.think(question)

    async def close(self) -> None:
        """Close graph backend and release resources (A-C1: cancel pending bg tasks)."""
        # Cancel and await all tracked background extraction tasks
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()
        if self._graph is not None:
            try:
                await self._graph.close()
            except Exception as e:
                logger.warning("EngramClient: close failed: %s", e)
        self._episodic = None
        self._graph = None
        self._engine = None

    # ------------------------------------------------------------------ #
    #  Sync wrappers                                                       #
    # ------------------------------------------------------------------ #

    def chat_sync(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        """Synchronous wrapper for chat()."""
        return run_async(self.chat(messages, **kwargs))

    def remember_sync(self, content: str, **kwargs) -> str:
        """Synchronous wrapper for remember()."""
        return run_async(self.remember(content, **kwargs))

    def recall_sync(self, query: str, **kwargs) -> list[Any]:
        """Synchronous wrapper for recall()."""
        return run_async(self.recall(query, **kwargs))

    def think_sync(self, question: str) -> str:
        """Synchronous wrapper for think()."""
        return run_async(self.think(question))

    # ------------------------------------------------------------------ #
    #  Context manager                                                     #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "EngramClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def __enter__(self) -> "EngramClient":
        """A-C3: Sync context manager — allows `with EngramClient() as client:`."""
        return self

    def __exit__(self, *_: Any) -> None:
        """A-C3: Sync exit — delegates close() through run_async."""
        run_async(self.close())

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    async def _recall_for_context(self, messages: list[dict[str, Any]]) -> str:
        """Search episodic and semantic memory using last user message as query.

        Returns formatted context string (max _MAX_MEMORY_CONTEXT_CHARS).
        """
        query = _extract_last_user_content(messages)
        if not query:
            return ""

        lines: list[str] = []

        # Episodic search
        try:
            episodic_results = await self._episodic.search(query, limit=5)
            for mem in episodic_results:
                lines.append(f"[{mem.memory_type.value}] {mem.content}")
        except Exception as e:
            logger.debug("Episodic search failed: %s", e)

        # Semantic graph keyword search
        try:
            graph_nodes = await self._graph.query(query)
            for node in graph_nodes[:3]:
                lines.append(f"[entity:{node.type}] {node.name}")
        except Exception as e:
            logger.debug("Graph query failed: %s", e)

        if not lines:
            return ""

        context = "\n".join(f"- {line}" for line in lines)
        # Cap to avoid bloating context window
        return context[:_MAX_MEMORY_CONTEXT_CHARS]

    def _inject_memories(
        self, messages: list[dict[str, Any]], context: str
    ) -> list[dict[str, Any]]:
        """Prepend or merge memory context as a system message.

        If first message is already a system message, appends context to it.
        Otherwise inserts a new system message at the front.
        Returns a new list — does not mutate original.
        """
        memory_block = f"Relevant memories:\n{context}"
        msgs = list(messages)

        if msgs and msgs[0].get("role") == "system":
            existing = msgs[0].get("content", "")
            msgs[0] = {**msgs[0], "content": f"{existing}\n\n{memory_block}"}
        else:
            msgs.insert(0, {"role": "system", "content": memory_block})

        return msgs

    async def _extract_and_store(self, user_msg: str, assistant_msg: str) -> None:
        """Background task: extract facts from conversation turn and store them.

        Runs deduplication before storage. All errors are caught and logged.
        """
        try:
            await self._ensure_stores()
            extractor = self._ensure_extractor()
            items = await extractor.extract(user_msg, assistant_msg)
            if not items:
                return

            # Dedup: skip items too similar to existing memories
            new_items = await self._dedup_memories(items)
            if not new_items:
                return

            # Batch store
            await self._episodic.remember_batch(new_items)
            logger.debug("EngramClient: stored %d extracted memories", len(new_items))
        except Exception as e:
            logger.warning("EngramClient: background extraction failed: %s", e)

    async def _dedup_memories(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove items that are too similar to existing memories.

        Uses ChromaDB cosine distance: distance < 0.15 means ~similarity > 0.85.
        """
        new_items: list[dict[str, Any]] = []
        for item in items:
            try:
                existing = await self._episodic.search(item["content"], limit=1)
                if existing:
                    # ChromaDB cosine distance — lower = more similar
                    # EpisodicMemory doesn't expose distance, so we do a basic
                    # content-overlap check as a lightweight dedup heuristic.
                    # A full distance check would require raw ChromaDB access.
                    existing_content = existing[0].content.lower()
                    candidate = item["content"].lower()
                    # Skip if content is very similar (> 85% word overlap)
                    if _high_overlap(candidate, existing_content, threshold=0.85):
                        logger.debug("Dedup skip: '%s'", item["content"][:60])
                        continue
            except Exception:
                pass  # fail-open: if dedup check fails, store the item
            new_items.append(item)
        return new_items


# ------------------------------------------------------------------ #
#  Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _extract_last_user_content(messages: list[dict[str, Any]]) -> str:
    """Return content of the last user message in the list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle list-of-parts format (OpenAI vision API)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def _high_overlap(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two strings share > threshold fraction of words (Jaccard)."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return (intersection / union) >= threshold
