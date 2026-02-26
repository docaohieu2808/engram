"""MCP tools for reasoning and status: think, ask, status, summarize."""

from __future__ import annotations


def register(mcp, get_engine, get_episodic, get_graph, get_providers=None) -> None:
    """Register reasoning MCP tools on the FastMCP instance."""

    @mcp.tool()
    async def engram_ask(query: str) -> str:
        """Smart query that auto-routes to recall or think based on intent.

        Use this as the default entry point instead of engram_recall/engram_think.
        Questions with why/how/explain/compare → think (LLM reasoning).
        Simple keyword lookups → recall (vector search + federated).

        Args:
            query: Any question or search query
        """
        from engram.providers.router import classify_intent
        intent = classify_intent(query)
        if intent == "think":
            engine = get_engine()
            result = await engine.think(query)
            text = result["answer"]
            if result.get("degraded"):
                text = "[degraded mode] " + text
            return text
        # Recall path: episodic + graph + federated, no LLM synthesis
        store = get_episodic()
        results = await store.search(query, limit=5)
        lines = []
        graph = get_graph()
        graph_nodes = await graph.query(query) if graph else []
        for node in graph_nodes[:3]:
            lines.append(f"[graph] {node.type}:{node.name}")
        if get_providers:
            from engram.providers.router import federated_search
            providers = get_providers()
            if providers:
                for r in await federated_search(query, providers, limit=3):
                    lines.append(f"[{r.source}] {r.content[:300]}")
        for mem in results:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}")
        return "\n".join(lines) if lines else "No memories found."

    @mcp.tool()
    async def engram_think(question: str) -> str:
        """Combined reasoning across both episodic and semantic memory.

        Searches vector DB for relevant experiences, traverses knowledge graph
        for entity relationships, then synthesizes an answer using LLM.

        Args:
            question: Question to reason about (e.g. "What database issues happened recently?")
        """
        engine = get_engine()
        result = await engine.think(question)
        text = result["answer"]
        if result.get("degraded"):
            text = "[degraded mode] " + text
        return text

    @mcp.tool()
    async def engram_summarize(count: int = 20, save: bool = False) -> str:
        """Summarize recent memories into key insights using LLM.

        Args:
            count: Number of recent memories to include in summary (default 20)
            save: If True, store the summary as a new memory with type=context
        """
        engine = get_engine()
        return await engine.summarize(n=count, save=save)

    @mcp.tool()
    async def engram_status() -> str:
        """Show engram memory statistics - counts for both episodic and semantic stores."""
        episodic = get_episodic()
        graph = get_graph()

        ep_stats = await episodic.stats()
        sem_stats = await graph.stats()

        lines = [
            "Engram Memory Status:",
            f"  Episodic: {ep_stats.get('count', 0)} memories",
            f"  Semantic: {sem_stats.get('node_count', 0)} nodes, {sem_stats.get('edge_count', 0)} edges",
        ]
        if "node_types" in sem_stats:
            for t, c in sem_stats["node_types"].items():
                lines.append(f"    {t}: {c}")
        return "\n".join(lines)
