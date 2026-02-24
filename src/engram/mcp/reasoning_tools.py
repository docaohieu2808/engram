"""MCP tools for reasoning and status: think, status, summarize."""

from __future__ import annotations


def register(mcp, get_engine, get_episodic, get_graph) -> None:
    """Register reasoning MCP tools on the FastMCP instance."""

    @mcp.tool()
    async def engram_think(question: str) -> str:
        """Combined reasoning across both episodic and semantic memory.

        Searches vector DB for relevant experiences, traverses knowledge graph
        for entity relationships, then synthesizes an answer using LLM.

        Args:
            question: Question to reason about (e.g. "What database issues happened recently?")
        """
        engine = get_engine()
        return await engine.think(question)

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
