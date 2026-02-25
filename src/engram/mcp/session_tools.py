"""MCP tools for session lifecycle: start, end, summary, context."""

from __future__ import annotations

from engram.models import MemoryType


def register(mcp, get_session_store, get_episodic) -> None:
    """Register session MCP tools on the FastMCP instance."""

    @mcp.tool()
    async def engram_session_start(namespace: str = "default") -> str:
        """Start a new memory session. All subsequent memories are tagged with session_id.

        Args:
            namespace: Namespace to associate the session with
        """
        store = get_session_store()
        session = store.start(namespace=namespace)
        return f"Session started (id={session.id[:8]})"

    @mcp.tool()
    async def engram_session_end() -> str:
        """End the current session without a summary. Use engram_session_summary for structured closure."""
        store = get_session_store()
        session = store.end()
        if not session:
            return "No active session."
        return f"Session ended (id={session.id[:8]}, started={session.started_at[:19]})"

    @mcp.tool()
    async def engram_session_summary(
        goal: str,
        discoveries: list[str],
        accomplished: list[str],
        files: list[str] | None = None,
    ) -> str:
        """End the current session with a structured summary stored as a memory.

        This is the recommended way to close a session. The summary is stored as a
        DECISION memory for future context recovery.

        Args:
            goal: What was being worked on this session
            discoveries: Key findings, decisions, or insights from this session
            accomplished: List of things completed in this session
            files: Optional list of file paths touched during the session
        """
        sess_store = get_session_store()
        session = sess_store.end(
            goal=goal, discoveries=discoveries,
            accomplished=accomplished, files=files or [],
        )
        if not session:
            return "No active session to summarize."

        # Store summary as a decision memory
        episodic = get_episodic()
        summary_lines = [f"Session summary: {goal}"]
        if accomplished:
            summary_lines.append(f"Accomplished: {'; '.join(accomplished)}")
        if discoveries:
            summary_lines.append(f"Discoveries: {'; '.join(discoveries)}")
        if files:
            summary_lines.append(f"Files: {', '.join(files)}")
        content = "\n".join(summary_lines)

        await episodic.remember(
            content,
            memory_type=MemoryType.DECISION,
            tags=["session-summary"],
            metadata={"session_id": session.id},
        )
        return f"Session summarized and stored (id={session.id[:8]})"

    @mcp.tool()
    async def engram_session_context(limit: int = 5) -> str:
        """Return context from recent past sessions to orient a new session.

        Call this at the start of a new session or after context compaction
        to recover what was done previously.

        Args:
            limit: Number of recent sessions to return (default 5)
        """
        store = get_session_store()
        sessions = store.get_recent(n=limit)
        if not sessions:
            return "No previous sessions found."
        lines = []
        for s in sessions:
            lines.append(f"[{s.started_at[:10]}] {s.goal or 'no goal'}")
            for a in s.accomplished[:3]:
                lines.append(f"  ✓ {a}")
            if s.discoveries:
                for d in s.discoveries[:2]:
                    lines.append(f"  → {d}")
        return "\n".join(lines)
