"""Format recalled memories by type for optimal LLM context injection.

Groups results by memory_type and formats with type labels.
Priority: preference > fact > lesson > decision > other
"""

from __future__ import annotations

from typing import Any

# Order determines output priority — lower index = higher priority
TYPE_PRIORITY: list[str] = [
    "preference",
    "fact",
    "lesson",
    "decision",
    "context",
    "observation",
]

_MAX_ENTRY_CHARS = 200


def _get_type(result: Any) -> str:
    """Extract memory_type string from a SearchResult or EpisodicMemory."""
    # SearchResult has .memory_type as str attribute
    if hasattr(result, "memory_type") and result.memory_type:
        mt = result.memory_type
        # EpisodicMemory stores MemoryType enum; SearchResult stores plain str
        if hasattr(mt, "value"):
            return mt.value
        return str(mt)
    # Fallback: check .metadata dict
    if hasattr(result, "metadata") and isinstance(result.metadata, dict):
        return result.metadata.get("memory_type", "memory") or "memory"
    return "memory"


def _get_content(result: Any) -> str:
    """Extract content string from a result object."""
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def format_for_llm(results: list, max_chars: int = 2000) -> str:
    """Format search results grouped by memory type.

    Args:
        results: List of objects with .content and .memory_type attributes
        max_chars: Max total characters in output

    Returns:
        Formatted string like:
        [preference] User prefers dark mode
        [fact] User lives in HCMC
        [lesson] Don't deploy on Friday
    """
    if not results:
        return ""

    # Group by type
    groups: dict[str, list[str]] = {}
    for r in results:
        mem_type = _get_type(r)
        # Normalize: types not in TYPE_PRIORITY become "memory"
        if mem_type not in TYPE_PRIORITY:
            mem_type = "memory"
        content = _get_content(r)
        # Flag outdated memories for LLM awareness
        meta = getattr(r, "metadata", None) or {}
        if meta.get("outdated") == "true":
            reason = meta.get("outdated_reason", "")
            content = f"[OUTDATED: {reason}] {content}" if reason else f"[OUTDATED] {content}"
        # Truncate individual entries
        if len(content) > _MAX_ENTRY_CHARS:
            content = content[:_MAX_ENTRY_CHARS - 3] + "..."
        groups.setdefault(mem_type, []).append(content)

    # Sort groups by TYPE_PRIORITY, unknown types ("memory") go last
    def _sort_key(t: str) -> int:
        try:
            return TYPE_PRIORITY.index(t)
        except ValueError:
            return len(TYPE_PRIORITY)

    lines: list[str] = []
    total_chars = 0

    for mem_type in sorted(groups.keys(), key=_sort_key):
        for content in groups[mem_type]:
            entry = f"[{mem_type}] {content}"
            if total_chars + len(entry) + 1 > max_chars:
                return "\n".join(lines)
            lines.append(entry)
            total_chars += len(entry) + 1  # +1 for newline

    return "\n".join(lines)
