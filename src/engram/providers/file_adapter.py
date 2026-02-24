"""File-based adapter for markdown/text folder memory stores (OpenClaw, Obsidian, etc.)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from engram.providers.base import MemoryProvider, ProviderResult

logger = logging.getLogger("engram.providers.file")


class FileAdapter(MemoryProvider):
    """Searches markdown/text files in a directory by keyword matching."""

    def __init__(
        self,
        name: str,
        path: str,
        pattern: str = "*.md",
        **kwargs: Any,
    ):
        super().__init__(name=name, provider_type="file", **kwargs)
        self.path = Path(os.path.expanduser(path))
        self.pattern = pattern

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        if not self.path.exists():
            return []

        query_lower = query.lower()
        query_words = query_lower.split()
        scored: list[tuple[float, str, str]] = []

        for filepath in self.path.rglob(self.pattern):
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            text_lower = text.lower()
            # Score by number of query words found in file
            hits = sum(1 for w in query_words if w in text_lower)
            if hits == 0:
                continue

            score = hits / len(query_words) if query_words else 0.0

            # Extract best matching paragraph (first match context)
            snippet = _extract_snippet(text, query_words)
            scored.append((score, filepath.name, snippet))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, filename, snippet in scored[:limit]:
            results.append(ProviderResult(
                content=snippet,
                score=score,
                source=self.name,
                metadata={"file": filename},
            ))

        if self.debug:
            logger.debug("[%s] found %d matches for '%s'", self.name, len(results), query)

        return results

    async def health(self) -> bool:
        return self.path.exists() and self.path.is_dir()


def _extract_snippet(text: str, query_words: list[str], max_len: int = 500) -> str:
    """Extract a snippet around the first query word match."""
    text_lower = text.lower()
    best_pos = len(text)

    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1 and pos < best_pos:
            best_pos = pos

    if best_pos == len(text):
        # No match found — return beginning
        return text[:max_len].strip()

    # Expand to paragraph boundaries
    start = text.rfind("\n\n", 0, best_pos)
    start = start + 2 if start != -1 else max(0, best_pos - 100)
    end = text.find("\n\n", best_pos)
    end = end if end != -1 else min(len(text), best_pos + 400)

    snippet = text[start:end].strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet
