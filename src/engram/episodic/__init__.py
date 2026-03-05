"""Episodic memory - vector DB storage for experiences and context."""

from __future__ import annotations

from engram.episodic.store import EpisodicStore

__all__ = ["EpisodicStore", "make_episodic_backend"]


def make_episodic_backend(config: "EpisodicConfig"):  # type: ignore[name-defined]
    """Factory: return the appropriate EpisodicBackend for the given config.

    Only Qdrant is supported as of v0.5. ChromaDB support has been removed.

    Args:
        config: EpisodicConfig instance with at least `host`/`port`/`api_key`.

    Returns:
        An uninitialised QdrantBackend instance (call `await backend.initialize(...)` before use).

    Raises:
        ValueError: If provider is "chromadb" or "chromadb_http" (removed in v0.5).
    """
    from engram.config import EpisodicConfig  # local import to avoid circular imports

    provider = getattr(config, "provider", "qdrant")
    mode = getattr(config, "mode", "qdrant")

    if provider in ("chromadb", "chromadb_http") or mode in ("embedded", "http"):
        raise ValueError(
            f"ChromaDB support was removed in v0.5. Use provider: qdrant"
        )

    from engram.episodic.qdrant_backend import QdrantBackend
    return QdrantBackend(
        host=getattr(config, "host", "localhost"),
        port=getattr(config, "port", 6333),
        api_key=getattr(config, "api_key", None),
    )
