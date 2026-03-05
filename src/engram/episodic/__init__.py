"""Episodic memory - vector DB storage for experiences and context."""

from __future__ import annotations

from engram.episodic.store import EpisodicStore

__all__ = ["EpisodicStore", "make_episodic_backend"]


def make_episodic_backend(config: "EpisodicConfig"):  # type: ignore[name-defined]
    """Factory: return the appropriate EpisodicBackend for the given config.

    Only Qdrant is supported as of v0.5. ChromaDB support has been removed.

    Modes:
    - ``mode: qdrant`` (default) — connects to a remote Qdrant server via HTTP.
    - ``mode: embedded`` — uses Qdrant's local file-based embedded mode; requires
      ``path`` to be set (e.g. ``~/.engram/qdrant``). No server needed.

    Args:
        config: EpisodicConfig instance.

    Returns:
        An uninitialised QdrantBackend instance (call ``await backend.initialize(...)`` before use).

    Raises:
        ValueError: If provider is "chromadb" or "chromadb_http" (removed in v0.5).
    """
    from engram.config import EpisodicConfig  # local import to avoid circular imports

    provider = getattr(config, "provider", "qdrant")
    mode = getattr(config, "mode", "qdrant")

    if provider in ("chromadb", "chromadb_http"):
        raise ValueError("ChromaDB support was removed in v0.5. Use provider: qdrant")

    from engram.episodic.qdrant_backend import QdrantBackend

    if mode == "embedded":
        # Local file-based embedded Qdrant — no server needed
        path = getattr(config, "path", "~/.engram/qdrant")
        return QdrantBackend(path=path)

    # Remote server mode (default)
    return QdrantBackend(
        host=getattr(config, "host", "localhost"),
        port=getattr(config, "port", 6333),
        api_key=getattr(config, "api_key", None),
    )
