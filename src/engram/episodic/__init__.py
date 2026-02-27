"""Episodic memory - vector DB storage for experiences and context."""

from __future__ import annotations

from engram.episodic.store import EpisodicStore

__all__ = ["EpisodicStore", "make_episodic_backend"]


def make_episodic_backend(config: "EpisodicConfig"):  # type: ignore[name-defined]
    """Factory: return the appropriate EpisodicBackend for the given config.

    Lazy-imports backend implementations so unused dependencies are never loaded.

    Args:
        config: EpisodicConfig instance with at least `mode` and `path`/`host`/`port`.

    Returns:
        An uninitialised backend instance (call `await backend.initialize(...)` before use).
    """
    from engram.config import EpisodicConfig  # local import to avoid circular imports

    mode = getattr(config, "mode", "embedded")
    if mode == "http":
        from engram.episodic.chromadb_http_backend import ChromaDBHttpBackend
        return ChromaDBHttpBackend(
            host=getattr(config, "host", "localhost"),
            port=getattr(config, "port", 8000),
        )
    # Default: embedded PersistentClient
    from engram.episodic.chromadb_backend import ChromaDBBackend
    return ChromaDBBackend(db_path=config.path)
