"""Embedding helpers for episodic memory store.

Uses gemini-embedding-001 (3072d) exclusively.
Supports multiple keys via GEMINI_API_KEY + GEMINI_API_KEY_FALLBACK env vars.

Key strategy (config.yaml → embedding.key_strategy):
  - "failover" (default): use primary key, switch to fallback only on failure
  - "round-robin": rotate keys evenly across calls to spread quota usage
"""

from __future__ import annotations

import itertools
import logging
import os

EMBEDDING_DIM = 3072
logger = logging.getLogger("engram")

# Round-robin counter — persists across calls within the same process
_rr_counter = itertools.count()


def _get_api_keys() -> list[str]:
    """Return list of available Gemini API keys (primary + fallback)."""
    keys = []
    primary = os.environ.get("GEMINI_API_KEY")
    if primary:
        keys.append(primary)
    fallback = os.environ.get("GEMINI_API_KEY_FALLBACK")
    if fallback:
        keys.append(fallback)
    return keys


def _get_key_strategy() -> str:
    """Read key_strategy from config, fall back to env, then default."""
    # Env override takes precedence (for systemd services)
    env_val = os.environ.get("GEMINI_KEY_STRATEGY")
    if env_val:
        return env_val.lower()
    # Read from config file
    try:
        from engram.config import load_config
        cfg = load_config()
        return cfg.embedding.key_strategy.lower()
    except Exception:
        return "failover"


def _order_keys(keys: list[str]) -> list[str]:
    """Order keys based on configured strategy (failover or round-robin)."""
    if len(keys) <= 1:
        return keys
    strategy = _get_key_strategy()
    if strategy == "round-robin":
        start = next(_rr_counter) % len(keys)
        return keys[start:] + keys[:start]
    # failover: natural order (primary first)
    return keys


def _get_embeddings(model: str, texts: list[str], expected_dim: int | None = None) -> list[list[float]]:
    """Generate embeddings via litellm with key rotation on failure."""
    import litellm

    keys = _get_api_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY configured. Set GEMINI_API_KEY env var.")

    ordered = _order_keys(keys)
    last_err = None
    for i, key in enumerate(ordered):
        try:
            response = litellm.embedding(model=model, input=texts, api_key=key)
            return [item["embedding"] for item in response.data]
        except litellm.AuthenticationError as e:
            last_err = e
            logger.warning("Gemini key #%d failed auth, trying next...", i + 1)
        except Exception as e:
            last_err = e
            logger.warning("Gemini key #%d error: %s, trying next...", i + 1, type(e).__name__)

    raise RuntimeError(f"All {len(keys)} Gemini API keys failed. Last error: {last_err}")


def _detect_embedding_dim_from_model(embed_model: str) -> int | None:
    """Return 3072 for gemini-embedding-001, None otherwise."""
    model_name = embed_model.split("/")[-1] if "/" in embed_model else embed_model
    if model_name == "gemini-embedding-001":
        return EMBEDDING_DIM
    return None
