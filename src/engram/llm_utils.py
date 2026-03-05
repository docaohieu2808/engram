"""Shared LLM utilities — fallback chain and model detection."""

from __future__ import annotations

import logging
from typing import Any

import litellm

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

# Fallback model chain: try each on quota/rate/auth errors
_MODEL_FALLBACK_CHAIN = [
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3-pro-preview",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
]


def is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower() or "anthropic" in model.lower()


async def _llm_call_with_fallback(kwargs: dict, primary_model: str | None = None) -> Any:
    """Call litellm with model fallback chain on quota/rate/auth errors."""
    primary = primary_model or kwargs.get("model", "")

    seen = {primary}
    fallback_models = [primary]
    for m in _MODEL_FALLBACK_CHAIN:
        if m not in seen:
            fallback_models.append(m)
            seen.add(m)

    last_exc: Exception | None = None
    for model in fallback_models:
        kwargs["model"] = model
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_retriable = any(k in err_str for k in (
                "429", "rate", "quota", "resource_exhausted",
                "auth", "api key", "403", "401",
                "404", "not found", "not_found",
            ))
            if is_retriable and model != fallback_models[-1]:
                logger.warning("LLM %s failed (%s), trying next model", model, type(e).__name__)
                continue
            raise
    raise last_exc  # type: ignore[misc]
