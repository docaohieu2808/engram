"""Live model list fetchers — call provider REST APIs directly.

Only returns currently active models (retired ones are automatically excluded
by the provider). Falls back to empty list on API failure (no key, timeout, etc.).
Each fetcher accepts an api_key param (resolved from config + env by caller).
"""

from __future__ import annotations

# Keywords that indicate non-text-generation models
_NON_TEXT_KW = frozenset({
    "image", "vision", "audio", "realtime", "tts", "transcribe", "search",
    "embed", "live", "native-audio", "computer-use", "customtools",
    "gemma", "learnlm", "aqa", "bison", "gecko", "robotics",
    "deep-research", "nano-banana",
})


# Curated list for OAuth tokens (can't call /v1/models)
_ANTHROPIC_CURATED = sorted([
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-5-20250514",
    "anthropic/claude-sonnet-4-5-20250514",
    "anthropic/claude-haiku-4-5-20251001",
])


def _is_text_model(model_id: str) -> bool:
    """Filter out non-text models by keyword."""
    low = model_id.lower()
    return not any(kw in low for kw in _NON_TEXT_KW)


async def fetch_anthropic_models(api_key: str = "") -> list[str]:
    """Fetch active models from Anthropic API.

    Note: OAuth tokens (sk-ant-oat*) don't support /v1/models.
    Only standard API keys (sk-ant-api*) work.
    """
    import httpx

    if not api_key:
        return []
    # OAuth tokens can't list models — return curated list
    if "oat" in api_key[:15]:
        return _ANTHROPIC_CURATED
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            models = []
            url = "https://api.anthropic.com/v1/models?limit=100"
            while url:
                resp = await client.get(url, headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                })
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("data", []):
                    mid = m.get("id", "")
                    if _is_text_model(mid):
                        models.append(f"anthropic/{mid}")
                if data.get("has_more") and data.get("last_id"):
                    url = f"https://api.anthropic.com/v1/models?limit=100&after_id={data['last_id']}"
                else:
                    url = None
            return sorted(models)
    except Exception:
        return []


async def fetch_gemini_models(api_key: str = "") -> list[str]:
    """Fetch active models from Google Gemini API.

    Returns only the best variant per version (no -001, -lite, -latest aliases).
    """
    import httpx

    if not api_key:
        return []
    _SKIP_SUFFIXES = ("-001", "-002", "-lite", "-latest")
    _SKIP_PREFIXES = ("gemini-2.0-",)  # 2.0 series retired by Google

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            models = []
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
            while url:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models", []):
                    name = m.get("name", "")  # "models/gemini-2.5-flash"
                    actions = m.get("supportedGenerationMethods", [])
                    if "generateContent" not in actions:
                        continue
                    model_id = name.replace("models/", "")
                    if not _is_text_model(model_id):
                        continue
                    if any(model_id.startswith(p) for p in _SKIP_PREFIXES):
                        continue
                    if any(model_id.endswith(s) for s in _SKIP_SUFFIXES):
                        continue
                    if "latest" in model_id or "-lite" in model_id:
                        continue
                    models.append(f"gemini/{model_id}")
                next_token = data.get("nextPageToken")
                if next_token:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100&pageToken={next_token}"
                else:
                    url = None
            return sorted(models)
    except Exception:
        return []


async def fetch_openai_models(api_key: str = "") -> list[str]:
    """Fetch active models from OpenAI API."""
    import httpx

    if not api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            _OPENAI_PREFIXES = ("gpt-4", "gpt-5", "o1", "o3", "o4")
            models = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid.startswith(_OPENAI_PREFIXES) and _is_text_model(mid):
                    models.append(mid)
            return sorted(models)
    except Exception:
        return []


PROVIDER_FETCHERS = {
    "anthropic": fetch_anthropic_models,
    "gemini": fetch_gemini_models,
    "openai": fetch_openai_models,
}
