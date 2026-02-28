"""Live model list fetchers — call provider REST APIs directly.

Only returns currently active models (retired ones are automatically excluded
by the provider). Falls back to empty list on API failure (no key, timeout, etc.).
"""

from __future__ import annotations

# Keywords that indicate non-text-generation models
_NON_TEXT_KW = frozenset({
    "image", "vision", "audio", "realtime", "tts", "transcribe", "search",
    "embed", "live", "native-audio", "computer-use", "customtools",
    "gemma", "learnlm", "aqa", "bison", "gecko", "robotics",
    "deep-research", "nano-banana",
})


def _is_text_model(model_id: str) -> bool:
    """Filter out non-text models by keyword."""
    low = model_id.lower()
    return not any(kw in low for kw in _NON_TEXT_KW)


async def fetch_anthropic_models() -> list[str]:
    """Fetch active models from Anthropic API."""
    import os
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return []
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


async def fetch_gemini_models() -> list[str]:
    """Fetch active models from Google Gemini API."""
    import os
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []
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
                    if _is_text_model(model_id):
                        models.append(f"gemini/{model_id}")
                next_token = data.get("nextPageToken")
                if next_token:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100&pageToken={next_token}"
                else:
                    url = None
            return sorted(models)
    except Exception:
        return []


async def fetch_openai_models() -> list[str]:
    """Fetch active models from OpenAI API."""
    import os
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY", "")
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
