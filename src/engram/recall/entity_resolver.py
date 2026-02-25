"""Entity resolver — pronoun and temporal reference resolution.

Temporal resolution is pure regex (no LLM needed).
Pronoun resolution uses a cheap LLM (gemini-flash) with conversation context.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from engram.models import Entity, ResolvedText

logger = logging.getLogger("engram")

# --- Pronoun patterns (Vietnamese + English) ---

PRONOUN_MAP: dict[str, str] = {
    # Vietnamese
    "cô ấy": "female_reference",
    "anh ấy": "male_reference",
    "chị ấy": "female_reference",
    "nó": "neutral_reference",
    "họ": "plural_reference",
    "ông ấy": "male_elder_reference",
    "bà ấy": "female_elder_reference",
    # English
    "he": "male_reference",
    "she": "female_reference",
    "they": "plural_reference",
    "it": "neutral_reference",
    "him": "male_reference",
    "her": "female_reference",
}

# Compiled word-boundary patterns for pronoun detection
_PRONOUN_PATTERNS = [
    re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE | re.UNICODE)
    for p in PRONOUN_MAP
]


def _month_ago(d: datetime) -> str:
    """Compute date one month ago, handling month boundaries."""
    month = d.month - 1 or 12
    year = d.year if d.month > 1 else d.year - 1
    day = min(d.day, 28)  # safe for all months
    return datetime(year, month, day).strftime("%Y-%m-%d")


# --- Temporal patterns (Vietnamese + English) ---

TEMPORAL_PATTERNS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"\bhôm nay\b", re.IGNORECASE), lambda d: d.strftime("%Y-%m-%d")),
    (re.compile(r"\bhôm qua\b", re.IGNORECASE), lambda d: (d - timedelta(days=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\bhôm kia\b", re.IGNORECASE), lambda d: (d - timedelta(days=2)).strftime("%Y-%m-%d")),
    (re.compile(r"\bngày mai\b", re.IGNORECASE), lambda d: (d + timedelta(days=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\bngày kia\b", re.IGNORECASE), lambda d: (d + timedelta(days=2)).strftime("%Y-%m-%d")),
    (re.compile(r"\btuần trước\b", re.IGNORECASE), lambda d: (d - timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\btuần sau\b", re.IGNORECASE), lambda d: (d + timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\btháng trước\b", re.IGNORECASE), lambda d: _month_ago(d)),
    (re.compile(r"\bnăm ngoái\b", re.IGNORECASE), lambda d: str(d.year - 1)),
    # English
    (re.compile(r"\btoday\b", re.IGNORECASE), lambda d: d.strftime("%Y-%m-%d")),
    (re.compile(r"\byesterday\b", re.IGNORECASE), lambda d: (d - timedelta(days=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), lambda d: (d + timedelta(days=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\blast week\b", re.IGNORECASE), lambda d: (d - timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (re.compile(r"\blast month\b", re.IGNORECASE), lambda d: _month_ago(d)),
    (re.compile(r"\blast year\b", re.IGNORECASE), lambda d: str(d.year - 1)),
]


def resolve_temporal(text: str, reference_date: datetime | None = None) -> tuple[str, dict[str, str]]:
    """Replace temporal references with actual dates.

    Returns (resolved_text, mapping_of_original_to_date).
    Pure regex, no LLM needed.
    """
    ref = reference_date or datetime.now()
    resolved = text
    refs: dict[str, str] = {}

    for pattern, resolver in TEMPORAL_PATTERNS:
        match = pattern.search(resolved)
        if match:
            original = match.group(0)
            replacement = resolver(ref)
            refs[original] = replacement
            resolved = pattern.sub(replacement, resolved)

    return resolved, refs


def has_pronouns(text: str) -> bool:
    """Check if text contains any known pronouns."""
    text_lower = text.lower()
    return any(p.search(text_lower) for p in _PRONOUN_PATTERNS)


def _format_context(context: list[dict], max_messages: int = 10) -> str:
    """Format conversation context for LLM prompt."""
    recent = context[-max_messages:] if len(context) > max_messages else context
    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def resolve_pronouns(
    text: str,
    context: list[dict],
    model: str = "gemini/gemini-2.0-flash",
) -> ResolvedText:
    """Use LLM to resolve pronouns given conversation context.

    Falls back to original text if LLM call fails.
    """
    if not has_pronouns(text):
        return ResolvedText(original=text, resolved=text)

    if not context:
        return ResolvedText(original=text, resolved=text)

    prompt = (
        "Given this conversation context:\n"
        f"{_format_context(context)}\n\n"
        "Resolve all pronouns in this message to their actual names/entities:\n"
        f'"{text}"\n\n'
        'Return ONLY valid JSON: {"resolved": "...", "entities": [{"name": "...", "type": "person"}]}\n'
        "Only resolve pronouns clearly identifiable from context. If unsure, keep original."
    )

    try:
        import litellm
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        entities = [Entity(**e) for e in parsed.get("entities", [])]
        return ResolvedText(
            original=text,
            resolved=parsed.get("resolved", text),
            entities=entities,
        )
    except Exception as e:
        logger.debug("Pronoun resolution failed, using original text: %s", e)
        return ResolvedText(original=text, resolved=text)


async def resolve(
    text: str,
    context: list[dict] | None = None,
    reference_date: datetime | None = None,
    model: str = "gemini/gemini-2.0-flash",
    resolve_temporal_refs: bool = True,
    resolve_pronoun_refs: bool = True,
) -> ResolvedText:
    """Full resolution pipeline: temporal + pronoun.

    Temporal is applied first (regex), then pronoun (LLM).
    """
    resolved_text = text
    temporal_refs: dict[str, str] = {}
    entities: list[Entity] = []

    # Step 1: Temporal resolution (cheap, regex-based)
    if resolve_temporal_refs:
        resolved_text, temporal_refs = resolve_temporal(resolved_text, reference_date)

    # Step 2: Pronoun resolution (LLM-assisted)
    if resolve_pronoun_refs and context:
        pronoun_result = await resolve_pronouns(resolved_text, context, model)
        resolved_text = pronoun_result.resolved
        entities = pronoun_result.entities

    return ResolvedText(
        original=text,
        resolved=resolved_text,
        entities=entities,
        temporal_refs=temporal_refs,
    )
