"""Resolve pronouns to named entities using conversation context.

Maps 'anh ấy', 'he', 'she' etc. to actual names from recent context.
This is a lightweight regex-based resolver (no LLM required) suitable
for the hot recall path. For full LLM-assisted resolution see entity_resolver.py.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Pronoun pattern groups — ordered longest match first to avoid partial hits
# ---------------------------------------------------------------------------

# Vietnamese pronouns → gender hint (used for future entity gender filtering)
_VI_PRONOUNS: list[tuple[re.Pattern, str]] = [
    # Multi-word first
    (re.compile(r"\banh ấy\b", re.IGNORECASE | re.UNICODE), "male"),
    (re.compile(r"\bchị ấy\b", re.IGNORECASE | re.UNICODE), "female"),
    (re.compile(r"\bcô ấy\b", re.IGNORECASE | re.UNICODE), "female"),
    (re.compile(r"\bông ấy\b", re.IGNORECASE | re.UNICODE), "male"),
    (re.compile(r"\bbà ấy\b", re.IGNORECASE | re.UNICODE), "female"),
    # Short forms (single word, match after multi-word to avoid partial clash)
    (re.compile(r"\bảnh\b", re.IGNORECASE | re.UNICODE), "male"),
    (re.compile(r"\bcổ\b", re.IGNORECASE | re.UNICODE), "female"),
    (re.compile(r"\bhắn\b", re.IGNORECASE | re.UNICODE), "neutral"),
    (re.compile(r"\bnó\b", re.IGNORECASE | re.UNICODE), "neutral"),
    # Plural — keep as-is (ambiguous), not resolved
]

# English pronouns → gender hint
_EN_PRONOUNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhis\b", re.IGNORECASE), "male"),
    (re.compile(r"\bhim\b", re.IGNORECASE), "male"),
    (re.compile(r"\bhe\b", re.IGNORECASE), "male"),
    (re.compile(r"\bhers\b", re.IGNORECASE), "female"),
    (re.compile(r"\bher\b", re.IGNORECASE), "female"),
    (re.compile(r"\bshe\b", re.IGNORECASE), "female"),
    # "it", "they", "them" — ambiguous, skip
]

# All resolvable pronouns (plural/ambiguous are excluded by design)
_ALL_PRONOUNS: list[tuple[re.Pattern, str]] = _VI_PRONOUNS + _EN_PRONOUNS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_pronouns(content: str, context_entities: list[str]) -> str:
    """Resolve pronouns in content using recently mentioned entities.

    Replaces unambiguous singular pronouns with the first entity in
    context_entities. Plural or genuinely ambiguous pronouns ('họ', 'they',
    'them', 'it') are left unchanged.

    Args:
        content: Text with possible pronouns.
        context_entities: Recently mentioned person names (most recent first).
                          If empty, content is returned unchanged.

    Returns:
        Content with pronouns resolved, or unchanged if no resolution possible.

    Example:
        >>> resolve_pronouns("anh ấy thích cà phê", ["Max"])
        "Max thích cà phê"
    """
    if not content or not context_entities:
        return content

    # Use the most recently mentioned entity as the resolution target
    target = context_entities[0]
    resolved = content

    for pattern, _gender in _ALL_PRONOUNS:
        resolved = pattern.sub(target, resolved)

    return resolved


def has_resolvable_pronouns(content: str) -> bool:
    """Return True if content contains any resolvable pronoun."""
    return any(p.search(content) for p, _ in _ALL_PRONOUNS)
