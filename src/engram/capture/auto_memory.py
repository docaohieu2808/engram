"""Auto-memory detection — identify messages worth saving automatically.

Scans user messages for patterns indicating save-worthy content
(identity, preferences, explicit save requests) while skipping sensitive data.
"""

from __future__ import annotations

import re

from engram.models import MemoryCandidate, MemoryType

# (pattern, importance, category, memory_type)
# Ordered by priority — first match wins for each pattern group
SAVE_PATTERNS: list[tuple[re.Pattern, int, str, MemoryType]] = [
    # Manual save: "Save: ..." prefix
    (re.compile(r"^save:\s*(.+)", re.IGNORECASE), 4, "manual", MemoryType.FACT),
    # Identity patterns
    (re.compile(r"\b(my name is|tên tôi là|i am called|tôi tên)\b", re.IGNORECASE), 3, "identity", MemoryType.FACT),
    # Preference patterns
    (re.compile(r"\b(i prefer|tôi thích|i like|i hate|tôi ghét|i love|tôi yêu)\b", re.IGNORECASE), 3, "preference", MemoryType.PREFERENCE),
    # Explicit memory requests
    (re.compile(r"\b(remember that|nhớ là|note that|ghi nhớ|don't forget)\b", re.IGNORECASE), 3, "explicit", MemoryType.FACT),
    # Behavioral patterns
    (re.compile(r"\b(i always|i never|tôi luôn|tôi không bao giờ)\b", re.IGNORECASE), 2, "pattern", MemoryType.PREFERENCE),
    # Decision patterns
    (re.compile(r"\b(i decided|we decided|quyết định|i chose|we chose)\b", re.IGNORECASE), 3, "decision", MemoryType.DECISION),
]

# Patterns that should NEVER be saved (sensitive data)
SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(password|mật khẩu|passwd)\b", re.IGNORECASE),
    re.compile(r"\b(api[_\s]?key|secret[_\s]?key|access[_\s]?token)\b", re.IGNORECASE),
    re.compile(r"\b(credential|private[_\s]?key|ssh[_\s]?key)\b", re.IGNORECASE),
    re.compile(r"\b(credit[_\s]?card|cvv|ssn|social security)\b", re.IGNORECASE),
]


def detect_candidates(message: str) -> list[MemoryCandidate]:
    """Analyze a user message for auto-save candidates.

    Returns list of MemoryCandidate objects. Empty list if nothing worth saving
    or if sensitive data detected.
    """
    msg = message.strip()
    if not msg or len(msg) < 5:
        return []

    # Check for sensitive data — skip entirely if found
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(msg):
            return []

    candidates: list[MemoryCandidate] = []
    for pattern, importance, category, memory_type in SAVE_PATTERNS:
        match = pattern.search(msg)
        if match:
            # For manual save, extract content after "Save:" prefix
            if category == "manual":
                content = match.group(1).strip()
            else:
                content = msg
            candidates.append(MemoryCandidate(
                content=content,
                importance=importance,
                category=category,
                memory_type=memory_type,
            ))

    return candidates
