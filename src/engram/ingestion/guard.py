"""Poisoning guard — prevent prompt injection in ingested content.

Checks content for suspicious patterns commonly used in prompt injection
attacks before allowing it to enter the memory store.
"""

from __future__ import annotations

import re

# Patterns commonly used in prompt injection attacks
SUSPICIOUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore (all |any )?(previous|prior|above) (instructions|prompts|rules)", re.IGNORECASE),
     "ignore instructions"),
    (re.compile(r"disregard (all|any) (prior|previous)", re.IGNORECASE),
     "disregard prior"),
    (re.compile(r"you are now\b", re.IGNORECASE),
     "identity override"),
    (re.compile(r"act as if\b", re.IGNORECASE),
     "role impersonation"),
    (re.compile(r"pretend (that |to be )", re.IGNORECASE),
     "role pretend"),
    (re.compile(r"(system prompt|system message|system instruction)", re.IGNORECASE),
     "system prompt reference"),
    (re.compile(r"<\|.*?\|>"),
     "special tokens"),
    (re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE),
     "chat template tokens"),
    (re.compile(r"(new instructions|override (the |all )?instructions)", re.IGNORECASE),
     "instruction override"),
    (re.compile(r"forget (everything|all|what)", re.IGNORECASE),
     "memory wipe attempt"),
]


def check_content(content: str) -> tuple[bool, str]:
    """Check if content is safe to ingest.

    Returns (is_safe, reason). If not safe, content should be rejected.
    """
    if not content or not content.strip():
        return True, "empty"

    for pattern, label in SUSPICIOUS_PATTERNS:
        if pattern.search(content):
            return False, f"Suspicious pattern: {label}"

    return True, "OK"
