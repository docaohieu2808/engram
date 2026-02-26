"""Input sanitization utilities for engram.

Provides content validation and sanitization before storing to memory backends.
"""

from __future__ import annotations

import re
import secrets

from engram.models import MemoryType

# Control characters to strip: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F
# Keep: 0x09 (tab), 0x0A (newline), 0x0D (carriage return)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PRIVATE_TAG_RE = re.compile(r"<private>.*?</private>", re.DOTALL | re.IGNORECASE)


def sanitize_content(text: str, max_length: int = 10240) -> str:
    """Strip control characters and enforce maximum content length.

    Args:
        text: Raw input content.
        max_length: Maximum allowed byte length (default 10240 = 10KB).

    Returns:
        Sanitized string with control characters removed.

    Raises:
        ValueError: If content exceeds max_length after sanitization.
    """
    if not isinstance(text, str):
        raise TypeError(f"Content must be a string, got {type(text).__name__}")

    # Redact private tagged content before any other processing
    text = _PRIVATE_TAG_RE.sub("[REDACTED]", text)

    # Strip disallowed control characters (keep tab, newline, carriage return)
    cleaned = _CONTROL_CHAR_RE.sub("", text)

    # Enforce max length on byte representation to prevent memory exhaustion
    encoded = cleaned.encode("utf-8")
    if len(encoded) > max_length:
        raise ValueError(
            f"Content length {len(encoded)} bytes exceeds maximum {max_length} bytes"
        )

    return cleaned


def sanitize_llm_input(text: str, max_len: int = 2000) -> str:
    """Wrap user input in randomized delimiters and strip control chars for LLM prompts.

    Defends against prompt injection by:
    1. Stripping control characters that could break prompt structure.
    2. Truncating to max_len to prevent token flooding.
    3. Wrapping in randomized delimiters so attackers cannot pre-craft the end
       delimiter to escape the user-input boundary.

    Args:
        text: Raw user-controlled input.
        max_len: Maximum character length before truncation (default 2000).

    Returns:
        Sanitized string wrapped in randomized delimiter tokens.
    """
    if not isinstance(text, str):
        text = str(text)
    cleaned = _CONTROL_CHAR_RE.sub("", text)[:max_len]
    token = secrets.token_hex(4)
    return f"---USER-INPUT-{token}-START---\n{cleaned}\n---USER-INPUT-{token}-END---"


def validate_memory_type(value: str) -> MemoryType:
    """Parse memory type string into MemoryType enum safely.

    Args:
        value: String representation of memory type.

    Returns:
        MemoryType enum member.

    Raises:
        ValueError: If value is not a valid MemoryType.
    """
    try:
        return MemoryType(value)
    except ValueError:
        valid = [m.value for m in MemoryType]
        raise ValueError(f"Invalid memory_type '{value}'. Must be one of: {valid}")
