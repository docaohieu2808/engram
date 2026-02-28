"""Data Constitution for Engram — immutable rules injected into every LLM reasoning prompt.

Three laws governing memory system behavior:
I.   Never return memories from a different namespace (tenant isolation is absolute)
II.  Never fabricate memories — if nothing relevant exists, say so honestly
III. Every memory access is logged — no deletion without explicit user/admin action

Constitution is loaded from ~/.engram/constitution.md or uses built-in defaults.
Hash is verified at startup; warns if file was modified externally.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("engram")

# Built-in default constitution (used if no file exists)
DEFAULT_CONSTITUTION = """# Engram Data Constitution

## Law I — Namespace Isolation
Never return memories belonging to a different tenant or namespace.
Namespace isolation is absolute. Cross-namespace access requires explicit authorization.

## Law II — No Fabrication
Never fabricate, hallucinate, or invent memories that do not exist in the store.
If no relevant memory is found, state so honestly. Do not fill gaps with speculation.
In synthesis, only reference facts present in the retrieved memories.

## Law III — Audit & Consent
Every memory access must be logged. No memory may be deleted without explicit
user or admin action. The creator has full audit rights over all memory operations.
"""

# Prefix injected into LLM prompts (compact version for token efficiency)
CONSTITUTION_PROMPT_PREFIX = """[CONSTITUTION — IMMUTABLE RULES]
1. NEVER return memories from a different namespace. Namespace isolation is absolute.
2. NEVER fabricate memories (inventing events, dates, quotes that didn't happen). You MAY and SHOULD reason, analyze, and give advice using available context + general knowledge.
3. Every memory access is logged. Creator has full audit rights.
[END CONSTITUTION]

"""


def get_constitution_path() -> Path:
    """Return path to constitution file."""
    return Path.home() / ".engram" / "constitution.md"


def load_constitution() -> str:
    """Load constitution from file or return built-in default.

    Creates default file if it doesn't exist.
    """
    path = get_constitution_path()
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("constitution: failed to read %s, using default", path)
            return DEFAULT_CONSTITUTION

    # Create default constitution file
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONSTITUTION, encoding="utf-8")
        logger.info("constitution: created default at %s", path)
    except OSError:
        logger.warning("constitution: failed to write default to %s", path)

    return DEFAULT_CONSTITUTION


def compute_constitution_hash(content: str) -> str:
    """Compute SHA-256 hash of constitution content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def verify_constitution(stored_hash: str | None = None) -> tuple[bool, str]:
    """Verify constitution file integrity against stored hash.

    Returns (is_valid, current_hash). If no stored hash, always returns (True, hash).
    """
    content = load_constitution()
    current_hash = compute_constitution_hash(content)

    if stored_hash is None:
        return True, current_hash

    is_valid = current_hash == stored_hash
    if not is_valid:
        logger.warning(
            "constitution: hash mismatch (stored=%s, current=%s) — file was modified externally",
            stored_hash, current_hash,
        )
    return is_valid, current_hash


def get_constitution_prompt_prefix() -> str:
    """Return the compact constitution prefix for LLM prompt injection."""
    return CONSTITUTION_PROMPT_PREFIX
