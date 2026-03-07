"""Content quality filters for entity extraction pipeline.

Prevents junk content (code snippets, file paths, debug noise, system messages)
from being sent to the LLM for entity extraction.
"""

from __future__ import annotations

import re

# Patterns that indicate content is clearly code
_CODE_PATTERNS = re.compile(
    r"```|"
    r"\bdef |class |import |async def |function |const |let |var |\breturn |"
    r"elif |except |finally |lambda ",
    re.MULTILINE,
)

# File paths and media attachments
_PATH_PATTERNS = re.compile(
    r"\[media attached|"
    r"(?:^|[\s\"'])(?:/home/|/usr/|/var/|/etc/|C:\\\\|\.\.?/)|"
    r"\.(png|jpg|jpeg|gif|mp4|mp3|pdf|docx|xlsx)\b",
    re.IGNORECASE,
)

# Shell/CLI commands at line start
_COMMAND_PATTERNS = re.compile(
    r"^\s*(?:git |systemctl |docker |curl |wget |sudo |apt |pip |npm |yarn |pnpm )",
    re.MULTILINE,
)

# System/debug noise markers
_SYSTEM_PATTERNS = re.compile(
    r"\[\[reply_to|\bhook\b.*\bpgrep\b|\(line \d+\)|^\[system\]",
    re.IGNORECASE | re.MULTILINE,
)


def should_extract(content: str) -> bool:
    """Return False if content is clearly junk — pure code blocks, debug noise.

    Conservative: when in doubt, return True to allow extraction.
    Mixed content (natural language + code/paths/commands) is allowed through
    since it often contains important context worth remembering.

    Args:
        content: Raw message content string.

    Returns:
        True if content is worth sending to LLM for entity extraction.
    """
    if not content or len(content.strip()) < 30:
        return False
    # Only skip if content is PREDOMINANTLY code/noise, not mixed content
    stripped = content.strip()
    lines = stripped.splitlines()
    if not lines:
        return False
    # Skip pure code blocks (>80% of lines match code/command patterns)
    noise_lines = sum(
        1 for line in lines
        if _CODE_PATTERNS.search(line) or _COMMAND_PATTERNS.search(line)
    )
    if len(lines) > 3 and noise_lines / len(lines) > 0.8:
        return False
    if _SYSTEM_PATTERNS.search(content):
        return False
    return True
