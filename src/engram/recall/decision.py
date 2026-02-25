"""Decision layer — skip memory recall for trivial messages.

Checks if a message is trivial (greetings, acknowledgments, emoji-only)
and doesn't warrant a memory lookup.
"""

from __future__ import annotations

import re

# Patterns that indicate trivial messages not worth recalling memory for
TRIVIAL_PATTERNS = [
    r"^(ok|okay|oke|ổn|được|yes|no|không|có|ừ|ờ|uh|um)$",
    r"^(thanks|thank you|cảm ơn|cám ơn|tks|thx|ty)$",
    r"^(hello|hi|hey|chào|xin chào|yo|sup)$",
    r"^(bye|goodbye|tạm biệt|bye bye|bb)$",
    r"^[👍👌✅❤️🎉😊😄🙏💪🔥🤔😅😂🥲]+$",  # emoji-only messages
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRIVIAL_PATTERNS]


def should_skip_recall(message: str) -> bool:
    """Return True if message is trivial and doesn't need memory recall.

    Trivial = greetings, acks, emoji-only, or very short messages.
    """
    msg = message.strip()
    if not msg:
        return True
    if len(msg) < 2:
        return True
    msg_lower = msg.lower()
    return any(p.match(msg_lower) for p in _COMPILED)
