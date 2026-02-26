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
    # Emoji-only messages: matches sequences of emoji including ZWJ sequences (M18)
    # \U000fe0f = variation selector, \U0000200d = ZWJ
    # Note: excludes bare digits/hash/asterisk to avoid matching "42" as emoji
    r"^[\U00002600-\U000027ff\U0001f000-\U0001faff\U0000fe0f\U0000200d\U00002194-\U00002199\U000025aa-\U000025fe\U00002614-\U00002615\U00002648-\U00002653\U0000267f\U00002693\U000026a1\U000026aa-\U000026ab\U000026bd-\U000026be\U000026c4-\U000026c5\U000026ce\U000026d4\U000026ea\U000026f2-\U000026f3\U000026f5\U000026fa\U000026fd\U00002702\U00002705\U00002708-\U0000270d\U0000270f\U00002712\U00002714\U00002716\U00002733-\U00002734\U00002744\U00002747\U0000274c\U0000274e\U00002753-\U00002755\U00002757\U00002763-\U00002764\U00002795-\U00002797\U000027a1\U000027b0\U000027bf\U00002934-\U00002935\U000025b6\U000025c0\U0000203c\U00002049\U000020e3\U000021aa\U000021a9]+$",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRIVIAL_PATTERNS]


def should_skip_recall(message: str) -> bool:
    """Return True if message is trivial and doesn't need memory recall.

    Trivial = greetings, acks, emoji-only, or very short messages.
    """
    msg = message.strip()
    if not msg:
        return True
    # M15: allow single meaningful characters (e.g. "?" is meaningful, "" is not)
    if len(msg) < 1:
        return True
    msg_lower = msg.lower()
    return any(p.match(msg_lower) for p in _COMPILED)
