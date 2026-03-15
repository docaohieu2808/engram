"""Heuristic memory type classifier for ingested messages.

Classifies message content into MemoryType categories based on
keyword/pattern matching. Fast, no LLM cost. Falls back to FACT.
"""

from __future__ import annotations

import re

from engram.models import MemoryType

# Patterns ordered by specificity (first match wins)
_PATTERNS: list[tuple[MemoryType, list[re.Pattern]]] = [
    (MemoryType.TODO, [
        re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE),
        re.compile(r"\b(need to|should|must|have to|gotta|gonna)\b.*\b(fix|implement|add|create|update|change|refactor|migrate|deploy|set ?up)\b", re.IGNORECASE),
        re.compile(r"\b(task|action item|next step|follow[- ]?up)\b", re.IGNORECASE),
        re.compile(r"\bcần\b.*\b(làm|fix|sửa|thêm|tạo|update|deploy)\b", re.IGNORECASE),
        re.compile(r"\bphải\b.*\b(làm|fix|sửa|thêm|tạo)\b", re.IGNORECASE),
    ]),
    (MemoryType.DECISION, [
        re.compile(r"\b(decided|decision|we('ll| will) (go with|use)|let's (go with|use)|chose|choosing|picked)\b", re.IGNORECASE),
        re.compile(r"\b(switch(ed|ing)? (to|from)|migrat(e|ed|ing) (to|from)|replac(e|ed|ing) .+ with)\b", re.IGNORECASE),
        re.compile(r"\b(quyết định|chọn|dùng|chuyển sang|đổi sang)\b", re.IGNORECASE),
    ]),
    (MemoryType.PREFERENCE, [
        re.compile(r"\b(i prefer|i like|i want|i always|i never|don't like|hate)\b", re.IGNORECASE),
        re.compile(r"\b(tôi thích|tôi muốn|luôn luôn|không bao giờ|ưu tiên)\b", re.IGNORECASE),
        re.compile(r"\b(always use|never use|default to|convention is)\b", re.IGNORECASE),
    ]),
    (MemoryType.ERROR, [
        re.compile(r"\b(error|exception|traceback|stack ?trace|crash(ed|es)?|bug|broken)\b", re.IGNORECASE),
        re.compile(r"\b(fix(ed)?|root cause|workaround|hotfix|patch)\b.*\b(by|with|via|using)\b", re.IGNORECASE),
        re.compile(r"\b(lỗi|bug|crash|fail|sập|hỏng)\b", re.IGNORECASE),
    ]),
    (MemoryType.WORKFLOW, [
        re.compile(r"\b(workflow|pipeline|process|procedure|CI/?CD|deploy(ment)?|release)\b", re.IGNORECASE),
        re.compile(r"\b(step \d|first .+ then|after that|finally)\b", re.IGNORECASE),
    ]),
    (MemoryType.LESSON, [
        re.compile(r"\b(lesson learned|takeaway|insight|realized|turns out|TIL)\b", re.IGNORECASE),
        re.compile(r"\b(rút ra|bài học|hóa ra)\b", re.IGNORECASE),
    ]),
]


# Default priority by memory type (higher = more important to recall)
_TYPE_PRIORITY: dict[MemoryType, int] = {
    MemoryType.TODO: 7,
    MemoryType.DECISION: 8,
    MemoryType.PREFERENCE: 8,
    MemoryType.ERROR: 6,
    MemoryType.WORKFLOW: 5,
    MemoryType.LESSON: 7,
    MemoryType.FACT: 5,
}


def classify_memory_type(content: str) -> MemoryType:
    """Classify content into a MemoryType using heuristic patterns.

    Returns the first matching type, or FACT as default.
    """
    for memory_type, patterns in _PATTERNS:
        for pattern in patterns:
            if pattern.search(content):
                return memory_type
    return MemoryType.FACT


def classify_priority(memory_type: MemoryType) -> int:
    """Return default priority for a given memory type."""
    return _TYPE_PRIORITY.get(memory_type, 5)
