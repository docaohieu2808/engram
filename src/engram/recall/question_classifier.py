"""Classify questions as general-knowledge vs personal to skip irrelevant memory retrieval.

General questions (K8s best practices, how DNS works) should NOT pull episodic memories.
Personal questions (who is X, what's my server setup) SHOULD pull memories.
Mixed questions (how should I deploy MY app) get both.

Uses lightweight regex heuristics — no LLM call needed.
"""

from __future__ import annotations

import re
from enum import Enum


class QuestionType(str, Enum):
    GENERAL = "general"    # Pure general knowledge — skip episodic/semantic
    PERSONAL = "personal"  # About user's life/data — full memory search
    MIXED = "mixed"        # Has both signals — search but with higher relevance bar


# --- Strong personal signals: possessive/personal-context patterns only ---
# Bare pronouns (tôi, I, me) are NOT enough — they appear in any Vietnamese sentence.
# Only match when pronoun indicates the user is asking about THEIR stuff.

_STRONG_PERSONAL_PATTERNS = [
    # English possessive + noun (my server, my app, our infra)
    r"\b(my|our)\s+\w+",
    r"\bmine\b",
    # Vietnamese possessive patterns: "của tôi", "server tôi", "máy mình"
    r"\bcủa\s+(tôi|mình|em|anh|chị|ông|bà)\b",
    r"\b(máy|server|hệ thống|dự án|project|app|ứng dụng|code)\s+(tôi|mình|em|anh)\b",
    r"\b(tôi|mình|em)\s+(đang|đã|sẽ|có|muốn|cần)\s+(chạy|dùng|setup|cài|triển khai)\b",
    # "Who is X" / "What did I" / "When did I" / recall patterns
    r"\bwho\s+is\b",
    r"\bwhat\s+did\s+(i|we)\b",
    r"\bwhen\s+did\s+(i|we)\b",
    r"\bwhere\s+did\s+(i|we)\b",
    r"\b(i|we)\s+(did|have|had|was|were|am|are)\b",
    r"\b(nhớ|remember|recall)\b",
    # Explicit self-reference patterns
    r"\b(i'm|i've|i'd|i'll|myself|ourselves)\b",
    r"\b(tell me about my|show me my|what's my|what is my)\b",
]

_STRONG_PERSONAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in _STRONG_PERSONAL_PATTERNS]

# --- General knowledge signals: technical/abstract questions ---

_GENERAL_PATTERNS = [
    # "What is X" / "How does X work" / "Explain X" (generic knowledge-seeking)
    r"^(what\s+is|what\s+are|how\s+does|how\s+do|how\s+to|explain|define)\b",
    r"^(là gì|giải thích|cách|hướng dẫn|so sánh)\b",
    # "Best practices" / "pros and cons" / "differences between"
    r"\b(best\s+practices?|pros?\s+and\s+cons?|difference\s+between|advantages?\s+of)\b",
    r"\b(so sánh|ưu điểm|nhược điểm|khác nhau)\b",
    # "How to do X" in Vietnamese
    r"\b(làm sao|thế nào|cho chuẩn|triển khai)\b",
    # Technical domain keywords (strong signal)
    r"\b(kubernetes|k8s|docker|nginx|redis|postgres|mysql|mongodb)\b",
    r"\b(react|vue|angular|typescript|javascript|python|golang|rust|java)\b",
    r"\b(dns|tcp|http|ssl|tls|oauth|jwt|rest|graphql|grpc|websocket)\b",
    r"\b(aws|gcp|azure|terraform|ansible|helm|istio|prometheus|grafana)\b",
    r"\b(linux|unix|bash|shell|systemd|cron|iptables|firewall)\b",
    r"\b(algorithm|data\s+structure|design\s+pattern|architecture|microservice)\b",
    r"\b(machine\s+learning|neural\s+network|deep\s+learning|transformer|llm|embedding)\b",
    r"\b(server|cluster|node|container|pod|deployment|ingress|service)\b",
]

_GENERAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in _GENERAL_PATTERNS]


def classify_question(
    question: str,
    known_entities: list[str] | None = None,
) -> QuestionType:
    """Classify a question as general, personal, or mixed.

    Args:
        question: The user's question text.
        known_entities: List of personal entity names (Person/Project only).
            If the question mentions a known personal entity, it's personal.

    Returns:
        QuestionType indicating how to handle memory retrieval.
    """
    q = question.strip()
    if not q:
        return QuestionType.GENERAL

    has_personal = any(p.search(q) for p in _STRONG_PERSONAL_COMPILED)
    has_general = any(p.search(q) for p in _GENERAL_COMPILED)

    # Check if question mentions known personal entities from user's graph
    if known_entities:
        q_lower = q.lower()
        for entity in known_entities:
            if len(entity) > 2 and entity.lower() in q_lower:
                has_personal = True
                break

    # Classification logic:
    # When question has BOTH general tech terms AND strong personal context → MIXED
    # When only general tech terms (even with bare pronouns like "tôi") → GENERAL
    # When only strong personal signals → PERSONAL
    if has_personal and has_general:
        return QuestionType.MIXED
    if has_personal:
        return QuestionType.PERSONAL
    if has_general:
        return QuestionType.GENERAL

    # Default: no strong signals → MIXED (safe fallback)
    return QuestionType.MIXED
