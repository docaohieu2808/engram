"""LLM-powered episodic memory extraction from conversation turns.

Extracts facts, decisions, preferences, errors, TODOs, and workflow patterns
from a conversation exchange and returns structured items for storage.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm

from engram.config import ExtractionConfig
from engram.sanitize import sanitize_llm_input

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

EXTRACTION_PROMPT = """Extract important items from this conversation to remember long-term.
Return a JSON array of objects with fields: content, type, priority.

Field rules:
- content: concise description of the fact/decision/preference (max 200 chars)
- type: one of "fact", "decision", "preference", "error", "todo", "workflow"
- priority: integer 1-10 (1=trivial, 10=critical)

EXTRACT:
- Decisions made (tech choices, architecture, trade-offs) → priority 7-9
- User preferences explicitly stated → priority 6-8
- Errors encountered and their fixes → priority 7-8
- TODOs mentioned → priority 5-7
- Key facts about project/codebase → priority 5-7
- Workflow patterns → priority 5-6

DO NOT EXTRACT:
- Routine file reads/greps/searches
- Intermediate thinking steps
- Code snippets (too verbose)
- Greetings, acknowledgments, filler
- Items with priority < 4 (too noisy)

Conversation:
User: {user_msg}
Assistant: {assistant_msg}

Output (JSON array only, no markdown, no extra text):
"""


class MemoryExtractor:
    """Extract episodic memory items from a conversation turn using LLM."""

    def __init__(
        self,
        model: str | None = None,
        config: ExtractionConfig | None = None,
        disable_thinking: bool = False,
    ):
        self._config = config or ExtractionConfig()
        self._model = model or (self._config.llm_model or None)
        self._disable_thinking = disable_thinking
        if not self._model:
            # Fallback to default; caller should pass llm.model from top-level config
            self._model = "gemini/gemini-2.5-flash"

    async def extract(self, user_msg: str, assistant_msg: str) -> list[dict[str, Any]]:
        """Extract memorable items from a conversation turn.

        Args:
            user_msg: The user's message content.
            assistant_msg: The assistant's response content.

        Returns:
            List of dicts with keys: content, type, priority.
            Empty list on failure (fail-open).
        """
        # I-C4: sanitize user/assistant messages before LLM prompt interpolation
        cfg = self._config
        prompt = EXTRACTION_PROMPT.format(
            user_msg=sanitize_llm_input(user_msg, max_len=cfg.user_msg_max_len),
            assistant_msg=sanitize_llm_input(assistant_msg, max_len=cfg.assistant_msg_max_len),
        )

        try:
            kwargs: dict[str, Any] = dict(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
            )
            if self._disable_thinking:
                kwargs["thinking"] = {"type": "disabled"}
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            return self._parse(content)
        except Exception as e:
            logger.warning("MemoryExtractor LLM call failed: %s", e)
            return []

    def _parse(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM JSON response into list of extraction dicts.

        Strips markdown fences, parses JSON, filters low-priority items.
        """
        # Strip markdown code blocks if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")

        # Handle cases where the model wraps in an outer object
        if content.startswith("{"):
            try:
                data = json.loads(content)
                # Try common wrapper keys
                for key in ("items", "memories", "results", "data"):
                    if key in data and isinstance(data[key], list):
                        content = json.dumps(data[key])
                        break
                else:
                    # Single object — wrap in array
                    content = json.dumps([data])
            except json.JSONDecodeError:
                return []

        try:
            items = json.loads(content)
        except json.JSONDecodeError:
            logger.debug("MemoryExtractor: failed to parse JSON: %.200s", content)
            return []

        if not isinstance(items, list):
            return []

        valid: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content_val = item.get("content", "").strip()
            if not content_val:
                continue
            try:
                priority = int(item.get("priority", 5))
            except (ValueError, TypeError):
                priority = 5
            # Filter out noise
            if priority < 4:
                continue
            item_type = item.get("type", "fact")
            # Normalise type to known values
            if item_type not in ("fact", "decision", "preference", "error", "todo", "workflow"):
                item_type = "fact"
            valid.append({
                "content": content_val,
                "type": item_type,
                "priority": priority,
            })

        return valid
