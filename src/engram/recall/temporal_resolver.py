"""Resolve relative temporal references to ISO dates.

Converts 'hôm nay', 'yesterday', 'tuần trước' etc. to concrete ISO dates
so memories remain meaningful over time.

This module provides a store-compatible API (returns primary date string)
complementing entity_resolver.py which returns a full mapping dict.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_offset(d: date, months: int) -> date:
    """Shift date by ±months, clamping day to safe maximum."""
    total_months = d.month + months
    year = d.year + (total_months - 1) // 12
    month = ((total_months - 1) % 12) + 1
    day = min(d.day, 28)  # safe for all months (handles Feb 28/29)
    return date(year, month, day)


def _to_date(d: date | datetime | None) -> date:
    """Normalize to date, defaulting to today."""
    if d is None:
        return date.today()
    if isinstance(d, datetime):
        return d.date()
    return d


# ---------------------------------------------------------------------------
# Pattern registry — (compiled regex, resolver fn returning ISO date string)
# Ordered: multi-word patterns BEFORE single-word to avoid partial matches.
# ---------------------------------------------------------------------------

def _build_patterns(ref: date) -> list[tuple[re.Pattern, str]]:
    """Build list of (pattern, iso_date) for a given reference date."""
    return [
        # --- Vietnamese multi-word (check before single-word) ---
        (re.compile(r"\btuần rồi\b", re.IGNORECASE | re.UNICODE),
         (ref - timedelta(weeks=1)).isoformat()),
        (re.compile(r"\btuần trước\b", re.IGNORECASE | re.UNICODE),
         (ref - timedelta(weeks=1)).isoformat()),
        (re.compile(r"\btuần tới\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(weeks=1)).isoformat()),
        (re.compile(r"\btuần sau\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(weeks=1)).isoformat()),
        (re.compile(r"\btháng rồi\b", re.IGNORECASE | re.UNICODE),
         _month_offset(ref, -1).isoformat()),
        (re.compile(r"\btháng trước\b", re.IGNORECASE | re.UNICODE),
         _month_offset(ref, -1).isoformat()),
        (re.compile(r"\btháng tới\b", re.IGNORECASE | re.UNICODE),
         _month_offset(ref, 1).isoformat()),
        (re.compile(r"\btháng sau\b", re.IGNORECASE | re.UNICODE),
         _month_offset(ref, 1).isoformat()),
        (re.compile(r"\bnăm ngoái\b", re.IGNORECASE | re.UNICODE),
         str(ref.year - 1)),
        (re.compile(r"\bnăm trước\b", re.IGNORECASE | re.UNICODE),
         str(ref.year - 1)),
        (re.compile(r"\bnăm tới\b", re.IGNORECASE | re.UNICODE),
         str(ref.year + 1)),
        (re.compile(r"\bnăm sau\b", re.IGNORECASE | re.UNICODE),
         str(ref.year + 1)),
        (re.compile(r"\bngày mai\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=1)).isoformat()),
        # "mai" standalone — but NOT if preceded by sáng/chiều/tối/ngày (already matched)
        (re.compile(r"(?<!sáng )(?<!chiều )(?<!tối )(?<!ngày )(?<!\w)mai(?!\w)(?! \(ngày)", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=1)).isoformat()),
        (re.compile(r"\bngày mốt\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=2)).isoformat()),
        (re.compile(r"\bngày kia\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=2)).isoformat()),
        (re.compile(r"\bhôm kia\b", re.IGNORECASE | re.UNICODE),
         (ref - timedelta(days=2)).isoformat()),
        (re.compile(r"\bhôm qua\b", re.IGNORECASE | re.UNICODE),
         (ref - timedelta(days=1)).isoformat()),
        (re.compile(r"\bhôm nay\b", re.IGNORECASE | re.UNICODE),
         ref.isoformat()),
        # Time-of-day Vietnamese (append day tag in annotation)
        (re.compile(r"\bsáng nay\b", re.IGNORECASE | re.UNICODE),
         ref.isoformat()),
        (re.compile(r"\bchiều nay\b", re.IGNORECASE | re.UNICODE),
         ref.isoformat()),
        (re.compile(r"\btối nay\b", re.IGNORECASE | re.UNICODE),
         ref.isoformat()),
        (re.compile(r"\bsáng mai\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=1)).isoformat()),
        (re.compile(r"\bchiều mai\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=1)).isoformat()),
        (re.compile(r"\btối mai\b", re.IGNORECASE | re.UNICODE),
         (ref + timedelta(days=1)).isoformat()),
        (re.compile(r"\btối qua\b", re.IGNORECASE | re.UNICODE),
         (ref - timedelta(days=1)).isoformat()),
        (re.compile(r"\bsáng qua\b", re.IGNORECASE | re.UNICODE),
         (ref - timedelta(days=1)).isoformat()),
        # --- English multi-word ---
        (re.compile(r"\blast week\b", re.IGNORECASE),
         (ref - timedelta(weeks=1)).isoformat()),
        (re.compile(r"\blast month\b", re.IGNORECASE),
         _month_offset(ref, -1).isoformat()),
        (re.compile(r"\blast year\b", re.IGNORECASE),
         str(ref.year - 1)),
        (re.compile(r"\blast night\b", re.IGNORECASE),
         (ref - timedelta(days=1)).isoformat()),
        (re.compile(r"\bthis morning\b", re.IGNORECASE),
         ref.isoformat()),
        (re.compile(r"\bnext week\b", re.IGNORECASE),
         (ref + timedelta(weeks=1)).isoformat()),
        (re.compile(r"\bnext month\b", re.IGNORECASE),
         _month_offset(ref, 1).isoformat()),
        # --- English single-word ---
        (re.compile(r"\byesterday\b", re.IGNORECASE),
         (ref - timedelta(days=1)).isoformat()),
        (re.compile(r"\btomorrow\b", re.IGNORECASE),
         (ref + timedelta(days=1)).isoformat()),
        (re.compile(r"\btoday\b", re.IGNORECASE),
         ref.isoformat()),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_temporal(
    content: str,
    reference_date: date | datetime | None = None,
) -> tuple[str, str | None]:
    """Resolve relative dates in content to ISO format.

    Args:
        content: Raw text with possible temporal references.
        reference_date: Reference date (defaults to today).

    Returns:
        (resolved_content, resolved_date_iso) — content with dates annotated,
        and the primary resolved date as ISO string, or None if no match found.

    Example:
        >>> resolve_temporal("Đi mall hôm nay", date(2026, 2, 25))
        ("Đi mall hôm nay (ngày 2026-02-25)", "2026-02-25")
    """
    if not content:
        return content, None

    ref = _to_date(reference_date)
    patterns = _build_patterns(ref)

    resolved = content
    primary_date: str | None = None

    # Idempotency: pattern "(ngày YYYY-MM-DD)" marks already-annotated phrases.
    # Skip any pattern whose match is immediately followed by this annotation
    # to avoid stacking "(ngày ...)(ngày ...)" on repeated calls.
    _already_annotated = re.compile(r"\(ngày \d{4}-\d{2}-\d{2}\)")

    for pattern, iso_date in patterns:
        match = pattern.search(resolved)
        if not match:
            continue
        # Check if this match is already annotated (followed by "(ngày ...)")
        end_pos = match.end()
        suffix = resolved[end_pos:end_pos + 20]
        if _already_annotated.match(suffix.lstrip()):
            continue
        resolved = pattern.sub(
            lambda m, d=iso_date: f"{m.group(0)} (ngày {d})",
            resolved,
        )
        if primary_date is None:
            primary_date = iso_date  # record first matched date

    return resolved, primary_date
