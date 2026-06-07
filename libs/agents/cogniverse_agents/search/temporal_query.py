"""Extract an explicit temporal range from a search query.

Returns a ``(start, end)`` UTC datetime window when the query carries clear
temporal intent (relative ranges, named periods, explicit years), else ``None``
so the reranker keeps its neutral temporal score for non-temporal queries.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_REL_N = re.compile(
    r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b", re.I
)
_REL_1 = re.compile(r"\b(?:last|past|previous)\s+(day|week|month|year)\b", re.I)
_THIS = re.compile(r"\bthis\s+(week|month|year)\b", re.I)
_YESTERDAY = re.compile(r"\byesterday\b", re.I)
_TODAY = re.compile(r"\btoday\b", re.I)
_YEAR = re.compile(r"\b(?:in|from|since|during)\s+((?:19|20)\d{2})\b", re.I)


def extract_time_range(
    query: str, *, now: Optional[datetime] = None
) -> Optional[Tuple[datetime, datetime]]:
    """Return a UTC ``(start, end)`` window for the query's temporal intent.

    ``None`` when the query has no clear temporal expression — callers must then
    omit temporal context so reranking stays neutral.
    """
    if not query:
        return None
    now = now or datetime.now(timezone.utc)
    q = query.lower()

    m = _REL_N.search(q)
    if m:
        days = int(m.group(1)) * _UNIT_DAYS[m.group(2)]
        return (now - timedelta(days=days), now)

    m = _REL_1.search(q)
    if m:
        return (now - timedelta(days=_UNIT_DAYS[m.group(1)]), now)

    if _YESTERDAY.search(q):
        start = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (start, start + timedelta(days=1) - timedelta(microseconds=1))

    if _TODAY.search(q):
        return (now.replace(hour=0, minute=0, second=0, microsecond=0), now)

    m = _THIS.search(q)
    if m:
        unit = m.group(1)
        if unit == "week":
            start = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif unit == "month":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
        return (start, now)

    m = _YEAR.search(q)
    if m:
        year = int(m.group(1))
        return (
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        )

    return None
