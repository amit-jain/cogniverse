"""Seconds-vs-milliseconds-safe, tz-aware timestamp normalization for memory.

The ingestion write path validates ms-vs-s magnitude (``ingestion_client``
``_validate_ms_timestamp``/``_validate_s_timestamp``); the memory read and
age-compute paths did not. So a millisecond ``created_at`` clamped ages to 0
(never aging out), raised inside naive ``datetime.fromtimestamp`` (swallowed →
all search results dropped), and naive ISO strings were read in local tz.
These helpers apply the same magnitude guard and assume UTC for naive input.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

# 2100-01-01 UTC in seconds; a value larger than this is almost certainly ms.
_MAX_S_EPOCH = 4_102_444_800


def _numeric_epoch_seconds(ts: float) -> Optional[int]:
    if not math.isfinite(ts):  # inf would never exit the loop below; nan int()s raise
        return None
    while ts > _MAX_S_EPOCH:  # collapse ms/us/ns magnitudes down to seconds
        ts /= 1000.0
    return int(ts)


def to_epoch_seconds(value: Any) -> Optional[int]:
    """Normalize a ``created_at`` (epoch s/ms — numeric or numeric string — or
    ISO string) to UTC epoch seconds.

    Returns ``None`` for missing or unparseable input.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    # numpy scalars (np.int64/int32/float32) are not int/float subclasses, so
    # coerce them to a Python scalar before the isinstance gate — otherwise a
    # numpy created_at fell through to None and the write path substituted now().
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bool):
            return None
    if isinstance(value, (int, float)):
        return _numeric_epoch_seconds(float(value))
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            # Not ISO — a stringified epoch ("1700000000", "1.7e12") is still
            # a valid created_at; parse it through the same magnitude guard.
            try:
                return _numeric_epoch_seconds(float(value.strip()))
            except ValueError:
                return None
        if dt.tzinfo is None:  # naive → UTC, not the host's local tz
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    return None


def epoch_to_iso_utc(value: Any) -> Optional[str]:
    """Convert an epoch (seconds or milliseconds) to a tz-aware UTC ISO string."""
    secs = to_epoch_seconds(value)
    if secs is None:
        return None
    return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()
