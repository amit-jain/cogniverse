"""Memory timestamp normalization — ms-vs-seconds magnitude + tz-aware parsing.

A millisecond ``created_at`` must not clamp ``_compute_age_seconds`` to 0 or
raise inside ``datetime.fromtimestamp`` on the read path, and a naive ISO
``created_at`` must be read as UTC, not the host's local tz.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cogniverse_core.memory._timestamps import epoch_to_iso_utc, to_epoch_seconds

_S = 1_700_000_000  # 2023-11-14T22:13:20 UTC, in seconds
_MS = 1_700_000_000_000  # same instant, in milliseconds


def test_to_epoch_seconds_milliseconds_collapse_to_seconds():
    assert to_epoch_seconds(_MS) == _S
    assert to_epoch_seconds(_S) == _S


def test_to_epoch_seconds_naive_iso_is_utc_not_local():
    expected = int(datetime(2023, 11, 15, 3, 43, 20, tzinfo=timezone.utc).timestamp())
    assert to_epoch_seconds("2023-11-15T03:43:20") == expected
    assert to_epoch_seconds("2023-11-15T03:43:20Z") == expected


def test_to_epoch_seconds_aware_iso_respects_offset():
    tz = timezone(timedelta(hours=5, minutes=30))
    expected = int(datetime(2023, 11, 15, 3, 43, 20, tzinfo=tz).timestamp())
    assert to_epoch_seconds("2023-11-15T03:43:20+05:30") == expected


def test_to_epoch_seconds_rejects_missing_and_garbage():
    assert to_epoch_seconds(None) is None
    assert to_epoch_seconds("") is None
    assert to_epoch_seconds(True) is None
    assert to_epoch_seconds("not-a-date") is None


def test_epoch_to_iso_utc_handles_milliseconds_without_raising():
    # datetime.fromtimestamp(_MS) raises ValueError (year out of range) — the
    # old read path swallowed that and dropped the whole search result.
    assert epoch_to_iso_utc(_MS) == "2023-11-14T22:13:20+00:00"
    assert epoch_to_iso_utc(_S).endswith("+00:00")


def test_compute_age_seconds_normalizes_milliseconds():
    from cogniverse_core.memory.manager import Mem0MemoryManager

    now = _S + 100
    # ms created_at must yield a real positive age (was clamped to 0).
    assert Mem0MemoryManager._compute_age_seconds({"created_at": _MS}, now) == 100
    assert Mem0MemoryManager._compute_age_seconds({"created_at": _S}, now) == 100
    assert Mem0MemoryManager._compute_age_seconds({"created_at": None}, now) is None


def test_to_epoch_seconds_collapses_microsecond_and_nanosecond_magnitudes():
    assert to_epoch_seconds(_MS * 1000) == _S  # microseconds -> seconds
    assert to_epoch_seconds(_S * 10**9) == _S  # nanoseconds -> seconds


def test_epoch_to_iso_utc_survives_absurd_magnitude():
    # A value far beyond ms used to reach datetime.fromtimestamp and raise
    # OSError/OverflowError (swallowed on the read path); now it collapses.
    assert epoch_to_iso_utc(_S * 10**9) == "2023-11-14T22:13:20+00:00"
