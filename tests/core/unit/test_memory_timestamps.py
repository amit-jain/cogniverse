"""Memory timestamp normalization — ms-vs-seconds magnitude + tz-aware parsing.

A millisecond ``created_at`` must not clamp ``_compute_age_seconds`` to 0 or
raise inside ``datetime.fromtimestamp`` on the read path, and a naive ISO
``created_at`` must be read as UTC, not the host's local tz.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_core.memory._timestamps import epoch_to_iso_utc, to_epoch_seconds

_S = 1_700_000_000  # 2023-11-14T22:13:20 UTC, in seconds
_MS = 1_700_000_000_000  # same instant, in milliseconds


@pytest.mark.unit
@pytest.mark.ci_fast
def test_to_epoch_seconds_milliseconds_collapse_to_seconds():
    assert to_epoch_seconds(_MS) == _S
    assert to_epoch_seconds(_S) == _S


@pytest.mark.unit
@pytest.mark.ci_fast
def test_to_epoch_seconds_naive_iso_is_utc_not_local():
    expected = int(datetime(2023, 11, 15, 3, 43, 20, tzinfo=timezone.utc).timestamp())
    assert to_epoch_seconds("2023-11-15T03:43:20") == expected
    assert to_epoch_seconds("2023-11-15T03:43:20Z") == expected


def test_to_epoch_seconds_aware_iso_respects_offset():
    tz = timezone(timedelta(hours=5, minutes=30))
    expected = int(datetime(2023, 11, 15, 3, 43, 20, tzinfo=tz).timestamp())
    assert to_epoch_seconds("2023-11-15T03:43:20+05:30") == expected


@pytest.mark.unit
@pytest.mark.ci_fast
def test_to_epoch_seconds_rejects_missing_and_garbage():
    assert to_epoch_seconds(None) is None
    assert to_epoch_seconds("") is None
    assert to_epoch_seconds(True) is None
    assert to_epoch_seconds("not-a-date") is None


@pytest.mark.unit
@pytest.mark.ci_fast
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


def test_to_epoch_seconds_parses_stringified_epochs():
    # Mem0/Vespa payloads round-trip created_at through JSON as strings; a
    # digit-string epoch used to fall through the ISO parser and return None.
    assert to_epoch_seconds(str(_S)) == _S
    assert to_epoch_seconds(str(_MS)) == _S  # ms-magnitude string collapses too
    assert to_epoch_seconds(f"  {_S}  ") == _S
    assert to_epoch_seconds(f"{_S}.5") == _S
    assert to_epoch_seconds("1.7e12") == 1_700_000_000  # scientific ms → seconds


def test_to_epoch_seconds_string_non_finite_and_garbage_still_none():
    assert to_epoch_seconds("inf") is None
    assert to_epoch_seconds("nan") is None
    assert to_epoch_seconds("-inf") is None
    assert to_epoch_seconds("12abc") is None


def test_to_epoch_seconds_iso_still_wins_over_numeric_lookalike():
    # "20231115" is a valid ISO 8601 basic-format date; it must keep parsing
    # as 2023-11-15, not as a ~1970 epoch of 20,231,115 seconds.
    expected = int(datetime(2023, 11, 15, tzinfo=timezone.utc).timestamp())
    assert to_epoch_seconds("20231115") == expected


def test_to_epoch_seconds_returns_none_for_non_finite():
    # inf would spin the ms-collapse loop forever; nan/-inf int() raises.
    assert to_epoch_seconds(float("inf")) is None
    assert to_epoch_seconds(float("-inf")) is None
    assert to_epoch_seconds(float("nan")) is None
