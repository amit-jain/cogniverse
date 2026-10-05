"""The overview's run age: whole units since Argo's startedAt."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_dashboard.tabs.optimization import _format_run_age

NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "started_at,expected",
    [
        ("2026-10-05T12:00:03Z", "0m ago"),
        ("2026-10-05T12:00:59Z", "0m ago"),
        ("2026-10-05T12:00:00Z", "0m ago"),
        ("2026-10-05T11:59:01Z", "0m ago"),
        ("2026-10-05T11:59:00Z", "1m ago"),
        ("2026-10-05T11:00:00Z", "1h ago"),
        ("2026-10-03T12:00:00Z", "2d ago"),
        (None, "not started"),
        ("not a time", "unknown"),
    ],
)
def test_the_age_is_whole_units_and_never_negative(started_at, expected):
    """A start a few seconds ahead of the reader's clock (controller skew, or
    the second Argo rounds to) is no time ago, not "-1m ago"."""
    assert _format_run_age(started_at, NOW) == expected
