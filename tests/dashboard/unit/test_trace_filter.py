"""filter_traces_df must not crash the Trace Explorer on empty windows or
regex-metacharacter queries.

The old inline logic indexed traces_df['trace_id'] unguarded (KeyError on an
empty frame with no such column) and used str.contains default regex=True
(re.error on a query like 'op('). The helper guards the missing column and
matches literal substrings (regex=False, na=False).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from cogniverse_dashboard.utils.traces import filter_traces_df, span_window_end

pytestmark = [pytest.mark.unit]


def test_empty_frame_returns_empty_no_keyerror():
    for stype in ("Trace ID", "Operation", "All"):
        out = filter_traces_df(pd.DataFrame(), stype, "abc")
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0


def test_regex_metacharacters_matched_literally():
    df = pd.DataFrame(
        [
            {"trace_id": "t1", "operation": "video_op(x)"},
            {"trace_id": "t2", "operation": "search"},
            {"trace_id": "t3", "operation": "op"},
        ]
    )
    out = filter_traces_df(df, "Operation", "op(")
    # regex=False => 'op(' is a literal substring: only 'video_op(x)' contains
    # it; bare 'op' does not. regex=True would raise re.error on 'op('.
    assert list(out["operation"]) == ["video_op(x)"]
    assert len(out) == 1


def test_empty_search_text_returns_frame_unchanged():
    df = pd.DataFrame(
        [
            {"trace_id": "t1", "operation": "a"},
            {"trace_id": "t2", "operation": "b"},
        ]
    )
    out = filter_traces_df(df, "Trace ID", "")
    assert len(out) == 2


def test_nan_cell_excluded_without_exception():
    df = pd.DataFrame(
        [
            {"trace_id": "abc123", "operation": "search"},
            {"trace_id": None, "operation": "other"},
        ]
    )
    out = filter_traces_df(df, "Trace ID", "abc")
    assert list(out["trace_id"]) == ["abc123"]
    assert len(out) == 1


@pytest.mark.parametrize(
    ("now", "end"),
    [
        (
            datetime(2026, 9, 17, 5, 36, 54, 687000, tzinfo=timezone.utc),
            datetime(2026, 9, 17, 5, 37, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 9, 17, 5, 36, 30, tzinfo=timezone.utc),
            datetime(2026, 9, 17, 5, 37, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 9, 17, 5, 36, 29, 999999, tzinfo=timezone.utc),
            datetime(2026, 9, 17, 5, 36, 30, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 9, 17, 23, 59, 45, tzinfo=timezone.utc),
            datetime(2026, 9, 18, 0, 0, 0, tzinfo=timezone.utc),
        ),
    ],
)
def test_span_window_end_is_the_bucket_boundary_after_now(now, end):
    """A page rendered at ``now`` holds every span started up to ``now``.

    The first case is the dashboard's render time for a tenant whose routing
    decisions started at 05:36:42-05:36:48: a window ending at 05:36:30 held
    none of them.
    """
    assert span_window_end(now) == end


def test_renders_within_one_bucket_share_the_window_end():
    """Reruns inside one 30-second bucket reuse one cache key."""
    ends = {
        span_window_end(datetime(2026, 9, 17, 5, 36, second, tzinfo=timezone.utc))
        for second in range(30, 60)
    }
    assert ends == {datetime(2026, 9, 17, 5, 37, 0, tzinfo=timezone.utc)}
