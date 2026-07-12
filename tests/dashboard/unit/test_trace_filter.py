"""filter_traces_df must not crash the Trace Explorer on empty windows or
regex-metacharacter queries.

The old inline logic indexed traces_df['trace_id'] unguarded (KeyError on an
empty frame with no such column) and used str.contains default regex=True
(re.error on a query like 'op('). The helper guards the missing column and
matches literal substrings (regex=False, na=False).
"""

from __future__ import annotations

import pandas as pd
import pytest

from cogniverse_dashboard.utils.traces import filter_traces_df

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
