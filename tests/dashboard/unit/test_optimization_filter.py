"""_filter_search_spans tolerates null span names.

Phoenix span DataFrames can carry a null `name`; `str.contains` without
`na=False` returns NaN, which raises when used as a boolean mask — crashing
the optimization tab on render.
"""

from __future__ import annotations

import pandas as pd

from cogniverse_dashboard.tabs.optimization import _filter_search_spans


def test_null_span_name_does_not_crash_and_is_excluded():
    df = pd.DataFrame(
        {"name": ["search_agent.process", None, "summarizer.process"], "v": [1, 2, 3]}
    )

    result = _filter_search_spans(df)

    assert list(result["name"]) == ["search_agent.process"]


def test_no_search_spans_returns_empty():
    df = pd.DataFrame({"name": ["summarizer.process", None], "v": [1, 2]})
    assert _filter_search_spans(df).empty
