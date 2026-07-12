"""Pure trace-search filtering for the dashboard Trace Explorer.

Extracted from app.py so the filter is importable and unit-testable. The
search must never abort the dashboard: an empty trace window (no ``trace_id``
column) previously raised KeyError, and a query with regex metacharacters
(e.g. ``op(``) raised ``re.error``. Matching is literal-substring
(``regex=False``), guarded against a missing column and NaN cells.
"""

from __future__ import annotations

import pandas as pd


def filter_traces_df(
    traces_df: pd.DataFrame, search_type: str, search_text: str
) -> pd.DataFrame:
    if not search_text or traces_df.empty or "trace_id" not in traces_df.columns:
        return traces_df

    def _contains(col: str) -> pd.Series:
        if col not in traces_df.columns:
            return pd.Series(False, index=traces_df.index)
        return traces_df[col].str.contains(
            search_text, case=False, regex=False, na=False
        )

    if search_type == "Trace ID":
        mask = _contains("trace_id")
    elif search_type == "Operation":
        mask = _contains("operation")
    else:
        mask = _contains("trace_id") | _contains("operation")
    return traces_df[mask]
