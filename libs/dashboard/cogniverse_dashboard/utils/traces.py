"""Pure trace-search filtering for the dashboard Trace Explorer.

Extracted from app.py so the filter is importable and unit-testable. The
search must never abort the dashboard: an empty trace window (no ``trace_id``
column) previously raised KeyError, and a query with regex metacharacters
(e.g. ``op(``) raised ``re.error``. Matching is literal-substring
(``regex=False``), guarded against a missing column and NaN cells.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def fetch_tenant_traces(
    analytics: Any,
    tenant_id: str,
    start_time: datetime,
    end_time: datetime,
    operation_filter: str | None,
    limit: int = 10000,
) -> list:
    """Fetch traces scoped to a tenant's Phoenix project.

    Extracted so the Analytics tab cannot forget the ``project_name`` — omitting
    it made get_traces query Phoenix's ``default`` project, so real tenant
    traffic showed "No traces found". The tenant is required, so the project is
    always derived here.
    """
    return analytics.get_traces(
        start_time=start_time,
        end_time=end_time,
        operation_filter=operation_filter,
        limit=limit,
        project_name=f"cogniverse-{tenant_id}",
    )


def fetch_tenant_traces_safely(
    analytics: Any,
    tenant_id: str,
    start_time: datetime,
    end_time: datetime,
    operation_filter: str | None,
    limit: int = 10000,
) -> tuple[list, str | None]:
    """Fetch traces, mapping a telemetry failure to an error message.

    Returns (traces, None) on success and ([], message) on failure so the
    Analytics tab can show the outage instead of the indistinguishable
    "No traces found" empty state.
    """
    try:
        return (
            fetch_tenant_traces(
                analytics, tenant_id, start_time, end_time, operation_filter, limit
            ),
            None,
        )
    except Exception as exc:
        return [], f"Failed to fetch traces from the telemetry backend: {exc}"


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
