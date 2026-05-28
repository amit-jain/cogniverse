"""Profile metrics tab — per-modality runtime observability from Phoenix spans.

Reads ``cogniverse.profile_selection`` spans for the selected tenant and
aggregates them by the ``profile_selection.modality`` attribute the
ProfileSelectionAgent emits on every dispatch. Replaces the deleted
Multi-Modal Performance tab whose backing tracker
(``ModalityMetricsTracker``) was never populated by the runtime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_PROFILE_SELECTION,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

logger = logging.getLogger(__name__)


def _modality_from_row(row: pd.Series) -> str | None:
    """Pull the modality value from a Phoenix span row.

    Phoenix flattens span attributes into prefixed columns
    (``attributes.profile_selection`` is a dict). Returns the
    ``modality`` field or None when the attribute is missing.
    """
    cell = row.get("attributes.profile_selection")
    if isinstance(cell, dict):
        value = cell.get("modality")
        if value:
            return str(value)
    return None


def _aggregate(spans_df: pd.DataFrame) -> pd.DataFrame:
    """Group spans by modality, compute count + latency stats + success rate."""
    durations_ms = (
        spans_df["end_time"] - spans_df["start_time"]
    ).dt.total_seconds() * 1000
    spans_df = spans_df.assign(
        modality=spans_df.apply(_modality_from_row, axis=1),
        duration_ms=durations_ms,
        ok=spans_df["status_code"].fillna("OK").eq("OK"),
    ).dropna(subset=["modality"])

    if spans_df.empty:
        return pd.DataFrame()

    grouped = spans_df.groupby("modality", as_index=False).agg(
        count=("duration_ms", "size"),
        p50_ms=("duration_ms", lambda s: s.quantile(0.50)),
        p95_ms=("duration_ms", lambda s: s.quantile(0.95)),
        p99_ms=("duration_ms", lambda s: s.quantile(0.99)),
        success_rate=("ok", "mean"),
    )
    return grouped.sort_values("count", ascending=False)


def render_profile_metrics_tab() -> None:
    """Render per-modality runtime metrics from ProfileSelectionAgent spans."""
    st.subheader("Profile Routing Metrics")
    st.markdown(
        "Per-modality runtime observability sourced from "
        "``cogniverse.profile_selection`` spans in Phoenix. Reflects what "
        "the ProfileSelectionAgent actually classified and routed to."
    )

    tenant_id = st.session_state.get("current_tenant", "")
    if not tenant_id:
        st.info("Select a tenant in the sidebar to view profile routing metrics.")
        return

    col1, col2 = st.columns(2)
    with col1:
        lookback_hours = st.number_input(
            "Lookback (hours)", min_value=1, max_value=720, value=24, step=1
        )
    with col2:
        st.text(f"Tenant: {tenant_id}")

    try:
        manager = get_telemetry_manager()
        provider = manager.get_provider(tenant_id=tenant_id)
        project_name = TelemetryConfig().get_project_name(
            tenant_id, service="cogniverse-orchestration"
        )
    except Exception as exc:
        st.error(f"Failed to initialise telemetry provider: {exc}")
        return

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=int(lookback_hours))

    with st.spinner(f"Querying Phoenix project `{project_name}`..."):
        try:
            import asyncio

            spans_df = asyncio.run(
                provider.traces.get_spans(
                    project=project_name, start_time=start, end_time=end
                )
            )
        except Exception as exc:
            st.error(f"Phoenix span query failed: {exc}")
            return

    if spans_df is None or spans_df.empty:
        st.info(
            f"No spans found in `{project_name}` for the last {int(lookback_hours)}h."
        )
        return

    profile_spans = spans_df[spans_df["name"] == SPAN_NAME_PROFILE_SELECTION]
    if profile_spans.empty:
        st.info(
            f"No `{SPAN_NAME_PROFILE_SELECTION}` spans in the last "
            f"{int(lookback_hours)}h. Drive traffic through "
            "``profile_selection_agent`` first."
        )
        return

    aggregated = _aggregate(profile_spans)
    if aggregated.empty:
        st.warning(
            "Spans found but none carried a ``profile_selection.modality`` "
            "attribute. Verify ProfileSelectionAgent is emitting it."
        )
        return

    st.markdown("### Per-modality metrics")
    cols = st.columns(min(len(aggregated), 5))
    for col, (_, row) in zip(cols, aggregated.iterrows()):
        with col:
            st.metric(
                label=str(row["modality"]).upper(),
                value=int(row["count"]),
                delta=f"P95 {row['p95_ms']:.0f} ms",
            )

    st.markdown("### Detailed stats")
    display = aggregated.copy()
    display["success_rate"] = (display["success_rate"] * 100).map(lambda v: f"{v:.1f}%")
    for col in ("p50_ms", "p95_ms", "p99_ms"):
        display[col] = display[col].map(lambda v: f"{v:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Query distribution")
    fig = px.pie(
        aggregated,
        names="modality",
        values="count",
        title="Queries per modality",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latency by modality")
    fig_latency = px.bar(
        aggregated,
        x="modality",
        y="p95_ms",
        title="P95 latency (ms)",
        labels={"modality": "Modality", "p95_ms": "P95 latency (ms)"},
    )
    st.plotly_chart(fig_latency, use_container_width=True)
