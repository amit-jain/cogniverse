"""Dashboard tile — RLM A/B comparison spans.

Reads ``rlm.ab_compare`` spans (emitted by ``cogniverse-optim --mode
ab-compare``) from Phoenix and renders the per-row + aggregate view.
Each span carries the ``RLMABRunner.to_telemetry_dict()`` payload as
``openinference.*`` attributes, including the per-row ``ab_id`` that
ties paired arms together.

The data loading lives in :func:`load_ab_compare_data` and the aggregation
in ``cogniverse_foundation.telemetry.span_metrics``, so both can be
integration-tested against real Phoenix without the Streamlit runtime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from cogniverse_dashboard.utils import tenant_project_name
from cogniverse_foundation.telemetry.span_metrics import (
    AB_COMPARE_SPAN_NAME,
    ABCompareAggregate,
    aggregate_ab_compare,
)

logger = logging.getLogger(__name__)


def _fmt_delta(value: Optional[float]) -> str:
    """Format a delta metric; a real 0.0 must show as '0.0', not '—'."""
    return f"{value:.1f}" if value is not None else "—"


async def load_ab_compare_data(
    *,
    phoenix_http_endpoint: str,
    phoenix_grpc_endpoint: str,
    tenant_id: str,
    lookback_hours: float = 24.0,
) -> ABCompareAggregate:
    """Fetch ``rlm.ab_compare`` spans from a real Phoenix instance.

    Returns an :class:`ABCompareAggregate`. The Streamlit tile renders
    this; the integration test asserts against this directly so the
    dashboard's data path is exercised end-to-end without a Streamlit
    runtime.
    """
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_http_endpoint,
            "grpc_endpoint": phoenix_grpc_endpoint,
        }
    )

    # ``cogniverse-optim --mode ab-compare`` emits into the tenant project the
    # telemetry config names, so the tile derives it from the same config.
    project_name = tenant_project_name(get_telemetry_manager(), tenant_id)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    # Query through the generic TraceStore interface so any telemetry
    # backend — not just Phoenix — can serve the tile. get_spans pushes
    # the name filter down as a server-side predicate and returns the
    # span attributes flattened; aggregate_ab_compare already tolerates
    # whichever attribute-column shape the backend emits.
    try:
        spans_df = await provider.traces.get_spans(
            project=project_name,
            start_time=start_time,
            end_time=end_time,
            filters={"name": AB_COMPARE_SPAN_NAME},
            limit=10000,
        )
    except Exception as exc:
        raise RuntimeError(
            f"ab-compare spans for tenant {tenant_id!r} could not be read from "
            f"Phoenix at {phoenix_http_endpoint}: {exc}"
        ) from exc

    if spans_df.empty:
        return ABCompareAggregate()

    return aggregate_ab_compare(spans_df)


def render_rlm_ab_compare_tab():
    """Streamlit tile — renders the A/B comparison view.

    Imports streamlit lazily so the module is importable in test
    contexts that don't have streamlit set up.
    """
    import streamlit as st

    st.header("RLM A/B Comparison")
    st.caption(
        "Spans emitted by `cogniverse-optim --mode ab-compare`. "
        "Each row is one (query, context) pair from the input dataset, "
        "with both arms (RLM-on / RLM-off) tied by a shared `ab_id`."
    )

    # The app shell stores the gate-selected tenant under "current_tenant";
    # "tenant_id" was never set, so the tab always fell back to the text input.
    tenant_id = st.session_state.get("current_tenant") or st.text_input(
        "Tenant id", value="default"
    )
    phoenix_url = st.session_state.get("phoenix_url")
    collector_endpoint = st.session_state.get("telemetry_collector_endpoint")
    if not phoenix_url or not collector_endpoint:
        st.error(
            "Phoenix is not configured for this dashboard: "
            f"telemetry_url={phoenix_url!r}, "
            f"telemetry_collector_endpoint={collector_endpoint!r}"
        )
        return
    lookback_hours = st.number_input(
        "Lookback (hours)", min_value=0.1, value=24.0, step=1.0
    )

    if st.button("Load A/B comparison data"):
        import asyncio

        with st.spinner("Querying Phoenix…"):
            try:
                agg = asyncio.run(
                    load_ab_compare_data(
                        phoenix_http_endpoint=phoenix_url,
                        phoenix_grpc_endpoint=collector_endpoint,
                        tenant_id=tenant_id,
                        lookback_hours=lookback_hours,
                    )
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return

        if agg.rows == 0:
            st.info(
                "No `rlm.ab_compare` spans in this window. Run "
                "`cogniverse-optim --mode ab-compare --tenant-id "
                f"{tenant_id} --queries-dataset <name>` to populate."
            )
            return

        cols = st.columns(4)
        cols[0].metric("Comparisons", agg.rows)
        cols[1].metric("Δ latency (ms)", _fmt_delta(agg.avg_latency_delta_ms))
        cols[2].metric("Δ tokens", _fmt_delta(agg.avg_tokens_delta))
        cols[3].metric(
            "Δ judge",
            f"{agg.avg_judge_delta:.3f}" if agg.avg_judge_delta is not None else "—",
        )
        if agg.fallback_rate is not None:
            st.metric("RLM fallback rate", f"{100 * agg.fallback_rate:.1f}%")

        if not agg.per_dataset.empty:
            st.subheader("Per-dataset")
            st.dataframe(agg.per_dataset, width="stretch")

        st.subheader("Per-row")
        display_cols = [
            c
            for c in (
                "ab_id",
                "ab_query",
                "ab_latency_delta_ms",
                "ab_tokens_delta",
                "ab_judge_delta",
                "ab_with_rlm_was_fallback",
                "queries_dataset",
                "start_time",
            )
            if c in agg.per_row.columns
        ]
        st.dataframe(agg.per_row[display_cols], width="stretch")
