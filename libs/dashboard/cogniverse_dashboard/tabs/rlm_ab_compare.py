"""Dashboard tile — RLM A/B comparison spans.

Reads ``rlm.ab_compare`` spans (emitted by ``cogniverse-optim --mode
ab-compare``) from Phoenix and renders the per-row + aggregate view.
Each span carries the ``RLMABRunner.to_telemetry_dict()`` payload as
``openinference.*`` attributes, including the per-row ``ab_id`` that
ties paired arms together.

The data-loading + aggregation logic lives in :func:`load_ab_compare_data`
as a pure async function so it can be integration-tested against real
Phoenix without spinning up the Streamlit runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SPAN_NAME = "rlm.ab_compare"


@dataclass
class ABCompareAggregate:
    """Aggregated per-tenant A/B comparison stats over a time window."""

    rows: int = 0
    avg_latency_delta_ms: Optional[float] = None
    avg_tokens_delta: Optional[float] = None
    avg_judge_delta: Optional[float] = None
    fallback_rate: Optional[float] = None
    per_row: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_dataset: pd.DataFrame = field(default_factory=pd.DataFrame)


def _fmt_delta(value: Optional[float]) -> str:
    """Format a delta metric; a real 0.0 must show as '0.0', not '—'."""
    return f"{value:.1f}" if value is not None else "—"


def _attribute_to_dict(spans_df: pd.DataFrame) -> pd.DataFrame:
    """Lift each ``openinference.*`` attribute into a typed value.

    Phoenix returns the same span attribute under different keys
    depending on the version + transport:
      * a nested dict column named ``attributes`` (in-memory exporter);
      * flat columns prefixed ``attributes.openinference.X``;
      * flat columns prefixed just ``openinference.X`` (Phoenix HTTP API).
    Tolerate all three so the dashboard tile works regardless of how
    the spans landed.
    """
    if spans_df.empty:
        return pd.DataFrame()

    # Strip both possible attribute-name prefixes; the rest is the
    # logical attribute name the harness set (e.g. ``ab_id``).
    def _strip_prefix(key: str) -> Optional[str]:
        for prefix in ("attributes.openinference.", "openinference."):
            if key.startswith(prefix):
                return key.removeprefix(prefix)
        return None

    out_records: List[Dict[str, Any]] = []
    for _, row in spans_df.iterrows():
        rec: Dict[str, Any] = {
            "trace_id": row.get("trace_id"),
            "span_id": row.get("context.span_id") or row.get("span_id"),
            "start_time": row.get("start_time"),
        }
        # Case 1: nested ``attributes`` dict (in-memory exporter).
        attrs = row.get("attributes")
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                stripped = _strip_prefix(k)
                if stripped is not None:
                    rec[stripped] = v
        # Case 2: flat ``[attributes.]openinference.X`` columns. We always
        # iterate them too — some Phoenix builds return BOTH (a nested
        # ``attributes`` blob AND flat columns); flat wins on conflict
        # because the live API surface is the source of truth for the tile.
        for col, val in row.items():
            if not isinstance(col, str):
                continue
            stripped = _strip_prefix(col)
            if stripped is not None and val is not None:
                # Don't overwrite a flat value with a nested NaN.
                if stripped not in rec or rec.get(stripped) is None:
                    rec[stripped] = val
        out_records.append(rec)
    df = pd.DataFrame(out_records)
    # Coerce numeric columns even if Phoenix returned them as strings.
    for c in (
        "ab_without_rlm_latency_ms",
        "ab_with_rlm_latency_ms",
        "ab_without_rlm_tokens",
        "ab_with_rlm_tokens",
        "ab_latency_delta_ms",
        "ab_tokens_delta",
        "ab_judge_delta",
        "ab_with_rlm_judge",
        "ab_without_rlm_judge",
        "ab_context_chars",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_ab_compare(spans_df: pd.DataFrame) -> ABCompareAggregate:
    """Compute the per-tenant + per-dataset aggregate from raw spans.

    Pure function over a Phoenix span DataFrame. Used by the Streamlit
    tile and by the integration test alike. Empty input returns a
    zero-row aggregate, never None.
    """
    df = _attribute_to_dict(spans_df)
    if df.empty:
        return ABCompareAggregate()

    n = len(df)

    def _avg(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        vals = df[col].dropna()
        return float(vals.mean()) if len(vals) else None

    # Fallback rate: spans where ab_with_rlm_was_fallback was set truthy.
    fallback_rate: Optional[float] = None
    if "ab_with_rlm_was_fallback" in df.columns:
        flags = df["ab_with_rlm_was_fallback"].fillna(False)

        def _truthy(x):
            return bool(x) and str(x).lower() not in ("false", "0", "")

        fallback_rate = sum(_truthy(v) for v in flags) / n

    per_dataset = pd.DataFrame()
    if "queries_dataset" in df.columns:
        grp = df.groupby("queries_dataset", dropna=False)
        # Build the agg-spec dynamically — the optional ``ab_judge_delta``
        # only exists when a judge was supplied to the harness; including
        # it unconditionally would raise KeyError on judgeless runs.
        agg_spec = {
            "rows": ("ab_id", "count"),
            "avg_latency_delta_ms": ("ab_latency_delta_ms", "mean"),
            "avg_tokens_delta": ("ab_tokens_delta", "mean"),
        }
        if "ab_judge_delta" in df.columns:
            agg_spec["avg_judge_delta"] = ("ab_judge_delta", "mean")
        per_dataset = grp.agg(**agg_spec).reset_index()

    return ABCompareAggregate(
        rows=n,
        avg_latency_delta_ms=_avg("ab_latency_delta_ms"),
        avg_tokens_delta=_avg("ab_tokens_delta"),
        avg_judge_delta=_avg("ab_judge_delta"),
        fallback_rate=fallback_rate,
        per_row=df,
        per_dataset=per_dataset,
    )


async def load_ab_compare_data(
    *,
    phoenix_http_endpoint: str,
    tenant_id: str,
    lookback_hours: float = 24.0,
) -> ABCompareAggregate:
    """Fetch ``rlm.ab_compare`` spans from a real Phoenix instance.

    Returns an :class:`ABCompareAggregate`. The Streamlit tile renders
    this; the integration test asserts against this directly so the
    dashboard's data path is exercised end-to-end without a Streamlit
    runtime.
    """
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_http_endpoint,
            # gRPC endpoint isn't used for queries but the provider
            # contract requires it.
            "grpc_endpoint": "localhost:4317",
        }
    )

    # Phoenix project naming: the provider derives project_name from
    # tenant_id + service_template. The harness emits spans with the
    # default tracer, which lands in the tenant's project.
    project_name = f"cogniverse-{tenant_id}"

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    # Phoenix's default get_spans_dataframe returns rows with span
    # metadata only — no flattened ``attributes.*`` columns. To get the
    # ``openinference.*`` attributes the harness emits, we run a
    # SpanQuery that explicitly selects them.
    try:
        from phoenix.client.types.spans import SpanQuery

        client = provider.traces._get_client()  # type: ignore[attr-defined]
        select_fields = [
            "name",
            "context.span_id",
            "trace_id",
            "start_time",
            "attributes.openinference.ab_id",
            "attributes.openinference.ab_query",
            "attributes.openinference.ab_context_chars",
            "attributes.openinference.ab_latency_delta_ms",
            "attributes.openinference.ab_tokens_delta",
            "attributes.openinference.ab_judge_delta",
            "attributes.openinference.ab_with_rlm_was_fallback",
            "attributes.openinference.ab_with_rlm_latency_ms",
            "attributes.openinference.ab_without_rlm_latency_ms",
            "attributes.openinference.ab_with_rlm_tokens",
            "attributes.openinference.ab_without_rlm_tokens",
            "attributes.openinference.ab_with_rlm_judge",
            "attributes.openinference.ab_without_rlm_judge",
            "attributes.openinference.queries_dataset",
            "attributes.openinference.tenant_id",
        ]
        query = SpanQuery().select(*select_fields).where(f"name == '{SPAN_NAME}'")
        spans_df = await client.spans.get_spans_dataframe(
            query=query,
            project_identifier=project_name,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
            timeout=120,
        )
    except Exception as exc:
        logger.warning(
            "ab-compare tile: span query failed for tenant=%s: %s",
            tenant_id,
            exc,
        )
        return ABCompareAggregate()

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
    phoenix_url = st.session_state.get("phoenix_url") or st.text_input(
        "Phoenix HTTP URL", value="http://localhost:6006"
    )
    lookback_hours = st.number_input(
        "Lookback (hours)", min_value=0.1, value=24.0, step=1.0
    )

    if st.button("Load A/B comparison data"):
        import asyncio

        with st.spinner("Querying Phoenix…"):
            agg = asyncio.run(
                load_ab_compare_data(
                    phoenix_http_endpoint=phoenix_url,
                    tenant_id=tenant_id,
                    lookback_hours=lookback_hours,
                )
            )

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
