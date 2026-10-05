"""Metrics over a tenant's telemetry spans for the operations views.

Every route reads all of the tenant's spans of one name in the window (not a
first page of them) from the tenant's telemetry project and aggregates them
with ``cogniverse_foundation.telemetry.span_metrics``. A telemetry backend
that fails the read answers 502; it never reads as an empty window.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import SPAN_NAME_PROFILE_SELECTION
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_foundation.telemetry.span_metrics import (
    AB_COMPARE_SPAN_NAME,
    aggregate_ab_compare,
    profile_selection_metrics,
    recorded_flag,
    trace_rows,
    trace_statistics,
)
from cogniverse_runtime.http_errors import failure_response

logger = logging.getLogger(__name__)

router = APIRouter()

Lookback = Query(24, ge=1, le=24 * 30)


class ModalityMetrics(BaseModel):
    modality: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float


class ProfileSelectionMetrics(BaseModel):
    modalities: List[ModalityMetrics]


class DatasetComparison(BaseModel):
    queries_dataset: Optional[str]
    rows: int
    avg_latency_delta_ms: Optional[float]
    avg_tokens_delta: Optional[float]
    avg_judge_delta: Optional[float]


class Comparison(BaseModel):
    ab_id: Optional[str]
    query: Optional[str]
    queries_dataset: Optional[str]
    latency_delta_ms: Optional[float]
    tokens_delta: Optional[float]
    judge_delta: Optional[float]
    with_rlm_was_fallback: bool
    start_time: Optional[str]


class RlmAbComparison(BaseModel):
    rows: int
    avg_latency_delta_ms: Optional[float]
    avg_tokens_delta: Optional[float]
    avg_judge_delta: Optional[float]
    fallback_rate: Optional[float]
    per_dataset: List[DatasetComparison]
    comparisons: List[Comparison]


class Trace(BaseModel):
    trace_id: Optional[str]
    span_id: Optional[str]
    start_time: str
    duration_ms: float
    operation: str
    succeeded: bool
    profile: Optional[str]
    strategy: Optional[str]
    error: Optional[str]


class Latency(BaseModel):
    mean: Optional[float]
    min: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    max: Optional[float]


class OperationStatistics(BaseModel):
    operation: str
    count: int
    mean_ms: float
    p95_ms: float
    error_rate: float


class TraceStatistics(BaseModel):
    requests: int
    succeeded: int
    failed: int
    success_rate: Optional[float]
    latency_ms: Latency
    outlier_bounds_ms: Optional[Dict[str, float]]
    by_operation: List[OperationStatistics]


class TraceFacets(BaseModel):
    operations: List[str]
    profiles: List[str]
    strategies: List[str]


class TraceAnalytics(BaseModel):
    facets: TraceFacets
    statistics: TraceStatistics
    traces: List[Trace]


async def _window_spans(
    tenant_id: str, lookback_hours: int, *, span_name: str = "", roots_only=False
):
    manager = get_telemetry_manager()
    project = manager.config.get_project_name(tenant_id)
    end = datetime.now(timezone.utc)
    filters: Dict[str, Any] = {"roots_only": True} if roots_only else {}
    if span_name:
        filters["name"] = span_name
    try:
        provider = manager.get_provider(tenant_id=tenant_id, project_name=project)
        return await provider.traces.get_all_spans(
            project=project,
            start_time=end - timedelta(hours=lookback_hours),
            end_time=end,
            filters=filters,
        )
    except Exception as exc:
        what = f"the {span_name} spans" if span_name else "the traces"
        raise failure_response(
            502,
            "telemetry_unavailable",
            f"Could not read {what} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


@router.get(
    "/{tenant_id}/telemetry/profile-selection",
    response_model=ProfileSelectionMetrics,
)
async def profile_selection(tenant_id: str, lookback_hours: int = Lookback):
    """Per-modality count, latency and success rate of the tenant's profile
    selections in the last ``lookback_hours``."""
    tenant_id = canonical_tenant_id(tenant_id)
    spans = await _window_spans(
        tenant_id, lookback_hours, span_name=SPAN_NAME_PROFILE_SELECTION
    )
    return ProfileSelectionMetrics(modalities=profile_selection_metrics(spans))


@router.get("/{tenant_id}/telemetry/rlm-ab", response_model=RlmAbComparison)
async def rlm_ab(tenant_id: str, lookback_hours: int = Lookback):
    """The tenant's RLM A/B comparisons in the last ``lookback_hours``:
    averages, per queries dataset, and each compared row newest first."""
    tenant_id = canonical_tenant_id(tenant_id)
    spans = await _window_spans(
        tenant_id, lookback_hours, span_name=AB_COMPARE_SPAN_NAME
    )
    aggregate = aggregate_ab_compare(spans)
    return RlmAbComparison(
        rows=aggregate.rows,
        avg_latency_delta_ms=aggregate.avg_latency_delta_ms,
        avg_tokens_delta=aggregate.avg_tokens_delta,
        avg_judge_delta=aggregate.avg_judge_delta,
        fallback_rate=aggregate.fallback_rate,
        per_dataset=[
            DatasetComparison(
                queries_dataset=_text(row.get("queries_dataset")),
                rows=int(row["rows"]),
                avg_latency_delta_ms=_number(row.get("avg_latency_delta_ms")),
                avg_tokens_delta=_number(row.get("avg_tokens_delta")),
                avg_judge_delta=_number(row.get("avg_judge_delta")),
            )
            for _, row in aggregate.per_dataset.iterrows()
        ],
        comparisons=[
            Comparison(
                ab_id=_text(row.get("ab_id")),
                query=_text(row.get("ab_query")),
                queries_dataset=_text(row.get("queries_dataset")),
                latency_delta_ms=_number(row.get("ab_latency_delta_ms")),
                tokens_delta=_number(row.get("ab_tokens_delta")),
                judge_delta=_number(row.get("ab_judge_delta")),
                with_rlm_was_fallback=recorded_flag(
                    row.get("ab_with_rlm_was_fallback")
                ),
                start_time=_text(row.get("start_time")),
            )
            for _, row in aggregate.per_row.iterrows()
        ],
    )


@router.get("/{tenant_id}/telemetry/traces", response_model=TraceAnalytics)
async def traces(
    tenant_id: str,
    lookback_hours: int = Lookback,
    operation: str = "",
    profile: List[str] = Query([]),
    strategy: List[str] = Query([]),
):
    """The tenant's traces (root spans) in the last ``lookback_hours``,
    newest first, with their statistics.

    ``operation`` keeps traces whose name contains it (any case); ``profile``
    and ``strategy`` keep traces with one of the given values. ``facets``
    lists the values present in the whole window.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    rows = trace_rows(await _window_spans(tenant_id, lookback_hours, roots_only=True))
    facets = TraceFacets(
        operations=sorted({row["operation"] for row in rows}),
        profiles=sorted({row["profile"] for row in rows if row["profile"]}),
        strategies=sorted({row["strategy"] for row in rows if row["strategy"]}),
    )
    needle = operation.casefold()
    kept = [
        row
        for row in rows
        if needle in row["operation"].casefold()
        and (not profile or row["profile"] in profile)
        and (not strategy or row["strategy"] in strategy)
    ]
    return TraceAnalytics(
        facets=facets,
        statistics=TraceStatistics(**trace_statistics(kept)),
        traces=[Trace(**row) for row in kept],
    )
