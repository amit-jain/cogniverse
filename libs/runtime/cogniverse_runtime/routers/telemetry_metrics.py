"""Metrics over a tenant's telemetry spans for the operations views.

Every route reads all of the tenant's spans in the window (not a first page
of them) from the tenant's telemetry project and aggregates them with
``cogniverse_foundation.telemetry.span_metrics``, or scores them against the
tenant's golden set with ``cogniverse_evaluation.recorded_searches``. A
telemetry backend that fails the read answers 502; it never reads as an empty
window.
"""

import asyncio
import logging
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.optimizer.golden_set_ground_truth import (
    GoldenSetGroundTruthError,
    GoldenSetGroundTruthMissingError,
    GoldenSetGroundTruthStoreUnavailableError,
    load_golden_set_ground_truth_rows,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_evaluation.analysis.root_cause_analysis import RootCauseAnalyzer
from cogniverse_evaluation.recorded_searches import (
    SEARCH_SPAN_NAME,
    score_recorded_searches,
)
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
from cogniverse_runtime.quality_monitor_cli import GOLDEN_SET_UPLOAD_ROUTE

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_WINDOW = timedelta(days=30)
Lookback = Query(24, ge=1, le=MAX_WINDOW.days * 24)
WindowStart = Query(
    None,
    description="Start of the window (ISO 8601 with a timezone); with ``end`` "
    "it replaces ``lookback_hours``.",
)
WindowEnd = Query(None, description="End of the window (ISO 8601 with a timezone).")

_phoenix_public_url: Optional[str] = None


def set_phoenix_public_url(url: Optional[str]) -> None:
    """The Phoenix UI address a browser reaches, for the links the
    ``/telemetry/phoenix`` route answers; ``None`` turns the links off."""
    global _phoenix_public_url
    _phoenix_public_url = url.rstrip("/") if url else None


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


class RootCause(BaseModel):
    hypothesis: str
    confidence: float
    category: str
    evidence: List[str]
    affected_traces: List[str]
    suggested_action: str


class Recommendation(BaseModel):
    priority: str
    category: str
    recommendation: str
    details: List[str]
    affected_components: List[str]


class Tally(BaseModel):
    value: str
    count: int


class HourlyFailures(BaseModel):
    hour: int
    requests: int
    failed: int
    failure_rate: float


class FailureBurst(BaseModel):
    start_time: str
    end_time: str
    failures: int
    duration_minutes: float
    trace_ids: List[str]


class FailureAnalysis(BaseModel):
    """What the failed traces have in common, each tally most frequent
    first."""

    error_types: List[Tally]
    operations: List[Tally]
    profiles: List[Tally]
    strategies: List[Tally]
    hours: List[HourlyFailures]
    bursts: List[FailureBurst]


class SlowOperation(BaseModel):
    operation: str
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    sample_ms: List[float]


class LatencyShift(BaseModel):
    slow_mean_ms: float
    slow_std_ms: float
    normal_mean_ms: float
    normal_std_ms: float
    slowdown_factor: float


class PerformanceAnalysis(BaseModel):
    """Where the slow traces sit and how much slower they are than the
    rest of the successful traces."""

    percentile: int
    threshold_ms: float
    operations: List[SlowOperation]
    profiles: List[Tally]
    strategies: List[Tally]
    latency: LatencyShift


class RootCauseAnalysis(BaseModel):
    traces: int
    failed: int
    slow: int
    failure_rate: float
    slow_threshold_ms: Optional[float]
    root_causes: List[RootCause]
    recommendations: List[Recommendation]
    failure_analysis: Optional[FailureAnalysis]
    performance_analysis: Optional[PerformanceAnalysis]


class PhoenixLinks(BaseModel):
    phoenix_url: Optional[str]
    project: str
    project_url: Optional[str]


class StrategyScores(BaseModel):
    profile: str
    strategy: str
    queries: int
    mrr: float
    ndcg: float
    recall_at_1: float
    recall_at_5: float
    precision_at_5: float
    success_rate: float


class QueryScores(BaseModel):
    profile: str
    strategy: str
    query: str
    expected: List[str]
    retrieved: List[str]
    searched_at: str
    trace_id: Optional[str]
    mrr: float
    ndcg: float
    recall_at_1: float
    recall_at_5: float
    precision_at_5: float


class GoldenEvaluation(BaseModel):
    golden_queries: int
    strategies: List[StrategyScores]
    queries: List[QueryScores]
    unsearched_queries: List[str]
    failed_searches: int
    unscored_searches: int


def _window(
    lookback_hours: int, start: Optional[datetime], end: Optional[datetime]
) -> tuple[datetime, datetime]:
    """``start`` to ``end`` when both are given, else the last
    ``lookback_hours``.

    Raises:
        HTTPException 422: only one bound given, a bound without a timezone,
            an empty window or one longer than ``MAX_WINDOW``.
    """
    if start is None and end is None:
        now = datetime.now(timezone.utc)
        return now - timedelta(hours=lookback_hours), now
    if start is None or end is None:
        raise HTTPException(422, "Give both start and end, or neither.")
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(422, "start and end must carry a timezone.")
    if start >= end:
        raise HTTPException(422, "start must be before end.")
    if end - start > MAX_WINDOW:
        raise HTTPException(422, f"The window may span at most {MAX_WINDOW.days} days.")
    return start, end


async def _window_spans(
    tenant_id: str,
    lookback_hours: int,
    *,
    span_name: str = "",
    roots_only=False,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
):
    manager = get_telemetry_manager()
    project = manager.config.get_project_name(tenant_id)
    start, end = _window(lookback_hours, start, end)
    filters: Dict[str, Any] = {"roots_only": True} if roots_only else {}
    if span_name:
        filters["name"] = span_name
    try:
        provider = manager.get_provider(tenant_id=tenant_id, project_name=project)
        return await provider.traces.get_all_spans(
            project=project,
            start_time=start,
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
    start: Optional[datetime] = WindowStart,
    end: Optional[datetime] = WindowEnd,
    operation: str = "",
    profile: List[str] = Query([]),
    strategy: List[str] = Query([]),
):
    """The tenant's traces (root spans) in the last ``lookback_hours`` (or
    from ``start`` to ``end``), newest first, with their statistics.

    ``operation`` is a regular expression (any case) a trace's name must
    match somewhere; ``profile`` and ``strategy`` keep traces with one of the
    given values. ``facets`` lists the values present in the whole window.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    pattern = _operation_pattern(operation)
    rows = trace_rows(
        await _window_spans(
            tenant_id, lookback_hours, roots_only=True, start=start, end=end
        )
    )
    facets = TraceFacets(
        operations=sorted({row["operation"] for row in rows}),
        profiles=sorted({row["profile"] for row in rows if row["profile"]}),
        strategies=sorted({row["strategy"] for row in rows if row["strategy"]}),
    )
    kept = _filtered(rows, pattern, profile, strategy)
    return TraceAnalytics(
        facets=facets,
        statistics=TraceStatistics(**trace_statistics(kept)),
        traces=[Trace(**row) for row in kept],
    )


def _operation_pattern(operation: str) -> re.Pattern:
    """``operation`` compiled as a case-insensitive regular expression.

    Raises:
        HTTPException 422: it is not a valid regular expression.
    """
    try:
        return re.compile(operation, re.IGNORECASE)
    except re.error as exc:
        raise HTTPException(
            422, f"operation is not a valid regular expression: {exc}"
        ) from exc


def _filtered(
    rows: List[Dict[str, Any]],
    operation: re.Pattern,
    profile: List[str],
    strategy: List[str],
) -> List[Dict[str, Any]]:
    """Rows whose operation matches ``operation`` and whose profile and
    strategy are among the given ones, when any are given."""
    return [
        row
        for row in rows
        if operation.search(row["operation"])
        and (not profile or row["profile"] in profile)
        and (not strategy or row["strategy"] in strategy)
    ]


def root_cause_analysis(
    rows: List[Dict[str, Any]], *, include_slow: bool, slow_percentile: int
) -> RootCauseAnalysis:
    """``RootCauseAnalyzer`` over trace rows: failures, and (when
    ``include_slow``) successful traces slower than ``slow_percentile`` of
    the successful ones."""
    traces = [
        SimpleNamespace(
            trace_id=row["trace_id"] or row["span_id"] or "",
            status="success" if row["succeeded"] else "error",
            error=row["error"],
            operation=row["operation"],
            profile=row["profile"],
            strategy=row["strategy"],
            duration_ms=row["duration_ms"],
            timestamp=datetime.fromisoformat(row["start_time"]),
        )
        for row in rows
    ]
    analysis = RootCauseAnalyzer().analyze_failures(
        traces,
        include_performance=include_slow,
        performance_threshold_percentile=slow_percentile,
    )
    summary = analysis["summary"]
    threshold = analysis["performance_analysis"].get("threshold")
    return RootCauseAnalysis(
        traces=summary["total_traces"],
        failed=summary["failed_traces"],
        slow=summary["performance_degraded"],
        failure_rate=summary["failure_rate"],
        slow_threshold_ms=float(threshold) if threshold is not None else None,
        root_causes=[
            RootCause(
                hypothesis=cause.hypothesis,
                confidence=float(cause.confidence),
                category=cause.category,
                evidence=list(cause.evidence),
                affected_traces=list(cause.affected_traces),
                suggested_action=cause.suggested_action,
            )
            for cause in analysis["root_causes"]
        ],
        recommendations=[
            Recommendation(
                priority=item["priority"],
                category=item["category"],
                recommendation=item["recommendation"],
                details=list(item["details"]),
                affected_components=sorted(item["affected_components"]),
            )
            for item in analysis["recommendations"]
        ],
        failure_analysis=_failure_analysis(analysis["failure_analysis"]),
        performance_analysis=_performance_analysis(analysis["performance_analysis"]),
    )


def _tallies(counter: Counter) -> List[Tally]:
    return [
        Tally(value=str(value), count=count) for value, count in counter.most_common()
    ]


def _failure_analysis(found: Dict[str, Any]) -> Optional[FailureAnalysis]:
    """The analyzer's failure patterns; ``None`` when nothing failed."""
    if not found:
        return None
    temporal = found["temporal_patterns"]
    return FailureAnalysis(
        error_types=_tallies(found["error_types"]),
        operations=_tallies(found["failed_operations"]),
        profiles=_tallies(found["failed_profiles"]),
        strategies=_tallies(found["failed_strategies"]),
        hours=sorted(
            (
                HourlyFailures(
                    hour=item["hour"],
                    requests=item["total_requests"],
                    failed=item["failed_requests"],
                    failure_rate=float(item["failure_rate"]),
                )
                for item in temporal
                if item["type"] == "hourly"
            ),
            key=lambda item: item.hour,
        ),
        bursts=[
            FailureBurst(
                start_time=item["start_time"],
                end_time=item["end_time"],
                failures=item["failure_count"],
                duration_minutes=float(item["duration_minutes"]),
                trace_ids=list(item["trace_ids"]),
            )
            for item in temporal
            if item["type"] == "burst"
        ],
    )


def _performance_analysis(found: Dict[str, Any]) -> Optional[PerformanceAnalysis]:
    """The analyzer's slow-trace patterns; ``None`` when no trace was slow."""
    if not found:
        return None
    latency = found["latency_distribution"]
    return PerformanceAnalysis(
        percentile=int(found["threshold_percentile"]),
        threshold_ms=float(found["threshold"]),
        operations=sorted(
            (
                SlowOperation(
                    operation=operation,
                    count=int(stats["count"]),
                    mean_ms=float(stats["avg_duration"]),
                    min_ms=float(stats["min_duration"]),
                    max_ms=float(stats["max_duration"]),
                    sample_ms=[float(value) for value in stats["durations"]],
                )
                for operation, stats in found["slow_operations"].items()
            ),
            key=lambda item: (-item.count, item.operation),
        ),
        profiles=_tallies(found["slow_profiles"]),
        strategies=_tallies(found["slow_strategies"]),
        latency=LatencyShift(
            slow_mean_ms=float(latency["slow_mean"]),
            slow_std_ms=float(latency["slow_std"]),
            normal_mean_ms=float(latency["normal_mean"]),
            normal_std_ms=float(latency["normal_std"]),
            slowdown_factor=float(latency["slowdown_factor"]),
        ),
    )


@router.get("/{tenant_id}/telemetry/root-causes", response_model=RootCauseAnalysis)
async def root_causes(
    tenant_id: str,
    lookback_hours: int = Lookback,
    start: Optional[datetime] = WindowStart,
    end: Optional[datetime] = WindowEnd,
    operation: str = "",
    profile: List[str] = Query([]),
    strategy: List[str] = Query([]),
    include_slow: bool = True,
    slow_percentile: int = Query(95, ge=50, le=99),
):
    """Root-cause hypotheses for the failed (and slow) traces among the
    tenant's traces in the window, filtered as ``/telemetry/traces`` filters
    them."""
    tenant_id = canonical_tenant_id(tenant_id)
    pattern = _operation_pattern(operation)
    rows = trace_rows(
        await _window_spans(
            tenant_id, lookback_hours, roots_only=True, start=start, end=end
        )
    )
    kept = _filtered(rows, pattern, profile, strategy)
    return await asyncio.to_thread(
        root_cause_analysis,
        kept,
        include_slow=include_slow,
        slow_percentile=slow_percentile,
    )


@router.get("/{tenant_id}/telemetry/phoenix", response_model=PhoenixLinks)
async def phoenix_links(tenant_id: str):
    """Where a browser opens the tenant's traces in Phoenix: the Phoenix UI
    address set with ``set_phoenix_public_url`` and the tenant's project
    page there. Either is ``null`` when the address is not set or Phoenix
    has no project for the tenant yet."""
    tenant_id = canonical_tenant_id(tenant_id)
    manager = get_telemetry_manager()
    project = manager.config.get_project_name(tenant_id)
    if _phoenix_public_url is None:
        return PhoenixLinks(phoenix_url=None, project=project, project_url=None)
    try:
        provider = manager.get_provider(tenant_id=tenant_id, project_name=project)
        project_id = await provider.project_id(project)
    except Exception as exc:
        raise failure_response(
            502,
            "telemetry_unavailable",
            f"Could not read the Phoenix project of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    return PhoenixLinks(
        phoenix_url=_phoenix_public_url,
        project=project,
        project_url=(
            f"{_phoenix_public_url}/projects/{project_id}" if project_id else None
        ),
    )


async def _golden_rows(tenant_id: str) -> List[Dict[str, Any]]:
    manager = get_telemetry_manager()
    try:
        provider = manager.get_provider(tenant_id=tenant_id)
        return await load_golden_set_ground_truth_rows(
            ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)
        )
    except GoldenSetGroundTruthMissingError as exc:
        raise failure_response(
            404,
            exc.reason,
            f"Tenant {tenant_id} has no golden set. Upload one with "
            f"{GOLDEN_SET_UPLOAD_ROUTE.format(tenant_id=tenant_id)}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except GoldenSetGroundTruthStoreUnavailableError as exc:
        raise failure_response(
            502,
            exc.reason,
            f"Could not read the golden set of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except GoldenSetGroundTruthError as exc:
        raise failure_response(
            409,
            exc.reason,
            f"The golden set of tenant {tenant_id} cannot be used. Upload it "
            f"again with {GOLDEN_SET_UPLOAD_ROUTE.format(tenant_id=tenant_id)}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except Exception as exc:
        raise failure_response(
            502,
            "golden_set_store_unavailable",
            f"Could not read the golden set of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc


@router.get("/{tenant_id}/evaluation/golden", response_model=GoldenEvaluation)
async def golden_evaluation(
    tenant_id: str, lookback_hours: int = Query(168, ge=1, le=24 * 90)
):
    """The tenant's searches of its golden queries in the last
    ``lookback_hours``, scored against its golden set: per profile and
    strategy, and per query for the latest search of each."""
    tenant_id = canonical_tenant_id(tenant_id)
    golden_rows = await _golden_rows(tenant_id)
    spans = await _window_spans(tenant_id, lookback_hours, span_name=SEARCH_SPAN_NAME)
    return GoldenEvaluation(**score_recorded_searches(spans, golden_rows))
