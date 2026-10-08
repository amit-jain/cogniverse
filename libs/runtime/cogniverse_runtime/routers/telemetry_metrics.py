"""Metrics over a tenant's telemetry spans for the operations views.

Every route reads all of the tenant's spans in the window (not a first page
of them) from the tenant's telemetry project and aggregates them with
``cogniverse_foundation.telemetry.span_metrics``, or scores them against the
tenant's golden set with ``cogniverse_evaluation.recorded_searches``. A
telemetry backend that fails the read answers 502, one that does not answer
within ``SPAN_READ_BUDGET_S`` answers 504, and a tenant no telemetry provider
can be built for answers 503; none of them reads as an empty window.
"""

import asyncio
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import httpx
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
    dataset_golden_rows,
    score_recorded_searches,
)
from cogniverse_foundation.telemetry.config import SPAN_NAME_PROFILE_SELECTION
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_foundation.telemetry.providers.base import DatasetSummary
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

Lookback = Query(24, ge=1, le=24 * 30)

# The longest a route waits for the spans of its window before it answers
# that the store is slow.
SPAN_READ_BUDGET_S = 60.0


class ModalityMetrics(BaseModel):
    modality: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float


class ProfileSelectionMetrics(BaseModel):
    project: str
    spans: int
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


class RootCauseAnalysis(BaseModel):
    traces: int
    failed: int
    slow: int
    failure_rate: float
    slow_threshold_ms: Optional[float]
    root_causes: List[RootCause]
    recommendations: List[Recommendation]


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


class EvaluationDataset(BaseModel):
    id: str
    name: str
    example_count: int
    created_at: str
    description: str


class EvaluationDatasets(BaseModel):
    phoenix_url: Optional[str]
    datasets: List[EvaluationDataset]


class DatasetEvaluation(GoldenEvaluation):
    dataset: EvaluationDataset


# The address a browser opens the telemetry store's UI at; datasets link
# to it when it is set.
PHOENIX_UI_URL_ENV = "PHOENIX_UI_URL"


def _project(tenant_id: str) -> str:
    return get_telemetry_manager().config.get_project_name(tenant_id)


async def _window_spans(
    tenant_id: str, lookback_hours: float, *, span_name: str = "", roots_only=False
):
    manager = get_telemetry_manager()
    project = _project(tenant_id)
    end = datetime.now(timezone.utc)
    filters: Dict[str, Any] = {"roots_only": True} if roots_only else {}
    if span_name:
        filters["name"] = span_name
    what = f"the {span_name} spans" if span_name else "the traces"
    try:
        provider = manager.get_provider(tenant_id=tenant_id, project_name=project)
    except Exception as exc:
        raise failure_response(
            503,
            "telemetry_unconfigured",
            f"No telemetry provider could be built for tenant {tenant_id}, so "
            f"{what} cannot be read; the runtime log names the cause.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    try:
        return await asyncio.wait_for(
            provider.traces.get_all_spans(
                project=project,
                start_time=end - timedelta(hours=lookback_hours),
                end_time=end,
                filters=filters,
            ),
            SPAN_READ_BUDGET_S,
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise failure_response(
            504,
            "telemetry_slow",
            f"The telemetry store did not return {what} of tenant {tenant_id} "
            f"within {SPAN_READ_BUDGET_S:g} s. It is slow, not empty; retry "
            "shortly.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except Exception as exc:
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
    selections in the last ``lookback_hours``, the project they are read
    from, and how many selection spans the window holds (spans without a
    modality are counted there but not in ``modalities``)."""
    tenant_id = canonical_tenant_id(tenant_id)
    spans = await _window_spans(
        tenant_id, lookback_hours, span_name=SPAN_NAME_PROFILE_SELECTION
    )
    return ProfileSelectionMetrics(
        project=_project(tenant_id),
        spans=len(spans),
        modalities=profile_selection_metrics(spans),
    )


@router.get("/{tenant_id}/telemetry/rlm-ab", response_model=RlmAbComparison)
async def rlm_ab(tenant_id: str, lookback_hours: float = Query(24, ge=0.1, le=24 * 30)):
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
    kept = _filtered(rows, operation, profile, strategy)
    return TraceAnalytics(
        facets=facets,
        statistics=TraceStatistics(**trace_statistics(kept)),
        traces=[Trace(**row) for row in kept],
    )


def _filtered(
    rows: List[Dict[str, Any]], operation: str, profile: List[str], strategy: List[str]
) -> List[Dict[str, Any]]:
    """Rows whose operation contains ``operation`` (any case) and whose
    profile and strategy are among the given ones, when any are given."""
    needle = operation.casefold()
    return [
        row
        for row in rows
        if needle in row["operation"].casefold()
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
    )


@router.get("/{tenant_id}/telemetry/root-causes", response_model=RootCauseAnalysis)
async def root_causes(
    tenant_id: str,
    lookback_hours: int = Lookback,
    operation: str = "",
    profile: List[str] = Query([]),
    strategy: List[str] = Query([]),
    include_slow: bool = True,
    slow_percentile: int = Query(95, ge=50, le=99),
):
    """Root-cause hypotheses for the failed (and slow) traces among the
    tenant's traces in the last ``lookback_hours``, filtered as
    ``/telemetry/traces`` filters them."""
    tenant_id = canonical_tenant_id(tenant_id)
    rows = trace_rows(await _window_spans(tenant_id, lookback_hours, roots_only=True))
    kept = _filtered(rows, operation, profile, strategy)
    return await asyncio.to_thread(
        root_cause_analysis,
        kept,
        include_slow=include_slow,
        slow_percentile=slow_percentile,
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


def _dataset(summary: DatasetSummary) -> EvaluationDataset:
    return EvaluationDataset(
        id=summary.id,
        name=summary.name,
        example_count=summary.example_count,
        created_at=summary.created_at.isoformat(),
        description=summary.description,
    )


async def _tenant_datasets(tenant_id: str) -> Tuple[Any, List[DatasetSummary]]:
    """The tenant's telemetry provider and the evaluation datasets it owns,
    newest first."""
    try:
        provider = get_telemetry_manager().get_provider(tenant_id=tenant_id)
    except Exception as exc:
        raise failure_response(
            503,
            "telemetry_unconfigured",
            f"No telemetry provider could be built for tenant {tenant_id}, so "
            "its datasets cannot be listed; the runtime log names the cause.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    try:
        summaries = await provider.datasets.describe_datasets()
    except Exception as exc:
        raise failure_response(
            502,
            "dataset_store_unavailable",
            f"Could not list the datasets of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    return provider, [s for s in summaries if s.tenant_id == tenant_id]


@router.get("/{tenant_id}/evaluation/datasets", response_model=EvaluationDatasets)
async def evaluation_datasets(tenant_id: str):
    """The evaluation datasets the tenant owns, newest first, and the
    address of the telemetry store's UI (``PHOENIX_UI_URL``; null when the
    runtime is not given one)."""
    tenant_id = canonical_tenant_id(tenant_id)
    _, owned = await _tenant_datasets(tenant_id)
    return EvaluationDatasets(
        phoenix_url=os.environ.get(PHOENIX_UI_URL_ENV, "").rstrip("/") or None,
        datasets=[_dataset(summary) for summary in owned],
    )


@router.get("/{tenant_id}/evaluation/dataset", response_model=DatasetEvaluation)
async def dataset_evaluation(
    tenant_id: str,
    dataset_id: str,
    lookback_hours: int = Query(168, ge=1, le=24 * 90),
):
    """The tenant's searches of the queries of its dataset ``dataset_id`` in
    the last ``lookback_hours``, scored against the dataset's expected
    sources as ``/evaluation/golden`` scores them against the golden set."""
    tenant_id = canonical_tenant_id(tenant_id)
    provider, owned = await _tenant_datasets(tenant_id)
    summary = next((d for d in owned if d.id == dataset_id), None)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} has no dataset {dataset_id}.",
        )
    try:
        examples = await provider.datasets.get_dataset(summary.name)
    except Exception as exc:
        raise failure_response(
            502,
            "dataset_store_unavailable",
            f"Could not read dataset {summary.name} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    rows = dataset_golden_rows(examples)
    spans = await _window_spans(tenant_id, lookback_hours, span_name=SEARCH_SPAN_NAME)
    return DatasetEvaluation(
        dataset=_dataset(summary), **score_recorded_searches(spans, rows)
    )
