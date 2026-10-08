"""Training-data and model routes behind the web client's optimization view.

Search-quality annotation of recorded searches, golden datasets built from
those annotations, the results of a synthetic-data run, the tenant's training
datasets, the XGBoost profile recommender and the optimization metrics all
read the tenant's telemetry project. A telemetry backend that fails a read
answers 502 (504 when it does not answer in time); it never reads as an empty
window.
"""

import asyncio
import json
import logging
import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from cogniverse_agents.approval import ApprovalStorageImpl
from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.routing.profile_performance_optimizer import (
    FEATURE_NAMES,
    ProfilePerformanceOptimizer,
)
from cogniverse_core.approval.interfaces import ApprovalBatch, ReviewItem
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_evaluation.data.datasets import DatasetManager
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingEvaluator
from cogniverse_evaluation.recorded_searches import SEARCH_SPAN_NAME
from cogniverse_foundation.config.unified_config import ApprovalConfig
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_foundation.telemetry.span_contract import read_span_io
from cogniverse_runtime.http_errors import canonical_tenant_or_400, failure_response
from cogniverse_runtime.routers import tenant as tenant_router
from cogniverse_sdk.document import result_source_title_key
from cogniverse_synthetic.approval.corrections import (
    correction_template,
    review_reasoning,
)
from cogniverse_synthetic.registry import (
    APPROVED_TRAINING_AGENT_BY_OPTIMIZER,
    get_optimizer_config,
)
from cogniverse_synthetic.schemas import SAMPLING_STRATEGIES

logger = logging.getLogger(__name__)

router = APIRouter()

SEARCH_ANNOTATION_NAME = "search_quality_annotation"
# Search results a reviewer sees, and a golden query keeps, per search.
TOP_RESULTS = 5
# Bound on one telemetry read. A store that does not answer in time is slow,
# not empty, and the caller says so.
TELEMETRY_READ_TIMEOUT_S = 60.0
# Key of the trained recommender in the tenant's artifact store.
PROFILE_MODEL_BLOB = ("model", "profile_performance_xgboost")
# Spans a span name must mention to count as an evaluation or a training run.
EVALUATION_SPAN_PATTERN = "eval|ndcg"
TRAINING_SPAN_PATTERN = "train|optim"
# The fewest labelled searches the recommender trains on.
PROFILE_TRAINING_MIN_SAMPLES = 10


def tenant_dataset_name(tenant_id: str, name: str) -> str:
    """The stored name of the tenant's dataset ``name``: dataset names carry
    their tenant, as the approved synthetic dataset's does."""
    return f"{name}-{canonical_tenant_id(tenant_id)}"


def _project(tenant_id: str) -> str:
    return get_telemetry_manager().config.get_project_name(tenant_id)


def _provider(tenant_id: str):
    return get_telemetry_manager().get_provider(
        tenant_id=tenant_id, project_name=_project(tenant_id)
    )


async def _bounded(read, what: str, tenant_id: str):
    """Await ``read``; an unanswered or failed read raises with context."""
    try:
        return await asyncio.wait_for(read, timeout=TELEMETRY_READ_TIMEOUT_S)
    except TimeoutError as exc:
        raise failure_response(
            504,
            "telemetry_timeout",
            f"Telemetry did not return {what} of tenant {tenant_id} within "
            f"{TELEMETRY_READ_TIMEOUT_S:g}s; the store is slow, not empty.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise failure_response(
            502,
            "telemetry_unavailable",
            f"Could not read {what} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc


async def _spans(
    tenant_id: str,
    start: datetime,
    end: datetime,
    what: str,
    *,
    name: Optional[str] = None,
) -> pd.DataFrame:
    provider = _provider(tenant_id)
    filters = {"name": name} if name else {}
    return await _bounded(
        provider.traces.get_all_spans(
            project=_project(tenant_id),
            start_time=start,
            end_time=end,
            filters=filters,
        ),
        what,
        tenant_id,
    )


def _text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _named_frame(spans: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Rows whose span name matches ``pattern``, case-insensitively. A null
    name (Phoenix can return one) does not match."""
    if spans.empty or "name" not in spans.columns:
        return spans.iloc[0:0]
    return spans[spans["name"].str.contains(pattern, case=False, na=False)]


# ---------------------------------------------------------------- annotations


class SearchAnnotation(BaseModel):
    label: str
    score: float
    annotation_type: Optional[str]
    notes: Optional[str]


class AnnotatableSearch(BaseModel):
    span_id: str
    trace_id: Optional[str]
    start_time: Optional[str]
    query: str
    results: List[str]
    profile: Optional[str]
    strategy: Optional[str]
    latency_ms: Optional[float]
    annotation: Optional[SearchAnnotation]


class AnnotatableSearches(BaseModel):
    searches: List[AnnotatableSearch]


def _result_title(row: Dict[str, Any]) -> str:
    return str(row.get("source_title") or row.get("id") or "Unknown")


def _latency_ms(span: Dict[str, Any]) -> Optional[float]:
    recorded = _number(span.get("attributes.latency_ms"))
    if recorded is not None:
        return recorded
    start, end = span.get("start_time"), span.get("end_time")
    if isinstance(start, (pd.Timestamp, datetime)) and isinstance(
        end, (pd.Timestamp, datetime)
    ):
        return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() * 1000
    return None


def _annotations_by_span(annotations: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """The newest search-quality annotation of each span; the frame is
    indexed by span id."""
    if annotations is None or annotations.empty:
        return {}
    if "created_at" in annotations.columns:
        annotations = annotations.sort_values("created_at")
    return {
        str(span_id): row
        for span_id, row in zip(
            annotations.index, annotations.to_dict("records"), strict=True
        )
    }


def _metadata(row: Dict[str, Any], key: str) -> Optional[str]:
    value = row.get(f"metadata.{key}")
    if value is None:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get(key)
    return _text(value)


async def _search_annotations(
    tenant_id: str, searches: pd.DataFrame, what: str
) -> pd.DataFrame:
    if searches.empty:
        return pd.DataFrame()
    provider = _provider(tenant_id)
    return await _bounded(
        provider.annotations.get_annotations(
            searches,
            project=_project(tenant_id),
            annotation_names=[SEARCH_ANNOTATION_NAME],
        ),
        what,
        tenant_id,
    )


@router.get("/{tenant_id}/search-annotations", response_model=AnnotatableSearches)
async def annotatable_searches(
    tenant_id: str, lookback_hours: int = Query(24, ge=1, le=168)
):
    """The tenant's recorded searches in the window, newest first, each with
    its top results and the search-quality annotation it already carries."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    end = datetime.now(timezone.utc)
    spans = await _spans(
        tenant_id,
        end - timedelta(hours=lookback_hours),
        end,
        "the recorded searches",
        name=SEARCH_SPAN_NAME,
    )
    annotations = _annotations_by_span(
        await _search_annotations(tenant_id, spans, "the search-quality annotations")
    )
    if not spans.empty and "start_time" in spans.columns:
        spans = spans.sort_values("start_time", ascending=False)
    searches = []
    for span in spans.to_dict("records"):
        io = read_span_io(span)
        output = io["output"]
        rows = (
            [row for row in output if isinstance(row, dict)]
            if isinstance(output, list)
            else []
        )
        span_id = str(span["context.span_id"])
        annotation = annotations.get(span_id)
        searches.append(
            AnnotatableSearch(
                span_id=span_id,
                trace_id=_text(span.get("context.trace_id")),
                start_time=_text(span.get("start_time")),
                query=str(io["input"] or ""),
                results=[_result_title(row) for row in rows[:TOP_RESULTS]],
                profile=_text(span.get("attributes.profile")),
                strategy=_text(span.get("attributes.strategy")),
                latency_ms=_latency_ms(span),
                annotation=SearchAnnotation(
                    label=str(annotation.get("result.label")),
                    score=float(annotation.get("result.score")),
                    annotation_type=_metadata(annotation, "annotation_type"),
                    notes=_metadata(annotation, "explanation"),
                )
                if annotation is not None
                else None,
            )
        )
    return AnnotatableSearches(searches=searches)


class AnnotationRequest(BaseModel):
    kind: Literal["thumbs", "stars", "relevance"]
    value: float
    notes: str = ""


class AnnotationResponse(BaseModel):
    span_id: str
    label: str
    score: float
    annotation_type: str


def annotation_score(kind: str, value: float) -> float:
    """The 0-1 score of a thumbs (0 or 1), star (1-5) or relevance (0-1)
    rating."""
    if kind == "thumbs":
        if value not in (0, 1):
            raise ValueError("A thumbs rating is 1 (good) or 0 (bad).")
        return float(value)
    if kind == "stars":
        if value != int(value) or not 1 <= value <= 5:
            raise ValueError("A star rating is a whole number from 1 to 5.")
        return value / 5.0
    if not 0 <= value <= 1:
        raise ValueError("A relevance score is between 0 and 1.")
    return float(value)


def annotation_label(score: float) -> str:
    """``positive`` from 0.6, ``negative`` up to 0.4, ``neutral`` between."""
    if score >= 0.6:
        return "positive"
    if score <= 0.4:
        return "negative"
    return "neutral"


@router.post(
    "/{tenant_id}/search-annotations/{span_id}", response_model=AnnotationResponse
)
async def annotate_search(tenant_id: str, span_id: str, body: AnnotationRequest):
    """Record a reviewer's rating of one recorded search as its
    ``search_quality_annotation``, replacing the one it carried."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    try:
        score = annotation_score(body.kind, body.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    provider = _provider(tenant_id)
    project = _project(tenant_id)
    found = await _bounded(
        provider.traces.get_spans(
            project=project, filters={"span_id": [span_id]}, limit=1
        ),
        f"search {span_id}",
        tenant_id,
    )
    if found.empty or SEARCH_SPAN_NAME not in set(found["name"]):
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} has no recorded search {span_id}.",
        )
    label = annotation_label(score)
    metadata = {
        "label": label,
        "score": score,
        "explanation": body.notes.strip() or "User annotation",
        "annotation_type": body.kind,
        "annotator": "human",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await _bounded(
        provider.annotations.add_annotation(
            span_id=span_id,
            name=SEARCH_ANNOTATION_NAME,
            label=label,
            score=score,
            metadata=metadata,
            project=project,
        ),
        f"the annotation of search {span_id}",
        tenant_id,
    )
    return AnnotationResponse(
        span_id=span_id, label=label, score=score, annotation_type=body.kind
    )


class AnnotationCount(BaseModel):
    lookback_days: int
    annotated_searches: int


@router.get("/{tenant_id}/search-annotations/count", response_model=AnnotationCount)
async def annotation_count(tenant_id: str, lookback_days: int = Query(30, ge=1, le=90)):
    """How many of the tenant's recorded searches in the window carry a
    search-quality annotation."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    end = datetime.now(timezone.utc)
    spans = await _spans(
        tenant_id,
        end - timedelta(days=lookback_days),
        end,
        "the recorded searches",
        name=SEARCH_SPAN_NAME,
    )
    annotations = _annotations_by_span(
        await _search_annotations(tenant_id, spans, "the search-quality annotations")
    )
    return AnnotationCount(
        lookback_days=lookback_days, annotated_searches=len(annotations)
    )


# ------------------------------------------------------------ golden dataset


class GoldenEntry(BaseModel):
    expected_videos: List[str]
    relevance_scores: Dict[str, float]
    avg_relevance: float
    profile: str
    timestamp: str


class GoldenDatasetRequest(BaseModel):
    min_rating: float = Field(0.8, ge=0, le=1)
    lookback_days: int = Field(30, ge=1, le=90)


class GoldenDataset(BaseModel):
    dataset: Dict[str, GoldenEntry]
    untitled_results: int


def build_golden_dataset(
    searches: pd.DataFrame, annotations: pd.DataFrame, min_rating: float
) -> tuple[Dict[str, Dict[str, Any]], int]:
    """Golden entries from annotated searches, and how many results were left
    out for carrying no source title.

    A search whose search-quality scores average ``min_rating`` or higher
    contributes its query and its top results, keyed by
    ``result_source_title_key``; each source's relevance is its reciprocal
    rank. A search left with no titled result contributes nothing.
    """
    if searches.empty or annotations is None or annotations.empty:
        return {}, 0
    ratings = annotations["result.score"].groupby(level=0).mean()
    if "start_time" in searches.columns:
        searches = searches.sort_values("start_time")
    dataset: Dict[str, Dict[str, Any]] = {}
    untitled = 0
    for span in searches.to_dict("records"):
        rating = ratings.get(str(span["context.span_id"]))
        if rating is None or pd.isna(rating) or float(rating) < min_rating:
            continue
        io = read_span_io(span)
        query = str(io["input"] or "").strip()
        rows = io["output"] if isinstance(io["output"], list) else []
        expected: List[str] = []
        for row in rows[:TOP_RESULTS]:
            try:
                key = result_source_title_key(row)
            except ValueError:
                untitled += 1
                continue
            if key not in expected:
                expected.append(key)
        if not query or not expected:
            continue
        dataset[query] = {
            "expected_videos": expected,
            "relevance_scores": {
                source: 1.0 / (rank + 1) for rank, source in enumerate(expected)
            },
            "avg_relevance": float(rating),
            "profile": _text(span.get("attributes.profile")) or "unknown",
            "timestamp": _text(span.get("start_time")) or "",
        }
    return dataset, untitled


@router.post("/{tenant_id}/golden-dataset", response_model=GoldenDataset)
async def golden_dataset(tenant_id: str, body: GoldenDatasetRequest):
    """A golden dataset from the tenant's annotated searches: query to the
    sources its well-rated search found."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    end = datetime.now(timezone.utc)
    spans = await _spans(
        tenant_id,
        end - timedelta(days=body.lookback_days),
        end,
        "the recorded searches",
        name=SEARCH_SPAN_NAME,
    )
    annotations = await _search_annotations(
        tenant_id, spans, "the search-quality annotations"
    )
    dataset, untitled = build_golden_dataset(spans, annotations, body.min_rating)
    return GoldenDataset(dataset=dataset, untitled_results=untitled)


# ------------------------------------------------------------ synthetic data


class SyntheticOptimizer(BaseModel):
    name: str
    description: str
    schema_name: str
    agent_type: str
    backend_query_strategy: str


class SyntheticSettings(BaseModel):
    confidence_threshold: float
    sampling_strategies: List[str]
    optimizers: List[SyntheticOptimizer]


@router.get("/{tenant_id}/synthetic/settings", response_model=SyntheticSettings)
async def synthetic_settings(tenant_id: str):
    """The auto-approval threshold a synthetic run applies, the sampling
    strategies it accepts and the optimizers it generates for."""
    optimizers = []
    for name in sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER):
        config = get_optimizer_config(name)
        optimizers.append(
            SyntheticOptimizer(
                name=name,
                description=config.description,
                schema_name=config.schema_class.__name__,
                agent_type=APPROVED_TRAINING_AGENT_BY_OPTIMIZER[name],
                backend_query_strategy=config.backend_query_strategy,
            )
        )
    return SyntheticSettings(
        confidence_threshold=ApprovalConfig().confidence_threshold,
        sampling_strategies=sorted(SAMPLING_STRATEGIES),
        optimizers=optimizers,
    )


class GeneratedItem(BaseModel):
    item_id: str
    status: str
    confidence: float
    query: Optional[str]
    reasoning: str
    entities: List[Any]
    schema_name: Optional[str]
    retry_count: Optional[int]
    generation_metadata: Dict[str, Any]
    data: Dict[str, Any]


class OptimizerOutcome(BaseModel):
    optimizer: str
    status: str
    error: Optional[str] = None
    batch_id: Optional[str] = None
    schema_name: Optional[str] = None
    selected_profiles: List[str] = []
    profile_selection_reasoning: Optional[str] = None
    generation_time_ms: Optional[float] = None
    examples_generated: int = 0
    auto_approved: int = 0
    pending_review: int = 0
    avg_confidence: Optional[float] = None
    items: List[GeneratedItem] = []


class SyntheticRunResults(BaseModel):
    workflow_name: str
    phase: Optional[str]
    settled: bool
    status: Optional[str]
    parameters: Dict[str, Any]
    outcomes: List[OptimizerOutcome]


def _run_optimizer_result(data: Dict[str, Any]) -> Optional[str]:
    """The stdout Argo captured from the run's optimizer pod, or ``None``
    while it has none."""
    nodes = [
        node
        for node in ((data.get("status") or {}).get("nodes") or {}).values()
        if isinstance(node, dict)
        and node.get("type") == "Pod"
        and node.get("templateName") == "run-optimizer"
    ]
    if len(nodes) != 1:
        return None
    return (nodes[0].get("outputs") or {}).get("result")


def _generated_item(item: ReviewItem) -> GeneratedItem:
    data = dict(item.data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    generation = metadata.get("_generation_metadata") or {}
    try:
        schema_name, _ = correction_template(data)
        reasoning = review_reasoning(data)
    except ValueError:
        schema_name, reasoning = None, ""
    retries = generation.get("retry_count")
    entities = data.get("entities")
    return GeneratedItem(
        item_id=item.item_id,
        status=item.status.value,
        confidence=item.confidence,
        query=_text(data.get("query")),
        reasoning=reasoning,
        entities=entities if isinstance(entities, list) else [],
        schema_name=schema_name,
        retry_count=int(retries) if retries is not None else None,
        generation_metadata=generation if isinstance(generation, dict) else {},
        data=data,
    )


async def _approval_batch(tenant_id: str, batch_id: str) -> ApprovalBatch:
    from cogniverse_runtime.routers import approvals

    try:
        storage = await asyncio.to_thread(
            ApprovalStorageImpl.from_system_config,
            approvals._require_config_manager(),
            get_telemetry_manager(),
            tenant_id,
        )
        batch = await storage.get_batch(batch_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise failure_response(
            502,
            "approval_store_unavailable",
            f"Could not read synthetic batch {batch_id} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    if batch is None:
        raise HTTPException(
            status_code=502,
            detail=(
                f"The run reported synthetic batch {batch_id}, which the "
                f"approval store of tenant {tenant_id} does not hold."
            ),
        )
    return batch


@router.get(
    "/{tenant_id}/optimize/runs/{workflow_name}/synthetic",
    response_model=SyntheticRunResults,
)
async def synthetic_run_results(tenant_id: str, workflow_name: str):
    """What a synthetic run generated: per optimizer, its outcome, the
    profiles it sampled and why, and the examples with their review state.

    Until the run settles there is nothing to read; a settled run's outcome
    is the document its optimizer pod printed.
    """
    data = await tenant_router._argo_get_workflow_data(workflow_name, tenant_id)
    labels = (data.get("metadata") or {}).get("labels") or {}
    if labels.get("cogniverse.ai/mode") != tenant_router._SYNTHETIC_MODE:
        raise HTTPException(
            status_code=400,
            detail=f"Run {workflow_name} is not a synthetic run.",
        )
    tenant_id = canonical_tenant_or_400(tenant_id)
    phase = (data.get("status") or {}).get("phase")
    try:
        options = json.loads(tenant_router._workflow_parameter(data, "options") or "{}")
    except ValueError as exc:
        raise failure_response(
            502,
            "synthetic_options_unreadable",
            f"Run {workflow_name} carries options that are not JSON.",
            exc,
            workflow=workflow_name,
        ) from exc
    parameters = {
        "optimizers": [
            name
            for name in (tenant_router._workflow_parameter(data, "agents") or "").split(
                ","
            )
            if name
        ],
        **options,
    }
    settled = phase in tenant_router._FINISHED_WORKFLOW_PHASES
    if not settled:
        return SyntheticRunResults(
            workflow_name=workflow_name,
            phase=phase,
            settled=False,
            status=None,
            parameters=parameters,
            outcomes=[],
        )
    printed = _run_optimizer_result(data)
    if printed is None:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Run {workflow_name} ended {phase} without printing its "
                "outcome; its pod log says why."
            ),
        )
    try:
        document = json.loads(printed)
        results: Dict[str, Dict[str, Any]] = document["results"]
    except (ValueError, KeyError, TypeError) as exc:
        raise failure_response(
            502,
            "synthetic_result_unreadable",
            f"Run {workflow_name} printed an outcome that is not a synthetic "
            "run result.",
            exc,
            workflow=workflow_name,
        ) from exc
    outcomes = []
    for optimizer in sorted(results):
        result = results[optimizer]
        outcome = OptimizerOutcome(
            optimizer=optimizer,
            status=str(result.get("status")),
            error=result.get("error"),
            batch_id=result.get("batch_id"),
            schema_name=result.get("schema_name"),
            selected_profiles=list(result.get("selected_profiles") or []),
            profile_selection_reasoning=result.get("profile_selection_reasoning"),
            generation_time_ms=_number(result.get("generation_time_ms")),
            examples_generated=int(result.get("examples_generated") or 0),
            auto_approved=int(result.get("auto_approved") or 0),
            pending_review=int(result.get("pending_review") or 0),
            avg_confidence=_number(result.get("avg_confidence")),
        )
        if outcome.batch_id:
            batch = await _approval_batch(tenant_id, outcome.batch_id)
            outcome.items = [_generated_item(item) for item in batch.items]
        outcomes.append(outcome)
    return SyntheticRunResults(
        workflow_name=workflow_name,
        phase=phase,
        settled=True,
        status=document.get("status"),
        parameters=parameters,
        outcomes=outcomes,
    )


# ------------------------------------------------------------------- datasets


class TrainingDataset(BaseModel):
    name: str
    examples: Optional[int]
    created_at: Optional[str]
    description: Optional[str]


class TrainingDatasets(BaseModel):
    datasets: List[TrainingDataset]


@router.get("/{tenant_id}/datasets", response_model=TrainingDatasets)
async def training_datasets(tenant_id: str):
    """The tenant's telemetry datasets, newest first."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    suffix = tenant_dataset_name(tenant_id, "")
    listed = await _bounded(
        _provider(tenant_id).datasets.list_datasets(),
        "the datasets",
        tenant_id,
    )
    owned = [entry for entry in listed if str(entry["name"]).endswith(suffix)]
    owned.sort(key=lambda entry: str(entry.get("created_at") or ""), reverse=True)
    return TrainingDatasets(
        datasets=[
            TrainingDataset(
                name=entry["name"],
                examples=entry.get("example_count"),
                created_at=_text(entry.get("created_at")),
                description=entry.get("description") or None,
            )
            for entry in owned
        ]
    )


class CreatedDataset(BaseModel):
    name: str
    dataset_id: str
    examples: int


@router.post("/{tenant_id}/datasets", response_model=CreatedDataset)
async def upload_dataset(
    tenant_id: str, name: str = Form(..., min_length=1), file: UploadFile = File(...)
):
    """Create a telemetry dataset of the tenant from a CSV of ``query``,
    ``expected_videos`` (comma-separated) and optional ``category`` columns."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    stored_name = tenant_dataset_name(tenant_id, name.strip())
    content = await file.read()

    def create() -> tuple[str, int]:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "upload.csv"
            path.write_bytes(content)
            rows = len(pd.read_csv(path))
            manager = DatasetManager(
                tenant_id, dataset_store=_provider(tenant_id).datasets
            )
            return (
                manager.create_from_csv(
                    csv_path=str(path),
                    dataset_name=stored_name,
                    description="Uploaded from the web client's optimization view",
                ),
                rows,
            )

    try:
        dataset_id, rows = await asyncio.to_thread(create)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(
            status_code=400, detail=f"The CSV cannot become a dataset: {exc}"
        ) from exc
    except Exception as exc:
        raise failure_response(
            502,
            "dataset_store_unavailable",
            f"Could not create dataset {stored_name} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    return CreatedDataset(name=stored_name, dataset_id=str(dataset_id), examples=rows)


# ------------------------------------------------------- profile recommender


class ColumnStatistics(BaseModel):
    column: str
    statistics: Dict[str, Optional[float]]


class ProfileQuality(BaseModel):
    profile_column: str
    quality_column: str
    rows: List[Dict[str, Any]]


class ProfileSpanAnalysis(BaseModel):
    lookback_days: int
    search_spans: int
    columns: List[str]
    profile_usage: Dict[str, Dict[str, int]]
    quality: List[ColumnStatistics]
    profile_quality: List[ProfileQuality]


def _quality_columns(columns: List[str]) -> List[str]:
    return [
        column
        for column in columns
        if any(metric in column.lower() for metric in ("ndcg", "score", "quality"))
    ]


def analyze_profile_spans(search_spans: pd.DataFrame) -> Dict[str, Any]:
    """Profile usage, quality-metric statistics and per-profile quality of a
    frame of search spans."""
    columns = [str(column) for column in search_spans.columns]
    profile_columns = [column for column in columns if "profile" in column.lower()]
    quality_columns = [
        column
        for column in _quality_columns(columns)
        if pd.api.types.is_numeric_dtype(search_spans[column])
    ]
    usage = {
        column: dict(
            sorted(
                (
                    (str(value), int(count))
                    for value, count in search_spans[column].value_counts().items()
                ),
                key=lambda entry: (-entry[1], entry[0]),
            )
        )
        for column in profile_columns
    }
    quality = [
        ColumnStatistics(
            column=column,
            statistics={
                key: _number(value)
                for key, value in search_spans[column].describe().to_dict().items()
            },
        )
        for column in quality_columns
    ]
    profile_quality = []
    for profile_column in profile_columns:
        for quality_column in quality_columns:
            grouped = search_spans.groupby(profile_column)[quality_column].agg(
                ["mean", "count"]
            )
            profile_quality.append(
                ProfileQuality(
                    profile_column=profile_column,
                    quality_column=quality_column,
                    rows=[
                        {
                            "profile": str(profile),
                            "mean": _number(row["mean"]),
                            "count": int(row["count"]),
                        }
                        for profile, row in grouped.iterrows()
                    ],
                )
            )
    return {
        "search_spans": len(search_spans),
        "columns": columns,
        "profile_usage": usage,
        "quality": quality,
        "profile_quality": profile_quality,
    }


@router.get(
    "/{tenant_id}/profile-selection/analysis", response_model=ProfileSpanAnalysis
)
async def profile_span_analysis(
    tenant_id: str, lookback_days: int = Query(30, ge=1, le=90)
):
    """How the tenant's search spans in the window used each profile and
    scored, as the recommender's training data sees them."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    end = datetime.now(timezone.utc)
    spans = await _spans(
        tenant_id, end - timedelta(days=lookback_days), end, "the search spans"
    )
    searches = _named_frame(spans, "search")
    return ProfileSpanAnalysis(
        lookback_days=lookback_days, **analyze_profile_spans(searches)
    )


class TrainRequest(BaseModel):
    lookback_days: int = Field(30, ge=1, le=90)


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class TrainedRecommender(BaseModel):
    train_accuracy: float
    test_accuracy: float
    samples: int
    features: int
    profiles: List[str]
    feature_importance: List[FeatureImportance]


class RecommenderState(BaseModel):
    trained: bool
    profiles: List[str]


class PredictRequest(BaseModel):
    query: str = Field(min_length=1)


class Prediction(BaseModel):
    profile: str
    confidence: float
    features: Dict[str, float]


def _artifacts(tenant_id: str) -> ArtifactManager:
    return ArtifactManager(_provider(tenant_id), tenant_id)


@router.post("/{tenant_id}/profile-selection/train", response_model=TrainedRecommender)
async def train_recommender(tenant_id: str, body: TrainRequest):
    """Train the XGBoost profile recommender on the tenant's search and
    evaluation spans in the window and store it as the tenant's model."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    end = datetime.now(timezone.utc)
    optimizer = ProfilePerformanceOptimizer(
        model_dir=Path(tempfile.gettempdir()) / "cogniverse-profile-performance"
    )
    try:
        features, labels, profiles = await optimizer.extract_training_data_from_phoenix(
            tenant_id=tenant_id,
            project_name=_project(tenant_id),
            start_time=end - timedelta(days=body.lookback_days),
            end_time=end,
            min_samples=PROFILE_TRAINING_MIN_SAMPLES,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Cannot train the recommender: {exc}"
        ) from exc
    except Exception as exc:
        raise failure_response(
            502,
            "telemetry_unavailable",
            f"Could not read the training spans of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    try:
        metrics = await asyncio.to_thread(optimizer.train, features, labels, 0.2)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Cannot train the recommender: {exc}"
        ) from exc
    try:
        await _artifacts(tenant_id).save_blob(*PROFILE_MODEL_BLOB, optimizer.to_blob())
    except Exception as exc:
        raise failure_response(
            502,
            "artifact_store_unavailable",
            f"The recommender of tenant {tenant_id} trained but was not stored.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    importance = sorted(
        (
            FeatureImportance(feature=name, importance=float(value))
            for name, value in zip(
                FEATURE_NAMES, optimizer.model.feature_importances_, strict=True
            )
        ),
        key=lambda entry: entry.importance,
        reverse=True,
    )
    return TrainedRecommender(
        train_accuracy=float(metrics["train_accuracy"]),
        test_accuracy=float(metrics["test_accuracy"]),
        samples=int(metrics["n_samples"]),
        features=int(metrics["n_features"]),
        profiles=list(profiles),
        feature_importance=importance,
    )


async def _stored_recommender(tenant_id: str) -> Optional[ProfilePerformanceOptimizer]:
    try:
        blob = await _artifacts(tenant_id).load_blob(*PROFILE_MODEL_BLOB)
    except Exception as exc:
        raise failure_response(
            502,
            "artifact_store_unavailable",
            f"Could not read the recommender of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    if blob is None:
        return None
    optimizer = ProfilePerformanceOptimizer(
        model_dir=Path(tempfile.gettempdir()) / "cogniverse-profile-performance"
    )
    optimizer.load_blob(blob)
    return optimizer


@router.get("/{tenant_id}/profile-selection/model", response_model=RecommenderState)
async def recommender_state(tenant_id: str):
    """Whether the tenant has a trained recommender, and its profiles."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    optimizer = await _stored_recommender(tenant_id)
    if optimizer is None:
        return RecommenderState(trained=False, profiles=[])
    return RecommenderState(
        trained=True, profiles=optimizer.label_encoder.classes_.tolist()
    )


@router.post("/{tenant_id}/profile-selection/predict", response_model=Prediction)
async def predict_profile(tenant_id: str, body: PredictRequest):
    """The stored recommender's profile for ``query``, its confidence and
    the features it read from the query."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    optimizer = await _stored_recommender(tenant_id)
    if optimizer is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} has no trained profile recommender.",
        )
    profile, confidence = optimizer.predict_best_profile(body.query)
    features = optimizer.extract_query_features(body.query)
    return Prediction(
        profile=str(profile),
        confidence=confidence,
        features={
            name: float(value)
            for name, value in zip(FEATURE_NAMES, features.to_array(), strict=True)
        },
    )


# -------------------------------------------------------------------- metrics


class AgentScores(BaseModel):
    agent: str
    precision: float
    recall: float
    f1: float


class RoutingScores(BaseModel):
    accuracy: float
    total_decisions: int
    avg_latency_ms: float
    confidence_calibration: float
    per_agent: List[AgentScores]


class EvaluationActivity(BaseModel):
    spans: int
    queries: int


class TrainingDay(BaseModel):
    date: str
    runs: int


class OptimizationMetrics(BaseModel):
    lookback_days: int
    spans: int
    routing: Optional[RoutingScores]
    evaluation: EvaluationActivity
    training: List[TrainingDay]


def _routing_scores(
    evaluator: RoutingEvaluator, routing_spans: List[Dict[str, Any]]
) -> Optional[RoutingScores]:
    if not routing_spans:
        return None
    metrics = evaluator.calculate_metrics(routing_spans)
    return RoutingScores(
        accuracy=metrics.routing_accuracy,
        total_decisions=metrics.total_decisions,
        avg_latency_ms=metrics.avg_routing_latency,
        confidence_calibration=metrics.confidence_calibration,
        per_agent=[
            AgentScores(
                agent=agent,
                precision=metrics.per_agent_precision.get(agent, 0.0),
                recall=metrics.per_agent_recall.get(agent, 0.0),
                f1=metrics.per_agent_f1.get(agent, 0.0),
            )
            for agent in sorted(metrics.per_agent_precision)
        ],
    )


@router.get("/{tenant_id}/optimization-metrics", response_model=OptimizationMetrics)
async def optimization_metrics(
    tenant_id: str, lookback_days: int = Query(7, ge=1, le=90)
):
    """Routing accuracy with per-agent precision, recall and F1, evaluation
    activity, and optimization runs per day over the window."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    spans = await _spans(tenant_id, start, end, "the spans")
    evaluator = RoutingEvaluator(_provider(tenant_id), project_name=_project(tenant_id))
    routing_spans = await _bounded(
        evaluator.query_routing_spans(start_time=start, end_time=end, limit=1000),
        "the routing spans",
        tenant_id,
    )
    evaluations = _named_frame(spans, EVALUATION_SPAN_PATTERN)
    queries = (
        {
            str(io["input"])
            for io in (read_span_io(span) for span in evaluations.to_dict("records"))
            if io["input"]
        }
        if not evaluations.empty
        else set()
    )
    training = _named_frame(spans, TRAINING_SPAN_PATTERN)
    days: List[TrainingDay] = []
    if not training.empty and "start_time" in training.columns:
        per_day = (
            pd.to_datetime(training["start_time"], utc=True)
            .dt.date.value_counts()
            .sort_index()
        )
        days = [
            TrainingDay(date=day.isoformat(), runs=int(count))
            for day, count in per_day.items()
        ]
    return OptimizationMetrics(
        lookback_days=lookback_days,
        spans=len(spans),
        routing=_routing_scores(evaluator, routing_spans),
        evaluation=EvaluationActivity(spans=len(evaluations), queries=len(queries)),
        training=days,
    )
