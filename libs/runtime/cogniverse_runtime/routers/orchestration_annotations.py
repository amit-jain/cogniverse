"""Human review of a tenant's orchestration workflows.

The orchestrator records each workflow as a ``cogniverse.orchestration`` span
in the tenant's telemetry project: the query in ``input.value`` and the
workflow (pattern, agent sequence, execution order, timing, outcome) in
``output.value``. A reviewer's verdict is stored as the span's
``orchestration_quality`` annotation, which the optimization feedback path
reads as ground truth.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from cogniverse_agents.routing.annotation_storage import _meta_get
from cogniverse_agents.routing.orchestration_annotation_storage import (
    ORCHESTRATION_ANNOTATION_NAME,
    OrchestrationAnnotation,
    OrchestrationAnnotationStorage,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION
from cogniverse_foundation.telemetry.span_contract import read_span_io
from cogniverse_runtime.http_errors import failure_response

logger = logging.getLogger(__name__)

router = APIRouter()

QualityLabel = Literal["failed", "poor", "acceptable", "good", "excellent"]
Pattern = Literal["parallel", "sequential", "conditional", "mixed"]


class WorkflowReview(BaseModel):
    annotator: str
    label: str
    score: float
    annotation_source: str
    pattern_is_optimal: bool
    agents_are_correct: bool
    execution_order_is_optimal: bool
    improvement_notes: Optional[str]


class Workflow(BaseModel):
    span_id: str
    start_time: str
    query: str
    workflow_id: str
    pattern: str
    agent_sequence: List[str]
    execution_order: List[str]
    execution_time: float
    tasks_completed: int
    success: bool
    error_summary: Optional[str]
    review: Optional[WorkflowReview]


class Workflows(BaseModel):
    workflows: List[Workflow]


class AnnotationRequest(BaseModel):
    # The span's start time as the list served it; the span is read back
    # from that instant rather than trusted from the request.
    start_time: datetime
    annotator: str = Field(min_length=1)
    quality_label: QualityLabel
    quality_score: float = Field(ge=0.0, le=1.0)
    pattern_is_optimal: bool
    suggested_pattern: Optional[Pattern] = None
    pattern_feedback: Optional[str] = None
    agents_are_correct: bool
    missing_agents: List[str] = []
    unnecessary_agents: List[str] = []
    execution_order_is_optimal: bool
    suggested_execution_order: Optional[List[str]] = None
    execution_order_feedback: Optional[str] = None
    what_went_well: Optional[str] = None
    what_went_wrong: Optional[str] = None
    improvement_notes: Optional[str] = None


def _telemetry_failed(exc: Exception, tenant_id: str, message: str) -> HTTPException:
    return failure_response(
        502, "telemetry_unavailable", message, exc, tenant_id=tenant_id
    )


def _workflow(span_row: pd.Series, review: Optional[WorkflowReview]) -> Workflow:
    io = read_span_io(span_row)
    output = io["output"] if isinstance(io["output"], dict) else {}
    return Workflow(
        span_id=span_row["context.span_id"],
        start_time=span_row["start_time"].isoformat(),
        query=io["input"] or "",
        workflow_id=str(output.get("workflow_id", "")),
        pattern=str(output.get("pattern", "")),
        agent_sequence=list(output.get("agent_sequence") or []),
        execution_order=list(output.get("execution_order") or []),
        execution_time=float(output.get("execution_time", 0.0)),
        tasks_completed=int(output.get("tasks_completed", 0)),
        success=bool(output.get("success", False)),
        error_summary=output.get("error_summary"),
        review=review,
    )


def _review(annotation: pd.Series) -> WorkflowReview:
    def flag(key: str) -> bool:
        return bool(_meta_get(annotation, key, False))

    return WorkflowReview(
        annotator=str(_meta_get(annotation, "annotator_id", "")),
        label=str(annotation.get("result.label")),
        score=float(annotation.get("result.score")),
        annotation_source=str(_meta_get(annotation, "annotation_source", "")),
        pattern_is_optimal=flag("pattern_is_optimal"),
        agents_are_correct=flag("agents_are_correct"),
        execution_order_is_optimal=flag("execution_order_is_optimal"),
        improvement_notes=_meta_get(annotation, "improvement_notes"),
    )


async def _reviews(
    storage: OrchestrationAnnotationStorage, spans: pd.DataFrame
) -> Dict[str, WorkflowReview]:
    """The latest ``orchestration_quality`` annotation of each span."""
    annotations = await storage.provider.annotations.get_annotations(
        spans_df=spans,
        project=storage.project_name,
        annotation_names=[ORCHESTRATION_ANNOTATION_NAME],
    )
    if annotations is None or annotations.empty:
        return {}
    if "updated_at" in annotations.columns:
        annotations = annotations.sort_values("updated_at")
    return {
        span_id: _review(row)
        for span_id, row in annotations.iterrows()  # the latest wins
    }


@router.get("/{tenant_id}/orchestration-workflows", response_model=Workflows)
async def list_workflows(
    tenant_id: str,
    lookback_hours: int = Query(24, ge=1, le=24 * 30),
    limit: int = Query(50, ge=1, le=500),
):
    """The tenant's orchestration workflows of the last ``lookback_hours``,
    newest first, each with its latest review."""
    tenant_id = canonical_tenant_id(tenant_id)
    storage = OrchestrationAnnotationStorage(tenant_id=tenant_id)
    end = datetime.now(timezone.utc)
    try:
        spans = await storage.provider.traces.get_all_spans(
            project=storage.project_name,
            start_time=end - timedelta(hours=lookback_hours),
            end_time=end,
            filters={"name": SPAN_NAME_ORCHESTRATION},
        )
        if spans.empty:
            return Workflows(workflows=[])
        spans = spans.sort_values("start_time", ascending=False).head(limit)
        reviews = await _reviews(storage, spans)
    except Exception as exc:
        raise _telemetry_failed(
            exc,
            tenant_id,
            f"Could not read the orchestration workflows of tenant {tenant_id}.",
        ) from exc
    return Workflows(
        workflows=[
            _workflow(row, reviews.get(row["context.span_id"]))
            for _, row in spans.iterrows()
        ]
    )


@router.post(
    "/{tenant_id}/orchestration-workflows/{span_id}/annotation",
    response_model=Workflow,
)
async def annotate_workflow(tenant_id: str, span_id: str, request: AnnotationRequest):
    """Store a reviewer's verdict on one workflow as its
    ``orchestration_quality`` annotation; answers the workflow with it."""
    tenant_id = canonical_tenant_id(tenant_id)
    if request.start_time.utcoffset() is None:
        raise HTTPException(
            status_code=422, detail="start_time must include a timezone."
        )
    storage = OrchestrationAnnotationStorage(tenant_id=tenant_id)
    try:
        spans = await storage.provider.traces.get_all_spans(
            project=storage.project_name,
            start_time=request.start_time - timedelta(seconds=1),
            end_time=request.start_time + timedelta(seconds=1),
            filters={"name": SPAN_NAME_ORCHESTRATION},
        )
    except Exception as exc:
        raise _telemetry_failed(
            exc, tenant_id, f"Could not read workflow span {span_id}."
        ) from exc
    matches = spans[spans["context.span_id"] == span_id] if not spans.empty else spans
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No orchestration workflow span {span_id} started at "
                f"{request.start_time.isoformat()} for tenant {tenant_id}."
            ),
        )
    workflow = _workflow(matches.iloc[0], None)
    suggested_agents = [
        agent
        for agent in [*workflow.agent_sequence, *request.missing_agents]
        if agent not in request.unnecessary_agents
    ]
    annotation = OrchestrationAnnotation(
        workflow_id=workflow.workflow_id,
        span_id=span_id,
        query=workflow.query,
        orchestration_pattern=workflow.pattern,
        agents_used=workflow.agent_sequence,
        execution_order=workflow.execution_order,
        execution_time=workflow.execution_time,
        pattern_is_optimal=request.pattern_is_optimal,
        suggested_pattern=request.suggested_pattern,
        pattern_feedback=request.pattern_feedback,
        agents_are_correct=request.agents_are_correct,
        missing_agents=request.missing_agents,
        unnecessary_agents=request.unnecessary_agents,
        suggested_agents=suggested_agents,
        execution_order_is_optimal=request.execution_order_is_optimal,
        suggested_execution_order=request.suggested_execution_order,
        execution_order_feedback=request.execution_order_feedback,
        workflow_quality_label=request.quality_label,
        quality_score=request.quality_score,
        improvement_notes=request.improvement_notes,
        what_went_well=request.what_went_well,
        what_went_wrong=request.what_went_wrong,
        annotator_id=request.annotator,
        annotation_timestamp=datetime.now(timezone.utc),
        workflow_succeeded=workflow.success,
        error_details=workflow.error_summary,
    )
    try:
        await storage.store_annotation(annotation)
    except Exception as exc:
        raise failure_response(
            502,
            "annotation_not_stored",
            f"The review of workflow span {span_id} was not stored.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    logger.info(
        "Orchestration review of %s for tenant %s: %s by %s",
        span_id,
        tenant_id,
        request.quality_label,
        request.annotator,
    )
    workflow.review = WorkflowReview(
        annotator=request.annotator,
        label=request.quality_label,
        score=request.quality_score,
        annotation_source="human",
        pattern_is_optimal=request.pattern_is_optimal,
        agents_are_correct=request.agents_are_correct,
        execution_order_is_optimal=request.execution_order_is_optimal,
        improvement_notes=request.improvement_notes,
    )
    return workflow
