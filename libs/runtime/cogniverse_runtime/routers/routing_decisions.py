"""A tenant's routing decisions, their quality, and their review.

The gateway records each decision as a ``cogniverse.routing`` span in the
tenant's telemetry project. A decision's label (from the LLM annotator or a
reviewer) is the span's ``routing_annotation``, which the optimization
feedback path reads as ground truth.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from cogniverse_agents.routing.annotation_storage import (
    AnnotationStorage,
    LLMAnnotationNotFoundError,
    NotAnLLMAnnotationError,
)
from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_evaluation.evaluators.routing_evaluator import (
    summarize_routing_decisions,
)
from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
from cogniverse_runtime.http_errors import failure_response

logger = logging.getLogger(__name__)

router = APIRouter()

ReviewLabel = Literal["correct", "wrong", "ambiguous", "insufficient_info"]


class DecisionLabel(BaseModel):
    label: str
    confidence: Optional[float]
    reasoning: Optional[str]
    suggested_agent: Optional[str]
    annotator: Optional[str]
    human_reviewed: bool
    requires_review: bool
    approved_by: Optional[str]


class RoutingDecision(BaseModel):
    span_id: Optional[str]
    trace_id: Optional[str]
    start_time: str
    query: Optional[str]
    chosen_agent: str
    confidence: float
    outcome: str
    reason: str
    latency_ms: float
    entity_extraction_failed: bool
    label: Optional[DecisionLabel]


class AgentRouting(BaseModel):
    agent: str
    decisions: int
    successes: int
    failures: int
    ambiguous: int
    success_rate: float
    mean_confidence: float
    mean_latency_ms: float


class RoutingLatency(BaseModel):
    mean: Optional[float]
    p50: Optional[float]
    p95: Optional[float]


class RoutingDecisions(BaseModel):
    total: int
    successes: int
    failures: int
    ambiguous: int
    unreadable: int
    accuracy: Optional[float]
    confidence_calibration: Optional[float]
    latency_ms: RoutingLatency
    per_agent: List[AgentRouting]
    decisions: List[RoutingDecision]


class DecisionRef(BaseModel):
    # The span's start time as the list served it; the span is read back
    # from that instant rather than trusted from the request.
    start_time: datetime
    reviewer: str = Field(min_length=1)


class LabelRequest(DecisionRef):
    label: ReviewLabel
    reasoning: str = ""
    suggested_agent: Optional[str] = None


def _value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _label(label, score, metadata) -> DecisionLabel:
    metadata = metadata if isinstance(metadata, dict) else {}
    score = _value(score)
    return DecisionLabel(
        label=str(label),
        confidence=None if score is None else float(score),
        reasoning=metadata.get("reasoning"),
        suggested_agent=metadata.get("suggested_agent"),
        annotator=metadata.get("annotator"),
        human_reviewed=metadata.get("human_reviewed") is True,
        requires_review=metadata.get("requires_review") is True,
        approved_by=metadata.get("approved_by"),
    )


async def _labels(
    storage: AnnotationStorage, spans: pd.DataFrame
) -> Dict[str, DecisionLabel]:
    """The latest ``routing_annotation`` of each span."""
    if spans.empty:
        return {}
    annotations = await storage.provider.annotations.get_annotations(
        spans_df=spans,
        project=storage.project_name,
        annotation_names=[storage.annotation_name],
    )
    if annotations is None or annotations.empty:
        return {}
    if "updated_at" in annotations.columns:
        annotations = annotations.sort_values("updated_at")
    return {
        span_id: _label(
            row.get("result.label"), row.get("result.score"), row.get("metadata")
        )
        for span_id, row in annotations.iterrows()
    }


def _decisions(spans: pd.DataFrame, labels: Dict[str, DecisionLabel]) -> dict:
    summary = summarize_routing_decisions(spans)
    summary["decisions"] = [
        RoutingDecision(**decision, label=labels.get(decision["span_id"]))
        for decision in summary["decisions"]
    ]
    return summary


@router.get("/{tenant_id}/routing-decisions", response_model=RoutingDecisions)
async def list_routing_decisions(
    tenant_id: str, lookback_hours: int = Query(24, ge=1, le=24 * 30)
):
    """The tenant's routing decisions in the last ``lookback_hours``, newest
    first with their labels, and their outcome counts, accuracy, confidence
    calibration, latency and per-agent figures."""
    tenant_id = canonical_tenant_id(tenant_id)
    storage = AnnotationStorage(tenant_id=tenant_id)
    end = datetime.now(timezone.utc)
    try:
        spans = await storage.provider.traces.get_all_spans(
            project=storage.project_name,
            start_time=end - timedelta(hours=lookback_hours),
            end_time=end,
            filters={"name": SPAN_NAME_ROUTING},
        )
        labels = await _labels(storage, spans)
    except Exception as exc:
        raise failure_response(
            502,
            "telemetry_unavailable",
            f"Could not read the routing decisions of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    return RoutingDecisions(**_decisions(spans, labels))


async def _read_decision(
    storage: AnnotationStorage, tenant_id: str, span_id: str, ref: DecisionRef
) -> RoutingDecision:
    """The routing decision ``span_id`` of the tenant, read back from its
    start time, with its current label; 404 when the tenant has no such
    decision then."""
    if ref.start_time.utcoffset() is None:
        raise HTTPException(
            status_code=422, detail="start_time must include a timezone."
        )
    try:
        spans = await storage.provider.traces.get_all_spans(
            project=storage.project_name,
            start_time=ref.start_time - timedelta(seconds=1),
            end_time=ref.start_time + timedelta(seconds=1),
            filters={"name": SPAN_NAME_ROUTING},
        )
        if not spans.empty:
            spans = spans[spans["context.span_id"] == span_id]
        labels = await _labels(storage, spans)
    except Exception as exc:
        raise failure_response(
            502,
            "telemetry_unavailable",
            f"Could not read routing decision {span_id} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    decisions = _decisions(spans, labels)["decisions"]
    if not decisions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No routing decision {span_id} started at "
                f"{ref.start_time.isoformat()} for tenant {tenant_id}."
            ),
        )
    return decisions[0]


def _not_stored(exc: Exception, tenant_id: str, span_id: str) -> HTTPException:
    return failure_response(
        502,
        "annotation_not_stored",
        f"The label of routing decision {span_id} was not stored.",
        exc,
        tenant_id=tenant_id,
    )


@router.post(
    "/{tenant_id}/routing-decisions/{span_id}/approve",
    response_model=RoutingDecision,
)
async def approve_llm_label(tenant_id: str, span_id: str, request: DecisionRef):
    """Approve the LLM annotator's label of one decision as the reviewer's,
    keeping its label and reasoning; answers the decision with it."""
    tenant_id = canonical_tenant_id(tenant_id)
    storage = AnnotationStorage(tenant_id=tenant_id)
    decision = await _read_decision(storage, tenant_id, span_id, request)
    try:
        approved = await storage.approve_llm_annotation(
            span_id, annotator_id=request.reviewer
        )
    except LLMAnnotationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Routing decision {span_id} has no LLM label to approve.",
        ) from exc
    except NotAnLLMAnnotationError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Routing decision {span_id} is labelled by "
                f"{decision.label.annotator if decision.label else 'a reviewer'}, "
                "not the LLM."
            ),
        ) from exc
    except Exception as exc:
        raise _not_stored(exc, tenant_id, span_id) from exc
    return decision.model_copy(
        update={
            "label": _label(approved["label"], approved["score"], approved["metadata"])
        }
    )


@router.put(
    "/{tenant_id}/routing-decisions/{span_id}/label",
    response_model=RoutingDecision,
)
async def label_decision(tenant_id: str, span_id: str, request: LabelRequest):
    """Store the reviewer's label of one decision, replacing any LLM label;
    answers the decision with it."""
    tenant_id = canonical_tenant_id(tenant_id)
    storage = AnnotationStorage(tenant_id=tenant_id)
    decision = await _read_decision(storage, tenant_id, span_id, request)
    suggested_agent = request.suggested_agent or None
    try:
        await storage.store_human_annotation(
            span_id=span_id,
            label=AnnotationLabel(request.label),
            reasoning=request.reasoning,
            suggested_agent=suggested_agent,
            annotator_id=request.reviewer,
        )
    except Exception as exc:
        raise _not_stored(exc, tenant_id, span_id) from exc
    return decision.model_copy(
        update={
            "label": DecisionLabel(
                label=request.label,
                confidence=1.0,
                reasoning=request.reasoning,
                suggested_agent=suggested_agent,
                annotator=request.reviewer,
                human_reviewed=True,
                requires_review=False,
                approved_by=None,
            )
        }
    )
