"""Human review of a tenant's synthetic examples.

Generated examples the confidence extractor did not auto-approve wait in the
tenant's approval store (Phoenix spans and annotations, with Redis electing
each decision). A reviewer approves an item into the tenant's approved
training dataset, or rejects it with feedback and schema corrections, which
regenerates it with the tenant's primary LM for another review. The review
history lists every approved and rejected item with its decision, and a
rejected item that was never regenerated can be regenerated from its
recorded decision.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cogniverse_agents.approval import (
    ApprovalStatus,
    ApprovalStorageImpl,
    HumanApprovalAgent,
    ReviewDecision,
    ReviewDecisionConflictError,
)
from cogniverse_core.approval.interfaces import ApprovalBatch, ReviewItem
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import ApprovalConfig
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_runtime.http_errors import canonical_tenant_or_400, failure_response
from cogniverse_synthetic.approval import (
    SyntheticDataConfidenceExtractor,
    SyntheticDataFeedbackHandler,
)
from cogniverse_synthetic.approval.corrections import (
    CORRECTION_ONLY_SCHEMAS,
    correction_template,
    parse_corrections,
    review_reasoning,
    schema_for_item_data,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# A rejection regenerates with the tenant's LM, retrying per attempt.
DECISION_TIMEOUT_S = 900.0

_config_manager: Optional[ConfigManager] = None


def set_config_manager(config_manager: Optional[ConfigManager]) -> None:
    """Inject ConfigManager (called from main.py lifespan)."""
    global _config_manager
    _config_manager = config_manager


class PendingItem(BaseModel):
    item_id: str
    batch_id: str
    status: str
    confidence: float
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: Optional[str]
    # The example's schema and its correctable fields; None for data no
    # synthetic example schema describes, which can be approved but not
    # corrected.
    schema_name: Optional[str]
    correction_template: Optional[Dict[str, Any]]
    # A rejection of this item merges the corrections instead of
    # regenerating, so it needs at least one.
    corrections_required: bool
    reasoning: str


class PendingItems(BaseModel):
    items: List[PendingItem]


class DecisionRequest(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1)
    feedback: str = ""
    corrections: Dict[str, Any] = {}


class DecisionResponse(BaseModel):
    status: str
    item: PendingItem


class ReviewedItem(BaseModel):
    item_id: str
    batch_id: str
    status: str
    confidence: float
    query: str
    data: Dict[str, Any]
    created_at: Optional[str]
    reviewed_at: Optional[str]
    schema_name: Optional[str]
    reviewer: Optional[str]
    feedback: Optional[str]
    corrections: Dict[str, Any]
    # The item that replaced a rejected one, and where its review stands.
    replacement_id: Optional[str]
    replacement_status: Optional[str]


class ReviewHistory(BaseModel):
    approved: List[ReviewedItem]
    rejected: List[ReviewedItem]


class ReviewStats(BaseModel):
    total: int
    pending: int
    auto_approved: int
    approved: int
    rejected: int
    # Share of all items approved by a reviewer or the confidence threshold.
    approval_rate: float
    # Mean confidence of each group above that holds items.
    average_confidence: Dict[str, float]


# How the statistics group item statuses.
_STATS_GROUPS = {
    ApprovalStatus.PENDING_REVIEW: "pending",
    ApprovalStatus.REGENERATED: "pending",
    ApprovalStatus.AUTO_APPROVED: "auto_approved",
    ApprovalStatus.APPROVED: "approved",
    ApprovalStatus.REJECTED: "rejected",
}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _pending_item(item: ReviewItem, batch_id: str) -> PendingItem:
    try:
        schema_name, template = correction_template(item.data)
        reasoning = review_reasoning(item.data)
        corrections_required = (
            schema_for_item_data(item.data) in CORRECTION_ONLY_SCHEMAS
        )
    except ValueError:
        schema_name, template, reasoning = None, None, ""
        corrections_required = False
    return PendingItem(
        item_id=item.item_id,
        batch_id=batch_id,
        status=item.status.value,
        confidence=item.confidence,
        data=item.data,
        metadata=item.metadata,
        created_at=_iso(item.created_at),
        schema_name=schema_name,
        correction_template=template,
        corrections_required=corrections_required,
        reasoning=reasoning,
    )


def _store_unconfigured(exc: Exception, tenant_id: str) -> HTTPException:
    return failure_response(
        503,
        "approval_store_unavailable",
        f"The approval store of tenant {tenant_id} is not configured.",
        exc,
        tenant_id=tenant_id,
    )


def _require_config_manager() -> ConfigManager:
    if _config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialised")
    return _config_manager


def _agent(tenant_id: str) -> HumanApprovalAgent:
    """The agent over ``tenant_id``'s approval store, without regeneration."""
    config_manager = _require_config_manager()
    try:
        storage = ApprovalStorageImpl.from_system_config(
            config_manager, get_telemetry_manager(), tenant_id
        )
    except Exception as exc:
        raise _store_unconfigured(exc, tenant_id) from exc
    return HumanApprovalAgent.from_approval_config(
        ApprovalConfig(),
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        storage=storage,
    )


def _feedback_handler(tenant_id: str) -> SyntheticDataFeedbackHandler:
    """Regeneration with ``tenant_id``'s primary LM."""
    try:
        return SyntheticDataFeedbackHandler.for_tenant(
            _require_config_manager(), tenant_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise failure_response(
            503,
            "regeneration_unavailable",
            f"Tenant {tenant_id} has no language model to regenerate with.",
            exc,
            tenant_id=tenant_id,
        ) from exc


async def _pending(agent: HumanApprovalAgent, tenant_id: str) -> List[ReviewItem]:
    try:
        return await agent.get_pending_items()
    except Exception as exc:
        raise failure_response(
            502,
            "approval_store_unavailable",
            f"Could not read the items awaiting review for tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc


async def _batches(agent: HumanApprovalAgent, tenant_id: str) -> List[ApprovalBatch]:
    try:
        return await agent.storage.get_batches()
    except Exception as exc:
        raise failure_response(
            502,
            "approval_store_unavailable",
            f"Could not read the review history of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc


def _schema_name(data: Dict[str, Any]) -> Optional[str]:
    try:
        return schema_for_item_data(data).__name__
    except ValueError:
        return None


def _reviewed_item(
    item: ReviewItem, batch_id: str, replacement: Optional[ReviewItem]
) -> ReviewedItem:
    """``item`` with the decision that settled it: its own, or for a
    regenerated item the one its replacement records."""
    decision = item.metadata.get("decision")
    if not isinstance(decision, dict) and replacement is not None:
        decision = replacement.metadata.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    query = item.data.get("query")
    return ReviewedItem(
        item_id=item.item_id,
        batch_id=batch_id,
        status=item.status.value,
        confidence=item.confidence,
        query=query if isinstance(query, str) else "",
        data=item.data,
        created_at=_iso(item.created_at),
        reviewed_at=_iso(item.reviewed_at),
        schema_name=_schema_name(item.data),
        reviewer=decision.get("reviewer"),
        feedback=decision.get("feedback"),
        corrections=decision.get("corrections") or {},
        replacement_id=replacement.item_id if replacement else None,
        replacement_status=replacement.status.value if replacement else None,
    )


def _replacement_of(batch: ApprovalBatch, item_id: str) -> Optional[ReviewItem]:
    return next(
        (
            candidate
            for candidate in batch.items
            if candidate.metadata.get("original_item_id") == item_id
        ),
        None,
    )


def _newest_first(items: List[ReviewedItem]) -> List[ReviewedItem]:
    return sorted(
        items, key=lambda item: item.reviewed_at or item.created_at or "", reverse=True
    )


@router.get("/{tenant_id}/approvals/history", response_model=ReviewHistory)
async def review_history(tenant_id: str):
    """Every approved (by a reviewer or the confidence threshold) and every
    rejected item of the tenant, most recently reviewed first, each with the
    reviewer, feedback and corrections of its decision and, for a rejected
    item, the replacement it was regenerated as."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    agent = await asyncio.to_thread(_agent, tenant_id)
    approved: List[ReviewedItem] = []
    rejected: List[ReviewedItem] = []
    for batch in await _batches(agent, tenant_id):
        for item in batch.items:
            if item.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED):
                approved.append(_reviewed_item(item, batch.batch_id, None))
            elif item.status is ApprovalStatus.REJECTED:
                rejected.append(
                    _reviewed_item(
                        item, batch.batch_id, _replacement_of(batch, item.item_id)
                    )
                )
    return ReviewHistory(
        approved=_newest_first(approved), rejected=_newest_first(rejected)
    )


@router.get("/{tenant_id}/approvals/stats", response_model=ReviewStats)
async def review_stats(tenant_id: str):
    """How many of the tenant's items await review, were approved (by a
    reviewer or the confidence threshold) or were rejected, the share
    approved, and the mean confidence of each group."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    agent = await asyncio.to_thread(_agent, tenant_id)
    confidences: Dict[str, List[float]] = {
        "pending": [],
        "auto_approved": [],
        "approved": [],
        "rejected": [],
    }
    for batch in await _batches(agent, tenant_id):
        for item in batch.items:
            confidences[_STATS_GROUPS[item.status]].append(item.confidence)
    counts = {group: len(values) for group, values in confidences.items()}
    total = sum(counts.values())
    return ReviewStats(
        total=total,
        **counts,
        approval_rate=(
            (counts["approved"] + counts["auto_approved"]) / total if total else 0.0
        ),
        average_confidence={
            group: sum(values) / len(values)
            for group, values in confidences.items()
            if values
        },
    )


@router.get("/{tenant_id}/approvals", response_model=PendingItems)
async def list_pending(tenant_id: str):
    """The tenant's items awaiting review, newest batch first as the store
    returns them, each with its schema's correctable fields."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    agent = await asyncio.to_thread(_agent, tenant_id)
    items = await _pending(agent, tenant_id)
    return PendingItems(
        items=[
            _pending_item(item, item.metadata["approval_batch_id"]) for item in items
        ]
    )


async def _apply(
    agent: HumanApprovalAgent,
    tenant_id: str,
    batch_id: str,
    decision: ReviewDecision,
    expected: ApprovalStatus,
) -> ReviewItem:
    """Apply ``decision`` and return the item it left ``expected``."""
    item_id = decision.item_id
    try:
        result = await asyncio.wait_for(
            agent.apply_decision(batch_id, decision), timeout=DECISION_TIMEOUT_S
        )
    except TimeoutError as exc:
        raise failure_response(
            504,
            "approval_decision_timed_out",
            f"The decision on item {item_id} did not finish within "
            f"{DECISION_TIMEOUT_S:g} seconds.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except ReviewDecisionConflictError as exc:
        raise failure_response(
            409,
            "approval_decision_conflict",
            f"Item {item_id} was already decided by another reviewer.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except Exception as exc:
        raise failure_response(
            502,
            "approval_decision_failed",
            f"The decision on item {item_id} was not recorded.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    if result.status is not expected:
        raise HTTPException(
            status_code=502,
            detail=f"The decision on item {item_id} left it {result.status.value}.",
        )
    return result


@router.post(
    "/{tenant_id}/approvals/{batch_id}/{item_id}", response_model=DecisionResponse
)
async def decide(tenant_id: str, batch_id: str, item_id: str, request: DecisionRequest):
    """Approve or reject one pending item.

    An approval answers ``approved`` with the approved item. A rejection's
    corrections must be fields the item's schema lets a reviewer change. An
    item of a synthetic example schema is regenerated and answers
    ``regenerated`` with the replacement awaiting review: an item of a
    correction-only schema takes the corrections, which it needs at least one
    of, and any other is regenerated by the tenant's LM, which needs the
    reviewer's feedback. An item no schema describes answers ``rejected``.
    """
    tenant_id = canonical_tenant_or_400(tenant_id)
    agent = await asyncio.to_thread(_agent, tenant_id)
    item = next(
        (
            candidate
            for candidate in await _pending(agent, tenant_id)
            if candidate.item_id == item_id
            and candidate.metadata.get("approval_batch_id") == batch_id
        ),
        None,
    )
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} of batch {batch_id} is not awaiting review.",
        )
    pending = _pending_item(item, batch_id)
    regenerable = pending.schema_name is not None
    corrections: Dict[str, Any] = {}
    if request.corrections:
        try:
            corrections = parse_corrections(item.data, request.corrections)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not request.approved and pending.corrections_required and not corrections:
        raise HTTPException(
            status_code=400,
            detail=f"A {pending.schema_name} rejection needs at least one correction.",
        )
    if (
        not request.approved
        and regenerable
        and not pending.corrections_required
        and not request.feedback.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Regenerating a {pending.schema_name} item needs feedback.",
        )
    if not request.approved and regenerable:
        agent.feedback_handler = await asyncio.to_thread(_feedback_handler, tenant_id)

    feedback = request.feedback.strip()
    decision = ReviewDecision(
        item_id=item_id,
        approved=request.approved,
        # A replacement records its rejection's feedback as text, even empty.
        feedback=feedback if not request.approved and regenerable else feedback or None,
        corrections=corrections,
        reviewer=request.reviewer,
    )
    expected = (
        ApprovalStatus.APPROVED
        if request.approved
        else ApprovalStatus.REGENERATED
        if regenerable
        else ApprovalStatus.REJECTED
    )
    result = await _apply(agent, tenant_id, batch_id, decision, expected)
    logger.info(
        "Review of %s/%s for tenant %s: %s by %s",
        batch_id,
        item_id,
        tenant_id,
        result.status.value,
        request.reviewer,
    )
    return DecisionResponse(
        status=result.status.value, item=_pending_item(result, batch_id)
    )


@router.post(
    "/{tenant_id}/approvals/{batch_id}/{item_id}/regenerate",
    response_model=DecisionResponse,
)
async def regenerate(tenant_id: str, batch_id: str, item_id: str):
    """Regenerate a rejected item that has no replacement from its recorded
    rejection, and answer ``regenerated`` with the replacement awaiting
    review. Two requests at once elect one replacement."""
    tenant_id = canonical_tenant_or_400(tenant_id)
    agent = await asyncio.to_thread(_agent, tenant_id)
    try:
        batch = await agent.storage.get_batch(batch_id)
    except Exception as exc:
        raise failure_response(
            502,
            "approval_store_unavailable",
            f"Could not read batch {batch_id} of tenant {tenant_id}.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    item = (
        next(
            (candidate for candidate in batch.items if candidate.item_id == item_id),
            None,
        )
        if batch
        else None
    )
    if item is None:
        raise HTTPException(
            status_code=404, detail=f"Item {item_id} of batch {batch_id} was not found."
        )
    if item.status is not ApprovalStatus.REJECTED:
        raise HTTPException(
            status_code=409,
            detail=f"Item {item_id} is {item.status.value}, not rejected.",
        )
    replacement = _replacement_of(batch, item_id)
    if replacement is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Item {item_id} was already regenerated as {replacement.item_id}.",
        )
    if _schema_name(item.data) is None:
        raise HTTPException(
            status_code=400,
            detail=f"No example schema describes item {item_id}, so it cannot be "
            "regenerated.",
        )
    recorded = item.metadata.get("decision")
    if not isinstance(recorded, dict):
        raise HTTPException(
            status_code=409,
            detail=f"Item {item_id} has no recorded rejection to regenerate from.",
        )
    schema = schema_for_item_data(item.data)
    if schema in CORRECTION_ONLY_SCHEMAS and not recorded.get("corrections"):
        raise HTTPException(
            status_code=400,
            detail=f"Item {item_id} was rejected without corrections, which a "
            f"{schema.__name__} needs to be regenerated.",
        )
    if schema not in CORRECTION_ONLY_SCHEMAS and not recorded.get("feedback"):
        raise HTTPException(
            status_code=400,
            detail=f"Item {item_id} was rejected without feedback, which a "
            f"{schema.__name__} needs to be regenerated.",
        )
    agent.feedback_handler = await asyncio.to_thread(_feedback_handler, tenant_id)
    decision = ReviewDecision(
        item_id=item_id,
        approved=False,
        feedback=recorded.get("feedback"),
        corrections=dict(recorded.get("corrections") or {}),
        reviewer=recorded.get("reviewer"),
    )
    result = await _apply(
        agent, tenant_id, batch_id, decision, ApprovalStatus.REGENERATED
    )
    logger.info(
        "Regenerated %s/%s for tenant %s as %s",
        batch_id,
        item_id,
        tenant_id,
        result.item_id,
    )
    return DecisionResponse(
        status=result.status.value, item=_pending_item(result, batch_id)
    )
