"""Human review of a tenant's synthetic examples.

Generated examples the confidence extractor did not auto-approve wait in the
tenant's approval store (Phoenix spans and annotations, with Redis electing
each decision). A reviewer approves an item into the tenant's approved
training dataset, or rejects it with feedback and schema corrections, which
regenerates it with the tenant's primary LM for another review.
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
from cogniverse_core.approval.interfaces import ReviewItem
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import ApprovalConfig
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_runtime.http_errors import failure_response
from cogniverse_synthetic.approval import (
    SyntheticDataConfidenceExtractor,
    SyntheticDataFeedbackHandler,
)
from cogniverse_synthetic.approval.corrections import (
    correction_template,
    parse_corrections,
    review_reasoning,
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


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _pending_item(item: ReviewItem, batch_id: str) -> PendingItem:
    try:
        schema_name, template = correction_template(item.data)
        reasoning = review_reasoning(item.data)
    except ValueError:
        schema_name, template, reasoning = None, None, ""
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


@router.get("/{tenant_id}/approvals", response_model=PendingItems)
async def list_pending(tenant_id: str):
    """The tenant's items awaiting review, newest batch first as the store
    returns them, each with its schema's correctable fields."""
    tenant_id = canonical_tenant_id(tenant_id)
    agent = await asyncio.to_thread(_agent, tenant_id)
    items = await _pending(agent, tenant_id)
    return PendingItems(
        items=[
            _pending_item(item, item.metadata["approval_batch_id"]) for item in items
        ]
    )


@router.post(
    "/{tenant_id}/approvals/{batch_id}/{item_id}", response_model=DecisionResponse
)
async def decide(tenant_id: str, batch_id: str, item_id: str, request: DecisionRequest):
    """Approve or reject one pending item.

    An approval answers ``approved`` with the approved item. A rejection
    needs feedback; its corrections must be fields the item's schema lets a
    reviewer change. An item of a synthetic example schema is regenerated and
    answers ``regenerated`` with the replacement awaiting review; any other
    item answers ``rejected``.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    if not request.approved and not request.feedback.strip():
        raise HTTPException(status_code=400, detail="A rejection needs feedback.")
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
    regenerable = _pending_item(item, batch_id).schema_name is not None
    corrections: Dict[str, Any] = {}
    if request.corrections:
        try:
            corrections = parse_corrections(item.data, request.corrections)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not request.approved and regenerable:
        agent.feedback_handler = await asyncio.to_thread(_feedback_handler, tenant_id)

    decision = ReviewDecision(
        item_id=item_id,
        approved=request.approved,
        feedback=request.feedback.strip() or None,
        corrections=corrections,
        reviewer=request.reviewer,
    )
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

    expected = (
        ApprovalStatus.APPROVED
        if request.approved
        else ApprovalStatus.REGENERATED
        if regenerable
        else ApprovalStatus.REJECTED
    )
    if result.status is not expected:
        raise HTTPException(
            status_code=502,
            detail=f"The decision on item {item_id} left it {result.status.value}.",
        )
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
