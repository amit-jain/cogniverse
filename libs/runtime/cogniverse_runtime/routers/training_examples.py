"""Operator-written training examples for a tenant's optimizers.

An upload carries examples in one optimizer's synthetic example schema. They
are validated as a whole, saved as a review batch, and approved by the
uploader into the tenant's approved training dataset, which that optimizer's
runs read alongside its production spans.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cogniverse_agents.approval import (
    ApprovalStorageImpl,
    HumanApprovalAgent,
    ReviewedBatchIncompleteError,
)
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import ApprovalConfig
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_runtime.http_errors import failure_response
from cogniverse_synthetic.approval import SyntheticDataConfidenceExtractor
from cogniverse_synthetic.approval.uploads import (
    MAX_UPLOADED_EXAMPLES,
    UploadedExamplesError,
    parse_uploaded_examples,
    upload_templates,
)
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER

logger = logging.getLogger(__name__)

router = APIRouter()

_config_manager: Optional[ConfigManager] = None


def set_config_manager(config_manager: Optional[ConfigManager]) -> None:
    """Inject ConfigManager (called from main.py lifespan)."""
    global _config_manager
    _config_manager = config_manager


class TrainingExampleTemplates(BaseModel):
    templates: Dict[str, Dict[str, Any]]
    max_examples: int


class TrainingExamplesUpload(BaseModel):
    optimizer: str
    reviewer: str = Field(min_length=1)
    source: str = Field("", description="The uploaded file's name")
    examples: List[Any]


class TrainingExamplesUploaded(BaseModel):
    batch_id: str
    optimizer: str
    dataset: str
    item_ids: List[str]


@router.get("/training-example-templates", response_model=TrainingExampleTemplates)
async def training_example_templates():
    """Per optimizer: its example schema's fields, the required ones and a
    valid example; and how many examples one upload may hold."""
    return TrainingExampleTemplates(
        templates=upload_templates(), max_examples=MAX_UPLOADED_EXAMPLES
    )


def _agent(tenant_id: str) -> HumanApprovalAgent:
    if _config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialised")
    try:
        storage = ApprovalStorageImpl.from_system_config(
            _config_manager, get_telemetry_manager(), tenant_id
        )
    except Exception as exc:
        raise failure_response(
            503,
            "approval_store_unavailable",
            f"The approval store of tenant {tenant_id} is not configured.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    return HumanApprovalAgent.from_approval_config(
        ApprovalConfig(),
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        storage=storage,
    )


@router.post("/{tenant_id}/training-examples", response_model=TrainingExamplesUploaded)
async def upload_training_examples(tenant_id: str, body: TrainingExamplesUpload):
    """Approve ``body.examples`` into the tenant's training dataset for
    ``body.optimizer``, in order, with the uploader as reviewer.

    Nothing is written when any example is invalid. An upload that fails part
    way leaves the examples not yet approved awaiting review in the tenant's
    approval queue.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    reviewer = body.reviewer.strip()
    if not reviewer:
        raise HTTPException(status_code=400, detail="Name the reviewer.")
    try:
        records = parse_uploaded_examples(body.optimizer, body.examples)
    except UploadedExamplesError as exc:
        raise HTTPException(
            status_code=400, detail={"message": str(exc), "errors": exc.errors}
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent_type = APPROVED_TRAINING_AGENT_BY_OPTIMIZER[body.optimizer]
    batch_id = f"upload_{body.optimizer}_{uuid.uuid4().hex}"
    source = body.source.strip()
    batch = ApprovalBatch(
        batch_id=batch_id,
        items=[
            ReviewItem(
                item_id=f"{batch_id}_{index}",
                data=record,
                confidence=1.0,
                status=ApprovalStatus.PENDING_REVIEW,
                metadata={"agent_type": agent_type, "optimizer_type": body.optimizer},
            )
            for index, record in enumerate(records)
        ],
        context={
            "tenant_id": tenant_id,
            "agent_type": agent_type,
            "optimizer": body.optimizer,
            "purpose": "optimizer_training",
            "source": "upload",
            "source_file": source,
        },
    )
    agent = await asyncio.to_thread(_agent, tenant_id)
    feedback = f"Uploaded from {source}" if source else "Uploaded"
    try:
        await agent.submit_reviewed_batch(batch, reviewer=reviewer, feedback=feedback)
    except ReviewedBatchIncompleteError as exc:
        raise failure_response(
            502,
            "training_examples_incomplete",
            f"{len(exc.approved_item_ids)} of {exc.total} uploaded examples "
            f"were approved into the training dataset; the rest of batch "
            f"{batch_id} await review in the approval queue.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except Exception as exc:
        raise failure_response(
            502,
            "training_examples_not_stored",
            f"The upload of {len(batch.items)} examples was not stored.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    logger.info(
        "Uploaded %d %s examples for tenant %s by %s",
        len(batch.items),
        body.optimizer,
        tenant_id,
        reviewer,
    )
    return TrainingExamplesUploaded(
        batch_id=batch_id,
        optimizer=body.optimizer,
        dataset=approved_synthetic_dataset_name(tenant_id),
        item_ids=[item.item_id for item in batch.items],
    )
