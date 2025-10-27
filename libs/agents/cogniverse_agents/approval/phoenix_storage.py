"""
Phoenix Approval Storage

Stores approval data as Phoenix spans with annotations.
Enables approval workflow tracing and analysis.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register

from cogniverse_agents.approval.interfaces import (
    ApprovalBatch,
    ApprovalStorage,
    ReviewDecision,
    ReviewItem,
)

logger = logging.getLogger(__name__)


class PhoenixApprovalStorage(ApprovalStorage):
    """
    Store approval data in Phoenix as spans

    Structure:
    - approval_batch (root span): Contains batch metadata
      - approval_item (child span): One per review item
        - Attributes: item_id, confidence, status, data
        - Annotations: Human decisions with feedback

    Benefits:
    - Integrated with existing Phoenix infrastructure
    - Trace approval workflows alongside optimization
    - Query and analyze approval patterns
    - No additional database needed
    """

    def __init__(
        self,
        phoenix_endpoint: str = "http://localhost:6006",
        project_name: str = "approval_system",
    ):
        """
        Initialize Phoenix storage

        Args:
            phoenix_endpoint: Phoenix collector endpoint
            project_name: Phoenix project for approval data
        """
        self.phoenix_endpoint = phoenix_endpoint
        self.project_name = project_name

        # Register with Phoenix
        register(
            project_name=project_name,
            endpoint=phoenix_endpoint,
        )

        self.tracer = trace.get_tracer(__name__)
        logger.info(
            f"Initialized PhoenixApprovalStorage (project: {project_name}, "
            f"endpoint: {phoenix_endpoint})"
        )

    async def save_batch(self, batch: ApprovalBatch) -> str:
        """
        Save approval batch as Phoenix span tree

        Creates:
        - Root span for batch with context attributes
        - Child span for each item with confidence and status

        Args:
            batch: Batch to save

        Returns:
            Batch ID
        """
        with self.tracer.start_as_current_span(
            "approval_batch",
            attributes={
                "batch_id": batch.batch_id,
                "total_items": len(batch.items),
                "auto_approved": len(batch.auto_approved),
                "pending_review": len(batch.pending_review),
                "context": json.dumps(batch.context),
                "created_at": batch.created_at.isoformat() if batch.created_at else None,
            },
        ) as batch_span:
            # Create child span for each item
            for item in batch.items:
                self._create_item_span(item)

            batch_span.set_status(Status(StatusCode.OK))
            logger.info(f"Saved batch {batch.batch_id} to Phoenix")

        return batch.batch_id

    def _create_item_span(self, item: ReviewItem) -> None:
        """Create Phoenix span for a review item"""
        with self.tracer.start_as_current_span(
            "approval_item",
            attributes={
                "item_id": item.item_id,
                "confidence": item.confidence,
                "status": item.status.value,
                "data": json.dumps(item.data),
                "metadata": json.dumps(item.metadata),
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
            },
        ) as item_span:
            item_span.set_status(Status(StatusCode.OK))

    async def get_batch(self, batch_id: str) -> Optional[ApprovalBatch]:
        """
        Retrieve approval batch from Phoenix

        Note: This is a simplified implementation that queries Phoenix spans.
        In production, you may want to add caching or use Phoenix query API.

        Args:
            batch_id: Batch ID to retrieve

        Returns:
            ApprovalBatch if found, None otherwise
        """
        # Query Phoenix spans for this batch
        # This is a placeholder - actual implementation would use Phoenix query API
        logger.warning(
            f"get_batch not fully implemented for Phoenix storage (batch_id: {batch_id})"
        )
        return None

    async def update_item(self, item: ReviewItem) -> None:
        """
        Update review item status

        Creates new span with updated status and links to original.
        Preserves full audit trail.

        Args:
            item: Item with updated status
        """
        with self.tracer.start_as_current_span(
            "approval_item_update",
            attributes={
                "item_id": item.item_id,
                "new_status": item.status.value,
                "confidence": item.confidence,
                "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else datetime.utcnow().isoformat(),
            },
        ) as update_span:
            # Add metadata about the update
            if "original_item_id" in item.metadata:
                update_span.set_attribute(
                    "original_item_id", item.metadata["original_item_id"]
                )

            if "regeneration_feedback" in item.metadata:
                update_span.set_attribute(
                    "regeneration_feedback", item.metadata["regeneration_feedback"]
                )

            # Create event for the status change
            update_span.add_event(
                "status_changed",
                attributes={
                    "item_id": item.item_id,
                    "new_status": item.status.value,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            update_span.set_status(Status(StatusCode.OK))
            logger.info(f"Updated item {item.item_id} status to {item.status.value}")

    async def get_pending_batches(
        self, context_filter: Optional[Dict[str, Any]] = None
    ) -> List[ApprovalBatch]:
        """
        Get batches with pending reviews

        Note: This is a simplified implementation.
        In production, query Phoenix for spans with pending_review > 0.

        Args:
            context_filter: Optional filter by batch context

        Returns:
            List of batches with pending items
        """
        logger.warning(
            "get_pending_batches not fully implemented for Phoenix storage"
        )
        return []

    async def record_decision(
        self, decision: ReviewDecision, item: ReviewItem
    ) -> None:
        """
        Record human decision as Phoenix annotation

        Args:
            decision: Human decision
            item: Review item being decided on
        """
        with self.tracer.start_as_current_span(
            "approval_decision",
            attributes={
                "item_id": decision.item_id,
                "approved": decision.approved,
                "reviewer": decision.reviewer or "unknown",
                "timestamp": decision.timestamp.isoformat() if decision.timestamp else datetime.utcnow().isoformat(),
                "feedback": decision.feedback or "",
                "corrections": json.dumps(decision.corrections),
            },
        ) as decision_span:
            # Add event for the decision
            decision_span.add_event(
                "human_decision",
                attributes={
                    "item_id": decision.item_id,
                    "approved": decision.approved,
                    "has_feedback": bool(decision.feedback),
                    "has_corrections": len(decision.corrections) > 0,
                },
            )

            decision_span.set_status(Status(StatusCode.OK))
            logger.info(
                f"Recorded decision for {decision.item_id}: "
                f"{'APPROVED' if decision.approved else 'REJECTED'}"
            )
