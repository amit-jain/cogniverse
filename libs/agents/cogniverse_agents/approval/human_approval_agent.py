"""
Human Approval Agent

Generic human-in-the-loop approval agent using dependency injection.
Works for any domain by accepting ConfidenceExtractor and FeedbackHandler.
"""

import logging
from typing import Any, Dict, List, Optional

from cogniverse_agents.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ApprovalStorage,
    ConfidenceExtractor,
    FeedbackHandler,
    ReviewDecision,
    ReviewItem,
)

logger = logging.getLogger(__name__)


class HumanApprovalAgent:
    """
    Generic human-in-the-loop approval agent

    Uses dependency injection to support any domain:
    - ConfidenceExtractor: Domain-specific confidence scoring
    - FeedbackHandler: Domain-specific rejection handling
    - ApprovalStorage: Persistence backend (Phoenix, database, etc.)

    Example usage:
        # For synthetic data
        agent = HumanApprovalAgent(
            confidence_extractor=SyntheticDataConfidenceExtractor(),
            feedback_handler=SyntheticDataFeedbackHandler(generator),
            confidence_threshold=0.85
        )

        # For optimization results
        agent = HumanApprovalAgent(
            confidence_extractor=OptimizationConfidenceExtractor(),
            feedback_handler=OptimizationFeedbackHandler(),
            confidence_threshold=0.90
        )
    """

    def __init__(
        self,
        confidence_extractor: ConfidenceExtractor,
        feedback_handler: Optional[FeedbackHandler] = None,
        confidence_threshold: float = 0.85,
        storage: Optional[ApprovalStorage] = None,
    ):
        """
        Initialize approval agent

        Args:
            confidence_extractor: Extract confidence from domain data
            feedback_handler: Handle rejection feedback (optional)
            confidence_threshold: Auto-approve above this score (0-1)
            storage: Storage backend for approval data
        """
        self.confidence_extractor = confidence_extractor
        self.feedback_handler = feedback_handler
        self.threshold = confidence_threshold
        self.storage = storage

        logger.info(
            f"Initialized HumanApprovalAgent (threshold: {confidence_threshold}, "
            f"storage: {'configured' if storage else 'none'})"
        )

    async def process_batch(
        self, items: List[Dict[str, Any]], batch_id: str, context: Dict[str, Any]
    ) -> ApprovalBatch:
        """
        Process batch of items, splitting by confidence

        Items with confidence >= threshold are auto-approved.
        Items with confidence < threshold require human review.

        Args:
            items: List of domain-specific data dictionaries
            batch_id: Unique identifier for this batch
            context: Additional batch context (tenant, optimizer, etc.)

        Returns:
            ApprovalBatch with items split by confidence
        """
        logger.info(f"Processing batch {batch_id} with {len(items)} items")

        # Convert to ReviewItems with confidence scores
        review_items = []
        for i, item in enumerate(items):
            confidence = self.confidence_extractor.extract(item)
            status = (
                ApprovalStatus.AUTO_APPROVED
                if confidence >= self.threshold
                else ApprovalStatus.PENDING_REVIEW
            )

            review_item = ReviewItem(
                item_id=f"{batch_id}_{i}",
                data=item,
                confidence=confidence,
                metadata={
                    "batch_id": batch_id,
                    "index": i,
                },
                status=status,
            )
            review_items.append(review_item)

        # Create batch
        batch = ApprovalBatch(batch_id=batch_id, items=review_items, context=context)

        # Save to storage if available
        if self.storage:
            await self.storage.save_batch(batch)
            logger.info(
                f"Saved batch {batch_id} to storage "
                f"(auto_approved: {len(batch.auto_approved)}, "
                f"pending: {len(batch.pending_review)})"
            )

        logger.info(
            f"Batch {batch_id} processed: "
            f"{len(batch.auto_approved)} auto-approved, "
            f"{len(batch.pending_review)} pending review"
        )

        return batch

    async def apply_decision(
        self, batch_id: str, decision: ReviewDecision
    ) -> Optional[ReviewItem]:
        """
        Apply human decision to a review item

        If approved, mark as approved.
        If rejected and feedback_handler is available, attempt regeneration.

        Args:
            batch_id: Batch containing the item
            decision: Human decision with feedback

        Returns:
            Updated ReviewItem (or regenerated item if rejected)
        """
        logger.info(
            f"Applying decision for {decision.item_id} in batch {batch_id}: "
            f"{'APPROVED' if decision.approved else 'REJECTED'}"
        )

        # Get batch and item
        if not self.storage:
            raise ValueError("Storage required for apply_decision")

        batch = await self.storage.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")

        item = next((i for i in batch.items if i.item_id == decision.item_id), None)
        if not item:
            raise ValueError(
                f"Item {decision.item_id} not found in batch {batch_id}"
            )

        # Apply decision
        if decision.approved:
            item.status = ApprovalStatus.APPROVED
            await self.storage.update_item(item, batch_id=batch.batch_id)

            # Log approval decision as Phoenix annotation (SpanEvaluation)
            # Find span_id for this item and annotate it
            from cogniverse_agents.approval.phoenix_storage import (
                PhoenixApprovalStorage,
            )
            if isinstance(self.storage, PhoenixApprovalStorage):
                span_id = await self.storage.get_item_span_id(item.item_id, batch_id=batch.batch_id)
                if span_id:
                    await self.storage.log_approval_decision(
                        span_id=span_id,
                        item_id=item.item_id,
                        approved=True,
                        feedback=decision.feedback,
                        reviewer=decision.reviewer
                    )

                # Add approved item to training dataset
                dataset_name = batch.context.get("dataset_name", "approved_synthetic_data")
                await self.storage.append_to_training_dataset(
                    dataset_name=dataset_name,
                    items=[item],
                    project_context=batch.context
                )

            logger.info(f"Item {item.item_id} approved and added to training dataset")
            return item

        else:
            item.status = ApprovalStatus.REJECTED
            await self.storage.update_item(item, batch_id=batch.batch_id)

            # Log rejection decision as Phoenix annotation (SpanEvaluation)
            from cogniverse_agents.approval.phoenix_storage import (
                PhoenixApprovalStorage,
            )
            if isinstance(self.storage, PhoenixApprovalStorage):
                span_id = await self.storage.get_item_span_id(item.item_id, batch_id=batch.batch_id)
                if span_id:
                    await self.storage.log_approval_decision(
                        span_id=span_id,
                        item_id=item.item_id,
                        approved=False,
                        feedback=decision.feedback,
                        reviewer=decision.reviewer
                    )

            # Attempt regeneration if feedback handler available
            if self.feedback_handler:
                logger.info(
                    f"Item {item.item_id} rejected, attempting regeneration "
                    f"with feedback: {decision.feedback}"
                )
                regenerated = await self.feedback_handler.process_rejection(
                    item, decision
                )

                if regenerated:
                    regenerated.status = ApprovalStatus.REGENERATED
                    regenerated.metadata["original_item_id"] = item.item_id
                    regenerated.metadata["regeneration_feedback"] = decision.feedback
                    await self.storage.update_item(regenerated, batch_id=batch.batch_id)
                    logger.info(
                        f"Item {item.item_id} regenerated successfully as "
                        f"{regenerated.item_id}"
                    )
                    return regenerated
                else:
                    logger.warning(f"Failed to regenerate item {item.item_id}")
            else:
                logger.info(
                    f"Item {item.item_id} rejected, no feedback handler available"
                )

            return item

    async def apply_batch_decisions(
        self, batch_id: str, decisions: List[ReviewDecision]
    ) -> ApprovalBatch:
        """
        Apply multiple decisions at once

        Args:
            batch_id: Batch containing the items
            decisions: List of human decisions

        Returns:
            Updated ApprovalBatch
        """
        logger.info(
            f"Applying {len(decisions)} decisions to batch {batch_id}"
        )

        for decision in decisions:
            await self.apply_decision(batch_id, decision)

        # Return updated batch
        if self.storage:
            batch = await self.storage.get_batch(batch_id)
            if batch:
                return batch

        raise ValueError(f"Failed to retrieve updated batch {batch_id}")

    async def get_pending_items(
        self, context_filter: Optional[Dict[str, Any]] = None
    ) -> List[ReviewItem]:
        """
        Get all items awaiting human review

        Args:
            context_filter: Optional filter by batch context
                Example: {"tenant_id": "acme_corp", "optimizer_type": "routing"}

        Returns:
            List of ReviewItems with PENDING_REVIEW status
        """
        if not self.storage:
            raise ValueError("Storage required for get_pending_items")

        batches = await self.storage.get_pending_batches(context_filter)

        pending_items = []
        for batch in batches:
            pending_items.extend(batch.pending_review)

        logger.info(
            f"Found {len(pending_items)} pending items across {len(batches)} batches"
        )
        return pending_items

    def get_approval_stats(self, batch: ApprovalBatch) -> Dict[str, Any]:
        """
        Get approval statistics for a batch

        Args:
            batch: Approval batch

        Returns:
            Dictionary with approval metrics
        """
        total = len(batch.items)
        auto_approved = len(batch.auto_approved)
        pending = len(batch.pending_review)
        approved = len(batch.approved)
        rejected = len(batch.rejected)

        return {
            "batch_id": batch.batch_id,
            "total_items": total,
            "auto_approved": auto_approved,
            "auto_approved_pct": auto_approved / total if total > 0 else 0,
            "pending_review": pending,
            "pending_review_pct": pending / total if total > 0 else 0,
            "human_approved": approved,
            "human_approved_pct": approved / total if total > 0 else 0,
            "rejected": rejected,
            "rejected_pct": rejected / total if total > 0 else 0,
            "overall_approval_rate": batch.approval_rate,
            "avg_confidence": (
                sum(item.confidence for item in batch.items) / total
                if total > 0
                else 0
            ),
        }
