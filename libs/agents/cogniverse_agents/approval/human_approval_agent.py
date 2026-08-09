"""
Human Approval Agent

Generic human-in-the-loop approval agent using dependency injection.
Works for any domain by accepting ConfidenceExtractor and FeedbackHandler.
"""

import copy
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ApprovalStorage,
    ConfidenceExtractor,
    FeedbackHandler,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_core.common.tenant_utils import require_tenant_id

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
        regenerator = ValidatedSyntheticExampleRegenerator(max_retries=3)
        regenerator.lm = primary_lm
        agent = HumanApprovalAgent(
            confidence_extractor=SyntheticDataConfidenceExtractor(),
            feedback_handler=SyntheticDataFeedbackHandler(
                generator=regenerator,
                generation_timeout_seconds=primary_config.request_timeout,
            ),
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
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be a finite number in [0, 1]")
        self.confidence_extractor = confidence_extractor
        self.feedback_handler = feedback_handler
        self.threshold = confidence_threshold
        self.storage = storage

        logger.info(
            f"Initialized HumanApprovalAgent (threshold: {confidence_threshold}, "
            f"storage: {'configured' if storage else 'none'})"
        )

    @classmethod
    def from_approval_config(
        cls,
        approval_config: Any,
        *,
        confidence_extractor: ConfidenceExtractor,
        feedback_handler: Optional[FeedbackHandler] = None,
        storage: Optional[ApprovalStorage] = None,
    ) -> "HumanApprovalAgent":
        """Build an agent using the auto-approval threshold from an
        ``ApprovalConfig`` (``cogniverse_foundation.config.unified_config``)
        rather than a hard-coded value, so the threshold has a single typed
        source of truth.
        """
        return cls(
            confidence_extractor=confidence_extractor,
            feedback_handler=feedback_handler,
            confidence_threshold=approval_config.confidence_threshold,
            storage=storage,
        )

    @staticmethod
    def _validated_confidence(item_id: str, confidence: Any) -> float:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(
                f"Review item {item_id!r} confidence must be a finite number in [0, 1]"
            )
        return float(confidence)

    async def submit_for_review(self, batch: ApprovalBatch) -> ApprovalBatch:
        """Register a pre-built batch for human review and persist it.

        Unlike :meth:`process_batch` (which builds a batch from raw items via
        the confidence extractor), this accepts a caller-built
        :class:`ApprovalBatch` whose items already carry confidence scores —
        e.g. the finetuning synthetic-data path. Each item is (re)classified
        against ``confidence_threshold``: ``>= threshold`` is auto-approved,
        the rest stay ``PENDING_REVIEW`` for a human to resolve in the
        dashboard. The batch is persisted (when storage is configured) so the
        approval queue surfaces it, then returned immediately — review is
        asynchronous, so callers must resume work from the persisted batch
        after a human acts (via :meth:`apply_decision` /
        :meth:`apply_batch_decisions`), not from this return value's pending
        items.
        """
        for item in batch.items:
            item.confidence = self._validated_confidence(item.item_id, item.confidence)
            item.status = (
                ApprovalStatus.AUTO_APPROVED
                if item.confidence >= self.threshold
                else ApprovalStatus.PENDING_REVIEW
            )

        if self.storage:
            await self._persist_submitted_batch(batch)

        logger.info(
            f"Submitted batch {batch.batch_id} for review: "
            f"{len(batch.auto_approved)} auto-approved, "
            f"{len(batch.pending_review)} pending human review"
        )
        return batch

    async def _persist_submitted_batch(self, batch: ApprovalBatch) -> None:
        batch_tenant = require_tenant_id(
            batch.context.get("tenant_id"),
            source=f"approval batch {batch.batch_id} context",
        )
        storage_tenant = getattr(self.storage, "tenant_id", None)
        if not isinstance(storage_tenant, str) or not storage_tenant.strip():
            raise RuntimeError("Approval storage must expose a non-empty tenant_id")
        if storage_tenant != batch_tenant:
            raise ValueError(
                "Approval batch tenant does not match its storage: "
                f"batch={batch.batch_id} context_tenant={batch_tenant} "
                f"storage_tenant={storage_tenant}"
            )
        batch.context["tenant_id"] = batch_tenant
        persist_approved = getattr(self.storage, "persist_approved_item", None)
        if batch.auto_approved and not callable(persist_approved):
            raise RuntimeError(
                "Approval storage must implement persist_approved_item before "
                "auto-approved items can be submitted"
            )
        await self.storage.save_batch(batch)

        for index, item in enumerate(list(batch.items)):
            if item.status is not ApprovalStatus.AUTO_APPROVED:
                continue
            decision = ReviewDecision(
                item_id=item.item_id,
                approved=True,
                feedback=(
                    f"confidence {item.confidence} met automatic approval "
                    f"threshold {self.threshold}"
                ),
                reviewer="cogniverse:auto-approval",
                timestamp=item.created_at,
            )
            batch.items[index] = await persist_approved(
                batch_id=batch.batch_id,
                dataset_name=approved_synthetic_dataset_name(batch_tenant),
                item=item,
                decision=decision,
                project_context=batch.context,
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
        agent_type = context.get("agent_type")
        if not isinstance(agent_type, str) or not agent_type.strip():
            raise ValueError("context.agent_type must be a non-empty string")
        logger.info(f"Processing batch {batch_id} with {len(items)} items")

        # Convert to ReviewItems with confidence scores
        review_items = []
        for i, item in enumerate(items):
            item_id = f"{batch_id}_{i}"
            confidence = self._validated_confidence(
                item_id, self.confidence_extractor.extract(item)
            )
            status = (
                ApprovalStatus.AUTO_APPROVED
                if confidence >= self.threshold
                else ApprovalStatus.PENDING_REVIEW
            )

            review_item = ReviewItem(
                item_id=item_id,
                data=item,
                confidence=confidence,
                metadata={
                    "agent_type": agent_type,
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
            await self._persist_submitted_batch(batch)
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

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        if isinstance(self.storage, ApprovalStorageImpl):
            batch_tenant = require_tenant_id(
                batch.context.get("tenant_id"),
                source=f"approval batch {batch.batch_id} context",
            )
            if self.storage.tenant_id != batch_tenant:
                raise ValueError(
                    "Approval batch tenant does not match its storage: "
                    f"batch={batch.batch_id} context_tenant={batch_tenant} "
                    f"storage_tenant={self.storage.tenant_id}"
                )
            batch.context["tenant_id"] = batch_tenant

        item = next((i for i in batch.items if i.item_id == decision.item_id), None)
        if not item:
            raise ValueError(f"Item {decision.item_id} not found in batch {batch_id}")

        if isinstance(self.storage, ApprovalStorageImpl):
            await self.storage.select_review_decision(
                batch_id=batch.batch_id,
                original_item_id=item.item_id,
                decision=decision,
            )

        if decision.approved:
            if isinstance(self.storage, ApprovalStorageImpl):
                batch_tenant = require_tenant_id(
                    batch.context.get("tenant_id"),
                    source=f"approval batch {batch.batch_id} context",
                )
                if self.storage.tenant_id != batch_tenant:
                    raise ValueError(
                        "Approval batch tenant does not match its storage: "
                        f"batch={batch.batch_id} context_tenant={batch_tenant} "
                        f"storage_tenant={self.storage.tenant_id}"
                    )
                batch.context["tenant_id"] = batch_tenant
                approved = await self.storage.persist_approved_item(
                    batch_id=batch.batch_id,
                    dataset_name=approved_synthetic_dataset_name(batch_tenant),
                    item=item,
                    decision=decision,
                    project_context=batch.context,
                )
            else:
                if not isinstance(decision.timestamp, datetime):
                    raise ValueError("Review decision timestamp is required")
                if (
                    decision.timestamp.tzinfo is None
                    or decision.timestamp.utcoffset() is None
                ):
                    raise ValueError(
                        "Review decision timestamp must include timezone information"
                    )
                approved = copy.deepcopy(item)
                approved.status = ApprovalStatus.APPROVED
                approved.reviewed_at = decision.timestamp
                approved.metadata["decision"] = {
                    "reviewer": decision.reviewer,
                    "feedback": decision.feedback,
                    "corrections": copy.deepcopy(decision.corrections),
                    "timestamp": decision.timestamp.isoformat(),
                }
                await self.storage.update_item(approved, batch_id=batch.batch_id)

            logger.info(
                f"Item {approved.item_id} approved and added to training dataset"
            )
            return approved

        else:
            if self.feedback_handler:
                logger.info(
                    f"Item {item.item_id} rejected, attempting regeneration "
                    f"with feedback: {decision.feedback}"
                )
                regenerated = await self.feedback_handler.process_rejection(
                    item, decision
                )
                if regenerated:
                    if decision.timestamp is None:
                        raise ValueError("Review decision timestamp is required")
                    agent_type = item.metadata.get("agent_type")
                    if not isinstance(agent_type, str) or not agent_type.strip():
                        raise ValueError(
                            "original item metadata.agent_type must be a non-empty string"
                        )
                    regenerated.status = ApprovalStatus.REGENERATED
                    regenerated.metadata["agent_type"] = agent_type
                    regenerated.metadata["original_item_id"] = item.item_id
                    regenerated.metadata["regeneration_feedback"] = decision.feedback
                    regenerated.metadata["decision"] = {
                        "reviewer": decision.reviewer,
                        "feedback": decision.feedback,
                        "corrections": dict(decision.corrections),
                        "timestamp": decision.timestamp.isoformat(),
                    }
                    await self.storage.replace_item(batch.batch_id, item, regenerated)
                    persisted_batch = await self.storage.get_batch(batch.batch_id)
                    if persisted_batch is None:
                        raise RuntimeError(
                            "Replacement batch disappeared after persistence: "
                            f"batch={batch.batch_id} original={item.item_id}"
                        )
                    canonical_replacements = [
                        candidate
                        for candidate in persisted_batch.items
                        if candidate.status is ApprovalStatus.REGENERATED
                        and candidate.metadata.get("original_item_id") == item.item_id
                    ]
                    if len(canonical_replacements) != 1:
                        raise RuntimeError(
                            "Expected one canonical replacement after persistence: "
                            f"batch={batch.batch_id} original={item.item_id} "
                            f"count={len(canonical_replacements)}"
                        )
                    regenerated = canonical_replacements[0]
                    logger.info(
                        f"Item {item.item_id} regenerated successfully as "
                        f"{regenerated.item_id}"
                    )
                    return regenerated

                logger.warning(f"Failed to regenerate item {item.item_id}")
            else:
                logger.info(
                    f"Item {item.item_id} rejected, no feedback handler available"
                )

            # If regeneration produced no replacement, record the rejection
            # before changing the item status so a failed write remains retryable.
            from cogniverse_agents.approval.approval_storage import (
                ApprovalStorageImpl,
            )

            if not isinstance(decision.timestamp, datetime):
                raise ValueError("Review decision timestamp is required")
            if (
                decision.timestamp.tzinfo is None
                or decision.timestamp.utcoffset() is None
            ):
                raise ValueError(
                    "Review decision timestamp must include timezone information"
                )
            if isinstance(self.storage, ApprovalStorageImpl):
                span_id = await self.storage.get_item_span_id(
                    item.item_id, batch_id=batch.batch_id
                )
                if span_id:
                    await self.storage.log_approval_decision(
                        span_id=span_id,
                        item_id=item.item_id,
                        approved=False,
                        feedback=decision.feedback,
                        reviewer=decision.reviewer,
                        decision_timestamp=decision.timestamp,
                    )

            item.status = ApprovalStatus.REJECTED
            item.reviewed_at = decision.timestamp
            item.metadata["decision"] = {
                "reviewer": decision.reviewer,
                "feedback": decision.feedback,
                "corrections": copy.deepcopy(decision.corrections),
                "timestamp": decision.timestamp.isoformat(),
            }
            await self.storage.update_item(item, batch_id=batch.batch_id)
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
        logger.info(f"Applying {len(decisions)} decisions to batch {batch_id}")

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
            for item in batch.pending_review:
                item.metadata["approval_batch_id"] = batch.batch_id
                pending_items.append(item)

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
                sum(item.confidence for item in batch.items) / total if total > 0 else 0
            ),
        }
