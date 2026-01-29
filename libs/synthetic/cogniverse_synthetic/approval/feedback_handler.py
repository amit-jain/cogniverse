"""
Synthetic Data Feedback Handler

Handle rejection feedback and regenerate synthetic data with DSPy.
"""

import logging
from typing import Any, Dict, Optional

from cogniverse_agents.approval.interfaces import (
    ApprovalStatus,
    FeedbackHandler,
    ReviewDecision,
    ReviewItem,
)
from cogniverse_synthetic.dspy_modules import ValidatedEntityQueryGenerator

logger = logging.getLogger(__name__)


class SyntheticDataFeedbackHandler(FeedbackHandler):
    """
    Handle rejection feedback for synthetic data

    When humans reject synthetic queries:
    1. Log rejection reason and corrections
    2. Regenerate query with DSPy using feedback
    3. Return new ReviewItem with regenerated data

    Example feedback corrections:
    {
        "entities": ["PyTorch", "Tutorial"],  # Updated entities
        "query_hint": "focus on beginner tutorials"  # Additional guidance
    }
    """

    def __init__(
        self,
        generator: Optional[ValidatedEntityQueryGenerator] = None,
        max_regeneration_attempts: int = 2,
    ):
        """
        Initialize feedback handler

        Args:
            generator: DSPy generator for regeneration (creates default if None)
            max_regeneration_attempts: Maximum regeneration attempts per item
        """
        self.generator = generator or ValidatedEntityQueryGenerator(max_retries=3)
        self.max_attempts = max_regeneration_attempts

        logger.info(
            f"Initialized SyntheticDataFeedbackHandler "
            f"(max_attempts: {max_regeneration_attempts})"
        )

    async def process_rejection(
        self, item: ReviewItem, decision: ReviewDecision
    ) -> Optional[ReviewItem]:
        """
        Process rejection and regenerate with feedback

        Args:
            item: Original rejected item
            decision: Human decision with feedback and corrections

        Returns:
            New ReviewItem with regenerated data, or None if regeneration failed
        """
        logger.info(f"Processing rejection for {item.item_id}: {decision.feedback}")

        # Extract original data
        original_data = item.data
        topics = original_data.get("topics", "")
        entities = original_data.get("entities", [])
        entity_types = original_data.get("entity_types", [])

        # Apply corrections from human
        corrections = decision.corrections
        if "entities" in corrections:
            entities = corrections["entities"]
            logger.info(f"Using corrected entities: {entities}")

        if "topics" in corrections:
            topics = corrections["topics"]
            logger.info(f"Using corrected topics: {topics}")

        # Attempt regeneration
        for attempt in range(self.max_attempts):
            try:
                logger.info(
                    f"Regeneration attempt {attempt + 1}/{self.max_attempts} "
                    f"for {item.item_id}"
                )

                # Regenerate with DSPy
                result = self.generator.forward(
                    topics=topics if isinstance(topics, str) else ", ".join(topics),
                    entities=(
                        entities if isinstance(entities, str) else ", ".join(entities)
                    ),
                    entity_types=(
                        entity_types
                        if isinstance(entity_types, str)
                        else ", ".join(entity_types)
                    ),
                )

                # Create new data with regenerated query
                regenerated_data = {
                    "query": result.query,
                    "reasoning": (
                        result.reasoning if hasattr(result, "reasoning") else ""
                    ),
                    "topics": topics,
                    "entities": entities,
                    "entity_types": entity_types,
                    "_generation_metadata": {
                        "retry_count": attempt,
                        "max_retries": self.max_attempts,
                        "regeneration": True,
                        "original_query": original_data.get("query", ""),
                        "human_feedback": decision.feedback,
                        "corrections_applied": corrections,
                    },
                }

                # Create new ReviewItem
                new_item = ReviewItem(
                    item_id=f"{item.item_id}_regen_{attempt}",
                    data=regenerated_data,
                    confidence=0.8,  # Regenerated items start with medium confidence
                    metadata={
                        "original_item_id": item.item_id,
                        "regeneration_attempt": attempt + 1,
                        "feedback": decision.feedback,
                    },
                    status=ApprovalStatus.REGENERATED,
                )

                logger.info(
                    f"Successfully regenerated {item.item_id}: " f"'{result.query}'"
                )

                return new_item

            except Exception as e:
                logger.warning(
                    f"Regeneration attempt {attempt + 1} failed for {item.item_id}: {e}"
                )

        # All attempts failed
        logger.error(
            f"Failed to regenerate {item.item_id} after {self.max_attempts} attempts"
        )
        return None

    def get_regeneration_stats(self, items: list[ReviewItem]) -> Dict[str, Any]:
        """
        Get regeneration statistics

        Args:
            items: List of review items (original + regenerated)

        Returns:
            Dictionary with regeneration metrics
        """
        regenerated = [
            item for item in items if item.status == ApprovalStatus.REGENERATED
        ]
        successful = [
            item
            for item in regenerated
            if item.metadata.get("regeneration_attempt", 0) <= self.max_attempts
        ]

        return {
            "total_items": len(items),
            "regenerated_count": len(regenerated),
            "successful_regenerations": len(successful),
            "failed_regenerations": len(regenerated) - len(successful),
            "regeneration_rate": (
                len(regenerated) / len(items) if len(items) > 0 else 0
            ),
        }
