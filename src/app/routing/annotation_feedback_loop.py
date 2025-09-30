"""
Annotation Feedback Loop

Periodically queries annotated routing spans from Phoenix
and feeds them to the AdvancedRoutingOptimizer as ground truth training data.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from src.app.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer,
    RoutingExperience,
)
from src.app.routing.annotation_storage import AnnotationStorage
from src.app.routing.llm_auto_annotator import AnnotationLabel

logger = logging.getLogger(__name__)


class AnnotationFeedbackLoop:
    """
    Feedback loop that converts human annotations into optimizer training data

    This loop:
    1. Periodically queries Phoenix for newly annotated routing spans
    2. Converts annotations into ground truth experiences
    3. Feeds experiences to AdvancedRoutingOptimizer
    4. Triggers optimizer updates when sufficient annotations collected
    """

    def __init__(
        self,
        optimizer: AdvancedRoutingOptimizer,
        tenant_id: str = "default",
        poll_interval_minutes: int = 15,
        min_annotations_for_update: int = 10
    ):
        """
        Initialize feedback loop

        Args:
            optimizer: Routing optimizer to train
            tenant_id: Tenant identifier
            poll_interval_minutes: How often to check for new annotations
            min_annotations_for_update: Minimum annotations before triggering optimizer update
        """
        self.optimizer = optimizer
        self.tenant_id = tenant_id
        self.poll_interval_minutes = poll_interval_minutes
        self.min_annotations_for_update = min_annotations_for_update

        # Initialize storage
        self.annotation_storage = AnnotationStorage(tenant_id=tenant_id)

        # Track last processed time
        self._last_processed_time = datetime.now()
        self._processed_annotation_ids = set()

        # Statistics
        self._total_annotations_processed = 0
        self._total_experiences_created = 0
        self._optimizer_updates_triggered = 0

        logger.info(
            f"🔄 Initialized AnnotationFeedbackLoop for tenant '{tenant_id}' "
            f"(poll_interval: {poll_interval_minutes}m, "
            f"min_annotations: {min_annotations_for_update})"
        )

    async def start(self):
        """Start the feedback loop (runs continuously)"""
        logger.info("🚀 Starting annotation feedback loop")

        while True:
            try:
                await self.process_new_annotations()
                await asyncio.sleep(self.poll_interval_minutes * 60)
            except Exception as e:
                logger.error(f"❌ Error in feedback loop: {e}")
                # Wait 1 minute before retrying on error
                await asyncio.sleep(60)

    async def process_new_annotations(self) -> Dict:
        """
        Process newly added annotations and feed to optimizer

        Returns:
            Dictionary with processing statistics
        """
        logger.info("🔍 Processing new annotations")

        # Query annotations since last check
        end_time = datetime.now()
        start_time = self._last_processed_time

        try:
            annotated_spans = self.annotation_storage.query_annotated_spans(
                start_time=start_time,
                end_time=end_time,
                only_human_reviewed=True  # Only use human-reviewed annotations
            )
        except Exception as e:
            logger.error(f"❌ Error querying annotated spans: {e}")
            return {
                "annotations_found": 0,
                "experiences_created": 0,
                "optimizer_updated": False,
                "error": str(e)
            }

        if not annotated_spans:
            logger.info("📭 No new annotations found")
            return {
                "annotations_found": 0,
                "experiences_created": 0,
                "optimizer_updated": False
            }

        logger.info(f"📊 Found {len(annotated_spans)} new annotations")

        # Convert annotations to experiences and feed to optimizer
        experiences_created = 0
        new_annotation_ids = []

        for span_data in annotated_spans:
            try:
                # Skip if already processed
                annotation_id = f"{span_data.get('span_id')}_{span_data.get('annotation_timestamp')}"
                if annotation_id in self._processed_annotation_ids:
                    continue

                # Convert to routing experience
                experience = self._annotation_to_experience(span_data)
                if not experience:
                    continue

                # Feed to optimizer
                reward = await self.optimizer.record_routing_experience(
                    query=experience.query,
                    entities=experience.entities,
                    relationships=experience.relationships,
                    enhanced_query=experience.enhanced_query,
                    chosen_agent=experience.chosen_agent,
                    routing_confidence=experience.routing_confidence,
                    search_quality=experience.search_quality,
                    agent_success=experience.agent_success,
                    processing_time=experience.processing_time,
                    user_satisfaction=experience.user_satisfaction
                )

                experiences_created += 1
                new_annotation_ids.append(annotation_id)

                logger.info(
                    f"✅ Created experience from annotation: {experience.chosen_agent} "
                    f"(label: {span_data.get('annotation_label')}, reward: {reward:.3f})"
                )

            except Exception as e:
                logger.error(f"❌ Error processing annotation: {e}")
                continue

        # Update tracking
        self._processed_annotation_ids.update(new_annotation_ids)
        self._total_annotations_processed += len(new_annotation_ids)
        self._total_experiences_created += experiences_created
        self._last_processed_time = end_time

        # Trigger optimizer update if enough annotations
        optimizer_updated = False
        if experiences_created >= self.min_annotations_for_update:
            logger.info(
                f"🎯 Triggering optimizer update "
                f"({experiences_created} new experiences)"
            )
            try:
                # Optimizer automatically updates its model when recording experiences
                # Additional explicit update could be triggered here if needed
                optimizer_updated = True
                self._optimizer_updates_triggered += 1
            except Exception as e:
                logger.error(f"❌ Error triggering optimizer update: {e}")

        result = {
            "annotations_found": len(annotated_spans),
            "experiences_created": experiences_created,
            "optimizer_updated": optimizer_updated,
            "total_annotations_processed": self._total_annotations_processed,
            "total_experiences_created": self._total_experiences_created,
            "optimizer_updates_triggered": self._optimizer_updates_triggered
        }

        logger.info(
            f"✅ Processed {experiences_created} annotations into experiences "
            f"(optimizer_updated: {optimizer_updated})"
        )

        return result

    def _annotation_to_experience(self, span_data: Dict) -> Optional[RoutingExperience]:
        """
        Convert annotated span data to RoutingExperience

        Args:
            span_data: Dictionary with span + annotation data

        Returns:
            RoutingExperience if conversion successful, None otherwise
        """
        try:
            # Extract required fields
            query = span_data.get("query")
            chosen_agent = span_data.get("chosen_agent")
            routing_confidence = span_data.get("routing_confidence")
            annotation_label = span_data.get("annotation_label")

            if not all([query, chosen_agent, routing_confidence is not None, annotation_label]):
                logger.warning("⚠️ Missing required fields in span data")
                return None

            # Convert annotation label to success/failure
            agent_success = self._label_to_success(annotation_label)

            # Derive search quality from annotation
            # High quality if annotation is "correct_routing", lower for others
            search_quality = self._label_to_search_quality(annotation_label)

            # Extract context if available
            context = span_data.get("context", {})
            entities = context.get("entities", []) if isinstance(context, dict) else []
            relationships = context.get("relationships", []) if isinstance(context, dict) else []
            enhanced_query = context.get("enhanced_query", query) if isinstance(context, dict) else query

            # Create experience
            experience = RoutingExperience(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
                chosen_agent=chosen_agent,
                routing_confidence=float(routing_confidence),
                search_quality=search_quality,
                agent_success=agent_success,
                processing_time=0.0,  # Not available from annotations
                user_satisfaction=1.0 if agent_success else 0.0,  # Derive from success
                timestamp=datetime.now()
            )

            return experience

        except Exception as e:
            logger.error(f"❌ Error converting annotation to experience: {e}")
            return None

    def _label_to_success(self, label: str) -> bool:
        """
        Convert annotation label to agent success boolean

        Args:
            label: Annotation label string

        Returns:
            True if routing was correct, False otherwise
        """
        try:
            label_enum = AnnotationLabel(label)
            return label_enum == AnnotationLabel.CORRECT_ROUTING
        except ValueError:
            logger.warning(f"⚠️ Unknown annotation label: {label}")
            return False

    def _label_to_search_quality(self, label: str) -> float:
        """
        Convert annotation label to search quality score

        Args:
            label: Annotation label string

        Returns:
            Quality score 0.0-1.0
        """
        try:
            label_enum = AnnotationLabel(label)

            quality_map = {
                AnnotationLabel.CORRECT_ROUTING: 0.9,  # High quality
                AnnotationLabel.WRONG_ROUTING: 0.3,  # Low quality
                AnnotationLabel.AMBIGUOUS: 0.6,  # Medium quality
                AnnotationLabel.INSUFFICIENT_INFO: 0.5  # Neutral
            }

            return quality_map.get(label_enum, 0.5)

        except ValueError:
            logger.warning(f"⚠️ Unknown annotation label: {label}")
            return 0.5

    def get_statistics(self) -> Dict:
        """
        Get feedback loop statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "total_annotations_processed": self._total_annotations_processed,
            "total_experiences_created": self._total_experiences_created,
            "optimizer_updates_triggered": self._optimizer_updates_triggered,
            "last_processed_time": self._last_processed_time.isoformat(),
            "poll_interval_minutes": self.poll_interval_minutes,
            "min_annotations_for_update": self.min_annotations_for_update
        }
