"""
Orchestration Feedback Loop

Periodically queries annotated orchestration spans from Phoenix
and feeds them to WorkflowIntelligence as ground truth training data.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from cogniverse_agents.workflow_intelligence import (
    WorkflowExecution,
    WorkflowIntelligence,
)
from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotationStorage,
)

logger = logging.getLogger(__name__)


class OrchestrationFeedbackLoop:
    """
    Feedback loop that converts orchestration annotations into optimizer training data

    This loop:
    1. Periodically queries Phoenix for newly annotated orchestration spans
    2. Converts human feedback into corrected ground truth workflows
    3. Feeds ground truth to WorkflowIntelligence for learning
    4. Triggers DSPy optimization when sufficient annotations collected
    """

    def __init__(
        self,
        workflow_intelligence: WorkflowIntelligence,
        tenant_id: str = "default",
        poll_interval_minutes: int = 15,
        min_annotations_for_update: int = 10,
    ):
        """
        Initialize orchestration feedback loop

        Args:
            workflow_intelligence: Workflow optimizer to train
            tenant_id: Tenant identifier
            poll_interval_minutes: How often to check for new annotations
            min_annotations_for_update: Minimum annotations before triggering optimization
        """
        self.workflow_intelligence = workflow_intelligence
        self.tenant_id = tenant_id
        self.poll_interval_minutes = poll_interval_minutes
        self.min_annotations_for_update = min_annotations_for_update

        # Initialize storage
        self.annotation_storage = OrchestrationAnnotationStorage(tenant_id=tenant_id)

        # Track last processed time
        self._last_processed_time = datetime.now()
        self._processed_annotation_ids = set()

        # Statistics
        self._total_annotations_processed = 0
        self._total_workflows_learned = 0
        self._optimizer_updates_triggered = 0

        logger.info(
            f"🔄 Initialized OrchestrationFeedbackLoop for tenant '{tenant_id}' "
            f"(poll_interval: {poll_interval_minutes}m, "
            f"min_annotations: {min_annotations_for_update})"
        )

    async def start(self):
        """Start the feedback loop (runs continuously)"""
        logger.info("🚀 Starting orchestration feedback loop")

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
        logger.info("🔍 Processing new orchestration annotations")

        # Query annotations since last check
        end_time = datetime.now()
        start_time = self._last_processed_time

        try:
            annotated_spans = await self.annotation_storage.query_annotated_spans(
                start_time=start_time,
                end_time=end_time,
                only_human_reviewed=True,  # Only use human-reviewed annotations
            )
        except Exception as e:
            logger.error(f"❌ Error querying annotated spans: {e}")
            return {
                "annotations_found": 0,
                "workflows_learned": 0,
                "optimizer_updated": False,
                "error": str(e),
            }

        if not annotated_spans:
            logger.info("📭 No new orchestration annotations found")
            return {
                "annotations_found": 0,
                "workflows_learned": 0,
                "optimizer_updated": False,
            }

        logger.info(f"📊 Found {len(annotated_spans)} new orchestration annotations")

        # Convert annotations to ground truth workflows
        workflows_learned = 0
        new_annotation_ids = []

        for span_data in annotated_spans:
            try:
                # Skip if already processed
                annotation_timestamp = (
                    span_data.get("annotations", [{}])[0]
                    .get("metadata", {})
                    .get("annotation_timestamp")
                )
                annotation_id = f"{span_data.get('span_id')}_{annotation_timestamp}"
                if annotation_id in self._processed_annotation_ids:
                    continue

                # Convert to ground truth workflow execution
                ground_truth_workflow = self._annotation_to_ground_truth(span_data)
                if not ground_truth_workflow:
                    continue

                # Feed to WorkflowIntelligence as ground truth
                await self.workflow_intelligence.record_ground_truth_execution(
                    ground_truth_workflow
                )

                self._processed_annotation_ids.add(annotation_id)
                new_annotation_ids.append(annotation_id)
                workflows_learned += 1

            except Exception as e:
                logger.error(f"❌ Error processing annotation: {e}")

        self._total_annotations_processed += len(annotated_spans)
        self._total_workflows_learned += workflows_learned
        self._last_processed_time = end_time

        # Trigger optimizer update if we have enough annotations
        optimizer_updated = False
        if workflows_learned >= self.min_annotations_for_update:
            logger.info(
                f"🎯 Sufficient annotations collected ({workflows_learned}), "
                f"triggering WorkflowIntelligence optimization"
            )

            try:
                await self.workflow_intelligence.optimize_from_ground_truth()
                optimizer_updated = True
                self._optimizer_updates_triggered += 1
            except Exception as e:
                logger.error(f"❌ Optimization failed: {e}")

        result = {
            "annotations_found": len(annotated_spans),
            "workflows_learned": workflows_learned,
            "optimizer_updated": optimizer_updated,
            "total_annotations_processed": self._total_annotations_processed,
            "total_workflows_learned": self._total_workflows_learned,
            "optimizer_updates_triggered": self._optimizer_updates_triggered,
        }

        logger.info(
            f"✅ Orchestration feedback processing complete: "
            f"{workflows_learned} workflows learned, "
            f"optimizer_updated={optimizer_updated}"
        )

        return result

    def _annotation_to_ground_truth(
        self, span_data: Dict
    ) -> Optional[WorkflowExecution]:
        """
        Convert human annotation to ground truth WorkflowExecution

        Key insight: Use human-corrected values as ground truth:
        - If human suggests different pattern → use suggested_pattern
        - If human suggests different agents → use suggested_agents
        - If human suggests different order → use suggested_execution_order
        - Quality score comes from human feedback

        This creates training data showing what the "correct" orchestration should have been.
        """
        try:
            annotation = span_data.get("annotations", [{}])[0]
            metadata = annotation.get("metadata", {})
            span_info = span_data.get("span_data", {})
            attributes = span_info.get("attributes", {})

            # Extract actual workflow data
            workflow_id = attributes.get("orchestration.workflow_id")
            query = attributes.get("orchestration.query")

            if not workflow_id or not query:
                logger.warning("Missing workflow_id or query in span data")
                return None

            # Use CORRECTED values from human annotation
            # This is the key: we're creating "ideal" workflow for training

            # Pattern: Use suggested if provided, otherwise actual
            if not metadata.get("pattern_is_optimal"):
                orchestration_pattern = metadata.get("suggested_pattern")
            else:
                orchestration_pattern = metadata.get("actual_pattern")

            # Agents: Use suggested if provided, otherwise actual
            if not metadata.get("agents_are_correct"):
                suggested_agents_str = metadata.get("suggested_agents", "")
                agent_sequence = (
                    suggested_agents_str.split(",") if suggested_agents_str else []
                )
            else:
                actual_agents_str = metadata.get("actual_agents", "")
                agent_sequence = (
                    actual_agents_str.split(",") if actual_agents_str else []
                )

            # Execution order: Use suggested if provided, otherwise actual
            if not metadata.get("execution_order_is_optimal"):
                suggested_order_str = metadata.get("suggested_execution_order", "")
                execution_order = (
                    suggested_order_str.split(",") if suggested_order_str else []
                )
            else:
                actual_order_str = metadata.get("actual_execution_order", "")
                execution_order = (
                    actual_order_str.split(",") if actual_order_str else []
                )

            # Quality: Use human quality score
            quality_label = annotation.get("result", {}).get("label", "acceptable")
            quality_score = annotation.get("result", {}).get("score", 0.5)

            # Success: Based on human quality assessment
            success = quality_label in ["excellent", "good"]

            # Classify query type for pattern learning
            query_type = self._classify_query_type(query, orchestration_pattern)

            # Determine final agent sequence
            # Use execution_order if it matches agent_sequence length, otherwise use agent_sequence
            final_agent_sequence = agent_sequence
            if execution_order and len(execution_order) == len(agent_sequence):
                final_agent_sequence = execution_order

            # Create ground truth WorkflowExecution
            ground_truth = WorkflowExecution(
                workflow_id=f"{workflow_id}_corrected",
                query=query,
                query_type=query_type,
                execution_time=float(metadata.get("execution_time", 0.0)),
                success=success,
                agent_sequence=final_agent_sequence,
                task_count=len(agent_sequence),
                parallel_efficiency=1.0 if orchestration_pattern == "parallel" else 0.0,
                confidence_score=1.0,  # Ground truth is high confidence
                user_satisfaction=quality_score,
                metadata={
                    "source": "human_annotation",
                    "original_workflow_id": workflow_id,
                    "orchestration_pattern": orchestration_pattern,
                    "annotation_quality": quality_label,
                    "improvement_notes": metadata.get("improvement_notes"),
                    "what_went_well": metadata.get("what_went_well"),
                    "what_went_wrong": metadata.get("what_went_wrong"),
                },
            )

            logger.debug(
                f"Converted annotation to ground truth: "
                f"pattern={orchestration_pattern}, agents={agent_sequence}"
            )

            return ground_truth

        except Exception as e:
            logger.error(f"Error converting annotation to ground truth: {e}")
            return None

    def _classify_query_type(self, query: str, pattern: str) -> str:
        """Classify query type for pattern learning"""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["videos and documents", "images and text"]
        ):
            return "multi_modal_search"

        if any(word in query_lower for word in ["detailed", "analysis", "report"]):
            if pattern == "sequential":
                return "sequential_report"
            return "detailed_analysis"

        if any(word in query_lower for word in ["summarize", "summary"]):
            return "summarization"

        return f"{pattern}_query"
