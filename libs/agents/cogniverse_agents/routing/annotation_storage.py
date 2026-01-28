"""
Annotation Storage for Telemetry

Stores human and LLM annotations in telemetry backend using annotations API.
Provides query capabilities for the feedback loop.
Reuses common evaluation patterns from cogniverse_evaluation.span_evaluator.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel, AutoAnnotation
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class RoutingAnnotationStorage:
    """
    Stores and retrieves routing annotations in telemetry backend

    Annotations are stored using the telemetry provider's annotation API with the following metadata:
    - label: The annotation label (correct_routing, wrong_routing, etc.)
    - score: Confidence score (0-1)
    - metadata.reasoning: Human or LLM reasoning
    - metadata.annotator: Who provided the annotation (human or llm)
    - metadata.timestamp: When annotation was created
    - metadata.suggested_agent: Suggested correct agent (if wrong_routing)
    - metadata.human_reviewed: Whether human has reviewed this
    """

    def __init__(self, tenant_id: str = "default"):
        """
        Initialize annotation storage

        Args:
            tenant_id: Tenant identifier
        """
        self.tenant_id = tenant_id

        # Get telemetry manager and use its config (shared singleton config)
        telemetry_manager = get_telemetry_manager()
        self.telemetry_config = telemetry_manager.config

        # Get unified tenant project name for routing annotations
        self.project_name = self.telemetry_config.get_project_name(tenant_id)

        # Get telemetry provider for annotations
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=tenant_id
        )

        logger.info(
            f"💾 Initialized RoutingAnnotationStorage for tenant '{tenant_id}' (project: {self.project_name})"
        )

    async def store_llm_annotation(
        self, span_id: str, annotation: AutoAnnotation
    ) -> bool:
        """
        Store LLM-generated annotation for a span

        Args:
            span_id: Span ID
            annotation: LLM annotation to store

        Returns:
            True if stored successfully
        """
        logger.info(f"💾 Storing LLM annotation for span {span_id}")

        annotation_data = {
            "annotation.label": annotation.label.value,
            "annotation.confidence": annotation.confidence,
            "annotation.reasoning": annotation.reasoning,
            "annotation.annotator": "llm",
            "annotation.timestamp": datetime.now().isoformat(),
            "annotation.human_reviewed": False,
            "annotation.requires_review": annotation.requires_human_review,
        }

        if annotation.suggested_correct_agent:
            annotation_data["annotation.suggested_agent"] = (
                annotation.suggested_correct_agent
            )

        return await self._update_span_attributes(span_id, annotation_data)

    async def store_human_annotation(
        self,
        span_id: str,
        label: AnnotationLabel,
        reasoning: str,
        suggested_agent: Optional[str] = None,
        annotator_id: str = "human",
    ) -> bool:
        """
        Store human annotation for a span

        Args:
            span_id: Span ID
            label: Annotation label
            reasoning: Human reasoning
            suggested_agent: Suggested correct agent (if wrong_routing)
            annotator_id: Human annotator identifier

        Returns:
            True if stored successfully
        """
        logger.info(f"💾 Storing human annotation for span {span_id}")

        annotation_data = {
            "annotation.label": label.value,
            "annotation.confidence": 1.0,  # Human annotations have full confidence
            "annotation.reasoning": reasoning,
            "annotation.annotator": annotator_id,
            "annotation.timestamp": datetime.now().isoformat(),
            "annotation.human_reviewed": True,
            "annotation.requires_review": False,
        }

        if suggested_agent:
            annotation_data["annotation.suggested_agent"] = suggested_agent

        return await self._update_span_attributes(span_id, annotation_data)

    async def approve_llm_annotation(
        self, span_id: str, annotator_id: str = "human"
    ) -> bool:
        """
        Mark LLM annotation as reviewed and approved by human

        Args:
            span_id: Span ID
            annotator_id: Human annotator identifier

        Returns:
            True if updated successfully
        """
        logger.info(f"✅ Approving LLM annotation for span {span_id}")

        update_data = {
            "annotation.human_reviewed": True,
            "annotation.requires_review": False,
            "annotation.approved_by": annotator_id,
            "annotation.approval_timestamp": datetime.now().isoformat(),
        }

        return await self._update_span_attributes(span_id, update_data)

    async def _update_span_attributes(self, span_id: str, attributes: Dict) -> bool:
        """
        Store annotation using telemetry provider's annotation API

        Args:
            span_id: Span ID
            attributes: Dictionary of annotation attributes

        Returns:
            True if stored successfully
        """
        try:
            # Extract label, score, and metadata from attributes
            label = attributes.get("annotation.label", "")
            score = attributes.get("annotation.confidence", 0.0)

            # Build metadata dictionary (remove annotation. prefix)
            metadata = {}
            for key, value in attributes.items():
                if key.startswith("annotation."):
                    # Strip prefix for metadata
                    meta_key = key.replace("annotation.", "")
                    metadata[meta_key] = value

            # Use provider's annotation API
            await self.provider.annotations.add_annotation(
                span_id=span_id,
                name="routing_annotation",
                label=label,
                score=score,
                metadata=metadata,
                project=self.project_name,
            )

            logger.info(f"✅ Stored routing annotation for span {span_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store annotation for span {span_id}: {e}")
            raise

    async def query_annotated_spans(
        self, start_time: datetime, end_time: datetime, only_human_reviewed: bool = True
    ) -> List[Dict]:
        """
        Query annotated spans for feedback loop

        Args:
            start_time: Start of time range
            end_time: End of time range
            only_human_reviewed: Only return human-reviewed annotations

        Returns:
            List of dictionaries containing span data + annotations
        """
        logger.info(
            f"🔍 Querying annotated spans "
            f"(time range: {start_time} to {end_time}, "
            f"human_reviewed_only: {only_human_reviewed})"
        )

        try:
            # Get all spans in time range
            spans_df = await self.provider.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                limit=10000,
            )

            if spans_df.empty:
                logger.info("📭 No spans found in time range")
                return []

            # Filter for spans with annotations
            # Check if annotation.label attribute exists
            annotated_spans = []

            for _, span_row in spans_df.iterrows():
                # Look for annotation.label in flattened attributes
                annotation_label = span_row.get("attributes.annotation.label")
                if not annotation_label:
                    continue

                # Filter by human_reviewed if requested
                if only_human_reviewed:
                    human_reviewed = span_row.get(
                        "attributes.annotation.human_reviewed", False
                    )
                    if not human_reviewed:
                        continue

                # Extract all annotation data
                annotation_data = {
                    "span_id": span_row.get("context.span_id"),
                    "query": span_row.get("attributes.routing.query"),
                    "chosen_agent": span_row.get("attributes.routing.chosen_agent"),
                    "routing_confidence": span_row.get("attributes.routing.confidence"),
                    "annotation_label": annotation_label,
                    "annotation_confidence": span_row.get(
                        "attributes.annotation.confidence", 1.0
                    ),
                    "annotation_reasoning": span_row.get(
                        "attributes.annotation.reasoning", ""
                    ),
                    "annotation_timestamp": span_row.get(
                        "attributes.annotation.timestamp"
                    ),
                    "suggested_agent": span_row.get(
                        "attributes.annotation.suggested_agent"
                    ),
                    "context": span_row.get("attributes.routing.context", {}),
                }

                annotated_spans.append(annotation_data)

            logger.info(f"✅ Found {len(annotated_spans)} annotated spans")
            return annotated_spans

        except Exception as e:
            logger.error(f"❌ Error querying annotated spans: {e}")
            return []

    async def get_annotation_statistics(self) -> Dict:
        """
        Get statistics about stored annotations

        Returns:
            Dictionary with annotation statistics
        """
        try:
            # Query recent annotations (last 30 days)
            from datetime import timedelta

            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)

            annotated_spans = await self.query_annotated_spans(
                start_time=start_time, end_time=end_time, only_human_reviewed=False
            )

            if not annotated_spans:
                return {
                    "total": 0,
                    "human_reviewed": 0,
                    "pending_review": 0,
                    "by_label": {},
                }

            human_reviewed = sum(
                1
                for span in annotated_spans
                if span.get("annotation_label") and span.get("human_reviewed", False)
            )

            by_label = {}
            for span in annotated_spans:
                label = span.get("annotation_label", "unknown")
                by_label[label] = by_label.get(label, 0) + 1

            return {
                "total": len(annotated_spans),
                "human_reviewed": human_reviewed,
                "pending_review": len(annotated_spans) - human_reviewed,
                "by_label": by_label,
            }

        except Exception as e:
            logger.error(f"❌ Error getting annotation statistics: {e}")
            return {
                "total": 0,
                "human_reviewed": 0,
                "pending_review": 0,
                "by_label": {},
                "error": str(e),
            }
