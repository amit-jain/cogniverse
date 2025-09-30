"""
Annotation Storage for Phoenix

Stores human and LLM annotations in Phoenix using SpanEvaluations.
Provides query capabilities for the feedback loop.
Reuses common evaluation patterns from src.evaluation.span_evaluator.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import phoenix as px
from phoenix.trace import SpanEvaluations

from src.app.routing.llm_auto_annotator import AnnotationLabel, AutoAnnotation
from src.app.telemetry.config import SERVICE_NAME_ORCHESTRATION, TelemetryConfig

logger = logging.getLogger(__name__)


class AnnotationStorage:
    """
    Stores and retrieves routing annotations in Phoenix

    Annotations are stored as span attributes with the following structure:
    - annotation.label: The annotation label (correct_routing, wrong_routing, etc.)
    - annotation.confidence: Confidence score (0-1)
    - annotation.reasoning: Human or LLM reasoning
    - annotation.annotator: Who provided the annotation (human or llm)
    - annotation.timestamp: When annotation was created
    - annotation.suggested_agent: Suggested correct agent (if wrong_routing)
    - annotation.human_reviewed: Whether human has reviewed this
    """

    def __init__(self, tenant_id: str = "default"):
        """
        Initialize annotation storage

        Args:
            tenant_id: Tenant identifier
        """
        self.tenant_id = tenant_id
        self.telemetry_config = TelemetryConfig.from_env()
        self.phoenix_client = px.Client()
        self.project_name = self.telemetry_config.get_project_name(
            tenant_id, service=SERVICE_NAME_ORCHESTRATION
        )

        logger.info(
            f"💾 Initialized AnnotationStorage for tenant '{tenant_id}' (project: {self.project_name})"
        )

    def store_llm_annotation(self, span_id: str, annotation: AutoAnnotation) -> bool:
        """
        Store LLM-generated annotation for a span

        Args:
            span_id: Phoenix span ID
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

        return self._update_span_attributes(span_id, annotation_data)

    def store_human_annotation(
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
            span_id: Phoenix span ID
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

        return self._update_span_attributes(span_id, annotation_data)

    def approve_llm_annotation(self, span_id: str, annotator_id: str = "human") -> bool:
        """
        Mark LLM annotation as reviewed and approved by human

        Args:
            span_id: Phoenix span ID
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

        return self._update_span_attributes(span_id, update_data)

    def _update_span_attributes(self, span_id: str, attributes: Dict) -> bool:
        """
        Store annotation as span evaluation in Phoenix using SpanEvaluations API

        This follows the pattern from src.evaluation.span_evaluator for consistency.

        Args:
            span_id: Phoenix span ID
            attributes: Dictionary of attributes to set

        Returns:
            True if stored successfully
        """
        try:
            # Prepare evaluation data following Phoenix SpanEvaluations format
            # This matches the pattern in src.evaluation.span_evaluator.upload_evaluations_to_phoenix
            eval_data = {
                "context.span_id": [span_id],
                "label": [attributes.get("annotation.label", "")],
                "score": [attributes.get("annotation.confidence", 0.0)],
                "explanation": [attributes.get("annotation.reasoning", "")],
            }

            # Add custom metadata fields
            for key, value in attributes.items():
                if not key.startswith("annotation."):
                    continue
                # Store as separate columns for querying
                safe_key = key.replace(".", "_")
                eval_data[safe_key] = [value]

            # Create evaluations dataframe
            eval_df = pd.DataFrame(eval_data)

            # Log using SpanEvaluations with eval_name (like span_evaluator does)
            span_evals = SpanEvaluations(
                eval_name="routing_annotation", dataframe=eval_df
            )
            self.phoenix_client.log_evaluations(span_evals)

            logger.info(f"✅ Stored routing annotation for span {span_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store annotation for span {span_id}: {e}")
            raise

    def query_annotated_spans(
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
            spans_df = self.phoenix_client.get_spans_dataframe(
                project_name=self.project_name, start_time=start_time, end_time=end_time
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

    def get_annotation_statistics(self) -> Dict:
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

            annotated_spans = self.query_annotated_spans(
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
