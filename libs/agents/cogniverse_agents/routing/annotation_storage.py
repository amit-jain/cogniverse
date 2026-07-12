"""
Annotation Storage for Telemetry

Stores human and LLM annotations in telemetry backend using annotations API.
Provides query capabilities for the feedback loop.
Reuses common evaluation patterns from cogniverse_evaluation.span_evaluator.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel, AutoAnnotation
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


def _meta_get(ann_row: "pd.Series", key: str, default: Any = None) -> Any:
    """Read an annotation metadata field. Phoenix returns metadata either as a
    nested dict in a ``metadata`` column or flattened to ``metadata.<key>``
    columns (see preference_extractor); handle both."""
    meta = ann_row.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta.get(key, default)
    col = f"metadata.{key}"
    if col in ann_row.index:
        val = ann_row[col]
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return val
    return default


def _routing_get(span_row: "pd.Series", field: str, default: Any = None) -> Any:
    """Read a routing.* span attribute. Phoenix nests dotted attributes into an
    ``attributes.routing`` dict column; fall back to a flat column if present."""
    routing = span_row.get("attributes.routing")
    if isinstance(routing, dict) and field in routing:
        return routing.get(field, default)
    col = f"attributes.routing.{field}"
    if col in span_row.index:
        val = span_row[col]
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return val
    return default


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

    def __init__(
        self,
        tenant_id: str,
    ):
        """
        Initialize annotation storage

        Args:
            tenant_id: Tenant identifier
        """
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
        except Exception as e:
            # A backend failure is not "no annotated spans" — swallowing it
            # into [] hid Phoenix outages from every caller.
            logger.error(f"❌ Error querying annotated spans: {e!r}")
            raise

        if spans_df.empty:
            logger.info("📭 No spans found in time range")
            return []

        # Annotations live in Phoenix's separate annotation store, not on the
        # span attributes — fetch them and join to spans by span_id. The old
        # read of ``attributes.annotation.label`` was never populated, so the
        # feedback loop and dashboard always saw zero annotations.
        annotations_df = await self.provider.annotations.get_annotations(
            spans_df=spans_df,
            project=self.project_name,
            annotation_names=["routing_annotation"],
        )
        if annotations_df is None or annotations_df.empty:
            logger.info("📭 No routing annotations found in time range")
            return []

        # annotations_df is indexed by span_id (no span_id column).
        annotations_by_span = {sid: row for sid, row in annotations_df.iterrows()}

        annotated_spans = []
        for _, span_row in spans_df.iterrows():
            span_id = span_row.get("context.span_id")
            ann_row = annotations_by_span.get(span_id)
            if ann_row is None:
                continue

            human_reviewed = bool(_meta_get(ann_row, "human_reviewed", False))
            if only_human_reviewed and not human_reviewed:
                continue

            annotated_spans.append(
                {
                    "span_id": span_id,
                    "query": _routing_get(span_row, "query"),
                    "chosen_agent": _routing_get(span_row, "chosen_agent"),
                    "routing_confidence": _routing_get(span_row, "confidence"),
                    "annotation_label": ann_row.get("result.label"),
                    "annotation_confidence": ann_row.get("result.score", 1.0),
                    "annotation_reasoning": _meta_get(ann_row, "reasoning", ""),
                    "annotation_timestamp": _meta_get(ann_row, "timestamp"),
                    "suggested_agent": _meta_get(ann_row, "suggested_agent"),
                    "human_reviewed": human_reviewed,
                    "context": _routing_get(span_row, "context", {}),
                }
            )

        logger.info(f"✅ Found {len(annotated_spans)} annotated spans")
        return annotated_spans

    async def get_annotation_statistics(self) -> Dict:
        """
        Get statistics about stored annotations

        Returns:
            Dictionary with annotation statistics
        """
        # Query recent annotations (last 30 days). UTC-aware so the Phoenix
        # window is not shifted by the host's local offset.
        from datetime import timedelta

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)

        try:
            annotated_spans = await self.query_annotated_spans(
                start_time=start_time, end_time=end_time, only_human_reviewed=False
            )
        except Exception as e:
            # A backend failure must not read as "zero annotations".
            logger.error(f"❌ Error getting annotation statistics: {e!r}")
            raise

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
