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


class LLMAnnotationNotFoundError(LookupError):
    """The span carries no annotation to approve."""


class NotAnLLMAnnotationError(ValueError):
    """The span's annotation was not made by the LLM annotator."""


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


# Span ids per by-id span query; bounds the query string and each response.
_SPAN_ID_BATCH = 200


class AnnotationStorage:
    """
    Stores and retrieves per-agent-type annotations in the telemetry backend.

    Each agent type persists under its own Phoenix annotation name
    (``{agent_type}_annotation``); ``routing`` keeps the historical
    ``routing_annotation`` name so stored data stays readable.

    Annotations are stored using the telemetry provider's annotation API with the following metadata:
    - label: The annotation label (correct, wrong, correct_routing, etc.)
    - score: Confidence score (0-1)
    - metadata.reasoning: Human or LLM reasoning
    - metadata.annotator: Who provided the annotation (human or llm)
    - metadata.timestamp: When annotation was created
    - metadata.suggested_agent: Suggested correct agent (if wrong)
    - metadata.human_reviewed: Whether human has reviewed this
    """

    def __init__(
        self,
        tenant_id: str,
        agent_type: str = "routing",
    ):
        """
        Initialize annotation storage

        Args:
            tenant_id: Tenant identifier
            agent_type: Which agent's decisions this storage annotates
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        # The runtime emits spans under the canonical tenant project; read
        # and write the same one regardless of how the caller spelled the
        # tenant, or real-traffic spans are invisible to the loop.
        tenant_id = canonical_tenant_id(tenant_id)
        self.tenant_id = tenant_id

        # Get telemetry manager and use its config (shared singleton config)
        telemetry_manager = get_telemetry_manager()
        self.telemetry_config = telemetry_manager.config
        self.agent_type = agent_type
        self.annotation_name = f"{agent_type}_annotation"

        # Get unified tenant project name for annotations
        self.project_name = self.telemetry_config.get_project_name(tenant_id)

        # Get telemetry provider for annotations
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=tenant_id
        )

        logger.info(
            f"💾 Initialized AnnotationStorage for tenant '{tenant_id}' "
            f"(agent_type: {agent_type}, project: {self.project_name})"
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
            "annotation.timestamp": datetime.now(timezone.utc).isoformat(),
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
            "annotation.timestamp": datetime.now(timezone.utc).isoformat(),
            "annotation.human_reviewed": True,
            "annotation.requires_review": False,
        }

        if suggested_agent:
            annotation_data["annotation.suggested_agent"] = suggested_agent

        return await self._update_span_attributes(span_id, annotation_data)

    async def get_annotation(self, span_id: str) -> Optional[Dict[str, Any]]:
        """The span's latest annotation of this storage's name as
        ``{"label", "score", "metadata"}``, or ``None`` when it has none."""
        annotations = await self.provider.annotations.get_annotations(
            spans_df=pd.DataFrame({"context.span_id": [span_id]}),
            project=self.project_name,
            annotation_names=[self.annotation_name],
        )
        if annotations is None or annotations.empty:
            return None
        if "updated_at" in annotations.columns:
            annotations = annotations.sort_values("updated_at")
        latest = annotations.iloc[-1]
        metadata = latest.get("metadata")
        return {
            "label": latest.get("result.label"),
            "score": latest.get("result.score"),
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        }

    async def approve_llm_annotation(
        self, span_id: str, annotator_id: str = "human"
    ) -> Dict[str, Any]:
        """Mark the span's LLM annotation as reviewed and approved by
        ``annotator_id``, keeping its label, confidence and reasoning.

        Returns the approved annotation. Raises
        ``LLMAnnotationNotFoundError`` when the span has no annotation and
        ``NotAnLLMAnnotationError`` when a person made it.
        """
        logger.info(f"✅ Approving LLM annotation for span {span_id}")

        annotation = await self.get_annotation(span_id)
        if annotation is None:
            raise LLMAnnotationNotFoundError(
                f"span {span_id} has no {self.annotation_name} to approve"
            )
        if annotation["metadata"].get("annotator") != "llm":
            raise NotAnLLMAnnotationError(
                f"the {self.annotation_name} of span {span_id} was made by "
                f"{annotation['metadata'].get('annotator')!r}, not the LLM"
            )
        metadata = {
            **annotation["metadata"],
            "human_reviewed": True,
            "requires_review": False,
            "approved_by": annotator_id,
            "approval_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.provider.annotations.add_annotation(
            span_id=span_id,
            name=self.annotation_name,
            label=annotation["label"],
            score=annotation["score"],
            metadata=metadata,
            project=self.project_name,
        )
        return {**annotation, "metadata": metadata}

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
                name=self.annotation_name,
                label=label,
                score=score,
                metadata=metadata,
                project=self.project_name,
            )

            logger.info(f"✅ Stored {self.annotation_name} for span {span_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store annotation for span {span_id}: {e}")
            raise

    async def fetch_project_spans(
        self, start_time: datetime, end_time: datetime
    ) -> "pd.DataFrame":
        """Pull the ids of the tenant project's spans for a time window.

        The annotation join needs every span name (annotations attach to
        whichever span each agent emitted), so the pull is unfiltered, and it
        carries only ``context.span_id``: ``query_annotated_spans`` reads the
        attributes of the annotated spans alone. Callers that query several
        agent types over one window fetch this frame once and pass it to
        ``query_annotated_spans(spans_df=...)``.
        """
        try:
            frame = await self.provider.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                limit=10000,
                columns=["span_id"],
            )
            if "context.span_id" not in frame.columns:
                frame = frame.reset_index()
            return frame
        except Exception as e:
            # A backend failure is not "no annotated spans" — swallowing it
            # into [] hid Phoenix outages from every caller.
            logger.error(f"❌ Error querying annotated spans: {e!r}")
            raise

    async def _fetch_spans_by_id(
        self, span_ids: List[str], *, start_time: datetime, end_time: datetime
    ) -> Dict[str, "pd.Series"]:
        """The full rows of ``span_ids``, keyed by span id, fetched in
        batches of ``_SPAN_ID_BATCH``."""
        rows: Dict[str, "pd.Series"] = {}
        for offset in range(0, len(span_ids), _SPAN_ID_BATCH):
            batch = span_ids[offset : offset + _SPAN_ID_BATCH]
            frame = await self.provider.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                filters={"span_id": batch},
                limit=len(batch),
            )
            if "context.span_id" not in frame.columns:
                frame = frame.reset_index()
            for _, row in frame.iterrows():
                rows[row["context.span_id"]] = row
        return rows

    async def query_annotated_spans(
        self,
        start_time: datetime,
        end_time: datetime,
        only_human_reviewed: bool = True,
        spans_df: Optional["pd.DataFrame"] = None,
    ) -> List[Dict]:
        """
        Query annotated spans for feedback loop

        Args:
            start_time: Start of time range
            end_time: End of time range
            only_human_reviewed: Only return human-reviewed annotations
            spans_df: Optional pre-fetched project spans over the same
                window (see ``fetch_project_spans``). When provided, the
                per-call whole-project span pull is skipped.

        Returns:
            List of dictionaries containing span data + annotations
        """
        logger.info(
            f"🔍 Querying annotated spans "
            f"(time range: {start_time} to {end_time}, "
            f"human_reviewed_only: {only_human_reviewed})"
        )

        if spans_df is None:
            spans_df = await self.fetch_project_spans(start_time, end_time)

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
            annotation_names=[self.annotation_name],
        )
        if annotations_df is None or annotations_df.empty:
            logger.info(f"📭 No {self.annotation_name} found in time range")
            return []

        # annotations_df is indexed by span_id (no span_id column).
        annotations_by_span = {sid: row for sid, row in annotations_df.iterrows()}
        annotated_ids = [
            span_id
            for span_id in spans_df["context.span_id"]
            if span_id in annotations_by_span
        ]
        span_rows = await self._fetch_spans_by_id(
            annotated_ids, start_time=start_time, end_time=end_time
        )

        from cogniverse_foundation.telemetry.span_contract import read_span_io

        annotated_spans = []
        for span_id in annotated_ids:
            span_row = span_rows.get(span_id)
            if span_row is None:
                continue
            ann_row = annotations_by_span[span_id]

            human_reviewed = bool(_meta_get(ann_row, "human_reviewed", False))
            if only_human_reviewed and not human_reviewed:
                continue

            # Span fields come from the canonical input.value/output.value
            # slots the emitters write; the legacy attributes.routing dict is
            # a fallback for pre-migration spans (the emitter never set it, so
            # reading only the legacy attribute returned empty fields for
            # every real span).
            io = read_span_io(span_row)
            output = io["output"] if isinstance(io.get("output"), dict) else {}
            annotated_spans.append(
                {
                    "span_id": span_id,
                    "agent_type": self.agent_type,
                    "query": io.get("input") or _routing_get(span_row, "query"),
                    "chosen_agent": (
                        output.get("chosen_agent")
                        or output.get("selected_profile")
                        or _routing_get(span_row, "chosen_agent")
                    ),
                    "routing_confidence": (
                        output.get("confidence")
                        if output.get("confidence") is not None
                        else _routing_get(span_row, "confidence")
                    ),
                    "output": output,
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


# Back-compat alias from when the storage was routing-only.
RoutingAnnotationStorage = AnnotationStorage
