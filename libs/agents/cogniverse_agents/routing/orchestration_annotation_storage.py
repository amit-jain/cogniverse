"""
Orchestration Annotation Storage

Stores human annotations for orchestration workflow quality and corrections.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cogniverse_core.telemetry.manager import get_telemetry_manager

if TYPE_CHECKING:
    from cogniverse_core.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationAnnotation:
    """
    Human annotation for an orchestration workflow

    Captures human feedback on:
    - Was the orchestration pattern optimal?
    - Were the right agents selected?
    - Was the execution order correct?
    - Overall workflow quality
    - Suggested improvements
    """

    # Workflow identification (required fields)
    workflow_id: str
    span_id: str
    query: str

    # Actual orchestration details (required fields)
    orchestration_pattern: str  # parallel, sequential, etc.
    agents_used: List[str]
    execution_order: List[str]
    execution_time: float

    # Human feedback - pattern optimality (required fields)
    pattern_is_optimal: bool

    # Human feedback - agent selection (required fields)
    agents_are_correct: bool

    # Human feedback - execution order (required fields)
    execution_order_is_optimal: bool

    # Human feedback - overall quality (required fields)
    workflow_quality_label: str  # excellent, good, acceptable, poor, failed
    quality_score: float  # 0.0-1.0

    # Annotation metadata (required fields)
    annotator_id: str

    # Optional fields with defaults
    suggested_pattern: Optional[str] = None
    pattern_feedback: Optional[str] = None
    missing_agents: List[str] = field(default_factory=list)
    unnecessary_agents: List[str] = field(default_factory=list)
    suggested_agents: List[str] = field(default_factory=list)
    suggested_execution_order: Optional[List[str]] = None
    execution_order_feedback: Optional[str] = None
    improvement_notes: Optional[str] = None
    what_went_well: Optional[str] = None
    what_went_wrong: Optional[str] = None
    annotation_timestamp: datetime = field(default_factory=datetime.now)
    annotation_source: str = "human"  # human, llm_auto, hybrid
    workflow_succeeded: bool = True
    error_details: Optional[str] = None


class OrchestrationAnnotationStorage:
    """
    Storage for orchestration workflow annotations

    Stores annotations in telemetry backend using annotations API.
    """

    def __init__(self, tenant_id: str = "default", project_name: str = "cogniverse"):
        """Initialize annotation storage"""
        self.tenant_id = tenant_id
        self.project_name = project_name

        # Get telemetry provider for annotations
        telemetry_manager = get_telemetry_manager()
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=tenant_id
        )

        logger.info(
            f"🗄️  Initialized OrchestrationAnnotationStorage for tenant '{tenant_id}' (project: {project_name})"
        )

    async def store_annotation(self, annotation: OrchestrationAnnotation) -> bool:
        """
        Store orchestration annotation in telemetry backend

        Args:
            annotation: Orchestration annotation to store

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Build metadata with all annotation details
            metadata = {
                # Pattern feedback
                "workflow_id": annotation.workflow_id,
                "query": annotation.query,
                "pattern_is_optimal": annotation.pattern_is_optimal,
                "suggested_pattern": annotation.suggested_pattern,
                "pattern_feedback": annotation.pattern_feedback,
                # Agent selection feedback
                "agents_are_correct": annotation.agents_are_correct,
                "missing_agents": ",".join(annotation.missing_agents),
                "unnecessary_agents": ",".join(annotation.unnecessary_agents),
                "suggested_agents": ",".join(annotation.suggested_agents),
                # Execution order feedback
                "execution_order_is_optimal": annotation.execution_order_is_optimal,
                "suggested_execution_order": ",".join(
                    annotation.suggested_execution_order or []
                ),
                "execution_order_feedback": annotation.execution_order_feedback,
                # Improvement notes
                "improvement_notes": annotation.improvement_notes,
                "what_went_well": annotation.what_went_well,
                "what_went_wrong": annotation.what_went_wrong,
                # Original workflow data
                "actual_pattern": annotation.orchestration_pattern,
                "actual_agents": ",".join(annotation.agents_used),
                "actual_execution_order": ",".join(annotation.execution_order),
                "execution_time": annotation.execution_time,
                "workflow_succeeded": annotation.workflow_succeeded,
                # Annotator
                "annotator_id": annotation.annotator_id,
                "annotation_source": annotation.annotation_source,
                "annotation_timestamp": annotation.annotation_timestamp.isoformat(),
            }

            # Use provider's annotation API
            await self.provider.annotations.add_annotation(
                span_id=annotation.span_id,
                name="orchestration_quality",
                label=annotation.workflow_quality_label,
                score=annotation.quality_score,
                metadata=metadata,
                project=self.project_name,
            )

            logger.info(
                f"✅ Stored orchestration annotation for workflow {annotation.workflow_id}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error storing annotation: {e}")
            return False

    async def query_annotated_spans(
        self,
        start_time: datetime,
        end_time: datetime,
        only_human_reviewed: bool = True,
        min_quality_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query orchestration spans that have been annotated

        Args:
            start_time: Start of time range
            end_time: End of time range
            only_human_reviewed: Only return human annotations (not LLM auto-annotations)
            min_quality_score: Filter by minimum quality score

        Returns:
            List of annotated span data with annotations
        """
        try:
            # Query telemetry provider for spans with orchestration_quality evaluations
            spans_df = await self.provider.traces.get_spans(
                start_time=start_time,
                end_time=end_time,
                project_name=self.project_name,
            )

            # Filter for orchestration spans with evaluations
            annotated_spans = []

            for _, span_row in spans_df.iterrows():
                # Check if span has orchestration_quality evaluation
                evaluations = await self._get_span_evaluations(span_row)

                if not evaluations:
                    continue

                # Filter by annotation source
                if only_human_reviewed:
                    evaluations = [
                        e for e in evaluations if e.get("annotator_kind") == "human"
                    ]

                # Filter by quality score
                if min_quality_score is not None:
                    evaluations = [
                        e
                        for e in evaluations
                        if e.get("result", {}).get("score", 0.0) >= min_quality_score
                    ]

                if evaluations:
                    annotated_spans.append(
                        {
                            "span_id": span_row.get("context.span_id"),
                            "span_data": span_row.to_dict(),
                            "annotations": evaluations,
                        }
                    )

            logger.info(
                f"📊 Found {len(annotated_spans)} annotated orchestration spans"
            )
            return annotated_spans

        except Exception as e:
            logger.error(f"❌ Error querying annotated spans: {e}")
            return []

    async def _get_span_evaluations(self, span_row) -> List[Dict]:
        """
        Extract evaluations for a span

        Note: This uses Phoenix API to get evaluations attached to spans.
        The actual implementation depends on Phoenix client capabilities.
        """
        try:
            span_id = span_row.get("context.span_id")
            if not span_id:
                return []

            # Get evaluations for this span from provider
            evals_df = await self.provider.evaluations.get_evaluations(span_ids=[span_id])

            if evals_df.empty:
                return []

            # Filter for orchestration_quality evaluations
            orch_evals = evals_df[evals_df["name"] == "orchestration_quality"]

            evaluations = []
            for _, eval_row in orch_evals.iterrows():
                evaluations.append(
                    {
                        "annotator_kind": eval_row.get("annotator_kind", "unknown"),
                        "result": {
                            "label": eval_row.get("label"),
                            "score": eval_row.get("score", 0.0),
                        },
                        "metadata": eval_row.get("metadata", {}),
                    }
                )

            return evaluations

        except Exception as e:
            logger.error(f"Error extracting evaluations for span {span_id}: {e}")
            raise
