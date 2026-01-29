"""
Annotation Agent for Routing Optimization

Identifies routing decisions that need human review based on:
- Low confidence scores
- Ambiguous outcomes from RoutingEvaluator
- Downstream execution failures
- Edge cases where optimizer is uncertain
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from cogniverse_evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingOutcome,
)
from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class AnnotationPriority(Enum):
    """Priority levels for annotation requests"""

    HIGH = "high"  # Critical failures or very low confidence
    MEDIUM = "medium"  # Ambiguous cases
    LOW = "low"  # Edge cases for training data diversity


@dataclass
class AnnotationRequest:
    """Request for human annotation on a routing decision"""

    span_id: str
    timestamp: datetime
    query: str
    chosen_agent: str
    routing_confidence: float
    outcome: RoutingOutcome
    priority: AnnotationPriority
    reason: str
    context: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "span_id": self.span_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "chosen_agent": self.chosen_agent,
            "routing_confidence": self.routing_confidence,
            "outcome": self.outcome.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "context": self.context,
        }


class AnnotationAgent:
    """
    Agent that identifies routing spans needing human annotation

    This agent:
    1. Queries Phoenix for recent routing spans
    2. Evaluates each span using RoutingEvaluator
    3. Identifies spans that need human review based on configurable criteria
    4. Prioritizes annotation requests
    5. Returns queue of spans needing annotation
    """

    def __init__(
        self,
        tenant_id: str = "default",
        confidence_threshold: float = 0.6,
        failure_lookback_hours: int = 24,
        max_annotations_per_run: int = 50,
    ):
        """
        Initialize annotation agent

        Args:
            tenant_id: Tenant to analyze
            confidence_threshold: Confidence below this needs review
            failure_lookback_hours: How far back to look for failures
            max_annotations_per_run: Maximum annotations to request per run
        """
        self.tenant_id = tenant_id
        self.confidence_threshold = confidence_threshold
        self.failure_lookback_hours = failure_lookback_hours
        self.max_annotations_per_run = max_annotations_per_run

        # Initialize components - use shared telemetry manager config
        telemetry_manager = get_telemetry_manager()
        self.telemetry_config = telemetry_manager.config

        # Get project name for unified tenant project (routing operations)
        self.project_name = self.telemetry_config.get_project_name(tenant_id)

        # Get telemetry provider for querying spans
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=tenant_id
        )

        # Initialize evaluator for outcome classification (using provider, not Phoenix)
        self.evaluator = RoutingEvaluator(
            provider=self.provider, project_name=self.project_name
        )

        logger.info(
            f"🎯 Initialized AnnotationAgent for tenant '{tenant_id}' "
            f"(confidence_threshold: {confidence_threshold}, "
            f"project: {self.project_name})"
        )

    async def identify_spans_needing_annotation(
        self, lookback_hours: Optional[int] = None
    ) -> List[AnnotationRequest]:
        """
        Identify routing spans that need human annotation

        Args:
            lookback_hours: How far back to look (defaults to failure_lookback_hours)

        Returns:
            List of AnnotationRequest objects prioritized by importance
        """
        lookback_hours = lookback_hours or self.failure_lookback_hours

        logger.info(
            f"🔍 Identifying spans needing annotation "
            f"(lookback: {lookback_hours}h, max: {self.max_annotations_per_run})"
        )

        # Query routing spans from telemetry provider
        # Use UTC timezone-aware datetime to avoid timezone confusion
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=lookback_hours)

        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                limit=10000,
            )
        except Exception as e:
            logger.error(f"❌ Error querying spans: {e}")
            return []

        if spans_df.empty:
            logger.info(f"📭 No spans found in project {self.project_name}")
            return []

        # Filter for routing spans
        routing_spans_df = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]

        if routing_spans_df.empty:
            logger.info("📭 No routing spans found")
            return []

        logger.info(f"📊 Found {len(routing_spans_df)} routing spans to analyze")

        # Analyze each span and collect annotation requests
        annotation_requests = []

        for _, span_row in routing_spans_df.iterrows():
            try:
                annotation_request = self._analyze_span_for_annotation(span_row)
                if annotation_request:
                    annotation_requests.append(annotation_request)
            except Exception as e:
                logger.error(f"❌ Error analyzing span: {e}")
                continue

        # Prioritize and limit
        annotation_requests = self._prioritize_requests(annotation_requests)
        annotation_requests = annotation_requests[: self.max_annotations_per_run]

        logger.info(
            f"✅ Identified {len(annotation_requests)} spans needing annotation "
            f"(HIGH: {sum(1 for r in annotation_requests if r.priority == AnnotationPriority.HIGH)}, "
            f"MEDIUM: {sum(1 for r in annotation_requests if r.priority == AnnotationPriority.MEDIUM)}, "
            f"LOW: {sum(1 for r in annotation_requests if r.priority == AnnotationPriority.LOW)})"
        )

        return annotation_requests

    def _analyze_span_for_annotation(self, span_row) -> Optional[AnnotationRequest]:
        """
        Analyze a single span to determine if it needs annotation

        Args:
            span_row: Pandas Series with span data

        Returns:
            AnnotationRequest if span needs annotation, None otherwise
        """
        # Extract routing attributes
        routing_attrs = span_row.get("attributes.routing")
        if not routing_attrs or not isinstance(routing_attrs, dict):
            return None

        chosen_agent = routing_attrs.get("chosen_agent")
        confidence = routing_attrs.get("confidence")
        query = routing_attrs.get("query")
        context = routing_attrs.get("context", {})

        if not chosen_agent or confidence is None or not query:
            return None

        confidence = float(confidence)
        span_id = span_row.get("context.span_id", "")
        timestamp = span_row.get("start_time")
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        # Evaluate outcome using RoutingEvaluator
        try:
            outcome, outcome_details = self.evaluator._classify_routing_outcome(
                span_row
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to classify outcome: {e}")
            outcome = RoutingOutcome.AMBIGUOUS
            outcome_details = {"reason": "evaluation_error"}

        # Determine if annotation needed and priority
        needs_annotation, priority, reason = self._needs_annotation(
            confidence=confidence,
            outcome=outcome,
            outcome_details=outcome_details,
            span_row=span_row,
        )

        if not needs_annotation:
            return None

        return AnnotationRequest(
            span_id=span_id,
            timestamp=timestamp,
            query=query,
            chosen_agent=chosen_agent,
            routing_confidence=confidence,
            outcome=outcome,
            priority=priority,
            reason=reason,
            context={
                "routing_context": context,
                "outcome_details": outcome_details,
                "span_status": span_row.get("status"),
                "latency_ms": span_row.get("latency_ms", 0),
            },
        )

    def _needs_annotation(
        self,
        confidence: float,
        outcome: RoutingOutcome,
        outcome_details: Dict,
        span_row,
    ) -> Tuple[bool, AnnotationPriority, str]:
        """
        Determine if a span needs annotation and its priority

        Returns:
            Tuple of (needs_annotation, priority, reason)
        """
        # HIGH priority: Clear failures with low confidence
        if outcome == RoutingOutcome.FAILURE:
            if confidence < self.confidence_threshold:
                return (
                    True,
                    AnnotationPriority.HIGH,
                    f"Failure with low confidence ({confidence:.2f})",
                )
            else:
                return (
                    True,
                    AnnotationPriority.MEDIUM,
                    f"Failure despite high confidence ({confidence:.2f})",
                )

        # HIGH priority: Very low confidence regardless of outcome
        if confidence < 0.3:
            return (
                True,
                AnnotationPriority.HIGH,
                f"Very low confidence ({confidence:.2f})",
            )

        # MEDIUM priority: Ambiguous outcomes
        if outcome == RoutingOutcome.AMBIGUOUS:
            return (
                True,
                AnnotationPriority.MEDIUM,
                "Ambiguous outcome - unclear if routing was correct",
            )

        # MEDIUM priority: Low confidence with success (potential false positives)
        if outcome == RoutingOutcome.SUCCESS and confidence < self.confidence_threshold:
            return (
                True,
                AnnotationPriority.MEDIUM,
                f"Success but low confidence ({confidence:.2f}) - verify correctness",
            )

        # LOW priority: Edge cases for training diversity
        # (e.g., high confidence successes near decision boundaries)
        if outcome == RoutingOutcome.SUCCESS and 0.6 <= confidence <= 0.75:
            return (
                True,
                AnnotationPriority.LOW,
                f"Near decision boundary ({confidence:.2f}) - training data diversity",
            )

        # No annotation needed
        return False, AnnotationPriority.LOW, ""

    def _prioritize_requests(
        self, requests: List[AnnotationRequest]
    ) -> List[AnnotationRequest]:
        """
        Sort annotation requests by priority

        Args:
            requests: List of annotation requests

        Returns:
            Sorted list (HIGH first, then MEDIUM, then LOW)
        """
        priority_order = {
            AnnotationPriority.HIGH: 0,
            AnnotationPriority.MEDIUM: 1,
            AnnotationPriority.LOW: 2,
        }

        return sorted(
            requests,
            key=lambda r: (priority_order[r.priority], r.timestamp),
            reverse=False,  # Earlier timestamps first within same priority
        )

    def get_annotation_statistics(self, requests: List[AnnotationRequest]) -> Dict:
        """
        Get statistics about annotation requests

        Args:
            requests: List of annotation requests

        Returns:
            Dictionary with statistics
        """
        if not requests:
            return {
                "total": 0,
                "by_priority": {},
                "by_outcome": {},
                "by_agent": {},
                "avg_confidence": 0.0,
            }

        by_priority = {}
        for priority in AnnotationPriority:
            count = sum(1 for r in requests if r.priority == priority)
            by_priority[priority.value] = count

        by_outcome = {}
        for outcome in RoutingOutcome:
            count = sum(1 for r in requests if r.outcome == outcome)
            by_outcome[outcome.value] = count

        by_agent = {}
        for request in requests:
            agent = request.chosen_agent
            by_agent[agent] = by_agent.get(agent, 0) + 1

        avg_confidence = sum(r.routing_confidence for r in requests) / len(requests)

        return {
            "total": len(requests),
            "by_priority": by_priority,
            "by_outcome": by_outcome,
            "by_agent": by_agent,
            "avg_confidence": avg_confidence,
        }
