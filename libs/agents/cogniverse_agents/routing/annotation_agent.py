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

from cogniverse_agents._confidence import parse_confidence
from cogniverse_evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingOutcome,
)
from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

if TYPE_CHECKING:
    from cogniverse_agents.routing.config import AutomationRulesConfig
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class AnnotationPriority(Enum):
    """Priority levels for annotation requests"""

    HIGH = "high"  # Critical failures or very low confidence
    MEDIUM = "medium"  # Ambiguous cases
    LOW = "low"  # Edge cases for training data diversity


class AnnotationStatus(Enum):
    """Status of an annotation request in the queue"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    EXPIRED = "expired"


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
    status: AnnotationStatus = AnnotationStatus.PENDING
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    sla_deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # The human reviewer's annotation label, captured on completion.
    label: Optional[str] = None
    # Which agent's decision this request reviews.
    agent_type: str = "routing"
    # Tenant whose telemetry the span (and the persisted annotation) belong to.
    tenant_id: Optional[str] = None

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
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "sla_deadline": self.sla_deadline.isoformat()
            if self.sla_deadline
            else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "label": self.label,
            "agent_type": self.agent_type,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AnnotationRequest":
        """Rebuild a request from its ``to_dict`` shape (REST enqueue payload)."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        return cls(
            span_id=data["span_id"],
            timestamp=timestamp or datetime.now(timezone.utc),
            query=data.get("query", ""),
            chosen_agent=data.get("chosen_agent", ""),
            routing_confidence=float(data.get("routing_confidence", 0.0)),
            outcome=RoutingOutcome(data.get("outcome", "ambiguous")),
            priority=AnnotationPriority(data.get("priority", "medium")),
            reason=data.get("reason", ""),
            context=data.get("context") or {},
            label=data.get("label"),
            agent_type=data.get("agent_type", "routing"),
            tenant_id=data.get("tenant_id"),
        )


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
        tenant_id: str,
        confidence_threshold: float = 0.6,
        failure_lookback_hours: int = 24,
        max_annotations_per_run: int = 50,
        automation_rules: "AutomationRulesConfig | None" = None,
    ):
        """
        Initialize annotation agent

        Args:
            tenant_id: Tenant to analyze
            confidence_threshold: Confidence below this needs review
            failure_lookback_hours: How far back to look for failures
            max_annotations_per_run: Maximum annotations to request per run
            automation_rules: Optional declarative config (overrides individual kwargs)
        """
        if automation_rules is not None:
            thresholds = automation_rules.annotation_thresholds
            self.confidence_threshold = thresholds.confidence_threshold
            self.very_low_confidence = thresholds.very_low_confidence
            self.boundary_low = thresholds.boundary_low
            self.boundary_high = thresholds.boundary_high
            self.failure_lookback_hours = thresholds.failure_lookback_hours
            self.max_annotations_per_run = thresholds.max_annotations_per_run
        else:
            self.confidence_threshold = confidence_threshold
            self.very_low_confidence = 0.3
            self.boundary_low = 0.6
            self.boundary_high = 0.75
            self.failure_lookback_hours = failure_lookback_hours
            self.max_annotations_per_run = max_annotations_per_run

        telemetry_manager = get_telemetry_manager()
        self.telemetry_config = telemetry_manager.config
        self.project_name = self.telemetry_config.get_project_name(tenant_id)
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=tenant_id
        )
        self.evaluator = RoutingEvaluator(
            provider=self.provider, project_name=self.project_name
        )

        logger.info(
            f"🎯 Initialized AnnotationAgent for tenant '{tenant_id}' "
            f"(confidence_threshold: {confidence_threshold}, "
            f"project: {self.project_name})"
        )

    async def identify_spans_needing_annotation(
        self,
        lookback_hours: Optional[int] = None,
        agent_type: str = "routing",
    ) -> List[AnnotationRequest]:
        """
        Identify an agent type's spans that need human annotation

        Args:
            lookback_hours: How far back to look (defaults to failure_lookback_hours)
            agent_type: Which agent's spans to review (evaluator-registry key)

        Returns:
            List of AnnotationRequest objects prioritized by importance
        """
        from cogniverse_evaluation.evaluators.agent_evaluators import (
            get_agent_evaluator,
        )

        lookback_hours = lookback_hours or self.failure_lookback_hours
        entry = get_agent_evaluator(agent_type)
        span_name = entry.span_name if entry else SPAN_NAME_ROUTING

        logger.info(
            f"🔍 Identifying {agent_type} spans needing annotation "
            f"(lookback: {lookback_hours}h, max: {self.max_annotations_per_run})"
        )

        # Query the agent's spans from the telemetry provider
        # Use UTC timezone-aware datetime to avoid timezone confusion
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=lookback_hours)

        spans_df = await self.provider.traces.get_spans(
            project=self.project_name,
            start_time=start_time,
            end_time=end_time,
            filters={"name": span_name},
            limit=10000,
        )

        if spans_df.empty:
            logger.info(f"📭 No spans found in project {self.project_name}")
            return []

        agent_spans_df = spans_df[spans_df["name"] == span_name]

        if agent_spans_df.empty:
            logger.info(f"📭 No {agent_type} spans found")
            return []

        logger.info(f"📊 Found {len(agent_spans_df)} {agent_type} spans to analyze")

        annotation_requests = []

        for _, span_row in agent_spans_df.iterrows():
            annotation_request = self._analyze_span_for_annotation(
                span_row, agent_type=agent_type
            )
            if annotation_request:
                annotation_requests.append(annotation_request)

        annotation_requests = self._prioritize_requests(annotation_requests)
        annotation_requests = annotation_requests[: self.max_annotations_per_run]

        logger.info(
            f"✅ Identified {len(annotation_requests)} spans needing annotation "
            f"(HIGH: {sum(1 for r in annotation_requests if r.priority == AnnotationPriority.HIGH)}, "
            f"MEDIUM: {sum(1 for r in annotation_requests if r.priority == AnnotationPriority.MEDIUM)}, "
            f"LOW: {sum(1 for r in annotation_requests if r.priority == AnnotationPriority.LOW)})"
        )

        return annotation_requests

    def _analyze_span_for_annotation(
        self, span_row, agent_type: str = "routing"
    ) -> Optional[AnnotationRequest]:
        """
        Analyze a single span to determine if it needs annotation

        Args:
            span_row: Pandas Series with span data
            agent_type: Which agent's decision the span records

        Returns:
            AnnotationRequest if span needs annotation, None otherwise
        """
        from cogniverse_foundation.telemetry.span_contract import read_span_io

        # Canonical slots first (what the emitters actually write); the legacy
        # attributes.routing dict is a fallback for pre-migration spans — the
        # gateway never set it, so reading only the legacy attribute made this
        # agent identify zero spans on real data.
        io = read_span_io(span_row)
        output = io["output"] if isinstance(io.get("output"), dict) else {}
        routing_attrs = span_row.get("attributes.routing")
        if not isinstance(routing_attrs, dict):
            routing_attrs = {}

        chosen_agent = (
            output.get("chosen_agent")
            or output.get("selected_profile")
            or routing_attrs.get("chosen_agent")
        )
        confidence = output.get("confidence")
        if confidence is None:
            confidence = routing_attrs.get("confidence")
        query = io.get("input") or routing_attrs.get("query")
        context = routing_attrs.get("context", {})

        if confidence is None or not query:
            return None
        if agent_type in ("routing", "gateway") and not chosen_agent:
            return None

        confidence = parse_confidence(confidence)
        span_id = span_row.get("context.span_id", "")
        timestamp = span_row.get("start_time")
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        if agent_type in ("routing", "gateway"):
            outcome, outcome_details = self.evaluator._classify_routing_outcome(
                span_row
            )
        else:
            # Non-routing agents have no downstream-agent handoff to classify;
            # the span's own status is the outcome signal.
            status_code = span_row.get("status_code", "OK")
            if status_code == "ERROR":
                outcome, outcome_details = RoutingOutcome.FAILURE, "span_error"
            elif status_code == "OK":
                outcome, outcome_details = (
                    RoutingOutcome.SUCCESS,
                    "completed_successfully",
                )
            else:
                outcome, outcome_details = RoutingOutcome.AMBIGUOUS, "unclear_outcome"

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
            chosen_agent=chosen_agent or "",
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
            agent_type=agent_type,
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
        if confidence < self.very_low_confidence:
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
        if (
            outcome == RoutingOutcome.SUCCESS
            and self.boundary_low <= confidence <= self.boundary_high
        ):
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
