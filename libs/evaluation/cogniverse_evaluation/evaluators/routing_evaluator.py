"""
Routing-specific evaluator for analyzing routing decisions.

This evaluator processes telemetry spans containing routing decisions and calculates
metrics specific to routing quality, separate from search or generation quality.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from cogniverse_foundation.confidence import parse_confidence

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


def _coerce_latency(value: object) -> float:
    """Coerce a span's processing_time to milliseconds, defaulting to 0.0 for
    a missing/non-numeric value rather than raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class RoutingOutcome(Enum):
    """Classification of routing decision outcomes"""

    SUCCESS = "success"  # Agent completed task successfully
    FAILURE = "failure"  # Agent failed, timed out, or returned empty
    AMBIGUOUS = "ambiguous"  # Needs human annotation


def classify_routing_outcome(
    span_data: Dict[str, Any],
) -> Tuple[RoutingOutcome, str]:
    """Classify a routing span's outcome from its own data.

    Pure function (no provider access) so structural evaluators can share it
    without constructing a ``RoutingEvaluator``.
    """
    parent_span_id = span_data.get("parent_id")
    if not isinstance(parent_span_id, str) or not parent_span_id:
        return RoutingOutcome.AMBIGUOUS, "no_parent_span"

    status_code = span_data.get("status_code", "OK")
    if status_code == "ERROR":
        return RoutingOutcome.FAILURE, "routing_error"

    # Look for downstream agent execution indicators. The routing decision
    # is on the canonical output.value; fall back to the legacy attribute.
    from cogniverse_foundation.telemetry.span_contract import read_span_io

    output = read_span_io(span_data)["output"]
    attributes = span_data.get("attributes", {})
    chosen_agent = (
        output.get("chosen_agent") if isinstance(output, dict) else None
    ) or attributes.get("routing.chosen_agent")

    if "error" in span_data.get("events", []):
        return RoutingOutcome.FAILURE, "downstream_error"

    # A span that ends without setting a status is UNSET; only ERROR fails.
    if chosen_agent:
        return RoutingOutcome.SUCCESS, "completed_successfully"

    return RoutingOutcome.AMBIGUOUS, "unclear_outcome"


def evaluate_routing_span(
    span_data: Dict[str, Any],
) -> Tuple[RoutingOutcome, Dict[str, Any]]:
    """Extract and evaluate one routing decision from a routing span's data.

    Returns ``(outcome, metrics)``; ``metrics`` holds ``chosen_agent``,
    ``confidence``, ``latency_ms`` (the decision's recorded
    ``processing_time``), ``success`` and ``downstream_status``. Raises
    ``ValueError`` when the span is not a routing span or records no chosen
    agent or confidence.
    """
    # Validate this is a routing span
    span_name = span_data.get("name", "")
    if span_name != "cogniverse.routing":
        raise ValueError(f"Expected cogniverse.routing span, got: {span_name}")

    # Extract routing decision details - handle both Phoenix flattened and nested formats
    # Phoenix format: attributes.routing = {"chosen_agent": ..., "confidence": ...}
    # Unit test format: attributes = {"routing.chosen_agent": ..., "routing.confidence": ...}

    chosen_agent = None
    confidence = None
    latency_ms = 0.0

    # Canonical span contract: the routing decision is on output.value.
    from cogniverse_foundation.telemetry.span_contract import read_span_io

    output = read_span_io(span_data)["output"]
    if isinstance(output, dict):
        chosen_agent = output.get("chosen_agent") or output.get("recommended_agent")
        confidence = output.get("confidence")
        latency_ms = output.get("processing_time", 0.0)

    # Try Phoenix flattened format first (attributes.routing.*)
    if (
        (not chosen_agent or confidence is None)
        and "attributes.routing" in span_data
        and isinstance(span_data["attributes.routing"], dict)
    ):
        routing_attrs = span_data["attributes.routing"]
        chosen_agent = routing_attrs.get("chosen_agent") or routing_attrs.get(
            "recommended_agent"
        )
        confidence = routing_attrs.get("confidence")
        latency_ms = routing_attrs.get("processing_time", 0.0)

    # Try nested format (attributes with routing.* keys)
    if not chosen_agent or confidence is None:
        attributes = span_data.get("attributes", {})
        chosen_agent = (
            chosen_agent
            or attributes.get("routing.chosen_agent")
            or attributes.get("routing.recommended_agent")
        )
        confidence = (
            confidence
            if confidence is not None
            else attributes.get("routing.confidence")
        )
        latency_ms = latency_ms or attributes.get("routing.processing_time", 0.0)

    if not chosen_agent or confidence is None:
        raise ValueError(
            "Routing span missing required attributes: routing.chosen_agent or routing.confidence"
        )

    # Determine outcome by looking at downstream agent spans
    outcome, downstream_status = classify_routing_outcome(span_data)

    metrics = {
        "chosen_agent": chosen_agent,
        # Routers emit floats, labels ("high") or percents ("85%") —
        # parse_confidence maps them all into [0, 1].
        "confidence": parse_confidence(confidence),
        # Coerce defensively: a None/list processing_time must not raise a
        # TypeError that aborts the whole calculate_metrics batch.
        "latency_ms": _coerce_latency(latency_ms),
        "success": outcome == RoutingOutcome.SUCCESS,
        "downstream_status": downstream_status,
    }

    return outcome, metrics


def _confidence_success_correlation(
    confidences: List[float], successes: List[bool]
) -> Optional[float]:
    """Pearson correlation of confidence with success; ``None`` when it is
    undefined (fewer than two decisions, or either side constant)."""
    if len(set(confidences)) < 2 or len(set(successes)) < 2:
        return None
    correlation = pd.Series(confidences).corr(
        pd.Series([1.0 if success else 0.0 for success in successes])
    )
    return None if pd.isna(correlation) else float(correlation)


@dataclass
class RoutingMetrics:
    """Container for routing evaluation metrics"""

    routing_accuracy: float  # Percentage of successful routing decisions
    confidence_calibration: float  # Correlation between confidence and success
    avg_routing_latency: float  # Average time to make routing decision (ms)
    per_agent_precision: Dict[str, float]  # Precision per agent type
    per_agent_recall: Dict[str, float]  # Recall per agent type
    per_agent_f1: Dict[str, float]  # F1 score per agent type
    total_decisions: int  # Total routing decisions evaluated
    ambiguous_count: int  # Number of decisions needing human review


class RoutingEvaluator:
    """
    Evaluate routing decisions separately from search quality.

    Processes telemetry spans with cogniverse.routing child spans to calculate
    routing-specific metrics like accuracy, confidence calibration, and latency.
    """

    def __init__(
        self,
        provider: "TelemetryProvider",
        *,
        project_name: str,
    ):
        """
        Initialize routing evaluator.

        Args:
            provider: Telemetry provider for querying spans
            project_name: Project name for routing optimization
        """
        if not project_name:
            raise ValueError("project_name is required")

        self.provider = provider
        self.project_name = project_name
        self.logger = logging.getLogger(__name__)

    def evaluate_routing_decision(
        self, span_data: Dict[str, Any]
    ) -> Tuple[RoutingOutcome, Dict[str, Any]]:
        """
        Extract and evaluate a single routing decision from span data.

        Args:
            span_data: Dictionary containing span information including attributes and child spans

        Returns:
            Tuple of (RoutingOutcome, metrics_dict) where metrics_dict contains:
                - chosen_agent: str
                - confidence: float
                - latency_ms: float
                - success: bool
                - downstream_status: str

        Raises:
            ValueError: If span_data doesn't contain required routing information
        """
        return evaluate_routing_span(span_data)

    def _classify_routing_outcome(
        self, span_data: Dict[str, Any]
    ) -> Tuple[RoutingOutcome, str]:
        """
        Classify routing outcome based on downstream agent execution.

        Args:
            span_data: Routing span data

        Returns:
            Tuple of (RoutingOutcome, status_description)
        """
        return classify_routing_outcome(span_data)

    def calculate_metrics(self, routing_spans: List[Dict[str, Any]]) -> RoutingMetrics:
        """
        Calculate comprehensive routing metrics from a collection of routing spans.

        Args:
            routing_spans: List of routing span dictionaries

        Returns:
            RoutingMetrics object with calculated metrics

        Raises:
            ValueError: If routing_spans is empty
        """
        if not routing_spans:
            raise ValueError("Cannot calculate metrics from empty routing_spans list")

        # Collect evaluation results
        evaluations = []
        for span in routing_spans:
            try:
                outcome, metrics = self.evaluate_routing_decision(span)
                evaluations.append((outcome, metrics))
            except ValueError as e:
                self.logger.warning(f"Skipping invalid span: {e}")
                continue

        if not evaluations:
            raise ValueError("No valid routing spans found in input")

        # Calculate overall metrics
        total_decisions = len(evaluations)
        successful = sum(
            1 for outcome, _ in evaluations if outcome == RoutingOutcome.SUCCESS
        )
        ambiguous = sum(
            1 for outcome, _ in evaluations if outcome == RoutingOutcome.AMBIGUOUS
        )

        routing_accuracy = successful / total_decisions if total_decisions > 0 else 0.0

        # Calculate confidence calibration (correlation between confidence and success)
        confidence_calibration = self._calculate_confidence_calibration(evaluations)

        # Calculate average latency
        latencies = [metrics["latency_ms"] for _, metrics in evaluations]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate per-agent metrics
        per_agent_precision, per_agent_recall, per_agent_f1 = (
            self._calculate_per_agent_metrics(evaluations)
        )

        return RoutingMetrics(
            routing_accuracy=routing_accuracy,
            confidence_calibration=confidence_calibration,
            avg_routing_latency=avg_latency,
            per_agent_precision=per_agent_precision,
            per_agent_recall=per_agent_recall,
            per_agent_f1=per_agent_f1,
            total_decisions=total_decisions,
            ambiguous_count=ambiguous,
        )

    def _calculate_confidence_calibration(
        self, evaluations: List[Tuple[RoutingOutcome, Dict[str, Any]]]
    ) -> float:
        """
        Calculate how well confidence scores predict actual success.

        Uses Pearson correlation between confidence scores and success outcomes.

        Args:
            evaluations: List of (outcome, metrics) tuples

        Returns:
            Correlation coefficient between -1 and 1
        """
        if len(evaluations) < 2:
            return 0.0

        correlation = _confidence_success_correlation(
            [metrics["confidence"] for _, metrics in evaluations],
            [outcome == RoutingOutcome.SUCCESS for outcome, _ in evaluations],
        )
        return 0.0 if correlation is None else correlation

    def _calculate_per_agent_metrics(
        self, evaluations: List[Tuple[RoutingOutcome, Dict[str, Any]]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Calculate precision, recall, and F1 score for each agent type.

        Args:
            evaluations: List of (outcome, metrics) tuples

        Returns:
            Tuple of (precision_dict, recall_dict, f1_dict) for each agent
        """
        # Group by agent
        agent_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for outcome, metrics in evaluations:
            agent = metrics["chosen_agent"]
            success = outcome == RoutingOutcome.SUCCESS

            if success:
                agent_stats[agent]["tp"] += 1
            else:
                agent_stats[agent]["fp"] += 1
                # Note: FN (false negatives) would require ground truth of what agent
                # *should* have been chosen. For now, we only track TP and FP.

        # Calculate precision for each agent
        precision = {}
        recall = {}
        f1 = {}

        for agent, stats in agent_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            # Precision: TP / (TP + FP)
            precision[agent] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall: TP / (TP + FN) - without ground truth, this is limited
            # For now, we can only calculate this if we have FN data
            recall[agent] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1 score
            prec = precision[agent]
            rec = recall[agent]
            f1[agent] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return precision, recall, f1

    async def query_routing_spans(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query telemetry provider for routing spans within a time range from the routing optimization project.

        Args:
            start_time: Start of time range (None for no limit)
            end_time: End of time range (None for no limit)
            limit: Maximum number of spans to return

        Returns:
            List of routing span dictionaries

        Raises:
            RuntimeError: If telemetry query fails
        """
        try:
            # Query only routing spans — the name predicate runs server-side,
            # so the limit budget isn't consumed by unrelated span types and
            # only routing rows cross the wire.
            spans_df = await self.provider.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                filters={"name": "cogniverse.routing"},
                limit=limit,
            )

            if spans_df is None or spans_df.empty:
                return []

            routing_spans_df = spans_df[spans_df["name"] == "cogniverse.routing"]

            if routing_spans_df.empty:
                return []

            # Sort by start time (most recent first) and limit
            routing_spans_df = routing_spans_df.sort_values(
                "start_time", ascending=False
            )
            if limit:
                routing_spans_df = routing_spans_df.head(limit)

            # Convert DataFrame to list of dicts
            # evaluate_routing_decision() will handle both flattened and nested formats
            return routing_spans_df.to_dict("records")

        except Exception as e:
            raise RuntimeError(
                f"Failed to query routing spans from telemetry provider: {e}"
            ) from e


def summarize_routing_decisions(spans: pd.DataFrame) -> Dict[str, Any]:
    """The routing decisions in ``spans`` (a frame of ``cogniverse.routing``
    spans) and their aggregate quality.

    Each decision is read with ``evaluate_routing_span``; a span it cannot
    read counts in ``unreadable``. Returns ``decisions`` (newest first:
    ``span_id``, ``trace_id``, ``start_time``, ``query``, ``chosen_agent``,
    ``confidence``, ``outcome``, ``reason``, ``latency_ms`` (the span's
    duration) and ``entity_extraction_failed``), the outcome counts,
    ``accuracy`` (the share that succeeded), ``confidence_calibration`` (the
    correlation of confidence with success, ``None`` when undefined),
    ``latency_ms`` (``mean``, ``p50``, ``p95``) and ``per_agent`` (most
    decisions first).
    """
    from cogniverse_foundation.telemetry.span_contract import read_span_io

    decisions = []
    unreadable = 0
    for span in spans.to_dict("records"):
        try:
            outcome, metrics = evaluate_routing_span(span)
        except ValueError:
            unreadable += 1
            continue
        io = read_span_io(span)
        output = io["output"] if isinstance(io["output"], dict) else {}
        start = pd.Timestamp(span["start_time"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        duration = pd.Timestamp(span["end_time"]) - pd.Timestamp(span["start_time"])
        decisions.append(
            {
                "span_id": span.get("context.span_id"),
                "trace_id": span.get("context.trace_id"),
                "start_time": start.tz_convert("UTC").isoformat(),
                "query": io["input"],
                "chosen_agent": str(metrics["chosen_agent"]),
                "confidence": metrics["confidence"],
                "outcome": outcome.value,
                "reason": metrics["downstream_status"],
                "latency_ms": duration.total_seconds() * 1000,
                "entity_extraction_failed": output.get("entity_extraction_failed")
                is True,
            }
        )
    decisions.sort(key=lambda row: row["start_time"], reverse=True)

    def count(rows, outcome):
        return sum(row["outcome"] == outcome.value for row in rows)

    latencies = pd.Series([row["latency_ms"] for row in decisions], dtype=float)
    per_agent = []
    for agent in {row["chosen_agent"] for row in decisions}:
        rows = [row for row in decisions if row["chosen_agent"] == agent]
        per_agent.append(
            {
                "agent": agent,
                "decisions": len(rows),
                "successes": count(rows, RoutingOutcome.SUCCESS),
                "failures": count(rows, RoutingOutcome.FAILURE),
                "ambiguous": count(rows, RoutingOutcome.AMBIGUOUS),
                "success_rate": count(rows, RoutingOutcome.SUCCESS) / len(rows),
                "mean_confidence": sum(row["confidence"] for row in rows) / len(rows),
                "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
            }
        )
    per_agent.sort(key=lambda row: (-row["decisions"], row["agent"]))
    successes = count(decisions, RoutingOutcome.SUCCESS)
    return {
        "decisions": decisions,
        "total": len(decisions),
        "successes": successes,
        "failures": count(decisions, RoutingOutcome.FAILURE),
        "ambiguous": count(decisions, RoutingOutcome.AMBIGUOUS),
        "unreadable": unreadable,
        "accuracy": successes / len(decisions) if decisions else None,
        "confidence_calibration": _confidence_success_correlation(
            [row["confidence"] for row in decisions],
            [row["outcome"] == RoutingOutcome.SUCCESS.value for row in decisions],
        ),
        "latency_ms": {
            "mean": float(latencies.mean()) if decisions else None,
            "p50": float(latencies.quantile(0.5)) if decisions else None,
            "p95": float(latencies.quantile(0.95)) if decisions else None,
        },
        "per_agent": per_agent,
    }
