"""
Routing-specific evaluator for analyzing routing decisions.

This evaluator processes Phoenix spans containing routing decisions and calculates
metrics specific to routing quality, separate from search or generation quality.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import phoenix as px

logger = logging.getLogger(__name__)


class RoutingOutcome(Enum):
    """Classification of routing decision outcomes"""

    SUCCESS = "success"  # Agent completed task successfully
    FAILURE = "failure"  # Agent failed, timed out, or returned empty
    AMBIGUOUS = "ambiguous"  # Needs human annotation


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

    Processes Phoenix spans with cogniverse.routing child spans to calculate
    routing-specific metrics like accuracy, confidence calibration, and latency.
    """

    def __init__(self, phoenix_client: Optional[px.Client] = None):
        """
        Initialize routing evaluator.

        Args:
            phoenix_client: Phoenix client for querying spans. If None, creates new client.
        """
        self.client = phoenix_client or px.Client()
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
        # Extract routing span attributes
        attributes = span_data.get("attributes", {})

        # Validate this is a routing span
        span_name = span_data.get("name", "")
        if span_name != "cogniverse.routing":
            raise ValueError(f"Expected cogniverse.routing span, got: {span_name}")

        # Extract routing decision details
        chosen_agent = attributes.get("routing.chosen_agent")
        confidence = attributes.get("routing.confidence")
        latency_ms = attributes.get("routing.processing_time", 0.0)

        if not chosen_agent or confidence is None:
            raise ValueError(
                "Routing span missing required attributes: routing.chosen_agent or routing.confidence"
            )

        # Determine outcome by looking at downstream agent spans
        outcome, downstream_status = self._classify_routing_outcome(span_data)

        metrics = {
            "chosen_agent": chosen_agent,
            "confidence": float(confidence),
            "latency_ms": float(latency_ms),
            "success": outcome == RoutingOutcome.SUCCESS,
            "downstream_status": downstream_status,
        }

        return outcome, metrics

    def _classify_routing_outcome(
        self, span_data: Dict[str, Any]
    ) -> Tuple[RoutingOutcome, str]:
        """
        Classify routing outcome based on downstream agent execution.

        Examines the parent span (cogniverse.request) to find downstream agent
        spans and determine if the routed agent succeeded.

        Args:
            span_data: Routing span data

        Returns:
            Tuple of (RoutingOutcome, status_description)
        """
        # Get parent span context to access downstream agent spans
        parent_span_id = span_data.get("parent_id")
        if not parent_span_id:
            return RoutingOutcome.AMBIGUOUS, "no_parent_span"

        # Check span status
        status_code = span_data.get("status_code", "OK")
        if status_code == "ERROR":
            return RoutingOutcome.FAILURE, "routing_error"

        # Look for downstream agent execution indicators in attributes
        attributes = span_data.get("attributes", {})
        chosen_agent = attributes.get("routing.chosen_agent")

        # Check if there are any error indicators
        if "error" in span_data.get("events", []):
            return RoutingOutcome.FAILURE, "downstream_error"

        # If routing completed and no errors, consider it successful
        # More sophisticated logic can be added to check actual agent results
        if status_code == "OK" and chosen_agent:
            return RoutingOutcome.SUCCESS, "completed_successfully"

        return RoutingOutcome.AMBIGUOUS, "unclear_outcome"

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

        confidences = [metrics["confidence"] for _, metrics in evaluations]
        successes = [
            1.0 if outcome == RoutingOutcome.SUCCESS else 0.0
            for outcome, _ in evaluations
        ]

        # Calculate Pearson correlation
        df = pd.DataFrame({"confidence": confidences, "success": successes})
        correlation = df["confidence"].corr(df["success"])

        return float(correlation) if not pd.isna(correlation) else 0.0

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

    def query_routing_spans(
        self,
        project_name: str = "cogniverse",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query Phoenix for routing spans within a time range.

        Args:
            project_name: Phoenix project name (not currently used, for future multi-project support)
            start_time: Start of time range (None for no limit)
            end_time: End of time range (None for no limit)
            limit: Maximum number of spans to return

        Returns:
            List of routing span dictionaries

        Raises:
            RuntimeError: If Phoenix query fails
        """
        try:
            # Get all spans using Phoenix client API
            spans_df = self.client.get_spans_dataframe(
                start_time=start_time,
                end_time=end_time
            )

            if spans_df is None or spans_df.empty:
                return []

            # Filter to only routing spans
            routing_spans_df = spans_df[spans_df['name'] == 'cogniverse.routing']

            if routing_spans_df.empty:
                return []

            # Sort by start time (most recent first) and limit
            routing_spans_df = routing_spans_df.sort_values('start_time', ascending=False)
            if limit:
                routing_spans_df = routing_spans_df.head(limit)

            # Convert DataFrame to list of dicts
            return routing_spans_df.to_dict("records")

        except Exception as e:
            raise RuntimeError(
                f"Failed to query routing spans from Phoenix: {e}"
            ) from e
