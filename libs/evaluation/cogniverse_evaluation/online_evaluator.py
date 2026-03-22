"""
Online Evaluator for real-time routing span scoring.

Evaluates individual routing spans as they are extracted by the
RoutingSpanEvaluator, producing scores that are persisted as telemetry
annotations for drift detection and continuous quality monitoring.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cogniverse_evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingOutcome,
)

if TYPE_CHECKING:
    from cogniverse_agents.routing.config import OnlineEvaluationConfig
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class OnlineEvalResult:
    """Result of evaluating a single span online."""

    span_id: str
    evaluator_name: str
    score: float
    label: str
    explanation: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "evaluator_name": self.evaluator_name,
            "score": self.score,
            "label": self.label,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


class OnlineEvaluator:
    """
    Evaluates routing spans in real-time during the optimization loop.

    Wraps RoutingEvaluator with:
    - Configurable sampling rate to control evaluation overhead
    - Score persistence via telemetry annotations
    - Multiple evaluator dispatch (routing_outcome, confidence_calibration)
    """

    def __init__(
        self,
        provider: TelemetryProvider,
        project_name: str,
        config: OnlineEvaluationConfig | None = None,
    ):
        self.provider = provider
        self.project_name = project_name

        if config is not None:
            self.enabled = config.enabled
            self.sampling_rate = config.sampling_rate
            self.evaluator_names = list(config.evaluators)
            self.persist_scores = config.persist_scores
            self.annotation_name = config.score_annotation_name
        else:
            self.enabled = True
            self.sampling_rate = 1.0
            self.evaluator_names = ["routing_outcome", "confidence_calibration"]
            self.persist_scores = True
            self.annotation_name = "online_eval"

        self.routing_evaluator = RoutingEvaluator(
            provider=provider, project_name=project_name
        )

        self._total_evaluated = 0
        self._total_skipped = 0

        logger.info(
            f"Initialized OnlineEvaluator (enabled={self.enabled}, "
            f"sampling_rate={self.sampling_rate}, "
            f"evaluators={self.evaluator_names})"
        )

    async def evaluate_span(
        self, span_data: Dict[str, Any]
    ) -> List[OnlineEvalResult]:
        """
        Evaluate a single routing span and optionally persist scores.

        Args:
            span_data: Span row data (Pandas Series converted to dict, or raw dict)

        Returns:
            List of evaluation results (one per evaluator).
            Empty list if evaluation is disabled or span is not sampled.
        """
        if not self.enabled:
            return []

        if self.sampling_rate < 1.0 and random.random() > self.sampling_rate:
            self._total_skipped += 1
            return []

        span_id = span_data.get("context.span_id", span_data.get("span_id", ""))
        results: List[OnlineEvalResult] = []

        for eval_name in self.evaluator_names:
            try:
                result = self._run_evaluator(eval_name, span_data, span_id)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Evaluator '{eval_name}' failed for {span_id}: {e}")

        if results and self.persist_scores:
            await self._persist_results(results)

        self._total_evaluated += 1
        return results

    def _run_evaluator(
        self, evaluator_name: str, span_data: Dict[str, Any], span_id: str
    ) -> Optional[OnlineEvalResult]:
        """Run a single named evaluator against span data."""
        if evaluator_name == "routing_outcome":
            return self._eval_routing_outcome(span_data, span_id)
        elif evaluator_name == "confidence_calibration":
            return self._eval_confidence_calibration(span_data, span_id)
        else:
            logger.warning(f"Unknown evaluator: {evaluator_name}")
            return None

    def _eval_routing_outcome(
        self, span_data: Dict[str, Any], span_id: str
    ) -> OnlineEvalResult:
        """Classify routing outcome (success/failure/ambiguous) and score."""
        outcome, status_description = (
            self.routing_evaluator._classify_routing_outcome(span_data)
        )

        score_map = {
            RoutingOutcome.SUCCESS: 1.0,
            RoutingOutcome.FAILURE: 0.0,
            RoutingOutcome.AMBIGUOUS: 0.5,
        }

        return OnlineEvalResult(
            span_id=span_id,
            evaluator_name="routing_outcome",
            score=score_map.get(outcome, 0.5),
            label=outcome.value,
            explanation=status_description,
            timestamp=datetime.now(),
        )

    def _eval_confidence_calibration(
        self, span_data: Dict[str, Any], span_id: str
    ) -> OnlineEvalResult:
        """Score how well confidence predicts actual success."""
        routing_attrs = span_data.get("attributes.routing")
        if not routing_attrs or not isinstance(routing_attrs, dict):
            routing_attrs = {}

        confidence = float(routing_attrs.get("confidence", 0.5))

        status = span_data.get("status_code", span_data.get("status", ""))
        actual_success = status == "OK" if status else True

        if actual_success:
            calibration_score = confidence
        else:
            calibration_score = 1.0 - confidence

        if calibration_score >= 0.8:
            label = "well_calibrated"
        elif calibration_score >= 0.5:
            label = "moderately_calibrated"
        else:
            label = "poorly_calibrated"

        return OnlineEvalResult(
            span_id=span_id,
            evaluator_name="confidence_calibration",
            score=calibration_score,
            label=label,
            explanation=(
                f"confidence={confidence:.2f}, "
                f"actual_success={actual_success}, "
                f"calibration={calibration_score:.2f}"
            ),
            timestamp=datetime.now(),
        )

    async def _persist_results(self, results: List[OnlineEvalResult]) -> None:
        """Write evaluation results as telemetry annotations."""
        for result in results:
            try:
                annotation_name = f"{self.annotation_name}.{result.evaluator_name}"
                await self.provider.annotations.add_annotation(
                    span_id=result.span_id,
                    name=annotation_name,
                    label=result.label,
                    score=result.score,
                    explanation=result.explanation,
                    metadata={
                        "evaluator": result.evaluator_name,
                        "online": True,
                        "timestamp": result.timestamp.isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to persist score for {result.span_id}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Return running statistics."""
        total = self._total_evaluated + self._total_skipped
        return {
            "total_evaluated": self._total_evaluated,
            "total_skipped": self._total_skipped,
            "sampling_rate": self.sampling_rate,
            "effective_rate": (
                self._total_evaluated / total if total > 0 else 0.0
            ),
            "evaluators": self.evaluator_names,
        }
