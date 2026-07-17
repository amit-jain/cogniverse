"""Online Evaluator for real-time routing span scoring.

Scores individual `cogniverse.routing` spans produced by GatewayAgent and
persists the scores as telemetry annotations for drift detection and
continuous quality monitoring.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cogniverse_evaluation.evaluators.agent_evaluators import get_agent_evaluator
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingEvaluator

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
        agent_type: str = "routing",
    ):
        self.provider = provider
        self.project_name = project_name
        # Which per-agent evaluator registry entry this instance scores with.
        self.agent_type = agent_type

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

    async def evaluate_span(self, span_data: Dict[str, Any]) -> List[OnlineEvalResult]:
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
        """Run a single named evaluator from the per-agent registry."""
        entry = get_agent_evaluator(self.agent_type)
        result = entry.run_structural(evaluator_name, span_data) if entry else None
        if result is None:
            logger.warning(f"Unknown evaluator: {evaluator_name}")
            return None
        return OnlineEvalResult(
            span_id=span_id,
            evaluator_name=result.evaluator_name,
            score=result.score,
            label=result.label,
            explanation=result.explanation,
            timestamp=datetime.now(),
        )

    def _eval_routing_outcome(
        self, span_data: Dict[str, Any], span_id: str
    ) -> OnlineEvalResult:
        """Classify routing outcome (success/failure/ambiguous) and score."""
        return self._run_evaluator("routing_outcome", span_data, span_id)

    def _eval_confidence_calibration(
        self, span_data: Dict[str, Any], span_id: str
    ) -> OnlineEvalResult:
        """Score how well confidence predicts actual success."""
        return self._run_evaluator("confidence_calibration", span_data, span_id)

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
                    metadata={
                        "evaluator": result.evaluator_name,
                        "online": True,
                        "explanation": result.explanation,
                        "timestamp": result.timestamp.isoformat(),
                    },
                    project=self.project_name,
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
            "effective_rate": (self._total_evaluated / total if total > 0 else 0.0),
            "evaluators": self.evaluator_names,
        }
