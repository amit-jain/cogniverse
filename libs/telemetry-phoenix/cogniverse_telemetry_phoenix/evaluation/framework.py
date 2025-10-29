"""
Phoenix evaluator framework implementation.

Provides Phoenix-specific evaluator base classes and result types.
"""

from typing import Any, Dict, Optional

from cogniverse_core.evaluation.providers.base import EvaluatorFramework
from phoenix.experiments.evaluators.base import Evaluator as PhoenixEvaluator
from phoenix.experiments.types import EvaluationResult


class PhoenixEvaluatorFramework(EvaluatorFramework):
    """
    Phoenix implementation of evaluator framework.

    Provides access to Phoenix's evaluator base classes and result types.
    """

    def get_evaluator_base_class(self) -> type:
        """
        Return Phoenix's base evaluator class.

        Returns:
            phoenix.experiments.evaluators.base.Evaluator
        """
        return PhoenixEvaluator

    def get_evaluation_result_type(self) -> type:
        """
        Return Phoenix's evaluation result type.

        Returns:
            phoenix.experiments.types.EvaluationResult
        """
        return EvaluationResult

    def create_evaluation_result(
        self,
        score: float,
        label: str,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Create Phoenix evaluation result.

        Args:
            score: Numeric score for the evaluation
            label: Classification label
            explanation: Human-readable explanation
            metadata: Additional metadata

        Returns:
            Phoenix EvaluationResult instance
        """
        return EvaluationResult(
            score=score,
            label=label,
            explanation=explanation,
            metadata=metadata or {}
        )
