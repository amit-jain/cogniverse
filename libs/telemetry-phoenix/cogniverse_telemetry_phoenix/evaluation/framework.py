"""
Phoenix evaluator framework implementation.

Provides Phoenix-specific evaluator base classes and result types.
"""

from typing import Any, Dict, Optional

from phoenix.client.resources.experiments.types import (
    BaseEvaluator as PhoenixEvaluator,
)

from cogniverse_evaluation.providers.base import EvaluatorFramework


class EvaluationResult(dict):
    """Dict subclass that supports attribute access for backward compatibility.

    Phoenix v14 ExperimentEvaluation is a TypedDict (plain dict), but the
    codebase accesses result.score, result.label everywhere. This class
    bridges both worlds: dict-compatible for Phoenix, attribute-accessible
    for internal evaluators.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'EvaluationResult' has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class PhoenixEvaluatorFramework(EvaluatorFramework):
    """
    Phoenix implementation of evaluator framework.

    Provides access to Phoenix's evaluator base classes and result types.
    """

    def get_evaluator_base_class(self) -> type:
        """
        Return Phoenix's base evaluator class.

        Returns:
            phoenix.client.resources.experiments.types.BaseEvaluator
        """
        return PhoenixEvaluator

    def get_evaluation_result_type(self) -> type:
        """
        Return Phoenix's evaluation result type.

        Returns:
            EvaluationResult (dict subclass with attribute access)
        """
        return EvaluationResult

    def create_evaluation_result(
        self,
        score: float,
        label: str,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Create Phoenix evaluation result.

        Args:
            score: Numeric score for the evaluation
            label: Classification label
            explanation: Human-readable explanation
            metadata: Additional metadata

        Returns:
            EvaluationResult with both dict and attribute access
        """
        return EvaluationResult(
            score=score, label=label, explanation=explanation, metadata=metadata or {}
        )
