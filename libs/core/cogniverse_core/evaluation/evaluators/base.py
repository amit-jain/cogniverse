"""
Generic evaluator base classes using provider abstraction.

This module provides base classes that work with any telemetry provider.
"""

from typing import Any, Dict, Optional


def get_evaluator_base_class():
    """
    Get the evaluator base class from the current provider.

    Returns:
        Base evaluator class from the telemetry provider
    """
    from cogniverse_core.evaluation.providers import get_evaluator_provider

    provider = get_evaluator_provider()
    return provider.framework.get_evaluator_base_class()


def get_evaluation_result_type():
    """
    Get the evaluation result type from the current provider.

    Returns:
        Evaluation result type from the telemetry provider
    """
    from cogniverse_core.evaluation.providers import get_evaluator_provider

    provider = get_evaluator_provider()
    return provider.framework.get_evaluation_result_type()


def create_evaluation_result(
    score: float,
    label: str,
    explanation: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Create an evaluation result using the current provider.

    Args:
        score: Numeric score
        label: Classification label
        explanation: Human-readable explanation
        metadata: Additional metadata

    Returns:
        Evaluation result in provider-specific format
    """
    from cogniverse_core.evaluation.providers import get_evaluator_provider

    provider = get_evaluator_provider()
    return provider.framework.create_evaluation_result(
        score=score,
        label=label,
        explanation=explanation,
        metadata=metadata
    )


# For backward compatibility, provide a class that inherits from provider's evaluator
class Evaluator:
    """
    Generic evaluator base class that delegates to provider's evaluator.

    Subclasses should implement the evaluate method.
    """

    def __init__(self):
        from cogniverse_core.evaluation.providers import get_evaluator_provider

        provider = get_evaluator_provider()
        self._provider_evaluator_class = provider.framework.get_evaluator_base_class()

    def __init_subclass__(cls, **kwargs):
        """
        Called when a subclass is created.

        This dynamically adds the provider's evaluator base class to the MRO.
        """
        super().__init_subclass__(**kwargs)

        # Get provider's base class
        from cogniverse_core.evaluation.providers import get_evaluator_provider

        try:
            provider = get_evaluator_provider()
            provider_base = provider.framework.get_evaluator_base_class()

            # Add provider's base to the class bases if not already present
            if provider_base not in cls.__bases__:
                cls.__bases__ = (provider_base,) + cls.__bases__
        except Exception:
            # If provider not available during import, will be resolved at runtime
            pass


# Create a result type accessor
class EvaluationResult:
    """
    Wrapper for evaluation result that delegates to provider's result type.
    """

    def __new__(cls, score: float, label: str, explanation: str, metadata: Optional[Dict[str, Any]] = None):
        return create_evaluation_result(score, label, explanation, metadata)
