"""
Cogniverse Evaluation

Provider-agnostic evaluation framework for experiments, datasets, and evaluators.
Supports multiple evaluation backends (Phoenix, Langsmith, etc) via provider pattern.
"""

from cogniverse_evaluation.providers import (
    EvaluationProvider,
    get_evaluation_provider,
    set_evaluation_provider,
)

__version__ = "0.1.0"

__all__ = [
    "EvaluationProvider",
    "get_evaluation_provider",
    "set_evaluation_provider",
]
