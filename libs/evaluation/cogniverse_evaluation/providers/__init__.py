"""
Evaluation Provider System

Provides abstraction for evaluation backends (Phoenix, Langsmith, etc).
Similar to telemetry providers, but for experiment/dataset/evaluation concerns.
"""

from .base import (
    AnalyticsProvider,
    EvaluationProvider,
    TraceMetrics,
)
from .registry import (
    EvaluationRegistry,
    get_evaluation_provider,
    register_evaluation_provider,
    reset_evaluation_provider,
    set_evaluation_provider,
)

__all__ = [
    # Provider interfaces
    "EvaluationProvider",
    "AnalyticsProvider",
    # Data structures
    "TraceMetrics",
    # Registry functions
    "EvaluationRegistry",
    "get_evaluation_provider",
    "set_evaluation_provider",
    "register_evaluation_provider",
    "reset_evaluation_provider",
]
