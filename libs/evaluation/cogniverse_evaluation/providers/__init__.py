"""
Evaluation Provider System

Provides abstraction for evaluation backends (Phoenix, Langsmith, etc).
Similar to telemetry providers, but for experiment/dataset/evaluation concerns.
"""

from .base import (
    AnalyticsProvider,
    EvaluationProvider,
    MonitoringProvider,
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
    # Provider interfaces (Phoenix is the only concrete impl today —
    # RootCauseProvider + DatasetsProvider were removed because they
    # had zero implementations in any commit; the concrete
    # RootCauseAnalyzer in cogniverse_evaluation.analysis exists
    # outside this provider hierarchy).
    "EvaluationProvider",
    "AnalyticsProvider",
    "MonitoringProvider",
    # Data structures
    "TraceMetrics",
    # Registry functions
    "EvaluationRegistry",
    "get_evaluation_provider",
    "set_evaluation_provider",
    "register_evaluation_provider",
    "reset_evaluation_provider",
]
