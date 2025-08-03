"""
Comprehensive Evaluation Framework for Cogniverse

This module provides structured evaluation capabilities using:
- Inspect AI for task-based evaluation
- Arize Phoenix for tracing and observability
- Custom scoring and metrics
"""

from .pipeline.orchestrator import EvaluationPipeline
from .phoenix.instrumentation import CogniverseInstrumentor
from .phoenix.monitoring import RetrievalMonitor

__all__ = [
    'EvaluationPipeline',
    'CogniverseInstrumentor', 
    'RetrievalMonitor'
]