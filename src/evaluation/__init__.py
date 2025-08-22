"""
Cogniverse Evaluation Framework.

A comprehensive evaluation system for video RAG built on Inspect AI, RAGAS, and Phoenix.
"""

__version__ = "0.1.0"

# New evaluation framework imports
from .core.scorers import (
    diversity_scorer,
    get_configured_scorers,
    precision_scorer,
    recall_scorer,
    relevance_scorer,
    schema_aware_temporal_scorer,
)
from .core.solvers import (
    create_batch_solver,
    create_live_solver,
    create_retrieval_solver,
)
from .core.task import evaluation_task, run_evaluation
from .data.datasets import DatasetManager
from .data.storage import PhoenixStorage
from .data.traces import TraceManager

__all__ = [
    # Core
    "evaluation_task",
    "run_evaluation",
    "create_retrieval_solver",
    "create_batch_solver",
    "create_live_solver",
    "get_configured_scorers",
    # Scorers
    "relevance_scorer",
    "precision_scorer",
    "recall_scorer",
    "diversity_scorer",
    "schema_aware_temporal_scorer",
    # Data
    "PhoenixStorage",
    "DatasetManager",
    "TraceManager",
]
