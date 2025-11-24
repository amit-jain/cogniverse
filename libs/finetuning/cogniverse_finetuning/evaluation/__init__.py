"""
Fine-tuning evaluation module.

Evaluates adapter performance vs base model on held-out test sets.
"""

from cogniverse_finetuning.evaluation.adapter_evaluator import (
    AdapterEvaluator,
    ComparisonResult,
    EvaluationMetrics,
)

__all__ = ["AdapterEvaluator", "EvaluationMetrics", "ComparisonResult"]
