"""
Synchronous golden dataset evaluator for Phoenix experiments
"""

import logging
from typing import Any

import numpy as np
from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult

logger = logging.getLogger(__name__)


class SyncGoldenDatasetEvaluator(Evaluator):
    """
    Synchronous evaluator that compares results against golden dataset
    """

    def __init__(self, golden_dataset: dict[str, dict[str, Any]]):
        """
        Initialize with golden dataset

        Args:
            golden_dataset: Dict mapping query to expected results
        """
        self.golden_dataset = golden_dataset

    def evaluate(
        self, *, input=None, output=None, expected=None, **kwargs
    ) -> EvaluationResult:
        """
        Evaluate results against golden dataset
        """
        # Extract query
        query = ""
        if hasattr(input, "query"):
            query = input.query
        elif isinstance(input, dict) and "query" in input:
            query = input["query"]
        elif isinstance(input, str):
            query = input

        # Extract results from output
        results = []
        if hasattr(output, "results"):
            results = output.results
        elif isinstance(output, dict) and "results" in output:
            results = output["results"]
        elif isinstance(output, list):
            results = output

        # Check if we have golden data for this query
        golden_data = self.golden_dataset.get(query)

        # Also check expected from the example
        expected_videos = []
        if expected and isinstance(expected, list):
            expected_videos = expected
        elif hasattr(expected, "expected_videos"):
            expected_videos = expected.expected_videos
        elif golden_data:
            expected_videos = golden_data.get("expected_videos", [])

        if not expected_videos and not golden_data:
            # Not a golden query, return a neutral score instead of -1
            # This allows the experiment to proceed without penalizing non-golden queries
            return EvaluationResult(
                score=0.5,
                label="not_golden",
                explanation="Query not in golden dataset - neutral score assigned",
            )

        # Extract video IDs from results
        retrieved_videos = []
        for result in results:
            if isinstance(result, dict):
                video_id = result.get("video_id", result.get("source_id", ""))
            else:
                video_id = getattr(result, "video_id", getattr(result, "source_id", ""))

            if video_id:
                retrieved_videos.append(video_id)

        # Calculate metrics
        metrics = self._calculate_metrics(retrieved_videos, expected_videos)

        # Use MRR as primary score
        score = metrics["mrr"]

        # Determine label
        if score >= 0.8:
            label = "excellent"
        elif score >= 0.5:
            label = "good"
        elif score > 0:
            label = "poor"
        else:
            label = "failed"

        return EvaluationResult(
            score=float(score),
            label=label,
            explanation=f"MRR: {metrics['mrr']:.3f}, P@5: {metrics['precision_at_5']:.3f}, R@5: {metrics['recall_at_5']:.3f}",
            metadata={
                "metrics": metrics,
                "expected_videos": expected_videos,
                "retrieved_videos": retrieved_videos[:10],
            },
        )

    def _calculate_metrics(
        self, retrieved: list[str], expected: list[str]
    ) -> dict[str, float]:
        """Calculate retrieval metrics"""
        metrics = {}

        if not expected:
            # No expected results, can't calculate metrics
            return {
                "mrr": 0.0,
                "precision_at_1": 0.0,
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
                "recall_at_1": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "ndcg": 0.0,
            }

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, video in enumerate(retrieved):
            if video in expected:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr

        # Precision at k
        for k in [1, 5, 10]:
            if k <= len(retrieved):
                relevant_at_k = sum(1 for v in retrieved[:k] if v in expected)
                metrics[f"precision_at_{k}"] = relevant_at_k / k
            else:
                metrics[f"precision_at_{k}"] = 0.0

        # Recall at k
        for k in [1, 5, 10]:
            if k <= len(retrieved):
                relevant_at_k = sum(1 for v in retrieved[:k] if v in expected)
                metrics[f"recall_at_{k}"] = relevant_at_k / len(expected)
            else:
                metrics[f"recall_at_{k}"] = 0.0

        # NDCG@10
        relevances = [1 if vid in expected else 0 for vid in retrieved[:10]]

        # DCG
        dcg = relevances[0] if relevances else 0
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 2)

        # Ideal DCG
        ideal_relevances = [1] * min(len(expected), 10) + [0] * max(
            0, 10 - len(expected)
        )
        idcg = ideal_relevances[0] if ideal_relevances else 0
        for i in range(1, len(ideal_relevances)):
            idcg += ideal_relevances[i] / np.log2(i + 2)

        metrics["ndcg"] = dcg / idcg if idcg > 0 else 0

        return metrics


def create_sync_evaluators_with_golden(
    golden_dataset: dict | None = None,
) -> list[Evaluator]:
    """
    Create synchronous evaluators including golden dataset evaluator

    Args:
        golden_dataset: Optional golden dataset for evaluation

    Returns:
        List of evaluator instances
    """
    from .sync_reference_free import (
        SyncQueryResultRelevanceEvaluator,
        SyncResultDiversityEvaluator,
    )

    evaluators = [SyncQueryResultRelevanceEvaluator(), SyncResultDiversityEvaluator()]

    if golden_dataset:
        evaluators.append(SyncGoldenDatasetEvaluator(golden_dataset))

    return evaluators
