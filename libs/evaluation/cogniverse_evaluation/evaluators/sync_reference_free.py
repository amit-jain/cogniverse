"""
Synchronous reference-free evaluators for Phoenix experiments

These evaluators don't require golden datasets and can evaluate any retrieval result
"""

import logging
from typing import Any

import numpy as np

from .base import Evaluator, create_evaluation_result

logger = logging.getLogger(__name__)


class SyncQueryResultRelevanceEvaluator(Evaluator):
    """
    Synchronous evaluator for query-result relevance
    """

    def __init__(self, min_score_threshold: float = 0.5):
        self.min_score_threshold = min_score_threshold

    def evaluate(self, *, input=None, output=None, **kwargs) -> Any:
        """
        Evaluate query-result relevance without golden dataset
        """
        # Handle Phoenix experiment format
        if hasattr(output, "results"):
            results = output.results
        elif isinstance(output, dict) and "results" in output:
            results = output["results"]
        else:
            results = output if isinstance(output, list) else []

        if not results:
            return create_evaluation_result(
                score=0.0,
                label="no_results",
                explanation="No results returned for query",
            )

        # Check if top results have high scores
        top_scores = []
        for _i, result in enumerate(results[:5]):  # Top 5
            if isinstance(result, dict):
                score = result.get("score", 0)
            else:
                score = getattr(result, "score", 0)
            top_scores.append(score)

        avg_top_score = np.mean(top_scores) if top_scores else 0

        # Determine label based on average score
        if avg_top_score >= 0.8:
            label = "highly_relevant"
        elif avg_top_score >= self.min_score_threshold:
            label = "relevant"
        else:
            label = "low_relevance"

        return create_evaluation_result(
            score=float(avg_top_score),
            label=label,
            explanation=f"Average relevance score of top {len(top_scores)} results: {avg_top_score:.3f}",
        )


class SyncResultDiversityEvaluator(Evaluator):
    """
    Synchronous evaluator for result diversity
    """

    def evaluate(self, *, input=None, output=None, **kwargs) -> Any:
        """
        Evaluate result diversity
        """
        # Handle Phoenix experiment format
        if hasattr(output, "results"):
            results = output.results
        elif isinstance(output, dict) and "results" in output:
            results = output["results"]
        else:
            results = output if isinstance(output, list) else []

        if len(results) < 2:
            return create_evaluation_result(
                score=0.0,
                label="insufficient_results",
                explanation="Need at least 2 results to evaluate diversity",
            )

        # Extract unique video IDs
        video_ids = set()
        for result in results:
            if isinstance(result, dict):
                video_id = result.get("video_id")
            else:
                video_id = getattr(result, "video_id", None)

            if video_id:
                video_ids.add(video_id)

        # Diversity score: ratio of unique videos to total results
        diversity_score = len(video_ids) / len(results) if results else 0

        # Determine label
        if diversity_score >= 0.8:
            label = "high_diversity"
        elif diversity_score >= 0.5:
            label = "moderate_diversity"
        else:
            label = "low_diversity"

        return create_evaluation_result(
            score=float(diversity_score),
            label=label,
            explanation=f"{len(video_ids)} unique videos out of {len(results)} results",
        )


class SyncResultDistributionEvaluator(Evaluator):
    """
    Evaluates the distribution of scores in results
    """

    def evaluate(self, *, input=None, output=None, **kwargs) -> Any:
        """
        Evaluate score distribution for quality assessment
        """
        # Handle Phoenix experiment format
        if hasattr(output, "results"):
            results = output.results
        elif isinstance(output, dict) and "results" in output:
            results = output["results"]
        else:
            results = output if isinstance(output, list) else []

        if not results:
            return create_evaluation_result(
                score=0.0,
                label="no_results",
                explanation="No results to evaluate distribution",
            )

        # Extract scores
        scores = []
        for result in results:
            if isinstance(result, dict):
                score = result.get("score", 0)
            else:
                score = getattr(result, "score", 0)
            scores.append(score)

        # Calculate distribution metrics
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            score_range = max(scores) - min(scores)

            # Quality based on distribution characteristics
            # Good: high mean, low variance (consistent high scores)
            quality_score = (
                mean_score * (1 - std_score / 2) if std_score < 1 else mean_score * 0.5
            )

            if quality_score >= 0.7:
                label = "excellent_distribution"
            elif quality_score >= 0.5:
                label = "good_distribution"
            elif quality_score >= 0.3:
                label = "moderate_distribution"
            else:
                label = "poor_distribution"
        else:
            quality_score = 0.0
            label = "no_scores"
            mean_score = std_score = score_range = 0

        return create_evaluation_result(
            score=float(quality_score),
            label=label,
            explanation=f"Mean: {mean_score:.3f}, Std: {std_score:.3f}, Range: {score_range:.3f}",
        )


class SyncTemporalCoverageEvaluator(Evaluator):
    """
    Evaluates temporal coverage for video results
    """

    def evaluate(self, *, input=None, output=None, **kwargs) -> Any:
        """
        Evaluate temporal coverage of video results
        """
        # Handle Phoenix experiment format
        if hasattr(output, "results"):
            results = output.results
        elif isinstance(output, dict) and "results" in output:
            results = output["results"]
        else:
            results = output if isinstance(output, list) else []

        if not results:
            return create_evaluation_result(
                score=0.0,
                label="no_temporal_data",
                explanation="No results with temporal information",
            )

        # Extract temporal segments
        segments = []
        for result in results:
            if isinstance(result, dict):
                temporal_info = result.get("temporal_info")
                if temporal_info:
                    start = temporal_info.get("start_time", 0)
                    end = temporal_info.get("end_time", 0)
                    if end > start:
                        segments.append((start, end))

        if not segments:
            # No temporal data, but still give partial score based on result count
            coverage_score = min(len(results) / 10, 0.5)  # Max 0.5 for non-temporal
            return create_evaluation_result(
                score=float(coverage_score),
                label="no_temporal_coverage",
                explanation=f"{len(results)} results without temporal information",
            )

        # Merge overlapping segments
        segments.sort()
        merged = []
        for start, end in segments:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Calculate total coverage
        total_duration = sum(end - start for start, end in merged)
        unique_segments = len(merged)

        # Score based on coverage and segment count
        coverage_score = (
            min(total_duration / 300, 1.0) * 0.7 + min(unique_segments / 5, 1.0) * 0.3
        )

        if coverage_score >= 0.8:
            label = "excellent_coverage"
        elif coverage_score >= 0.6:
            label = "good_coverage"
        elif coverage_score >= 0.4:
            label = "moderate_coverage"
        else:
            label = "poor_coverage"

        return create_evaluation_result(
            score=float(coverage_score),
            label=label,
            explanation=f"{unique_segments} segments covering {total_duration:.1f}s total",
        )


def create_sync_evaluators() -> list[Evaluator]:
    """
    Create synchronous evaluators for Phoenix experiments

    Returns:
        List of evaluator instances
    """
    return [SyncQueryResultRelevanceEvaluator(), SyncResultDiversityEvaluator()]


def create_quality_evaluators() -> list[Evaluator]:
    """
    Create comprehensive quality evaluators

    Returns:
        List of quality evaluator instances
    """
    return [
        SyncQueryResultRelevanceEvaluator(min_score_threshold=0.5),
        SyncResultDiversityEvaluator(),
        SyncResultDistributionEvaluator(),
        SyncTemporalCoverageEvaluator(),
    ]
