"""
Synchronous reference-free evaluators for Phoenix experiments

These evaluators don't require golden datasets and can evaluate any retrieval result
"""

import logging
from typing import List, Dict, Any

import numpy as np
from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult

logger = logging.getLogger(__name__)


class SyncQueryResultRelevanceEvaluator(Evaluator):
    """
    Synchronous evaluator for query-result relevance
    """
    
    def __init__(self, min_score_threshold: float = 0.5):
        self.min_score_threshold = min_score_threshold
    
    def evaluate(self, *, input=None, output=None, **kwargs) -> EvaluationResult:
        """
        Evaluate query-result relevance without golden dataset
        """
        # Handle Phoenix experiment format
        if hasattr(output, 'results'):
            results = output.results
        elif isinstance(output, dict) and 'results' in output:
            results = output['results']
        else:
            results = output if isinstance(output, list) else []
        
        if not results:
            return EvaluationResult(
                score=0.0,
                label="no_results",
                explanation="No results returned for query"
            )
        
        # Check if top results have high scores
        top_scores = []
        for i, result in enumerate(results[:5]):  # Top 5
            if isinstance(result, dict):
                score = result.get("score", 0)
            else:
                score = getattr(result, 'score', 0)
            top_scores.append(score)
        
        avg_top_score = np.mean(top_scores) if top_scores else 0
        
        # Determine label based on average score
        if avg_top_score >= 0.8:
            label = "highly_relevant"
        elif avg_top_score >= self.min_score_threshold:
            label = "relevant"
        else:
            label = "low_relevance"
        
        return EvaluationResult(
            score=float(avg_top_score),
            label=label,
            explanation=f"Average relevance score of top {len(top_scores)} results: {avg_top_score:.3f}"
        )


class SyncResultDiversityEvaluator(Evaluator):
    """
    Synchronous evaluator for result diversity
    """
    
    def evaluate(self, *, input=None, output=None, **kwargs) -> EvaluationResult:
        """
        Evaluate result diversity
        """
        # Handle Phoenix experiment format
        if hasattr(output, 'results'):
            results = output.results
        elif isinstance(output, dict) and 'results' in output:
            results = output['results']
        else:
            results = output if isinstance(output, list) else []
        
        if len(results) < 2:
            return EvaluationResult(
                score=0.0,
                label="insufficient_results",
                explanation="Need at least 2 results to evaluate diversity"
            )
        
        # Extract unique video IDs
        video_ids = set()
        for result in results:
            if isinstance(result, dict):
                video_id = result.get("video_id")
            else:
                video_id = getattr(result, 'video_id', None)
            
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
        
        return EvaluationResult(
            score=float(diversity_score),
            label=label,
            explanation=f"{len(video_ids)} unique videos out of {len(results)} results"
        )


def create_sync_evaluators() -> List[Evaluator]:
    """
    Create synchronous evaluators for Phoenix experiments
    
    Returns:
        List of evaluator instances
    """
    return [
        SyncQueryResultRelevanceEvaluator(),
        SyncResultDiversityEvaluator()
    ]