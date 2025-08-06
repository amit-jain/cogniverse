"""
Golden dataset evaluator for spans marked with dataset identifiers

This evaluator can evaluate spans that have been marked as belonging to a test dataset
"""

import logging
from typing import List, Dict, Any, Optional, Set
import json

from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult
import numpy as np

logger = logging.getLogger(__name__)


class GoldenDatasetEvaluator(Evaluator):
    """
    Evaluates spans against a golden dataset when they have matching identifiers
    """
    
    def __init__(self, golden_dataset: Dict[str, Dict[str, Any]]):
        """
        Initialize with golden dataset
        
        Args:
            golden_dataset: Dict mapping query to expected results
                           {query: {"expected_videos": [...], "relevance_scores": {...}}}
        """
        self.golden_dataset = golden_dataset
        
    async def evaluate(
        self,
        input: str,
        output: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate results against golden dataset if applicable
        
        Args:
            input: The search query
            output: List of retrieved results
            metadata: Span metadata that may contain dataset identifiers
            
        Returns:
            EvaluationResult with metrics against golden dataset
        """
        # Check if this span is marked for evaluation
        if not metadata:
            return EvaluationResult(
                score=-1.0,
                label="not_evaluable",
                explanation="No metadata to identify if this is a test query"
            )
        
        dataset_id = metadata.get("dataset_id")
        is_test_query = metadata.get("is_test_query", False)
        
        if not dataset_id and not is_test_query:
            return EvaluationResult(
                score=-1.0,
                label="not_test_query",
                explanation="This span is not marked as a test query"
            )
        
        # Find golden data for this query
        golden_data = self.golden_dataset.get(input)
        if not golden_data:
            return EvaluationResult(
                score=0.0,
                label="no_golden_data",
                explanation=f"No golden data found for query: {input}",
                metadata={"dataset_id": dataset_id}
            )
        
        # Extract expected results
        expected_videos = golden_data.get("expected_videos", [])
        relevance_scores = golden_data.get("relevance_scores", {})
        
        # Extract actual results
        retrieved_videos = []
        for result in output:
            video_id = result.get("source_id", result.get("video_id"))
            if video_id:
                retrieved_videos.append(video_id)
        
        # Calculate metrics
        metrics = self._calculate_metrics(retrieved_videos, expected_videos, relevance_scores)
        
        # Determine overall score and label
        overall_score = metrics["mrr"]  # Use MRR as primary metric
        
        if overall_score >= 0.8:
            label = "excellent"
        elif overall_score >= 0.5:
            label = "good"
        elif overall_score > 0:
            label = "poor"
        else:
            label = "failed"
        
        return EvaluationResult(
            score=float(overall_score),
            label=label,
            explanation=f"MRR: {metrics['mrr']:.3f}, P@5: {metrics['precision_at_5']:.3f}, NDCG: {metrics['ndcg']:.3f}",
            metadata={
                "dataset_id": dataset_id,
                "metrics": metrics,
                "expected_videos": expected_videos,
                "retrieved_videos": retrieved_videos[:10]  # Top 10
            }
        )
    
    def _calculate_metrics(
        self,
        retrieved: List[str],
        expected: List[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        metrics = {}
        
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
        if expected:
            for k in [1, 5, 10]:
                if k <= len(retrieved):
                    relevant_at_k = sum(1 for v in retrieved[:k] if v in expected)
                    metrics[f"recall_at_{k}"] = relevant_at_k / len(expected)
                else:
                    metrics[f"recall_at_{k}"] = 0.0
        else:
            for k in [1, 5, 10]:
                metrics[f"recall_at_{k}"] = 0.0
        
        # NDCG@10
        if relevance_scores:
            # Use provided relevance scores
            relevances = [relevance_scores.get(vid, 0) for vid in retrieved[:10]]
        else:
            # Binary relevance
            relevances = [1 if vid in expected else 0 for vid in retrieved[:10]]
        
        dcg = relevances[0] if relevances else 0
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 2)
        
        # Ideal DCG
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = ideal_relevances[0] if ideal_relevances else 0
        for i in range(1, len(ideal_relevances)):
            idcg += ideal_relevances[i] / np.log2(i + 2)
        
        metrics["ndcg"] = dcg / idcg if idcg > 0 else 0
        
        return metrics


def create_low_scoring_golden_dataset() -> Dict[str, Dict[str, Any]]:
    """
    Create a golden dataset with queries known to have low scores
    This helps test the evaluation system with challenging queries
    
    Returns:
        Golden dataset dictionary
    """
    # Import here to avoid circular imports
    from tests.comprehensive_video_query_test_v2 import VISUAL_TEST_QUERIES
    
    golden_dataset = {
        # Ambiguous queries that should return diverse results
        "object in motion": {
            "expected_videos": ["v_gkSMwfO1q1I", "v_0NIKVT3kmT4"],
            "relevance_scores": {"v_gkSMwfO1q1I": 0.5, "v_0NIKVT3kmT4": 0.5}
        },
        
        # Very specific queries that might not match well
        "purple elephant dancing in rain": {
            "expected_videos": [],  # No videos should match this
            "relevance_scores": {}
        },
        
        # Temporal queries that are challenging
        "something happening at exactly 2:15": {
            "expected_videos": ["v_WFrgou1LD2Q"],  # Only if we have temporal alignment
            "relevance_scores": {"v_WFrgou1LD2Q": 1.0}
        },
        
        # Abstract concept queries
        "feeling of nostalgia": {
            "expected_videos": ["v_HWFrgou1LD2Q", "v_-IMXSEIabMM"],
            "relevance_scores": {"v_HWFrgou1LD2Q": 0.3, "v_-IMXSEIabMM": 0.3}
        },
        
        # Multi-modal challenging query
        "loud noise with bright flash": {
            "expected_videos": ["v_J0nA4VgnoCo"],
            "relevance_scores": {"v_J0nA4VgnoCo": 0.8}
        },
        
        # Negation query (often challenging)
        "not indoor scene": {
            "expected_videos": ["v_-IMXSEIabMM", "v_HWFrgou1LD2Q", "v_gkSMwfO1q1I"],
            "relevance_scores": {
                "v_-IMXSEIabMM": 0.9,  # Outdoor snow scene
                "v_HWFrgou1LD2Q": 0.9,  # Outdoor scene
                "v_gkSMwfO1q1I": 0.9   # Outdoor activity
            }
        },
        
        # Query with typos/misspellings
        "preson wering winer clotes": {  # person wearing winter clothes
            "expected_videos": ["v_-IMXSEIabMM"],
            "relevance_scores": {"v_-IMXSEIabMM": 0.7}  # Lower score due to typos
        },
        
        # Very long, complex query
        "a scene showing multiple people engaged in various activities with complex interactions and environmental factors": {
            "expected_videos": ["v_0NIKVT3kmT4", "v_gkSMwfO1q1I"],
            "relevance_scores": {"v_0NIKVT3kmT4": 0.4, "v_gkSMwfO1q1I": 0.4}
        }
    }


def mark_span_as_test_query(span_attributes: Dict[str, Any], dataset_id: str = "golden_test_v1"):
    """
    Helper function to mark a span as belonging to a test dataset
    
    Args:
        span_attributes: The span attributes dictionary to update
        dataset_id: The dataset identifier
    """
    span_attributes.update({
        "is_test_query": True,
        "dataset_id": dataset_id,
        "evaluation_enabled": True
    })