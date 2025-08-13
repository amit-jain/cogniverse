"""
Scorers for evaluation including RAGAS metrics and custom metrics.
"""

from typing import List, Dict, Any, Optional
import logging

from inspect_ai.scorer import scorer, Score
from ragas.metrics import context_relevancy, context_precision, context_recall

logger = logging.getLogger(__name__)


def get_configured_scorers(config: Optional[Dict[str, Any]] = None) -> List:
    """
    Get list of scorers based on configuration.
    
    Args:
        config: Configuration dictionary specifying which scorers to use
        
    Returns:
        List of configured scorer functions
    """
    if not config:
        # Default scorers if no config provided
        return [
            ragas_context_relevancy_scorer(),
            custom_diversity_scorer(),
            custom_temporal_coherence_scorer()
        ]
    
    scorers = []
    
    # Add RAGAS scorers
    if config.get("use_ragas", True):
        ragas_metrics = config.get("ragas_metrics", ["context_relevancy"])
        if "context_relevancy" in ragas_metrics:
            scorers.append(ragas_context_relevancy_scorer())
        if "context_precision" in ragas_metrics:
            scorers.append(ragas_context_precision_scorer())
        if "context_recall" in ragas_metrics:
            scorers.append(ragas_context_recall_scorer())
    
    # Add custom scorers
    if config.get("use_custom", True):
        custom_metrics = config.get("custom_metrics", ["diversity", "temporal_coherence"])
        if "diversity" in custom_metrics:
            scorers.append(custom_diversity_scorer())
        if "temporal_coherence" in custom_metrics:
            scorers.append(custom_temporal_coherence_scorer())
        if "result_count" in custom_metrics:
            scorers.append(custom_result_count_scorer())
    
    # Add visual/LLM scorers if configured
    if config.get("use_visual", False):
        scorers.append(visual_quality_scorer(config.get("visual_config", {})))
    
    logger.info(f"Configured {len(scorers)} scorers")
    return scorers


@scorer
def ragas_context_relevancy_scorer():
    """
    RAGAS context relevancy scorer - works WITHOUT ground truth!
    Evaluates how relevant the retrieved contexts are to the query.
    """
    def score(state) -> Score:
        query = state.input.get("query", "")
        
        # Aggregate results from all configurations
        all_scores = {}
        
        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue
                
            results = output.get("results", [])
            
            if not results:
                all_scores[config_key] = 0.0
                continue
            
            try:
                # Extract contexts from results
                contexts = [r.get("content", "") for r in results if r.get("content")]
                
                if not contexts:
                    all_scores[config_key] = 0.0
                    continue
                
                # Use RAGAS to evaluate relevancy
                # Note: In production, this would use an LLM to judge relevancy
                # For now, we'll use a simplified heuristic
                relevancy_score = calculate_simple_relevancy(query, contexts)
                all_scores[config_key] = relevancy_score
                
            except Exception as e:
                logger.error(f"Failed to calculate relevancy for {config_key}: {e}")
                all_scores[config_key] = 0.0
        
        # Calculate overall score
        if all_scores:
            avg_score = sum(all_scores.values()) / len(all_scores)
            explanation = f"Context relevancy scores: {', '.join(f'{k}={v:.3f}' for k, v in all_scores.items())}"
        else:
            avg_score = 0.0
            explanation = "No results to evaluate"
        
        return Score(
            value=avg_score,
            explanation=explanation,
            metadata={"individual_scores": all_scores}
        )
    
    return score


@scorer
def ragas_context_precision_scorer():
    """
    RAGAS context precision scorer - requires ground truth.
    Measures what fraction of retrieved contexts are relevant.
    """
    def score(state) -> Score:
        # Get expected results if available
        expected = state.output.get("expected_videos", []) if hasattr(state, 'output') else []
        
        if not expected:
            return Score(
                value=None,
                explanation="No ground truth available for precision calculation"
            )
        
        all_scores = {}
        
        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue
                
            results = output.get("results", [])
            retrieved_videos = [r.get("video_id") for r in results]
            
            # Calculate precision
            if retrieved_videos:
                relevant_retrieved = len([v for v in retrieved_videos if v in expected])
                precision = relevant_retrieved / len(retrieved_videos)
                all_scores[config_key] = precision
            else:
                all_scores[config_key] = 0.0
        
        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        
        return Score(
            value=avg_score,
            explanation=f"Context precision: {avg_score:.3f}",
            metadata={"individual_scores": all_scores}
        )
    
    return score


@scorer
def ragas_context_recall_scorer():
    """
    RAGAS context recall scorer - requires ground truth.
    Measures what fraction of relevant contexts were retrieved.
    """
    def score(state) -> Score:
        # Get expected results if available
        expected = state.output.get("expected_videos", []) if hasattr(state, 'output') else []
        
        if not expected:
            return Score(
                value=None,
                explanation="No ground truth available for recall calculation"
            )
        
        all_scores = {}
        
        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue
                
            results = output.get("results", [])
            retrieved_videos = [r.get("video_id") for r in results]
            
            # Calculate recall
            if expected:
                relevant_retrieved = len([v for v in expected if v in retrieved_videos])
                recall = relevant_retrieved / len(expected)
                all_scores[config_key] = recall
            else:
                all_scores[config_key] = 0.0
        
        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        
        return Score(
            value=avg_score,
            explanation=f"Context recall: {avg_score:.3f}",
            metadata={"individual_scores": all_scores}
        )
    
    return score


@scorer
def custom_diversity_scorer():
    """
    Custom diversity metric - evaluates result diversity without ground truth.
    """
    def score(state) -> Score:
        all_scores = {}
        
        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue
                
            results = output.get("results", [])
            
            if not results:
                all_scores[config_key] = 0.0
                continue
            
            # Calculate diversity as ratio of unique videos
            video_ids = [r.get("video_id") for r in results if r.get("video_id")]
            if video_ids:
                unique_videos = len(set(video_ids))
                diversity = unique_videos / len(video_ids)
                all_scores[config_key] = diversity
            else:
                all_scores[config_key] = 0.0
        
        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        
        return Score(
            value=avg_score,
            explanation=f"Result diversity: {avg_score:.3f} (unique/total ratio)",
            metadata={"individual_scores": all_scores}
        )
    
    return score


@scorer
def custom_temporal_coherence_scorer():
    """
    Evaluate temporal coherence for time-based queries.
    """
    def score(state) -> Score:
        query = state.input.get("query", "").lower()
        
        # Check if this is a temporal query
        temporal_keywords = ['when', 'after', 'before', 'during', 'timeline', 'first', 'last', 'then']
        is_temporal = any(kw in query for kw in temporal_keywords)
        
        if not is_temporal:
            return Score(
                value=None,
                explanation="Not a temporal query"
            )
        
        all_scores = {}
        
        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue
                
            results = output.get("results", [])
            
            # Check if results have temporal information
            timestamps = []
            for r in results:
                temporal_info = r.get("temporal_info", {})
                if "timestamp" in temporal_info:
                    timestamps.append(temporal_info["timestamp"])
            
            if timestamps and len(timestamps) > 1:
                # Check if timestamps are properly ordered
                is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
                all_scores[config_key] = 1.0 if is_ordered else 0.0
            else:
                all_scores[config_key] = 0.5  # Neutral if no temporal info
        
        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        
        return Score(
            value=avg_score,
            explanation=f"Temporal coherence: {avg_score:.3f}",
            metadata={"individual_scores": all_scores}
        )
    
    return score


@scorer
def custom_result_count_scorer():
    """
    Simple scorer that checks if we got enough results.
    """
    def score(state) -> Score:
        all_counts = {}
        
        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_counts[config_key] = 0
                continue
                
            results = output.get("results", [])
            all_counts[config_key] = len(results)
        
        # Score based on whether we got at least 5 results
        all_scores = {k: min(1.0, v/5.0) for k, v in all_counts.items()}
        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        
        return Score(
            value=avg_score,
            explanation=f"Result counts: {all_counts}",
            metadata={"counts": all_counts, "scores": all_scores}
        )
    
    return score


@scorer
def visual_quality_scorer(visual_config: Dict[str, Any]):
    """
    LLM-based visual quality scorer.
    """
    def score(state) -> Score:
        # This would use an LLM to evaluate visual quality
        # For now, return a placeholder
        return Score(
            value=None,
            explanation="Visual quality scoring not yet implemented"
        )
    
    return score


def calculate_simple_relevancy(query: str, contexts: List[str]) -> float:
    """
    Simple heuristic for relevancy calculation.
    In production, this would use an LLM or embedding similarity.
    """
    if not contexts:
        return 0.0
    
    # Simple keyword overlap heuristic
    query_words = set(query.lower().split())
    
    relevancy_scores = []
    for context in contexts:
        context_words = set(context.lower().split())
        overlap = len(query_words & context_words)
        relevancy = overlap / len(query_words) if query_words else 0.0
        relevancy_scores.append(min(1.0, relevancy))
    
    return sum(relevancy_scores) / len(relevancy_scores)