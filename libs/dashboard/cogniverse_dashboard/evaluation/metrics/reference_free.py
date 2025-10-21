"""
Reference-free evaluation metrics that don't require ground truth.
"""

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class DiversityEvaluator:
    """
    Evaluate result diversity without ground truth.
    """

    def evaluate(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate diversity of search results.

        Args:
            query: Search query
            results: List of search results

        Returns:
            Evaluation metrics
        """
        if not results:
            return {
                "diversity_score": 0.0,
                "unique_videos": 0,
                "total_results": 0,
                "explanation": "No results to evaluate",
            }

        # Extract video IDs
        video_ids = [r.get("video_id") for r in results if r.get("video_id")]

        if not video_ids:
            return {
                "diversity_score": 0.0,
                "unique_videos": 0,
                "total_results": len(results),
                "explanation": "No video IDs found in results",
            }

        # Calculate diversity metrics
        unique_videos = len(set(video_ids))
        total_videos = len(video_ids)
        diversity_score = unique_videos / total_videos

        # Calculate distribution
        video_counts = Counter(video_ids)
        max_occurrences = max(video_counts.values())

        return {
            "diversity_score": diversity_score,
            "unique_videos": unique_videos,
            "total_results": total_videos,
            "max_occurrences": max_occurrences,
            "distribution": dict(video_counts),
            "explanation": f"Diversity: {unique_videos}/{total_videos} unique videos ({diversity_score:.2%})",
        }


class TemporalCoherenceEvaluator:
    """
    Evaluate temporal coherence for time-based queries.
    """

    def evaluate(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate if temporal queries return temporally coherent results.

        Args:
            query: Search query
            results: List of search results

        Returns:
            Evaluation metrics
        """
        # Check if this is a temporal query
        temporal_keywords = [
            "when",
            "after",
            "before",
            "during",
            "timeline",
            "first",
            "last",
            "then",
            "sequence",
            "order",
        ]

        query_lower = query.lower()
        is_temporal = any(kw in query_lower for kw in temporal_keywords)

        if not is_temporal:
            return {
                "temporal_query": False,
                "score": None,
                "explanation": "Not a temporal query",
            }

        if not results:
            return {
                "temporal_query": True,
                "score": 0.0,
                "explanation": "No results to evaluate temporal coherence",
            }

        # Extract temporal information
        timestamps = []
        for r in results:
            temporal_info = r.get("temporal_info", {})
            if "timestamp" in temporal_info:
                timestamps.append(temporal_info["timestamp"])
            elif "start_time" in temporal_info:
                timestamps.append(temporal_info["start_time"])

        if not timestamps:
            return {
                "temporal_query": True,
                "score": 0.5,
                "has_temporal_info": False,
                "explanation": "Results lack temporal information",
            }

        # Check ordering
        is_ascending = all(
            timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
        )
        is_descending = all(
            timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)
        )
        is_ordered = is_ascending or is_descending

        # Calculate temporal spread
        if len(timestamps) > 1:
            time_range = max(timestamps) - min(timestamps)
            avg_gap = time_range / (len(timestamps) - 1) if len(timestamps) > 1 else 0
        else:
            time_range = 0
            avg_gap = 0

        # Calculate score
        score = 1.0 if is_ordered else 0.0

        return {
            "temporal_query": True,
            "score": score,
            "is_ordered": is_ordered,
            "is_ascending": is_ascending,
            "is_descending": is_descending,
            "num_timestamps": len(timestamps),
            "time_range": time_range,
            "avg_gap": avg_gap,
            "explanation": f"Temporal coherence: {'ordered' if is_ordered else 'unordered'} ({len(timestamps)} timestamps)",
        }


class ResultDistributionEvaluator:
    """
    Evaluate the distribution of results across different dimensions.
    """

    def evaluate(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate how results are distributed.

        Args:
            query: Search query
            results: List of search results

        Returns:
            Evaluation metrics
        """
        if not results:
            return {"score": 0.0, "explanation": "No results to evaluate distribution"}

        # Analyze score distribution
        scores = [r.get("score", 0.0) for r in results]

        if scores:
            max_score = max(scores)
            min_score = min(scores)
            avg_score = sum(scores) / len(scores)
            score_range = max_score - min_score

            # Check if scores are well-distributed
            if score_range > 0:
                # Normalize scores
                normalized_scores = [(s - min_score) / score_range for s in scores]

                # Calculate standard deviation
                variance = sum((s - 0.5) ** 2 for s in normalized_scores) / len(
                    normalized_scores
                )
                std_dev = variance**0.5

                # Good distribution has reasonable spread
                distribution_score = min(1.0, std_dev * 2)
            else:
                distribution_score = 0.0
        else:
            distribution_score = 0.0
            max_score = min_score = avg_score = score_range = 0.0

        # Analyze rank distribution
        ranks = [r.get("rank", i + 1) for i, r in enumerate(results)]
        expected_ranks = list(range(1, len(results) + 1))
        rank_correlation = 1.0 if ranks == expected_ranks else 0.5

        return {
            "distribution_score": distribution_score,
            "rank_correlation": rank_correlation,
            "score_stats": {
                "max": max_score,
                "min": min_score,
                "avg": avg_score,
                "range": score_range,
            },
            "num_results": len(results),
            "explanation": f"Distribution score: {distribution_score:.2f}, Rank correlation: {rank_correlation:.2f}",
        }


class SemanticCoherenceEvaluator:
    """
    Evaluate semantic coherence of results with the query.
    Note: This would typically use embeddings or an LLM.
    """

    def evaluate(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate semantic coherence between query and results.

        Args:
            query: Search query
            results: List of search results

        Returns:
            Evaluation metrics
        """
        if not results:
            return {
                "coherence_score": 0.0,
                "explanation": "No results to evaluate coherence",
            }

        # Simple heuristic: check keyword overlap
        query_words = set(query.lower().split())

        coherence_scores = []
        for r in results:
            content = r.get("content", "")
            if content:
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                coherence = overlap / len(query_words) if query_words else 0.0
                coherence_scores.append(min(1.0, coherence))
            else:
                coherence_scores.append(0.0)

        avg_coherence = (
            sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        )

        return {
            "coherence_score": avg_coherence,
            "num_evaluated": len(coherence_scores),
            "explanation": f"Average semantic coherence: {avg_coherence:.2%}",
        }
