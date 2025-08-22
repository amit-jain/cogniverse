"""
Scoring mechanisms for Cogniverse evaluation with Inspect AI
"""

import json
import logging
import numpy as np
from typing import List, Dict

from inspect_ai.scorer import Scorer, scorer, Score

logger = logging.getLogger(__name__)


# Factory functions for Inspect AI registration
@scorer(metrics=[], name="video_retrieval_scorer")
def video_retrieval_scorer(metrics: List[str] = None) -> Scorer:
    """Create a video retrieval scorer for Inspect AI."""
    scorer_instance = VideoRetrievalScorer(metrics)
    return scorer_instance


@scorer(metrics=[], name="temporal_accuracy_scorer")
def temporal_accuracy_scorer() -> Scorer:
    """Create a temporal accuracy scorer for Inspect AI."""
    scorer_instance = TemporalAccuracyScorer()
    return scorer_instance


@scorer(metrics=[], name="alignment_scorer")
def alignment_scorer(k: int = 10) -> Scorer:
    """Create an alignment scorer for Inspect AI."""
    scorer_instance = AlignmentScorer(k)
    return scorer_instance


@scorer(metrics=[], name="failure_analysis_scorer")
def failure_analysis_scorer() -> Scorer:
    """Create a failure analysis scorer for Inspect AI."""
    scorer_instance = FailureAnalysisScorer()
    return scorer_instance


class VideoRetrievalScorer:
    """Comprehensive scorer for video retrieval tasks"""

    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["mrr", "ndcg", "precision", "recall"]
        self.metric_calculators = {
            "mrr": self._calculate_mrr,
            "ndcg": self._calculate_ndcg,
            "precision": self._calculate_precision,
            "recall": self._calculate_recall,
            "map": self._calculate_map,
        }

    async def __call__(self, state, target) -> Score:
        """Score the retrieval results"""
        if not hasattr(state, "metadata") or "retrieval_results" not in state.metadata:
            return Score(value=0.0, explanation="No retrieval results found")

        results = state.metadata["retrieval_results"]

        # Get expected videos
        expected = []
        if hasattr(state, "metadata") and "expected_videos" in state.metadata:
            expected = state.metadata["expected_videos"]
        elif (
            hasattr(state.input, "metadata")
            and "expected_videos" in state.input.metadata
        ):
            expected = state.input.metadata["expected_videos"]
        else:
            # Try to parse from target
            try:
                expected = eval(target) if isinstance(target, str) else target
            except Exception:
                logger.warning(f"Could not parse expected videos from target: {target}")

        if not expected:
            return Score(value=0.0, explanation="No expected videos specified")

        # Calculate scores - handle both list and dict formats
        scores = {}

        if isinstance(results, dict):
            # Dict of configs with results
            for config, search_results in results.items():
                config_scores = {}
                for metric in self.metrics:
                    if metric in self.metric_calculators:
                        config_scores[metric] = self.metric_calculators[metric](
                            search_results, expected
                        )
                scores[config] = config_scores
        elif isinstance(results, list):
            # Direct list of results
            config_scores = {}
            for metric in self.metrics:
                if metric in self.metric_calculators:
                    config_scores[metric] = self.metric_calculators[metric](
                        results, expected
                    )
            scores["default"] = config_scores
        else:
            # Unknown format
            scores["default"] = {metric: 0.0 for metric in self.metrics}

        # Log to Phoenix using OpenTelemetry
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("scoring") as span:
            span.set_attribute("scores", json.dumps(scores))
            span.set_attribute(
                "query",
                state.input.text if hasattr(state.input, "text") else str(state.input),
            )

        # Calculate overall score (average across all metrics and configs)
        all_scores = []
        for config_scores in scores.values():
            all_scores.extend(config_scores.values())

        overall_score = np.mean(all_scores) if all_scores else 0.0

        return Score(
            value=overall_score,
            answer=json.dumps(scores, indent=2),
            explanation=self._generate_explanation(scores),
            metadata={
                "metrics": scores.get("default", scores) if len(scores) == 1 else scores
            },
        )

    def _calculate_mrr(self, results: List, expected: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not results:
            return 0.0

        for i, result in enumerate(results):
            # Handle both string and dict results
            if isinstance(result, str):
                video_id = result
            elif isinstance(result, dict):
                video_id = result.get("video_id", result.get("id", ""))
            else:
                continue

            if video_id in expected:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_ndcg(self, results: List, expected: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not results:
            return 0.0

        # Binary relevance: 1 if video is expected, 0 otherwise
        relevances = []
        for r in results[:k]:
            if isinstance(r, str):
                relevances.append(1 if r in expected else 0)
            elif isinstance(r, dict):
                video_id = r.get("video_id", r.get("id", ""))
                relevances.append(1 if video_id in expected else 0)
            else:
                relevances.append(0)

        # Calculate DCG
        dcg = relevances[0] if relevances else 0
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 2)

        # Calculate ideal DCG
        ideal_relevances = [1] * min(len(expected), k) + [0] * max(0, k - len(expected))
        idcg = ideal_relevances[0] if ideal_relevances else 0
        for i in range(1, len(ideal_relevances)):
            idcg += ideal_relevances[i] / np.log2(i + 2)

        # Calculate NDCG
        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_precision(
        self, results: List, expected: List[str], k: int = 10
    ) -> float:
        """Calculate Precision@k"""
        if not results:
            return 0.0

        top_k_results = results[:k]
        relevant_found = 0
        for r in top_k_results:
            if isinstance(r, str):
                if r in expected:
                    relevant_found += 1
            elif isinstance(r, dict):
                video_id = r.get("video_id", r.get("id", ""))
                if video_id in expected:
                    relevant_found += 1

        return relevant_found / len(top_k_results)

    def _calculate_recall(
        self, results: List, expected: List[str], k: int = 10
    ) -> float:
        """Calculate Recall@k"""
        if not results or not expected:
            return 0.0

        top_k_results = results[:k]
        relevant_found = 0
        for r in top_k_results:
            if isinstance(r, str):
                if r in expected:
                    relevant_found += 1
            elif isinstance(r, dict):
                video_id = r.get("video_id", r.get("id", ""))
                if video_id in expected:
                    relevant_found += 1

        return relevant_found / len(expected)

    def _calculate_map(self, results: List, expected: List[str]) -> float:
        """Calculate Mean Average Precision"""
        if not results or not expected:
            return 0.0

        num_relevant = 0
        sum_precision = 0.0

        for i, result in enumerate(results):
            video_id = None
            if isinstance(result, str):
                video_id = result
            elif isinstance(result, dict):
                video_id = result.get("video_id", result.get("id", ""))

            if video_id and video_id in expected:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i

        return sum_precision / len(expected) if expected else 0.0

    def _generate_explanation(self, scores: Dict[str, Dict[str, float]]) -> str:
        """Generate human-readable explanation of scores"""
        if not scores:
            return "No scores calculated"

        explanations = []

        # Find best performing configuration
        best_config = None
        best_score = -1

        for config, config_scores in scores.items():
            avg_score = np.mean(list(config_scores.values()))
            if avg_score > best_score:
                best_score = avg_score
                best_config = config

        explanations.append(
            f"Best configuration: {best_config} (avg score: {best_score:.3f})"
        )

        # Add per-metric summaries
        for metric in self.metrics:
            metric_scores = [
                config_scores.get(metric, 0) for config_scores in scores.values()
            ]
            if metric_scores:
                avg = np.mean(metric_scores)
                explanations.append(f"{metric.upper()}: {avg:.3f} (avg across configs)")

        return "\n".join(explanations)


class TemporalAccuracyScorer(Scorer):
    """Scorer for temporal understanding tasks"""

    def __init__(self):
        self.name = "mrr"

    async def __call__(self, state, target) -> Score:
        """Score temporal understanding"""
        if not hasattr(state, "metadata"):
            return Score(value=0.0, explanation="No metadata found")

        temporal_info = state.metadata.get("temporal_info", {})

        # Get expected time range
        expected_range = None
        if "expected_time_range" in state.metadata:
            expected_range = state.metadata["expected_time_range"]
        elif (
            hasattr(state.input, "metadata")
            and "expected_time_range" in state.input.metadata
        ):
            expected_range = state.input.metadata["expected_time_range"]
        else:
            try:
                expected_range = eval(target) if isinstance(target, str) else target
            except (SyntaxError, NameError, ValueError) as e:
                logger.warning(f"Failed to parse expected range from target: {e}")
                expected_range = None

        if not expected_range or not temporal_info.get("extracted_range"):
            return Score(
                value=0.0, explanation="Could not extract or compare time ranges"
            )

        # Calculate overlap between extracted and expected ranges
        extracted = temporal_info["extracted_range"]
        overlap = self._calculate_range_overlap(extracted, expected_range)

        return Score(
            value=overlap,
            answer=f"Extracted: {extracted}, Expected: {expected_range}",
            explanation=f"Time range overlap: {overlap:.2%}",
            metadata={
                "extracted_range": extracted,
                "expected_range": expected_range,
                "overlap": overlap,
            },
        )

    def _calculate_range_overlap(
        self, range1: List[float], range2: List[float]
    ) -> float:
        """Calculate overlap between two time ranges"""
        if len(range1) != 2 or len(range2) != 2:
            return 0.0

        # Convert negative indices to positive if needed
        # (assuming video duration for simplicity)
        r1_start, r1_end = range1
        r2_start, r2_end = range2

        # Calculate overlap
        overlap_start = max(r1_start, r2_start)
        overlap_end = min(r1_end, r2_end)

        if overlap_start >= overlap_end:
            return 0.0  # No overlap

        # Calculate IoU (Intersection over Union)
        intersection = overlap_end - overlap_start
        union = (r1_end - r1_start) + (r2_end - r2_start) - intersection

        return intersection / union if union > 0 else 0.0


class AlignmentScorer(Scorer):
    """Scorer for multimodal alignment tasks"""

    def __init__(self, k: int = 10):
        self.name = "ndcg"
        self.k = k

    async def __call__(self, state, target) -> Score:
        """Score multimodal alignment"""
        if not hasattr(state, "metadata"):
            return Score(value=0.0, explanation="No metadata found")

        alignment_score = state.metadata.get("alignment_score", 0.0)
        alignment_check = state.metadata.get("alignment_check", False)

        # Get expected alignment
        expected_alignment = None
        if "expected_alignment" in state.metadata:
            expected_alignment = state.metadata["expected_alignment"]
        elif (
            hasattr(state.input, "metadata")
            and "expected_alignment" in state.input.metadata
        ):
            expected_alignment = state.input.metadata["expected_alignment"]
        else:
            try:
                expected_alignment = eval(target) if isinstance(target, str) else target
            except (SyntaxError, NameError, ValueError) as e:
                logger.warning(f"Failed to parse expected alignment from target: {e}")
                expected_alignment = None

        # Calculate accuracy
        if expected_alignment is not None:
            accuracy = 1.0 if alignment_check == expected_alignment else 0.0
        else:
            accuracy = alignment_score

        return Score(
            value=accuracy,
            answer=f"Alignment: {alignment_check}, Score: {alignment_score:.3f}",
            explanation=f"Alignment {'correct' if accuracy == 1.0 else 'incorrect'}",
            metadata={
                "alignment_score": alignment_score,
                "alignment_check": alignment_check,
                "expected": expected_alignment,
            },
        )


class FailureAnalysisScorer(Scorer):
    """Scorer for failure analysis tasks"""

    async def __call__(self, state, target) -> Score:
        """Score based on failure analysis"""
        if not hasattr(state, "metadata"):
            return Score(value=0.0, explanation="No metadata found")

        failure_analysis = state.metadata.get("failure_analysis", {})
        error_patterns = state.metadata.get("error_patterns", {})

        # Calculate failure rate
        total_configs = len(state.metadata.get("retrieval_results", {}))
        failed_configs = len(failure_analysis)

        if total_configs == 0:
            failure_rate = 1.0
        else:
            failure_rate = failed_configs / total_configs

        # Success score is inverse of failure rate
        success_score = 1.0 - failure_rate

        # Generate detailed analysis
        analysis = []

        if error_patterns:
            analysis.append(
                f"No results errors: {error_patterns.get('no_results_count', 0)}"
            )
            analysis.append(
                f"No relevant errors: {error_patterns.get('no_relevant_count', 0)}"
            )

            if error_patterns.get("affected_profiles"):
                analysis.append(
                    f"Affected profiles: {', '.join(error_patterns['affected_profiles'])}"
                )
            if error_patterns.get("affected_strategies"):
                analysis.append(
                    f"Affected strategies: {', '.join(error_patterns['affected_strategies'])}"
                )

        return Score(
            value=success_score,
            answer=json.dumps(failure_analysis, indent=2),
            explanation=f"Failure rate: {failure_rate:.2%}\n" + "\n".join(analysis),
            metadata={
                "failure_rate": failure_rate,
                "failure_analysis": failure_analysis,
                "error_patterns": error_patterns,
            },
        )
