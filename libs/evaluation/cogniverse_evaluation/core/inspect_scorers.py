"""
Scorers adapted for Inspect AI's interface.

These scorers unpack the structured output from our solvers and evaluate
the search results using the same logic as our original scorers.
"""

import logging
from typing import Any

from inspect_ai.scorer import Score, Target, mean, scorer

from .solver_output import unpack_solver_output

logger = logging.getLogger(__name__)


def get_configured_scorers(config: dict[str, Any]) -> list:
    """Get list of scorers based on configuration."""
    if config is None:
        config = {}

    scorers = []

    # Add scorers based on config
    if config.get("use_relevance", True):
        scorers.append(relevance_scorer())

    if config.get("use_diversity", True):
        scorers.append(diversity_scorer())

    if config.get("use_result_count", True):
        scorers.append(result_count_scorer())

    # Always have at least one scorer
    if not scorers:
        scorers.append(relevance_scorer())

    return scorers


@scorer(metrics=[mean()])
def relevance_scorer():
    """
    Relevance scorer that unpacks solver output and evaluates keyword matching.
    """

    async def score(state, target: Target) -> Score:
        try:
            # Extract the completion string from state.output.choices[0].message.content
            output_str = ""
            if state.output and state.output.choices and len(state.output.choices) > 0:
                output_str = state.output.choices[0].message.content or ""

            # Unpack the structured output
            eval_output = unpack_solver_output(output_str)

            if not eval_output.query:
                return Score(value=0.0, explanation="No query found in output")

            # Calculate relevance for each configuration
            config_scores = {}

            for config_key, results_data in eval_output.search_configs.items():
                if not results_data.get("success", False):
                    config_scores[config_key] = 0.0
                    continue

                results = results_data.get("results", [])
                if not results:
                    config_scores[config_key] = 0.0
                    continue

                # Extract text content and calculate relevance
                contexts = []
                for r in results:
                    content = (
                        r.get("content")
                        or r.get("text")
                        or r.get("description")
                        or r.get("transcript")
                        or ""
                    )
                    if content:
                        contexts.append(str(content))

                # Simple keyword matching
                if contexts:
                    relevance = _calculate_keyword_relevance(
                        eval_output.query, contexts
                    )
                    config_scores[config_key] = relevance
                else:
                    config_scores[config_key] = 0.0

            # Average across all configurations
            avg_score = (
                sum(config_scores.values()) / len(config_scores)
                if config_scores
                else 0.0
            )

            explanation = f"Relevance scores: {', '.join(f'{k}={v:.3f}' for k, v in config_scores.items())}"

            return Score(
                value=avg_score,
                explanation=explanation,
                metadata={
                    "individual_scores": config_scores,
                    "phoenix_trace_id": eval_output.phoenix_trace_id,
                },
            )

        except Exception as e:
            logger.error(f"Error in relevance scorer: {e}")
            return Score(value=0.0, explanation=f"Scorer error: {e}")

    return score


@scorer(metrics=[mean()])
def diversity_scorer():
    """
    Diversity scorer that measures uniqueness of results.
    """

    async def score(state, target: Target) -> Score:
        try:
            # Extract the completion string from state.output.choices[0].message.content
            output_str = ""
            if state.output and state.output.choices and len(state.output.choices) > 0:
                output_str = state.output.choices[0].message.content or ""

            # Unpack the structured output
            eval_output = unpack_solver_output(output_str)

            config_scores = {}

            for config_key, results_data in eval_output.search_configs.items():
                if not results_data.get("success", False):
                    config_scores[config_key] = 0.0
                    continue

                results = results_data.get("results", [])
                if not results:
                    config_scores[config_key] = 0.0
                    continue

                # Extract unique IDs
                item_ids = []
                for r in results:
                    # Try different ID fields
                    item_id = (
                        r.get("video_id")
                        or r.get("item_id")
                        or r.get("document_id")
                        or r.get("id")
                    )
                    if item_id:
                        item_ids.append(item_id)

                # Calculate diversity
                if item_ids:
                    unique_ids = set(item_ids)
                    diversity = len(unique_ids) / len(item_ids)
                    config_scores[config_key] = diversity
                else:
                    config_scores[config_key] = 0.0

            avg_score = (
                sum(config_scores.values()) / len(config_scores)
                if config_scores
                else 0.0
            )

            explanation = f"Diversity scores: {', '.join(f'{k}={v:.3f}' for k, v in config_scores.items())}"

            return Score(
                value=avg_score,
                explanation=explanation,
                metadata={
                    "individual_scores": config_scores,
                    "phoenix_trace_id": eval_output.phoenix_trace_id,
                },
            )

        except Exception as e:
            logger.error(f"Error in diversity scorer: {e}")
            return Score(value=0.0, explanation=f"Scorer error: {e}")

    return score


@scorer(metrics=[mean()])
def result_count_scorer():
    """
    Simple scorer that checks if we got results.
    """

    async def score(state, target: Target) -> Score:
        try:
            # Extract the completion string from state.output.choices[0].message.content
            output_str = ""
            if state.output and state.output.choices and len(state.output.choices) > 0:
                output_str = state.output.choices[0].message.content or ""

            # Unpack the structured output
            eval_output = unpack_solver_output(output_str)

            config_scores = {}

            for config_key, results_data in eval_output.search_configs.items():
                if not results_data.get("success", False):
                    config_scores[config_key] = 0.0
                    continue

                results = results_data.get("results", [])
                # Score based on whether we got results (normalized)
                config_scores[config_key] = min(len(results) / 10.0, 1.0)

            avg_score = (
                sum(config_scores.values()) / len(config_scores)
                if config_scores
                else 0.0
            )

            explanation = f"Result count scores: {', '.join(f'{k}={v:.3f}' for k, v in config_scores.items())}"

            return Score(
                value=avg_score,
                explanation=explanation,
                metadata={
                    "individual_scores": config_scores,
                    "phoenix_trace_id": eval_output.phoenix_trace_id,
                },
            )

        except Exception as e:
            logger.error(f"Error in result count scorer: {e}")
            return Score(value=0.0, explanation=f"Scorer error: {e}")

    return score


def _calculate_keyword_relevance(query: str, contexts: list[str]) -> float:
    """Calculate keyword-based relevance score."""
    if not query or not contexts:
        return 0.0

    # Simple keyword matching
    query_words = set(query.lower().split())

    scores = []
    for context in contexts:
        context_words = set(context.lower().split())
        if context_words:
            overlap = len(query_words.intersection(context_words))
            score = overlap / len(query_words)
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0
