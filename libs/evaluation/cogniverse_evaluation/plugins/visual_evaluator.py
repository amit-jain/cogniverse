"""
Visual evaluator plugin for video evaluation.

This plugin provides visual judge and quality evaluators for video search results.
"""

import logging
from typing import Any

from inspect_ai.scorer import Score, scorer

logger = logging.getLogger(__name__)


class VisualEvaluatorPlugin:
    """Plugin for visual evaluation capabilities."""

    @staticmethod
    @scorer(metrics=[])
    def create_visual_judge_scorer(evaluator_name: str = "visual_judge"):
        """
        Create a visual judge scorer for video evaluation.

        Args:
            evaluator_name: Name of the visual evaluator configuration

        Returns:
            Scorer function for Inspect AI
        """

        async def score(state, target=None) -> Score:
            """Score video search results using visual judge."""
            from cogniverse_evaluation.evaluators.configurable_visual_judge import (
                ConfigurableVisualJudge,
            )
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
                get_config,
            )

            config_manager = create_default_config_manager()
            config = get_config(tenant_id="default", config_manager=config_manager)
            evaluator_config = config.get("evaluators", {}).get(evaluator_name, {})

            if not evaluator_config:
                return Score(
                    value=0.0,
                    explanation=f"Visual evaluator '{evaluator_name}' not configured",
                )

            visual_judge = ConfigurableVisualJudge(
                provider=evaluator_config.get("provider", "ollama"),
                model=evaluator_config.get("model"),
                base_url=evaluator_config.get("base_url"),
                api_key=evaluator_config.get("api_key"),
                temperature=evaluator_config.get("temperature", 0.1),
                frames_to_extract=evaluator_config.get("frames_to_extract", 3),
            )

            query = (
                state.input.get("query", "")
                if hasattr(state.input, "get")
                else str(state.input)
            )
            all_scores = {}

            for config_key, output in state.outputs.items():
                if not output.get("success", False):
                    all_scores[config_key] = 0.0
                    continue

                results = output.get("results", [])
                if not results:
                    all_scores[config_key] = 0.0
                    continue

                top_result = results[0]
                video_id = top_result.get("video_id", "")

                if video_id:
                    try:
                        eval_result = visual_judge.evaluate(
                            input={"query": query},
                            output={"video_id": video_id, "results": results},
                        )
                        all_scores[config_key] = (
                            eval_result.score if eval_result else 0.0
                        )
                    except Exception as e:
                        logger.error(f"Visual evaluation failed: {e}")
                        all_scores[config_key] = 0.0
                else:
                    all_scores[config_key] = 0.0

            avg_score = (
                sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
            )

            return Score(
                value=avg_score,
                explanation="Visual judge evaluation for video results",
                metadata={
                    "evaluator": evaluator_name,
                    "individual_scores": all_scores,
                    "plugin": "visual_evaluator",
                },
            )

        return score

    @staticmethod
    @scorer(metrics=[])
    def create_quality_scorer():
        """
        Create quality evaluators scorer for video evaluation.

        Returns:
            Scorer function for Inspect AI
        """

        async def score(state, target=None) -> Score:
            """Score using video quality evaluators."""
            from cogniverse_evaluation.evaluators.sync_reference_free import (
                create_sync_evaluators,
            )

            evaluators = create_sync_evaluators()

            if not evaluators:
                return Score(
                    value=0.0, explanation="No video quality evaluators available"
                )

            query = (
                state.input.get("query", "")
                if hasattr(state.input, "get")
                else str(state.input)
            )
            all_scores = {}

            for config_key, output in state.outputs.items():
                if not output.get("success", False):
                    all_scores[config_key] = {}
                    continue

                results = output.get("results", [])
                config_scores = {}

                for evaluator in evaluators:
                    try:
                        eval_result = evaluator.evaluate(
                            input={"query": query}, output={"results": results}
                        )
                        config_scores[evaluator.__class__.__name__] = (
                            eval_result.score if eval_result else 0.0
                        )
                    except Exception as e:
                        logger.error(f"Quality evaluation failed: {e}")
                        config_scores[evaluator.__class__.__name__] = 0.0

                all_scores[config_key] = config_scores

            total_scores = []
            for config_scores in all_scores.values():
                total_scores.extend(config_scores.values())

            avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0

            return Score(
                value=avg_score,
                explanation="Video quality evaluation scores",
                metadata={"scores_by_config": all_scores, "plugin": "visual_evaluator"},
            )

        return score


def register():
    """Register the visual evaluator plugin."""
    logger.info("Visual evaluator plugin registered")

    # Could register with a central registry if needed
    # For now, the plugin is available for import

    return True


def get_visual_scorers(config: dict[str, Any]) -> list:
    """
    Get visual evaluator scorers based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of visual scorers
    """
    scorers = []

    if config.get("enable_llm_evaluators", False):
        evaluator_name = config.get("evaluator_name", "visual_judge")
        scorers.append(VisualEvaluatorPlugin.create_visual_judge_scorer(evaluator_name))

    if config.get("enable_quality_evaluators", False):
        scorers.append(VisualEvaluatorPlugin.create_quality_scorer())

    return scorers
