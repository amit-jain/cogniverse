"""
Visual evaluator plugin for video evaluation.

This plugin provides visual judge and quality evaluators for video search results.
"""

import logging
from typing import Any, Optional

from inspect_ai.scorer import Score, scorer

logger = logging.getLogger(__name__)


class VisualEvaluatorPlugin:
    """Plugin for visual evaluation capabilities."""

    @staticmethod
    @scorer(metrics=[])
    def create_visual_judge_scorer(
        evaluator_name: str = "visual_judge",
        tenant_id: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Create a visual judge scorer for video evaluation.

        Args:
            evaluator_name: Name of the visual evaluator configuration
            tenant_id: Tenant whose evaluator/media config to use (defaults to
                the system tenant); threaded from the experiment config.
            model: Optional model override (the experiment's --llm-model).
            base_url: Optional base-URL override.

        Returns:
            Scorer function for Inspect AI
        """

        async def score(state, target=None) -> Score:
            """Score video search results using visual judge."""
            from cogniverse_core.common.media import MediaConfig, MediaLocator
            from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
            from cogniverse_evaluation.evaluators.configurable_visual_judge import (
                ConfigurableVisualJudge,
            )
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
                get_config,
            )

            resolved_tenant = tenant_id or SYSTEM_TENANT_ID
            config_manager = create_default_config_manager()
            config = get_config(
                tenant_id=resolved_tenant, config_manager=config_manager
            )
            evaluator_config = config.get("evaluators", {}).get(evaluator_name, {})

            if not evaluator_config:
                return Score(
                    value=0.0,
                    explanation=f"Visual evaluator '{evaluator_name}' not configured",
                )

            media_section = config.get("media", {})
            media_config = (
                MediaConfig.from_dict(media_section) if media_section else MediaConfig()
            )
            locator = MediaLocator(tenant_id=resolved_tenant, config=media_config)
            visual_judge = ConfigurableVisualJudge(
                locator=locator,
                evaluator_name=evaluator_name,
                tenant_id=resolved_tenant,
                model=model,
                base_url=base_url,
            )

            query = (
                state.input.get("query", "")
                if hasattr(state.input, "get")
                else str(state.input)
            )
            # Genuine zeros (no results / no video id) score 0.0; judge
            # FAILURES are excluded from the mean and surfaced in metadata —
            # folding them in as 0.0 would make a judge outage read as a
            # uniform quality collapse across every configuration.
            all_scores = {}
            failed = {}
            judged_ok = 0

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
                if not video_id:
                    all_scores[config_key] = 0.0
                    continue

                try:
                    eval_result = visual_judge.evaluate(
                        input={"query": query},
                        output={"video_id": video_id, "results": results},
                    )
                except Exception as e:
                    logger.error(f"Visual evaluation failed: {e}")
                    failed[config_key] = str(e)
                    continue

                if eval_result is None:
                    failed[config_key] = "visual judge returned no result"
                    continue

                all_scores[config_key] = eval_result.score
                judged_ok += 1

            if failed and not judged_ok:
                raise RuntimeError(
                    "visual judge failed for every attempted configuration — "
                    f"refusing to report a score for a judge outage: {failed}"
                )

            avg_score = (
                sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
            )

            return Score(
                value=avg_score,
                explanation="Visual judge evaluation for video results",
                metadata={
                    "evaluator": evaluator_name,
                    "individual_scores": all_scores,
                    "failed_evaluations": failed,
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
        scorers.append(
            VisualEvaluatorPlugin.create_visual_judge_scorer(
                evaluator_name,
                tenant_id=config.get("tenant_id"),
                model=config.get("llm_model"),
                base_url=config.get("llm_base_url"),
            )
        )

    if config.get("enable_quality_evaluators", False):
        scorers.append(VisualEvaluatorPlugin.create_quality_scorer())

    return scorers
