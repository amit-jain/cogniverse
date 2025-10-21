"""
Phoenix experiment integration plugin.

This plugin ensures Inspect AI evaluations are properly tracked in Phoenix experiments.
Phoenix handles storage/tracking while Inspect AI handles evaluation execution.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import phoenix as px
from phoenix.experiments import run_experiment

logger = logging.getLogger(__name__)


class PhoenixExperimentPlugin:
    """Plugin to integrate Phoenix experiments with the new evaluation framework."""

    @staticmethod
    def wrap_inspect_task_for_phoenix(
        inspect_solver,
        profiles: list[str],
        strategies: list[str],
        config: dict[str, Any],
    ) -> Callable:
        """
        Wrap an Inspect AI solver to work with Phoenix experiments.

        Args:
            inspect_solver: The Inspect AI solver
            profiles: Video processing profiles
            strategies: Ranking strategies
            config: Configuration

        Returns:
            Task function compatible with Phoenix run_experiment
        """

        def phoenix_task(example):
            """Phoenix-compatible task that uses Inspect solver logic."""
            query = (
                example.input.get("query", "")
                if hasattr(example, "input")
                else example.get("query", "")
            )

            # Import search service
            from cogniverse_agents.search.service import SearchService

            from cogniverse_core.config.utils import get_config

            main_config = get_config()
            all_results = {}

            # Run searches for each profile/strategy (similar to Inspect solver)
            for profile in profiles:
                for strategy in strategies:
                    config_key = f"{profile}_{strategy}"

                    try:
                        search_service = SearchService(main_config, profile)
                        search_results = search_service.search(
                            query=query,
                            top_k=config.get("top_k", 10),
                            ranking_strategy=strategy,
                        )

                        formatted_results = []
                        for i, result in enumerate(search_results):
                            result_dict = (
                                result.to_dict()
                                if hasattr(result, "to_dict")
                                else result
                            )
                            video_id = result_dict.get("source_id", "")
                            if not video_id and "document_id" in result_dict:
                                doc_id = result_dict["document_id"]
                                if "_frame_" in doc_id:
                                    video_id = doc_id.split("_frame_")[0]
                                else:
                                    video_id = doc_id

                            formatted_results.append(
                                {
                                    "video_id": video_id,
                                    "score": float(
                                        result_dict.get("score", 1.0 / (i + 1))
                                    ),
                                    "rank": i + 1,
                                    "content": result_dict.get("content", ""),
                                }
                            )

                        all_results[config_key] = {
                            "results": formatted_results,
                            "profile": profile,
                            "strategy": strategy,
                            "success": True,
                            "count": len(formatted_results),
                        }

                    except Exception as e:
                        logger.error(f"Search failed for {config_key}: {e}")
                        all_results[config_key] = {
                            "results": [],
                            "profile": profile,
                            "strategy": strategy,
                            "success": False,
                            "error": str(e),
                        }

            return {
                "query": query,
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
            }

        return phoenix_task

    @staticmethod
    def run_inspect_with_phoenix_tracking(
        dataset_name: str,
        profiles: list[str],
        strategies: list[str],
        evaluators: list[Any],
        config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Run Inspect AI evaluation with Phoenix experiment tracking.

        This is the RIGHT way to combine them:
        - Inspect AI handles the evaluation logic (solvers, scorers)
        - Phoenix tracks the experiment, stores results, maintains history

        Args:
            dataset_name: Phoenix dataset name
            profiles: Video processing profiles
            strategies: Ranking strategies
            evaluators: List of Phoenix evaluators (can include Inspect scorers)
            config: Additional configuration

        Returns:
            Experiment results with Phoenix tracking
        """
        config = config or {}
        client = px.Client()

        # Get dataset from Phoenix (Phoenix as storage)
        dataset = client.get_dataset(name=dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found in Phoenix")

        # Create task that combines Inspect AI solvers with Phoenix tracking
        from cogniverse_core.evaluation.core.solvers import create_retrieval_solver

        inspect_solver = create_retrieval_solver(profiles, strategies, config)

        # Wrap Inspect solver for Phoenix compatibility
        phoenix_task = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
            inspect_solver, profiles, strategies, config
        )

        # Run through Phoenix experiment API (Phoenix as tracker)
        # This creates an experiment in Phoenix that tracks the Inspect evaluation
        experiment_name = (
            f"inspect_eval_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        logger.info(
            f"Running Inspect AI evaluation tracked by Phoenix experiment: {experiment_name}"
        )

        result = run_experiment(
            dataset=dataset,
            task=phoenix_task,  # Inspect logic wrapped for Phoenix
            evaluators=evaluators,  # Can include both Phoenix and Inspect evaluators
            experiment_name=experiment_name,
            experiment_metadata={
                "profiles": profiles,
                "strategies": strategies,
                "config": config,
                "framework": "inspect_ai",
                "storage": "phoenix",
            },
        )

        # Result is stored in Phoenix and returned
        return result


def register():
    """Register the Phoenix experiment plugin."""
    logger.info("Phoenix experiment plugin registered")
    return True


def get_phoenix_evaluators(config: dict[str, Any]) -> list[Any]:
    """
    Get Phoenix-compatible evaluators based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of Phoenix evaluators
    """
    evaluators = []

    # Add visual evaluators if configured
    if config.get("enable_llm_evaluators", False):
        from cogniverse_core.evaluation.evaluators.configurable_visual_judge import (
            ConfigurableVisualJudge,
        )

        evaluator_name = config.get("evaluator_name", "visual_judge")
        evaluator_config = config.get("evaluators", {}).get(evaluator_name, {})

        visual_judge = ConfigurableVisualJudge(
            provider=evaluator_config.get("provider", "ollama"),
            model=evaluator_config.get("model"),
            base_url=evaluator_config.get("base_url"),
            api_key=evaluator_config.get("api_key"),
        )
        evaluators.append(visual_judge)

    # Add quality evaluators if configured
    if config.get("enable_quality_evaluators", False):
        from cogniverse_core.evaluation.evaluators.sync_reference_free import (
            create_sync_evaluators,
        )

        quality_evaluators = create_sync_evaluators()
        evaluators.extend(quality_evaluators)

    return evaluators
