"""
Evaluation task implementation using Inspect AI.
"""

import logging
from datetime import datetime
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

logger = logging.getLogger(__name__)


async def evaluation_task(
    mode: str,
    dataset_name: str,
    profiles: list[str] | None = None,
    strategies: list[str] | None = None,
    trace_ids: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> Task:
    """
    Unified evaluation task for all modes.

    Args:
        mode: One of "experiment", "batch", or "live"
        dataset_name: Phoenix dataset name
        profiles: Video processing profiles (for experiment mode)
        strategies: Ranking strategies (for experiment mode)
        trace_ids: Specific traces to evaluate (for batch mode)
        config: Additional configuration

    Returns:
        Configured Inspect AI Task that can be run with eval()
    """
    # Auto-register plugins based on config or dataset name
    if config and "plugins" in config.get("evaluation", {}):
        from cogniverse_dashboard.evaluation.plugins import auto_register_plugins

        auto_register_plugins(config)
    elif "video" in dataset_name.lower() or any(
        p for p in (profiles or []) if "video" in p.lower() or "frame" in p.lower()
    ):
        # Auto-activate video plugin for video-related tasks
        from cogniverse_dashboard.evaluation.plugins import register_video_plugin

        register_video_plugin()

    # Validate inputs based on mode
    if mode == "experiment" and not (profiles and strategies):
        raise ValueError("profiles and strategies required for experiment mode")

    from cogniverse_core.telemetry.manager import TelemetryManager

    telemetry_manager = TelemetryManager()
    provider = telemetry_manager.provider

    dataset_df = await provider.datasets.get_dataset(dataset_name)

    if dataset_df.empty:
        raise ValueError(f"Dataset '{dataset_name}' not found or empty")

    samples = []
    for idx, row in dataset_df.iterrows():
        query = row.get("query", "")

        expected_videos = []
        for field in ["expected_videos", "expected_items", "expected_video_ids"]:
            if field in row and row[field]:
                value = row[field]
                if isinstance(value, str):
                    expected_videos = [x.strip() for x in value.split(",") if x.strip()]
                elif isinstance(value, list):
                    expected_videos = value
                break

        sample = Sample(
            input=query,
            target=expected_videos,
            metadata={
                "example_id": str(idx),
                "category": row.get("category", "general"),
            },
        )
        samples.append(sample)

    # Create Inspect AI dataset
    dataset = MemoryDataset(samples)

    # Import solvers
    from .solvers import (
        create_batch_solver,
        create_live_solver,
        create_retrieval_solver,
    )

    # Choose solver based on mode
    if mode == "experiment":
        solver = create_retrieval_solver(profiles, strategies, config)
    elif mode == "batch":
        solver = create_batch_solver(trace_ids, config)
    elif mode == "live":
        solver = create_live_solver(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Get configured scorers adapted for Inspect AI
    from .inspect_scorers import get_configured_scorers

    scorers = get_configured_scorers(config or {})

    # Create the task
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        config=(
            GenerateConfig()
            if not config
            else GenerateConfig(
                **{
                    k: v
                    for k, v in (config or {}).items()
                    if k
                    in [
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "stop_seqs",
                        "max_retries",
                        "timeout",
                        "max_connections",
                    ]
                }
            )
        ),
        metadata={
            "mode": mode,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "profiles": profiles,
            "strategies": strategies,
        },
    )


async def run_evaluation(
    mode: str,
    dataset_name: str,
    profiles: list[str] | None = None,
    strategies: list[str] | None = None,
    trace_ids: list[str] | None = None,
    config: dict[str, Any] | None = None,
    use_phoenix_experiments: bool = False,
) -> dict[str, Any]:
    """
    Helper function to create and run evaluation task.

    Args:
        mode: Evaluation mode
        dataset_name: Dataset name
        profiles: Video processing profiles
        strategies: Ranking strategies
        trace_ids: Trace IDs for batch mode
        config: Configuration
        use_phoenix_experiments: Use Phoenix experiment API instead of Inspect AI

    Returns:
        Evaluation results dictionary
    """
    # If Phoenix experiment tracking requested
    if use_phoenix_experiments and mode == "experiment":
        from cogniverse_dashboard.evaluation.plugins.phoenix_experiment import (
            PhoenixExperimentPlugin,
            get_phoenix_evaluators,
        )

        # Get evaluators (these can be Inspect scorers wrapped for Phoenix)
        evaluators = get_phoenix_evaluators(config or {})

        # Run Inspect evaluation with Phoenix tracking
        # This uses Inspect AI's evaluation logic but stores everything in Phoenix
        return PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
            dataset_name=dataset_name,
            profiles=profiles,
            strategies=strategies,
            evaluators=evaluators,
            config=config,
        )

    # Use Inspect AI with its own logging system
    task = await evaluation_task(
        mode=mode,
        dataset_name=dataset_name,
        profiles=profiles,
        strategies=strategies,
        trace_ids=trace_ids,
        config=config,
    )

    # Run the evaluation - Inspect handles its own logging
    # Inspect logs to local filesystem for review and visualization
    results = eval(task)

    # Inspect creates EvalLog files that can be:
    # - Viewed with inspect view command
    # - Used for visualizations
    # - Analyzed for performance trends

    logger.info("Evaluation complete. Inspect logs saved locally.")
    if hasattr(results, "log_file"):
        logger.info(f"Log file: {results.log_file}")

    # Process and return results
    return {
        "mode": mode,
        "dataset": dataset_name,
        "results": results,  # This includes Inspect's EvalLog
        "timestamp": datetime.now().isoformat(),
    }
