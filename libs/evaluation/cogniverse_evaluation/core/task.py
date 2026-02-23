"""
Evaluation task implementation using Inspect AI.
"""

import logging
from datetime import datetime
from typing import Any

# Provider import moved to function scope to avoid circular deps
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

logger = logging.getLogger(__name__)


def evaluation_task(
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
        from cogniverse_evaluation.plugins import auto_register_plugins

        auto_register_plugins(config)
    elif "video" in dataset_name.lower() or any(
        p for p in (profiles or []) if "video" in p.lower() or "frame" in p.lower()
    ):
        # Auto-activate video plugin for video-related tasks
        from cogniverse_evaluation.plugins import register_video_plugin

        register_video_plugin()

    # Validate inputs based on mode
    if mode == "experiment" and not (profiles and strategies):
        raise ValueError("profiles and strategies required for experiment mode")

    # Load dataset from Phoenix using sync client directly
    # (avoids nested asyncio.run issues when called from async context)
    import pandas as pd
    import phoenix as px

    from cogniverse_evaluation.providers import get_evaluation_provider

    provider = get_evaluation_provider()

    sync_client = px.Client(endpoint=provider.http_endpoint)
    phoenix_dataset = sync_client.get_dataset(name=dataset_name)
    if phoenix_dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found or empty")
    dataset_data = phoenix_dataset.as_dataframe()

    # PhoenixDatasetStore.get_dataset() returns a DataFrame
    if isinstance(dataset_data, pd.DataFrame):
        if dataset_data.empty:
            raise ValueError(f"Dataset '{dataset_name}' is empty")

        # Convert DataFrame rows to Inspect AI Samples
        # Phoenix wraps CSV columns into a nested 'input' dict column
        # when no input_keys/output_keys specified during upload.
        # Format: {'input': {'query': '...', 'expected_videos': '...', ...}}
        samples = []
        for _, row in dataset_data.iterrows():
            # Handle Phoenix nested 'input' dict format
            if "input" in row.index and isinstance(row["input"], dict):
                record = row["input"]
            else:
                # Flat column format (direct CSV columns)
                record = row.to_dict()

            query = str(record.get("query", ""))
            if not query:
                continue

            expected_videos = record.get("expected_videos", "")
            # Handle comma-separated video IDs or single value
            if isinstance(expected_videos, str):
                target = [v.strip() for v in expected_videos.split(",") if v.strip()]
            elif isinstance(expected_videos, list):
                target = expected_videos
            else:
                target = [str(expected_videos)] if expected_videos else []

            sample = Sample(
                input=query,
                target=target,
                metadata={
                    "query_type": str(record.get("query_type", "general")),
                },
            )
            samples.append(sample)
    else:
        # Legacy dict format with "examples" key
        if not dataset_data or not dataset_data.get("examples"):
            raise ValueError(f"Dataset '{dataset_name}' not found or empty")

        samples = []
        examples = dataset_data.get("examples", [])
        for example in examples:
            query = example.get("input", {}).get("query", "")
            expected_videos = example.get("output", {}).get("expected_videos", [])

            sample = Sample(
                input=query,
                target=expected_videos,
                metadata={
                    "example_id": example.get("id", ""),
                    "category": example.get("input", {}).get("category", "general"),
                },
            )
            samples.append(sample)

    if not samples:
        raise ValueError(f"No valid samples in dataset '{dataset_name}'")

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


def run_evaluation(
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
        from cogniverse_evaluation.plugins.phoenix_experiment import (
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
    task = evaluation_task(
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
