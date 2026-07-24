"""
Evaluation task implementation using Inspect AI.
"""

import logging
from datetime import datetime, timezone
from typing import Any

# Provider import moved to function scope to avoid circular deps
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

logger = logging.getLogger(__name__)

# Dataset frames cached per (endpoint, dataset_name) — see evaluation_task.
_DATASET_FRAMES: dict[tuple[str, str], Any] = {}


def _row_to_record(row: Any) -> dict[str, Any]:
    """Flatten a Phoenix dataset row into one record.

    Phoenix ``to_dataframe()`` splits a dataset into ``input`` / ``output`` /
    ``metadata`` dict-columns keyed by the writer's ``input_keys`` /
    ``output_keys``. ``DatasetManager`` writes ``expected_videos`` under
    ``output_keys`` while ``ExperimentTracker`` writes everything under
    ``input`` — so the reader must merge all three slots, or a
    DatasetManager-created dataset yields an empty target for every sample.
    """
    record: dict[str, Any] = {}
    for slot in ("metadata", "output", "input"):
        if slot in getattr(row, "index", ()) and isinstance(row[slot], dict):
            record.update(row[slot])
    if not record:
        # Flat column format (direct CSV columns, no nesting)
        record = row.to_dict()
    return record


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
    # (avoids nested asyncio.run issues when called from async context).
    # Cached per (endpoint, dataset): an experiment sweep builds one task per
    # profile x strategy combination, so the identical dataset is fetched once
    # rather than per task.
    import pandas as pd

    from cogniverse_evaluation.providers import get_evaluation_provider

    provider = get_evaluation_provider()

    cache_key = (provider.http_endpoint, dataset_name)
    dataset_data = _DATASET_FRAMES.get(cache_key)
    if dataset_data is None:
        from phoenix.client import Client as PhoenixSyncClient

        sync_client = PhoenixSyncClient(base_url=provider.http_endpoint)
        phoenix_dataset = sync_client.datasets.get_dataset(dataset=dataset_name)
        if phoenix_dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found or empty")
        dataset_data = phoenix_dataset.to_dataframe()
        _DATASET_FRAMES[cache_key] = dataset_data

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
            record = _row_to_record(row)

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

            # query_type is the canonical metadata key; category (the
            # DatasetManager column) is the fallback.
            query_type = record.get("query_type") or record.get("category") or "general"
            sample = Sample(
                input=query,
                target=target,
                metadata={"query_type": str(query_type)},
            )
            samples.append(sample)
    else:
        raise TypeError(
            f"Expected a DataFrame from Phoenix for dataset '{dataset_name}', "
            f"got {type(dataset_data).__name__}"
        )

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profiles": profiles,
            "strategies": strategies,
        },
    )
