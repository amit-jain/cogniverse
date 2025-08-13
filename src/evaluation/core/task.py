"""
Main evaluation task orchestrator using Inspect AI.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
import phoenix as px

from .solvers import retrieval_solver, trace_loader_solver, live_trace_solver
from .scorers import get_configured_scorers

logger = logging.getLogger(__name__)


@task
def evaluation_task(
    mode: str,
    dataset_name: str,
    profiles: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    trace_ids: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Task:
    """
    Unified evaluation task for all modes.
    
    This is the main entry point for all evaluation workflows:
    - experiment: Run new searches with different configurations
    - batch: Evaluate existing traces
    - live: Real-time evaluation of incoming traces
    
    Args:
        mode: One of "experiment", "batch", or "live"
        dataset_name: Phoenix dataset name
        profiles: Video processing profiles (for experiment mode)
        strategies: Ranking strategies (for experiment mode)
        trace_ids: Specific traces to evaluate (for batch mode)
        config: Additional configuration for scorers and solvers
        
    Returns:
        Configured Inspect AI Task
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Validate mode
    valid_modes = ["experiment", "batch", "live"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    # Validate mode-specific requirements
    if mode == "experiment":
        if not profiles or not strategies:
            raise ValueError("profiles and strategies required for experiment mode")
    
    if mode == "batch" and not trace_ids:
        logger.warning("No trace_ids provided for batch mode, will evaluate recent traces")
    
    # Load dataset from Phoenix
    try:
        phoenix_client = px.Client()
        dataset = phoenix_client.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found in Phoenix")
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {e}")
    
    logger.info(f"Loaded dataset '{dataset_name}' with {len(dataset.examples)} examples")
    
    # Choose solver based on mode
    if mode == "experiment":
        solver = retrieval_solver(profiles, strategies, config)
        logger.info(f"Using retrieval solver for {len(profiles)} profiles and {len(strategies)} strategies")
    elif mode == "batch":
        solver = trace_loader_solver(trace_ids, config)
        logger.info(f"Using trace loader solver for {len(trace_ids) if trace_ids else 'recent'} traces")
    elif mode == "live":
        solver = live_trace_solver(config)
        logger.info("Using live trace solver for real-time evaluation")
    
    # Get configured scorers
    scorers = get_configured_scorers(config)
    logger.info(f"Configured {len(scorers)} scorers")
    
    # Create task with metadata
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        metadata={
            "mode": mode,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "profiles": profiles,
            "strategies": strategies,
            "trace_ids": trace_ids[:5] if trace_ids else None  # Log first 5 for brevity
        }
    )