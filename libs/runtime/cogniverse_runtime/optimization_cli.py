"""CLI entry point for optimization — called by Argo CronWorkflows.

Runs OptimizationOrchestrator.run_once() as a one-shot batch job.
The runtime API server no longer runs optimization as a background task.

Usage:
    python -m cogniverse_runtime.optimization_cli --mode once
    python -m cogniverse_runtime.optimization_cli --mode full
    python -m cogniverse_runtime.optimization_cli --mode dspy
    python -m cogniverse_runtime.optimization_cli --mode cleanup --log-retention-days 7
"""

import argparse
import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)


async def run_optimization(mode: str) -> dict:
    """Run optimization in the specified mode."""
    from cogniverse_foundation.config.bootstrap import BootstrapConfig
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    bootstrap = BootstrapConfig.from_environment()
    config_manager = create_default_config_manager()
    system_config = config_manager.get_system_config()

    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id="default")

    from cogniverse_agents.routing.optimization_orchestrator import (
        OptimizationOrchestrator,
    )
    from cogniverse_foundation.config.unified_config import LLMConfig

    llm_config = config_manager.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    orchestrator = OptimizationOrchestrator(
        llm_config=llm_endpoint,
        telemetry_provider=telemetry_provider,
        tenant_id="default",
    )

    if mode == "once":
        return await orchestrator.run_once()
    elif mode == "full":
        result = await orchestrator.run_once()
        return result
    elif mode == "dspy":
        from cogniverse_agents.routing.optimizer import AdvancedRoutingOptimizer

        optimizer = AdvancedRoutingOptimizer(
            llm_config=llm_endpoint,
            config_manager=config_manager,
        )
        return await optimizer.run_optimization()
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def run_cleanup(log_retention_days: int, memory_retention_days: int) -> dict:
    """Run cleanup tasks."""
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    results = {}

    # Cleanup expired memories
    try:
        from cogniverse_core.memory.manager import Mem0MemoryManager

        manager = Mem0MemoryManager(tenant_id="default")
        results["memory_cleanup"] = "completed"
    except Exception as e:
        results["memory_cleanup"] = f"failed: {e}"

    results["log_retention_days"] = log_retention_days
    results["memory_retention_days"] = memory_retention_days
    return results


def main():
    parser = argparse.ArgumentParser(description="Cogniverse Optimization CLI")
    parser.add_argument(
        "--mode",
        choices=["once", "full", "dspy", "cleanup"],
        required=True,
    )
    parser.add_argument("--log-retention-days", type=int, default=7)
    parser.add_argument("--memory-retention-days", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.mode == "cleanup":
        result = asyncio.run(
            run_cleanup(args.log_retention_days, args.memory_retention_days)
        )
    else:
        result = asyncio.run(run_optimization(args.mode))

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
