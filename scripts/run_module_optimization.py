#!/usr/bin/env python3
"""
Run Module Optimization

Optimizes routing/workflow modules (modality, cross_modal, routing, workflow, unified)
with automatic DSPy optimizer selection and synthetic data generation.

Usage:
    # Optimize specific module
    uv run python scripts/run_module_optimization.py \\
        --module modality \\
        --tenant-id default \\
        --output results.json

    # Optimize all modules
    uv run python scripts/run_module_optimization.py \\
        --module all \\
        --tenant-id default \\
        --use-synthetic-data \\
        --output results.json
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer
from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import get_config
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_llm_config(tenant_id: str) -> LLMEndpointConfig:
    """Resolve LLM config from centralized config system."""
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    system_config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    return system_config.get_llm_config().primary


async def optimize_modality(
    tenant_id: str,
    dataset_name: str | None = None,
    use_synthetic: bool = False,
    lookback_hours: int = 24,
    min_confidence: float = 0.7,
    force_training: bool = False,
) -> Dict[str, Any]:
    """
    Optimize per-modality routing

    Args:
        tenant_id: Tenant identifier
        use_synthetic: Generate synthetic data if insufficient real data
        lookback_hours: Hours to look back for Phoenix spans
        min_confidence: Minimum confidence threshold
        force_training: Force training regardless of XGBoost decision

    Returns:
        Optimization results per modality
    """
    logger.info(f"🎯 Optimizing MODALITY routing for tenant '{tenant_id}'")

    llm_config = _get_llm_config(tenant_id)
    optimizer = ModalityOptimizer(llm_config=llm_config, tenant_id=tenant_id)
    results = await optimizer.optimize_all_modalities(
        lookback_hours=lookback_hours,
        min_confidence=min_confidence
    )

    return {
        "module": "modality",
        "tenant_id": tenant_id,
        "results_by_modality": {
            modality.value: result
            for modality, result in results.items()
        },
        "summary": {
            "total_modalities": len(results),
            "trained_count": sum(1 for r in results.values() if r.get("trained", False)),
            "skipped_count": sum(1 for r in results.values() if not r.get("trained", False))
        }
    }


async def optimize_cross_modal(
    tenant_id: str,
    dataset_name: str | None = None,
    use_synthetic: bool = False,
    lookback_hours: int = 24,
) -> Dict[str, Any]:
    """
    Optimize cross-modal fusion decisions

    Args:
        tenant_id: Tenant identifier
        use_synthetic: Generate synthetic data if insufficient real data
        lookback_hours: Hours to look back for Phoenix spans

    Returns:
        Optimization results
    """
    logger.info(f"🎯 Optimizing CROSS-MODAL fusion for tenant '{tenant_id}'")

    optimizer = CrossModalOptimizer(tenant_id=tenant_id)
    results = optimizer.train_fusion_model()

    return {
        "module": "cross_modal",
        "tenant_id": tenant_id,
        "results": results
    }


async def optimize_routing(
    tenant_id: str,
    dataset_name: str | None = None,
    use_synthetic: bool = False,
    lookback_hours: int = 24,
) -> Dict[str, Any]:
    """
    Optimize advanced entity-based routing

    Args:
        tenant_id: Tenant identifier
        use_synthetic: Generate synthetic data if insufficient real data
        lookback_hours: Hours to look back for Phoenix spans

    Returns:
        Optimization results
    """
    logger.info(f"🎯 Optimizing ROUTING (entity-based) for tenant '{tenant_id}'")

    llm_config = _get_llm_config(tenant_id)
    telemetry_provider = get_telemetry_manager().get_provider(tenant_id=tenant_id)
    optimizer = AdvancedRoutingOptimizer(
        tenant_id=tenant_id,
        llm_config=llm_config,
        telemetry_provider=telemetry_provider,
    )
    results = await optimizer.optimize_routing_policy()

    return {
        "module": "routing",
        "tenant_id": tenant_id,
        "results": results
    }


async def optimize_workflow(
    tenant_id: str,
    dataset_name: str | None = None,
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Optimize workflow planning and orchestration

    Args:
        tenant_id: Tenant identifier
        use_synthetic: Generate synthetic data if insufficient real data

    Returns:
        Optimization results
    """
    logger.info(f"🎯 Optimizing WORKFLOW orchestration for tenant '{tenant_id}'")

    raise NotImplementedError(
        "WorkflowOptimizer is not yet implemented. "
        "Implement the workflow optimization module before calling this function."
    )


async def optimize_unified(
    tenant_id: str,
    dataset_name: str | None = None,
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Optimize unified routing + workflow planning

    Args:
        tenant_id: Tenant identifier
        use_synthetic: Generate synthetic data if insufficient real data

    Returns:
        Optimization results
    """
    logger.info(f"🎯 Optimizing UNIFIED routing+workflow for tenant '{tenant_id}'")

    raise NotImplementedError(
        "UnifiedOptimizer is not yet implemented. "
        "Implement the unified routing+workflow optimization module before calling this function."
    )


async def optimize_all_modules(
    tenant_id: str,
    dataset_name: str | None = None,
    use_synthetic: bool = False,
    lookback_hours: int = 24,
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """
    Optimize all modules sequentially

    Args:
        tenant_id: Tenant identifier
        dataset_name: Phoenix dataset name (optional)
        use_synthetic: Generate synthetic data if insufficient real data
        lookback_hours: Hours to look back for Phoenix spans
        min_confidence: Minimum confidence threshold

    Returns:
        Combined optimization results
    """
    logger.info(f"🚀 Optimizing ALL modules for tenant '{tenant_id}'")

    results = {}

    # Run each optimizer
    try:
        results["modality"] = await optimize_modality(
            tenant_id, dataset_name, use_synthetic, lookback_hours, min_confidence
        )
    except Exception as e:
        logger.error(f"❌ Modality optimization failed: {e}")
        results["modality"] = {"status": "error", "error": str(e)}

    try:
        results["cross_modal"] = await optimize_cross_modal(
            tenant_id, dataset_name, use_synthetic, lookback_hours
        )
    except Exception as e:
        logger.error(f"❌ Cross-modal optimization failed: {e}")
        results["cross_modal"] = {"status": "error", "error": str(e)}

    try:
        results["routing"] = await optimize_routing(
            tenant_id, dataset_name, use_synthetic, lookback_hours
        )
    except Exception as e:
        logger.error(f"❌ Routing optimization failed: {e}")
        results["routing"] = {"status": "error", "error": str(e)}

    try:
        results["workflow"] = await optimize_workflow(tenant_id, dataset_name, use_synthetic)
    except Exception as e:
        logger.error(f"❌ Workflow optimization failed: {e}")
        results["workflow"] = {"status": "error", "error": str(e)}

    try:
        results["unified"] = await optimize_unified(tenant_id, dataset_name, use_synthetic)
    except Exception as e:
        logger.error(f"❌ Unified optimization failed: {e}")
        results["unified"] = {"status": "error", "error": str(e)}

    # Summary
    summary = {
        "total_modules": len(results),
        "successful": sum(1 for r in results.values() if r.get("status") != "error"),
        "failed": sum(1 for r in results.values() if r.get("status") == "error"),
    }

    return {
        "module": "all",
        "tenant_id": tenant_id,
        "summary": summary,
        "results": results
    }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Optimize routing/workflow modules with automatic DSPy optimizer selection"
    )
    parser.add_argument(
        "--module",
        required=True,
        choices=["modality", "cross_modal", "routing", "workflow", "unified", "all"],
        help="Module to optimize (or 'all' for all modules)"
    )
    parser.add_argument(
        "--tenant-id",
        default="default",
        help="Tenant identifier (default: 'default')"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Phoenix dataset name to use (if provided, takes precedence over traces/synthetic)"
    )
    parser.add_argument(
        "--use-synthetic-data",
        action="store_true",
        help="Generate synthetic training data if insufficient Phoenix traces"
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Hours to look back for Phoenix spans (default: 24)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for span collection (default: 0.7)"
    )
    parser.add_argument(
        "--force-training",
        action="store_true",
        help="Force training regardless of XGBoost decision model"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum DSPy training iterations (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/optimization_results.json"),
        help="Output JSON file path (default: /tmp/optimization_results.json)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🎯 Module Optimization")
    logger.info("=" * 80)
    logger.info(f"Module: {args.module}")
    logger.info(f"Tenant: {args.tenant_id}")
    logger.info(f"Dataset Name: {args.dataset_name or 'None (use traces/synthetic)'}")
    logger.info(f"Synthetic Data: {args.use_synthetic_data}")
    logger.info(f"Lookback Hours: {args.lookback_hours}")
    logger.info(f"Min Confidence: {args.min_confidence}")
    logger.info(f"Force Training: {args.force_training}")
    logger.info(f"Max Iterations: {args.max_iterations}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Run appropriate optimizer
        if args.module == "modality":
            results = await optimize_modality(
                args.tenant_id,
                args.dataset_name,
                args.use_synthetic_data,
                args.lookback_hours,
                args.min_confidence,
                args.force_training
            )
        elif args.module == "cross_modal":
            results = await optimize_cross_modal(
                args.tenant_id,
                args.dataset_name,
                args.use_synthetic_data,
                args.lookback_hours
            )
        elif args.module == "routing":
            results = await optimize_routing(
                args.tenant_id,
                args.dataset_name,
                args.use_synthetic_data,
                args.lookback_hours
            )
        elif args.module == "workflow":
            results = await optimize_workflow(
                args.tenant_id,
                args.dataset_name,
                args.use_synthetic_data
            )
        elif args.module == "unified":
            results = await optimize_unified(
                args.tenant_id,
                args.dataset_name,
                args.use_synthetic_data
            )
        elif args.module == "all":
            results = await optimize_all_modules(
                args.tenant_id,
                args.dataset_name,
                args.use_synthetic_data,
                args.lookback_hours,
                args.min_confidence
            )
        else:
            raise ValueError(f"Unknown module: {args.module}")

        # Add metadata
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output = {
            "module": args.module,
            "tenant_id": args.tenant_id,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "success": True,
            "results": results
        }

        # Write output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("=" * 80)
        logger.info(f"✅ Optimization completed in {duration:.1f}s")
        logger.info(f"📄 Results written to: {args.output}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}", exc_info=True)

        # Write error output
        error_output = {
            "module": args.module,
            "tenant_id": args.tenant_id,
            "timestamp": start_time.isoformat(),
            "success": False,
            "error": str(e)
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(error_output, f, indent=2)

        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
