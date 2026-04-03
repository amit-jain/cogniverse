"""CLI entry point for optimization — called by Argo CronWorkflows.

Runs OptimizationOrchestrator.run_once() as a one-shot batch job.
The runtime API server no longer runs optimization as a background task.

Usage:
    python -m cogniverse_runtime.optimization_cli --mode once
    python -m cogniverse_runtime.optimization_cli --mode full
    python -m cogniverse_runtime.optimization_cli --mode dspy
    python -m cogniverse_runtime.optimization_cli --mode cleanup --log-retention-days 7
    python -m cogniverse_runtime.optimization_cli --mode triggered \
        --tenant-id default --agents search,summary \
        --trigger-dataset optimization-trigger-default-20260403_040000
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


async def run_optimization(mode: str) -> dict:
    """Run optimization in the specified mode."""
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    config_manager = create_default_config_manager()

    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id="default")

    from cogniverse_agents.routing.optimization_orchestrator import (
        OptimizationOrchestrator,
    )

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


async def run_triggered_optimization(
    tenant_id: str, agents: list[str], trigger_dataset: str
) -> dict:
    """Run optimization triggered by quality monitor.

    Loads scored examples from Phoenix trigger dataset, then compiles
    DSPy modules for each flagged agent using those examples as training data.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    # Load trigger dataset from Phoenix
    import phoenix as px

    system_config = config_manager.get_system_config()
    phoenix_endpoint = system_config.telemetry_url
    sync_client = px.Client(endpoint=phoenix_endpoint)

    try:
        dataset = sync_client.get_dataset(name=trigger_dataset)
        trigger_df = dataset.as_dataframe()
    except Exception as e:
        logger.error(f"Failed to load trigger dataset '{trigger_dataset}': {e}")
        return {"status": "failed", "error": str(e)}

    results = {}
    llm_config = config_manager.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    for agent_name in agents:
        agent_df = trigger_df[trigger_df["agent"] == agent_name]
        if agent_df.empty:
            logger.info(f"No training data for agent '{agent_name}', skipping")
            results[agent_name] = {"status": "skipped", "reason": "no_data"}
            continue

        low_scoring = agent_df[agent_df["category"] == "low_scoring"]
        high_scoring = agent_df[agent_df["category"] == "high_scoring"]

        logger.info(
            f"Optimizing {agent_name}: "
            f"{len(low_scoring)} negative, {len(high_scoring)} positive examples"
        )

        try:
            result = await _optimize_agent(
                agent_name=agent_name,
                low_scoring_df=low_scoring,
                high_scoring_df=high_scoring,
                llm_endpoint=llm_endpoint,
                config_manager=config_manager,
                telemetry_provider=telemetry_provider,
                tenant_id=tenant_id,
            )
            results[agent_name] = result
        except Exception as e:
            logger.error(f"Optimization failed for {agent_name}: {e}")
            results[agent_name] = {"status": "failed", "error": str(e)}

    # Post-optimization: run golden eval to verify improvement
    try:
        from cogniverse_evaluation.quality_monitor import QualityMonitor

        monitor = QualityMonitor(
            tenant_id=tenant_id,
            runtime_url=system_config.agent_registry_url,
            phoenix_http_endpoint=phoenix_endpoint,
            llm_base_url=llm_endpoint.api_base or "http://localhost:11434",
            llm_model=llm_endpoint.model,
            golden_dataset_path="data/testset/evaluation/sample_videos_retrieval_queries.json",
        )
        post_eval = await monitor.evaluate_golden_set()
        results["post_optimization_eval"] = {
            "mrr": post_eval.mean_mrr,
            "ndcg": post_eval.mean_ndcg,
            "precision_at_5": post_eval.mean_precision_at_5,
        }

        # Update baseline if scores improved
        if post_eval.mean_mrr > (monitor._last_golden_baseline_mrr or 0):
            await monitor.update_baseline(golden_result=post_eval)
            results["baseline_updated"] = True

        # Grow golden set with high-scoring live queries
        new_golden_candidates = []
        for _, row in high_scoring.iterrows():
            if row.get("score", 0) >= 0.8:
                new_golden_candidates.append(
                    {
                        "query": row.get("query", ""),
                        "expected_videos": [],
                        "ground_truth": "",
                        "query_type": "live_traffic",
                        "source": "quality_monitor",
                    }
                )
        if new_golden_candidates:
            await monitor.grow_golden_set(new_golden_candidates)
            results["golden_set_growth"] = len(new_golden_candidates)

        await monitor.close()

    except Exception as e:
        logger.warning(f"Post-optimization eval failed: {e}")

    return results


async def _optimize_agent(
    agent_name: str,
    low_scoring_df,
    high_scoring_df,
    llm_endpoint,
    config_manager,
    telemetry_provider,
    tenant_id: str,
) -> dict:
    """Run DSPy optimization for a specific agent using scored examples."""
    import json as _json

    from cogniverse_agents.optimizer.dspy_agent_optimizer import (
        DSPyAgentPromptOptimizer,
    )

    optimizer = DSPyAgentPromptOptimizer()
    optimizer.initialize_language_model(llm_endpoint)

    # Build DSPy training examples from scored data
    import dspy

    trainset = []
    for _, row in high_scoring_df.iterrows():
        query = row.get("query", "")
        output = row.get("output", "{}")
        if isinstance(output, str):
            try:
                output = _json.loads(output)
            except Exception:
                output = {}

        if agent_name == "search":
            example = dspy.Example(
                query=query,
                modality="video",
                top_k=10,
                search_strategy="colpali",
                enhanced_query=query,
                confidence=row.get("score", 0.8),
            ).with_inputs("query", "modality", "top_k")
        elif agent_name == "summary":
            example = dspy.Example(
                content=_json.dumps(output, default=str),
                summary_type="comprehensive",
                target_audience="general",
                summary=output.get("summary", ""),
                key_points=str(output.get("key_points", [])),
                confidence=row.get("score", 0.8),
            ).with_inputs("content", "summary_type", "target_audience")
        elif agent_name == "report":
            example = dspy.Example(
                search_results=_json.dumps(output, default=str),
                query_context=query,
                analysis_depth="detailed",
                executive_summary=output.get("executive_summary", ""),
                detailed_findings=output.get("detailed_findings", ""),
                recommendations=output.get("recommendations", ""),
                technical_details=output.get("technical_details", ""),
                confidence=row.get("score", 0.8),
            ).with_inputs("search_results", "query_context", "analysis_depth")
        elif agent_name == "routing":
            # Routing optimization already handled by OptimizationOrchestrator.
            # Feed misrouted queries as additional experiences.
            continue
        else:
            continue

        trainset.append(example)

    if not trainset:
        return {"status": "skipped", "reason": "no_training_examples"}

    # Select and compile the right DSPy module
    if agent_name == "search":
        signature = optimizer.create_query_analysis_signature()
    elif agent_name == "summary":
        signature = optimizer.create_summary_generation_signature()
    elif agent_name == "report":
        signature = optimizer.create_detailed_report_signature()
    else:
        return {"status": "skipped", "reason": f"no_signature_for_{agent_name}"}

    from dspy.teleprompt import BootstrapFewShot

    teleprompter = BootstrapFewShot(
        max_bootstrapped_demos=optimizer.optimization_settings[
            "max_bootstrapped_demos"
        ],
        max_labeled_demos=optimizer.optimization_settings["max_labeled_demos"],
        max_rounds=optimizer.optimization_settings["max_rounds"],
        max_errors=optimizer.optimization_settings["max_errors"],
    )

    module = dspy.ChainOfThought(signature)

    try:
        compiled = teleprompter.compile(module, trainset=trainset)

        # Store compiled module via ArtifactManager
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        artifact_manager = ArtifactManager(telemetry_provider=telemetry_provider)
        artifact_id = await artifact_manager.store_artifact(
            artifact_type=f"dspy_compiled_{agent_name}",
            data=compiled.dump_state(),
            metadata={
                "agent": agent_name,
                "tenant_id": tenant_id,
                "training_examples": len(trainset),
                "timestamp": str(datetime.now()),
            },
        )

        return {
            "status": "success",
            "artifact_id": artifact_id,
            "training_examples": len(trainset),
        }

    except Exception as e:
        logger.error(f"DSPy compilation failed for {agent_name}: {e}")
        return {"status": "failed", "error": str(e)}


async def run_cleanup(log_retention_days: int, memory_retention_days: int) -> dict:
    """Run cleanup tasks."""
    results = {}

    try:
        from cogniverse_core.memory.manager import Mem0MemoryManager

        mem_manager = Mem0MemoryManager(tenant_id="default")
        mem_manager.cleanup(retention_days=memory_retention_days)
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
        choices=["once", "full", "dspy", "cleanup", "triggered"],
        required=True,
    )
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument(
        "--agents",
        help="Comma-separated agent names for triggered mode",
    )
    parser.add_argument(
        "--trigger-dataset",
        help="Phoenix dataset name containing trigger payload",
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
    elif args.mode == "triggered":
        if not args.agents or not args.trigger_dataset:
            parser.error(
                "--agents and --trigger-dataset are required for triggered mode"
            )
        agents = [a.strip() for a in args.agents.split(",")]
        result = asyncio.run(
            run_triggered_optimization(
                tenant_id=args.tenant_id,
                agents=agents,
                trigger_dataset=args.trigger_dataset,
            )
        )
    else:
        result = asyncio.run(run_optimization(args.mode))

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
