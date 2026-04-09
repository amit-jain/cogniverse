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
    python -m cogniverse_runtime.optimization_cli --mode simba --tenant-id default
    python -m cogniverse_runtime.optimization_cli --mode workflow --tenant-id default
    python -m cogniverse_runtime.optimization_cli --mode gateway-thresholds --tenant-id default
    python -m cogniverse_runtime.optimization_cli --mode profile --tenant-id default
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta

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
    tenant_id: str,
    agents: list[str],
    trigger_dataset: str,
    config_manager=None,
    phoenix_endpoint: str = None,
) -> dict:
    """Run optimization triggered by quality monitor.

    Loads scored examples from Phoenix trigger dataset, then compiles
    DSPy modules for each flagged agent using those examples as training data.

    Args:
        tenant_id: Tenant to optimize for.
        agents: List of agent names to optimize.
        trigger_dataset: Phoenix dataset name with scored trace examples.
        config_manager: Optional ConfigManager (for testing). If None,
            creates default from config.json.
        phoenix_endpoint: Optional Phoenix HTTP URL (for testing). If None,
            reads from SystemConfig.telemetry_url.
    """
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    if config_manager is None:
        from cogniverse_foundation.config.utils import create_default_config_manager

        config_manager = create_default_config_manager()

    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    # Load trigger dataset from Phoenix
    import phoenix as px

    system_config = config_manager.get_system_config()
    if phoenix_endpoint is None:
        phoenix_endpoint = system_config.telemetry_url
    sync_client = px.Client(endpoint=phoenix_endpoint)

    try:
        dataset = sync_client.get_dataset(name=trigger_dataset)
        trigger_df = dataset.as_dataframe()

        # Phoenix wraps columns under input/output dicts — flatten
        if "input" in trigger_df.columns and "agent" not in trigger_df.columns:
            import pandas as _pd

            flat = []
            for _, row in trigger_df.iterrows():
                inp = row.get("input", {}) or {}
                out = row.get("output", {}) or {}
                flat.append({**inp, **out})
            trigger_df = _pd.DataFrame(flat)
    except Exception as e:
        logger.error(f"Failed to load trigger dataset '{trigger_dataset}': {e}")
        return {"status": "failed", "error": str(e)}

    results = {}
    from cogniverse_foundation.config.utils import get_config

    config_utils = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config_utils.get_llm_config()
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

    # Strategy distillation: learn reusable strategies from the trigger dataset
    try:
        from cogniverse_agents.optimizer.strategy_learner import StrategyLearner
        from cogniverse_core.memory.manager import Mem0MemoryManager

        if not system_config.backend_url:
            raise ValueError(
                "SystemConfig.backend_url is required for strategy distillation"
            )
        if not system_config.backend_port:
            raise ValueError(
                "SystemConfig.backend_port is required for strategy distillation"
            )
        if not llm_endpoint.api_base:
            raise ValueError(
                "LLMEndpointConfig.api_base is required for strategy distillation"
            )

        mem_manager = Mem0MemoryManager(tenant_id=tenant_id)
        if mem_manager.memory is None:
            mem_manager.initialize(
                backend_host=system_config.backend_url,
                backend_port=system_config.backend_port,
                llm_model=llm_endpoint.model,
                embedding_model="nomic-embed-text",
                llm_base_url=llm_endpoint.api_base,
                config_manager=config_manager,
                schema_loader=None,
            )

        learner = StrategyLearner(
            memory_manager=mem_manager,
            tenant_id=tenant_id,
            llm_config=llm_endpoint,
        )
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        results["strategies_distilled"] = len(strategies)
        logger.info(f"Distilled {len(strategies)} strategies from trigger dataset")
    except Exception as e:
        logger.warning(f"Strategy distillation failed (non-fatal): {e}")
        results["strategies_distilled"] = 0

    # Post-optimization: run golden eval to verify improvement (best-effort)
    try:
        from cogniverse_evaluation.quality_monitor import QualityMonitor

        if not llm_endpoint.api_base:
            raise ValueError("LLMEndpointConfig.api_base required for post-eval")

        monitor = QualityMonitor(
            tenant_id=tenant_id,
            runtime_url=system_config.agent_registry_url,
            phoenix_http_endpoint=phoenix_endpoint,
            llm_base_url=llm_endpoint.api_base,
            llm_model=llm_endpoint.model,
            golden_dataset_path="data/testset/evaluation/sample_videos_retrieval_queries.json",
        )
        post_eval = await monitor.evaluate_golden_set()
        results["post_optimization_eval"] = {
            "mrr": post_eval.mean_mrr,
            "ndcg": post_eval.mean_ndcg,
            "precision_at_5": post_eval.mean_precision_at_5,
        }

        if post_eval.mean_mrr > (monitor._last_golden_baseline_mrr or 0):
            await monitor.update_baseline(golden_result=post_eval)
            results["baseline_updated"] = True

        # Grow golden set with high-scoring live queries
        high_scoring = trigger_df[trigger_df["category"] == "high_scoring"]
        new_golden_candidates = []
        for _, row in high_scoring.iterrows():
            if float(row.get("score", 0)) >= 0.8:
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
        logger.warning(f"Post-optimization eval failed (non-fatal): {e}")
        results["post_optimization_eval"] = {"error": str(e)}

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


async def _query_spans_by_name(
    telemetry_provider,
    tenant_id: str,
    span_name: str,
    lookback_hours: int,
):
    """Query spans from Phoenix filtered by span name.

    Returns a DataFrame of matching spans, or an empty DataFrame if none found.
    """
    import pandas as pd

    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    telemetry_manager = get_telemetry_manager()
    project_name = telemetry_manager.config.get_project_name(tenant_id)

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)

    try:
        spans_df = await telemetry_provider.traces.get_spans(
            project=project_name,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )
    except Exception as e:
        logger.error("Failed to query spans for %s: %s", span_name, e)
        return pd.DataFrame()

    if spans_df.empty:
        return spans_df

    return spans_df[spans_df["name"] == span_name]


async def run_simba_optimization(
    tenant_id: str,
    lookback_hours: int = 24,
) -> dict:
    """SIMBA query enhancement optimization.

    Reads cogniverse.query_enhancement spans, builds training examples
    from (original_query -> enhanced_query) pairs, compiles the
    QueryEnhancementAgent's DSPy module via BootstrapFewShot, and
    saves the optimized module as an artifact.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting SIMBA optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_QUERY_ENHANCEMENT, lookback_hours
    )

    if spans_df.empty:
        logger.info("No query_enhancement spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d query_enhancement spans", len(spans_df))

    # Build DSPy training examples from span attributes
    import dspy

    trainset = []
    for _, row in spans_df.iterrows():
        original = row.get("attributes.query_enhancement.original_query", "")
        enhanced = row.get("attributes.query_enhancement.enhanced_query", "")
        confidence = float(row.get("attributes.query_enhancement.confidence", 0.0))

        if not original or not enhanced:
            continue

        example = dspy.Example(
            query=original,
            enhanced_query=enhanced,
            expansion_terms="",
            synonyms="",
            context="",
            confidence=str(confidence),
            reasoning="From production span",
        ).with_inputs("query")
        trainset.append(example)

    if not trainset:
        logger.info("No valid training examples extracted from spans")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    logger.info("Built %d training examples for SIMBA compilation", len(trainset))

    # Compile DSPy module
    from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule

    llm_config = config_manager.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{llm_endpoint.model}",
            api_base=llm_endpoint.api_base,
        )
    )

    module = QueryEnhancementModule()

    from dspy.teleprompt import BootstrapFewShot

    teleprompter = BootstrapFewShot(
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
        max_errors=5,
    )

    try:
        compiled = teleprompter.compile(module, trainset=trainset)

        # Save compiled module via ArtifactManager
        import json as _json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        dataset_id = await artifact_manager.save_blob(
            kind="model",
            key="simba_query_enhancement",
            content=_json.dumps(compiled.dump_state(), default=str),
        )

        logger.info("SIMBA optimization complete — artifact %s", dataset_id)
        return {
            "status": "success",
            "spans_found": len(spans_df),
            "training_examples": len(trainset),
            "artifact_id": dataset_id,
        }

    except Exception as e:
        logger.error("SIMBA compilation failed: %s", e)
        return {"status": "failed", "error": str(e)}


async def run_workflow_optimization(
    tenant_id: str,
    lookback_hours: int = 24,
) -> dict:
    """Workflow orchestration optimization.

    Reads cogniverse.orchestration spans, feeds them through
    OrchestrationEvaluator to extract WorkflowExecution records,
    then generates workflow templates and agent performance profiles
    and saves them as artifacts.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting workflow optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_ORCHESTRATION, lookback_hours
    )

    if spans_df.empty:
        logger.info("No orchestration spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d orchestration spans", len(spans_df))

    # Use OrchestrationEvaluator to extract workflow executions
    from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

    intelligence = WorkflowIntelligence(
        telemetry_provider=telemetry_provider,
        tenant_id=tenant_id,
    )

    from cogniverse_agents.routing.orchestration_evaluator import (
        OrchestrationEvaluator,
    )

    evaluator = OrchestrationEvaluator(
        workflow_intelligence=intelligence,
        tenant_id=tenant_id,
    )

    eval_result = await evaluator.evaluate_orchestration_spans(
        lookback_hours=lookback_hours,
    )

    workflows_extracted = eval_result.get("workflows_extracted", 0)
    logger.info("Extracted %d workflow executions from spans", workflows_extracted)

    if workflows_extracted == 0:
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "workflows_extracted": 0,
        }

    # Generate templates from workflow history
    import json as _json

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

    # Save workflow executions as demonstrations
    execution_demos = []
    for execution in intelligence.workflow_history:
        execution_demos.append(
            {
                "input": _json.dumps(
                    {
                        "workflow_id": execution.workflow_id,
                        "query": execution.query,
                        "query_type": execution.query_type,
                        "execution_time": execution.execution_time,
                        "success": execution.success,
                        "agent_sequence": execution.agent_sequence,
                        "task_count": execution.task_count,
                        "parallel_efficiency": execution.parallel_efficiency,
                        "confidence_score": execution.confidence_score,
                        "timestamp": datetime.now().isoformat(),
                    },
                    default=str,
                ),
                "output": _json.dumps(
                    {"success": execution.success, "execution_time": execution.execution_time},
                    default=str,
                ),
            }
        )

    if execution_demos:
        await artifact_manager.save_demonstrations("workflow", execution_demos)

    # Save agent performance profiles
    perf_report = intelligence.get_agent_performance_report()
    if perf_report:
        profile_demos = [
            {
                "input": _json.dumps(
                    {
                        "agent_name": agent_name,
                        **perf_data,
                        "last_updated": datetime.now().isoformat(),
                    },
                    default=str,
                ),
                "output": _json.dumps(
                    {"agent_name": agent_name}, default=str
                ),
            }
            for agent_name, perf_data in perf_report.items()
        ]
        await artifact_manager.save_demonstrations("agent_profiles", profile_demos)

    # Save query type patterns
    if intelligence.query_type_patterns:
        await artifact_manager.save_blob(
            kind="workflow",
            key="query_patterns",
            content=_json.dumps(dict(intelligence.query_type_patterns)),
        )

    logger.info("Workflow optimization complete")
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "workflows_extracted": workflows_extracted,
        "execution_demos_saved": len(execution_demos),
        "agent_profiles_saved": len(perf_report),
    }


async def run_gateway_thresholds_optimization(
    tenant_id: str,
    lookback_hours: int = 24,
) -> dict:
    """Gateway confidence threshold tuning.

    Reads cogniverse.gateway spans, analyzes classification accuracy
    (was "simple" routing correct? did "complex" queries actually need
    orchestration?), and updates GLiNER confidence thresholds.
    Saves the threshold config as an artifact.
    """
    import json as _json

    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_GATEWAY
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting gateway threshold optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_GATEWAY, lookback_hours
    )

    if spans_df.empty:
        logger.info("No gateway spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d gateway spans", len(spans_df))

    # Analyze classification patterns
    simple_spans = spans_df[
        spans_df.get("attributes.gateway.complexity", default="") == "simple"
    ]
    complex_spans = spans_df[
        spans_df.get("attributes.gateway.complexity", default="") == "complex"
    ]

    # Extract confidence scores
    confidences = spans_df.get("attributes.gateway.confidence", default=None)
    if confidences is None:
        logger.info("No confidence data in gateway spans")
        return {"status": "no_data", "spans_found": len(spans_df), "reason": "no_confidence_data"}

    confidences = confidences.dropna().astype(float)
    if confidences.empty:
        return {"status": "no_data", "spans_found": len(spans_df), "reason": "no_confidence_data"}

    # Check error rates by complexity class
    status_col = "status_code"
    simple_errors = 0
    simple_total = len(simple_spans)
    complex_errors = 0
    complex_total = len(complex_spans)

    if status_col in spans_df.columns:
        if simple_total > 0:
            simple_errors = len(
                simple_spans[simple_spans[status_col] == "ERROR"]
            )
        if complex_total > 0:
            complex_errors = len(
                complex_spans[complex_spans[status_col] == "ERROR"]
            )

    simple_error_rate = simple_errors / max(simple_total, 1)
    complex_error_rate = complex_errors / max(complex_total, 1)

    # Compute optimized thresholds:
    # If simple queries have high error rate, raise the fast-path threshold
    # (send more queries to orchestrator). If complex queries rarely fail,
    # we can lower it to keep more queries on the fast path.
    current_threshold = 0.7  # default from GatewayDeps
    mean_confidence = float(confidences.mean())

    if simple_error_rate > 0.2:
        # Too many simple-routed queries are failing — raise threshold
        optimized_threshold = min(current_threshold + 0.1, 0.95)
    elif complex_error_rate < 0.05 and mean_confidence > 0.8:
        # Complex routing is rarely needed and confidence is high — lower threshold
        optimized_threshold = max(current_threshold - 0.05, 0.5)
    else:
        optimized_threshold = current_threshold

    # Also tune gliner_threshold based on confidence distribution
    p25_confidence = float(confidences.quantile(0.25))
    optimized_gliner_threshold = max(0.15, min(p25_confidence * 0.8, 0.5))

    threshold_config = {
        "fast_path_confidence_threshold": optimized_threshold,
        "gliner_threshold": round(optimized_gliner_threshold, 3),
        "analysis": {
            "total_spans": len(spans_df),
            "simple_count": simple_total,
            "complex_count": complex_total,
            "simple_error_rate": round(simple_error_rate, 4),
            "complex_error_rate": round(complex_error_rate, 4),
            "mean_confidence": round(mean_confidence, 4),
            "p25_confidence": round(p25_confidence, 4),
        },
    }

    # Save threshold config as artifact
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
    dataset_id = await artifact_manager.save_blob(
        kind="config",
        key="gateway_thresholds",
        content=_json.dumps(threshold_config),
    )

    logger.info(
        "Gateway threshold optimization complete — threshold %.2f -> %.2f, artifact %s",
        current_threshold,
        optimized_threshold,
        dataset_id,
    )

    return {
        "status": "success",
        "spans_found": len(spans_df),
        "artifact_id": dataset_id,
        "thresholds": threshold_config,
    }


async def run_profile_optimization(
    tenant_id: str,
    lookback_hours: int = 24,
) -> dict:
    """Profile selection optimization.

    Reads cogniverse.profile_selection spans, builds training examples
    from (query, available_profiles) -> selected_profile pairs, compiles
    the ProfileSelectionAgent's DSPy module, and saves the optimized
    module as an artifact.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_PROFILE_SELECTION
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting profile selection optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_PROFILE_SELECTION, lookback_hours
    )

    if spans_df.empty:
        logger.info("No profile_selection spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d profile_selection spans", len(spans_df))

    # Build DSPy training examples from span attributes
    import dspy

    trainset = []
    for _, row in spans_df.iterrows():
        query = row.get("attributes.profile_selection.query", "")
        selected = row.get("attributes.profile_selection.selected_profile", "")
        modality = row.get("attributes.profile_selection.modality", "video")
        complexity = row.get("attributes.profile_selection.complexity", "simple")
        intent = row.get("attributes.profile_selection.intent", "")
        confidence = float(row.get("attributes.profile_selection.confidence", 0.0))

        if not query or not selected:
            continue

        # Only learn from high-confidence selections
        if confidence < 0.5:
            continue

        example = dspy.Example(
            query=query,
            available_profiles="video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s,video_videoprism_base_mv_chunk_30s,video_videoprism_large_mv_chunk_30s",
            selected_profile=selected,
            confidence=str(confidence),
            reasoning=f"Selected {selected} for {modality}/{complexity} query",
            query_intent=intent,
            modality=modality,
            complexity=complexity,
        ).with_inputs("query", "available_profiles")
        trainset.append(example)

    if not trainset:
        logger.info("No valid training examples extracted from spans")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    logger.info("Built %d training examples for profile optimization", len(trainset))

    # Compile DSPy module
    from cogniverse_agents.profile_selection_agent import ProfileSelectionModule

    llm_config = config_manager.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{llm_endpoint.model}",
            api_base=llm_endpoint.api_base,
        )
    )

    module = ProfileSelectionModule()

    from dspy.teleprompt import BootstrapFewShot

    teleprompter = BootstrapFewShot(
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
        max_errors=5,
    )

    try:
        compiled = teleprompter.compile(module, trainset=trainset)

        import json as _json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        dataset_id = await artifact_manager.save_blob(
            kind="model",
            key="profile_selection",
            content=_json.dumps(compiled.dump_state(), default=str),
        )

        logger.info("Profile optimization complete — artifact %s", dataset_id)
        return {
            "status": "success",
            "spans_found": len(spans_df),
            "training_examples": len(trainset),
            "artifact_id": dataset_id,
        }

    except Exception as e:
        logger.error("Profile DSPy compilation failed: %s", e)
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Cogniverse Optimization CLI")
    parser.add_argument(
        "--mode",
        choices=[
            "once",
            "full",
            "dspy",
            "cleanup",
            "triggered",
            "simba",
            "workflow",
            "gateway-thresholds",
            "profile",
        ],
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
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Hours of span history to analyze (simba, workflow, gateway-thresholds, profile)",
    )
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
    elif args.mode == "simba":
        result = asyncio.run(
            run_simba_optimization(
                tenant_id=args.tenant_id,
                lookback_hours=args.lookback_hours,
            )
        )
    elif args.mode == "workflow":
        result = asyncio.run(
            run_workflow_optimization(
                tenant_id=args.tenant_id,
                lookback_hours=args.lookback_hours,
            )
        )
    elif args.mode == "gateway-thresholds":
        result = asyncio.run(
            run_gateway_thresholds_optimization(
                tenant_id=args.tenant_id,
                lookback_hours=args.lookback_hours,
            )
        )
    elif args.mode == "profile":
        result = asyncio.run(
            run_profile_optimization(
                tenant_id=args.tenant_id,
                lookback_hours=args.lookback_hours,
            )
        )
    else:
        result = asyncio.run(run_optimization(args.mode))

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
