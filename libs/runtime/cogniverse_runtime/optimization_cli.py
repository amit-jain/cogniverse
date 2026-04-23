"""CLI entry point for optimization — called by Argo CronWorkflows.

Per-agent batch optimization modes. Each mode reads production spans from
Phoenix, builds DSPy training examples, compiles optimized modules, and
saves artifacts via ArtifactManager. Agents load artifacts at startup.

Usage:
    python -m cogniverse_runtime.optimization_cli --mode simba --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode workflow --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode gateway-thresholds --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode profile --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode entity-extraction --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode routing --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode cleanup --log-retention-days 7
    python -m cogniverse_runtime.optimization_cli --mode triggered \
        --tenant-id acme:production --agents search,summary \
        --trigger-dataset optimization-trigger-acme-production-20260403_040000
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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
    from phoenix.client import Client as PhoenixSyncClient

    system_config = config_manager.get_system_config()
    if phoenix_endpoint is None:
        phoenix_endpoint = system_config.telemetry_url
    sync_client = PhoenixSyncClient(base_url=phoenix_endpoint)

    try:
        dataset = sync_client.datasets.get_dataset(dataset=trigger_dataset)
        trigger_df = dataset.to_dataframe()

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


async def run_cleanup(
    tenant_id: str, log_retention_days: int, memory_retention_days: int
) -> dict:
    """Run cleanup tasks."""
    results = {}

    try:
        from cogniverse_core.memory.manager import Mem0MemoryManager

        mem_manager = Mem0MemoryManager(tenant_id=tenant_id)
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
    lookback_hours: float,
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


def _create_teleprompter(trainset_size: int):
    """Select DSPy optimizer config based on training set size.

    Scales BootstrapFewShot parameters for larger training sets:
    - < 50 examples: 4 bootstrapped demos, 8 labeled, 1 round
    - >= 50 examples: 8 bootstrapped demos, 16 labeled, 2 rounds
    """
    from dspy.teleprompt import BootstrapFewShot

    if trainset_size >= 50:
        logger.info(
            "Using scaled BootstrapFewShot for %d examples (>= 50 threshold)",
            trainset_size,
        )
        return BootstrapFewShot(
            max_bootstrapped_demos=8,
            max_labeled_demos=16,
            max_rounds=2,
            max_errors=10,
        )

    logger.info("Using BootstrapFewShot for %d examples", trainset_size)
    return BootstrapFewShot(
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
        max_errors=5,
    )


async def _load_approved_synthetic_data(
    telemetry_provider,
    tenant_id: str,
    optimizer_type: str,
) -> list:
    """Load approved synthetic datasets for an optimizer type.

    Returns list of demo dicts from datasets with status APPROVED or AUTO_APPROVED.
    Returns empty list if none found or if approval module is not available.
    """
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    am = ArtifactManager(telemetry_provider, tenant_id)
    demos = await am.load_demonstrations(f"synthetic_{optimizer_type}")
    if not demos:
        return []

    try:
        from cogniverse_agents.approval.interfaces import ApprovalStatus

        approved_statuses = {
            ApprovalStatus.APPROVED.value,
            ApprovalStatus.AUTO_APPROVED.value,
        }
    except ImportError:
        approved_statuses = {"approved", "auto_approved"}

    approved = []
    for demo in demos:
        metadata = demo.get("metadata", {})
        if isinstance(metadata, str):
            try:
                import json as _json

                metadata = _json.loads(metadata)
            except (ValueError, TypeError):
                metadata = {}
        status = metadata.get("approval_status", "")
        if status in approved_statuses:
            approved.append(demo)

    logger.info(
        "Loaded %d/%d approved synthetic examples for %s",
        len(approved),
        len(demos),
        optimizer_type,
    )
    return approved


async def run_simba_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
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
        # Phoenix stores custom attributes as dict in "attributes.query_enhancement"
        qe_attrs = row.get("attributes.query_enhancement", {})
        if not isinstance(qe_attrs, dict):
            qe_attrs = {}
        original = qe_attrs.get("original_query", "")
        enhanced = qe_attrs.get("enhanced_query", "")
        confidence = float(qe_attrs.get("confidence", 0.0))

        if not original or not enhanced:
            continue
        if enhanced.strip() == original.strip():
            # Belt-and-suspenders: QueryEnhancementAgent now guarantees
            # enhanced != query, so new spans won't hit this branch. Older
            # spans (from before that fix) can still be in the lookback
            # window; skip them so SIMBA doesn't train on identity pairs.
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

    # Merge approved synthetic data
    import json as _json

    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "simba"
    )
    production_count = len(trainset)
    for demo in synthetic_demos:
        inp = demo.get("input", "")
        if isinstance(inp, str):
            try:
                inp = _json.loads(inp)
            except (ValueError, TypeError):
                continue
        if isinstance(inp, dict):
            q = str(inp.get("query", "")).strip()
            eq = str(inp.get("enhanced_query", "")).strip()
            if q and eq and q == eq:
                continue
            example = dspy.Example(**inp).with_inputs("query")
            trainset.append(example)
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        production_count,
        len(trainset),
    )

    # Compile DSPy module
    from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{llm_endpoint.model}",
            api_base=llm_endpoint.api_base,
        )
    )

    module = QueryEnhancementModule()

    teleprompter = _create_teleprompter(len(trainset))

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
    lookback_hours: float = 24.0,
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
                    {
                        "success": execution.success,
                        "execution_time": execution.execution_time,
                    },
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
                "output": _json.dumps({"agent_name": agent_name}, default=str),
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
    lookback_hours: float = 24.0,
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

    # Phoenix stores custom span attributes as a dict in "attributes.gateway"
    # column, not as separate flattened columns. Extract fields from the dict.
    gw_attrs = spans_df.get("attributes.gateway")
    if gw_attrs is None:
        logger.info("No attributes.gateway column in spans")
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "reason": "no_gateway_attributes",
        }

    # Extract complexity and confidence from the nested dict
    spans_df = spans_df.copy()
    spans_df["_complexity"] = gw_attrs.apply(
        lambda d: d.get("complexity", "") if isinstance(d, dict) else ""
    )
    spans_df["_confidence"] = gw_attrs.apply(
        lambda d: d.get("confidence", None) if isinstance(d, dict) else None
    )

    simple_spans = spans_df[spans_df["_complexity"] == "simple"]
    complex_spans = spans_df[spans_df["_complexity"] == "complex"]

    confidences = spans_df["_confidence"].dropna().astype(float)
    if confidences.empty:
        logger.info("No confidence data in gateway spans")
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "reason": "no_confidence_data",
        }

    # Check error rates by complexity class
    status_col = "status_code"
    simple_errors = 0
    simple_total = len(simple_spans)
    complex_errors = 0
    complex_total = len(complex_spans)

    if status_col in spans_df.columns:
        if simple_total > 0:
            simple_errors = len(simple_spans[simple_spans[status_col] == "ERROR"])
        if complex_total > 0:
            complex_errors = len(complex_spans[complex_spans[status_col] == "ERROR"])

    simple_error_rate = simple_errors / max(simple_total, 1)
    complex_error_rate = complex_errors / max(complex_total, 1)

    # Compute optimized thresholds:
    # If simple queries have high error rate, raise the fast-path threshold
    # (send more queries to orchestrator). If complex queries rarely fail,
    # we can lower it to keep more queries on the fast path.
    current_threshold = 0.4  # default from GatewayDeps.fast_path_confidence_threshold
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
    lookback_hours: float = 24.0,
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
        # Phoenix stores custom attributes as dict in "attributes.profile_selection"
        ps_attrs = row.get("attributes.profile_selection", {})
        if not isinstance(ps_attrs, dict):
            ps_attrs = {}
        query = ps_attrs.get("query", "")
        selected = ps_attrs.get("selected_profile", "")
        modality = ps_attrs.get("modality", "video")
        complexity = ps_attrs.get("complexity", "simple")
        intent = ps_attrs.get("intent", "")
        confidence = float(ps_attrs.get("confidence", 0.0))

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

    # Merge approved synthetic data
    import json as _json

    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "profile"
    )
    production_count = len(trainset)
    for demo in synthetic_demos:
        inp = demo.get("input", "")
        if isinstance(inp, str):
            try:
                inp = _json.loads(inp)
            except (ValueError, TypeError):
                continue
        if isinstance(inp, dict):
            example = dspy.Example(**inp).with_inputs("query")
            trainset.append(example)
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        production_count,
        len(trainset),
    )

    # Compile DSPy module
    from cogniverse_agents.profile_selection_agent import ProfileSelectionModule
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{llm_endpoint.model}",
            api_base=llm_endpoint.api_base,
        )
    )

    module = ProfileSelectionModule()

    teleprompter = _create_teleprompter(len(trainset))

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


async def run_entity_extraction_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
) -> dict:
    """Entity extraction optimization.

    Reads cogniverse.entity_extraction spans, builds training examples
    from (query) -> (entities) pairs, compiles the EntityExtractionModule's
    DSPy module, and saves the optimized module as an artifact.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_ENTITY_EXTRACTION
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting entity extraction optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_ENTITY_EXTRACTION, lookback_hours
    )

    if spans_df.empty:
        logger.info("No entity_extraction spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d entity_extraction spans", len(spans_df))

    import json as _json

    import dspy

    trainset = []
    for _, row in spans_df.iterrows():
        ee_attrs = row.get("attributes.entity_extraction", {})
        if not isinstance(ee_attrs, dict):
            ee_attrs = {}
        query = ee_attrs.get("query", "")
        entity_count = int(ee_attrs.get("entity_count", 0))
        entities_json = ee_attrs.get("entities", "[]")

        if not query or entity_count == 0:
            continue

        if not isinstance(entities_json, str):
            entities_json = _json.dumps(entities_json)

        example = dspy.Example(
            query=query,
            entities=entities_json,
            entity_types="",
        ).with_inputs("query")
        trainset.append(example)

    if not trainset:
        logger.info("No valid training examples extracted from entity_extraction spans")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    logger.info(
        "Built %d training examples for entity extraction optimization", len(trainset)
    )

    # Merge approved synthetic data
    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "entity_extraction"
    )
    production_count = len(trainset)
    for demo in synthetic_demos:
        inp = demo.get("input", "")
        if isinstance(inp, str):
            try:
                inp = _json.loads(inp)
            except (ValueError, TypeError):
                continue
        if isinstance(inp, dict):
            example = dspy.Example(**inp).with_inputs("query")
            trainset.append(example)
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        production_count,
        len(trainset),
    )

    from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{llm_endpoint.model}",
            api_base=llm_endpoint.api_base,
        )
    )

    module = EntityExtractionModule()

    teleprompter = _create_teleprompter(len(trainset))

    try:
        compiled = teleprompter.compile(module, trainset=trainset)

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        dataset_id = await artifact_manager.save_blob(
            kind="model",
            key="entity_extraction",
            content=_json.dumps(compiled.dump_state(), default=str),
        )

        logger.info("Entity extraction optimization complete — artifact %s", dataset_id)
        return {
            "status": "success",
            "spans_found": len(spans_df),
            "training_examples": len(trainset),
            "artifact_id": dataset_id,
        }

    except Exception as e:
        logger.error("Entity extraction DSPy compilation failed: %s", e)
        return {"status": "failed", "error": str(e)}


async def run_routing_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
) -> dict:
    """Routing decision optimization.

    Reads cogniverse.routing spans, builds training examples from
    (query) -> (recommended_agent, primary_intent) pairs, compiles the
    DSPyAdvancedRoutingModule, and saves the optimized module as an artifact.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting routing optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_ROUTING, lookback_hours
    )

    if spans_df.empty:
        logger.info("No routing spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d routing spans", len(spans_df))

    import dspy

    trainset = []
    for _, row in spans_df.iterrows():
        r_attrs = row.get("attributes.routing", {})
        if not isinstance(r_attrs, dict):
            r_attrs = {}
        query = r_attrs.get("query", "")
        # Handle both new ("recommended_agent") and legacy ("chosen_agent") field names
        recommended_agent = r_attrs.get("recommended_agent", "") or r_attrs.get(
            "chosen_agent", ""
        )
        primary_intent = r_attrs.get("primary_intent", "")
        confidence = float(r_attrs.get("confidence", 0.0))

        if not query or not recommended_agent:
            continue

        if confidence < 0.5:
            continue

        example = dspy.Example(
            query=query,
            context="",
            primary_intent=primary_intent,
            needs_video_search=str(
                "video" in recommended_agent.lower()
                or "search" in recommended_agent.lower()
            ),
            recommended_agent=recommended_agent,
            confidence=str(confidence),
        ).with_inputs("query", "context")
        trainset.append(example)

    if not trainset:
        logger.info("No valid training examples extracted from routing spans")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    logger.info("Built %d training examples for routing optimization", len(trainset))

    # Merge approved synthetic data
    import json as _json

    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "routing"
    )
    production_count = len(trainset)
    for demo in synthetic_demos:
        inp = demo.get("input", "")
        if isinstance(inp, str):
            try:
                inp = _json.loads(inp)
            except (ValueError, TypeError):
                continue
        if isinstance(inp, dict):
            example = dspy.Example(**inp).with_inputs("query")
            trainset.append(example)
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        production_count,
        len(trainset),
    )

    try:
        from cogniverse_agents.routing.dspy_relationship_router import (
            DSPyAdvancedRoutingModule,
        )

        module = DSPyAdvancedRoutingModule(analysis_module=None)
    except (ImportError, Exception):
        from cogniverse_agents.routing.dspy_routing_signatures import (
            BasicQueryAnalysisSignature,
        )

        module = dspy.ChainOfThought(BasicQueryAnalysisSignature)

    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{llm_endpoint.model}",
            api_base=llm_endpoint.api_base,
        )
    )

    teleprompter = _create_teleprompter(len(trainset))

    try:
        compiled = teleprompter.compile(module, trainset=trainset)

        import json as _json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        dataset_id = await artifact_manager.save_blob(
            kind="model",
            key="routing_decision",
            content=_json.dumps(compiled.dump_state(), default=str),
        )

        logger.info("Routing optimization complete — artifact %s", dataset_id)
        return {
            "status": "success",
            "spans_found": len(spans_df),
            "training_examples": len(trainset),
            "artifact_id": dataset_id,
        }

    except Exception as e:
        logger.error("Routing DSPy compilation failed: %s", e)
        return {"status": "failed", "error": str(e)}


async def run_synthetic_generation(
    tenant_id: str,
    optimizer_types: list[str] | None = None,
    count: int = 50,
) -> dict:
    """Generate synthetic training data for optimizer types.

    Uses SyntheticDataService to create training examples, then saves
    them as demonstrations via ArtifactManager for later merge into
    batch optimization jobs.
    """
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    if optimizer_types is None:
        optimizer_types = ["simba", "routing", "profile", "workflow"]

    logger.info(
        "Starting synthetic generation for tenant=%s types=%s count=%d",
        tenant_id,
        optimizer_types,
        count,
    )

    config_manager = create_default_config_manager()
    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    am = ArtifactManager(telemetry_provider, tenant_id)

    results = {}
    for opt_type in optimizer_types:
        try:
            from cogniverse_synthetic.schemas import SyntheticDataRequest
            from cogniverse_synthetic.service import SyntheticDataService

            backend_config = config.get("backend", {})
            generator_config = config.get("synthetic", {})

            # Create backend instance for content sampling. Stamp the
            # unpacked dicts with the caller's tenant_id if they don't
            # already carry one — synthetic runs are always per-tenant.
            from cogniverse_foundation.config.unified_config import (
                BackendConfig,
                SyntheticGeneratorConfig,
            )

            if isinstance(backend_config, dict):
                backend_config = {**backend_config}
                backend_config.setdefault("tenant_id", tenant_id)
                bc = BackendConfig(**backend_config)
            else:
                bc = backend_config
            if isinstance(generator_config, dict):
                generator_config = {**generator_config}
                generator_config.setdefault("tenant_id", tenant_id)
                gc = SyntheticGeneratorConfig(**generator_config)
            else:
                gc = generator_config

            from cogniverse_core.registries.backend_registry import BackendRegistry

            registry = BackendRegistry(config_manager=config_manager)
            backend = registry.get_backend(tenant_id=tenant_id)

            service = SyntheticDataService(
                backend=backend,
                backend_config=bc,
                generator_config=gc,
            )

            request = SyntheticDataRequest(
                optimizer=opt_type,
                count=count,
                tenant_id=tenant_id,
            )
            response = await service.generate(request)

            # Save as demonstrations with approval_status=pending
            demos = []
            for item in response.data:
                demos.append(
                    {
                        "input": json.dumps(item, default=str),
                        "output": json.dumps(
                            item.get("expected_output", ""), default=str
                        ),
                        "metadata": json.dumps(
                            {
                                "approval_status": "pending",
                                "optimizer_type": opt_type,
                                "generated_at": datetime.now().isoformat(),
                            }
                        ),
                    }
                )

            if demos:
                dataset_id = await am.save_demonstrations(
                    f"synthetic_{opt_type}", demos
                )
                results[opt_type] = {
                    "status": "success",
                    "examples_generated": len(demos),
                    "dataset_id": dataset_id,
                }
            else:
                results[opt_type] = {"status": "no_data", "examples_generated": 0}

            logger.info("Generated %d synthetic examples for %s", len(demos), opt_type)

        except Exception as e:
            logger.error("Synthetic generation failed for %s: %s", opt_type, e)
            results[opt_type] = {"status": "failed", "error": str(e)}

    return {
        "status": "success"
        if any(r["status"] == "success" for r in results.values())
        else "failed",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Cogniverse Optimization CLI")
    parser.add_argument(
        "--mode",
        choices=[
            "cleanup",
            "triggered",
            "simba",
            "workflow",
            "gateway-thresholds",
            "profile",
            "entity-extraction",
            "routing",
            "synthetic",
        ],
        required=True,
    )
    parser.add_argument("--tenant-id", required=True)
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
        type=float,
        default=24.0,
        help="Hours of span history to analyze. Accepts fractions (e.g. 0.1 "
        "= 6 minutes) so e2e tests can scope to the current fixture "
        "window without picking up spans from earlier runs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.mode == "cleanup":
        result = asyncio.run(
            run_cleanup(
                args.tenant_id, args.log_retention_days, args.memory_retention_days
            )
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
    elif args.mode == "entity-extraction":
        result = asyncio.run(
            run_entity_extraction_optimization(
                tenant_id=args.tenant_id,
                lookback_hours=args.lookback_hours,
            )
        )
    elif args.mode == "routing":
        result = asyncio.run(
            run_routing_optimization(
                tenant_id=args.tenant_id,
                lookback_hours=args.lookback_hours,
            )
        )
    elif args.mode == "synthetic":
        optimizer_types = ["simba", "routing", "profile", "workflow"]
        if args.agents:
            optimizer_types = [a.strip() for a in args.agents.split(",")]
        result = asyncio.run(
            run_synthetic_generation(
                tenant_id=args.tenant_id,
                optimizer_types=optimizer_types,
            )
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
