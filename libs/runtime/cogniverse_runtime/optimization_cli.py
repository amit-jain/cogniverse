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
    python -m cogniverse_runtime.optimization_cli --mode cleanup --log-retention-days 7
    python -m cogniverse_runtime.optimization_cli --mode triggered \
        --tenant-id acme:production --agents search,summary \
        --trigger-dataset optimization-trigger-acme-production-20260403_040000
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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
        denseon_url = system_config.inference_service_urls.get("denseon")
        if not denseon_url:
            raise ValueError(
                "Mem0 strategy distillation requires the denseon inference "
                "service. Available: "
                f"{sorted(system_config.inference_service_urls)}"
            )

        mem_manager = Mem0MemoryManager(tenant_id=tenant_id)
        if mem_manager.memory is None:
            mem_manager.initialize(
                backend_host=system_config.backend_url,
                backend_port=system_config.backend_port,
                llm_model=llm_endpoint.model,
                embedding_model="lightonai/DenseOn",
                llm_base_url=llm_endpoint.api_base,
                embedder_base_url=denseon_url,
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
    tenant_id: Optional[str],
    log_retention_days: int,
    memory_retention_days: int,
) -> dict:
    """Run schema-driven memory cleanup across one or every tenant.

    If ``tenant_id`` is None, enumerates every tenant in every org via
    the live ``tenant_manager`` helpers and drives
    ``Mem0MemoryManager.cleanup_with_schema`` per tenant — the
    daily-cleanup CronWorkflow runs in this mode. Per-tenant
    exceptions are captured so a single bad tenant does not abort the
    sweep across the rest.

    ``log_retention_days`` and ``memory_retention_days`` are accepted
    for CLI/workflow compatibility but the real retention contract is
    schema-driven (per-kind TTLs in the ``KnowledgeRegistry``); the
    raw retention_days args are echoed back in the result so the
    workflow logs preserve them.
    """
    from pathlib import Path

    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.schema import build_default_registry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.memory_init import lazy_init_memory

    # tenant_manager.get_backend() refuses to initialise without a
    # SchemaLoader injected up-front. The daily-cleanup CronWorkflow
    # runs as a standalone process (not via the runtime FastAPI app),
    # so it has no app-startup lifespan to call set_schema_loader for
    # it. Wire it here using the same FilesystemSchemaLoader pattern
    # the synthetic mode uses.
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    tenant_manager.set_schema_loader(FilesystemSchemaLoader(schemas_dir))

    # cleanup_with_schema requires a fully-initialised Mem0 instance
    # (it touches mgr.memory.get_all). The Mem0MemoryManager singleton
    # cache returns a bare object on first construction — without
    # lazy_init_memory every tenant returns "Mem0MemoryManager not
    # initialized" and the workflow appears to Succeed while silently
    # processing nothing. Build a config_manager once and reuse for
    # every tenant in the sweep.
    config_manager = create_default_config_manager()
    registry = build_default_registry()

    results: Dict[str, Any] = {
        "log_retention_days": log_retention_days,
        "memory_retention_days": memory_retention_days,
    }

    def _cleanup_one(tid: str) -> str:
        try:
            mm = Mem0MemoryManager(tenant_id=tid)
            if not lazy_init_memory(mm, tid, config_manager):
                return "skipped: memory backend init failed (see workflow log)"
            deleted_by_kind = mm.cleanup_with_schema(registry)
            return f"completed: {dict(deleted_by_kind)}"
        except Exception as e:
            return f"failed: {e}"

    if tenant_id is not None:
        results["memory_cleanup"] = {tenant_id: _cleanup_one(tenant_id)}
        return results

    per_tenant: Dict[str, str] = {}
    org_ids = await tenant_manager.list_organizations_internal()
    for org_id in org_ids:
        for tenant in await tenant_manager.list_tenants_for_org_internal(org_id):
            tid = tenant.tenant_full_id
            if not tid:
                continue
            per_tenant[tid] = _cleanup_one(tid)

    results["memory_cleanup"] = per_tenant
    results["tenants_processed"] = len(per_tenant)
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
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(lm=create_dspy_lm(llm_endpoint))

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

    # Drop executions whose agent_sequence references an agent that no
    # longer exists in the current configuration. Phoenix retains
    # historical spans across schema changes (e.g. an old agent that was
    # renamed or split into multiple agents), and the workflow optimizer
    # would otherwise persist demos that point at deleted agents —
    # those demos can't be replayed and trip downstream consumers
    # asserting ``agent in known_agents``.
    #
    # Read the live set from ``configs/config.json``'s ``agents`` block.
    # That file is the canonical source of which agents the runtime
    # routes to today. AgentRegistry would be the obvious alternative,
    # but it's populated by HTTP self-registration and starts empty in
    # this process (the optimization CLI runs in its own pod with no
    # agents registered against it), so an AgentRegistry-backed filter
    # would drop *every* demo, not just the stale ones.
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    _cfg = get_config(
        tenant_id=SYSTEM_TENANT_ID,
        config_manager=create_default_config_manager(),
    )
    _agents_section = (_cfg or {}).get("agents", {})
    _live_agents = {
        name
        for name, body in _agents_section.items()
        if isinstance(body, dict) and body.get("enabled", True)
    }
    if not _live_agents:
        raise RuntimeError(
            "configs/config.json 'agents' block is empty or unreachable; "
            "cannot filter stale workflow demos. Refusing to save "
            "execution_demos because every demo would be flagged stale "
            "(or every demo would slip through unchecked, depending on "
            "the filter's defensive default) — both are wrong."
        )

    def _agents_live(seq) -> bool:
        if isinstance(seq, str):
            seq = [a.strip() for a in seq.split(",") if a.strip()]
        return bool(seq) and all(a in _live_agents for a in seq)

    # Save workflow executions as demonstrations
    execution_demos = []
    for execution in intelligence.workflow_history:
        if not _agents_live(execution.agent_sequence):
            logger.debug(
                "Skipping stale workflow demo (agents %r not all in registry)",
                execution.agent_sequence,
            )
            continue
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


GATEWAY_DEFAULT_THRESHOLD = 0.4


def _compute_gateway_thresholds(spans_df) -> dict:
    """Pure function: calibrate gateway thresholds from a spans DataFrame.

    Extracted from :func:`run_gateway_thresholds_optimization` so the
    calibration algorithm can be unit-tested against deterministic inputs.
    The async wrapper handles Phoenix I/O and artifact persistence.

    Returns one of:
    - ``{"status": "no_data", "spans_found": N, "reason": ...}`` when the
      input lacks the required attributes or has no confidence values.
    - ``{"status": "ready", "spans_found": N, "thresholds": {...}}`` with
      the calibrated ``fast_path_confidence_threshold``, ``gliner_threshold``
      and an ``analysis`` subdict.

    The ``ready`` status is not yet ``success`` because the artifact hasn't
    been persisted — the wrapper writes the dataset and converts.
    """
    if spans_df.empty:
        return {"status": "no_data", "spans_found": 0}

    # Phoenix stores custom span attributes as a dict in ``attributes.gateway``
    # column (not as separate flattened columns). Extract fields from it.
    gw_attrs = spans_df.get("attributes.gateway")
    if gw_attrs is None:
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "reason": "no_gateway_attributes",
        }

    df = spans_df.copy()
    df["_complexity"] = gw_attrs.apply(
        lambda d: d.get("complexity", "") if isinstance(d, dict) else ""
    )
    df["_confidence"] = gw_attrs.apply(
        lambda d: d.get("confidence", None) if isinstance(d, dict) else None
    )

    simple_spans = df[df["_complexity"] == "simple"]
    complex_spans = df[df["_complexity"] == "complex"]

    confidences = df["_confidence"].dropna().astype(float)
    if confidences.empty:
        return {
            "status": "no_data",
            "spans_found": len(df),
            "reason": "no_confidence_data",
        }

    simple_total = len(simple_spans)
    complex_total = len(complex_spans)
    status_col = "status_code"
    simple_errors = 0
    complex_errors = 0
    if status_col in df.columns:
        if simple_total > 0:
            simple_errors = len(simple_spans[simple_spans[status_col] == "ERROR"])
        if complex_total > 0:
            complex_errors = len(complex_spans[complex_spans[status_col] == "ERROR"])

    simple_error_rate = simple_errors / max(simple_total, 1)
    complex_error_rate = complex_errors / max(complex_total, 1)
    mean_confidence = float(confidences.mean())
    p25_confidence = float(confidences.quantile(0.25))

    # Threshold calibration — if simple routing is failing often, raise the
    # threshold so more queries go to orchestrator; if complex routing rarely
    # fails AND mean confidence is high, lower the threshold to keep more
    # queries on the fast path.
    current = GATEWAY_DEFAULT_THRESHOLD
    if simple_error_rate > 0.2:
        optimized_threshold = min(current + 0.1, 0.95)
    elif complex_error_rate < 0.05 and mean_confidence > 0.8:
        optimized_threshold = max(current - 0.05, 0.5)
    else:
        optimized_threshold = current

    optimized_gliner_threshold = max(0.15, min(p25_confidence * 0.8, 0.5))

    return {
        "status": "ready",
        "spans_found": len(df),
        "thresholds": {
            "fast_path_confidence_threshold": optimized_threshold,
            "gliner_threshold": round(optimized_gliner_threshold, 3),
            "analysis": {
                "total_spans": len(df),
                "simple_count": simple_total,
                "complex_count": complex_total,
                "simple_error_rate": round(simple_error_rate, 4),
                "complex_error_rate": round(complex_error_rate, 4),
                "mean_confidence": round(mean_confidence, 4),
                "p25_confidence": round(p25_confidence, 4),
            },
        },
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

    result = _compute_gateway_thresholds(spans_df)
    if result["status"] != "ready":
        logger.info("Gateway threshold calibration skipped: %s", result.get("reason"))
        return result

    threshold_config = result["thresholds"]

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
    dataset_id = await artifact_manager.save_blob(
        kind="config",
        key="gateway_thresholds",
        content=_json.dumps(threshold_config),
    )

    logger.info(
        "Gateway threshold optimization complete — threshold %.2f -> %.2f, artifact %s",
        GATEWAY_DEFAULT_THRESHOLD,
        threshold_config["fast_path_confidence_threshold"],
        dataset_id,
    )

    return {
        "status": "success",
        "spans_found": result["spans_found"],
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
            # ProfileSelectionSignature has query AND available_profiles as
            # InputFields; match the production trainset's input set so
            # BootstrapFewShot doesn't see split-shape demos.
            example = dspy.Example(**inp).with_inputs("query", "available_profiles")
            trainset.append(example)
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        production_count,
        len(trainset),
    )

    # Compile DSPy module
    from cogniverse_agents.profile_selection_agent import ProfileSelectionModule
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(lm=create_dspy_lm(llm_endpoint))

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
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(lm=create_dspy_lm(llm_endpoint))

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
        optimizer_types = ["simba", "profile", "workflow"]

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

    # Synthetic generators that wrap DSPy modules (RoutingGenerator)
    # need a configured LM. The other optimizer modes (simba, profile,
    # entity-extraction) call ``dspy.configure(lm=create_dspy_lm(...))``;
    # matching that pattern here so any DSPy-backed generator finds a
    # default LM at module-construction time.
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm

    llm_endpoint = config.get_llm_config().primary
    dspy.configure(lm=create_dspy_lm(llm_endpoint))

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
                # Use ``from_dict`` (not ``BackendConfig(**...)``)
                # because the chart-rendered config.json stores the
                # backend kind under the JSON-friendly ``type`` key
                # while the dataclass attribute is ``backend_type``.
                # ``from_dict`` does the rename + nested profile parse;
                # plain kwargs raise ``unexpected keyword argument 'type'``.
                bc = BackendConfig.from_dict(backend_config)
            else:
                bc = backend_config
            if isinstance(generator_config, dict):
                generator_config = {**generator_config}
                generator_config.setdefault("tenant_id", tenant_id)
                # Use ``from_dict`` (not ``SyntheticGeneratorConfig(**...)``)
                # because the nested ``optimizer_configs[key]`` values
                # need to be hydrated as ``OptimizerGenerationConfig``
                # instances — kwargs construction leaves them as raw
                # dicts, so the generator later trips on
                # ``'dict' object has no attribute 'profile_scoring_rules'``.
                gc = SyntheticGeneratorConfig.from_dict(generator_config)
            else:
                gc = generator_config

            from pathlib import Path

            from cogniverse_core.registries.backend_registry import BackendRegistry
            from cogniverse_core.schemas.filesystem_loader import (
                FilesystemSchemaLoader,
            )

            # BackendRegistry is a singleton — its __new__ takes no args.
            # get_search_backend is the public accessor; tenant isolation
            # is per-query via tenant_id in query_dict, so we don't pass
            # tenant_id here. Backend name comes from the resolved
            # backend config (defaults to "vespa"). schema_loader is
            # required for backend init — match what the ingestion v2
            # worker does (see ingestion_v2/worker.py).
            schemas_dir = Path(
                os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas")
            )
            registry = BackendRegistry()
            backend = registry.get_search_backend(
                name=bc.backend_type,
                config_manager=config_manager,
                schema_loader=FilesystemSchemaLoader(schemas_dir),
            )

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


async def run_ab_compare(
    *,
    tenant_id: str,
    queries_dataset: str,
    judge_substring: Optional[str] = None,
    rlm_max_iterations: int = 10,
    rlm_max_llm_calls: int = 30,
) -> Dict[str, Any]:
    """run RLMABRunner over a Phoenix queries dataset.

    The dataset must contain rows with at least ``query`` and ``context``
    columns (Phoenix wraps these under ``input``/``output`` dicts when
    saved with input_keys; we flatten on load). For each row we run both
    arms and emit a Phoenix span (``rlm.ab_compare``) with the harness's
    ``to_telemetry_dict()`` as attributes — that's what the dashboard
    tile will read.

    Optional ``judge_substring`` enables a deterministic substring-match
    judge (1.0 if the substring appears in the answer, 0.0 otherwise).
    Real eval-time judges should be wired by the caller; this is the
    minimum viable judge for getting a `judge_delta` populated in CI.

    Returns aggregated stats so the operator can see per-dataset trends
    without tailing Phoenix.
    """
    from opentelemetry import trace
    from phoenix.client import Client as PhoenixSyncClient

    from cogniverse_agents.inference.ab_harness import RLMABRunner
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    config_manager = create_default_config_manager()
    cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_primary = cfg.get_llm_config().primary

    phoenix_http = os.environ.get("PHOENIX_HTTP_ENDPOINT", "http://localhost:6006")
    sync_client = PhoenixSyncClient(base_url=phoenix_http)

    try:
        dataset = sync_client.datasets.get_dataset(dataset=queries_dataset)
        df = dataset.to_dataframe()
    except Exception as exc:
        logger.error("ab-compare: dataset %r not loadable: %s", queries_dataset, exc)
        return {"status": "failed", "error": str(exc)}

    # Flatten input/output dicts the way run_triggered_optimization does.
    if "input" in df.columns and "query" not in df.columns:
        import pandas as _pd

        flat = []
        for _, row in df.iterrows():
            inp = row.get("input", {}) or {}
            out = row.get("output", {}) or {}
            flat.append({**inp, **out})
        df = _pd.DataFrame(flat)

    if "query" not in df.columns or "context" not in df.columns:
        return {
            "status": "failed",
            "error": (
                f"dataset {queries_dataset!r} must expose 'query' and 'context' "
                f"columns; got {list(df.columns)}"
            ),
        }

    judge = None
    if judge_substring:
        token = judge_substring

        def _substring_judge(_q: str, _ctx: str, ans: str) -> float:
            return 1.0 if token.lower() in (ans or "").lower() else 0.0

        judge = _substring_judge

    runner = RLMABRunner(
        llm_config=llm_primary,
        judge=judge,
        rlm_max_iterations=rlm_max_iterations,
        rlm_max_llm_calls=rlm_max_llm_calls,
    )

    tracer = trace.get_tracer("cogniverse.ab_compare")

    rows: list = []
    for _, r in df.iterrows():
        query = str(r["query"])
        context = str(r["context"])
        try:
            result = runner.run(query=query, context=context)
        except Exception as exc:
            logger.warning("ab-compare: arm failure on query=%r: %s", query[:60], exc)
            continue

        # Emit a Phoenix span with the comparison attributes — the dashboard
        # tile (when added) will aggregate over these.
        with tracer.start_as_current_span("rlm.ab_compare") as span:
            for k, v in result.to_telemetry_dict().items():
                if v is None:
                    continue
                span.set_attribute(f"openinference.{k}", v)
            span.set_attribute("openinference.tenant_id", tenant_id)
            span.set_attribute("openinference.queries_dataset", queries_dataset)
        rows.append(result)

    if not rows:
        return {
            "status": "failed",
            "error": "no rows produced both arms successfully",
            "queries_dataset": queries_dataset,
        }

    n = len(rows)
    avg_latency_delta = sum(r.comparison.latency_delta_ms for r in rows) / n
    avg_tokens_delta = sum(r.comparison.tokens_delta for r in rows) / n
    judge_deltas = [
        r.comparison.judge_delta for r in rows if r.comparison.judge_delta is not None
    ]
    avg_judge_delta = sum(judge_deltas) / len(judge_deltas) if judge_deltas else None
    fallback_count = sum(1 for r in rows if r.with_rlm.was_fallback)

    summary = {
        "status": "ok",
        "queries_dataset": queries_dataset,
        "tenant_id": tenant_id,
        "rows_compared": n,
        "avg_latency_delta_ms": avg_latency_delta,
        "avg_tokens_delta": avg_tokens_delta,
        "avg_judge_delta": avg_judge_delta,
        "rlm_fallback_rate": fallback_count / n,
        "ab_ids": [r.ab_id for r in rows],
    }
    logger.info("A/B compare complete: %s", summary)
    return summary


def run_egress_netpol(
    *,
    policy_dir: str,
    output_dir: str,
    service_map: Dict[str, str],
    namespace: str = "cogniverse",
    pod_app_label: str = "cogniverse",
    helm_conditional: Optional[str] = None,
    unified_pod_selector: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """emit k8s NetworkPolicy CRDs from agent policy YAMLs.

    Reads every YAML in ``policy_dir`` whose
    ``network_policies.deny_all_other`` is true, translates the egress
    list into NetworkPolicy egress rules.

    Two emit modes:
      * **per-agent (default)**: writes one NetworkPolicy per agent under
        ``output_dir/<agent>-egress-netpol.yaml`` selecting on
        ``app=<pod_app_label>, cogniverse-agent=<agent>``. Use this when
        each agent runs in its own Deployment so the labels match.
      * **unified-runtime** (``unified_pod_selector`` set): emits ONE
        NetworkPolicy named ``runtime-egress-netpol.yaml`` whose
        ``spec.egress`` is the de-duplicated UNION of every agent's
        allowed destinations and whose ``spec.podSelector.matchLabels``
        come from ``unified_pod_selector``. Use this when every agent
        runs inside a single shared runtime pod (the default helm chart
        topology) — per-agent L4 enforcement is impossible there, but
        cluster-wide deny-all-other-egress with a union allowlist is
        still real defense-in-depth on top of the application-layer
        OpenShell sandbox enforcement.

    Why this exists: the agent policy YAMLs declare per-agent egress
    constraints (Vespa for SearchAgent, the configured LM for SummarizerAgent,
    etc.) but in-process Python enforcement is fundamentally weak — a
    compromised process can ``socket.connect`` past any httpx wrapper.
    NetworkPolicy is enforced in the kernel by the cluster's CNI
    plugin (Cilium / Calico / etc.), so it's process-bypass-proof and
    independent of which HTTP library the agent uses.

    Args:
        policy_dir: Where the agent policy YAMLs live (default
            ``configs/agent_policies/``).
        output_dir: Where to write the generated NetworkPolicy YAMLs.
            Operators check these into the helm chart's
            ``templates/agent-egress/`` so helm applies them at
            deploy time.
        service_map: Logical service name → ``namespace/service-name:port``
            mapping (e.g. ``vespa=cogniverse/vespa-service:8080``). The
            policy YAML's ``localhost:N`` entries are matched by port
            against this map's values; the resulting NetworkPolicy uses
            podSelectors that target those services.
        namespace: k8s namespace the NetworkPolicy lives in.
        pod_app_label: ``app=`` label that selects cogniverse pods in
            per-agent mode. Ignored when ``unified_pod_selector`` is
            set.
        helm_conditional: Wrap each emitted YAML in a helm ``{{- if X }}``
            … ``{{- end }}`` so a values flag toggles application.
        unified_pod_selector: When provided, emit a single union policy
            selecting on these labels (e.g.
            ``{"app.kubernetes.io/component": "runtime"}``).

    Returns a summary dict suitable for the CLI's stdout JSON.
    """
    from pathlib import Path as _Path

    import yaml as _yaml

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build a port → (svc-namespace, svc-name, svc-port) lookup so the
    # localhost:N entries in the YAMLs can be resolved to the right
    # in-cluster service.
    port_to_service: Dict[int, Dict[str, Any]] = {}
    for logical, target in service_map.items():
        # Format: "namespace/service-name:port"
        try:
            ns_part, port_str = target.rsplit(":", 1)
            svc_namespace, svc_name = ns_part.split("/", 1)
            port = int(port_str)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"--service-map {logical}={target!r} is malformed; expected "
                "'namespace/service:port'"
            ) from exc
        port_to_service[port] = {
            "logical": logical,
            "namespace": svc_namespace,
            "service": svc_name,
            "port": port,
        }

    written: List[str] = []
    skipped: List[Dict[str, str]] = []

    # First pass: read every eligible policy + collect (agent, [egress_rules]).
    per_agent_rules: List[tuple] = []  # (agent_name, [egress_rules])
    for yaml_path in sorted(_Path(policy_dir).glob("*.yaml")):
        agent_name = yaml_path.stem
        with open(yaml_path) as f:
            policy_blob = _yaml.safe_load(f) or {}

        netpols = policy_blob.get("network_policies") or {}
        if not netpols.get("deny_all_other"):
            skipped.append({"agent": agent_name, "reason": "deny_all_other not set"})
            continue

        egress_rules: List[Dict[str, Any]] = []
        unmapped_ports: List[int] = []
        for rule in netpols.get("egress") or []:
            port = int(rule.get("port", 0))
            svc = port_to_service.get(port)
            if svc is None:
                unmapped_ports.append(port)
                continue
            egress_rules.append(
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {
                                    "kubernetes.io/metadata.name": svc["namespace"]
                                }
                            },
                            "podSelector": {"matchLabels": {"app": svc["service"]}},
                        }
                    ],
                    "ports": [
                        {
                            "port": svc["port"],
                            "protocol": str(rule.get("protocol", "tcp")).upper(),
                        }
                    ],
                }
            )

        if unmapped_ports:
            skipped.append(
                {
                    "agent": agent_name,
                    "reason": (
                        f"egress ports {sorted(set(unmapped_ports))} not in "
                        "--service-map"
                    ),
                }
            )
            continue

        # DNS is mandatory for any egress to resolve service names.
        egress_rules.append(
            {
                "to": [
                    {
                        "namespaceSelector": {
                            "matchLabels": {
                                "kubernetes.io/metadata.name": "kube-system"
                            }
                        },
                        "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
                    }
                ],
                "ports": [
                    {"port": 53, "protocol": "UDP"},
                    {"port": 53, "protocol": "TCP"},
                ],
            }
        )

        per_agent_rules.append((agent_name, egress_rules))

    # Second pass: emit either one union policy (unified mode) or one
    # policy per agent (legacy per-agent mode).
    if unified_pod_selector:
        # De-duplicate egress rules across agents — two agents that both
        # need DNS or both need vespa shouldn't produce duplicate yaml
        # entries.
        union: List[Dict[str, Any]] = []
        seen_keys = set()
        for _agent, rules in per_agent_rules:
            for rule in rules:
                key = _yaml.safe_dump(rule, sort_keys=True)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                union.append(rule)

        netpol_doc = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "cogniverse-runtime-egress",
                "namespace": namespace,
                "labels": {"cogniverse-component": "runtime"},
            },
            "spec": {
                "podSelector": {"matchLabels": dict(unified_pod_selector)},
                "policyTypes": ["Egress"],
                "egress": union,
            },
        }
        out_path = out / "runtime-egress-netpol.yaml"
        with open(out_path, "w") as f:
            if helm_conditional:
                f.write("{{- if " + helm_conditional + " }}\n")
            _yaml.safe_dump(netpol_doc, f, sort_keys=False, default_flow_style=False)
            if helm_conditional:
                f.write("{{- end }}\n")
        written.append(str(out_path))
        logger.info(
            "Wrote unified NetworkPolicy → %s (%d egress rules from %d agents)",
            out_path,
            len(union),
            len(per_agent_rules),
        )
    else:
        for agent_name, egress_rules in per_agent_rules:
            netpol_doc = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": f"cogniverse-{agent_name.replace('_', '-')}-egress",
                    "namespace": namespace,
                    "labels": {"cogniverse-agent": agent_name},
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {
                            "app": pod_app_label,
                            "cogniverse-agent": agent_name,
                        }
                    },
                    "policyTypes": ["Egress"],
                    "egress": egress_rules,
                },
            }

            out_path = out / f"{agent_name}-egress-netpol.yaml"
            with open(out_path, "w") as f:
                if helm_conditional:
                    f.write("{{- if " + helm_conditional + " }}\n")
                _yaml.safe_dump(
                    netpol_doc, f, sort_keys=False, default_flow_style=False
                )
                if helm_conditional:
                    f.write("{{- end }}\n")
            written.append(str(out_path))
            logger.info(
                "Wrote NetworkPolicy for %s → %s (%d egress rules)",
                agent_name,
                out_path,
                len(egress_rules),
            )

    return {
        "status": "ok",
        "policy_dir": str(policy_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "service_map": service_map,
        "mode": "unified" if unified_pod_selector else "per-agent",
    }


def _build_phoenix_provider_for_cli(tenant_id: str):
    """Construct a PhoenixProvider directly from env vars for CLI runs.

    Operators (and integration tests) set ``PHOENIX_HTTP_ENDPOINT`` and
    ``PHOENIX_GRPC_ENDPOINT`` to point at the Phoenix instance the CLI
    should talk to. We build the provider directly here rather than
    going through ``get_telemetry_manager()`` so a CLI invocation can
    target a specific Phoenix without the global telemetry config (which
    is loaded from ConfigManager and pinned to the cluster's primary).
    """
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    http_endpoint = os.environ.get("PHOENIX_HTTP_ENDPOINT", "http://localhost:6006")
    grpc_endpoint = os.environ.get("PHOENIX_GRPC_ENDPOINT", "localhost:4317")
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": http_endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return provider


async def run_rollback(
    *,
    tenant_id: str,
    agent_type: str,
    prompts_version: Optional[int] = None,
    demos_version: Optional[int] = None,
) -> Dict[str, Any]:
    """restore active artefacts to a previously-snapshotted version.

    Wraps :meth:`ArtifactManager.rollback_to_version` so an operator can
    run e.g. ``cogniverse-optim --mode rollback --tenant-id acme
    --agent search_agent --prompts-version 3``.

    The current active artefacts are themselves snapshotted before the
    rollback (the manager method does this) so the rollback is itself
    reversible — the returned ``backup_versions`` dict contains the
    versions you'd pass to ``rollback`` again to undo this operation.
    """
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    telemetry_provider = _build_phoenix_provider_for_cli(tenant_id)
    am = ArtifactManager(telemetry_provider, tenant_id)
    logger.info(
        "Rollback: tenant=%s agent=%s prompts_v=%s demos_v=%s",
        tenant_id,
        agent_type,
        prompts_version,
        demos_version,
    )
    summary = await am.rollback_to_version(
        agent_type=agent_type,
        prompts_version=prompts_version,
        demos_version=demos_version,
    )
    logger.info("Rollback complete: %s", summary)
    return summary


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
            "synthetic",
            "rollback",
            "ab-compare",
            "egress-netpol",
        ],
        required=True,
    )
    # --tenant-id is required for most modes; cleanup mode is the
    # exception and runs globally when omitted, so the daily-cleanup
    # CronWorkflow (which has no tenant) doesn't exit 2 on argparse.
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="Tenant ID (required for all modes except --mode cleanup)",
    )
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
    # rollback mode args. Operators run e.g.
    #   cogniverse-optim --mode rollback --tenant-id acme \
    #       --agent search_agent --prompts-version 3
    # to restore search_agent's active prompts to v3. Demos rollback is
    # independent so a caller can roll back just one or both.
    parser.add_argument(
        "--agent",
        help="Single agent name (rollback mode)",
    )
    parser.add_argument(
        "--prompts-version",
        type=int,
        help="Prompts version to restore (rollback mode)",
    )
    parser.add_argument(
        "--demos-version",
        type=int,
        help="Demonstrations version to restore (rollback mode)",
    )
    # ab-compare mode args. Operators run e.g.
    #   cogniverse-optim --mode ab-compare --tenant-id acme \
    #       --queries-dataset golden_eval_v1 [--judge-substring 'Paris']
    parser.add_argument(
        "--queries-dataset",
        help="Phoenix dataset of (query, context) rows (ab-compare mode)",
    )
    parser.add_argument(
        "--judge-substring",
        help="Optional substring judge for ab-compare mode (1.0 if present)",
    )
    parser.add_argument(
        "--rlm-max-iterations",
        type=int,
        default=10,
        help="Per-arm RLM iteration cap (ab-compare mode)",
    )
    parser.add_argument(
        "--rlm-max-llm-calls",
        type=int,
        default=30,
        help="Per-arm RLM total LLM call cap (ab-compare mode)",
    )
    # egress-netpol mode args. Generates k8s NetworkPolicy CRDs from
    # the agent policy YAMLs in configs/agent_policies/. Operators run e.g.
    #   cogniverse-optim --mode egress-netpol \
    #       --policy-dir configs/agent_policies/ \
    #       --output-dir charts/cogniverse/templates/networkpolicies/ \
    #       --service-map vespa=cogniverse/vespa-service:8080 \
    #       --service-map llm=cogniverse/llm-service:11434
    parser.add_argument(
        "--policy-dir",
        default="configs/agent_policies",
        help="Source directory of agent policy YAMLs (egress-netpol mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write generated NetworkPolicy YAMLs (egress-netpol mode)",
    )
    parser.add_argument(
        "--service-map",
        action="append",
        default=[],
        help=(
            "Logical service mapping `name=namespace/service:port` "
            "(repeatable; egress-netpol mode)"
        ),
    )
    parser.add_argument(
        "--netpol-namespace",
        default="cogniverse",
        help="k8s namespace for the generated NetworkPolicies",
    )
    parser.add_argument(
        "--netpol-app-label",
        default="cogniverse",
        help="Pod `app=` label that scopes the policies to cogniverse pods",
    )
    parser.add_argument(
        "--helm-conditional",
        default=None,
        help=(
            "When set, wrap each emitted YAML in `{{- if <expr> }}` ... "
            "`{{- end }}` so the helm chart's values.yaml flag toggles "
            "whether the NetworkPolicy applies. Example: "
            "`.Values.networkPolicy.agentEgress.enabled`."
        ),
    )
    parser.add_argument(
        "--unified-pod-selector",
        action="append",
        default=[],
        help=(
            "key=value (repeatable). When set, emit ONE NetworkPolicy "
            "selecting on these labels with the de-duplicated UNION of "
            "every agent's egress destinations. Use this for the "
            "default unified-runtime topology where all agents run in "
            "the same pod. Example: "
            "`--unified-pod-selector app.kubernetes.io/component=runtime`. "
            "When omitted, emits one NetworkPolicy per agent (legacy "
            "per-agent-pod topology)."
        ),
    )
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

    if args.mode != "cleanup" and args.mode != "egress-netpol" and not args.tenant_id:
        parser.error(f"--tenant-id is required for mode={args.mode!r}")

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
    elif args.mode == "rollback":
        if not args.agent or (
            args.prompts_version is None and args.demos_version is None
        ):
            parser.error(
                "--agent is required for rollback mode, plus at least one of "
                "--prompts-version or --demos-version"
            )
        result = asyncio.run(
            run_rollback(
                tenant_id=args.tenant_id,
                agent_type=args.agent,
                prompts_version=args.prompts_version,
                demos_version=args.demos_version,
            )
        )
    elif args.mode == "egress-netpol":
        if not args.output_dir:
            parser.error("--output-dir is required for egress-netpol mode")
        if not args.service_map:
            parser.error(
                "at least one --service-map is required for egress-netpol mode"
            )
        # Parse `name=ns/svc:port` pairs into a dict.
        sm: Dict[str, str] = {}
        for pair in args.service_map:
            if "=" not in pair:
                parser.error(f"--service-map {pair!r} missing '=' separator")
            k, v = pair.split("=", 1)
            sm[k.strip()] = v.strip()
        unified_selectors: Optional[Dict[str, str]] = None
        if args.unified_pod_selector:
            unified_selectors = {}
            for pair in args.unified_pod_selector:
                if "=" not in pair:
                    parser.error(
                        f"--unified-pod-selector {pair!r} missing '=' separator"
                    )
                k, v = pair.split("=", 1)
                unified_selectors[k.strip()] = v.strip()
        result = run_egress_netpol(
            policy_dir=args.policy_dir,
            output_dir=args.output_dir,
            service_map=sm,
            namespace=args.netpol_namespace,
            pod_app_label=args.netpol_app_label,
            helm_conditional=args.helm_conditional,
            unified_pod_selector=unified_selectors,
        )
    elif args.mode == "ab-compare":
        if not args.queries_dataset:
            parser.error("--queries-dataset is required for ab-compare mode")
        result = asyncio.run(
            run_ab_compare(
                tenant_id=args.tenant_id,
                queries_dataset=args.queries_dataset,
                judge_substring=args.judge_substring,
                rlm_max_iterations=args.rlm_max_iterations,
                rlm_max_llm_calls=args.rlm_max_llm_calls,
            )
        )
    elif args.mode == "synthetic":
        optimizer_types = ["simba", "profile", "workflow"]
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
