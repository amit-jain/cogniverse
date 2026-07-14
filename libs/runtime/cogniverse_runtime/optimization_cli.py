"""CLI entry point for optimization — called by Argo CronWorkflows.

Per-agent batch optimization modes. Each mode reads production spans from
Phoenix, builds DSPy training examples, compiles optimized modules, and
saves artifacts via ArtifactManager. Agents load artifacts at startup.

Usage:
    python -m cogniverse_runtime.optimization_cli --mode simba --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode workflow --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode gateway-thresholds --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode online-routing-eval --tenant-id acme:production
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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from cogniverse_foundation.telemetry.span_contract import read_span_io

logger = logging.getLogger(__name__)


def _query_enhancement_pairs(spans_df) -> List[Dict[str, Any]]:
    """(original_query -> enhanced_query) training pairs from query_enhancement spans.

    Reads the canonical span slots: input.value holds the original query, and
    output.value holds ``{"enhanced_query", "confidence", ...}``. Identity pairs
    (enhanced == original) are dropped so SIMBA never trains on no-ops.
    """
    pairs: List[Dict[str, Any]] = []
    for _, row in spans_df.iterrows():
        span_io = read_span_io(row)
        original = span_io["input"] or ""
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}
        enhanced = output.get("enhanced_query", "")
        if not original or not enhanced:
            continue
        if enhanced.strip() == original.strip():
            continue
        pairs.append(
            {
                "query": original,
                "enhanced_query": enhanced,
                "confidence": float(output.get("confidence", 0.0) or 0.0),
            }
        )
    return pairs


def _entity_extraction_pairs(spans_df) -> List[Dict[str, Any]]:
    """(query -> entities) training pairs from entity_extraction spans.

    Reads the canonical span slots: input.value holds the query, output.value
    holds ``{"entities": [...], ...}``.
    """
    pairs: List[Dict[str, Any]] = []
    for _, row in spans_df.iterrows():
        span_io = read_span_io(row)
        query = span_io["input"] or ""
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}
        entities = output.get("entities", [])
        if not query or not entities:
            continue
        pairs.append({"query": query, "entities": entities})
    return pairs


def _profile_selection_pairs(spans_df) -> List[Dict[str, Any]]:
    """(query -> selected_profile) training pairs from profile_selection spans.

    Reads the canonical span slots: input.value holds the query, output.value
    holds ``{"selected_profile", "modality", "complexity", "intent",
    "confidence"}``. Only high-confidence (>= 0.5) selections are kept.
    """
    pairs: List[Dict[str, Any]] = []
    for _, row in spans_df.iterrows():
        span_io = read_span_io(row)
        query = span_io["input"] or ""
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}
        selected = output.get("selected_profile", "")
        confidence = float(output.get("confidence", 0.0) or 0.0)
        if not query or not selected or confidence < 0.5:
            continue
        pairs.append(
            {
                "query": query,
                "selected_profile": selected,
                "modality": output.get("modality", "video"),
                "complexity": output.get("complexity", "simple"),
                "intent": output.get("intent", ""),
                "confidence": confidence,
            }
        )
    return pairs


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
                teacher_endpoint=llm_config.resolve_teacher(),
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
    teacher_endpoint=None,
) -> dict:
    """Run DSPy optimization for a specific agent using scored examples."""
    import json as _json

    from cogniverse_agents.optimizer.dspy_agent_optimizer import (
        DSPyAgentPromptOptimizer,
    )

    optimizer = DSPyAgentPromptOptimizer()
    optimizer.initialize_language_model(
        llm_endpoint, teacher_endpoint_config=teacher_endpoint
    )

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
        teacher_settings=optimizer.optimization_settings["teacher_settings"],
    )

    module = dspy.ChainOfThought(signature)

    try:
        # The compile must run with the LM configured — initialize_language_model
        # only sets optimizer.lm, it does not configure DSPy's global LM (unlike
        # the sibling modes which call dspy.configure). Scope it to the compile.
        with dspy.context(lm=optimizer.lm):
            compiled = teleprompter.compile(module, trainset=trainset)

        # Store compiled module via ArtifactManager (same path the other modes use)
        import json as _json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        artifact_id = await artifact_manager.save_blob(
            kind="model",
            key=f"dspy_compiled_{agent_name}",
            content=_json.dumps(compiled.dump_state(), default=str),
        )

        return {
            "status": "success",
            "artifact_id": artifact_id,
            "training_examples": len(trainset),
        }

    except Exception as e:
        logger.error(f"DSPy compilation failed for {agent_name}: {e}")
        return {"status": "failed", "error": str(e)}


def _prune_aged_files(root: str, *, older_than_days: float) -> dict:
    """Delete files under ``root`` whose mtime is older than the cutoff.

    Returns a dict ``{"scanned": N, "deleted": M, "errors": [..]}`` so
    the workflow log captures exact numbers — the assertion contract
    for the daily-cleanup e2e test depends on tight outcome reporting,
    not opaque ``cleanup completed`` markers.

    Silent no-op when ``root`` does not exist or is not a directory —
    the cron container may run on a pod that doesn't mount that path
    (e.g. ``/logs`` only exists when the runtime container mounts a
    log PVC). Logged at INFO so the workflow run records "skipped: no
    such path".
    """
    import time as _t
    from pathlib import Path as _Path

    summary: dict = {"path": root, "scanned": 0, "deleted": 0, "errors": []}
    p = _Path(root)
    if not p.is_dir():
        summary["skipped"] = f"path {root!r} is not a directory"
        return summary

    cutoff = _t.time() - older_than_days * 86400
    for entry in p.rglob("*"):
        if not entry.is_file():
            continue
        summary["scanned"] += 1
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                summary["deleted"] += 1
        except OSError as exc:
            summary["errors"].append(f"{entry}: {exc}")
    return summary


def _vacuum_config_metadata(*, keep_versions: int) -> dict:
    """Drain config_metadata version bloat across every config_id.

    Per-write pruning in ``VespaConfigStore.set_config`` keeps fresh
    writes bounded, but a backlog can accumulate when ``keep_versions``
    is bumped or when a backend write path skipped the prune (e.g. an
    older runtime image). One-off sweep here brings stale rows down
    to ``keep_versions`` per config_id and returns the count dropped
    so the workflow log proves the work happened.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    cm = create_default_config_manager()
    store = cm.store
    if not isinstance(store, VespaConfigStore):
        return {
            "skipped": f"store is {type(store).__name__}, expected VespaConfigStore"
        }

    dropped = store.prune_all_configs(keep=keep_versions)
    return {"dropped": dropped, "keep_versions": keep_versions}


async def run_cleanup(
    tenant_id: Optional[str],
    log_retention_days: int,
    memory_retention_days: int,
) -> dict:
    """Daily-cleanup workflow body: memory + logs + temp + config vacuum.

    Per-tenant Mem0 cleanup is schema-driven (per-kind TTLs in the
    KnowledgeRegistry). The other three steps absorbed the
    standalone ``daily-cleanup`` CronWorkflow that the chart didn't
    previously cover:

      * Log rotation under ``LOG_DIR`` (default ``/logs``) — files
        older than ``log_retention_days`` are removed.
      * Temp file cleanup under ``TEMP_DIR`` (default ``/tmp``) —
        files older than 1 day are removed.
      * config_metadata version vacuum — each config_id is pruned to
        the latest ``CONFIG_KEEP_VERSIONS`` (default 10).

    Each section reports exact counts in the result dict so the
    workflow run log proves the work landed — bare "Succeeded" is too
    weak a signal for a maintenance cron.
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

    # --- Memory cleanup (per tenant) ---
    if tenant_id is not None:
        results["memory_cleanup"] = {tenant_id: _cleanup_one(tenant_id)}
    else:
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

    # --- Log rotation ---
    log_dir = os.environ.get("LOG_DIR", "/logs")
    results["log_cleanup"] = _prune_aged_files(
        log_dir, older_than_days=float(log_retention_days)
    )

    # --- Temp file cleanup ---
    temp_dir = os.environ.get("TEMP_DIR", "/tmp")
    temp_age_days = float(os.environ.get("TEMP_RETENTION_DAYS", "1"))
    results["temp_cleanup"] = _prune_aged_files(temp_dir, older_than_days=temp_age_days)

    # --- Config metadata vacuum ---
    keep_versions = int(os.environ.get("CONFIG_KEEP_VERSIONS", "10"))
    try:
        results["config_vacuum"] = _vacuum_config_metadata(keep_versions=keep_versions)
    except Exception as exc:  # noqa: BLE001 — best-effort vacuum
        results["config_vacuum"] = {"failed": str(exc)}

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

    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    telemetry_manager = get_telemetry_manager()
    project_name = telemetry_manager.config.get_project_name(tenant_id)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            spans_df = await telemetry_provider.traces.get_spans(
                project=project_name,
                start_time=start_time,
                end_time=end_time,
                # Server-side name predicate — pulling the whole project
                # window and filtering client-side costs a full scan of a
                # project that accumulates thousands of spans a day.
                filters={"name": span_name},
                limit=10000,
            )
            break
        except Exception as e:
            last_exc = e
            logger.warning(
                "Span query for %s failed (attempt %d/3): %s", span_name, attempt + 1, e
            )
            await asyncio.sleep(5)
    else:
        # A failed query is not "no spans" — reporting no_data here made a
        # Phoenix timeout look like an empty optimization window.
        raise RuntimeError(
            f"Failed to query {span_name} spans from Phoenix after 3 attempts"
        ) from last_exc

    if spans_df.empty:
        return spans_df

    return spans_df[spans_df["name"] == span_name]


def _create_teleprompter(trainset_size: int, teacher_settings: dict | None = None):
    """Select DSPy optimizer config based on training set size.

    Scales BootstrapFewShot parameters for larger training sets:
    - < 50 examples: 4 bootstrapped demos, 8 labeled, 1 round
    - >= 50 examples: 8 bootstrapped demos, 16 labeled, 2 rounds

    teacher_settings (e.g. ``{"lm": teacher_lm}``) makes DSPy run the
    bootstrap teacher on the configured teacher endpoint instead of the
    student model teaching itself.
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
            teacher_settings=teacher_settings,
        )

    logger.info("Using BootstrapFewShot for %d examples", trainset_size)
    return BootstrapFewShot(
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
        max_errors=5,
        teacher_settings=teacher_settings,
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
        from cogniverse_core.approval.interfaces import ApprovalStatus

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


async def run_monthly_reports(
    output_dir: str,
    lookback_hours: float = 24.0 * 30,
) -> dict:
    """Generate the monthly usage + performance report.

    Replaces the standalone ``monthly-reports`` CronWorkflow that was
    a kubectl-applied stub (echoed empty JSON). This version collects
    real data:

      * **usage**: total orgs, total tenants per org, total schemas
        deployed per tenant (from organization_metadata + tenant_metadata).
      * **performance**: per-tenant span count, mean / p50 / p95 latency
        across every span the project emitted in the lookback window,
        plus error rate (status_code != OK).

    Writes ``usage-YYYYMM.json`` and ``performance-YYYYMM.json`` to
    ``output_dir`` so a follow-up workflow step can upload to MinIO via
    ``mc cp``. Returns a summary the workflow log captures verbatim.
    """
    import json
    from pathlib import Path

    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager
    from cogniverse_runtime.admin import tenant_manager

    create_default_config_manager()  # warm config singletons
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    tenant_manager.set_schema_loader(FilesystemSchemaLoader(schemas_dir))

    period = datetime.now().strftime("%Y%m")
    generated_at = datetime.now(timezone.utc).isoformat() + "Z"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- usage ---
    org_ids = await tenant_manager.list_organizations_internal()
    usage_per_org: Dict[str, Any] = {}
    total_tenants = 0
    total_schemas = 0
    for oid in org_ids:
        tenants = await tenant_manager.list_tenants_for_org_internal(oid)
        per_org_tenants = []
        for t in tenants:
            schemas = list(t.schemas_deployed or [])
            per_org_tenants.append(
                {
                    "tenant_full_id": t.tenant_full_id,
                    "tenant_name": t.tenant_name,
                    "status": t.status,
                    "schema_count": len(schemas),
                    "schemas_deployed": schemas,
                }
            )
            total_schemas += len(schemas)
        total_tenants += len(per_org_tenants)
        usage_per_org[oid] = {
            "tenant_count": len(per_org_tenants),
            "tenants": per_org_tenants,
        }
    usage_report = {
        "period": period,
        "generated_at": generated_at,
        "summary": {
            "org_count": len(org_ids),
            "tenant_count": total_tenants,
            "schema_count": total_schemas,
        },
        "organizations": usage_per_org,
    }
    usage_path = out / f"usage-{period}.json"
    usage_path.write_text(json.dumps(usage_report, indent=2, default=str))

    # --- performance ---
    telemetry_manager = get_telemetry_manager()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    perf_per_tenant: Dict[str, Any] = {}
    tenant_ids = [
        t.tenant_full_id
        for oid in org_ids
        for t in await tenant_manager.list_tenants_for_org_internal(oid)
        if t.tenant_full_id
    ]
    for tid in tenant_ids:
        provider = telemetry_manager.get_provider(tenant_id=tid)
        project = telemetry_manager.config.get_project_name(tid)
        try:
            spans_df = await provider.traces.get_spans(
                project=project,
                start_time=start,
                end_time=end,
                limit=10000,
            )
        except Exception as exc:
            perf_per_tenant[tid] = {"error": f"phoenix query failed: {exc}"}
            continue
        if spans_df is None or spans_df.empty:
            perf_per_tenant[tid] = {
                "span_count": 0,
                "latency_ms_mean": None,
                "latency_ms_p50": None,
                "latency_ms_p95": None,
                "error_rate": 0.0,
            }
            continue

        # Pyhoenix dataframes expose `latency_ms` (start_time, end_time)
        # and a status_code column; fall back gracefully if either is
        # absent in older provider versions.
        latencies = []
        if "latency_ms" in spans_df.columns:
            latencies = [v for v in spans_df["latency_ms"].dropna() if v >= 0]
        elif {"start_time", "end_time"}.issubset(spans_df.columns):
            for s, e in zip(spans_df["start_time"], spans_df["end_time"]):
                try:
                    latencies.append((e - s).total_seconds() * 1000.0)
                except Exception:
                    continue
        errors = 0
        if "status_code" in spans_df.columns:
            errors = int(
                spans_df["status_code"].fillna("OK").str.upper().ne("OK").sum()
            )
        n = len(spans_df)
        latencies_sorted = sorted(latencies) if latencies else []

        def _pct(lst: list, q: float):
            if not lst:
                return None
            idx = max(0, min(len(lst) - 1, int(q * (len(lst) - 1))))
            return round(float(lst[idx]), 3)

        perf_per_tenant[tid] = {
            "span_count": int(n),
            "latency_ms_mean": (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            ),
            "latency_ms_p50": _pct(latencies_sorted, 0.50),
            "latency_ms_p95": _pct(latencies_sorted, 0.95),
            "error_rate": round(errors / n, 4) if n else 0.0,
        }
    perf_report = {
        "period": period,
        "generated_at": generated_at,
        "lookback_hours": lookback_hours,
        "tenants": perf_per_tenant,
    }
    perf_path = out / f"performance-{period}.json"
    perf_path.write_text(json.dumps(perf_report, indent=2, default=str))

    return {
        "period": period,
        "generated_at": generated_at,
        "output_dir": str(out),
        "files_written": [str(usage_path), str(perf_path)],
        "summary": {
            "org_count": len(org_ids),
            "tenant_count": total_tenants,
            "perf_tenants_with_data": sum(
                1
                for v in perf_per_tenant.values()
                if isinstance(v, dict) and v.get("span_count", 0) > 0
            ),
        },
    }


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

    # Build DSPy training examples from the canonical span slots.
    import dspy

    trainset = [
        dspy.Example(
            query=pair["query"],
            enhanced_query=pair["enhanced_query"],
            expansion_terms="",
            synonyms="",
            context="",
            confidence=str(pair["confidence"]),
            reasoning="From production span",
        ).with_inputs("query")
        for pair in _query_enhancement_pairs(spans_df)
    ]

    if not trainset:
        logger.info("No valid training examples extracted from spans")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    logger.info("Built %d training examples for SIMBA compilation", len(trainset))

    # Merge approved synthetic data
    import json as _json

    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "query_enhancement"
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

    teleprompter = _create_teleprompter(
        len(trainset),
        teacher_settings={"lm": create_dspy_lm(llm_config.resolve_teacher())},
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

    # Persist through the workflow store — the same registry-resolved store
    # WorkflowIntelligence reads back at orchestrator startup. Stale demos
    # (agents no longer in the live config) are dropped first; the store owns
    # serialization and the demonstration/blob layout.
    from cogniverse_core.registries import WorkflowStoreRegistry

    store = WorkflowStoreRegistry.get(name="telemetry")

    live_executions = [
        execution
        for execution in intelligence.workflow_history
        if _agents_live(execution.agent_sequence)
    ]
    await store.save_executions(tenant_id, live_executions)

    profiles = list(intelligence.agent_performance.values())
    await store.save_agent_profiles(tenant_id, profiles)

    if intelligence.query_type_patterns:
        await store.save_query_patterns(
            tenant_id, dict(intelligence.query_type_patterns)
        )

    logger.info("Workflow optimization complete")
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "workflows_extracted": workflows_extracted,
        "execution_demos_saved": len(live_executions),
        "agent_profiles_saved": len(profiles),
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

    # The gateway decision is on the canonical output.value slot.
    df = spans_df.copy()

    def _gateway_output(row) -> Dict[str, Any]:
        out = read_span_io(row)["output"]
        return out if isinstance(out, dict) else {}

    gw_outputs = df.apply(_gateway_output, axis=1)
    if gw_outputs.map(bool).sum() == 0:
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "reason": "no_gateway_attributes",
        }

    df["_complexity"] = gw_outputs.apply(lambda d: d.get("complexity", ""))
    df["_confidence"] = gw_outputs.apply(lambda d: d.get("confidence", None))

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


async def run_online_routing_evaluation(
    tenant_id: str,
    lookback_hours: float = 24.0,
) -> dict:
    """Online routing-span scoring.

    Reads cogniverse.routing spans and scores each one (routing_outcome +
    confidence_calibration) via OnlineEvaluator, persisting the scores as
    telemetry annotations for drift detection. Sampling rate, evaluator set,
    and persistence are driven by automation_rules.online_evaluation in config.
    """
    from cogniverse_agents.routing.config import OnlineEvaluationConfig
    from cogniverse_evaluation.online_evaluator import OnlineEvaluator
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )
    from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    config_manager = create_default_config_manager()
    cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
    online_dict = (cfg.get_all().get("automation_rules") or {}).get(
        "online_evaluation"
    ) or {}
    online_cfg = OnlineEvaluationConfig(**online_dict)

    if not online_cfg.enabled:
        logger.info("Online routing evaluation disabled in config")
        return {"status": "disabled"}

    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)
    project_name = telemetry_manager.config.get_project_name(tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, SPAN_NAME_ROUTING, lookback_hours
    )
    if spans_df.empty:
        logger.info("No routing spans found — nothing to evaluate")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d routing spans", len(spans_df))

    evaluator = OnlineEvaluator(
        provider=telemetry_provider,
        project_name=project_name,
        config=online_cfg,
    )

    scores_persisted = 0
    for _, row in spans_df.iterrows():
        results = await evaluator.evaluate_span(row.to_dict())
        scores_persisted += len(results)

    stats = evaluator.get_statistics()
    logger.info(
        "Online routing evaluation complete — evaluated %d spans, persisted %d scores",
        stats["total_evaluated"],
        scores_persisted,
    )
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "scores_persisted": scores_persisted,
        "statistics": stats,
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

    # Build DSPy training examples from the canonical span slots.
    import dspy

    trainset = [
        dspy.Example(
            query=pair["query"],
            available_profiles="video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s,video_videoprism_base_mv_chunk_30s,video_videoprism_large_mv_chunk_30s",
            selected_profile=pair["selected_profile"],
            confidence=str(pair["confidence"]),
            reasoning=f"Selected {pair['selected_profile']} for {pair['modality']}/{pair['complexity']} query",
            query_intent=pair["intent"],
            modality=pair["modality"],
            complexity=pair["complexity"],
        ).with_inputs("query", "available_profiles")
        for pair in _profile_selection_pairs(spans_df)
    ]

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

    teleprompter = _create_teleprompter(
        len(trainset),
        teacher_settings={"lm": create_dspy_lm(llm_config.resolve_teacher())},
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

    trainset = [
        dspy.Example(
            query=pair["query"],
            entities=_json.dumps(pair["entities"]),
            entity_types="",
        ).with_inputs("query")
        for pair in _entity_extraction_pairs(spans_df)
    ]

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

    teleprompter = _create_teleprompter(
        len(trainset),
        teacher_settings={"lm": create_dspy_lm(llm_config.resolve_teacher())},
    )

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
        optimizer_types = ["query_enhancement", "profile", "workflow"]

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
    # entity-extraction) call ``dspy.configure(lm=create_dspy_lm(...))``
    # at the synchronous top level; this function runs in an asyncio task
    # so ``dspy.configure`` would raise (it can only be called from the
    # same async task that first called it). Use a process-wide thread-
    # local equivalent: set ``dspy.settings.lm`` directly. DSPy modules
    # read this attribute when no explicit ``lm=`` is passed.
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm

    llm_endpoint = config.get_llm_config().primary
    dspy.settings.lm = create_dspy_lm(llm_endpoint)

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
            # worker does (see ingestion_worker/worker.py).
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
        tenant_id=tenant_id,
        config_manager=config_manager,
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
    # policy per agent (per-agent mode).
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


def build_parser() -> argparse.ArgumentParser:
    """Build the optimization CLI argument parser.

    Exposed separately from ``main`` so tests assert against the REAL parser
    (its modes, defaults, required flags) instead of a hand-built copy that
    can silently drift from production.
    """
    parser = argparse.ArgumentParser(description="Cogniverse Optimization CLI")
    parser.add_argument(
        "--mode",
        choices=[
            "cleanup",
            "triggered",
            "simba",
            "workflow",
            "gateway-thresholds",
            "online-routing-eval",
            "profile",
            "entity-extraction",
            "synthetic",
            "rollback",
            "ab-compare",
            "egress-netpol",
            "monthly-reports",
        ],
        required=True,
    )
    # monthly-reports writes its JSON output here for a follow-up
    # workflow step to upload via mc. Inside the cron pod this is a
    # mounted emptyDir / PVC; local CLI runs default to ./reports.
    parser.add_argument(
        "--reports-output-dir",
        default="./reports",
        help="Output directory for monthly-reports mode (default: ./reports)",
    )
    # --tenant-id is required for most modes; cleanup + monthly-reports
    # are the exceptions and run globally when omitted, so the
    # daily-cleanup / monthly-reports CronWorkflows (no tenant) don't
    # exit 2 on argparse.
    parser.add_argument(
        "--tenant-id",
        default=None,
        help=(
            "Tenant ID (required for all modes except --mode cleanup / "
            "--mode monthly-reports)"
        ),
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
            "When omitted, emits one NetworkPolicy per agent "
            "(per-agent-pod topology)."
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if (
        args.mode not in ("cleanup", "egress-netpol", "monthly-reports")
        and not args.tenant_id
    ):
        parser.error(f"--tenant-id is required for mode={args.mode!r}")

    if args.mode == "cleanup":
        result = asyncio.run(
            run_cleanup(
                args.tenant_id, args.log_retention_days, args.memory_retention_days
            )
        )
    elif args.mode == "monthly-reports":
        result = asyncio.run(
            run_monthly_reports(
                output_dir=args.reports_output_dir,
                lookback_hours=args.lookback_hours,
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
    elif args.mode == "online-routing-eval":
        result = asyncio.run(
            run_online_routing_evaluation(
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
        optimizer_types = ["query_enhancement", "profile", "workflow"]
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
