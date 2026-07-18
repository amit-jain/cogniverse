"""CLI entry point for quality monitor sidecar.

Runs the continuous quality evaluation loop that scores all agents
and triggers Argo optimization workflows when quality degrades.

Usage:
    python -m cogniverse_runtime.quality_monitor_cli \
        --runtime-url http://cogniverse-runtime:28000 \
        --phoenix-url http://cogniverse-phoenix:6006 \
        --argo-url http://argo-server:2746 \
        --argo-namespace cogniverse
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _load_automation_rules(tenant_id: str):
    """automation_rules from the tenant config, defaults on any failure.

    The sidecar must keep monitoring even when the config store is
    unreachable, so a read failure degrades to the declared defaults
    (with a warning) instead of killing the loop.
    """
    from cogniverse_agents.routing.config import AutomationRulesConfig

    try:
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        cfg = get_config(
            tenant_id=tenant_id, config_manager=create_default_config_manager()
        )
        section = cfg.get_all().get("automation_rules") or {}
        return AutomationRulesConfig.from_dict(section)
    except Exception as e:
        logger.warning("automation_rules config read failed (%r); using defaults", e)
        return AutomationRulesConfig()


async def run_annotation_cycle(
    tenant_id: str,
    runtime_url: str,
    lookback_hours: Optional[int] = None,
    agent_types: Optional[list] = None,
    http_client=None,
    automation_rules=None,
) -> dict:
    """Identify spans needing human review and enqueue them on the runtime.

    One stateless pass: for each agent type, ``AnnotationAgent`` flags spans
    per the annotation thresholds, spans already carrying an annotation are
    dropped, the batch is capped at
    ``optimization_triggers.max_annotations_per_cycle``, and the remainder is
    POSTed to the runtime's ``/agents/annotations/queue/enqueue`` worklist.
    The in-memory queue is re-derivable: a runtime restart just means the next
    cycle repopulates it.
    """
    import httpx

    import cogniverse_agents.routing.annotation_agent as annotation_agent_mod
    import cogniverse_agents.routing.annotation_storage as annotation_storage_mod
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    tenant_id = canonical_tenant_id(tenant_id)
    rules = automation_rules or _load_automation_rules(tenant_id)
    triggers = rules.optimization_triggers
    if lookback_hours is None:
        lookback_hours = triggers.annotation_lookback_hours
    if agent_types is None:
        from cogniverse_evaluation.evaluators.agent_evaluators import AGENT_EVALUATORS

        agent_types = list(AGENT_EVALUATORS)

    identified = 0
    already_annotated = 0
    to_enqueue: list = []
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    for agent_type in agent_types:
        agent = annotation_agent_mod.AnnotationAgent(
            tenant_id=tenant_id, automation_rules=rules
        )
        requests = await agent.identify_spans_needing_annotation(
            lookback_hours=lookback_hours, agent_type=agent_type
        )
        identified += len(requests)
        if not requests:
            continue

        storage = annotation_storage_mod.AnnotationStorage(
            tenant_id=tenant_id, agent_type=agent_type
        )
        annotated_rows = await storage.query_annotated_spans(
            start_time=start_time, end_time=end_time, only_human_reviewed=False
        )
        annotated_ids = {row.get("span_id") for row in annotated_rows}

        for request in requests:
            if request.span_id in annotated_ids:
                already_annotated += 1
                continue
            request.tenant_id = tenant_id
            to_enqueue.append(request)

    if len(to_enqueue) > triggers.max_annotations_per_cycle:
        logger.info(
            "Capping %d annotation requests to max_annotations_per_cycle=%d",
            len(to_enqueue),
            triggers.max_annotations_per_cycle,
        )
        to_enqueue = to_enqueue[: triggers.max_annotations_per_cycle]

    enqueued = 0
    if to_enqueue:
        payload = {"requests": [r.to_dict() for r in to_enqueue]}
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.post(
                f"{runtime_url.rstrip('/')}/agents/annotations/queue/enqueue",
                json=payload,
            )
            response.raise_for_status()
            enqueued = response.json().get("enqueued", 0)
        finally:
            if owns_client:
                await client.aclose()

    result = {
        "identified": identified,
        "already_annotated": already_annotated,
        "enqueued": enqueued,
    }
    logger.info("Annotation cycle complete: %s", result)
    return result


async def _annotation_loop(tenant_id: str, runtime_url: str) -> None:
    """Sidecar loop: run the annotation cycle on the configured interval."""
    while True:
        rules = _load_automation_rules(tenant_id)
        try:
            await run_annotation_cycle(
                tenant_id=tenant_id, runtime_url=runtime_url, automation_rules=rules
            )
        except Exception as e:
            logger.error("Annotation cycle failed: %r", e)
        await asyncio.sleep(rules.intervals.annotation_interval_minutes * 60)


# Which optimization_cli mode recompiles each agent type. search/summary/
# report consume the scored trigger dataset; the rest have dedicated modes
# whose trainsets come from their own spans, so human-annotation volume is
# the WHEN signal for them. The gateway is heuristic by design (no LLM to
# compile) — routing feedback refreshes its threshold calibration instead.
AGENT_COMPILE_MODE = {
    "search": "triggered",
    "summary": "triggered",
    "report": "triggered",
    "query_enhancement": "simba",
    "entity_extraction": "entity-extraction",
    "profile_selection": "profile",
    "gateway": "gateway-thresholds",
    "routing": "gateway-thresholds",
}

_LOOP_STATE_SERVICE = "optimization_loop"
_LOOP_STATE_KEY = "state"


def _load_loop_state(config_manager, tenant_id: str) -> dict:
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    try:
        entry = config_manager.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.AGENT,
            service=_LOOP_STATE_SERVICE,
            config_key=_LOOP_STATE_KEY,
        )
        return dict(entry.config_value) if entry is not None else {}
    except Exception as e:
        logger.warning("optimization loop state read failed (%r); empty state", e)
        return {}


def _save_loop_state(config_manager, tenant_id: str, state: dict) -> None:
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    config_manager.store.set_config(
        tenant_id=tenant_id,
        scope=ConfigScope.AGENT,
        service=_LOOP_STATE_SERVICE,
        config_key=_LOOP_STATE_KEY,
        config_value=state,
    )


def _workflow_pod_spec_from_env():
    """Wiring for spawned optimization pods, from this pod's chart-set env.

    Returns None when OPTIMIZATION_WORKFLOW_IMAGE is unset (bare-manifest
    fallback). Read once at entrypoint; the spec is passed down explicitly.
    """
    from cogniverse_evaluation.quality_monitor import OptimizationWorkflowPodSpec

    image = os.environ.get("OPTIMIZATION_WORKFLOW_IMAGE")
    if not image:
        return None
    passthrough = (
        "BACKEND_URL",
        "BACKEND_PORT",
        "TELEMETRY_HTTP_ENDPOINT",
        "TELEMETRY_OTLP_ENDPOINT",
    )
    return OptimizationWorkflowPodSpec(
        image=image,
        env={name: os.environ[name] for name in passthrough if name in os.environ},
        config_map=os.environ.get("OPTIMIZATION_CONFIG_MAP"),
        dev_source_hostpath=os.environ.get("OPTIMIZATION_DEV_HOSTPATH"),
    )


async def run_annotation_feedback_cycle(
    tenant_id: str,
    argo_url: str,
    argo_namespace: str = "cogniverse",
    automation_rules=None,
    config_manager=None,
    http_client=None,
    dataset_store=None,
    now=None,
    force: bool = False,
    pod_spec=None,
) -> dict:
    """Turn accumulated human annotations into optimization submissions.

    Per agent type over ``annotation_lookback_hours``: count human-reviewed
    annotations; at ``min_annotations_for_optimization`` submit the agent's
    compile mode (search/summary/report additionally get a trigger dataset
    scored via ``quality_map``); gateway/routing get a cheaper
    ``gateway-thresholds`` refresh at ``min_annotations_for_update``. A
    per-agent ``min_days_between_optimizations`` cooldown and a
    ``poll_interval_minutes`` self-gate (both persisted in the config store)
    make the stateless cron safe to schedule densely.
    """
    import pandas as pd

    import cogniverse_agents.routing.annotation_storage as annotation_storage_mod
    from cogniverse_core.common.tenant_utils import canonical_tenant_id
    from cogniverse_evaluation.quality_monitor import (
        submit_argo_optimization_workflow,
    )

    # The runtime keys everything (span projects, artifacts) by the
    # canonical tenant form — a bare id here would read/write a parallel
    # world the runtime never touches.
    tenant_id = canonical_tenant_id(tenant_id)
    rules = automation_rules or _load_automation_rules(tenant_id)
    triggers = rules.optimization_triggers
    feedback = rules.feedback
    quality_map = feedback.quality_map
    now = now or datetime.now(timezone.utc)

    if config_manager is None:
        from cogniverse_foundation.config.utils import create_default_config_manager

        config_manager = create_default_config_manager()

    state = _load_loop_state(config_manager, tenant_id)

    last_poll = state.get("last_feedback_run_at")
    if last_poll and not force:
        elapsed = now - datetime.fromisoformat(last_poll)
        if elapsed < timedelta(minutes=feedback.poll_interval_minutes):
            logger.info(
                "Feedback cycle polled %.0fs ago (< poll_interval_minutes=%d) "
                "— skipping",
                elapsed.total_seconds(),
                feedback.poll_interval_minutes,
            )
            return {"status": "skipped_recent_poll"}

    # Argo submissions: pass the injected client through (tests capture it);
    # None lets the submit helper build its own self-signed-TLS-tolerant
    # client per call — argo-server runs secure mode with an in-cluster cert,
    # so a plain-HTTP or default-verifying client can never submit.
    client = http_client
    if dataset_store is None:
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        telemetry_manager = get_telemetry_manager()
        dataset_store = telemetry_manager.get_provider(tenant_id=tenant_id).datasets

    end_time = now
    start_time = now - timedelta(hours=triggers.annotation_lookback_hours)
    last_optimization = dict(state.get("last_optimization_at") or {})
    per_agent: dict = {}
    thresholds_refresh_submitted = False

    # Per-agent isolation: one agent's fault (Phoenix blip, dataset write)
    # must not abort the agents after it. The finally persists the poll stamp
    # and any cooldowns recorded before a fault — without it, the next cron
    # tick re-submits workflows agents already got this cycle.
    try:
        for agent_type, mode in AGENT_COMPILE_MODE.items():
            try:
                storage = annotation_storage_mod.AnnotationStorage(
                    tenant_id=tenant_id, agent_type=agent_type
                )
                rows = await storage.query_annotated_spans(
                    start_time=start_time,
                    end_time=end_time,
                    only_human_reviewed=True,
                )
                count = len(rows)
                outcome = {"annotations": count, "action": "none"}
                per_agent[agent_type] = outcome
                if count == 0:
                    continue

                recompile = count >= triggers.min_annotations_for_optimization
                refresh = (
                    mode == "gateway-thresholds"
                    and count >= feedback.min_annotations_for_update
                )
                if not recompile and not refresh:
                    continue

                last = last_optimization.get(agent_type)
                if last is not None:
                    since = now - datetime.fromisoformat(last)
                    if since < timedelta(days=triggers.min_days_between_optimizations):
                        outcome["action"] = "cooldown"
                        continue

                if mode == "gateway-thresholds" and thresholds_refresh_submitted:
                    # gateway + routing share one threshold recalibration.
                    outcome["action"] = "thresholds_refresh"
                    last_optimization[agent_type] = now.isoformat()
                    continue

                parameters = [{"name": "tenant-id", "value": tenant_id}]
                container_args = [
                    "--mode",
                    mode,
                    "--tenant-id",
                    "{{workflow.parameters.tenant-id}}",
                    "--lookback-hours",
                    str(float(triggers.annotation_lookback_hours)),
                ]

                if mode == "triggered":
                    records = []
                    for row in rows:
                        label = row.get("annotation_label")
                        score = quality_map.get(label)
                        if score is None:
                            logger.warning(
                                "Unmapped annotation label %r on span %s — skipped",
                                label,
                                row.get("span_id"),
                            )
                            continue
                        records.append(
                            {
                                "agent": agent_type,
                                "category": (
                                    "high_scoring" if score >= 0.8 else "low_scoring"
                                ),
                                "query": row.get("query") or "",
                                "score": score,
                                "output": json.dumps(
                                    row.get("output") or {}, default=str
                                ),
                            }
                        )
                    if not records:
                        outcome["action"] = "no_mapped_labels"
                        continue
                    dataset_name = (
                        f"optimization-trigger-{tenant_id}-"
                        f"{now.strftime('%Y%m%d_%H%M%S')}"
                    )
                    await dataset_store.create_dataset(
                        name=dataset_name,
                        data=pd.DataFrame(records),
                        metadata={
                            "description": (
                                f"Annotation-feedback trigger for {tenant_id}: "
                                f"{agent_type}"
                            ),
                            "input_keys": ["agent", "category", "query"],
                            "output_keys": ["score", "output"],
                        },
                    )
                    parameters.append({"name": "agents", "value": agent_type})
                    parameters.append(
                        {"name": "trigger-dataset", "value": dataset_name}
                    )
                    container_args = [
                        "--mode",
                        "triggered",
                        "--tenant-id",
                        "{{workflow.parameters.tenant-id}}",
                        "--agents",
                        "{{workflow.parameters.agents}}",
                        "--trigger-dataset",
                        "{{workflow.parameters.trigger-dataset}}",
                    ]

                submitted = await submit_argo_optimization_workflow(
                    http_client=client,
                    argo_api_url=argo_url,
                    argo_namespace=argo_namespace,
                    tenant_id=tenant_id,
                    name_prefix=(
                        f"annotation-feedback-{agent_type.replace('_', '-')}-"
                        f"{now.strftime('%Y%m%d-%H%M%S')}"
                    ),
                    trigger_label="annotation-feedback",
                    parameters=parameters,
                    container_args=container_args,
                    pod_spec=pod_spec,
                )
                if submitted:
                    outcome["action"] = (
                        "recompile" if recompile else "thresholds_refresh"
                    )
                    last_optimization[agent_type] = now.isoformat()
                    if mode == "gateway-thresholds":
                        thresholds_refresh_submitted = True
                else:
                    outcome["action"] = "submit_failed"
            except Exception as e:
                logger.error(
                    "Annotation feedback step failed for %s: %r", agent_type, e
                )
                per_agent.setdefault(agent_type, {"annotations": 0})["action"] = "error"
    finally:
        state["last_feedback_run_at"] = now.isoformat()
        state["last_optimization_at"] = last_optimization
        _save_loop_state(config_manager, tenant_id, state)

    result = {"status": "success", "agents": per_agent}
    logger.info("Annotation feedback cycle complete: %s", result)
    return result


def _build_phoenix_provider(tenant_id: str, http_endpoint: str) -> Optional[object]:
    """Construct a PhoenixProvider for the QualityMonitor's XGBoost gate.

    The XGBoost training-decision block in
    ``QualityMonitor._apply_training_decision_model`` is gated on
    ``self._telemetry_provider is not None``; this helper provides it.
    Constructs a provider from the HTTP endpoint and an env-var-overridable
    gRPC endpoint, then initializes it. On failure it logs a warning and
    returns ``None`` so the monitor degrades to naive verdicts rather
    than crashing the sidecar.
    """
    grpc_endpoint = os.environ.get("PHOENIX_GRPC_ENDPOINT")
    if not grpc_endpoint:
        # Default: same host as http_endpoint, OTLP gRPC port 4317.
        try:
            parsed = urlparse(http_endpoint)
            host = parsed.hostname or "localhost"
            grpc_endpoint = f"{host}:4317"
        except Exception:
            grpc_endpoint = "localhost:4317"

    try:
        from cogniverse_telemetry_phoenix.provider import PhoenixProvider

        provider = PhoenixProvider()
        provider.initialize(
            {
                "tenant_id": tenant_id,
                "http_endpoint": http_endpoint,
                "grpc_endpoint": grpc_endpoint,
            }
        )
        logger.info(
            f"PhoenixProvider initialized for QualityMonitor "
            f"(tenant={tenant_id}, http={http_endpoint}, grpc={grpc_endpoint})"
        )
        return provider
    except Exception as exc:
        logger.warning(
            f"Failed to build PhoenixProvider for QualityMonitor: {exc}. "
            "XGBoost gate will be skipped — naive verdicts only."
        )
        return None


def main():
    parser = argparse.ArgumentParser(description="Cogniverse Quality Monitor")
    parser.add_argument("--tenant-id", required=True, help="Tenant ID to monitor")
    parser.add_argument(
        "--runtime-url",
        default="http://localhost:28000",
        help="Runtime API URL for running golden queries",
    )
    parser.add_argument(
        "--phoenix-url",
        default="http://localhost:6006",
        help="Phoenix HTTP endpoint for span queries and dataset storage",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://localhost:11434",
        help="LLM API base URL for LLM judge",
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help=(
            "LLM model name for LLM judge evaluations. Must match "
            "evaluators.llm_judge.model in config — pass the bare model id."
        ),
    )
    parser.add_argument(
        "--golden-dataset-path",
        default="data/testset/evaluation/sample_videos_retrieval_queries.json",
        help="Path to golden evaluation dataset JSON",
    )
    parser.add_argument(
        "--argo-url",
        default=None,
        help="Argo server API URL (omit to disable auto-submission)",
    )
    parser.add_argument(
        "--argo-namespace",
        default="cogniverse",
        help="Kubernetes namespace for Argo workflows",
    )
    parser.add_argument(
        "--golden-interval",
        type=int,
        default=7200,
        help="Seconds between golden set evaluations (default: 2h)",
    )
    parser.add_argument(
        "--live-interval",
        type=int,
        default=14400,
        help="Seconds between live traffic evaluations (default: 4h)",
    )
    parser.add_argument(
        "--live-sample-count",
        type=int,
        default=20,
        help="Number of spans to sample per agent for live eval",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run a single forced optimization cycle and exit. Used by Argo "
            "CronWorkflows for scheduled distillation. Bypasses the threshold "
            "check so distillation runs even when quality is stable."
        ),
    )
    parser.add_argument(
        "--annotation-cycle",
        action="store_true",
        help=(
            "Run a single annotation-identification cycle (flag spans needing "
            "human review and enqueue them on the runtime worklist) and exit."
        ),
    )
    parser.add_argument(
        "--annotation-feedback",
        action="store_true",
        help=(
            "Run a single annotation-feedback cycle (count accumulated human "
            "annotations per agent and submit compile workflows when the "
            "volume gates and cooldowns allow) and exit."
        ),
    )
    args = parser.parse_args()
    # One canonical tenant everywhere — the runtime keys span projects and
    # artifacts by the canonical form, so a bare chart value like "default"
    # must not fork a parallel tenant world.
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    args.tenant_id = canonical_tenant_id(args.tenant_id)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Readers built via the telemetry manager (AnnotationStorage,
    # AnnotationAgent, dataset store) derive Phoenix's HTTP endpoint from
    # TELEMETRY_OTLP_ENDPOINT's host on the fixed :6006 port; point them at
    # --phoenix-url before any provider is constructed and cached.
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    get_telemetry_manager().config.provider_config["http_endpoint"] = args.phoenix_url

    from cogniverse_evaluation.quality_monitor import QualityMonitor

    # Inject a PhoenixProvider so the XGBoost training-decision gating
    # block in QualityMonitor.check_thresholds is reachable. The grpc
    # endpoint defaults to port 4317 on the same host as the HTTP endpoint,
    # matching the standard Phoenix deployment in the Helm chart.
    telemetry_provider = _build_phoenix_provider(
        tenant_id=args.tenant_id,
        http_endpoint=args.phoenix_url,
    )

    workflow_pod_spec = _workflow_pod_spec_from_env()
    monitor = QualityMonitor(
        tenant_id=args.tenant_id,
        runtime_url=args.runtime_url,
        phoenix_http_endpoint=args.phoenix_url,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        golden_dataset_path=args.golden_dataset_path,
        argo_api_url=args.argo_url,
        argo_namespace=args.argo_namespace,
        golden_eval_interval_seconds=args.golden_interval,
        live_eval_interval_seconds=args.live_interval,
        live_sample_count=args.live_sample_count,
        telemetry_provider=telemetry_provider,
        workflow_pod_spec=workflow_pod_spec,
    )

    if args.annotation_cycle:
        # One-shot annotation-identification pass for Argo CronWorkflows:
        # flag spans needing human review and feed the runtime worklist.
        logger.info(f"Running annotation cycle for tenant={args.tenant_id}")
        result = asyncio.run(
            run_annotation_cycle(tenant_id=args.tenant_id, runtime_url=args.runtime_url)
        )
        logger.info(f"Annotation cycle result: {result}")
        sys.exit(0)

    if args.annotation_feedback:
        # One-shot feedback pass for Argo CronWorkflows: accumulated human
        # annotations become compile submissions when gates + cooldowns allow.
        if not args.argo_url:
            logger.error("--annotation-feedback requires --argo-url")
            sys.exit(2)
        logger.info(f"Running annotation feedback cycle for tenant={args.tenant_id}")
        result = asyncio.run(
            run_annotation_feedback_cycle(
                tenant_id=args.tenant_id,
                argo_url=args.argo_url,
                argo_namespace=args.argo_namespace,
                pod_spec=workflow_pod_spec,
            )
        )
        logger.info(f"Annotation feedback result: {result}")
        sys.exit(0)

    if args.once:
        # One-shot scheduled distillation for Argo CronWorkflows: force-build
        # a trigger from the current eval and submit it regardless of
        # thresholds, then exit cleanly so the CronWorkflow run completes.
        # close() must run in the SAME loop as the cycle — the monitor's
        # async clients are bound to it, and a second asyncio.run() raises
        # "Event loop is closed" after a fully successful cycle.
        logger.info(f"Running forced optimization cycle for tenant={args.tenant_id}")

        async def _cycle_and_close():
            try:
                return await monitor.force_optimization_cycle()
            finally:
                await monitor.close()

        result = asyncio.run(_cycle_and_close())
        logger.info(f"Forced cycle result: {result}")
        sys.exit(0 if result.get("status") == "ok" else 1)

    logger.info(
        f"Starting quality monitor for tenant={args.tenant_id} "
        f"(golden every {args.golden_interval}s, live every {args.live_interval}s)"
    )

    async def _run_and_close():
        # The annotation-identification loop rides in the sidecar next to the
        # quality-drop loop: detection belongs in the long-lived service, and
        # the runtime's in-memory worklist needs a feeder in the same pod.
        annotation_task = asyncio.create_task(
            _annotation_loop(args.tenant_id, args.runtime_url)
        )
        try:
            await monitor.run()
        finally:
            annotation_task.cancel()
            await monitor.close()

    try:
        asyncio.run(_run_and_close())
    except KeyboardInterrupt:
        logger.info("Quality monitor stopped by user")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
