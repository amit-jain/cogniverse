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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from cogniverse_evaluation.quality_monitor import QualityMonitor

    # Inject a PhoenixProvider so the XGBoost training-decision gating
    # block in QualityMonitor.check_thresholds is reachable. The grpc
    # endpoint defaults to port 4317 on the same host as the HTTP endpoint,
    # matching the standard Phoenix deployment in the Helm chart.
    telemetry_provider = _build_phoenix_provider(
        tenant_id=args.tenant_id,
        http_endpoint=args.phoenix_url,
    )

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
