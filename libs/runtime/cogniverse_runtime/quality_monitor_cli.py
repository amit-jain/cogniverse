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
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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
        default="qwen3:4b",
        help="LLM model name for LLM judge evaluations",
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

    if args.once:
        # One-shot scheduled distillation for Argo CronWorkflows: force-build
        # a trigger from the current eval and submit it regardless of
        # thresholds, then exit cleanly so the CronWorkflow run completes.
        logger.info(f"Running forced optimization cycle for tenant={args.tenant_id}")
        try:
            result = asyncio.run(monitor.force_optimization_cycle())
            logger.info(f"Forced cycle result: {result}")
            exit_code = 0 if result.get("status") == "ok" else 1
        finally:
            asyncio.run(monitor.close())
        sys.exit(exit_code)

    logger.info(
        f"Starting quality monitor for tenant={args.tenant_id} "
        f"(golden every {args.golden_interval}s, live every {args.live_interval}s)"
    )

    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        logger.info("Quality monitor stopped by user")
    finally:
        asyncio.run(monitor.close())
        sys.exit(0)


if __name__ == "__main__":
    main()
