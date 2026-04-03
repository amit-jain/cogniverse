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
import sys

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cogniverse Quality Monitor")
    parser.add_argument(
        "--tenant-id", default="default", help="Tenant ID to monitor"
    )
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from cogniverse_evaluation.quality_monitor import QualityMonitor

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
    )

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
