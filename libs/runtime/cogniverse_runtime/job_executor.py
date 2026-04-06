"""Scheduled job executor for tenant agent jobs.

This module is the entrypoint that Argo CronWorkflows invoke. It:
1. Reads job config from ConfigStore (query, post_actions)
2. Calls POST /agents/routing_agent/process with the query + tenant_id
3. For each post_action, calls routing_agent again with the action as query
   and the previous result as context

CLI usage:
    python -m cogniverse_runtime.job_executor \
        --job-id <id> \
        --tenant-id <tenant_id> \
        --runtime-url http://localhost:28000
"""

import argparse
import asyncio
import logging
import sys

import httpx

logger = logging.getLogger(__name__)


async def _call_agent(
    client: httpx.AsyncClient,
    runtime_url: str,
    tenant_id: str,
    query: str,
    context: str = "",
) -> str:
    """POST to routing_agent/process and return the response text."""
    payload: dict = {"query": query, "tenant_id": tenant_id}
    if context:
        payload["context"] = context

    url = f"{runtime_url}/agents/routing_agent/process"
    try:
        response = await client.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        # Accept either {"response": "..."} or {"result": "..."}
        return data.get("response") or data.get("result") or str(data)
    except httpx.HTTPStatusError as exc:
        logger.error("Agent call failed (%s): %s", exc.response.status_code, exc.response.text[:500])
        raise
    except Exception as exc:
        logger.error("Agent call error: %s", exc)
        raise


async def run_job(job_id: str, tenant_id: str, runtime_url: str) -> None:
    """Execute the job: run query then each post_action in sequence."""
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    cm = create_default_config_manager()
    entry = cm.store.get_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service="tenant_jobs",
        config_key=f"job_{job_id}",
    )
    if entry is None or not entry.config_value:
        raise RuntimeError(f"Job {job_id} not found in ConfigStore for tenant {tenant_id}")

    job = entry.config_value
    query: str = job["query"]
    post_actions: list = job.get("post_actions", [])

    logger.info("Starting job %s for tenant %s — query=%r", job_id, tenant_id, query)

    async with httpx.AsyncClient() as client:
        result = await _call_agent(client, runtime_url, tenant_id, query)
        logger.info("Job %s main query result: %s", job_id, result[:200])

        for action in post_actions:
            logger.info("Job %s running post_action: %r", job_id, action)
            result = await _call_agent(
                client, runtime_url, tenant_id, query=action, context=result
            )
            logger.info("Job %s post_action result: %s", job_id, result[:200])

    logger.info("Job %s completed", job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cogniverse job executor")
    parser.add_argument("--job-id", required=True, help="Job ID to execute")
    parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    parser.add_argument(
        "--runtime-url",
        default="http://localhost:28000",
        help="Cogniverse runtime base URL",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        asyncio.run(run_job(args.job_id, args.tenant_id, args.runtime_url))
    except Exception as exc:
        logger.error("Job execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
