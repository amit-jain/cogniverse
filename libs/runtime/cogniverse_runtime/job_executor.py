"""Scheduled job executor for tenant agent jobs.

This module is the entrypoint that Argo CronWorkflows invoke. It:
1. Reads job config from ConfigStore (query, post_actions)
2. Calls orchestrator_agent with the query to get a result
3. For each post_action, routes through orchestrator_agent to process the
   action, then delivers the result to any detected destination (wiki,
   Telegram)

Post_action examples:
  - "save to wiki" → save main result to wiki
  - "send me a summary on Telegram" → orchestrator summarizes → send to Telegram
  - "create a detailed report" → orchestrator creates report (no delivery)
  - "summarize and save to wiki" → orchestrator summarizes → save to wiki

CLI usage:
    python -m cogniverse_runtime.job_executor \
        --job-id <id> \
        --tenant-id <tenant_id> \
        --runtime-url http://localhost:28000
"""

import argparse
import asyncio
import logging
import math
import os
import sys

import httpx

logger = logging.getLogger(__name__)

_DELIVERY_DESCRIPTIONS = {
    "wiki": "save store add persist write to wiki knowledge base",
    "telegram": "send notify message telegram chat alert me",
}

_DELIVERY_THRESHOLD = 0.5
_delivery_embeddings: dict = {}


def _embed_text(text: str, ollama_url: str) -> list:
    """Get embedding from Ollama for semantic classification."""
    resp = httpx.post(
        f"{ollama_url}/api/embed",
        json={"model": "nomic-embed-text", "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _ensure_delivery_embeddings(ollama_url: str) -> None:
    """Lazily compute embeddings for known delivery destinations."""
    if _delivery_embeddings:
        return
    for dest, desc in _DELIVERY_DESCRIPTIONS.items():
        _delivery_embeddings[dest] = _embed_text(desc, ollama_url)
    logger.info(
        "Computed delivery embeddings for %d destinations", len(_delivery_embeddings)
    )


def _detect_deliveries(action: str, ollama_url: str) -> list:
    """Detect delivery destinations in a post_action via semantic similarity.

    Returns a list of matched destinations (e.g. ["wiki"], ["telegram"],
    ["wiki", "telegram"], or [] for agent-only actions).
    """
    _ensure_delivery_embeddings(ollama_url)
    action_emb = _embed_text(action, ollama_url)

    matched = []
    for dest, ref_emb in _delivery_embeddings.items():
        sim = _cosine_sim(action_emb, ref_emb)
        if sim > _DELIVERY_THRESHOLD:
            matched.append((dest, sim))
            logger.info(
                "Detected delivery %r in action %r (sim=%.3f)", dest, action, sim
            )

    return [dest for dest, _ in sorted(matched, key=lambda x: -x[1])]


def _is_pure_delivery(action: str) -> bool:
    """Check if the action is purely a delivery instruction with no processing.

    "save to wiki" → pure delivery (just save the existing result)
    "summarize and save to wiki" → NOT pure (needs summarization first)
    """
    words = set(action.lower().split())
    processing_words = {
        "summarize",
        "analyze",
        "report",
        "search",
        "find",
        "research",
        "compare",
        "explain",
        "list",
        "describe",
        "create",
        "generate",
        "write",
        "draft",
        "compile",
    }
    return not words.intersection(processing_words)


async def _call_agent(
    client: httpx.AsyncClient,
    runtime_url: str,
    tenant_id: str,
    query: str,
    context: str = "",
) -> str:
    """POST to orchestrator_agent/process and return the response text."""
    payload: dict = {"query": query, "tenant_id": tenant_id}
    if context:
        payload["context"] = context

    url = f"{runtime_url}/agents/orchestrator_agent/process"
    try:
        response = await client.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return data.get("response") or data.get("result") or str(data)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Agent call failed (%s): %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        raise
    except Exception as exc:
        logger.error("Agent call error: %s", exc)
        raise


async def _deliver_to_wiki(
    client: httpx.AsyncClient,
    runtime_url: str,
    tenant_id: str,
    query: str,
    content: str,
) -> None:
    """Save content to wiki via POST /wiki/save."""
    payload = {
        "query": query,
        "response": {"answer": content},
        "entities": [],
        "agent_name": "job_executor",
        "tenant_id": tenant_id,
    }
    try:
        response = await client.post(
            f"{runtime_url}/wiki/save", json=payload, timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        logger.info("Delivered to wiki: slug=%s", data.get("slug"))
    except Exception as exc:
        logger.error("Wiki delivery failed: %s", exc)


async def _deliver_to_telegram(
    client: httpx.AsyncClient,
    runtime_url: str,
    tenant_id: str,
    content: str,
) -> None:
    """Send content via the messaging gateway."""
    payload = {"tenant_id": tenant_id, "message": content}
    try:
        response = await client.post(
            f"{runtime_url}/messaging/send", json=payload, timeout=30.0
        )
        if response.status_code == 404:
            logger.warning("Messaging endpoint not available — skipping Telegram")
            return
        response.raise_for_status()
        logger.info("Delivered to Telegram")
    except Exception as exc:
        logger.error("Telegram delivery failed: %s", exc)


async def _execute_action(
    client: httpx.AsyncClient,
    runtime_url: str,
    tenant_id: str,
    action: str,
    query: str,
    context: str,
    ollama_url: str,
) -> str:
    """Execute a post_action: process through agent if needed, then deliver.

    Pure delivery actions (e.g. "save to wiki") skip the agent call and
    deliver the existing context directly. Actions with processing intent
    (e.g. "summarize and send on Telegram") go through orchestrator_agent
    first.
    """
    deliveries = _detect_deliveries(action, ollama_url)

    if _is_pure_delivery(action) and deliveries:
        result = context
    else:
        result = await _call_agent(
            client, runtime_url, tenant_id, query=action, context=context
        )

    for dest in deliveries:
        if dest == "wiki":
            await _deliver_to_wiki(client, runtime_url, tenant_id, query, result)
        elif dest == "telegram":
            await _deliver_to_telegram(client, runtime_url, tenant_id, result)

    return result


async def run_job(job_id: str, tenant_id: str, runtime_url: str) -> None:
    """Execute the job: run query then process each post_action."""
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
        raise RuntimeError(
            f"Job {job_id} not found in ConfigStore for tenant {tenant_id}"
        )

    job = entry.config_value
    query: str = job["query"]
    post_actions: list = job.get("post_actions", [])

    ollama_url = os.environ.get("LLM_ENDPOINT", "http://localhost:11434")

    logger.info("Starting job %s for tenant %s — query=%r", job_id, tenant_id, query)

    async with httpx.AsyncClient() as client:
        result = await _call_agent(client, runtime_url, tenant_id, query)
        logger.info("Job %s main query result: %s", job_id, result[:200])

        for action in post_actions:
            logger.info("Job %s executing post_action: %r", job_id, action)
            result = await _execute_action(
                client,
                runtime_url,
                tenant_id,
                action,
                query,
                result,
                ollama_url,
            )

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
