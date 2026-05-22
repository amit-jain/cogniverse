"""Ingestion worker — claim jobs from Redis, run the pipeline, ack.

Runs as a long-lived process in the ingestor pod (``python -m
cogniverse_runtime.ingestion_v2.worker``). Each worker joins the
configured consumer group; Redis Streams + consumer groups guarantee
exclusive delivery so adding replicas just scales horizontally.

For each claimed job:
  1. Resolve the source URL via ``MediaLocator`` (s3://, http://, file://).
  2. Construct ``VideoIngestionPipeline`` against the right schema.
  3. Run ``pipeline.process_video_async``, publishing progress events
     to ``ingest:status:<ingest_id>`` as they arrive.
  4. On terminal: mark done in idempotency, decrement active counter,
     XACK the queue message.

Errors from any step land as a ``failed`` event with the exception
message; the message is still ACKed so it doesn't get redelivered.
PEL-based retry of stuck-in-flight messages is a separate reaper
job (out of scope here).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis

from cogniverse_runtime.ingestion_v2 import idempotency, queue
from cogniverse_runtime.ingestion_v2.queue import IngestJob
from cogniverse_runtime.ingestion_v2.redis_client import close_redis, get_redis

logger = logging.getLogger(__name__)


class WorkerConfig:
    """All env-driven knobs in one place. Read once at startup."""

    def __init__(self) -> None:
        self.redis_url = os.environ.get("REDIS_URL")
        if not self.redis_url:
            raise RuntimeError("REDIS_URL must be set for the ingestion worker")
        self.consumer_group = os.environ.get("INGEST_CONSUMER_GROUP", "ingestors")
        self.consumer_id = os.environ.get(
            "INGEST_CONSUMER_ID", f"{socket.gethostname()}-{os.getpid()}"
        )
        self.idempotency_ttl = int(
            os.environ.get("INGEST_IDEMPOTENCY_TTL_SECONDS", "604800")
        )
        self.claim_block_ms = int(os.environ.get("INGEST_CLAIM_BLOCK_MS", "5000"))


def _media_config_from_env() -> "object":
    """Build a MediaConfig that points fsspec at MinIO when the worker
    runs alongside our object store. Reading the MINIO_* env once at
    job-processing time keeps MediaConfig env-agnostic and avoids
    mutating any global config singleton."""
    from cogniverse_core.common.media import MediaConfig

    minio_endpoint = os.environ.get("MINIO_ENDPOINT")
    if not minio_endpoint:
        return MediaConfig.from_dict({})

    # fsspec's s3 client picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    # from the process env. Mirror our MINIO_* secrets onto those names
    # for the duration of the job so the localize() call authenticates.
    access = os.environ.get("MINIO_ACCESS_KEY")
    secret = os.environ.get("MINIO_SECRET_KEY")
    if access:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access)
    if secret:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)

    return MediaConfig.from_dict(
        {"backends": {"s3": {"endpoint_url": minio_endpoint, "region": "us-east-1"}}}
    )


_GRAPH_FACTORY_INSTALLED = False


def _ensure_graph_manager_factory(config_manager, schema_loader) -> None:
    """Install the per-tenant GraphManager factory on the graph router
    so ``_extract_graph_per_segment`` can reach it during ingest.

    Idempotent — first call sets the factory, subsequent calls return
    immediately. Mirrors the per-tenant lazy-deploy pattern in
    ``main.py``: each tenant gets its own ``knowledge_graph_<tenant>``
    Vespa schema deployed on first access.
    """
    global _GRAPH_FACTORY_INSTALLED
    if _GRAPH_FACTORY_INSTALLED:
        return

    from cogniverse_agents.graph.graph_manager import GraphManager
    from cogniverse_core.common.tenant_utils import (
        SYSTEM_TENANT_ID,
        canonical_tenant_id,
    )
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_foundation.config.bootstrap import BootstrapConfig
    from cogniverse_runtime.routers import graph as graph_router

    bootstrap = BootstrapConfig.from_environment()
    graph_backend = BackendRegistry.get_instance().get_ingestion_backend(
        name=bootstrap.backend_type,
        tenant_id=SYSTEM_TENANT_ID,
        config={
            "backend": {
                "url": bootstrap.backend_url,
                "port": bootstrap.backend_port,
            }
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    _graph_managers: dict = {}

    def _factory(tenant_id: str) -> GraphManager:
        tenant_id = canonical_tenant_id(tenant_id)
        if tenant_id in _graph_managers:
            return _graph_managers[tenant_id]
        try:
            graph_backend.schema_registry.deploy_schema(
                tenant_id=tenant_id, base_schema_name="knowledge_graph"
            )
        except Exception:
            # Schema may already be deployed; first feed/query attempt
            # surfaces the real error.
            pass
        sys_cfg = config_manager.get_system_config()
        colbert_url = sys_cfg.inference_service_urls.get("colbert_pylate")
        if not colbert_url:
            raise RuntimeError(
                "knowledge_graph requires colbert_pylate in "
                "INFERENCE_SERVICE_URLS. Available: "
                f"{sorted(sys_cfg.inference_service_urls)}"
            )
        mgr = GraphManager(
            backend=graph_backend,
            tenant_id=tenant_id,
            schema_name=graph_backend.get_tenant_schema_name(
                tenant_id, "knowledge_graph"
            ),
            colbert_endpoint_url=colbert_url,
        )
        _graph_managers[tenant_id] = mgr
        return mgr

    graph_router.set_graph_manager_factory(_factory)
    _GRAPH_FACTORY_INSTALLED = True


async def _default_processor(job: IngestJob) -> dict:
    """Production processor: localise the source via MediaLocator, run
    the VideoIngestionPipeline, then run the per-segment KG extraction
    + back-ref PATCH so the graph state lands alongside the content
    documents. Returns the pipeline's result dict augmented with the
    graph counts; the worker passes it to ``_summarise`` for the
    status event payload.
    """
    from cogniverse_core.common.media import MediaLocator
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
    from cogniverse_runtime.routers.ingestion import _extract_graph_per_segment

    media_config = _media_config_from_env()
    locator = MediaLocator(tenant_id=job.tenant_id, config=media_config)
    local_path = await asyncio.to_thread(locator.localize, job.source_url)

    config_manager = create_default_config_manager()
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    schema_loader = FilesystemSchemaLoader(schemas_dir)

    # Install the per-tenant GraphManager factory before the pipeline
    # runs so the downstream graph-extraction call can reach it. The
    # factory mirrors the one main.py installs for the API runtime,
    # so the worker behaves identically.
    _ensure_graph_manager_factory(config_manager, schema_loader)

    pipeline = VideoIngestionPipeline(
        tenant_id=job.tenant_id,
        config_manager=config_manager,
        schema_loader=schema_loader,
        schema_name=job.profile,
    )
    processing_results = await pipeline.process_video_async(Path(local_path))

    # Run per-segment graph extraction + cross-modal linker + face
    # pipeline + back-ref PATCH on top of the content ingestion. Graph
    # path is fail-safe: any internal exception is logged + the content
    # ingestion's results are returned unchanged so an LM blip doesn't
    # fail the whole upload.
    source_doc_id = processing_results.get("video_id") or job.ingest_id
    try:
        graph_counts = await _extract_graph_per_segment(
            processing_results=processing_results,
            source_doc_id=source_doc_id,
            tenant_id=job.tenant_id,
        )
        processing_results["graph_nodes"] = graph_counts.get("nodes_upserted", 0)
        processing_results["graph_edges"] = graph_counts.get("edges_upserted", 0)
    except Exception as exc:  # noqa: BLE001 — log + degrade, never fail ingest
        import logging

        logging.getLogger(__name__).warning(
            "per-segment KG extraction failed for ingest=%s: %s — content "
            "ingestion already succeeded, returning without graph counts",
            job.ingest_id,
            exc,
        )
        processing_results["graph_nodes"] = 0
        processing_results["graph_edges"] = 0

    return processing_results


async def _process_job(
    redis: aioredis.Redis,
    job: IngestJob,
    config: WorkerConfig,
    processor=_default_processor,
) -> None:
    """Run one job end-to-end and publish events for every state
    change. Always ACKs the queue message — even on failure — so the
    PEL doesn't grow unbounded under repeated transient errors. Stuck
    jobs are surfaced through the per-tenant active counter and the
    SSE error event, not via Redis-level retry.

    ``processor`` is injectable for tests that don't need the full
    Vespa+ColPali stack — production uses ``_default_processor``.

    Event ordering matters: cleanup (clear_inflight, mark_done,
    decrement_active, ack) runs BEFORE the terminal event publishes,
    so when an SSE watcher observes ``state=complete|failed`` all
    invariants (active counter accurate, idempotency record settled,
    queue PEL drained) are guaranteed consistent.
    """
    await queue.publish_status(
        redis,
        job.ingest_id,
        {
            "state": "running",
            "ingest_id": job.ingest_id,
            "consumer_id": config.consumer_id,
        },
    )

    success = False
    terminal_event: dict
    try:
        result = await processor(job)
        success = True
        terminal_event = {
            "state": "complete",
            "ingest_id": job.ingest_id,
            "result": _summarise(result),
        }
    except Exception as exc:
        logger.exception("Ingest job %s failed", job.ingest_id)
        terminal_event = {
            "state": "failed",
            "ingest_id": job.ingest_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    try:
        if success:
            await idempotency.mark_done(
                redis, job.sha, job.ingest_id, ttl_seconds=config.idempotency_ttl
            )
        await idempotency.clear_inflight(redis, job.sha)
        await queue.decrement_active(redis, job.tenant_id)
        await queue.ack(redis, config.consumer_group, job.message_id)
    except Exception as cleanup_exc:
        # Cleanup failures are themselves observable. Don't let them
        # crash the worker loop — log, attach to the terminal event,
        # and continue. The next claim still fires.
        logger.exception(
            "Cleanup failed for %s; terminal event will still publish", job.ingest_id
        )
        terminal_event["cleanup_error"] = str(cleanup_exc)

    await queue.publish_status(redis, job.ingest_id, terminal_event)


def _summarise(pipeline_result: dict) -> dict:
    """Trim the pipeline result to a small JSON-serialisable payload
    suitable for the status stream. The full result lives in Vespa /
    Phoenix; the event stream is for progress UX, not data transfer."""
    if not isinstance(pipeline_result, dict):
        return {"raw_type": type(pipeline_result).__name__}
    out = {
        k: pipeline_result.get(k)
        for k in ("video_id", "schema_name", "tenant_id", "duration_seconds")
        if k in pipeline_result
    }
    results = pipeline_result.get("results", {})
    if isinstance(results, dict):
        keyframe_list = results.get("keyframes")
        if isinstance(keyframe_list, list):
            out["keyframes"] = len(keyframe_list)
        elif isinstance(keyframe_list, dict):
            out["keyframes"] = len(keyframe_list.get("keyframes", []))
        else:
            out["keyframes"] = 0

        embeddings = results.get("embeddings")
        if isinstance(embeddings, dict):
            out["documents_fed"] = embeddings.get("documents_fed", 0)
        else:
            out["documents_fed"] = 0

        chunks = out["documents_fed"] or out["keyframes"]
        if not chunks:
            chunks = sum(
                len(v.get("chunks", []))
                for v in results.values()
                if isinstance(v, dict)
            )
        out["chunks"] = chunks
    return out


async def _claim_loop(
    redis: aioredis.Redis,
    config: WorkerConfig,
    stop: asyncio.Event,
    processor=_default_processor,
) -> None:
    await queue.ensure_consumer_group(redis, config.consumer_group)
    while not stop.is_set():
        try:
            jobs = await queue.claim(
                redis,
                config.consumer_group,
                config.consumer_id,
                block_ms=config.claim_block_ms,
                count=1,
            )
        except Exception as exc:
            logger.exception("claim failed; backing off: %s", exc)
            await asyncio.sleep(2.0)
            continue

        for job in jobs:
            if stop.is_set():
                break
            await _process_job(redis, job, config, processor=processor)


async def run(
    stop: Optional[asyncio.Event] = None,
    processor=_default_processor,
) -> None:
    """Worker entry. Pass an ``asyncio.Event`` to drive shutdown from
    a test; production uses signal handlers below."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = WorkerConfig()
    if stop is None:
        stop = asyncio.Event()
        _install_signal_handlers(stop)

    redis = await get_redis(config.redis_url)
    logger.info(
        "Worker %s started: group=%s redis=%s",
        config.consumer_id,
        config.consumer_group,
        config.redis_url,
    )
    try:
        await _claim_loop(redis, config, stop, processor=processor)
    finally:
        logger.info("Worker %s stopping", config.consumer_id)
        await close_redis()


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows / some embedded loops don't support signal handlers.
            pass


if __name__ == "__main__":
    asyncio.run(run())
