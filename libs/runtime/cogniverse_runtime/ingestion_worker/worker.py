"""Ingestion worker — claim jobs from Redis, run the pipeline, ack.

Runs as a long-lived process in the ingestor pod (``python -m
cogniverse_runtime.ingestion_worker.worker``). Each worker joins the
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
import json
import logging
import os
import signal
import socket
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis

from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.queue import IngestJob
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

logger = logging.getLogger(__name__)


class IngestPipelineError(RuntimeError):
    """A pipeline envelope reported ``status='failed'`` / ``'cancelled'``.

    ``process_video_async`` returns a failed-status dict rather than raising;
    raising routes it through the worker's failure path (``state='failed'``,
    no ``mark_done``).
    """


def _raise_if_pipeline_failed(result: object) -> None:
    """Raise on a ``failed``/``cancelled`` status envelope. Statusless dicts
    (injectable test processors) are treated as success."""
    if isinstance(result, dict):
        status = result.get("status")
        if status in ("failed", "cancelled"):
            error = result.get("error") or f"pipeline reported status={status!r}"
            raise IngestPipelineError(str(error))


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
        return MediaConfig()

    # fsspec's s3 client picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    # from the process env. Mirror our MINIO_* secrets onto those names
    # for the duration of the job so the localize() call authenticates.
    access = os.environ.get("MINIO_ACCESS_KEY")
    secret = os.environ.get("MINIO_SECRET_KEY")
    if access:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access)
    if secret:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)

    return MediaConfig.for_object_store(minio_endpoint)


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

    def _factory(tenant_id: str, deploy: bool = True) -> GraphManager:
        # ``deploy`` MUST be False on read-only paths: deploy_schema
        # triggers a Vespa redeploy that can drop another process's
        # just-fed rows mid-read. Read-built managers are not cached so
        # the first writer still deploys.
        tenant_id = canonical_tenant_id(tenant_id)
        if tenant_id in _graph_managers:
            return _graph_managers[tenant_id]
        if deploy:
            try:
                graph_backend.schema_registry.deploy_schema(
                    tenant_id=tenant_id, base_schema_name="knowledge_graph"
                )
            except Exception as exc:  # noqa: BLE001 — log + degrade
                # The common case is "schema already deployed"; the deploy
                # call is idempotent at the Vespa convergence layer but the
                # client wrapper can raise on transient transport errors
                # or genuine schema validation failures. Log so a real
                # failure is visible — first feed/query attempt will then
                # surface the actual blocking error to the caller.
                logger.warning(
                    "Knowledge-graph schema deploy for tenant %s raised "
                    "(treating as already-deployed; real error surfaces on "
                    "first feed/query): %s",
                    tenant_id,
                    exc,
                )
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
        if deploy:
            _graph_managers[tenant_id] = mgr
        return mgr

    graph_router.set_graph_manager_factory(_factory)
    _configure_dspy_lm(config_manager)
    _GRAPH_FACTORY_INSTALLED = True


def _resolve_worker_llm_config(config_manager):
    """Resolve the worker-wide default ``llm_config.primary`` endpoint.

    Consults the system tenant's config (the worker default LM is
    process-wide, not per-tenant — per-tenant LMs are resolved at
    dispatch via ``routers.ingestion._resolve_tenant_llm_config``).
    Returns ``None`` when the config store has no primary endpoint or
    is unreachable, so the caller can fall back to env.
    """
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig
    from cogniverse_foundation.config.utils import get_config

    try:
        cfg = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
        endpoint = cfg.get("llm_config", {}).get("primary")
    except Exception as exc:  # noqa: BLE001 — config store down ≠ worker down
        logger.warning(
            "Could not resolve llm_config.primary from the config store "
            "(falling back to LLM_ENDPOINT/LLM_MODEL env): %s",
            exc,
        )
        return None
    if not endpoint:
        return None
    return LLMEndpointConfig(**endpoint)


def _configure_dspy_lm(config_manager) -> None:
    """Configure the DSPy default LM so ClaimExtractor + the
    sufficient-context gate run without an explicit dspy.configure.

    The runtime's main.py does this on startup but the worker has its
    own process. Resolves ``llm_config.primary`` from the config store
    first (so retries/timeout/seed/extra_headers reach the LM); falls
    back to ``LLM_ENDPOINT`` + ``LLM_MODEL`` env (the same env vars the
    runtime pod uses) when the store has no primary endpoint. Either
    way the LM is built via ``create_dspy_lm`` — the mandatory
    chokepoint for every dspy.LM construction. Idempotent —
    dspy.settings.lm is replaced each call but the side effect is the
    same.
    """
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig
    from cogniverse_foundation.dspy.model_format import ensure_provider_prefix

    llm_config = _resolve_worker_llm_config(config_manager)
    if llm_config is None:
        endpoint = os.environ.get("LLM_ENDPOINT")
        model = os.environ.get("LLM_MODEL")
        if not endpoint or not model:
            logger.warning(
                "No llm_config.primary in the config store and "
                "LLM_ENDPOINT / LLM_MODEL env not set — DSPy will be "
                "unconfigured and ClaimExtractor calls will raise "
                "'No LM is loaded'."
            )
            return
        llm_config = LLMEndpointConfig(
            model=ensure_provider_prefix(model),
            api_base=endpoint.rstrip("/"),
            temperature=0.0,
        )
    lm = create_dspy_lm(llm_config)
    dspy.configure(lm=lm)
    logger.info(
        "DSPy LM configured for worker: model=%s api_base=%s",
        llm_config.model,
        llm_config.api_base,
    )


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

    # main.py bridges INFERENCE_SERVICE_URLS env → SystemConfig at API-
    # startup and persists to Vespa. The worker is a separate pod whose
    # SystemConfig read can race ahead of that write (or hit a Vespa
    # instance where main.py hasn't run since the deployment was
    # changed). Mirror the env bridge in memory so the pipeline's
    # ``service_urls`` lookup sees the same dict an API-side dispatch
    # would. Local-only; no Vespa persist (main.py remains
    # authoritative).
    _service_urls_env = os.environ.get("INFERENCE_SERVICE_URLS", "")
    if _service_urls_env:
        try:
            _parsed = json.loads(_service_urls_env)
            if isinstance(_parsed, dict):
                _sys_cfg = config_manager.get_system_config()
                if _sys_cfg.inference_service_urls != _parsed:
                    _sys_cfg.inference_service_urls = _parsed
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "INFERENCE_SERVICE_URLS env is not valid JSON: %s",
                _service_urls_env[:200],
            )
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
    # Process the already-localized file, but record job.source_url (s3://…) as
    # the canonical source_url on every indexed document — answer-time keyframe
    # resolution derives the object-store bucket from it. Passing source_uri
    # keeps the worker's own object-store-configured localize (line above) as
    # the single download, without depending on the pipeline's locator config.
    pipeline_envelope = await pipeline.process_video_async(
        Path(local_path), source_uri=job.source_url
    )

    # process_video_async wraps the strategy outputs under
    # envelope["results"] (alongside top-level status/error/timing
    # fields). Unwrap that nested dict before passing to the graph
    # extractor — _iter_segments_for_graph reads keyframes/transcript/
    # descriptions from the top level of whatever dict it receives.
    if isinstance(pipeline_envelope, dict) and "results" in pipeline_envelope:
        processing_results = dict(pipeline_envelope.get("results") or {})
        processing_results.setdefault("video_id", pipeline_envelope.get("video_id"))
    else:
        processing_results = pipeline_envelope or {}

    # Tag schema_name + video_id so _write_backrefs_to_content can
    # derive (schema, doc_id) per segment without needing a top-level
    # fed_documents list. The schema name follows the convention
    # <profile>_<tenant_sanitised> applied by the pipeline's Vespa
    # client.
    safe_tenant = job.tenant_id.replace(":", "_")
    processing_results.setdefault("__schema_name__", f"{job.profile}_{safe_tenant}")
    processing_results.setdefault("__video_id__", processing_results.get("video_id"))

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
            config_manager=config_manager,
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

    # Re-attach graph counts onto the original envelope the caller's
    # _summarise reads from (so /ingestion/{id}/status surfaces them).
    if isinstance(pipeline_envelope, dict):
        pipeline_envelope["graph_nodes"] = processing_results.get("graph_nodes", 0)
        pipeline_envelope["graph_edges"] = processing_results.get("graph_edges", 0)
        return pipeline_envelope
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
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    tm = get_telemetry_manager()

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
    # Outer span wraps the full job lifecycle (processor + cleanup +
    # status publish). component=pipeline so the TelemetryLevel filter
    # admits at DETAILED+. Errors propagate into the span via the
    # contextmanager's try/yield/except path.
    with tm.span(
        "pipeline.worker.process_job",
        tenant_id=job.tenant_id,
        component="pipeline",
        attributes={
            "job.id": job.ingest_id,
            "job.source_url": getattr(job, "source_url", "") or "",
            "job.profile": getattr(job, "profile", "") or "",
            "job.consumer_id": config.consumer_id,
        },
    ) as job_span:
        try:
            result = await processor(job)
            _raise_if_pipeline_failed(result)
            success = True
            terminal_event = {
                "state": "complete",
                "ingest_id": job.ingest_id,
                "result": _summarise(result),
            }
            job_span.set_attribute("job.outcome", "success")
        except Exception as exc:
            logger.exception("Ingest job %s failed", job.ingest_id)
            terminal_event = {
                "state": "failed",
                "ingest_id": job.ingest_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            job_span.set_attribute("job.outcome", "failed")
            job_span.set_attribute("job.error_type", type(exc).__name__)

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
    # ``schema_name``/``tenant_id`` are not in the pipeline envelope (callers
    # merge them from the job context); ``duration`` is the actual key
    # (the previous ``duration_seconds`` read was always missing).
    out = {
        k: pipeline_result.get(k)
        for k in ("video_id", "duration", "source_url")
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
    # Carry the worker's per-segment KG counts to the terminal event so the
    # wait=true route surfaces the real graph size instead of re-extracting.
    for k in ("graph_nodes", "graph_edges"):
        if k in pipeline_result:
            out[k] = pipeline_result.get(k, 0)
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
