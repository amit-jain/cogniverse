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

Errors before durable content exists land as a terminal ``failed`` event and
are ACKed. Once content has been fed, a graph-stage error lands as nonterminal
``retrying`` and remains in the PEL; ``reaper.py`` XAUTOCLAIMs it back to a
live consumer and re-drives the stable document ids until the graph completes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import signal
import socket
import threading
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis

from cogniverse_runtime.inference_services import parse_inference_service_urls
from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.queue import IngestJob
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

logger = logging.getLogger(__name__)

GRAPH_PENDING_KEY_PREFIX = "ingest:graph-pending:"


class JobDeadlineExceeded(Exception):
    """Pipeline exceeded the per-job wall-clock deadline.

    The claim heartbeat makes a hung job indistinguishable from a live one
    to the reaper, so this deadline is the only bound on "alive but stuck".
    """


class IngestPipelineError(RuntimeError):
    """A pipeline envelope reported ``status='failed'`` / ``'cancelled'``.

    ``process_video_async`` returns a failed-status dict rather than raising;
    raising routes it through the worker's failure path (``state='failed'``,
    no ``mark_done``).
    """


class GraphStageIncomplete(RuntimeError):
    """Content is durable but its graph transaction has not completed.

    This state is retryable and nonterminal: clearing the submit marker or
    acknowledging the stream entry would strand content without its graph.
    """


async def _mark_graph_pending(redis: aioredis.Redis, job: IngestJob) -> None:
    """Persist the graph-stage marker before entering its system boundary."""
    await redis.set(f"{GRAPH_PENDING_KEY_PREFIX}{job.message_id}", job.ingest_id)


async def _clear_graph_pending(redis: aioredis.Redis, message_id: str) -> None:
    await redis.delete(f"{GRAPH_PENDING_KEY_PREFIX}{message_id}")


async def _is_graph_pending(redis: aioredis.Redis, message_id: str) -> bool:
    return bool(await redis.exists(f"{GRAPH_PENDING_KEY_PREFIX}{message_id}"))


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
        self.inference_service_urls = parse_inference_service_urls(
            os.environ.get("INFERENCE_SERVICE_URLS")
        )
        self.consumer_group = os.environ.get("INGEST_CONSUMER_GROUP", "ingestors")
        self.consumer_id = os.environ.get(
            "INGEST_CONSUMER_ID", f"{socket.gethostname()}-{os.getpid()}"
        )
        self.idempotency_ttl = int(
            os.environ.get("INGEST_IDEMPOTENCY_TTL_SECONDS", "604800")
        )
        self.claim_block_ms = int(os.environ.get("INGEST_CLAIM_BLOCK_MS", "5000"))
        # Must be well below reaper_min_idle_ms: the heartbeat is what keeps
        # a live long-running job's PEL entry from looking abandoned.
        self.heartbeat_interval_s = int(
            os.environ.get("INGEST_HEARTBEAT_INTERVAL_SECONDS", "60")
        )
        self.reaper_enabled = os.environ.get(
            "INGEST_REAPER_ENABLED", "true"
        ).lower() in ("1", "true", "yes")
        self.reaper_interval_s = int(
            os.environ.get("INGEST_REAPER_INTERVAL_SECONDS", "60")
        )
        self.reaper_min_idle_ms = int(
            os.environ.get("INGEST_REAPER_MIN_IDLE_MS", "300000")
        )
        self.reaper_max_deliveries = int(
            os.environ.get("INGEST_REAPER_MAX_DELIVERIES", "5")
        )
        # Hard wall-clock cap per job. The claim heartbeat keeps a HUNG
        # pipeline's PEL entry fresh forever, so without this deadline a
        # processor stuck on a timeoutless await leaves the job "running"
        # and its tenant slot held until pod restart. Must exceed the
        # longest legitimate pipeline (long-video KG extraction runs ~40min
        # here); 0 disables.
        self.job_deadline_s = int(os.environ.get("INGEST_JOB_DEADLINE_SECONDS", "7200"))
        self.graph_deadline_s = float(
            os.environ.get("INGEST_GRAPH_DEADLINE_SECONDS", "1800")
        )
        if not math.isfinite(self.graph_deadline_s) or self.graph_deadline_s <= 0:
            raise RuntimeError(
                "INGEST_GRAPH_DEADLINE_SECONDS must be positive and finite"
            )


def _media_config_from_defaults(
    runtime_defaults: dict[str, str | int | None],
) -> "object":
    """Build a MediaConfig from the resolved entrypoint defaults."""
    from cogniverse_core.common.media import MediaConfig

    minio_endpoint = runtime_defaults["minio_endpoint"]
    if not minio_endpoint:
        return MediaConfig()

    # fsspec's s3 client picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    # from the process env. Mirror our MINIO_* secrets onto those names
    # for the duration of the job so the localize() call authenticates.
    access = runtime_defaults["minio_access_key"]
    secret = runtime_defaults["minio_secret_key"]
    if access:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access)
    if secret:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)

    return MediaConfig.for_object_store(minio_endpoint)


_GRAPH_FACTORY_INSTALLED = False


def _build_worker_graph_factory(graph_backend, config_manager):
    """Build the worker's per-tenant GraphManager factory.

    Same single-cold-build contract as ``main._build_graph_manager_factory``:
    ``get_or_set`` holds the cache lock across the build, so concurrent
    first-touches for a fresh tenant deploy + construct exactly once.
    """
    from cogniverse_agents.graph.graph_manager import GraphManager
    from cogniverse_core.common.tenant_utils import canonical_tenant_id
    from cogniverse_foundation.caching import TenantLRUCache

    _graph_managers: TenantLRUCache[GraphManager] = TenantLRUCache(capacity=64)

    def _factory(tenant_id: str, deploy: bool = True) -> GraphManager:
        # ``deploy`` MUST be False on read-only paths: deploy_schema
        # triggers a Vespa redeploy that can drop another process's
        # just-fed rows mid-read. Read-built managers are not cached so
        # the first writer still deploys.
        tenant_id = canonical_tenant_id(tenant_id)

        def _build() -> GraphManager:
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
            gliner_url = sys_cfg.inference_service_urls.get("gliner")
            if not colbert_url:
                raise RuntimeError(
                    "knowledge_graph requires colbert_pylate in "
                    "INFERENCE_SERVICE_URLS. Available: "
                    f"{sorted(sys_cfg.inference_service_urls)}"
                )
            if not gliner_url:
                raise RuntimeError(
                    "knowledge_graph requires gliner in INFERENCE_SERVICE_URLS. "
                    f"Available: {sorted(sys_cfg.inference_service_urls)}"
                )
            return GraphManager(
                backend=graph_backend,
                tenant_id=tenant_id,
                schema_name=graph_backend.get_tenant_schema_name(
                    tenant_id, "knowledge_graph"
                ),
                colbert_endpoint_url=colbert_url,
                gliner_inference_url=gliner_url,
            )

        if deploy:
            # get_or_set holds the cache lock across the build, so N
            # concurrent first-touches for a fresh tenant run deploy_schema
            # + the manager construction exactly once.
            return _graph_managers.get_or_set(tenant_id, _build)
        cached = _graph_managers.get(tenant_id)
        if cached is not None:
            return cached
        return _build()

    return _factory


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

    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
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

    graph_router.set_graph_manager_factory(
        _build_worker_graph_factory(graph_backend, config_manager)
    )
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


def _build_worker_lm(llm_config):
    """Build the worker-wide default LM (env fallback when None).

    Falls back to ``LLM_ENDPOINT`` + ``LLM_MODEL`` env (the same env vars
    the runtime pod uses) when the store has no primary endpoint. Either
    way the LM is built via ``create_dspy_lm`` — the mandatory chokepoint
    for every dspy.LM construction. Returns ``None`` when neither source
    names an endpoint, so the caller binds nothing at all rather than
    binding a null LM over whatever DSPy already has.
    """
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig
    from cogniverse_foundation.dspy.model_format import ensure_provider_prefix

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
            return None
        llm_config = LLMEndpointConfig(
            model=ensure_provider_prefix(model),
            api_base=endpoint.rstrip("/"),
            temperature=0.0,
        )
    lm = create_dspy_lm(llm_config)
    logger.info(
        "DSPy LM built for worker: model=%s api_base=%s",
        llm_config.model,
        llm_config.api_base,
    )
    return lm


_WORKER_LM = None
_WORKER_LM_RESOLVED = False
_WORKER_LM_LOCK = threading.Lock()


def _worker_dspy_lm(config_manager):
    """The worker-wide default LM, resolved from the config store once per
    process.

    Blocking (config-store read), so callers offload it; the lock keeps
    concurrent first-touch jobs to a single resolve + build instead of one
    per job.
    """
    global _WORKER_LM, _WORKER_LM_RESOLVED

    with _WORKER_LM_LOCK:
        if not _WORKER_LM_RESOLVED:
            _WORKER_LM = _build_worker_lm(_resolve_worker_llm_config(config_manager))
            _WORKER_LM_RESOLVED = True
        return _WORKER_LM


async def _ensure_worker_dspy_lm(config_manager):
    """The worker default LM for the caller to bind around its job."""
    return await asyncio.to_thread(_worker_dspy_lm, config_manager)


def _prepare_job_context(service_urls: dict[str, str] | None):
    """Build the config manager + schema loader and install the per-tenant
    GraphManager factory (mirroring the one main.py installs for the API
    runtime, so the worker behaves identically).

    All of this is cold-start heavy — Vespa-backed config reads plus a
    schema deploy with convergence waits — so ``_default_processor`` runs it
    via ``asyncio.to_thread``. Inline on the worker loop it would block the
    ``add_signal_handler`` SIGTERM callback, turning graceful shutdown into
    a SIGKILL.
    """
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()

    # main.py bridges INFERENCE_SERVICE_URLS env → SystemConfig at API-
    # startup and persists to Vespa. The worker is a separate pod whose
    # SystemConfig read can race ahead of that write (or hit a Vespa
    # instance where main.py hasn't run since the deployment was
    # changed). Use the same validated explicit override in memory;
    # absence leaves persisted discovery unchanged. Local-only; no
    # Vespa persist (main.py remains authoritative).
    if service_urls is not None:
        explicit_urls = dict(service_urls)
        system_config = config_manager.get_system_config()
        if system_config.inference_service_urls != explicit_urls:
            system_config.inference_service_urls = explicit_urls
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    schema_loader = FilesystemSchemaLoader(schemas_dir)

    _ensure_graph_manager_factory(config_manager, schema_loader)
    return config_manager, schema_loader


async def _default_processor(
    job: IngestJob,
    *,
    service_urls: dict[str, str] | None,
    mark_graph_pending: Callable[[IngestJob], Awaitable[None]],
    graph_deadline_s: float,
    media_config: "object" | None = None,
) -> dict:
    """Production processor: localise the source, bind the worker's default
    LM for the job, and run the pipeline + per-segment KG extraction under
    that binding.

    The LM is bound with ``dspy.context``, not ``dspy.configure``: the
    ambient binding belongs to whichever async task configured it first and
    raises for every other task, so a per-job binding is the only shape that
    works in a process where anything else has configured DSPy. The binding
    lives in a ContextVar, so it reaches the ``to_thread`` offloads and
    gathered subtasks the KG pass runs its ClaimExtractor calls in. With no
    endpoint resolvable, nothing is bound and DSPy keeps whatever it had.
    """
    import dspy

    from cogniverse_core.common.media import MediaConfig, MediaLocator

    if media_config is None:
        media_config = MediaConfig()
    locator = MediaLocator(tenant_id=job.tenant_id, config=media_config)
    local_path = await asyncio.to_thread(locator.localize, job.source_url)

    config_manager, schema_loader = await asyncio.to_thread(
        _prepare_job_context, service_urls
    )
    worker_lm = await _ensure_worker_dspy_lm(config_manager)
    binding = (
        dspy.context(lm=worker_lm)
        if worker_lm is not None
        else contextlib.nullcontext()
    )

    with binding:
        return await _ingest_and_extract_graph(
            job,
            local_path=local_path,
            config_manager=config_manager,
            schema_loader=schema_loader,
            mark_graph_pending=mark_graph_pending,
            graph_deadline_s=graph_deadline_s,
        )


async def _ingest_and_extract_graph(
    job: IngestJob,
    *,
    local_path,
    config_manager,
    schema_loader,
    mark_graph_pending: Callable[[IngestJob], Awaitable[None]],
    graph_deadline_s: float,
) -> dict:
    """Run the VideoIngestionPipeline over the localised file, then the
    per-segment KG extraction + back-ref PATCH so the graph state lands
    alongside the content documents. Returns the pipeline's result dict
    augmented with the graph counts; the worker passes it to ``_summarise``
    for the status event payload.

    Runs under the caller's DSPy LM binding — every LM-backed step below
    reads it, directly or from a thread the binding's ContextVar reaches.
    """
    from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
    from cogniverse_runtime.ingestion_worker import minio_client
    from cogniverse_runtime.routers.ingestion import _extract_graph_per_segment

    original_filename = await asyncio.to_thread(
        minio_client.get_original_filename, job.source_url
    )
    pipeline = VideoIngestionPipeline(
        tenant_id=job.tenant_id,
        config_manager=config_manager,
        schema_loader=schema_loader,
        schema_name=job.profile,
    )
    pipeline.original_filename = original_filename
    # Process the already-localized file, but record job.source_url (s3://…) as
    # the canonical source_url on every indexed document — answer-time keyframe
    # resolution derives the object-store bucket from it. Passing source_uri
    # keeps the caller's object-store-configured localize as the single
    # download, without depending on the pipeline's locator config.
    pipeline_envelope = await pipeline.process_video_async(
        Path(local_path), source_uri=job.source_url
    )
    # The pipeline may already have fed content. Commit the marker before any
    # transformation or graph work so every later crash is resumable.
    try:
        await mark_graph_pending(job)
    except asyncio.CancelledError as exc:
        raise GraphStageIncomplete(
            f"graph marker write was interrupted after content feed for ingest "
            f"{job.ingest_id}"
        ) from exc
    except Exception as exc:
        raise GraphStageIncomplete(
            f"graph marker write failed after content feed for ingest {job.ingest_id}"
        ) from exc

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

    # Stable document ids make graph and back-reference writes safe to retry.
    source_doc_id = processing_results.get("video_id") or job.ingest_id
    try:
        graph_counts = await asyncio.wait_for(
            _extract_graph_per_segment(
                processing_results=processing_results,
                source_doc_id=source_doc_id,
                tenant_id=job.tenant_id,
                config_manager=config_manager,
            ),
            timeout=graph_deadline_s,
        )
    except TimeoutError:
        raise GraphStageIncomplete(
            f"graph extraction exceeded the {graph_deadline_s:g}s deadline "
            f"for ingest {job.ingest_id}"
        ) from None
    except asyncio.CancelledError as exc:
        raise GraphStageIncomplete(
            f"graph extraction was interrupted for ingest {job.ingest_id}"
        ) from exc
    except Exception as exc:
        raise GraphStageIncomplete(
            f"graph extraction failed for ingest {job.ingest_id}"
        ) from exc

    processing_results["graph_nodes"] = graph_counts.get("nodes_upserted", 0)
    processing_results["graph_edges"] = graph_counts.get("edges_upserted", 0)
    processing_results["graph_failed"] = graph_counts.get("graph_failed", 0)
    if processing_results["graph_failed"]:
        raise GraphStageIncomplete(
            f"graph extraction left {processing_results['graph_failed']} failed "
            f"writes for ingest {job.ingest_id}"
        )

    # Re-attach graph counts onto the original envelope the caller's
    # _summarise reads from (so /ingestion/{id}/status surfaces them).
    if isinstance(pipeline_envelope, dict):
        pipeline_envelope["graph_nodes"] = processing_results.get("graph_nodes", 0)
        pipeline_envelope["graph_edges"] = processing_results.get("graph_edges", 0)
        pipeline_envelope["graph_failed"] = processing_results.get("graph_failed", 0)
        return pipeline_envelope
    return processing_results


async def _claim_heartbeat(
    redis: aioredis.Redis,
    job: IngestJob,
    config: WorkerConfig,
    stop: asyncio.Event,
) -> None:
    """Keep the claimed entry's PEL idle clock fresh while the pipeline runs.

    Without this, any job whose pipeline outlives ``reaper_min_idle_ms``
    is XAUTOCLAIMed away mid-processing and re-driven concurrently: the
    pipeline runs twice, the tenant counter double-decrements, and after
    enough reclaims a legitimately long video is dead-lettered as poison.
    Best-effort: a failed refresh retries next interval; if Redis is down
    the pipeline's own Redis calls fail first.
    """
    while True:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=config.heartbeat_interval_s)
        if stop.is_set():
            return
        try:
            await queue.refresh_claim(
                redis, config.consumer_group, config.consumer_id, job.message_id
            )
        except Exception:
            logger.warning(
                "Claim heartbeat failed for %s; retrying next interval",
                job.ingest_id,
                exc_info=True,
            )


async def _process_job(
    redis: aioredis.Redis,
    job: IngestJob,
    config: WorkerConfig,
    *,
    processor,
    telemetry_otlp_endpoint: str | None = None,
) -> None:
    """Run one job end-to-end and publish events for every state
    change. ACKs terminal success or pre-content failure. A graph-stage
    failure is nonterminal and retains the PEL entry for an idempotent
    re-drive. Cleanup steps are each best-effort; a step that
    fails (including the ack) is named on the terminal event's
    ``cleanup_error`` and whatever it left behind is recovered by the
    reaper (``reaper.py``).

    ``processor`` is injectable for tests that don't need the full
    Vespa+ColPali stack — production uses ``_default_processor``.

    Event ordering matters: cleanup (clear_inflight, mark_done,
    decrement_active, ack) runs BEFORE the terminal event publishes,
    so when an SSE watcher observes ``state=complete|failed`` all
    invariants (active counter accurate, idempotency record settled,
    queue PEL drained) are guaranteed consistent.
    """
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    success = False
    retrying = False
    terminal_event: dict
    # The heartbeat starts before ANY other work: telemetry cold init and
    # the running publish can stall, and the claim must never look
    # abandoned while this coroutine owns the job.
    stop_heartbeat = asyncio.Event()
    heartbeat = asyncio.create_task(
        _claim_heartbeat(redis, job, config, stop_heartbeat)
    )
    # Outer span wraps the full job lifecycle (processor + cleanup +
    # status publish). component=pipeline so the TelemetryLevel filter
    # admits at DETAILED+. Errors propagate into the span via the
    # contextmanager's try/yield/except path.
    try:
        tm = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
        await queue.publish_status(
            redis,
            job.ingest_id,
            {
                "state": "running",
                "ingest_id": job.ingest_id,
                "consumer_id": config.consumer_id,
            },
        )
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
                if config.job_deadline_s > 0:
                    try:
                        result = await asyncio.wait_for(
                            processor(job), timeout=config.job_deadline_s
                        )
                    except TimeoutError:
                        raise JobDeadlineExceeded(
                            f"job exceeded the {config.job_deadline_s}s "
                            "wall-clock deadline"
                        ) from None
                else:
                    result = await processor(job)
                _raise_if_pipeline_failed(result)
                success = True
                terminal_event = {
                    "state": "complete",
                    "ingest_id": job.ingest_id,
                    "result": _summarise(result),
                }
                job_span.set_attribute("job.outcome", "success")
            except GraphStageIncomplete as exc:
                retrying = True
                terminal_event = {
                    "state": "retrying",
                    "ingest_id": job.ingest_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                job_span.set_attribute("job.outcome", "retrying")
                job_span.set_attribute("job.error_type", type(exc).__name__)
            except Exception as exc:
                logger.exception("Ingest job %s failed", job.ingest_id)
                # Guard str(exc): an exception whose __str__ itself raises would
                # otherwise propagate here before terminal_event is built, so no
                # failed terminal is published and the cleanup below never runs —
                # the client sees "running" until the reaper dead-letters it and
                # the tenant slot leaks for the active-counter TTL.
                try:
                    error_text = str(exc)
                except Exception:
                    error_text = f"<unprintable {type(exc).__name__}>"
                terminal_event = {
                    "state": "failed",
                    "ingest_id": job.ingest_id,
                    "error": error_text,
                    "error_type": type(exc).__name__,
                }
                job_span.set_attribute("job.outcome", "failed")
                job_span.set_attribute("job.error_type", type(exc).__name__)
    finally:
        stop_heartbeat.set()
        await heartbeat

    if retrying:
        # _default_processor writes this before entering the graph boundary.
        # Repeating it here makes injected processors obey the same state
        # machine and closes the exception-to-status-publish crash window.
        await _mark_graph_pending(redis, job)
        await queue.publish_status(redis, job.ingest_id, terminal_event)
        return

    # Each cleanup step is independently best-effort: a failure in one must
    # not skip the others (a clear_inflight blip would otherwise leave the
    # tenant slot occupied and the message stuck in the PEL). Failed steps
    # are named on the terminal event so a watcher can distinguish a fully
    # clean terminal from one needing reaper recovery, and the reaper can
    # pick up whatever an unacked entry left behind.
    cleanup_errors: list = []

    async def _cleanup_step(name: str, coro) -> None:
        try:
            await coro
        except Exception as exc:
            logger.exception(
                "Cleanup step %s failed for %s; continuing with remaining steps",
                name,
                job.ingest_id,
            )
            cleanup_errors.append(f"{name}: {exc}")

    if success:
        await _cleanup_step(
            "mark_done",
            idempotency.mark_done(
                redis, job.sha, job.ingest_id, ttl_seconds=config.idempotency_ttl
            ),
        )
    await _cleanup_step("clear_inflight", idempotency.clear_inflight(redis, job.sha))
    await _cleanup_step(
        "clear_graph_pending",
        _clear_graph_pending(redis, job.message_id),
    )
    await _cleanup_step(
        "decrement_active", queue.decrement_active(redis, job.tenant_id)
    )
    await _cleanup_step("ack", queue.ack(redis, config.consumer_group, job.message_id))
    if cleanup_errors:
        terminal_event["cleanup_error"] = "; ".join(cleanup_errors)

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
    *,
    processor,
    telemetry_otlp_endpoint: str | None = None,
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
            try:
                await _process_job(
                    redis,
                    job,
                    config,
                    processor=processor,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            except Exception:
                # Status-publish/telemetry blips escape the per-job guard;
                # they must not kill the consumer. Leave the entry in the
                # PEL for the reaper and keep serving, mirroring the claim
                # guard above.
                logger.exception(
                    "processing %s crashed; leaving it to the reaper",
                    getattr(job, "ingest_id", "?"),
                )
                await asyncio.sleep(2.0)


async def run(
    stop: Optional[asyncio.Event] = None,
    processor=None,
) -> None:
    """Worker entry. Pass an ``asyncio.Event`` to drive shutdown from
    a test; production uses signal handlers below."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from cogniverse_runtime.entrypoint_env import resolve_library_env_defaults

    runtime_defaults = resolve_library_env_defaults()
    telemetry_otlp_endpoint = runtime_defaults["telemetry_otlp_endpoint"]
    media_config = _media_config_from_defaults(runtime_defaults)
    config = WorkerConfig()
    if stop is None:
        stop = asyncio.Event()
        _install_signal_handlers(stop)

    redis = await get_redis(config.redis_url)
    if processor is None:
        processor = partial(
            _default_processor,
            service_urls=config.inference_service_urls,
            mark_graph_pending=partial(_mark_graph_pending, redis),
            graph_deadline_s=config.graph_deadline_s,
            media_config=media_config,
        )
    logger.info(
        "Worker %s started: group=%s redis=%s reaper=%s",
        config.consumer_id,
        config.consumer_group,
        config.redis_url,
        "on" if config.reaper_enabled else "off",
    )
    reaper_task = None
    try:
        if config.reaper_enabled:
            from cogniverse_runtime.ingestion_worker.reaper import reaper_loop

            reaper_task = asyncio.create_task(
                reaper_loop(redis, config, stop, processor=processor)
            )
        await _claim_loop(
            redis,
            config,
            stop,
            processor=processor,
            telemetry_otlp_endpoint=telemetry_otlp_endpoint,
        )
    finally:
        logger.info("Worker %s stopping", config.consumer_id)
        if reaper_task is not None:
            reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reaper_task
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
