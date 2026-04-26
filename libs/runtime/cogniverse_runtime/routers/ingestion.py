"""Ingestion endpoints - unified interface for content ingestion."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel

from cogniverse_core.common.tenant_utils import require_tenant_id
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestionRequest(BaseModel):
    """Ingestion request model."""

    video_dir: str
    profile: str
    backend: str = "vespa"
    tenant_id: Optional[str] = None
    org_id: Optional[str] = None
    max_videos: Optional[int] = None
    batch_size: int = 10


class IngestionStatus(BaseModel):
    """Ingestion status response."""

    job_id: str
    status: str
    videos_processed: int
    videos_total: int
    errors: List[str] = []


# In-memory job tracking (replace with Redis/DB in production)
ingestion_jobs: Dict[str, IngestionStatus] = {}


# FastAPI dependencies - will be overridden in main.py via app.dependency_overrides
def get_config_manager_dependency():
    """
    FastAPI dependency for ConfigManager.

    This function should be overridden in main.py using app.dependency_overrides.
    If not overridden, it raises an error to fail fast.

    Returns:
        ConfigManager instance

    Raises:
        RuntimeError: If not overridden via app.dependency_overrides
    """
    raise RuntimeError(
        "ConfigManager dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides."
    )


def get_schema_loader_dependency() -> SchemaLoader:
    """
    FastAPI dependency for SchemaLoader.

    This function should be overridden in main.py using app.dependency_overrides.
    If not overridden, it raises an error to fail fast.

    Returns:
        SchemaLoader instance

    Raises:
        RuntimeError: If not overridden via app.dependency_overrides
    """
    raise RuntimeError(
        "SchemaLoader dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides."
    )


@router.post("/start")
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Dict[str, Any]:
    """Start video ingestion process."""
    try:
        # Validate inputs
        video_dir = Path(request.video_dir)
        if not video_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Video directory not found: {request.video_dir}",
            )

        # Get backend with dependency injection
        backend_registry = BackendRegistry.get_instance()
        try:
            tenant_id = require_tenant_id(request.tenant_id, source="IngestionRequest")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _backend = backend_registry.get_ingestion_backend(
            name=request.backend,
            tenant_id=tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Create job ID
        import uuid

        job_id = str(uuid.uuid4())

        # Initialize job status
        ingestion_jobs[job_id] = IngestionStatus(
            job_id=job_id,
            status="started",
            videos_processed=0,
            videos_total=0,
        )

        # Run ingestion in background
        background_tasks.add_task(
            run_ingestion,
            job_id=job_id,
            request=request,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        return {
            "job_id": job_id,
            "status": "started",
            "message": "Ingestion job started successfully",
        }

    except Exception as e:
        logger.error(f"Ingestion start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_ingestion_status(job_id: str) -> IngestionStatus:
    """Get status of ingestion job."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return ingestion_jobs[job_id]


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    profile: str = Form("default"),
    backend: str = Form("vespa"),
    tenant_id: Optional[str] = Form(None),
    org_id: Optional[str] = Form(None),
    wait: bool = Query(
        default=False,
        description=(
            "Block until ingestion reaches a terminal state. Default is "
            "async (returns 202 + ingest_id immediately)."
        ),
    ),
    wait_timeout: int = Query(
        default=300, ge=10, le=900, description="Max seconds to block when wait=true."
    ),
    force: bool = Query(
        default=False,
        description="Bypass idempotency and re-enqueue even on a cache hit.",
    ),
) -> Dict[str, Any]:
    """Upload and ingest a single file.

    Streams bytes to MinIO under ``{tenant}/{uuid}.{ext}``, derives the
    ``s3://`` URL, submits to the redis ingestion queue. Workers pull
    the queue, fetch via ``MediaLocator`` (which already speaks
    ``s3://``), run the pipeline, publish status events.

    Default response is 202 + ingest_id. ``?wait=true`` polls the
    status stream and returns a result-shaped response — useful for
    short jobs and callers that need synchronous behaviour. ``?force=
    true`` bypasses idempotency.

    Requires ``REDIS_URL`` + ``MINIO_ENDPOINT`` to be configured. No
    in-process pipeline fallback — single ingestion path, single
    backpressure budget, no second code path to rot.
    """
    try:
        upload_tenant_id = require_tenant_id(tenant_id, source="/ingestion/upload")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    redis_url = os.environ.get("REDIS_URL")
    minio_endpoint = os.environ.get("MINIO_ENDPOINT")
    if not redis_url or not minio_endpoint:
        missing = [
            name
            for name, value in (
                ("REDIS_URL", redis_url),
                ("MINIO_ENDPOINT", minio_endpoint),
            )
            if not value
        ]
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "/ingestion/upload requires the redis queue and MinIO "
                    "object store to be deployed."
                ),
                "missing_env": missing,
            },
        )

    from cogniverse_runtime.ingestion_v2 import minio_client
    from cogniverse_runtime.ingestion_v2.redis_client import get_redis
    from cogniverse_runtime.ingestion_v2.submit_api import (
        BackpressureError,
        enqueue_ingestion,
    )

    content = await file.read()
    source_url = minio_client.upload_bytes(
        content,
        tenant_id=upload_tenant_id,
        filename=file.filename,
        content_type=file.content_type,
    )

    redis = await get_redis(redis_url)
    try:
        result = await enqueue_ingestion(
            redis,
            source_url=source_url,
            profile=profile,
            tenant_id=upload_tenant_id,
            force=force,
            wait=wait,
            wait_timeout=wait_timeout,
        )
    except BackpressureError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "axis": exc.rejection.axis,
                "current": exc.rejection.current,
                "limit": exc.rejection.limit,
                "message": exc.rejection.message,
            },
        )

    response: Dict[str, Any] = {
        "ingest_id": result.ingest_id,
        "sha": result.sha,
        "state": result.state,
        "existing": result.existing,
        "filename": file.filename,
        "source_url": source_url,
    }
    if result.final_event is not None:
        # ``wait=true`` mode: re-emit the legacy response fields so
        # synchronous callers don't have to re-shape against the new
        # event payload.
        pipeline_result = result.final_event.get("result", {}) or {}
        response["video_id"] = pipeline_result.get("video_id")
        response["chunks_created"] = pipeline_result.get(
            "chunks", pipeline_result.get("keyframes", 0)
        )
        response["status"] = "success" if result.state == "complete" else result.state
    else:
        response["status"] = "queued"
    return response


def _extract_text_for_graph(processing_results: Dict[str, Any]) -> str:
    """Pull any LLM-analyzable text out of ingestion pipeline results.

    Walks the pipeline output and concatenates every text-ish field the
    processors already produced: Whisper transcripts (audio/video), VLM
    keyframe descriptions (images/video), and existing "extracted_text"
    blobs. Returns an empty string if nothing useful is found.
    """
    if not isinstance(processing_results, dict):
        return ""

    parts: list = []

    transcript = processing_results.get("transcript", {})
    if isinstance(transcript, dict):
        full = transcript.get("full_text") or transcript.get("text")
        if full:
            parts.append(str(full))
        segments = transcript.get("segments", [])
        if isinstance(segments, list):
            for seg in segments:
                if isinstance(seg, dict):
                    text = seg.get("text", "")
                    if text:
                        parts.append(str(text))

    descriptions = processing_results.get("descriptions", {})
    if isinstance(descriptions, dict):
        inner = descriptions.get("descriptions", descriptions)
        if isinstance(inner, dict):
            for desc in inner.values():
                if isinstance(desc, str):
                    parts.append(desc)
                elif isinstance(desc, dict):
                    text = desc.get("description") or desc.get("text") or ""
                    if text:
                        parts.append(str(text))

    keyframes_data = processing_results.get("keyframes", {})
    if isinstance(keyframes_data, dict):
        for kf in keyframes_data.get("keyframes", []) or []:
            if isinstance(kf, dict):
                text = kf.get("ocr_text") or kf.get("caption") or ""
                if text:
                    parts.append(str(text))

    doc_files = processing_results.get("document_files", [])
    if isinstance(doc_files, list):
        for doc in doc_files:
            if isinstance(doc, dict):
                text = doc.get("extracted_text") or ""
                if text:
                    parts.append(str(text))

    return "\n\n".join(parts).strip()


async def _extract_graph_from_multimodal(
    text: str,
    source_doc_id: str,
    tenant_id: str,
) -> Dict[str, int]:
    """Run the DocExtractor on multimodal text outputs and upsert to graph.

    Uses the same tenant-scoped GraphManager factory the /graph router
    uses, so no new manager wiring is needed.
    """
    from cogniverse_agents.graph.doc_extractor import DocExtractor
    from cogniverse_runtime.routers import graph as graph_router

    if graph_router._graph_manager_factory is None:
        return {"nodes_upserted": 0, "edges_upserted": 0}

    mgr = graph_router._graph_manager_factory(tenant_id)
    result = DocExtractor().extract_from_text(
        text=text,
        tenant_id=tenant_id,
        source_doc_id=source_doc_id,
    )
    if not result.nodes and not result.edges:
        return {"nodes_upserted": 0, "edges_upserted": 0}

    counts = mgr.upsert(result)
    logger.info(
        "Multimodal graph extraction for %s: %d nodes, %d edges",
        source_doc_id,
        counts["nodes_upserted"],
        counts["edges_upserted"],
    )
    return counts


async def run_ingestion(
    job_id: str,
    request: IngestionRequest,
    config_manager: ConfigManager,
    schema_loader: SchemaLoader,
) -> None:
    """Run ingestion process (background task)."""
    try:
        from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

        # Background task: the originating /ingestion endpoint already
        # validated tenant_id via require_tenant_id before enqueuing, so
        # this is belt-and-suspenders.  Missing tenant here is a logic
        # bug, not a user error — raise and let the task status record
        # the failure rather than silently ingesting under a ghost tenant.
        tenant_id = require_tenant_id(
            request.tenant_id, source="run_ingestion background task"
        )

        pipeline = VideoIngestionPipeline(
            tenant_id=tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
            schema_name=request.profile,
        )

        # Get video files
        video_dir = Path(request.video_dir)
        video_files = list(video_dir.glob("**/*.mp4"))

        if request.max_videos:
            video_files = video_files[: request.max_videos]

        # Update total
        ingestion_jobs[job_id].videos_total = len(video_files)
        ingestion_jobs[job_id].status = "processing"

        # Process videos using the async concurrent method
        result = await pipeline.process_videos_concurrent(
            video_files=video_files,
            max_concurrent=request.batch_size,
        )

        # Update job status from pipeline result
        ingestion_jobs[job_id].videos_processed = result.get("successful", 0)
        for error in result.get("errors", []):
            ingestion_jobs[job_id].errors.append(str(error))

        # Mark complete
        ingestion_jobs[job_id].status = "completed"

    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}")
        ingestion_jobs[job_id].status = "failed"
        ingestion_jobs[job_id].errors.append(str(e))
