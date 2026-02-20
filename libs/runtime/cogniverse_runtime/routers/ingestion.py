"""Ingestion endpoints - unified interface for content ingestion."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

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
        tenant_id = request.tenant_id or "default"
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
    profile: str = "default",
    backend: str = "vespa",
    tenant_id: Optional[str] = None,
    org_id: Optional[str] = None,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Dict[str, Any]:
    """Upload and ingest a single video file."""
    try:
        # Save uploaded file temporarily
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Get backend with dependency injection
        backend_registry = BackendRegistry.get_instance()
        upload_tenant_id = tenant_id or "default"
        backend_instance = backend_registry.get_ingestion_backend(
            name=backend,
            tenant_id=upload_tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Process video
        from cogniverse_foundation.config.utils import get_config
        from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

        config = get_config(
            tenant_id=tenant_id or "default", config_manager=config_manager
        )

        pipeline = VideoIngestionPipeline(
            config=config, profile=profile, backend=backend_instance
        )

        result = pipeline.process_video(
            video_path=tmp_path, tenant_id=tenant_id, org_id=org_id
        )

        # Clean up temp file
        Path(tmp_path).unlink()

        return {
            "status": "success",
            "filename": file.filename,
            "video_id": result.get("video_id"),
            "chunks_created": result.get("chunks_created", 0),
        }

    except Exception as e:
        logger.error(f"Video upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_ingestion(
    job_id: str,
    request: IngestionRequest,
    config_manager: ConfigManager,
    schema_loader: SchemaLoader,
) -> None:
    """Run ingestion process (background task)."""
    try:
        from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

        tenant_id = request.tenant_id or "default"

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
