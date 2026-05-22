"""Ingestion endpoints - unified interface for content ingestion."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import httpx
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

from cogniverse_agents.graph.graph_schema import Mention
from cogniverse_core.common.tenant_utils import assert_tenant_exists, require_tenant_id
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
        # Tenant identity first — auth-class checks (existence, ownership)
        # belong before any filesystem or request-content inspection so we
        # don't leak server-side state (e.g. dir existence) to unauth'd
        # callers.
        try:
            tenant_id = require_tenant_id(request.tenant_id, source="IngestionRequest")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        await assert_tenant_exists(tenant_id)

        video_dir = Path(request.video_dir)
        if not video_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Video directory not found: {request.video_dir}",
            )

        backend_registry = BackendRegistry.get_instance()
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
    """Upload a file to MinIO and enqueue ingestion via Redis.

    Returns 202 + ingest_id by default. ``wait=true`` polls until terminal
    state and returns a synchronous result. ``force=true`` bypasses idempotency.
    """
    try:
        upload_tenant_id = require_tenant_id(tenant_id, source="/ingestion/upload")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Reject ingestion for tenants that haven't been registered. Without
    # this guard the worker auto-deploys per-tenant schemas on first
    # upload, which produces "schema-only tenants" — schema in Vespa,
    # registry entry present, but no tenant_metadata document. In a
    # production deployment with auth, the auth layer enforces this
    # already; this check is the consistent answer for unauthenticated
    # local dev clusters and for any pre-auth code path.
    await assert_tenant_exists(upload_tenant_id)

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

    from cogniverse_runtime.ingestion_worker import minio_client
    from cogniverse_runtime.ingestion_worker.redis_client import get_redis
    from cogniverse_runtime.ingestion_worker.submit_api import (
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
        pipeline_result = result.final_event.get("result", {}) or {}
        response["video_id"] = pipeline_result.get("video_id")
        response["chunks_created"] = pipeline_result.get(
            "chunks", pipeline_result.get("keyframes", 0)
        )
        response["documents_fed"] = pipeline_result.get("documents_fed", 0)
        response["status"] = "success" if result.state == "complete" else result.state

        response["graph_nodes"] = 0
        response["graph_edges"] = 0
        if result.state == "complete":
            source_doc_id = (
                pipeline_result.get("video_id") or result.ingest_id or file.filename
            )
            counts = await _extract_graph_per_segment(
                processing_results=pipeline_result.get("results", {}) or {},
                source_doc_id=source_doc_id,
                tenant_id=upload_tenant_id,
            )
            response["graph_nodes"] = counts.get("nodes_upserted", 0)
            response["graph_edges"] = counts.get("edges_upserted", 0)
    else:
        response["status"] = "queued"
    return response


# Hard cap on the verbatim evidence_span captured per SegmentRecord.
_MAX_EVIDENCE_CHARS = 200


@dataclass
class SegmentRecord:
    """Per-segment text + temporal/positional anchor for graph extraction."""

    text: str
    segment_anchor: Mention


def _iter_segments_for_graph(
    processing_results: Dict[str, Any],
    source_doc_id: str,
) -> Iterator[SegmentRecord]:
    """Yield one SegmentRecord per Whisper segment, VLM keyframe, OCR/caption
    block, or document file present in the pipeline output.

    Whisper segments use the actual ``start``/``end`` timestamps. VLM and OCR
    keyframes use the keyframe ``timestamp`` for both ``ts_start`` and
    ``ts_end``. Document files use ``ts_start == ts_end == 0.0`` and
    ``segment_id == f"file_{i}"``.
    """
    if not isinstance(processing_results, dict):
        return

    # Build the keyframe timeline first so the transcript iterator can
    # align Whisper segments to each keyframe's 1-second window and
    # emit KG SegmentRecords whose segment_id matches the content
    # schema's keyframe index (the back-ref PATCH path keys off this).
    keyframes_data = processing_results.get("keyframes", {})
    keyframes_list = []
    if isinstance(keyframes_data, dict):
        kfs = keyframes_data.get("keyframes", [])
        if isinstance(kfs, list):
            keyframes_list = kfs
    keyframe_ts_by_id: Dict[str, float] = {}
    keyframe_windows: List[tuple[int, float, float]] = []
    for idx, kf in enumerate(keyframes_list):
        if not isinstance(kf, dict):
            continue
        # The keyframe processor writes ``frame_number``; accept both
        # keys so this iterator works regardless of which extractor
        # wrote them.
        fid = kf.get("frame_id")
        if fid is None:
            fid = kf.get("frame_number")
        if fid is None:
            continue
        ts = float(kf.get("timestamp", 0.0) or 0.0)
        keyframe_ts_by_id[str(fid)] = ts
        # Frame-based profile: each keyframe owns a 1-second window
        # starting at its timestamp. The content schema's segment_id is
        # the integer index ``idx``; doc_ids are ``<video_id>_seg_<idx>``.
        keyframe_windows.append((idx, ts, ts + 1.0))

    transcript = processing_results.get("transcript", {})
    if isinstance(transcript, dict):
        segments = transcript.get("segments", [])
        if isinstance(segments, list) and segments:
            # Align Whisper segments to keyframe windows so KG
            # segment_id matches the content schema's keyframe index.
            # Without this alignment, transcript KG nodes live under
            # ``seg_<idx>`` and back-ref PATCHes can't find their
            # content doc. Falls back to the raw transcript layout
            # when keyframe_windows is empty (chunk-based profiles).
            if keyframe_windows:
                for idx, win_start, win_end in keyframe_windows:
                    overlap_text_parts: List[str] = []
                    span_start = win_end
                    span_end = win_start
                    for seg in segments:
                        if not isinstance(seg, dict):
                            continue
                        seg_start = float(seg.get("start", 0.0) or 0.0)
                        seg_end = float(seg.get("end", seg_start) or seg_start)
                        if seg_end <= win_start or seg_start >= win_end:
                            continue
                        text = str(seg.get("text") or "").strip()
                        if text:
                            overlap_text_parts.append(text)
                            span_start = min(span_start, seg_start)
                            span_end = max(span_end, seg_end)
                    if not overlap_text_parts:
                        continue
                    combined = " ".join(overlap_text_parts)
                    yield SegmentRecord(
                        text=combined,
                        segment_anchor=Mention(
                            source_doc_id=source_doc_id,
                            segment_id=str(idx),
                            ts_start=float(span_start),
                            ts_end=float(span_end),
                            modality="transcript",
                            evidence_span=combined[:_MAX_EVIDENCE_CHARS],
                        ),
                    )
            else:
                for idx, seg in enumerate(segments):
                    if not isinstance(seg, dict):
                        continue
                    text = seg.get("text") or ""
                    if not text:
                        continue
                    text = str(text)
                    yield SegmentRecord(
                        text=text,
                        segment_anchor=Mention(
                            source_doc_id=source_doc_id,
                            segment_id=f"seg_{idx}",
                            ts_start=float(seg.get("start", 0.0) or 0.0),
                            ts_end=float(seg.get("end", 0.0) or 0.0),
                            modality="transcript",
                            evidence_span=text[:_MAX_EVIDENCE_CHARS],
                        ),
                    )

    descriptions = processing_results.get("descriptions", {})
    if isinstance(descriptions, dict):
        inner = descriptions.get("descriptions", descriptions)
        if isinstance(inner, dict):
            for frame_id, desc in inner.items():
                if isinstance(desc, str):
                    text = desc
                elif isinstance(desc, dict):
                    text = desc.get("description") or desc.get("text") or ""
                else:
                    text = ""
                if not text:
                    continue
                text = str(text)
                ts = keyframe_ts_by_id.get(str(frame_id), 0.0)
                yield SegmentRecord(
                    text=text,
                    segment_anchor=Mention(
                        source_doc_id=source_doc_id,
                        segment_id=f"frame_{frame_id}",
                        ts_start=ts,
                        ts_end=ts,
                        modality="vlm",
                        evidence_span=text[:_MAX_EVIDENCE_CHARS],
                    ),
                )

    # OCR / caption blocks attached to keyframes.
    for kf in keyframes_list:
        if not isinstance(kf, dict):
            continue
        text = kf.get("ocr_text") or kf.get("caption") or ""
        if not text:
            continue
        text = str(text)
        # ``frame_id`` can legitimately be 0 — use explicit None checks
        # rather than ``or`` so 0 doesn't fall through to ``frame_number``
        # and end up as empty-string in the segment_id.
        fid = kf.get("frame_id")
        if fid is None:
            fid = kf.get("frame_number", "")
        ts = float(kf.get("timestamp", 0.0) or 0.0)
        yield SegmentRecord(
            text=text,
            segment_anchor=Mention(
                source_doc_id=source_doc_id,
                segment_id=f"frame_{fid}",
                ts_start=ts,
                ts_end=ts,
                modality="ocr",
                evidence_span=text[:_MAX_EVIDENCE_CHARS],
            ),
        )

    doc_files = processing_results.get("document_files", [])
    if isinstance(doc_files, list):
        for idx, doc in enumerate(doc_files):
            if not isinstance(doc, dict):
                continue
            text = doc.get("extracted_text") or ""
            if not text:
                continue
            text = str(text)
            yield SegmentRecord(
                text=text,
                segment_anchor=Mention(
                    source_doc_id=source_doc_id,
                    segment_id=f"file_{idx}",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="document",
                    evidence_span=text[:_MAX_EVIDENCE_CHARS],
                ),
            )


async def _extract_graph_per_segment(
    processing_results: Dict[str, Any],
    source_doc_id: str,
    tenant_id: str,
) -> Dict[str, Any]:
    """Run per-segment KG extraction, cross-modal linking, and upsert.

    For every ``SegmentRecord`` yielded by ``_iter_segments_for_graph``,
    invoke ``DocExtractor.extract_from_text(..., segment_anchor=...)`` and
    accumulate the resulting nodes and edges into a single
    ``ExtractionResult``. ``CrossModalLinker.link`` then adds ``same_as``
    edges across modalities. Finally, ``GraphManager.upsert`` persists the
    full result and per-segment back-refs are PATCHed onto the
    corresponding content documents in Vespa.
    """
    from cogniverse_agents.graph.claim_extractor import ClaimExtractor
    from cogniverse_agents.graph.cross_modal_linker import CrossModalLinker
    from cogniverse_agents.graph.doc_extractor import DocExtractor
    from cogniverse_agents.graph.graph_schema import ExtractionResult
    from cogniverse_runtime.routers import graph as graph_router

    empty: Dict[str, Any] = {
        "nodes_upserted": 0,
        "edges_upserted": 0,
        "backrefs_by_segment": {},
    }

    if graph_router._graph_manager_factory is None:
        return empty

    mgr = graph_router._graph_manager_factory(tenant_id)

    claim_extractor = ClaimExtractor(
        artifact_manager=_lookup_artifact_manager(tenant_id)
    )
    doc_ext = DocExtractor(claim_extractor=claim_extractor)

    accumulated_nodes = []
    accumulated_edges = []
    backrefs_by_segment: Dict[str, Dict[str, List[str]]] = {}
    # Entity names seen so far across this source_doc_id. Passed forward
    # as ``prior_entities`` so the ClaimExtractor can resolve pronoun
    # coreferences in later segments (``She later won the Nobel Prize``
    # binds ``She`` → ``Marie Curie`` when Marie Curie was already
    # extracted from an earlier segment).
    entity_pool: List[str] = []
    entity_pool_seen: set[str] = set()

    segments_list = list(_iter_segments_for_graph(processing_results, source_doc_id))
    logger.info(
        "KG extraction: %d segments yielded for source_doc_id=%s "
        "(transcript_keys=%s, descriptions_keys=%s, keyframes_keys=%s)",
        len(segments_list),
        source_doc_id,
        list((processing_results.get("transcript") or {}).keys())
        if isinstance(processing_results.get("transcript"), dict)
        else type(processing_results.get("transcript")).__name__,
        list((processing_results.get("descriptions") or {}).keys())
        if isinstance(processing_results.get("descriptions"), dict)
        else type(processing_results.get("descriptions")).__name__,
        list((processing_results.get("keyframes") or {}).keys())
        if isinstance(processing_results.get("keyframes"), dict)
        else type(processing_results.get("keyframes")).__name__,
    )
    for record in segments_list:
        result = doc_ext.extract_from_text(
            text=record.text,
            tenant_id=tenant_id,
            source_doc_id=source_doc_id,
            segment_anchor=record.segment_anchor,
            prior_entities=list(entity_pool),
        )
        accumulated_nodes.extend(result.nodes)
        accumulated_edges.extend(result.edges)
        for n in result.nodes:
            if n.name.lower() not in entity_pool_seen:
                entity_pool.append(n.name)
                entity_pool_seen.add(n.name.lower())

        bucket = backrefs_by_segment.setdefault(
            record.segment_anchor.segment_id,
            {"entity_ids": [], "relation_ids": [], "claim_ids": []},
        )
        for node in result.nodes:
            if node.node_id not in bucket["entity_ids"]:
                bucket["entity_ids"].append(node.node_id)
        for edge in result.edges:
            if edge.edge_id not in bucket["relation_ids"]:
                bucket["relation_ids"].append(edge.edge_id)
            if edge.edge_id not in bucket["claim_ids"]:
                bucket["claim_ids"].append(edge.edge_id)

    combined = ExtractionResult(
        source_doc_id=source_doc_id,
        nodes=accumulated_nodes,
        edges=accumulated_edges,
    )

    # CrossModalLinker no longer depends on the ColBERT sidecar — it
    # works purely off the existing Node.label tags and transcript
    # Person-mention frequencies. Always runs; cheap.
    linker = CrossModalLinker()
    linked = linker.link(combined)

    # Face pipeline — opt-in via INFERENCE_SERVICE_URLS['face_embed'].
    # Adds same_as edges for cross-modal face↔Person identity links
    # (temporal attribution + KG-overlap third chance).
    face_embed_url = _lookup_face_embed_endpoint()
    face_edges = (
        _run_face_pipeline(
            processing_results=processing_results,
            linked_extraction=linked,
            source_doc_id=source_doc_id,
            tenant_id=tenant_id,
            face_embed_url=face_embed_url,
        )
        if face_embed_url
        else []
    )
    if face_edges:
        linked = ExtractionResult(
            source_doc_id=linked.source_doc_id,
            nodes=linked.nodes,
            edges=list(linked.edges) + list(face_edges),
            file_sha256=linked.file_sha256,
        )

    # Account for the same_as edges the linker (and the face pipeline)
    # added against the original segment that anchored each new edge.
    original_edge_ids = {edge.edge_id for edge in combined.edges}
    for edge in linked.edges:
        if edge.edge_id in original_edge_ids:
            continue
        bucket = backrefs_by_segment.setdefault(
            edge.segment_id,
            {"entity_ids": [], "relation_ids": [], "claim_ids": []},
        )
        if edge.edge_id not in bucket["relation_ids"]:
            bucket["relation_ids"].append(edge.edge_id)
        if edge.edge_id not in bucket["claim_ids"]:
            bucket["claim_ids"].append(edge.edge_id)

    counts = mgr.upsert(linked)

    await _write_backrefs_to_content(
        backrefs_by_segment=backrefs_by_segment,
        processing_results=processing_results,
        source_doc_id=source_doc_id,
        tenant_id=tenant_id,
    )

    return {
        "nodes_upserted": counts.get("nodes_upserted", 0),
        "edges_upserted": counts.get("edges_upserted", 0),
        "backrefs_by_segment": backrefs_by_segment,
    }


def _lookup_artifact_manager(tenant_id: str):
    """Look up an ArtifactManager for the tenant.

    Returns ``None`` when Phoenix telemetry isn't wired up; ``ClaimExtractor``
    handles ``None`` gracefully (skips compiled-artifact loading).
    """
    try:
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_telemetry_phoenix.provider import PhoenixProvider
    except ImportError:
        return None

    http_endpoint = os.environ.get("PHOENIX_HTTP_ENDPOINT")
    grpc_endpoint = os.environ.get("PHOENIX_GRPC_ENDPOINT")
    if not http_endpoint and not grpc_endpoint:
        return None

    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": http_endpoint or "http://localhost:6006",
            "grpc_endpoint": grpc_endpoint or "localhost:4317",
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


def _lookup_face_embed_endpoint() -> Optional[str]:
    """Return the ``face_embed`` sidecar URL.

    Reads from ``INFERENCE_SERVICE_URLS`` (same env-var contract as
    ``_lookup_colbert_endpoint``). When the var is missing or doesn't
    contain the ``face_embed`` key, the face pipeline is skipped
    gracefully — the per-segment KG extraction + structural cross-modal
    linker still ship, just without face-cluster ``same_as`` edges.
    """
    import json as _json

    raw = os.environ.get("INFERENCE_SERVICE_URLS")
    if not raw:
        return None
    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed.get("face_embed")


def _run_face_pipeline(
    processing_results: Dict[str, Any],
    linked_extraction,
    source_doc_id: str,
    tenant_id: str,
    face_embed_url: str,
) -> list:
    """Run face extract → cluster → temporal-attribute → KG-overlap.

    Returns the list of new same_as Edges to append to ``linked_extraction.edges``
    and have GraphManager.upsert persist alongside the rest. Empty list
    on any internal failure — face-path errors are logged but never
    propagated, so a face-sidecar outage doesn't break ingestion.

    Two attribution passes run:
      1. ``attribute_clusters_to_persons`` — emits face_cluster_temporal
         edges for every cluster that overlaps a transcript Person.
      2. ``attribute_orphans_by_kg_overlap`` — third-chance attribution
         for clusters left orphan by pass 1, scoring caption tokens
         against each Person's KG profile bag.
    """
    from cogniverse_agents.graph.face_cluster_attributor import (
        attribute_clusters_to_persons,
    )
    from cogniverse_agents.graph.face_clusterer import cluster_faces
    from cogniverse_agents.graph.face_extractor import extract_faces_per_keyframe
    from cogniverse_agents.graph.kg_overlap_attributor import (
        attribute_orphans_by_kg_overlap,
        build_person_profile_bags,
    )

    try:
        face_mentions = extract_faces_per_keyframe(
            processing_results=processing_results,
            source_doc_id=source_doc_id,
            face_embed_url=face_embed_url,
        )
    except Exception as exc:  # noqa: BLE001 — log + degrade, never fail ingest
        logger.warning(
            "Face extraction failed for source_doc_id=%s; skipping face pipeline: %s",
            source_doc_id,
            exc,
        )
        return []

    if not face_mentions:
        return []

    clusters = cluster_faces(face_mentions)
    temporal_edges = attribute_clusters_to_persons(
        clusters,
        linked_extraction,
        source_doc_id=source_doc_id,
    )
    attributed_cluster_ids = {edge.source for edge in temporal_edges}
    orphan_clusters = [
        c for c in clusters if c.cluster_id not in attributed_cluster_ids
    ]

    # Orphans need caption-token bags to score against Person profiles.
    # The caption tokens come from any non-Person Node that has a
    # Mention overlapping the cluster's keyframe windows — typically the
    # VLM-emitted Concept node the keyframe was tagged with.
    orphan_tuples: list = []
    if orphan_clusters:
        candidate_bags = build_person_profile_bags(linked_extraction)
        for cluster in orphan_clusters:
            tokens = _caption_tokens_for_cluster(cluster, linked_extraction)
            if tokens:
                orphan_tuples.append((cluster, tokens))
        if orphan_tuples and candidate_bags:
            kg_edges = attribute_orphans_by_kg_overlap(
                orphan_tuples,
                candidate_bags,
                source_doc_id=source_doc_id,
                tenant_id=tenant_id,
            )
        else:
            kg_edges = []
    else:
        kg_edges = []

    return temporal_edges + kg_edges


def _caption_tokens_for_cluster(cluster, extraction_result) -> set:
    """Token bag drawn from non-Person Nodes whose mentions overlap the cluster.

    Matches Mention.segment_id against the cluster's member segment_ids
    so a VLM Concept node ("woman in lab coat") emitted on the same
    keyframe contributes its name + evidence_span tokens to the orphan's
    caption bag for KG-overlap scoring.
    """
    import re as _re

    cluster_segments = {m.segment_id for m in cluster.members}
    tokens: set = set()
    for node in extraction_result.nodes:
        if (node.label or "").strip() == "Person":
            continue
        for mention in node.mentions:
            if mention.segment_id in cluster_segments:
                for raw in _re.findall(r"[A-Za-z0-9_]+", node.name):
                    if len(raw) >= 3:
                        tokens.add(raw.lower())
                for raw in _re.findall(r"[A-Za-z0-9_]+", mention.evidence_span or ""):
                    if len(raw) >= 3:
                        tokens.add(raw.lower())
                break  # one overlapping mention is enough
    return tokens


def _lookup_colbert_endpoint() -> Optional[str]:
    """Return the ``colbert_pylate`` sidecar URL.

    Reads from ``INFERENCE_SERVICE_URLS`` (the same JSON dict ``main.py``
    parses at startup into ``SystemConfig.inference_service_urls``). When
    the var is missing or doesn't contain the ``colbert_pylate`` key, no
    URL is returned and ``CrossModalLinker`` is skipped — the per-segment
    KG extraction still ships, just without ``same_as`` edges.
    """
    import json as _json

    raw = os.environ.get("INFERENCE_SERVICE_URLS")
    if not raw:
        return None
    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed.get("colbert_pylate")


async def _write_backrefs_to_content(
    backrefs_by_segment: Dict[str, Dict[str, List[str]]],
    processing_results: Dict[str, Any],
    source_doc_id: str,
    tenant_id: str,
) -> None:
    """PATCH ``entity_ids`` / ``relation_ids`` / ``claim_ids`` onto content docs.

    For each segment with back-refs, look up its content schema and doc_id
    from ``processing_results`` and issue a Vespa Document v1 partial
    update with ``{"fields": {"entity_ids": {"assign": [...]}}}``. Uses
    ``httpx.AsyncClient`` so all PATCHes for a single ingest fire
    concurrently.
    """
    if not backrefs_by_segment:
        return

    base_url = os.environ.get("VESPA_URL") or os.environ.get("VESPA_ENDPOINT")
    if not base_url:
        # Fall back to the BACKEND_URL + BACKEND_PORT env the
        # cogniverse-ingestor pod sets by default. Same shape, just
        # different env names — both deployments use these.
        bu = os.environ.get("BACKEND_URL")
        bp = os.environ.get("BACKEND_PORT")
        if bu and bp:
            base_url = f"{bu.rstrip('/')}:{bp}"
    if not base_url:
        logger.debug(
            "Skipping content back-ref PATCH: neither VESPA_URL nor "
            "BACKEND_URL+BACKEND_PORT are configured"
        )
        return

    # Build segment_id → (schema, doc_id) lookup. The pipeline doesn't
    # emit a top-level fed_documents list, so derive the per-segment
    # content doc_ids from the keyframes the segmentation step
    # produced. doc_id convention from the embedding feed:
    #   <schema_name>::docid::<video_id>_seg_<segment_id>
    # Schema name is <profile>_<tenant_sanitized>; the worker stamps
    # processing_results["__profile__"] + ["__schema_name__"] for us
    # before invoking the back-ref PATCH.
    targets: Dict[str, List[tuple[str, str]]] = {}
    schema = processing_results.get("__schema_name__") or processing_results.get(
        "schema_name"
    )
    video_id = (
        processing_results.get("video_id")
        or processing_results.get("__video_id__")
        or source_doc_id
    )

    fed_docs = processing_results.get("fed_documents") or []
    if isinstance(fed_docs, list) and fed_docs:
        for doc in fed_docs:
            if not isinstance(doc, dict):
                continue
            d_schema = doc.get("schema") or doc.get("content_schema") or schema
            doc_id = doc.get("doc_id") or doc.get("id")
            segment_id = doc.get("segment_id")
            if not d_schema or not doc_id or segment_id is None:
                continue
            targets.setdefault(str(segment_id), []).append((str(d_schema), str(doc_id)))

    # Fall back: derive (schema, doc_id) from keyframes list + schema name.
    if not targets and schema and video_id:
        keyframes_section = processing_results.get("keyframes") or {}
        kf_list = (
            keyframes_section.get("keyframes")
            if isinstance(keyframes_section, dict)
            else None
        )
        if isinstance(kf_list, list):
            for idx, kf in enumerate(kf_list):
                if not isinstance(kf, dict):
                    continue
                # The embedding step writes doc_id=<video_id>_seg_<idx>.
                doc_id = f"{video_id}_seg_{idx}"
                # Worker may also report segment_id as int idx; record
                # under both string keys so either lookup hits.
                segment_keys = {str(idx)}
                fid = kf.get("frame_id") or kf.get("frame_number")
                if fid is not None:
                    segment_keys.add(f"frame_{fid}")
                    segment_keys.add(str(fid))
                for k in segment_keys:
                    targets.setdefault(k, []).append((schema, doc_id))

    if not targets:
        return

    async with httpx.AsyncClient(timeout=15.0) as client:
        for segment_id, backrefs in backrefs_by_segment.items():
            for schema, doc_id in targets.get(segment_id, []):
                # Vespa Document v1 URL format:
                #   /document/v1/<namespace>/<doctype>/docid/<id>
                # Content schemas live under the ``content`` namespace
                # (the embedding feed writes
                # ``id:content:<schema>::<id>``), not under the schema
                # name. Older versions of this function used
                # ``<schema>/<schema>/docid/<id>`` which silently 404s.
                url = (
                    f"{base_url.rstrip('/')}/document/v1/"
                    f"content/{schema}/docid/{doc_id}"
                )
                payload = {
                    "fields": {
                        "entity_ids": {"assign": list(backrefs.get("entity_ids", []))},
                        "relation_ids": {
                            "assign": list(backrefs.get("relation_ids", []))
                        },
                        "claim_ids": {"assign": list(backrefs.get("claim_ids", []))},
                    }
                }
                resp = await client.put(url, json=payload)
                if not resp.is_success:
                    logger.warning(
                        "Content back-ref PATCH failed for %s/%s (%s): %s",
                        schema,
                        doc_id,
                        resp.status_code,
                        resp.text[:500],
                    )


async def run_ingestion(
    job_id: str,
    request: IngestionRequest,
    config_manager: ConfigManager,
    schema_loader: SchemaLoader,
) -> None:
    """Run ingestion process (background task)."""
    try:
        from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

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
