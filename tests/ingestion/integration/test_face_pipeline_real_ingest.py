"""A real video's faces reach the knowledge graph through the production
ingestion path.

``worker._default_processor`` localises the clip, runs the real
``VideoIngestionPipeline`` (keyframes, and the cluster's Whisper for the
transcript), then the per-segment KG stage: GLiNER and the chat LLM (Modal)
extract the transcript's entities and claims, and the face pipeline sends the
keyframes ``KeyframeProcessor`` wrote to the cluster's face-embed service. The
graph lands in a local Vespa. The keyframes carry only ``frame_number``,
``timestamp``, ``filename`` and ``path``, the shape real ingests produce.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import requests

from cogniverse_core.common.tenant_utils import canonical_tenant_id

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.requires_inference("vllm_asr"),
    pytest.mark.requires_inference("gliner"),
    pytest.mark.requires_inference("colbert_pylate"),
    pytest.mark.requires_inference("face_embed"),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEOS = REPO_ROOT / "tests/system/resources/videos"
BASE_PROFILE = "video_colpali_smol500_mv_frame"
PROFILE = f"{BASE_PROFILE}_faces"


def _kg_documents(http_port: int, tenant_id: str) -> list[dict]:
    schema = f"knowledge_graph_{canonical_tenant_id(tenant_id).replace(':', '_')}"
    response = requests.get(
        f"http://localhost:{http_port}/search/",
        params={"yql": f"select * from {schema} where true", "hits": 400},
        timeout=30,
    )
    response.raise_for_status()
    return [
        child["fields"]
        for child in response.json().get("root", {}).get("children", []) or []
        if "fields" in child
    ]


@pytest.fixture
def worker_env(shared_vespa, inference_endpoints, monkeypatch, tmp_path, request):
    """Point the worker at the local Vespa, the cluster's inference services
    and the Modal chat LLM, the way the deployed worker's environment does."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        BackendProfileConfig,
        SystemConfig,
    )
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.ingestion_worker import worker
    from cogniverse_runtime.routers import graph as graph_router
    from tests.utils.hermetic_llm import MODEL, ensure_llm

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    http_port = shared_vespa["http_port"]
    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(http_port))
    # No object store: the original-filename lookup and keyframe upload take
    # their documented no-MinIO paths.
    monkeypatch.setenv("MINIO_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("AWS_RETRY_MODE", "standard")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")

    config_blob = json.loads((REPO_ROOT / "configs/config.json").read_text())
    config_blob["backend"]["url"] = "http://localhost"
    config_blob["backend"]["port"] = http_port
    config_blob["llm_config"]["primary"]["api_base"] = ensure_llm(model=MODEL)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "schemas").symlink_to(
        (REPO_ROOT / "configs/schemas").resolve(), target_is_directory=True
    )
    (config_dir / "config.json").write_text(json.dumps(config_blob))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_dir / "config.json"))

    service_urls = {
        service: endpoint.base_url for service, endpoint in inference_endpoints.items()
    }
    config_manager = create_default_config_manager()
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=http_port,
            inference_service_urls=service_urls,
            telemetry_url="",
            telemetry_collector_endpoint="",
        )
    )
    profile = json.loads(json.dumps(config_blob["backend"]["profiles"][BASE_PROFILE]))
    profile["pipeline_config"]["generate_descriptions"] = False
    profile["pipeline_config"]["generate_embeddings"] = False
    profile["inference_services"].pop("embedding")
    profile["strategies"].pop("embedding")
    profile["strategies"].pop("description")
    tenant_id = request.param
    config_manager.set_backend_config(
        BackendConfig(
            tenant_id=tenant_id,
            profiles={PROFILE: BackendProfileConfig.from_dict(PROFILE, profile)},
            default_profiles={"video": {"profile": PROFILE}},
        )
    )

    saved_factory = graph_router._graph_manager_factory
    monkeypatch.setattr(worker, "_WORKER_LM", None)
    monkeypatch.setattr(worker, "_WORKER_LM_RESOLVED", False)
    # Each test installs the worker's graph factory against its own backend
    # registry, as a fresh worker process does.
    monkeypatch.setattr(worker, "_GRAPH_FACTORY_INSTALLED", False)
    try:
        yield {
            "http_port": http_port,
            "service_urls": service_urls,
            "tenant_id": tenant_id,
        }
    finally:
        graph_router.set_graph_manager_factory(saved_factory)
        BackendRegistry._instance = None
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None


async def _ingest(worker_env, clip: str, caplog) -> tuple[dict, list[str], list[dict]]:
    from cogniverse_runtime.ingestion_worker import queue, worker

    tenant_id = worker_env["tenant_id"]
    job = queue.IngestJob(
        message_id=f"0-{clip}",
        ingest_id=f"ing-{clip}",
        source_url=(VIDEOS / clip).as_uri(),
        profile=PROFILE,
        tenant_id=tenant_id,
        sha=f"sha-{clip}",
    )

    async def mark_graph_pending(pending_job):
        return None

    with caplog.at_level("INFO", logger="cogniverse_runtime.routers.ingestion"):
        result = await worker._default_processor(
            job,
            service_urls=worker_env["service_urls"],
            mark_graph_pending=mark_graph_pending,
            graph_deadline_s=1800,
        )
    face_logs = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_runtime.routers.ingestion"
        and r.getMessage().startswith("Face pipeline for source_doc_id=")
    ]
    # The tenant's graph schema was deployed during the run; its search
    # side answers once the new application is activated.
    deadline = time.monotonic() + 60
    while True:
        try:
            documents = _kg_documents(worker_env["http_port"], tenant_id)
        except requests.HTTPError:
            if time.monotonic() > deadline:
                raise
            documents = []
        if documents or time.monotonic() > deadline:
            return result, face_logs, documents
        time.sleep(2)


def _face_edges(documents: list[dict]) -> list[tuple]:
    """``(target, segment_id, ts_start, confidence)`` of each face edge.

    The face model's boxes, and so the cluster ids built from them, move by
    a pixel between runs on the cluster; which keyframe and Person an edge
    ties is stable."""
    return sorted(
        (
            d["target_node_id"],
            d["segment_id"],
            round(d["ts_start"], 4),
            d["confidence"],
        )
        for d in documents
        if d.get("doc_type") == "edge"
        and d.get("relation") == "same_as"
        and d.get("provenance") == "face_cluster_temporal"
    )


def _face_nodes(documents: list[dict]) -> list[str]:
    """The keyframe segment of each anonymous face node, from its
    ``face_cluster::<segment>::<x>_<y>`` name."""
    return sorted(
        d["name"].split("::")[1] for d in documents if d.get("kind") == "anonymous_face"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_env", ["faceingest:speaker"], indirect=True)
async def test_a_speakers_faces_are_tied_to_the_person_the_transcript_names(
    worker_env, caplog
):
    # 18 s, one speaker on camera who names himself "Bear Grylls"; the
    # keyframes at 0.5 fps fall every 1.97 s.
    result, face_logs, documents = await _ingest(
        worker_env, "v_-D1gdv_gQyw.mp4", caplog
    )

    assert result["status"] == "completed", result.get("error")
    assert face_logs == [
        "Face pipeline for source_doc_id=v_-D1gdv_gQyw: 4 faces in 10 keyframes, "
        "3 clusters, 3 same_as edges, 0 anonymous face nodes"
    ]
    assert _face_edges(documents) == [
        ("bear_grylls", "0", 0.0, 1.0),
        ("bear_grylls", "2", 3.9373, 1.0),
        ("bear_grylls", "3", 5.9059, 1.0),
    ]
    assert _face_nodes(documents) == []
    assert "Bear Grylls" in {
        d["name"] for d in documents if d.get("doc_type") == "node"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_env", ["faceingest:crowd"], indirect=True)
async def test_faces_no_transcript_person_covers_become_anonymous_face_nodes(
    worker_env, caplog
):
    # 8 s of a crowd; the transcript's only Person-labelled mention ("Yeah")
    # covers the first four keyframes, not the fifth at 7.88 s.
    result, face_logs, documents = await _ingest(
        worker_env, "v_-6dz6tBH77I.mp4", caplog
    )

    assert result["status"] == "completed", result.get("error")
    assert face_logs == [
        "Face pipeline for source_doc_id=v_-6dz6tBH77I: 24 faces in 5 keyframes, "
        "23 clusters, 19 same_as edges, 4 anonymous face nodes"
    ]
    assert _face_nodes(documents) == ["4", "4", "4", "4"]
    # Which keyframe anchors a cluster that spans two moves between runs, so
    # the edges are pinned by count, Person and the keyframes they cover.
    edges = _face_edges(documents)
    assert len(edges) == 19
    assert {target for target, *_ in edges} == {"yeah"}
    assert {segment for _, segment, *_ in edges} == {"0", "1", "2", "3"}
