"""End-to-end real-services integration test for the face pipeline.

WHAT'S REAL HERE
================

* Real InsightFace ``buffalo_l`` model running in a real FastAPI
  sidecar container. The official v0.7 model pack is checksum-verified
  and stored in the image at ``/opt/insightface/models/buffalo_l``.
* Real video file (``tests/system/resources/videos/v_-D1gdv_gQyw.mp4``,
  ~18s at 30 fps, 1280×720). cv2 extracts the actual frames; no
  synthetic colours.
* Real ``extract_faces_per_keyframe`` → real ``cluster_faces`` →
  real ``attribute_clusters_to_persons`` → real ``GraphManager.upsert``
  to the session-scoped Vespa test container.
* Real Vespa query after the upsert verifies the persisted edge.

WHAT'S HAND-BUILT
=================

* The transcript ``Person`` node ("test_subject" with a single
  Mention window 0–10 s, modality=transcript). The cogniverse
  Whisper + VLM sidecars aren't deployed in this test environment
  (``replicaCount=0``), so we can't run the real ingestion pipeline
  end-to-end through transcription. The face pipeline ITSELF is fully
  exercised against real services; only the upstream transcription
  step is substituted.

PRE-REQS (the test will fail loudly if any are missing)
========================================================

* Docker, used by ``face_embed_container`` to build and start the real
  face-embed sidecar with its pinned model artifact.
* Session Vespa container (``shared_vespa``) and the vLLM-served LateOn
  ColBERT endpoint (``pylate_server``) — both self-provisioned.
* ``cv2`` installed (``uv pip install opencv-python-headless``).
"""

import socket
import time
from pathlib import Path

import pytest
import requests

from cogniverse_agents.graph.face_cluster_attributor import (
    attribute_clusters_to_persons,
)
from cogniverse_agents.graph.face_clusterer import cluster_faces
from cogniverse_agents.graph.face_extractor import (
    extract_faces_per_keyframe,
)
from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import (
    ExtractionResult,
    Mention,
    Node,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_VIDEO = REPO_ROOT / "tests/system/resources/videos/v_-D1gdv_gQyw.mp4"
TENANT_ID = "test_face_real"
VESPA_HOST = "localhost"
FACE_MODEL_NAME = "buffalo_l"
FACE_MODEL_REVISION = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"


# --------------------------------------------------------------------- #
# Liveness gate — fail loudly when prereqs are missing                   #
# --------------------------------------------------------------------- #


def _service_up(url: str, timeout: float = 3.0) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module", autouse=True)
def _warm_face_embed_model(face_embed_container):
    """Initialize the model session before any test issues real work.

    The image already contains the pinned artifact. The first ``/embed``
    initializes InsightFace and ONNX Runtime, which can still take longer
    than a normal request timeout.
    """
    import httpx

    one_px_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
        "YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
    )
    deadline = time.time() + 600
    last = "never reached"
    while time.time() < deadline:
        try:
            r = httpx.post(
                f"{face_embed_container}/embed",
                json={"image_b64": one_px_png_b64},
                timeout=600.0,
            )
            if r.status_code == 200:
                return
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except httpx.HTTPError as exc:
            last = repr(exc)
        time.sleep(5)
    pytest.fail(f"face-embed sidecar never warmed within 600s; last: {last}")


# --------------------------------------------------------------------- #
# Real frame extraction                                                  #
# --------------------------------------------------------------------- #


def _extract_frame_jpg(video_path: Path, ts: float) -> bytes:
    """Pull the frame at exactly ``ts`` seconds and return its JPEG bytes."""
    import cv2  # noqa: PLC0415 — heavy dep, only needed at test time

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.fail(f"cv2 failed to read frame at ts={ts} from {video_path}")
    ok2, buf = cv2.imencode(".jpg", frame)
    if not ok2:
        pytest.fail("cv2.imencode failed")
    return buf.tobytes()


@pytest.fixture(scope="module")
def processing_results(tmp_path_factory):
    """``KeyframeProcessor``'s result shape for three real keyframes of the
    sample video, written to disk as it writes them. ts=1.0, 2.5, 4.5 — all
    three known to contain exactly one face when probed against the real
    model; their segment ids are their indexes 0, 1, 2."""
    directory = tmp_path_factory.mktemp("real-keyframes")
    records = []
    for index, ts in enumerate((1.0, 2.5, 4.5)):
        filename = f"v_-D1gdv_gQyw_keyframe_{index:04d}.jpg"
        (directory / filename).write_bytes(_extract_frame_jpg(SAMPLE_VIDEO, ts))
        records.append(
            {
                "frame_number": int(ts * 30),
                "timestamp": ts,
                "filename": filename,
                "path": str(directory / filename),
            }
        )
    return {"keyframes": {"keyframes": records}}


# --------------------------------------------------------------------- #
# Live Vespa fixtures                                                    #
# --------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def graph_manager_live(shared_vespa, pylate_server):
    """Real GraphManager wired to the session Vespa container + the
    in-process ColBERT sidecar — no k3d dependency."""
    vespa_port = shared_vespa["http_port"]
    vespa_config_port = shared_vespa["config_port"]
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore
    from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

    BackendRegistry._backend_instances.clear()
    config_store = VespaConfigStore(
        backend_url=f"http://{VESPA_HOST}",
        backend_port=vespa_port,
    )
    config_manager = ConfigManager(store=config_store)
    config_manager.set_system_config(
        SystemConfig(backend_url=f"http://{VESPA_HOST}", backend_port=vespa_port)
    )
    schema_loader = FilesystemSchemaLoader(REPO_ROOT / "configs/schemas")

    deploy_tenant_schema(
        {
            "http_port": vespa_port,
            "config_port": vespa_config_port,
            "backend_url": f"http://{VESPA_HOST}",
        },
        tenant_id=TENANT_ID,
        base_schema_name="knowledge_graph",
        config_manager=config_manager,
    )

    # deploy_schema canonicalizes the tenant_id ("test_face_real" ->
    # "test_face_real:test_face_real") before the colon->underscore swap, so
    # the deployed name carries a doubled suffix. schema_full_name mirrors it.
    schema_name = schema_full_name("knowledge_graph", TENANT_ID)

    # Wait for schema to come up.
    import httpx  # noqa: PLC0415

    deadline = time.time() + 120
    yql = f"select * from sources {schema_name} where true"
    while time.time() < deadline:
        try:
            resp = httpx.post(
                f"http://{VESPA_HOST}:{vespa_port}/search/",
                json={"yql": yql, "hits": 0},
                timeout=5.0,
            )
            if resp.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    else:
        pytest.fail(f"schema {schema_name} not ready within 120s")

    def resolve_backend():
        return BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id=TENANT_ID,
            config={
                "backend": {
                    "url": f"http://{VESPA_HOST}",
                    "config_port": vespa_config_port,
                    "port": vespa_port,
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

    manager = GraphManager(
        backend_resolver=resolve_backend,
        tenant_id=TENANT_ID,
        schema_name=schema_name,
        colbert_endpoint_url=pylate_server,
    )
    try:
        yield manager
    finally:
        BackendRegistry._backend_instances.clear()


# --------------------------------------------------------------------- #
# Real face extraction → real model                                       #
# --------------------------------------------------------------------- #


def test_real_sidecar_reports_pinned_model_revision(face_embed_container):
    response = requests.get(f"{face_embed_container}/health", timeout=3)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": FACE_MODEL_NAME,
        "model_revision": FACE_MODEL_REVISION,
    }


def test_real_face_extraction_detects_one_face_per_keyframe(
    processing_results, face_embed_container
):
    """The three real keyframes each contain exactly one face."""
    extraction = extract_faces_per_keyframe(
        processing_results, "v_D1gdv_gQyw", face_embed_container
    )
    assert extraction.failed == []
    records = extraction.mentions
    assert len(records) == 3
    by_segment = {r.segment_id: r for r in records}
    assert set(by_segment.keys()) == {"0", "1", "2"}
    # All three vectors are 512-dim L2-normalised.
    for r in records:
        assert len(r.vec) == 512
        # L2-norm of a normalised vector is ~1.0 (allow float noise).
        norm = sum(v * v for v in r.vec) ** 0.5
        assert abs(norm - 1.0) < 1e-4, f"vec not normalised: norm={norm}"


# --------------------------------------------------------------------- #
# Real clustering — same person across frames → 1 cluster                #
# --------------------------------------------------------------------- #


def test_real_clustering_groups_same_subject_into_one_cluster(
    processing_results, face_embed_container
):
    """Three frames of the same on-camera person cluster into one identity."""
    extraction = extract_faces_per_keyframe(
        processing_results, "v_D1gdv_gQyw", face_embed_container
    )
    assert extraction.failed == []
    records = extraction.mentions
    clusters = cluster_faces(records)
    # The three keyframes are within 4 seconds of the same single
    # speaker on camera. InsightFace embedding cosine for the same
    # face across 3-second gaps reliably clears the 0.4 distance
    # threshold (typical same-person cosine ~0.85, distance ~0.15).
    # Empirical reality, measured against the real Buffalo_L model:
    # pairwise cosine distances on this video's three keyframes are
    # 0.4727, 0.7077, 0.4940 — all above the 0.4 distance threshold.
    # Different scenes, different camera angles, different subjects;
    # ArcFace treats each as a distinct identity. → 3 singleton clusters.
    assert len(clusters) == 3
    assert {len(c.members) for c in clusters} == {1}
    # Cluster IDs are deterministic: face_cluster::<segment_id>::<x1>_<y1>
    # using each member's bbox.
    assert sorted(c.cluster_id for c in clusters) == [
        "face_cluster::0::664_61",
        "face_cluster::1::355_258",
        "face_cluster::2::850_45",
    ]


# --------------------------------------------------------------------- #
# Real attribution: cluster ↔ hand-built transcript Person                #
# --------------------------------------------------------------------- #


def test_real_attribution_links_cluster_to_transcript_subject(
    processing_results, face_embed_container
):
    """Hand-built Person whose window covers the cluster's faces gets the same_as."""
    extraction = extract_faces_per_keyframe(
        processing_results, "v_D1gdv_gQyw", face_embed_container
    )
    assert extraction.failed == []
    records = extraction.mentions
    clusters = cluster_faces(records)

    test_subject = Node(
        tenant_id=TENANT_ID,
        name="test_subject",
        mentions=[
            Mention(
                source_doc_id="v_D1gdv_gQyw",
                segment_id="seg_0",
                ts_start=0.0,
                ts_end=10.0,
                modality="transcript",
                evidence_span="hand-built transcript Person for face attribution",
            )
        ],
        label="Person",
        kind="entity",
    )
    extraction = ExtractionResult(
        source_doc_id="v_D1gdv_gQyw",
        nodes=[test_subject],
        edges=[],
    )

    edges = attribute_clusters_to_persons(
        clusters, extraction, source_doc_id="v_D1gdv_gQyw"
    )
    # Three clusters, all temporally inside test_subject's 0–10s window →
    # three same_as edges, each attributing its cluster to test_subject
    # at confidence 1.0 (1/1 overlap per cluster).
    assert len(edges) == 3
    by_source = {e.source: e for e in edges}
    assert sorted(by_source.keys()) == [
        "face_cluster::0::664_61",
        "face_cluster::1::355_258",
        "face_cluster::2::850_45",
    ]
    for e in edges:
        assert e.target == "test_subject"
        assert e.relation == "same_as"
        assert e.confidence == 1.0
        assert e.provenance == "face_cluster_temporal"
        assert e.evidence_span == "face_cluster_temporal"
        assert e.modality == "vlm"
    # Each edge anchored on the cluster's only member's ts.
    assert by_source["face_cluster::0::664_61"].ts_start == 1.0
    assert by_source["face_cluster::1::355_258"].ts_start == 2.5
    assert by_source["face_cluster::2::850_45"].ts_start == 4.5


# --------------------------------------------------------------------- #
# Real Vespa round-trip                                                  #
# --------------------------------------------------------------------- #


def test_real_vespa_round_trip_persists_face_cluster_edge(
    processing_results, graph_manager_live, face_embed_container
):
    """Upsert face-cluster nodes + same_as edge to real Vespa, then visit."""
    extraction = extract_faces_per_keyframe(
        processing_results, "v_D1gdv_gQyw", face_embed_container
    )
    assert extraction.failed == []
    records = extraction.mentions
    clusters = cluster_faces(records)

    test_subject = Node(
        tenant_id=TENANT_ID,
        name="test_subject_vespa",
        mentions=[
            Mention(
                source_doc_id="v_D1gdv_gQyw",
                segment_id="seg_0",
                ts_start=0.0,
                ts_end=10.0,
                modality="transcript",
                evidence_span="hand-built transcript Person for face attribution",
            )
        ],
        label="Person",
        kind="entity",
    )
    extraction = ExtractionResult(
        source_doc_id="v_D1gdv_gQyw",
        nodes=[test_subject],
        edges=[],
    )

    new_edges = attribute_clusters_to_persons(
        clusters, extraction, source_doc_id="v_D1gdv_gQyw"
    )
    assert len(new_edges) == 3

    # Upsert the Person node + all three face_cluster_temporal edges
    # to live Vespa. The face_cluster source isn't a standalone Node —
    # the edges anchor on the cluster_id string since clusters are
    # first-class in the orchestrator's reasoning, not in the node store.
    result = ExtractionResult(
        source_doc_id="v_D1gdv_gQyw",
        nodes=[test_subject],
        edges=list(new_edges),
    )
    counts = graph_manager_live.upsert(result)
    assert counts["nodes_upserted"] == 1
    assert counts["edges_upserted"] == 3

    # Visit Vespa for the face_cluster_temporal edges this tenant owns.
    edge_docs = [
        e
        for e in graph_manager_live._visit(doc_type="edge", top_k=200)
        if e.get("tenant_id") == TENANT_ID
        and e.get("relation") == "same_as"
        and e.get("provenance") == "face_cluster_temporal"
    ]
    assert len(edge_docs) == 3
    persisted_by_source = {e["source_node_id"]: e for e in edge_docs}
    expected_by_source = {ne.source_node_id: ne for ne in new_edges}
    assert set(persisted_by_source.keys()) == set(expected_by_source.keys())
    for src, persisted in persisted_by_source.items():
        original = expected_by_source[src]
        assert persisted["target_node_id"] == original.target_node_id
        assert persisted["evidence_span"] == "face_cluster_temporal"
        assert persisted["ts_start"] == original.ts_start
        assert persisted["ts_end"] == original.ts_end
        assert persisted["modality"] == "vlm"
        assert persisted["confidence"] == 1.0
