"""Integration test for the ingestion router's face pipeline wiring.

Drives ``_run_face_pipeline`` (the helper called from
``_extract_graph_per_segment`` once per source_doc_id) against a real
in-process face-embed sidecar and a hand-built ``ExtractionResult``
that already carries the Person + Concept nodes the previous pipeline
stages would have produced. Locks the W1–W4 contract:

* W1 — Successful end-to-end: face_cluster_temporal edges emitted for
  every cluster that overlaps a Person; orphan clusters scored against
  the KG profile bags and attributed when a Person dominates.
* W2 — Empty keyframes → zero face edges (no error).
* W3 — Sidecar 503 → zero face edges, ingestion continues (RuntimeError
  caught and degraded).
* W4 — Idempotent: rerunning the helper twice with the same input
  produces byte-equal edge lists.
"""

import base64
import importlib.util
import io
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from cogniverse_agents.graph.graph_schema import (
    ExtractionResult,
    Mention,
    Node,
)

SERVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "libs/runtime/cogniverse_runtime/sidecars/face_embed.py"
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------- #
# Pinned face vectors (float32-exact, identical to the face_extraction  #
# fixture so cluster IDs end up matching across suites).                #
# --------------------------------------------------------------------- #


ALICE_VEC = tuple([0.125] * 512)
BOB_VEC = tuple([-0.125] * 512)
_ALICE_COLOR = (200, 100, 50)
_BOB_COLOR = (50, 100, 200)
_SPLIT_COLOR = (128, 128, 128)


class _FakeFace:
    def __init__(self, bbox, vec, det_score):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.normed_embedding = np.array(vec, dtype=np.float32)
        self.det_score = det_score


class _ColorAwareFaceAnalysis:
    def __init__(self, name: str = "buffalo_l") -> None:
        self.name = name

    def prepare(self, ctx_id: int = -1, det_size=(640, 640)) -> None:  # noqa: ARG002
        return None

    def get(self, image_bgr: np.ndarray):
        px = tuple(int(c) for c in image_bgr[0, 0, ::-1])
        if px == _ALICE_COLOR:
            return [_FakeFace((100, 40, 200, 140), ALICE_VEC, 0.987)]
        if px == _BOB_COLOR:
            return [_FakeFace((80, 40, 180, 140), BOB_VEC, 0.964)]
        if px == _SPLIT_COLOR:
            return [
                _FakeFace((20, 40, 120, 140), ALICE_VEC, 0.943),
                _FakeFace((300, 40, 400, 140), BOB_VEC, 0.928),
            ]
        return []


# --------------------------------------------------------------------- #
# In-process sidecar fixture                                             #
# --------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def face_embed_url():
    fake_insightface = types.ModuleType("insightface")
    fake_app = types.ModuleType("insightface.app")
    fake_app.FaceAnalysis = _ColorAwareFaceAnalysis
    fake_insightface.app = fake_app
    saved = (
        sys.modules.get("insightface"),
        sys.modules.get("insightface.app"),
    )
    sys.modules["insightface"] = fake_insightface
    sys.modules["insightface.app"] = fake_app

    spec = importlib.util.spec_from_file_location(
        "face_embed_server_wired", str(SERVER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._MODEL = None

    import uvicorn  # noqa: PLC0415

    port = _free_port()
    config = uvicorn.Config(mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("face-embed sidecar did not come up within 30s")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if saved[0] is not None:
            sys.modules["insightface"] = saved[0]
        else:
            sys.modules.pop("insightface", None)
        if saved[1] is not None:
            sys.modules["insightface.app"] = saved[1]
        else:
            sys.modules.pop("insightface.app", None)


# --------------------------------------------------------------------- #
# Fixtures: processing_results + linked ExtractionResult                  #
# --------------------------------------------------------------------- #


def _solid_b64(color, size=(640, 480)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _debate_keyframes_payload():
    return {
        "keyframes": {
            "items": [
                {
                    "segment_id": "frame_5_0",
                    "ts_start": 5.0,
                    "image_b64": _solid_b64(_ALICE_COLOR),
                },
                {
                    "segment_id": "frame_15_0",
                    "ts_start": 15.0,
                    "image_b64": _solid_b64(_BOB_COLOR),
                },
            ]
        }
    }


def _t_mention(seg_id, ts_start, ts_end, span):
    return Mention(
        source_doc_id="debate_30s",
        segment_id=seg_id,
        ts_start=ts_start,
        ts_end=ts_end,
        modality="transcript",
        evidence_span=span,
    )


def _debate_linked_extraction():
    alice = Node(
        tenant_id="test",
        name="Alice Chen",
        mentions=[_t_mention("seg_1", 0.0, 10.0, "Alice Chen presents.")],
        label="Person",
        kind="entity",
    )
    bob = Node(
        tenant_id="test",
        name="Bob Smith",
        mentions=[_t_mention("seg_2", 10.0, 20.0, "Bob Smith responds.")],
        label="Person",
        kind="entity",
    )
    return ExtractionResult(
        source_doc_id="debate_30s",
        nodes=[alice, bob],
        edges=[],
    )


# --------------------------------------------------------------------- #
# W1 — End-to-end happy path                                             #
# --------------------------------------------------------------------- #


def test_face_pipeline_emits_temporal_attribution_edges(face_embed_url):
    from cogniverse_runtime.routers.ingestion import _run_face_pipeline

    edges, nodes = _run_face_pipeline(
        processing_results=_debate_keyframes_payload(),
        linked_extraction=_debate_linked_extraction(),
        source_doc_id="debate_30s",
        tenant_id="test",
        face_embed_url=face_embed_url,
    )
    # Two clusters (Alice@5, Bob@15) each overlap exactly one Person.
    assert len(edges) == 2
    # Both clusters got attributed → zero orphans → zero anonymous Nodes.
    assert nodes == []
    by_source = {e.source: e for e in edges}
    assert sorted(by_source.keys()) == [
        "face_cluster::frame_15_0::80_40",
        "face_cluster::frame_5_0::100_40",
    ]
    alice_edge = by_source["face_cluster::frame_5_0::100_40"]
    bob_edge = by_source["face_cluster::frame_15_0::80_40"]
    # Alice cluster (face@5.0) ∈ Alice's window [0,10] → confidence 1.0.
    assert alice_edge.target == "Alice Chen"
    assert alice_edge.confidence == 1.0
    assert alice_edge.provenance == "face_cluster_temporal"
    # Bob cluster (face@15.0) ∈ Bob's window [10,20] → confidence 1.0.
    assert bob_edge.target == "Bob Smith"
    assert bob_edge.confidence == 1.0
    assert bob_edge.provenance == "face_cluster_temporal"


# --------------------------------------------------------------------- #
# W2 — Empty keyframes → zero face edges                                  #
# --------------------------------------------------------------------- #


def test_empty_keyframes_yields_no_face_edges(face_embed_url):
    from cogniverse_runtime.routers.ingestion import _run_face_pipeline

    edges, nodes = _run_face_pipeline(
        processing_results={"keyframes": {"items": []}},
        linked_extraction=_debate_linked_extraction(),
        source_doc_id="debate_30s",
        tenant_id="test",
        face_embed_url=face_embed_url,
    )
    assert edges == []
    assert nodes == []


# --------------------------------------------------------------------- #
# W3 — Sidecar HTTP error → no edges, no propagation                      #
# --------------------------------------------------------------------- #


def test_sidecar_failure_degrades_gracefully():
    """Unreachable URL → _run_face_pipeline returns ([], []) instead of raising."""
    from cogniverse_runtime.routers.ingestion import _run_face_pipeline

    edges, nodes = _run_face_pipeline(
        processing_results=_debate_keyframes_payload(),
        linked_extraction=_debate_linked_extraction(),
        source_doc_id="debate_30s",
        tenant_id="test",
        face_embed_url="http://127.0.0.1:1",  # nothing listening on port 1
    )
    assert edges == []
    assert nodes == []


# --------------------------------------------------------------------- #
# W4 — Idempotency                                                        #
# --------------------------------------------------------------------- #


def test_face_pipeline_is_idempotent(face_embed_url):
    from cogniverse_runtime.routers.ingestion import _run_face_pipeline

    first_edges, first_nodes = _run_face_pipeline(
        processing_results=_debate_keyframes_payload(),
        linked_extraction=_debate_linked_extraction(),
        source_doc_id="debate_30s",
        tenant_id="test",
        face_embed_url=face_embed_url,
    )
    second_edges, second_nodes = _run_face_pipeline(
        processing_results=_debate_keyframes_payload(),
        linked_extraction=_debate_linked_extraction(),
        source_doc_id="debate_30s",
        tenant_id="test",
        face_embed_url=face_embed_url,
    )
    assert [
        (e.source, e.target, e.relation, e.confidence, e.provenance)
        for e in sorted(first_edges, key=lambda e: e.source)
    ] == [
        (e.source, e.target, e.relation, e.confidence, e.provenance)
        for e in sorted(second_edges, key=lambda e: e.source)
    ]
    assert [n.name for n in first_nodes] == [n.name for n in second_nodes]


# --------------------------------------------------------------------- #
# W5 — Endpoint lookup honours absent / present sidecar config            #
# --------------------------------------------------------------------- #


def _config_manager_with_inference_service_urls(urls: dict | None):
    """Build a stub ``ConfigManager`` whose ``SystemConfig.inference_service_urls``
    matches what ``main.py`` would set after reading ``INFERENCE_SERVICE_URLS``
    at startup. The face-embed lookup indexes into the parsed dict, not the
    raw env var, so the test injects the dict directly."""
    from unittest.mock import Mock

    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig

    sys_cfg = SystemConfig(inference_service_urls=urls or {})
    cm = Mock(spec=ConfigManager)
    cm.get_system_config = Mock(return_value=sys_cfg)
    return cm


def test_lookup_face_embed_endpoint_returns_none_without_env():
    from cogniverse_runtime.routers.ingestion import _lookup_face_embed_endpoint

    cm = _config_manager_with_inference_service_urls(None)
    assert _lookup_face_embed_endpoint(cm) is None


def test_lookup_face_embed_endpoint_returns_url_when_present():
    from cogniverse_runtime.routers.ingestion import _lookup_face_embed_endpoint

    cm = _config_manager_with_inference_service_urls(
        {"face_embed": "http://10.0.0.42:8000"}
    )
    assert _lookup_face_embed_endpoint(cm) == "http://10.0.0.42:8000"
