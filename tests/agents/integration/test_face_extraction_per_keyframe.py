"""Integration test for per-keyframe face extraction.

Drives the production ``extract_faces_per_keyframe`` helper against a
real FastAPI face-embed sidecar running in-process. The sidecar's
InsightFace model is stubbed so vectors stay deterministic, but the
HTTP boundary (uvicorn + httpx + Pydantic parsing) is real.

Locks the F1–F7 assertion contract documented in
``docs/plan/face-extraction-assertions.md``: total record count,
byte-equal serialised record list (vectors pinned in a golden file),
idempotency, empty-keyframe → zero records, multi-face keyframe →
distinct records, sidecar HTTP failure → RuntimeError with the failing
segment_id, and the FaceMention dataclass shape.
"""

import base64
import io
import json
import socket
import sys
import threading
import time
import types
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np
import pytest
import requests
from cogniverse_cli.modal_inference.servers import face as face_embed_server
from PIL import Image

from cogniverse_agents.graph.face_extractor import (
    extract_faces_per_keyframe,
    face_mention_as_jsonable,
)
from cogniverse_agents.graph.graph_schema import FaceMention

GOLDEN_DIR = Path(__file__).parent / "goldens"
FACE_MODEL_NAME = "buffalo_l"
FACE_MODEL_REVISION = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
FACE_MODEL_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------- #
# Pinned vectors                                                         #
# --------------------------------------------------------------------- #

# Use values that survive a float64 → float32 → float64 round-trip
# byte-equal. 0.125 == 2**-3 is exactly representable in float32; the
# usual decimal-looking constants (0.0123 etc.) are not, and the test
# fails by ~6e-9 if we use them.
ALICE_VEC = tuple([0.125] * 512)
BOB_VEC = tuple([-0.125] * 512)


def _alice_face():
    return _FakeFace(bbox=(100, 40, 200, 140), vec=ALICE_VEC, det_score=0.987)


def _bob_face_left():
    return _FakeFace(bbox=(80, 40, 180, 140), vec=BOB_VEC, det_score=0.964)


def _split_alice_face():
    return _FakeFace(bbox=(20, 40, 120, 140), vec=ALICE_VEC, det_score=0.943)


def _split_bob_face():
    return _FakeFace(bbox=(300, 40, 400, 140), vec=BOB_VEC, det_score=0.928)


class _FakeFace:
    def __init__(self, bbox, vec, det_score):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.normed_embedding = np.array(vec, dtype=np.float32)
        self.det_score = det_score


# --------------------------------------------------------------------- #
# Image fixture                                                          #
# --------------------------------------------------------------------- #

# Each test image uses a distinct solid colour. The mocked FaceAnalysis
# picks faces based on the dominant pixel — that gives us deterministic
# per-frame face configurations without needing real face detection.
_ALICE_COLOR = (200, 100, 50)
_BOB_COLOR = (50, 100, 200)
_SPLIT_COLOR = (128, 128, 128)
_EMPTY_COLOR = (10, 10, 10)


def _solid_b64(color, size=(640, 480)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _ColorAwareFaceAnalysis:
    """Stand-in for InsightFace that returns faces based on image colour.

    The integration test serialises three colour codes into solid PNGs;
    this mock decodes the dominant pixel of the input ndarray and emits
    the corresponding pinned face list.
    """

    last_root: str | None = None

    def __init__(
        self,
        name: str = FACE_MODEL_NAME,
        root: str | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.name = name
        self.root = root
        self.providers = providers
        type(self).last_root = root

    def prepare(self, ctx_id: int = -1, det_size=(640, 640)) -> None:  # noqa: ARG002
        return None

    def get(self, image_bgr: np.ndarray):
        # Sidecar converts incoming PNG to BGR. Read the first pixel.
        px = tuple(int(c) for c in image_bgr[0, 0, ::-1])  # BGR → RGB
        if px == _ALICE_COLOR:
            return [_alice_face()]
        if px == _BOB_COLOR:
            return [_bob_face_left()]
        if px == _SPLIT_COLOR:
            return [_split_alice_face(), _split_bob_face()]
        return []


# --------------------------------------------------------------------- #
# In-process sidecar fixture                                             #
# --------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def face_model_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("face-model")
    model_dir = root / "models" / FACE_MODEL_NAME
    model_dir.mkdir(parents=True)
    for filename in FACE_MODEL_FILES:
        (model_dir / filename).write_bytes(b"test model artifact")
    return root


@pytest.fixture(scope="module")
def face_embed_url(face_model_root, monkeypatch_session=None):
    """Yield a live face-embed sidecar URL with the stub model loaded."""
    fake_insightface = types.ModuleType("insightface")
    fake_app_module = types.ModuleType("insightface.app")
    fake_app_module.FaceAnalysis = _ColorAwareFaceAnalysis
    fake_insightface.app = fake_app_module
    saved_insightface = sys.modules.get("insightface")
    saved_app_module = sys.modules.get("insightface.app")
    sys.modules["insightface"] = fake_insightface
    sys.modules["insightface.app"] = fake_app_module

    mod = face_embed_server
    mod._MODEL = None
    app = mod.build_app(mod.FaceEmbedConfig(model_root=str(face_model_root)))

    import uvicorn  # noqa: PLC0415

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=1)
            if r.status_code == 200:
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
        if saved_insightface is not None:
            sys.modules["insightface"] = saved_insightface
        else:
            sys.modules.pop("insightface", None)
        if saved_app_module is not None:
            sys.modules["insightface.app"] = saved_app_module
        else:
            sys.modules.pop("insightface.app", None)


# --------------------------------------------------------------------- #
# Processing results fixture                                             #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def debate_processing_results():
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
                {
                    "segment_id": "frame_22_5",
                    "ts_start": 22.5,
                    "image_b64": _solid_b64(_SPLIT_COLOR),
                },
                {
                    "segment_id": "frame_28_0",
                    "ts_start": 28.0,
                    "image_b64": _solid_b64(_EMPTY_COLOR),
                },
            ]
        }
    }


# --------------------------------------------------------------------- #
# F1 — Total record count exact                                          #
# --------------------------------------------------------------------- #


def test_total_record_count(face_embed_url, debate_processing_results):
    """1 (Alice@5) + 1 (Bob@15) + 2 (split@22.5) + 0 (empty@28) == 4."""
    records = extract_faces_per_keyframe(
        debate_processing_results, "debate_30s", face_embed_url
    )
    assert len(records) == 4


def test_sidecar_reports_pinned_model_revision(face_embed_url):
    response = requests.get(f"{face_embed_url}/health", timeout=1)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": FACE_MODEL_NAME,
        "model_revision": FACE_MODEL_REVISION,
    }


def test_sidecar_loads_model_from_configured_root(
    face_embed_url, face_model_root, debate_processing_results
):
    records = extract_faces_per_keyframe(
        debate_processing_results, "debate_30s", face_embed_url
    )

    assert len(records) == 4
    assert _ColorAwareFaceAnalysis.last_root == str(face_model_root)


# --------------------------------------------------------------------- #
# F2 — Records byte-equal sorted by (segment_id, bbox)                   #
# --------------------------------------------------------------------- #


def test_records_byte_equal_sorted(face_embed_url, debate_processing_results):
    records = extract_faces_per_keyframe(
        debate_processing_results, "debate_30s", face_embed_url
    )
    serialised = [face_mention_as_jsonable(m) for m in records]
    expected = [
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_15_0",
            "ts_start": 15.0,
            "ts_end": 15.0,
            "bbox": [80, 40, 180, 140],
            "vec": list(BOB_VEC),
            "det_score": 0.964,
        },
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_22_5",
            "ts_start": 22.5,
            "ts_end": 22.5,
            "bbox": [20, 40, 120, 140],
            "vec": list(ALICE_VEC),
            "det_score": 0.943,
        },
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_22_5",
            "ts_start": 22.5,
            "ts_end": 22.5,
            "bbox": [300, 40, 400, 140],
            "vec": list(BOB_VEC),
            "det_score": 0.928,
        },
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_5_0",
            "ts_start": 5.0,
            "ts_end": 5.0,
            "bbox": [100, 40, 200, 140],
            "vec": list(ALICE_VEC),
            "det_score": 0.987,
        },
    ]
    assert json.dumps(serialised, sort_keys=True) == json.dumps(
        expected, sort_keys=True
    )


# --------------------------------------------------------------------- #
# F3 — Idempotency                                                       #
# --------------------------------------------------------------------- #


def test_idempotent_byte_equal(face_embed_url, debate_processing_results):
    first = extract_faces_per_keyframe(
        debate_processing_results, "debate_30s", face_embed_url
    )
    second = extract_faces_per_keyframe(
        debate_processing_results, "debate_30s", face_embed_url
    )
    assert [face_mention_as_jsonable(m) for m in first] == [
        face_mention_as_jsonable(m) for m in second
    ]


# --------------------------------------------------------------------- #
# F4 — Empty keyframe contributes nothing                                #
# --------------------------------------------------------------------- #


def test_empty_keyframe_yields_empty_list(face_embed_url):
    empty_only = {
        "keyframes": {
            "items": [
                {
                    "segment_id": "frame_28_0",
                    "ts_start": 28.0,
                    "image_b64": _solid_b64(_EMPTY_COLOR),
                }
            ]
        }
    }
    assert extract_faces_per_keyframe(empty_only, "debate_30s", face_embed_url) == []


# --------------------------------------------------------------------- #
# F5 — Multiple faces in one keyframe each get a distinct record         #
# --------------------------------------------------------------------- #


def test_multi_face_keyframe_emits_distinct_records(face_embed_url):
    split_only = {
        "keyframes": {
            "items": [
                {
                    "segment_id": "frame_22_5",
                    "ts_start": 22.5,
                    "image_b64": _solid_b64(_SPLIT_COLOR),
                }
            ]
        }
    }
    records = extract_faces_per_keyframe(split_only, "debate_30s", face_embed_url)
    assert len(records) == 2
    assert {tuple(r.bbox) for r in records} == {
        (20, 40, 120, 140),
        (300, 40, 400, 140),
    }
    assert {r.det_score for r in records} == {0.943, 0.928}


# --------------------------------------------------------------------- #
# F6 — Sidecar HTTP error surfaces with the failing segment_id           #
# --------------------------------------------------------------------- #


def test_sidecar_http_failure_raises_runtime_error_with_segment_id():
    def boom_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model not warm"})

    transport = httpx.MockTransport(boom_handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError) as exc_info:
            extract_faces_per_keyframe(
                {
                    "keyframes": {
                        "items": [
                            {
                                "segment_id": "frame_5_0",
                                "ts_start": 5.0,
                                "image_b64": _solid_b64(_ALICE_COLOR),
                            }
                        ]
                    }
                },
                "debate_30s",
                "http://boom.invalid",
                client=client,
            )
    msg = str(exc_info.value)
    assert "frame_5_0" in msg
    assert "503" in msg


# --------------------------------------------------------------------- #
# F7 — FaceMention dataclass shape locked                                #
# --------------------------------------------------------------------- #


def test_facemention_shape_locked():
    m = FaceMention(
        source_doc_id="x",
        segment_id="frame_0",
        ts_start=0.0,
        ts_end=0.0,
        bbox=(0, 0, 10, 10),
        vec=tuple([0.1] * 512),
        det_score=0.9,
    )
    import dataclasses as _dc

    fields = _dc.fields(FaceMention)
    assert len(fields) == 7
    assert fields[0].name == "source_doc_id"
    assert fields[1].name == "segment_id"
    assert fields[2].name == "ts_start"
    assert fields[3].name == "ts_end"
    assert fields[4].name == "bbox"
    assert fields[5].name == "vec"
    assert fields[6].name == "det_score"
    d = asdict(m)
    assert d["vec"][0] == 0.1
    assert d["bbox"] == (0, 0, 10, 10)


# --------------------------------------------------------------------- #
# F8 — Keyframes are POSTed concurrently, not serially                   #
# --------------------------------------------------------------------- #


def test_keyframes_posted_concurrently():
    """A threading.Barrier of N only releases once all N keyframe POSTs are
    in flight at the same time. Serial POSTs (the pre-fix behaviour) leave the
    barrier one party short forever and time out."""
    import threading

    n = 4
    barrier = threading.Barrier(n, timeout=5)

    def handler(request: httpx.Request) -> httpx.Response:
        barrier.wait()
        return httpx.Response(200, json={"faces": []})

    items = [
        {
            "segment_id": f"frame_{i}_0",
            "ts_start": float(i),
            "image_b64": _solid_b64(_EMPTY_COLOR),
        }
        for i in range(n)
    ]

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        records = extract_faces_per_keyframe(
            {"keyframes": {"items": items}},
            "debate_30s",
            "http://sidecar.invalid",
            client=client,
        )

    assert records == []


def test_modal_face_client_uses_environment_credential(monkeypatch):
    token = "shared-production-key"
    created = []

    class _CredentialCapturingClient:
        def __init__(self, *, headers):
            self.headers = dict(headers)
            self.calls = []
            self.closed = False
            created.append(self)

        def post(self, url, *, json, timeout):
            self.calls.append((url, json, timeout))
            return httpx.Response(200, json={"faces": []})

        def close(self):
            self.closed = True

    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", token)
    monkeypatch.setattr(
        "cogniverse_agents.graph.face_extractor.httpx.Client",
        _CredentialCapturingClient,
    )
    image_b64 = _solid_b64(_EMPTY_COLOR)

    records = extract_faces_per_keyframe(
        {
            "keyframes": {
                "items": [
                    {
                        "segment_id": "frame_0_0",
                        "ts_start": 0.0,
                        "image_b64": image_b64,
                    }
                ]
            }
        },
        "debate_30s",
        "https://face.modal.run",
    )

    assert records == []
    assert len(created) == 1
    assert created[0].headers == {"Authorization": f"Bearer {token}"}
    assert created[0].calls == [
        (
            "https://face.modal.run/embed",
            {"image_b64": image_b64},
            30.0,
        )
    ]
    assert created[0].closed is True


def test_modal_face_client_requires_environment_credential(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        extract_faces_per_keyframe(
            {"keyframes": {"items": []}},
            "debate_30s",
            "https://face.modal.run",
        )


def test_modal_face_client_rejects_caller_headers(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

    with pytest.raises(ValueError, match="headers.*Modal"):
        extract_faces_per_keyframe(
            {"keyframes": {"items": []}},
            "debate_30s",
            "https://face.modal.run",
            headers={"Authorization": "Bearer caller-specific-key"},
        )
