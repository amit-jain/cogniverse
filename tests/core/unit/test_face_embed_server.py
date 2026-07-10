"""Contract test for the face-embed FastAPI sidecar.

Loads ``cogniverse_runtime/sidecars/face_embed.py`` with InsightFace's
``FaceAnalysis`` class mocked so the test doesn't pull the 210 MiB
Buffalo_L weights. Verifies:

* ``GET /health`` returns ``{"status": "ok"}``.
* ``POST /embed`` with a valid base64-encoded image returns a response
  whose face count, bbox shape, embedding dimensionality, and detection
  scores match what the cogniverse face-cluster consumer expects.
* Input validation rejects requests with both / neither image_url and
  image_b64, and surfaces image-decode failures as HTTP 400.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

SERVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "libs/runtime/cogniverse_runtime/sidecars/face_embed.py"
)


class _FakeFace:
    """Stand-in for an InsightFace ``Face`` object."""

    def __init__(self, bbox, embedding, det_score):
        self.bbox = np.array(bbox, dtype=np.float32)
        # Real Buffalo_L emits 512-dim L2-normalised embeddings under
        # the ``normed_embedding`` attribute.
        self.normed_embedding = np.array(embedding, dtype=np.float32)
        self.det_score = det_score


class _FakeFaceAnalysis:
    """Stand-in for ``insightface.app.FaceAnalysis`` used only in tests."""

    def __init__(self, name: str = "buffalo_l") -> None:
        self.name = name

    def prepare(self, ctx_id: int = -1, det_size=(640, 640)) -> None:  # noqa: ARG002
        return None

    def get(self, image_bgr: np.ndarray):
        # Deterministic two-face response so we can assert ordering and
        # exact dimensionality without depending on real detection.
        h, w = image_bgr.shape[:2]
        return [
            _FakeFace(
                bbox=[10, 10, 110, 110],
                embedding=np.full(512, 0.0123, dtype=np.float32),
                det_score=0.987,
            ),
            _FakeFace(
                bbox=[w - 100, h - 100, w - 10, h - 10],
                embedding=np.full(512, -0.0123, dtype=np.float32),
                det_score=0.812,
            ),
        ]


@pytest.fixture
def server_module(monkeypatch):
    """Import ``server.py`` with insightface mocked. Return the module."""
    fake_insightface = types.ModuleType("insightface")
    fake_app_module = types.ModuleType("insightface.app")
    fake_app_module.FaceAnalysis = _FakeFaceAnalysis
    fake_insightface.app = fake_app_module
    monkeypatch.setitem(sys.modules, "insightface", fake_insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app_module)

    spec = importlib.util.spec_from_file_location(
        "face_embed_server_under_test", str(SERVER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Reset the module-level singleton between tests.
    mod._MODEL = None
    return mod


@pytest.fixture
def client(server_module):
    return TestClient(server_module.app)


@pytest.fixture
def sample_png_b64() -> str:
    """A minimal valid 200x200 RGB image encoded as base64 PNG."""
    img = Image.new("RGB", (200, 200), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.unit
@pytest.mark.ci_fast
class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestEmbedResponseShape:
    def test_two_faces_returned_with_expected_shape(self, client, sample_png_b64):
        resp = client.post("/embed", json={"image_b64": sample_png_b64})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["n"] == 2
        assert len(body["faces"]) == 2

        # bbox: list[int] of length 4
        face0 = body["faces"][0]
        assert face0["bbox"] == [10, 10, 110, 110]
        assert isinstance(face0["bbox"], list)
        assert len(face0["bbox"]) == 4
        assert all(isinstance(v, int) for v in face0["bbox"])

        # vec: 512-dim list of float, deterministic from the fake
        assert len(face0["vec"]) == 512
        assert all(isinstance(v, float) for v in face0["vec"])
        # ``np.full(512, 0.0123, dtype=float32)`` → exact-equal once cast
        assert face0["vec"][0] == pytest.approx(0.0123, abs=1e-7)

        # det_score: float
        assert face0["det_score"] == pytest.approx(0.987, abs=1e-6)

        # Second face uses (w-100, h-100, w-10, h-10) on a 200×200 input
        assert body["faces"][1]["bbox"] == [100, 100, 190, 190]
        assert body["faces"][1]["det_score"] == pytest.approx(0.812, abs=1e-6)

    def test_model_loaded_lazily(self, server_module, client, sample_png_b64):
        # Pre-request: model not initialised.
        assert server_module._MODEL is None
        client.post("/embed", json={"image_b64": sample_png_b64})
        # Post-request: model loaded and reused.
        assert server_module._MODEL is not None
        first_instance = server_module._MODEL
        client.post("/embed", json={"image_b64": sample_png_b64})
        assert server_module._MODEL is first_instance


class TestInputValidation:
    @pytest.mark.unit
    @pytest.mark.ci_fast
    def test_neither_field_supplied_returns_400(self, client):
        resp = client.post("/embed", json={})
        assert resp.status_code == 400
        assert "Exactly one of" in resp.json()["detail"]

    def test_both_fields_supplied_returns_400(self, client, sample_png_b64):
        resp = client.post(
            "/embed",
            json={"image_b64": sample_png_b64, "image_url": "http://x/y.png"},
        )
        assert resp.status_code == 400
        assert "Exactly one of" in resp.json()["detail"]

    @pytest.mark.unit
    @pytest.mark.ci_fast
    def test_garbage_b64_returns_400(self, client):
        resp = client.post("/embed", json={"image_b64": "@@@ not base64 @@@"})
        assert resp.status_code == 400
        assert "image_b64 decode failed" in resp.json()["detail"]

    def test_undecodable_image_returns_400(self, client):
        not_an_image = base64.b64encode(b"not an image").decode("ascii")
        resp = client.post("/embed", json={"image_b64": not_an_image})
        assert resp.status_code == 400
        assert "image decode failed" in resp.json()["detail"]
