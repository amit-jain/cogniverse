"""Unit tests for the video-embed sidecar.

Imports ``cogniverse_cli.modal_inference.servers.video_embed`` with the heavy
model patched by a deterministic stand-in. The video decode path runs for real
(cv2 over a generated MP4), so the request -> sampled-frames -> processor ->
768-vec contract is exercised end to end in-process.
"""

import base64
import os
import sys
import tempfile
import time as _time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from cogniverse_cli.modal_inference.servers import video_embed as video_embed_server
from fastapi.testclient import TestClient

MODEL = "microsoft/xclip-large-patch14"
REVISION = "a9dd1429a16cf305df2aaea232d5e8dceba1c675"

SOURCE_FRAMES = 16
FRAME_SIZE = 64
# The frames _sample_frames must pick out of a 16-frame clip when the model
# takes 8: numpy.linspace(0, 15, 8) rounded down to int.
EXPECTED_SAMPLED_FRAMES = [0, 2, 4, 6, 8, 10, 12, 15]


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def squeeze(self) -> "_FakeTensor":
        return _FakeTensor(np.squeeze(self._arr))

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeVideoModel:
    """Deterministic stand-in: constant 768-dim outputs."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    def get_video_features(self, **inputs):  # noqa: ARG002
        return _FakeTensor(np.full((1, self._dim), 0.25, dtype=np.float32))

    def get_text_features(self, **inputs):  # noqa: ARG002
        return _FakeTensor(np.full((1, self._dim), -0.5, dtype=np.float32))


class _FakeProcessor:
    """Records what it was called with so tests can pin the sampled frames."""

    def __init__(self) -> None:
        self.video_calls: list = []
        self.text_calls: list = []

    def __call__(self, videos=None, text=None, **_kw):
        if videos is not None:
            self.video_calls.append(videos)
        if text is not None:
            self.text_calls.append(text)
        return {}


@pytest.fixture
def server_module():
    module = video_embed_server
    module._MODEL = _FakeVideoModel()
    module._PROCESSOR = _FakeProcessor()
    return module


@pytest.fixture
def client(server_module):
    return TestClient(server_module.app)


def _mp4_b64(frames: int = SOURCE_FRAMES, size: int = FRAME_SIZE) -> str:
    """A real MP4 whose frame i carries blue=i*16, green=64, red=192.

    The channels are deliberately different: a grey ramp would survive a
    dropped BGR->RGB conversion unnoticed, whereas these levels make the
    channel order observable in the decoded frame. The levels survive the
    codec well enough to round back to exact multiples of 16.
    """
    import cv2

    path = tempfile.mktemp(suffix=".mp4")
    try:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (size, size)
        )
        assert writer.isOpened(), "could not open an mp4 writer for the fixture video"
        for index in range(frames):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            frame[:, :, 0] = index * 16
            frame[:, :, 1] = 64
            frame[:, :, 2] = 192
            writer.write(frame)
        writer.release()
        return base64.b64encode(open(path, "rb").read()).decode()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _channel_levels(frames) -> list[tuple[int, int, int]]:
    """Per-frame (r, g, b) levels in units of 16.

    Frame i was written as blue=i*16, green=64, red=192, so a correctly
    RGB-converted sample reads (12, 4, i) and one still in BGR order reads
    (i, 4, 12).
    """
    return [
        tuple(int(round(np.asarray(frame)[:, :, c].mean() / 16)) for c in range(3))
        for frame in frames
    ]


@pytest.mark.unit
@pytest.mark.ci_fast
def test_health_reports_exact_model_identity(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ready",
        "model": MODEL,
        "model_revision": REVISION,
        "embedding_dim": 768,
        "num_frames": 8,
    }


@pytest.mark.unit
@pytest.mark.ci_fast
def test_concurrent_cold_load_initializes_one_exact_model(monkeypatch, server_module):
    events: list[tuple] = []
    loaded_processor = object()
    simultaneous = Barrier(12)

    class LoadedModel:
        def to(self, device):
            events.append(("device", device))

        def eval(self):
            events.append(("eval",))

    loaded_model = LoadedModel()

    class Processor:
        @staticmethod
        def from_pretrained(model_id, *, revision):
            # Loading a checkpoint is slow; without a pause here the whole
            # load completes inside one GIL slice and the test passes even
            # with the lock removed.
            _time.sleep(0.05)
            events.append(("processor", model_id, revision))
            return loaded_processor

    class Model:
        @staticmethod
        def from_pretrained(model_id, *, revision):
            events.append(("model", model_id, revision))
            return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=Model, AutoProcessor=Processor),
    )
    server_module._MODEL = None
    server_module._PROCESSOR = None
    config = server_module.VideoEmbedConfig(device="cuda")

    def load():
        simultaneous.wait(timeout=3)
        return server_module._load_model(config)

    with ThreadPoolExecutor(max_workers=12) as executor:
        loaded = list(executor.map(lambda _: load(), range(12)))

    assert not simultaneous.broken
    assert events == [
        ("processor", MODEL, REVISION),
        ("model", MODEL, REVISION),
        ("device", "cuda"),
        ("eval",),
    ]
    assert loaded == [(loaded_model, loaded_processor)] * 12


@pytest.mark.unit
@pytest.mark.ci_fast
def test_model_load_failure_propagates_exact_context(
    monkeypatch, client, server_module
):
    class Broken:
        @staticmethod
        def from_pretrained(model_id, *, revision):
            raise OSError("checkpoint unavailable")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=Broken, AutoProcessor=Broken),
    )
    server_module._MODEL = None
    server_module._PROCESSOR = None

    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["detail"] == (
        f"video_embed: model {MODEL} load failed (OSError): checkpoint unavailable"
    )


@pytest.mark.unit
@pytest.mark.ci_fast
def test_model_inference_failure_propagates_exact_context(client, server_module):
    class Exploding:
        def get_text_features(self, **inputs):  # noqa: ARG002
            raise RuntimeError("kernel blew up")

    server_module._MODEL = Exploding()

    resp = client.post("/embed/text", json={"text": "a lion roaring"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == (
        f"video_embed: model {MODEL} inference failed (RuntimeError): kernel blew up"
    )


@pytest.mark.unit
@pytest.mark.ci_fast
def test_embed_video_samples_the_expected_frames_and_returns_768(client, server_module):
    resp = client.post("/embed/video", json={"video_b64": _mp4_b64()})

    assert resp.status_code == 200
    assert resp.json() == {"vec": [0.25] * 768}

    assert len(server_module._PROCESSOR.video_calls) == 1
    (batch,) = server_module._PROCESSOR.video_calls
    assert len(batch) == 1, "the processor takes one clip per request"
    frames = batch[0]
    assert len(frames) == 8, (
        "the model's input length is fixed, so a 16-frame clip must be reduced "
        f"to exactly 8 frames, got {len(frames)}"
    )
    assert _channel_levels(frames) == [(12, 4, i) for i in EXPECTED_SAMPLED_FRAMES], (
        "each sampled frame must be the evenly-spaced source frame, converted "
        "to RGB; (i, 4, 12) triples would mean the frames are still BGR"
    )
    assert [np.asarray(frame).shape for frame in frames] == [
        (FRAME_SIZE, FRAME_SIZE, 3)
    ] * 8, "frames reach the processor unresized; the processor owns the resize"


@pytest.mark.unit
@pytest.mark.ci_fast
def test_embed_text_returns_768_and_passes_the_query(client, server_module):
    resp = client.post("/embed/text", json={"text": "a lion roaring"})

    assert resp.status_code == 200
    assert resp.json() == {"vec": [-0.5] * 768}
    assert server_module._PROCESSOR.text_calls == [["a lion roaring"]]


@pytest.mark.unit
@pytest.mark.ci_fast
def test_embed_video_rejects_invalid_b64(client):
    resp = client.post("/embed/video", json={"video_b64": "not base64!!"})
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("video_b64 decode failed:")


@pytest.mark.unit
@pytest.mark.ci_fast
def test_embed_video_rejects_undecodable_bytes(client):
    resp = client.post(
        "/embed/video",
        json={"video_b64": base64.b64encode(b"not a video").decode()},
    )
    assert resp.status_code == 400
    assert "could not read frames from video" in resp.json()["detail"]


@pytest.mark.unit
@pytest.mark.ci_fast
def test_wrong_embedding_width_is_refused_at_the_sidecar(client, server_module):
    """A checkpoint of the wrong width would be rejected by Vespa far from
    here, with an error that names neither the model nor the endpoint."""
    server_module._MODEL = _FakeVideoModel(dim=512)

    resp = client.post("/embed/text", json={"text": "a lion roaring"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == (
        "video_embed: model emitted 512 dims, expected 768; the served "
        "checkpoint does not match the configured embedding width"
    )
