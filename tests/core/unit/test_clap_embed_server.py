"""Unit tests for the CLAP-embed sidecar.

Imports ``cogniverse_cli.modal_inference.servers.clap`` with the heavy model
patched by a deterministic stand-in. The audio decode path runs for real
(librosa over a generated WAV), so the request → 48 kHz-mono-array →
processor → 512-vec contract is exercised end to end in-process.
"""

import base64
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from cogniverse_cli.modal_inference.servers import clap as clap_embed_server
from fastapi.testclient import TestClient


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def squeeze(self) -> "_FakeTensor":
        return _FakeTensor(np.squeeze(self._arr))

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeClapModel:
    """Deterministic CLAP stand-in: constant 512-dim outputs."""

    def get_audio_features(self, **inputs):  # noqa: ARG002
        return _FakeTensor(np.full((1, 512), 0.25, dtype=np.float32))

    def get_text_features(self, **inputs):  # noqa: ARG002
        return _FakeTensor(np.full((1, 512), -0.5, dtype=np.float32))


class _FakeClapProcessor:
    """Records what it was called with so tests can pin the audio shape."""

    def __init__(self) -> None:
        self.audio_calls: list = []
        self.text_calls: list = []

    def __call__(self, audios=None, text=None, sampling_rate=None, **_kw):
        if audios is not None:
            self.audio_calls.append((np.asarray(audios), sampling_rate))
        if text is not None:
            self.text_calls.append(text)
        return {}


@pytest.fixture
def server_module():
    mod = clap_embed_server
    mod._MODEL = _FakeClapModel()
    mod._PROCESSOR = _FakeClapProcessor()
    return mod


@pytest.fixture
def client(server_module):
    return TestClient(server_module.app)


def _wav_b64(duration_s: float = 0.5, sr: int = 16000) -> str:
    """A real 16 kHz sine-tone WAV, base64-encoded."""
    import soundfile as sf

    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.unit
@pytest.mark.ci_fast
def test_health_reports_exact_model_identity(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ready",
        "model": "laion/clap-htsat-unfused",
        "model_revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
    }


@pytest.mark.unit
@pytest.mark.ci_fast
def test_concurrent_cold_load_initializes_one_exact_model(
    monkeypatch,
    server_module,
):
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
        SimpleNamespace(ClapModel=Model, ClapProcessor=Processor),
    )
    server_module._MODEL = None
    server_module._PROCESSOR = None
    config = server_module.ClapEmbedConfig(device="cuda")

    def load():
        simultaneous.wait(timeout=3)
        return server_module._load_model(config)

    with ThreadPoolExecutor(max_workers=12) as executor:
        loaded = list(executor.map(lambda _: load(), range(12)))

    revision = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
    assert not simultaneous.broken
    assert events == [
        ("processor", "laion/clap-htsat-unfused", revision),
        ("model", "laion/clap-htsat-unfused", revision),
        ("device", "cuda"),
        ("eval",),
    ]
    assert loaded == [(loaded_model, loaded_processor)] * 12


@pytest.mark.unit
@pytest.mark.ci_fast
def test_model_load_failure_propagates_exact_context(
    monkeypatch,
    client,
    server_module,
):
    class BrokenArtifact:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("checkpoint is truncated")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(ClapModel=BrokenArtifact, ClapProcessor=BrokenArtifact),
    )
    server_module._MODEL = None
    server_module._PROCESSOR = None

    response = client.post("/embed/text", json={"text": "rain on a roof"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "clap_embed: model laion/clap-htsat-unfused load failed "
            "(OSError): checkpoint is truncated"
        )
    }


@pytest.mark.unit
@pytest.mark.ci_fast
def test_model_inference_failure_propagates_exact_context(client, server_module):
    class BrokenModel:
        def get_text_features(self, **inputs):
            raise RuntimeError("device execution failed")

    server_module._MODEL = BrokenModel()
    server_module._PROCESSOR = lambda **kwargs: {}

    response = client.post("/embed/text", json={"text": "rain on a roof"})

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "clap_embed: model laion/clap-htsat-unfused inference failed "
            "(RuntimeError): device execution failed"
        )
    }


@pytest.mark.unit
@pytest.mark.ci_fast
def test_embed_text_returns_exact_512_vector(client):
    resp = client.post("/embed/text", json={"text": "rain on a tin roof"})
    assert resp.status_code == 200
    vec = resp.json()["vec"]
    assert vec == [-0.5] * 512


def test_embed_text_passes_query_to_processor(client, server_module):
    client.post("/embed/text", json={"text": "dog barking"})
    assert server_module._PROCESSOR.text_calls == [["dog barking"]]


def test_embed_audio_decodes_resamples_and_returns_512(client, server_module):
    resp = client.post("/embed/audio", json={"audio_b64": _wav_b64()})
    assert resp.status_code == 200
    assert resp.json()["vec"] == [0.25] * 512

    # The 0.5 s 16 kHz tone must reach the processor as a mono float
    # array resampled to the configured 48 kHz: exactly 24000 samples.
    (audio_arr, sampling_rate) = server_module._PROCESSOR.audio_calls[0]
    assert sampling_rate == 48000
    assert audio_arr.ndim == 1
    assert audio_arr.shape[0] == 24000


@pytest.mark.unit
@pytest.mark.ci_fast
def test_embed_audio_rejects_invalid_b64(client):
    resp = client.post("/embed/audio", json={"audio_b64": "not-base64!!"})
    assert resp.status_code == 400
    assert "audio_b64 decode failed" in resp.json()["detail"]


@pytest.mark.filterwarnings("ignore:PySoundFile failed:UserWarning")
@pytest.mark.filterwarnings("ignore:librosa.core.audio.__audioread_load:FutureWarning")
def test_embed_audio_rejects_undecodable_bytes(client):
    junk = base64.b64encode(b"definitely not audio bytes").decode()
    resp = client.post("/embed/audio", json={"audio_b64": junk})
    assert resp.status_code == 400
    assert "audio decode failed" in resp.json()["detail"]
