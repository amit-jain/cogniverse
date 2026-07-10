"""Unit tests for the CLAP-embed sidecar.

Loads ``cogniverse_runtime/sidecars/clap_embed.py`` with the heavy model
patched by a deterministic stand-in. The audio decode path runs for real
(librosa over a generated WAV), so the request → 48 kHz-mono-array →
processor → 512-vec contract is exercised end to end in-process.
"""

import base64
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

SERVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "libs/runtime/cogniverse_runtime/sidecars/clap_embed.py"
)


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
    spec = importlib.util.spec_from_file_location(
        "clap_embed_server_under_test", str(SERVER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


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


def test_embed_audio_rejects_undecodable_bytes(client):
    junk = base64.b64encode(b"definitely not audio bytes").decode()
    resp = client.post("/embed/audio", json={"audio_b64": junk})
    assert resp.status_code == 400
    assert "audio decode failed" in resp.json()["detail"]
