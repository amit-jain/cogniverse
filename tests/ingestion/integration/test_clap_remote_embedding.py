"""Round-trip: AudioEmbeddingGenerator's remote path ↔ clap_embed sidecar.

Runs the real sidecar app over HTTP (uvicorn thread) with a deterministic
CLAP stand-in, and drives the real client serialisation: file-path bytes,
in-memory array → WAV, and text queries. The boundary under test is the
HTTP contract — request shapes, audio decode/resample, exact 512-dim
responses — which the in-process unit tests cannot prove.
"""

import importlib.util
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)

SERVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "libs/runtime/cogniverse_runtime/sidecars/clap_embed.py"
)


class _FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def squeeze(self):
        return _FakeTensor(np.squeeze(self._arr))

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeClapModel:
    def get_audio_features(self, **_kw):
        return _FakeTensor(np.full((1, 512), 0.25, dtype=np.float32))

    def get_text_features(self, **_kw):
        return _FakeTensor(np.full((1, 512), -0.5, dtype=np.float32))


class _FakeClapProcessor:
    def __init__(self):
        self.audio_calls = []

    def __call__(self, audios=None, text=None, sampling_rate=None, **_kw):
        if audios is not None:
            self.audio_calls.append((np.asarray(audios), sampling_rate))
        return {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def clap_sidecar():
    """The real sidecar module served over real HTTP, model stubbed."""
    import uvicorn

    spec = importlib.util.spec_from_file_location(
        "clap_embed_under_test", str(SERVER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._MODEL = _FakeClapModel()
    mod._PROCESSOR = _FakeClapProcessor()

    port = _free_port()
    config = uvicorn.Config(mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        pytest.fail("clap_embed sidecar did not come up within 30s")

    try:
        yield {"url": base_url, "module": mod}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_remote_text_embedding_round_trip(clap_sidecar):
    gen = AudioEmbeddingGenerator(clap_endpoint_url=clap_sidecar["url"])
    vec = gen.generate_acoustic_text_embedding("rain on a tin roof")
    assert vec.dtype == np.float32
    assert vec.shape == (512,)
    assert vec.tolist() == [-0.5] * 512


def test_remote_audio_path_round_trip(clap_sidecar, tmp_path):
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    wav = tmp_path / "tone.wav"
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

    gen = AudioEmbeddingGenerator(clap_endpoint_url=clap_sidecar["url"])
    vec = gen.generate_acoustic_embedding(audio_path=wav)
    assert vec.tolist() == [0.25] * 512

    # The sidecar must have resampled the 0.5 s 16 kHz file to 48 kHz mono.
    audio_arr, sampling_rate = clap_sidecar["module"]._PROCESSOR.audio_calls[-1]
    assert sampling_rate == 48000
    assert audio_arr.shape == (24000,)


def test_remote_audio_array_round_trip(clap_sidecar):
    sr = 48000
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.25, sr // 4))).astype(
        np.float32
    )

    gen = AudioEmbeddingGenerator(clap_endpoint_url=clap_sidecar["url"])
    vec = gen.generate_acoustic_embedding(audio_array=tone, sample_rate=sr)
    assert vec.tolist() == [0.25] * 512

    audio_arr, sampling_rate = clap_sidecar["module"]._PROCESSOR.audio_calls[-1]
    assert sampling_rate == 48000
    assert audio_arr.shape == (sr // 4,)
