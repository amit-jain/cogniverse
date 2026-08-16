"""Round-trip: AudioEmbeddingGenerator's remote path ↔ clap_embed sidecar.

Runs the real sidecar app over HTTP (uvicorn thread) with a deterministic
CLAP stand-in, and drives the real client serialisation: file-path bytes,
in-memory array → WAV, and text queries. The boundary under test is the
HTTP contract — request shapes, audio decode/resample, exact 512-dim
responses — which the in-process unit tests cannot prove.
"""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests
from cogniverse_cli.modal_inference.servers import clap as clap_embed_server

from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)

pytestmark = pytest.mark.integration


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
        self.text_calls = []

    def __call__(self, audios=None, text=None, sampling_rate=None, **_kw):
        if audios is not None:
            self.audio_calls.append((np.asarray(audios), sampling_rate))
        if text is not None:
            self.text_calls.append(text)
        return {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def clap_sidecar():
    """The real sidecar module served over real HTTP, model stubbed."""
    import uvicorn

    mod = clap_embed_server
    mod._MODEL = _FakeClapModel()
    mod._PROCESSOR = _FakeClapProcessor()
    mod._RECEIVED_AUTHORIZATIONS = []
    app = mod.build_app(mod.ClapEmbedConfig())

    @app.middleware("http")
    async def capture_authorization(request, call_next):
        if request.url.path.startswith("/embed/"):
            mod._RECEIVED_AUTHORIZATIONS.append(request.headers.get("Authorization"))
        return await call_next(request)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
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


def test_concurrent_text_calls_share_one_authenticated_client(clap_sidecar):
    token = "modal-clap-key"
    module = clap_sidecar["module"]
    module._RECEIVED_AUTHORIZATIONS.clear()
    module._PROCESSOR.text_calls.clear()
    generator = AudioEmbeddingGenerator(
        clap_endpoint_url=clap_sidecar["url"],
        clap_headers={"Authorization": f"Bearer {token}"},
    )
    callers = threading.Barrier(12)
    texts = [f"acoustic query {index}" for index in range(12)]

    def encode(text):
        callers.wait(timeout=5)
        vector = generator.generate_acoustic_text_embedding(text)
        return id(generator._get_http_client()), vector.tolist()

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(encode, texts))

    assert not callers.broken
    assert {client_id for client_id, _ in results} == {id(generator._http_client)}
    assert [vector for _, vector in results] == [[-0.5] * 512] * 12
    assert module._RECEIVED_AUTHORIZATIONS == [f"Bearer {token}"] * 12
    assert sorted(module._PROCESSOR.text_calls) == sorted([[text] for text in texts])
    generator.close()
    assert generator._http_client is None


def test_connection_refusal_includes_endpoint_without_credential():
    token = "modal-clap-key"
    refused_url = f"http://127.0.0.1:{_free_port()}"
    generator = AudioEmbeddingGenerator(
        clap_endpoint_url=refused_url,
        clap_headers={"Authorization": f"Bearer {token}"},
    )

    with pytest.raises(RuntimeError) as caught:
        generator.generate_acoustic_text_embedding("rain")

    message = str(caught.value)
    assert message.startswith(
        f"CLAP request to {refused_url}/embed/text failed: ConnectionError:"
    )
    assert "Connection refused" in message
    assert token not in message
