"""The EmbeddingGemma 2 client: the exact message it posts to the vLLM pooling
server, the vector it returns, and how it fails.

A test-owned HTTP server stands in for ``vllm_omni_embed``, records every
request body, and answers from a per-test script.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import subprocess
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.common.models.model_loaders import (
    OMNI_DOCUMENT_PROMPT,
    OMNI_QUERY_PROMPT,
    ModelLoaderFactory,
    RemoteInferenceClient,
    _extract_audio_wav_b64,
    audio_windows,
)
from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

CORPUS_VIDEO = Path("tests/system/resources/videos/v_-6dz6tBH77I.mp4")


class _OmniServer:
    """Answers ``POST /v1/embeddings`` with ``answer(body) -> (status, payload)``."""

    def __init__(self, answer) -> None:
        self.bodies: list[dict] = []
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                with owner._lock:
                    owner.bodies.append(body)
                status, payload = answer(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_OmniServer":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _vector(*leading: float) -> list[float]:
    return list(leading) + [0.0] * (768 - len(leading))


def _ok(vector: list[float]):
    return 200, {"data": [{"index": 0, "embedding": vector}]}


def _loader(url: str, modalities: list[str]):
    loader = ModelLoaderFactory.create_loader(
        "google/embeddinggemma-2",
        {
            "model_loader": "embeddinggemma2",
            "remote_inference_url": url,
            "modalities": modalities,
        },
    )
    wrapper, _ = loader.load_model()
    return wrapper


def _tone_video(path: Path, seconds: float) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size=64x64:rate=10",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-shortest",
            "-y",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def _silent_track_video(path: Path, seconds: float) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size=64x64:rate=10",
            "-y",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def _wav_samples(b64: str) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(base64.b64decode(b64))) as wav:
        return wav.getframerate(), wav.getnchannels(), wav.getnframes()


def test_document_request_is_the_exact_message(tmp_path):
    video = _tone_video(tmp_path / "tone.mp4", 2.0)
    with _OmniServer(lambda body: _ok(_vector(1.0))) as server:
        _loader(server.url, ["video", "audio"]).process_video_segment(video, 0.0, 2.0)

    [body] = server.bodies
    assert set(body) == {"model", "messages", "add_special_tokens"}
    assert (body["model"], body["add_special_tokens"]) == (
        "google/embeddinggemma-2",
        True,
    )
    [message] = body["messages"]
    assert set(message) == {"role", "content"}
    assert message["role"] == "user"
    content = message["content"]
    assert [part["type"] for part in content] == ["text", "video_url", "input_audio"]
    assert content[0] == {"type": "text", "text": "title: none | text: "}
    assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert content[2]["input_audio"]["format"] == "wav"
    assert _wav_samples(content[2]["input_audio"]["data"]) == (16000, 1, 32000)


def test_a_frames_only_profile_sends_no_audio(tmp_path):
    video = _tone_video(tmp_path / "tone.mp4", 2.0)
    with _OmniServer(lambda body: _ok(_vector(1.0))) as server:
        _loader(server.url, ["video"]).process_video_segment(video, 0.0, 2.0)

    [body] = server.bodies
    assert [part["type"] for part in body["messages"][0]["content"]] == [
        "text",
        "video_url",
    ]


def test_query_text_carries_the_query_prompt():
    with _OmniServer(lambda body: _ok(_vector(1.0))) as server:
        _loader(server.url, ["video", "audio"]).embed_text("a dog barks")

    [body] = server.bodies
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "task: search result | query: a dog barks"}
    ]
    assert OMNI_QUERY_PROMPT == "task: search result | query: "
    assert OMNI_DOCUMENT_PROMPT == "title: none | text: "


def test_vector_is_l2_normalized():
    with _OmniServer(lambda body: _ok(_vector(3.0, 4.0))) as server:
        vector = _loader(server.url, ["audio"]).embed_text("cat")

    assert vector.dtype == np.float32
    assert vector.shape == (768,)
    np.testing.assert_allclose(vector[:3], [0.6, 0.8, 0.0], atol=1e-7)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)


def test_wrong_width_raises():
    with _OmniServer(lambda body: _ok([1.0] * 767)) as server:
        with pytest.raises(ValueError) as caught:
            _loader(server.url, ["audio"]).embed_text("cat")

    assert str(caught.value) == (
        f"vllm_omni_embed at {server.url} returned a 767-dim vector; "
        "the service contract is 768"
    )


def test_audio_windows_golden():
    assert audio_windows(25.0) == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    assert audio_windows(0.4) == [(0.0, 0.4)]
    assert audio_windows(10.0) == [(0.0, 10.0)]
    assert audio_windows(18.065125) == [(0.0, 10.0), (10.0, 18.065125)]


def test_audio_window_of_a_chunk_past_the_end_stops_at_the_end():
    rate, channels, samples = _wav_samples(
        _extract_audio_wav_b64(CORPUS_VIDEO, 5.0, 11.0)
    )

    assert (rate, channels) == (16000, 1)
    assert abs(samples - round((8.057324 - 5.0) * 16000)) <= 160


def test_audio_window_of_a_video_without_an_audio_stream_raises(tmp_path):
    video = _silent_track_video(tmp_path / "no_audio.mp4", 1.0)

    with pytest.raises(ValueError) as caught:
        _extract_audio_wav_b64(video, 0.0, 1.0)

    assert str(caught.value) == f"{video} has no audio stream to embed"


def test_concurrent_embeds_land_on_their_own_text():
    def answer(body):
        text = body["messages"][0]["content"][0]["text"]
        index = int(text.rsplit(" ", 1)[1])
        return _ok(_vector(*([0.0] * index), 1.0))

    with _OmniServer(answer) as server:
        wrapper = _loader(server.url, ["audio"])
        barrier = threading.Barrier(8)
        results: dict[int, np.ndarray] = {}

        def embed(index: int) -> None:
            barrier.wait()
            results[index] = wrapper.embed_text(f"item {index}")

        threads = [threading.Thread(target=embed, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert sorted(results) == list(range(8))
    assert {index: int(np.argmax(vector)) for index, vector in results.items()} == {
        index: index for index in range(8)
    }
    assert len(server.bodies) == 8


def _dead_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_down_service_raises_naming_the_service():
    url = f"http://127.0.0.1:{_dead_port()}"
    client = RemoteInferenceClient(url)

    with pytest.raises(InferenceServiceUnavailableError) as caught:
        client.embed_messages([{"type": "text", "text": "x"}], timeout=2.0)

    assert caught.value.service == "vllm_omni_embed"
    assert str(caught.value).startswith(f"vllm_omni_embed at {url} did not answer: ")


def test_a_5xx_raises_with_the_server_body():
    def answer(body):
        return 500, {"error": "engine dead"}

    with _OmniServer(answer) as server:
        client = RemoteInferenceClient(server.url)
        with pytest.raises(InferenceServiceUnavailableError) as caught:
            client.embed_messages([{"type": "text", "text": "x"}], timeout=2.0)

    assert caught.value.service == "vllm_omni_embed"
    assert str(caught.value) == (
        f'vllm_omni_embed at {server.url} failed (500): {{"error": "engine dead"}}'
    )
    assert len(server.bodies) == 3


def test_a_400_is_not_retried_and_carries_the_server_body():
    def answer(body):
        return 400, {"error": "prompt exceeds 8192 tokens"}

    with _OmniServer(answer) as server:
        client = RemoteInferenceClient(server.url)
        with pytest.raises(ValueError) as caught:
            client.embed_messages([{"type": "text", "text": "x"}], timeout=2.0)

    assert str(caught.value) == (
        f"vllm_omni_embed at {server.url} rejected the request (400): "
        '{"error": "prompt exceeds 8192 tokens"}'
    )
    assert len(server.bodies) == 1


def test_a_hung_service_times_out_within_the_retry_budget():
    def answer(body):
        time.sleep(3.0)
        return _ok(_vector(1.0))

    with _OmniServer(answer) as server:
        client = RemoteInferenceClient(server.url)
        started = time.monotonic()
        with pytest.raises(InferenceServiceUnavailableError) as caught:
            client.embed_messages([{"type": "text", "text": "x"}], timeout=0.5)
        elapsed = time.monotonic() - started

    assert caught.value.service == "vllm_omni_embed"
    # Three 0.5 s attempts plus at most 0.5 s and 1.0 s of backoff (+ jitter).
    assert elapsed < 4.5
    assert len(server.bodies) == 3


def test_unknown_modalities_are_refused_at_load():
    with pytest.raises(ValueError) as caught:
        _loader("http://127.0.0.1:1", ["video", "smell"])

    assert str(caught.value) == (
        "embeddinggemma2 modalities must be a non-empty subset of "
        "['audio', 'video'], got ['video', 'smell']"
    )
