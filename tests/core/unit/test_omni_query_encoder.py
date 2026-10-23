"""EmbeddingGemma 2 query encoding through the profile and service factories.

A test-owned HTTP server stands in for ``vllm_omni_embed`` and records each
request, so the tests pin the exact text the encoder sends.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from cogniverse_core.query.encoders import (
    SERVICE_TEXT_ENCODERS,
    EmbeddingGemma2QueryEncoder,
    QueryEncoderFactory,
)
from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
)
from cogniverse_foundation.config.unified_config import SystemConfig

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

PROFILE = {
    "type": "video",
    "embedding_model": "google/embeddinggemma-2",
    "model_loader": "embeddinggemma2",
    "modalities": ["video", "audio"],
    "embedding_type": "single_vector",
    "inference_services": {"embedding": "vllm_omni_embed"},
    "schema_config": {"embedding_dim": 768, "binary_dim": 96},
}


class _Server:
    def __init__(self) -> None:
        self.texts: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                [part] = body["messages"][0]["content"]
                owner.texts.append(part["text"])
                vector = [0.0] * 768
                vector[len(owner.texts) % 768] = 2.0
                data = json.dumps({"data": [{"index": 0, "embedding": vector}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data.encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def clean_encoder_cache():
    QueryEncoderFactory._encoder_cache.clear()
    yield
    QueryEncoderFactory._encoder_cache.clear()


def _system_config(**urls) -> SystemConfig:
    return SystemConfig(
        backend_url="http://127.0.0.1", backend_port=9, inference_service_urls=urls
    )


def test_profile_factory_builds_the_omni_encoder(clean_encoder_cache):
    with _Server() as server:
        encoder = QueryEncoderFactory.create_for_profile(
            "video_embeddinggemma2_sv_chunk_6s",
            PROFILE,
            _system_config(vllm_omni_embed=server.url),
        )
        vector = encoder.encode("a man lifts a barbell")

    assert type(encoder) is EmbeddingGemma2QueryEncoder
    assert server.texts == ["task: search result | query: a man lifts a barbell"]
    assert int(np.argmax(vector)) == 1
    assert float(vector[1]) == pytest.approx(1.0)
    assert encoder.get_embedding_dim() == 768


def test_service_factory_builds_one_shared_omni_encoder(clean_encoder_cache):
    assert SERVICE_TEXT_ENCODERS["vllm_omni_embed"] is EmbeddingGemma2QueryEncoder
    with _Server() as server:
        config = _system_config(vllm_omni_embed=server.url)
        first = QueryEncoderFactory.create_service_encoder(
            "audio_embeddinggemma2", "vllm_omni_embed", config
        )
        second = QueryEncoderFactory.create_service_encoder(
            "audio_embeddinggemma2", "vllm_omni_embed", config
        )
        first.encode("rain on a tin roof")

    assert first is second
    assert type(first) is EmbeddingGemma2QueryEncoder
    assert server.texts == ["task: search result | query: rain on a tin roof"]


def test_unreachable_service_is_an_unavailable_service(clean_encoder_cache):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    encoder = EmbeddingGemma2QueryEncoder(url)

    with pytest.raises(InferenceServiceUnavailableError) as caught:
        encoder.encode("thunder")

    assert caught.value.service == "vllm_omni_embed"
    assert str(caught.value).startswith(f"vllm_omni_embed at {url} did not answer: ")


def test_encoder_refuses_a_profile_width_the_service_does_not_serve():
    with pytest.raises(ValueError) as caught:
        EmbeddingGemma2QueryEncoder("http://127.0.0.1:1", embedding_dim=512)

    assert str(caught.value) == (
        "vllm_omni_embed serves 768-dim vectors but the profile indexes 512"
    )
