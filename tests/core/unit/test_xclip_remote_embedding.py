"""X-CLIP ingestion and query embeddings speak the video_embed contract.

Driven through the real ``RemoteInferenceClient`` against a local emulator of
the sidecar's HTTP API, so route, payload and response handling are validated
on the same requests path production uses rather than a mock of the client.

The emulator parses each request with the SIDECAR'S OWN pydantic models and
answers with its response model, so renaming a field in
``cogniverse_cli.modal_inference.servers.video_embed`` fails these tests
instead of leaving an emulator that agrees only with itself.
"""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from cogniverse_cli.modal_inference.servers.video_embed import (
    EmbedResponse,
    TextEmbedRequest,
    VideoEmbedConfig,
    VideoEmbedRequest,
)
from cogniverse_core.common.models.model_loaders import RemoteInferenceClient
from cogniverse_core.query.encoders import QueryEncoderFactory, XClipQueryEncoder

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

DIM = VideoEmbedConfig().embedding_dim


def _vector_for(seed: int) -> list[float]:
    """A deterministic vector distinct per seed, so cross-talk is detectable."""
    return [float((seed + i) % 97) for i in range(DIM)]


def _dead_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _SidecarEmulator(http.server.BaseHTTPRequestHandler):
    """Models /embed/video and /embed/text of the video_embed sidecar."""

    status_override: int | None = None
    empty_vector: bool = False
    seen_routes: list[str] = []

    def log_message(self, *args):  # noqa: D102 - silence the default stderr log
        pass

    def _reply(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's required name
        length = int(self.headers.get("Content-Length", 0))
        raw = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen_routes.append(self.path)

        if type(self).status_override is not None:
            self._reply(type(self).status_override, {"detail": "sidecar unavailable"})
            return

        if self.path == "/embed/video":
            # Parsed by the sidecar's own model: a renamed field fails here.
            request = VideoEmbedRequest(**raw)
            seed = len(request.video_b64) % 13
        elif self.path == "/embed/text":
            request = TextEmbedRequest(**raw)
            seed = sum(request.text.encode()) % 13
        else:
            self._reply(404, {"detail": f"no route {self.path}"})
            return

        vec = [] if type(self).empty_vector else _vector_for(seed)
        self._reply(200, EmbedResponse(vec=vec).model_dump())


@pytest.fixture
def sidecar():
    """Run the emulator on its own port and yield its base URL."""
    _SidecarEmulator.status_override = None
    _SidecarEmulator.empty_vector = False
    _SidecarEmulator.seen_routes = []

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SidecarEmulator)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A real 2-second video, so the ffmpeg segment cut runs for real."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_video_segment_embedding_posts_the_sidecar_route(sidecar, clip):
    client = RemoteInferenceClient(sidecar)
    vector = client.embed_video_segment(clip, 0.0, 1.0)

    assert _SidecarEmulator.seen_routes == ["/embed/video"]
    assert vector.dtype == np.float32
    assert vector.shape == (DIM,)


def test_text_embedding_returns_the_exact_vector_the_service_sent(sidecar):
    client = RemoteInferenceClient(sidecar)
    vector = client.embed_text("a person riding a bicycle")

    expected = _vector_for(sum(b"a person riding a bicycle") % 13)
    assert _SidecarEmulator.seen_routes == ["/embed/text"]
    assert vector.tolist() == expected


def test_query_encoder_encodes_through_the_same_service(sidecar):
    encoder = XClipQueryEncoder(
        "microsoft/xclip-large-patch14", inference_service_url=sidecar
    )
    vector = encoder.encode("a dog catching a frisbee")

    expected = _vector_for(sum(b"a dog catching a frisbee") % 13)
    assert vector.tolist() == expected
    assert encoder.get_embedding_dim() == DIM


def test_query_encoder_refuses_a_width_the_profile_does_not_index(sidecar):
    """A vector of the wrong width would score against a different space."""
    encoder = XClipQueryEncoder(
        "microsoft/xclip-large-patch14",
        inference_service_url=sidecar,
        embedding_dim=DIM + 1,
    )
    with pytest.raises(ValueError, match="different space"):
        encoder.encode("anything")


def test_service_outage_raises_rather_than_returning_no_vector(sidecar):
    """An outage must not read as a document with no content."""
    _SidecarEmulator.status_override = 503
    client = RemoteInferenceClient(sidecar)

    with pytest.raises(Exception) as excinfo:
        client.embed_text("query during an outage")
    assert "503" in str(excinfo.value)


def test_empty_vector_response_raises(sidecar):
    _SidecarEmulator.empty_vector = True
    client = RemoteInferenceClient(sidecar)

    with pytest.raises(ValueError, match="returned no vector"):
        client.embed_text("query answered with nothing")


def test_unreachable_service_raises(sidecar):
    client = RemoteInferenceClient(f"http://127.0.0.1:{_dead_port()}")

    with pytest.raises(Exception):
        client.embed_text("query to a dead port")


def test_concurrent_queries_do_not_cross_talk(sidecar):
    """One shared client under N threads returns each caller its OWN vector."""
    client = RemoteInferenceClient(sidecar)
    queries = [f"query number {i}" for i in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(client.embed_text, queries))

    expected = [_vector_for(sum(q.encode()) % 13) for q in queries]
    assert [r.tolist() for r in results] == expected


def test_factory_routes_the_xclip_profile_to_the_xclip_encoder():
    """The shipped profile's model_loader picks the joint-space encoder."""

    class _SystemConfig:
        inference_service_urls = {"video_embed": "http://video-embed.invalid:8000"}

    encoder = QueryEncoderFactory._create_encoder_instance(
        model_name="microsoft/xclip-large-patch14",
        profile="video_xclip_sv_chunk_6s",
        profile_config={
            "model_loader": "xclip",
            "inference_services": {"embedding": "video_embed"},
            "schema_config": {"embedding_dim": DIM},
        },
        system_config=_SystemConfig(),
    )

    assert isinstance(encoder, XClipQueryEncoder)
    assert encoder.get_embedding_dim() == DIM
