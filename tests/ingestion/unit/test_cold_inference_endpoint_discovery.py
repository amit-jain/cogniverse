"""Model-id discovery survives a cold scale-to-zero inference endpoint.

Ingestion asked ``/v1/models`` for the served model id with a 10s read timeout
and no retry. A Modal deployment that has scaled to zero takes longer than that
to boot, so every ingest job that arrived while the endpoint was cold died in
discovery — the exact moment production ingestion first needs it. Discovery now
runs against the service's cold-start budget, caches the answer per process,
and still fails loudly, naming the endpoint, when nothing ever answers.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from cogniverse_runtime.ingestion.processors.served_model import ServedModelUnavailable
from cogniverse_runtime.ingestion.processors.vlm_descriptor import VLMDescriptor

pytestmark = [pytest.mark.unit]

# The read timeout the probe used to carry. A cold Modal container takes longer
# than this to answer, so the stand-in endpoint delays past it.
OLD_PROBE_TIMEOUT_SECONDS = 10.0
COLD_BOOT_SECONDS = 11.0


class _Endpoint(ThreadingHTTPServer):
    """A vLLM ``/v1`` stand-in whose first ``/models`` answer can be delayed,
    failed, or withheld entirely."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        handler,
        *,
        model_id: str = "served-model",
        first_answer_delay: float = 0.0,
        fail_first_n: int = 0,
        never_answer: bool = False,
        models_status: int = 200,
    ):
        super().__init__(("127.0.0.1", 0), handler)
        self.lock = threading.Lock()
        self.model_id = model_id
        self.first_answer_delay = first_answer_delay
        self.fail_first_n = fail_first_n
        self.never_answer = never_answer
        self.models_status = models_status
        self.models_request_count = 0
        self.chat_models_seen: list[str] = []
        self.transcribe_models_seen: list[str] = []
        self.released = threading.Event()
        self.probe_arrived = threading.Event()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _EndpointHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.path.endswith("/models"):
            self.send_error(404)
            return
        server = self.server
        with server.lock:
            server.models_request_count += 1
            nth = server.models_request_count
        server.probe_arrived.set()
        if server.never_answer:
            server.released.wait(60)
            return
        if server.models_status != 200:
            self._json({"error": "not here"}, status=server.models_status)
            return
        if nth <= server.fail_first_n:
            self._json({"error": "engine still loading"}, status=500)
            return
        if nth == 1 and server.first_answer_delay:
            time.sleep(server.first_answer_delay)
        self._json({"data": [{"id": server.model_id}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        server = self.server
        if self.path.endswith("/chat/completions"):
            payload = json.loads(body)
            with server.lock:
                server.chat_models_seen.append(payload["model"])
            parts = payload["messages"][0]["content"]
            url = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
            frame = base64.b64decode(url.split("base64,", 1)[1]).decode()
            self._json({"choices": [{"message": {"content": f"desc for {frame}"}}]})
            return
        if self.path.endswith("/audio/transcriptions"):
            marker = b'name="model"\r\n\r\n'
            start = body.index(marker) + len(marker)
            model = body[start : body.index(b"\r\n", start)].decode()
            with server.lock:
                server.transcribe_models_seen.append(model)
            self._json(
                {
                    "text": " hello cold world ",
                    "language": "en",
                    "duration": 2.0,
                    "segments": [{"start": 0.0, "end": 2.0, "text": " hello cold "}],
                }
            )
            return
        self.send_error(404)


@pytest.fixture
def endpoint():
    servers: list[_Endpoint] = []

    def start(**kwargs) -> _Endpoint:
        server = _Endpoint(_EndpointHandler, **kwargs)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    yield start
    for server in servers:
        server.released.set()
        server.shutdown()
        server.server_close()


def _keyframes(tmp_path: Path, count: int, prefix: str = "f") -> list[dict]:
    frames = []
    for i in range(count):
        path = tmp_path / f"{prefix}{i}.jpg"
        path.write_bytes(f"{prefix}{i}".encode())
        frames.append({"frame_id": f"{prefix}{i}", "path": str(path)})
    return frames


@pytest.mark.slow
class TestColdEndpointDiscovery:
    def test_cold_models_answer_after_the_old_timeout_still_describes_frames(
        self, tmp_path, endpoint
    ):
        """The first ``/models`` answer lands after the 10s the probe used to
        allow; the job proceeds and the chat call carries the served id."""
        server = endpoint(
            model_id="google/gemma-4-e4b-it", first_answer_delay=COLD_BOOT_SECONDS
        )
        descriptor = VLMDescriptor(
            vlm_endpoint=f"{server.base}/v1", batch_size=500, timeout=60
        )

        started = time.monotonic()
        result = descriptor.generate_descriptions(
            {"video_id": "cold_video", "keyframes": _keyframes(tmp_path, 2)},
            output_dir=tmp_path,
        )
        elapsed = time.monotonic() - started

        assert elapsed > OLD_PROBE_TIMEOUT_SECONDS
        assert result["video_id"] == "cold_video"
        assert result["descriptions"] == {"f0": "desc for f0", "f1": "desc for f1"}
        assert result["total_descriptions"] == 2
        assert server.chat_models_seen == ["google/gemma-4-e4b-it"] * 2
        assert server.models_request_count == 1
        assert json.loads(
            (tmp_path / "descriptions" / "cold_video.json").read_text()
        ) == {
            "f0": "desc for f0",
            "f1": "desc for f1",
        }

    def test_endpoint_that_never_answers_fails_the_job_naming_the_endpoint(
        self, tmp_path, endpoint
    ):
        server = endpoint(never_answer=True)
        descriptor = VLMDescriptor(
            vlm_endpoint=f"{server.base}/v1",
            batch_size=500,
            timeout=60,
            model_discovery_deadline_seconds=1.0,
            model_discovery_retry_interval_seconds=0.05,
        )

        with pytest.raises(ServedModelUnavailable) as raised:
            descriptor.generate_descriptions(
                {"video_id": "dead_video", "keyframes": _keyframes(tmp_path, 1)},
                output_dir=tmp_path,
            )

        message = str(raised.value)
        assert f"{server.base}/v1" in message
        assert "vllm_llm_student" in message
        assert "did not report a served model id within 1s" in message
        assert server.chat_models_seen == []
        assert not (tmp_path / "descriptions" / "dead_video.json").exists()

    def test_models_probe_retries_through_a_server_error(self, tmp_path, endpoint):
        """Fault contract: a 500 from a half-booted engine is retried, not fatal."""
        server = endpoint(model_id="served-after-500", fail_first_n=1)
        descriptor = VLMDescriptor(
            vlm_endpoint=f"{server.base}/v1",
            batch_size=500,
            timeout=60,
            model_discovery_retry_interval_seconds=0.05,
        )

        result = descriptor._process_vlm_batch(_keyframes(tmp_path, 1))

        assert result == {"f0": "desc for f0"}
        assert server.models_request_count == 2
        assert server.chat_models_seen == ["served-after-500"]

    def test_a_status_waiting_cannot_repair_fails_on_the_first_attempt(
        self, tmp_path, endpoint
    ):
        """A 404 means the URL is not an OpenAI ``/v1`` root — retrying it for
        the whole cold-start budget would stall the job for nothing."""
        server = endpoint(models_status=404)
        descriptor = VLMDescriptor(
            vlm_endpoint=f"{server.base}/v1",
            batch_size=500,
            timeout=60,
            model_discovery_deadline_seconds=600.0,
            model_discovery_retry_interval_seconds=30.0,
        )

        started = time.monotonic()
        with pytest.raises(ServedModelUnavailable) as raised:
            descriptor._process_vlm_batch(_keyframes(tmp_path, 1))
        elapsed = time.monotonic() - started

        assert elapsed < 5.0
        assert server.models_request_count == 1
        assert str(raised.value) == (
            f"Inference endpoint {server.base}/v1 (vllm_llm_student) answered "
            f"404 for {server.base}/v1/models; waiting cannot repair that, so "
            f"discovery is not retried"
        )

    def test_two_concurrent_jobs_resolve_the_model_id_once(self, tmp_path, endpoint):
        """Concurrency invariant: two jobs against one endpoint issue a single
        ``/models`` GET — the second waits on the in-flight discovery rather
        than launching its own."""
        server = endpoint(model_id="shared-id", first_answer_delay=0.5)
        jobs = [
            VLMDescriptor(vlm_endpoint=f"{server.base}/v1", batch_size=500, timeout=60),
            VLMDescriptor(vlm_endpoint=f"{server.base}/v1", batch_size=500, timeout=60),
        ]
        frames = {
            0: _keyframes(tmp_path, 1, prefix="a"),
            1: _keyframes(tmp_path, 1, prefix="b"),
        }
        barrier = threading.Barrier(2)
        results: dict[int, dict] = {}

        def run(index: int):
            barrier.wait(10)
            results[index] = jobs[index]._process_vlm_batch(frames[index])

        threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)

        assert results == {0: {"a0": "desc for a0"}, 1: {"b0": "desc for b0"}}
        assert server.models_request_count == 1
        assert sorted(server.chat_models_seen) == ["shared-id", "shared-id"]
        assert jobs[0]._openai_model == "shared-id"
        assert jobs[1]._openai_model == "shared-id"

    def test_queued_jobs_share_one_dead_endpoints_verdict(self, tmp_path, endpoint):
        """Two jobs waiting on the same dead endpoint spend one cold-start
        budget between them, not one each, and both name the endpoint."""
        server = endpoint(never_answer=True)
        jobs = [
            VLMDescriptor(
                vlm_endpoint=f"{server.base}/v1",
                batch_size=500,
                timeout=60,
                model_discovery_deadline_seconds=2.0,
                model_discovery_retry_interval_seconds=0.05,
            )
            for _ in range(2)
        ]
        frames = {
            0: _keyframes(tmp_path, 1, prefix="a"),
            1: _keyframes(tmp_path, 1, prefix="b"),
        }
        barrier = threading.Barrier(2)
        failures: dict[int, Exception] = {}

        def run(index: int):
            barrier.wait(10)
            try:
                jobs[index]._process_vlm_batch(frames[index])
            except Exception as exc:
                failures[index] = exc

        threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
        started = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        elapsed = time.monotonic() - started

        assert sorted(failures) == [0, 1]
        for index in (0, 1):
            assert isinstance(failures[index], ServedModelUnavailable)
            assert f"{server.base}/v1 (vllm_llm_student)" in str(failures[index])
            assert "did not report a served model id within 2s" in str(failures[index])
        assert server.models_request_count == 1
        assert elapsed < 3.0
        assert server.chat_models_seen == []


@pytest.mark.slow
class TestColdAsrEndpointDiscovery:
    def test_cold_asr_endpoint_transcribes_with_the_served_model_id(
        self, tmp_path, endpoint, monkeypatch
    ):
        """The ASR probe carried the same 10s timeout and swallowed its failure,
        posting the configured name instead of the served one."""
        server = endpoint(
            model_id="openai/whisper-large-v3-turbo",
            first_answer_delay=COLD_BOOT_SECONDS,
        )
        monkeypatch.setattr(
            AudioProcessor, "_extract_audio_wav", staticmethod(lambda p: b"RIFF")
        )
        processor = AudioProcessor(
            logging.getLogger("test"),
            model="base",
            language="en",
            endpoint=server.base,
        )

        transcript = processor._transcribe_remote(Path("clip.mp4"), "clip")

        assert server.transcribe_models_seen == ["openai/whisper-large-v3-turbo"]
        assert transcript == {
            "video_id": "clip",
            "video_path": "clip.mp4",
            "model": "openai/whisper-large-v3-turbo",
            "language": "en",
            "duration": 2.0,
            "full_text": "hello cold world",
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello cold"}],
        }
