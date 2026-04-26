"""End-to-end round-trip of the remote Whisper transcription path.

Spins up a real HTTP stub that mimics the ``deploy/whisper`` sidecar's
``/v1/transcribe`` contract, routes a tiny audio file through
``AudioTranscriptionStrategy`` → ``ProcessorManager`` URL substitution →
``AudioProcessor.transcribe_audio``, and asserts both the request shape
sent to the pod and the response shape returned to the pipeline.

This is the wiring-correctness test required by CLAUDE.md — it would
catch:
- A strategy that drops ``inference_service`` so the URL never gets
  resolved.
- A processor-manager substitution that picks the wrong service or
  forgets to ``pop`` the placeholder.
- An AudioProcessor remote path that ships the wrong body shape
  (e.g. raw bytes instead of base64) or fails to parse the canonical
  ``deploy/whisper`` response back into the dict the pipeline expects.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import (
    ProcessingStrategySet,
)
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from cogniverse_runtime.ingestion.strategies import AudioTranscriptionStrategy


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _WhisperStub(BaseHTTPRequestHandler):
    """Mimics ``deploy/whisper/server.py`` for the round-trip test.

    Records each POST so the test can assert the body shape, then replies
    with a fixed transcript matching the ``TranscribeResponse`` schema.
    """

    captured_requests: list[dict] = []  # class-level, reset per test

    def log_message(self, format, *args):  # silence stderr
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = json.dumps(
                {"status": "ok", "model": "stub-base", "engine": "stub"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/v1/transcribe":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        _WhisperStub.captured_requests.append(body)

        # Echo back the canonical response shape from deploy/whisper/server.py
        resp = {
            "text": "hello world",
            "language": body.get("language") or "en",
            "duration_seconds": 1.5,
            "processing_time": 0.42,
            "model": "stub-base",
            "segments": [
                {"start": 0.0, "end": 0.7, "text": "hello"},
                {"start": 0.7, "end": 1.5, "text": " world"},
            ],
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def stub_whisper():
    _WhisperStub.captured_requests = []
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _WhisperStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", _WhisperStub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_strategy_to_pod_roundtrip(stub_whisper, tmp_path):
    """End-to-end: strategy declares whisper, manager resolves URL, processor POSTs."""
    url, stub = stub_whisper

    # Audio file with arbitrary bytes; the stub doesn't run a real model.
    audio_path = tmp_path / "clip.wav"
    audio_bytes = b"FAKE-WAV-PAYLOAD-" + b"x" * 100
    audio_path.write_bytes(audio_bytes)

    # Wire: strategy → ProcessorManager → resolved URL → AudioProcessor.endpoint
    strategy = AudioTranscriptionStrategy(
        model="base", language="en", inference_service="whisper"
    )
    strategy_set = ProcessingStrategySet(transcription=strategy)
    manager = ProcessorManager(logging.getLogger("test"))
    manager.initialize_from_strategies(strategy_set, service_urls={"whisper": url})

    processor = manager.get_processor("audio")
    assert isinstance(processor, AudioProcessor)
    assert processor.endpoint == url, (
        "ProcessorManager must rewrite inference_service → endpoint URL"
    )

    # Run the actual transcription; this hits the stub over HTTP.
    transcript = processor.transcribe_audio(audio_path, output_dir=tmp_path)

    # 1. The processor POSTed the right body shape to /v1/transcribe.
    assert len(stub.captured_requests) == 1
    sent = stub.captured_requests[0]
    assert "audio_b64" in sent, "remote path must base64-encode the audio bytes"
    assert base64.b64decode(sent["audio_b64"]) == audio_bytes, (
        "audio bytes must round-trip through base64 unchanged"
    )
    assert sent.get("language") == "en", (
        "non-auto language hint must be forwarded; auto must be omitted"
    )

    # 2. The processor parsed the response into the canonical transcript dict.
    assert transcript["full_text"] == "hello world"
    assert transcript["language"] == "en"
    assert transcript["model"] == "stub-base"
    assert transcript["video_id"] == "clip"
    assert transcript["duration"] == pytest.approx(1.5)
    assert len(transcript["segments"]) == 2
    assert transcript["segments"][0] == {"start": 0.0, "end": 0.7, "text": "hello"}
    assert transcript["segments"][1] == {"start": 0.7, "end": 1.5, "text": "world"}

    # 3. Transcript was written to disk in the expected location.
    written = Path(tmp_path) / "transcripts" / "clip_transcript.json"
    assert written.exists()
    on_disk = json.loads(written.read_text())
    assert on_disk["full_text"] == "hello world"


def test_language_auto_is_omitted_from_request(stub_whisper, tmp_path):
    """When language='auto' the processor must NOT send the field; the pod
    treats absent ``language`` as 'detect'. Sending the literal string
    'auto' would be interpreted as ISO-639-1 by the pod and silently
    return wrong-language transcripts."""
    url, stub = stub_whisper

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"AUDIO")

    strategy = AudioTranscriptionStrategy(
        model="base", language="auto", inference_service="whisper"
    )
    strategy_set = ProcessingStrategySet(transcription=strategy)
    manager = ProcessorManager(logging.getLogger("test"))
    manager.initialize_from_strategies(strategy_set, service_urls={"whisper": url})

    processor = manager.get_processor("audio")
    processor.transcribe_audio(audio_path, output_dir=tmp_path)

    sent = stub.captured_requests[0]
    assert "language" not in sent, (
        "language='auto' must be omitted; pod treats absence as auto-detect"
    )


def test_pod_500_surfaces_as_error_dict(stub_whisper, tmp_path):
    """A pod-side 5xx must produce the error-shaped result dict (the
    same shape ``transcribe_audio`` produces for any exception in the
    local path), not a silently-empty 'success' that downstream stages
    would treat as a real but empty transcript."""
    url, _ = stub_whisper

    # Stub the handler to fail just for this test
    class _Failing(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self) -> None:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"detail": "stub failure"}')

    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Failing)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        audio_path = tmp_path / "clip.wav"
        audio_path.write_bytes(b"AUDIO")

        processor = AudioProcessor(
            logging.getLogger("test"),
            model="base",
            endpoint=f"http://127.0.0.1:{port}",
        )
        result = processor.transcribe_audio(audio_path, output_dir=tmp_path)

        assert "error" in result, (
            "pod-side 500 must produce an error-shaped result, "
            "not a silently-empty success"
        )
        assert result["full_text"] == ""
        assert result["segments"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
