"""End-to-end round-trip of the remote Whisper transcription path.

Spins up a real HTTP stub that mimics vLLM's
``/v1/audio/transcriptions`` multipart contract, routes a tiny audio
file through ``AudioTranscriptionStrategy`` → ``ProcessorManager`` URL
substitution → ``AudioProcessor.transcribe_audio``, and asserts both
the request shape sent to the pod and the response shape returned to
the pipeline.

This is the wiring-correctness test required by CLAUDE.md — it would
catch:
- A strategy that drops ``inference_service`` so the URL never gets
  resolved.
- A processor-manager substitution that picks the wrong service or
  forgets to ``pop`` the placeholder.
- An AudioProcessor remote path that ships the wrong body shape
  (e.g. JSON instead of multipart) or fails to parse the OpenAI-
  compatible response back into the dict the pipeline expects.
"""

from __future__ import annotations

import cgi
import io
import json
import logging
import math
import socket
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import (
    ProcessingStrategySet,
)
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from cogniverse_runtime.ingestion.strategies import AudioTranscriptionStrategy


def _valid_wav_bytes(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """A real 16 kHz mono PCM WAV the processor's pyav extraction can decode.

    The remote path re-encodes audio to 16 kHz mono before POSTing, so the
    fixture must supply genuinely decodable audio (not placeholder bytes).
    """
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(
        b"".join(
            struct.pack("<h", int(2000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(seconds * rate))
        )
    )
    w.close()
    return buf.getvalue()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _WhisperStub(BaseHTTPRequestHandler):
    """Mimics vLLM's OpenAI-compatible ``/v1/audio/transcriptions``.

    Records each POST so the test can assert the multipart shape, then
    replies with a fixed transcript matching vLLM's verbose_json response.
    """

    captured_requests: list[dict] = []  # class-level, reset per test

    def log_message(self, format, *args):  # silence stderr
        return

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            payload = json.dumps(
                {"data": [{"id": "openai/whisper-large-v3-turbo"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/health":
            payload = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/v1/audio/transcriptions":
            self.send_response(404)
            self.end_headers()
            return

        ctype = self.headers.get("content-type", "")
        length = int(self.headers.get("content-length", "0"))
        # cgi.FieldStorage is the stdlib multipart parser — fine for tests.
        env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype}
        body_bytes = self.rfile.read(length)
        form = cgi.FieldStorage(
            fp=io.BytesIO(body_bytes),
            headers=self.headers,
            environ=env,
            keep_blank_values=True,
        )
        captured: dict = {}
        for key in form.keys():
            field = form[key]
            if getattr(field, "filename", None):
                captured[key] = {
                    "filename": field.filename,
                    "bytes": field.file.read(),
                }
            else:
                captured[key] = field.value
        _WhisperStub.captured_requests.append(captured)

        resp = {
            "text": "hello world",
            "language": captured.get("language") or "en",
            "duration": 1.5,
            "model": captured.get("model", "openai/whisper-large-v3-turbo"),
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
    """End-to-end: strategy declares vllm_asr, manager resolves URL, processor POSTs."""
    url, stub = stub_whisper

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(_valid_wav_bytes())

    strategy = AudioTranscriptionStrategy(
        model="base", language="en", inference_service="vllm_asr"
    )
    strategy_set = ProcessingStrategySet(transcription=strategy)
    manager = ProcessorManager(logging.getLogger("test"))
    manager.initialize_from_strategies(strategy_set, service_urls={"vllm_asr": url})

    processor = manager.get_processor("audio")
    assert isinstance(processor, AudioProcessor)
    assert processor.endpoint == url, (
        "ProcessorManager must rewrite inference_service → endpoint URL"
    )

    transcript = processor.transcribe_audio(audio_path, output_dir=tmp_path)

    assert len(stub.captured_requests) == 1
    sent = stub.captured_requests[0]
    assert "file" in sent and isinstance(sent["file"], dict), (
        "remote path must POST the audio as a multipart 'file' part"
    )
    posted = sent["file"]["bytes"]
    assert posted[:4] == b"RIFF" and len(posted) > 44, (
        "remote path must POST re-encoded 16 kHz mono PCM WAV bytes"
    )
    assert sent.get("model"), "model id must be present in form data"
    assert sent.get("language") == "en", (
        "non-auto language hint must be forwarded; auto must be omitted"
    )

    assert transcript["full_text"] == "hello world"
    assert transcript["language"] == "en"
    assert transcript["video_id"] == "clip"
    assert transcript["duration"] == pytest.approx(1.5)
    assert len(transcript["segments"]) == 2
    assert transcript["segments"][0] == {"start": 0.0, "end": 0.7, "text": "hello"}
    assert transcript["segments"][1] == {"start": 0.7, "end": 1.5, "text": "world"}

    written = Path(tmp_path) / "transcripts" / "clip_transcript.json"
    assert written.exists()
    on_disk = json.loads(written.read_text())
    assert on_disk["full_text"] == "hello world"


def test_language_auto_is_omitted_from_request(stub_whisper, tmp_path):
    """When language='auto' the processor must NOT send the field; the pod
    treats absent ``language`` as detect. Sending the literal string
    'auto' would be interpreted as ISO-639-1 by the pod and silently
    return wrong-language transcripts."""
    url, stub = stub_whisper

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(_valid_wav_bytes())

    strategy = AudioTranscriptionStrategy(
        model="base", language="auto", inference_service="vllm_asr"
    )
    strategy_set = ProcessingStrategySet(transcription=strategy)
    manager = ProcessorManager(logging.getLogger("test"))
    manager.initialize_from_strategies(strategy_set, service_urls={"vllm_asr": url})

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

    class _Failing(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self) -> None:
            self.send_response(404)
            self.end_headers()

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
        audio_path.write_bytes(_valid_wav_bytes())

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
