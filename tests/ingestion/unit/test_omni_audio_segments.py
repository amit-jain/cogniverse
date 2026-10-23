"""EmbeddingGemma 2 audio ingestion: one document per window of at most 10 s.

A test-owned HTTP server stands in for ``vllm_omni_embed`` and answers each
window with a distinct vector keyed by the WAV length it received, so every
document can be matched to the window it was embedded from. The backend is a
recorder; the Vespa round trip is pinned by the real-service ingestion test.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
    EmbeddingGeneratorImpl,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _window_seconds(body: dict) -> float:
    [audio] = [
        part for part in body["messages"][0]["content"] if part["type"] == "input_audio"
    ]
    with wave.open(io.BytesIO(base64.b64decode(audio["input_audio"]["data"]))) as wav:
        return wav.getnframes() / wav.getframerate()


class _Server:
    def __init__(self, fail_on_window: int | None = None) -> None:
        self.windows: list[float] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seconds = _window_seconds(body)
                owner.windows.append(round(seconds, 2))
                if fail_on_window is not None and round(seconds) == fail_on_window:
                    status, payload = 500, {"error": "engine dead"}
                else:
                    vector = [0.0] * 768
                    vector[round(seconds)] = 1.0
                    status, payload = 200, {"data": [{"index": 0, "embedding": vector}]}
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

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _RecordingBackend:
    def __init__(self) -> None:
        self.fed: list = []

    def ingest_documents(self, documents, schema_name):
        self.fed.extend(documents)
        return {"success_count": len(documents), "failed_documents": []}


def _tone(path: Path, seconds: float) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "16000",
            "-y",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def _generator(url: str, backend) -> EmbeddingGeneratorImpl:
    return EmbeddingGeneratorImpl(
        {
            "schema_name": "audio_embeddinggemma2",
            "embedding_model": "google/embeddinggemma-2",
            "embedding_type": "single_vector",
            "model_loader": "embeddinggemma2",
            "modalities": ["audio"],
            "remote_inference_url": url,
        },
        logging.getLogger("test_omni_audio"),
        backend,
    )


TRANSCRIPT = {
    "full_text": "a b c",
    "language": "en",
    "duration": 25.0,
    "segments": [
        {"start": 0.0, "end": 4.0, "text": "a"},
        {"start": 9.0, "end": 12.0, "text": "b"},
        {"start": 21.0, "end": 25.0, "text": "c"},
    ],
}


def test_each_window_becomes_one_document_with_its_own_vector(tmp_path):
    clip = _tone(tmp_path / "clip.wav", 25.0)
    backend = _RecordingBackend()
    with _Server() as server:
        result = _generator(server.url, backend).generate_embeddings(
            {
                "video_id": "clip",
                "original_filename": "rain.wav",
                "source_url": "s3://bucket/rain.wav",
                "transcript": TRANSCRIPT,
                "audio_files": [{"path": str(clip), "audio_id": "clip"}],
            },
            tmp_path,
        )

    assert sorted(server.windows) == [5.0, 10.0, 10.0]
    assert (result.documents_processed, result.documents_fed, result.errors) == (
        3,
        3,
        [],
    )
    assert [
        (
            doc.id,
            doc.metadata["chunk_index"],
            doc.metadata["chunk_count"],
            doc.metadata["chunk_start"],
            doc.metadata["chunk_end"],
            doc.metadata["audio_transcript"],
        )
        for doc in backend.fed
    ] == [
        ("clip_clip_w0000", 0, 3, 0.0, 10.0, "a b"),
        ("clip_clip_w0001", 1, 3, 10.0, 20.0, "b"),
        ("clip_clip_w0002", 2, 3, 20.0, 25.0, "c"),
    ]
    assert [
        int(np.argmax(doc.embeddings["embedding"]["data"])) for doc in backend.fed
    ] == [
        10,
        10,
        5,
    ]
    assert {
        (
            doc.metadata["audio_id"],
            doc.metadata["audio_title"],
            doc.metadata["audio_path"],
            doc.metadata["audio_language"],
            doc.metadata["audio_duration"],
            doc.metadata["source_url"],
        )
        for doc in backend.fed
    } == {("clip", "rain.wav", str(clip), "en", 25.0, "s3://bucket/rain.wav")}


def test_a_failed_window_feeds_nothing_for_the_file(tmp_path):
    clip = _tone(tmp_path / "clip.wav", 25.0)
    backend = _RecordingBackend()
    with _Server(fail_on_window=5) as server:
        result = _generator(server.url, backend).generate_embeddings(
            {
                "video_id": "clip",
                "transcript": TRANSCRIPT,
                "audio_files": [{"path": str(clip), "audio_id": "clip"}],
            },
            tmp_path,
        )

    assert backend.fed == []
    assert (result.documents_processed, result.documents_fed) == (0, 0)
    assert result.errors == [
        f"Audio 0: vllm_omni_embed at {server.url} failed (500): "
        '{"error": "engine dead"}'
    ]
