"""AudioProcessor's remote path asks each Whisper chunk with timestamps and
without, and checks both answers.

The cluster's vLLM Whisper answers some requests with HTTP 200 and an empty
transcript, or a repetition loop, for audio that carries sound. These run the
processor against a real HTTP server speaking the ``/v1/audio/transcriptions``
contract and pin that such an answer is asked again for the same chunk at the
next temperature, and that one which never recovers fails the transcription
naming the chunk, alone and under concurrent jobs.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.common.models.whisper_transcription import (
    FALLBACK_TEMPERATURES,
    TRANSCRIBE_ATTEMPTS,
    compression_ratio,
    loudest_frame_dbfs,
    pcm16_wav_samples,
    split_for_whisper,
)
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

RATE = 16000
MODEL = "openai/whisper-large-v3-turbo"
LOOP = " over and over" * 40


def _tone_wav(path: Path, seconds: float, frequency: float) -> np.ndarray:
    """A sound-bearing 16 kHz mono clip whose first cut is at 29.5 s."""
    t = np.arange(round(seconds * RATE)) / RATE
    samples = (3000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    samples[472000:473600] = 0
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(samples.tobytes())
    return samples


class _Whisper:
    """A real HTTP server answering each chunk with ``<file>-<chunk start>``.

    ``empty_first`` lists (file, chunk start) pairs whose first answer is the
    empty 200 the cluster's server gives; ``always_empty`` ones never recover;
    ``timestamped_empty`` ones are empty whenever timestamps are asked for;
    ``untimed_loop`` ones answer a repetition loop whenever asked without;
    ``untimed_unavailable`` ones answer 503 when asked without;
    ``timed_to_end`` ones are timed to the end of the chunk.
    """

    def __init__(
        self,
        empty_first=(),
        always_empty=(),
        timestamped_empty=(),
        untimed_loop=(),
        untimed_unavailable=(),
        timed_to_end=(),
    ) -> None:
        self.empty_first = set(empty_first)
        self.always_empty = set(always_empty)
        self.timestamped_empty = set(timestamped_empty)
        self.untimed_loop = set(untimed_loop)
        self.untimed_unavailable = set(untimed_unavailable)
        self.timed_to_end = set(timed_to_end)
        self.requests: list[tuple[str, float, str | None, bytes]] = []
        self.formats: list[tuple[str, float, str, float]] = []
        self._answered: set[tuple[str, float]] = set()
        self._lock = threading.Lock()
        whisper = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": MODEL}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                answer = whisper.answer(raw)
                if answer is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def answer(self, raw: bytes) -> dict | None:
        name = re.search(rb'filename="([^"]+)\.wav"', raw).group(1).decode()
        language = re.search(rb'name="language"\r\n\r\n([a-z]+)\r\n', raw)
        timed = (
            re.search(rb'name="response_format"\r\n\r\n([a-z_]+)\r\n', raw)
            .group(1)
            .decode()
        )
        temperature = float(
            re.search(rb'name="temperature"\r\n\r\n([0-9.]+)\r\n', raw).group(1)
        )
        audio = raw[raw.index(b"RIFF") : raw.rindex(b"\r\n--")]
        samples = pcm16_wav_samples(audio)
        # The chunk's first sample identifies where it starts in its file.
        start = _chunk_start(name, samples)
        key = (name, start)
        with self._lock:
            self.requests.append(
                (name, start, language.group(1).decode() if language else None, audio)
            )
            self.formats.append((name, start, timed, temperature))
            empty = (
                key in self.always_empty
                or (key in self.empty_first and key not in self._answered)
                or (key in self.timestamped_empty and timed == "verbose_json")
            )
            self._answered.add(key)
        duration = len(samples) / RATE
        # Timed text stops a second short of the chunk's end, so the chunk is
        # asked without timestamps too, unless the chunk is timed to its end.
        end = duration if key in self.timed_to_end else duration - 1.0
        text = "" if empty else f" {name}-{start:g}"
        if timed == "json":
            if key in self.untimed_unavailable:
                return None
            return {"text": LOOP if key in self.untimed_loop else text}
        return {
            "text": text,
            "language": "en",
            "duration": str(duration),
            "segments": ([{"start": 0.0, "end": end, "text": text}] if text else []),
        }

    def __enter__(self) -> "_Whisper":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


_CLIPS: dict[str, np.ndarray] = {}


def _chunk_start(name: str, chunk: np.ndarray) -> float:
    clip = _CLIPS[name]
    for candidate in split_for_whisper(clip):
        if np.array_equal(candidate.samples, chunk):
            return candidate.start_s
    raise AssertionError(f"{name}: request carried audio that is no chunk of it")


def _clip(tmp_path: Path, name: str, frequency: float = 440.0) -> Path:
    path = tmp_path / f"{name}.wav"
    _CLIPS[name] = _tone_wav(path, 45.0, frequency)
    return path


def _processor(url: str) -> AudioProcessor:
    return AudioProcessor(
        logging.getLogger("chunked-remote-test"),
        model="base",
        language="auto",
        endpoint=url,
    )


V, J = "verbose_json", "json"


def test_an_empty_answer_for_a_chunk_is_asked_again_and_the_transcript_is_whole(
    tmp_path,
):
    clip = _clip(tmp_path, "spoken")
    with _Whisper(empty_first=[("spoken", 29.5)]) as whisper:
        transcript = _processor(whisper.url).transcribe_audio(clip, tmp_path)

    assert [
        (name, start, language) for name, start, language, _ in whisper.requests
    ] == [
        ("spoken", 0.0, None),
        ("spoken", 0.0, "en"),
        ("spoken", 29.5, "en"),
        ("spoken", 29.5, "en"),
        ("spoken", 29.5, "en"),
    ]
    assert whisper.formats == [
        ("spoken", 0.0, V, 0.0),
        ("spoken", 0.0, J, 0.0),
        ("spoken", 29.5, V, 0.0),
        ("spoken", 29.5, J, 0.0),
        ("spoken", 29.5, V, 0.2),
    ]
    assert whisper.requests[2][3] == whisper.requests[3][3] == whisper.requests[4][3]
    assert {
        key: transcript[key] for key in transcript if key != "transcription_time"
    } == {
        "video_id": "spoken",
        "video_path": str(clip),
        "model": MODEL,
        "language": "en",
        "duration": 45.0,
        "full_text": "spoken-0 spoken-29.5",
        "segments": [
            {"start": 0.0, "end": 28.5, "text": "spoken-0"},
            {"start": 29.5, "end": 44.0, "text": "spoken-29.5"},
        ],
    }
    written = json.loads(
        (tmp_path / "transcripts" / "spoken_transcript.json").read_text()
    )
    assert written["full_text"] == "spoken-0 spoken-29.5"


def test_a_chunk_empty_on_every_attempt_fails_the_transcription_naming_it(tmp_path):
    clip = _clip(tmp_path, "spoken")
    loudest = loudest_frame_dbfs(_CLIPS["spoken"][472000:])
    with _Whisper(always_empty=[("spoken", 29.5)]) as whisper:
        transcript = _processor(whisper.url).transcribe_audio(clip, tmp_path)

    assert transcript == {
        "video_id": "spoken",
        "error": (
            f"{clip}: chunk 1 (29.50-45.00s, loudest frame {loudest:.1f} dBFS) "
            "carries sound but came back with an empty transcript on all "
            f"{TRANSCRIBE_ATTEMPTS} attempts"
        ),
        "full_text": "",
        "segments": [],
    }
    assert whisper.formats == [("spoken", 0.0, V, 0.0), ("spoken", 0.0, J, 0.0)] + [
        ("spoken", 29.5, fmt, t) for t in FALLBACK_TEMPERATURES for fmt in (V, J)
    ]
    assert not (tmp_path / "transcripts" / "spoken_transcript.json").exists()


def test_a_chunk_whose_text_loops_on_every_attempt_fails_the_transcription(
    tmp_path,
):
    clip = _clip(tmp_path, "spoken")
    ratio = compression_ratio(LOOP)
    with _Whisper(untimed_loop=[("spoken", 29.5)]) as whisper:
        transcript = _processor(whisper.url).transcribe_audio(clip, tmp_path)

    assert transcript == {
        "video_id": "spoken",
        "error": (
            f"{clip}: chunk 1 (29.50-45.00s) came back unusable on all "
            f"{TRANSCRIBE_ATTEMPTS} attempts: 6 repetition loops (compression "
            f"ratio {', '.join([f'{ratio:.2f}'] * 6)}, above 2.4) and 0 empty"
        ),
        "full_text": "",
        "segments": [],
    }
    assert whisper.formats == [
        ("spoken", 0.0, V, 0.0),
        ("spoken", 0.0, J, 0.0),
        ("spoken", 29.5, V, 0.0),
    ] + [("spoken", 29.5, J, t) for t in FALLBACK_TEMPERATURES]
    assert not (tmp_path / "transcripts" / "spoken_transcript.json").exists()


def test_a_failed_untimed_request_fails_the_transcription_without_another_attempt(
    tmp_path,
):
    clip = _clip(tmp_path, "spoken")
    with _Whisper(untimed_unavailable=[("spoken", 29.5)]) as whisper:
        transcript = _processor(whisper.url).transcribe_audio(clip, tmp_path)

    assert transcript == {
        "video_id": "spoken",
        "error": (
            "503 Server Error: Service Unavailable for url: "
            f"{whisper.url}/v1/audio/transcriptions"
        ),
        "full_text": "",
        "segments": [],
    }
    assert whisper.formats == [
        ("spoken", 0.0, V, 0.0),
        ("spoken", 0.0, J, 0.0),
        ("spoken", 29.5, V, 0.0),
        ("spoken", 29.5, J, 0.0),
    ]
    assert not (tmp_path / "transcripts" / "spoken_transcript.json").exists()


def test_concurrent_jobs_each_retry_their_own_chunk_and_keep_their_own_text(
    tmp_path,
):
    names = [f"job{index}" for index in range(8)]
    clips = {
        name: _clip(tmp_path, name, frequency=300.0 + 40.0 * index)
        for index, name in enumerate(names)
    }
    ready = threading.Barrier(len(names))

    with _Whisper(empty_first=[(name, 29.5) for name in names]) as whisper:
        processor = _processor(whisper.url)

        def run(name: str) -> dict:
            ready.wait(timeout=10)
            return processor.transcribe_audio(clips[name], tmp_path / name)

        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            transcripts = dict(zip(names, pool.map(run, names)))

    assert {name: transcripts[name]["full_text"] for name in names} == {
        name: f"{name}-0 {name}-29.5" for name in names
    }
    assert sorted(whisper.formats) == sorted(
        entry
        for name in names
        for entry in [
            (name, 0.0, V, 0.0),
            (name, 0.0, J, 0.0),
            (name, 29.5, V, 0.0),
            (name, 29.5, J, 0.0),
            (name, 29.5, V, 0.2),
        ]
    )
    for name in names:
        retried = [
            audio
            for sent, start, _, audio in whisper.requests
            if (sent, start) == (name, 29.5)
        ]
        assert len(retried) == 3
        assert retried[0] == retried[1] == retried[2]


def test_a_chunk_that_never_decodes_with_timestamps_keeps_its_untimed_text(
    tmp_path,
):
    clip = _clip(tmp_path, "spoken")
    with _Whisper(timestamped_empty=[("spoken", 29.5)]) as whisper:
        transcript = _processor(whisper.url).transcribe_audio(clip, tmp_path)

    assert whisper.formats == [
        ("spoken", 0.0, V, 0.0),
        ("spoken", 0.0, J, 0.0),
        ("spoken", 29.5, V, 0.0),
        ("spoken", 29.5, J, 0.0),
    ] + [("spoken", 29.5, V, t) for t in FALLBACK_TEMPERATURES[1:]]
    assert transcript["full_text"] == "spoken-0 spoken-29.5"
    assert transcript["segments"] == [
        {"start": 0.0, "end": 28.5, "text": "spoken-0"},
        {"start": 29.5, "end": 45.0, "text": "spoken-29.5"},
    ]


def test_a_chunk_timed_to_its_end_is_not_asked_without_timestamps(tmp_path):
    clip = _clip(tmp_path, "spoken")
    with _Whisper(timed_to_end=[("spoken", 0.0)]) as whisper:
        transcript = _processor(whisper.url).transcribe_audio(clip, tmp_path)

    assert whisper.formats == [
        ("spoken", 0.0, V, 0.0),
        ("spoken", 29.5, V, 0.0),
        ("spoken", 29.5, J, 0.0),
    ]
    assert transcript["segments"] == [
        {"start": 0.0, "end": 29.5, "text": "spoken-0"},
        {"start": 29.5, "end": 44.0, "text": "spoken-29.5"},
    ]
