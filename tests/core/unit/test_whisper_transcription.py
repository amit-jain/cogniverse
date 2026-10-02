"""Chunked Whisper transcription: the server's cut points, and no empty answer
for sound accepted."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import subprocess
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.common.models import whisper_transcription
from cogniverse_core.common.models.model_loaders import RemoteWhisperLoader
from cogniverse_core.common.models.whisper_transcription import (
    TRANSCRIBE_ATTEMPTS,
    AudioChunk,
    ChunkTranscript,
    EmptyTranscriptError,
    decode_audio,
    loudest_frame_dbfs,
    pcm16_wav_samples,
    split_for_whisper,
    transcribe_in_chunks,
    wav_bytes,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

RATE = 16000
LOGGER = logging.getLogger("whisper-transcription-test")


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    return _noise_samples(round(seconds * RATE), seed)


def _noise_samples(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, count, dtype=np.int16)


class _Server:
    """Answers each chunk from a script; records every request.

    A timestamped answer is one segment at 0.5-1.5 s; an untimed one names no
    language and spans the chunk.
    """

    def __init__(self, answers: dict[int, list[str]], language: str = "en"):
        self.answers = {index: list(texts) for index, texts in answers.items()}
        self.language = language
        self.requests: list[tuple[int, str | None]] = []
        self.timestamps: list[tuple[int, bool]] = []

    def __call__(
        self, chunk: AudioChunk, language: str | None, timestamps: bool
    ) -> ChunkTranscript:
        self.requests.append((chunk.index, language))
        self.timestamps.append((chunk.index, timestamps))
        text = self.answers[chunk.index].pop(0)
        if not timestamps:
            span = chunk.end_s - chunk.start_s
            segments = (
                [{"start": 0.0, "end": span, "text": text.strip()}]
                if text.strip()
                else []
            )
            return ChunkTranscript(text=text, language=None, segments=segments)
        segments = (
            [{"start": 0.5, "end": 1.5, "text": text.strip()}] if text.strip() else []
        )
        return ChunkTranscript(text=text, language=self.language, segments=segments)


def test_audio_up_to_thirty_seconds_is_one_chunk():
    whole = _noise(30.0)
    assert [
        (c.index, c.start_sample, len(c.samples)) for c in split_for_whisper(whole)
    ] == [(0, 0, 480000)]
    longer = _noise_samples(480001)
    longer[465600:467200] = 0
    assert [
        (c.index, c.start_sample, len(c.samples)) for c in split_for_whisper(longer)
    ] == [(0, 0, 465600), (1, 465600, 14401)]


def test_each_cut_is_the_quietest_tenth_of_a_second_in_the_chunks_last_second():
    audio = _noise(75.0)
    # Chunk 0 searches 29.0-30.0 s in 0.1 s windows; chunk 1, which starts at
    # the first cut, searches the second before 30 s later.
    audio[472000:473600] = 0
    audio[939200:940800] = 0

    chunks = split_for_whisper(audio)

    assert [(c.index, c.start_sample, len(c.samples)) for c in chunks] == [
        (0, 0, 472000),
        (1, 472000, 467200),
        (2, 939200, 260800),
    ]
    assert np.array_equal(np.concatenate([c.samples for c in chunks]), audio)
    assert [(c.start_s, c.end_s) for c in chunks] == [
        (0.0, 29.5),
        (29.5, 58.7),
        (58.7, 75.0),
    ]


def test_the_loudest_frame_decides_whether_a_chunk_carries_sound():
    silent = np.zeros(RATE, dtype=np.int16)
    square = np.tile(np.array([16384, -16384], dtype=np.int16), RATE // 2)
    faint = silent.copy()
    faint[8000:8400] = 33  # one 25 ms frame at -59.9 dBFS
    fainter = silent.copy()
    fainter[8000:8400] = 32  # -60.2 dBFS

    assert loudest_frame_dbfs(silent) == -math.inf
    assert loudest_frame_dbfs(square) == pytest.approx(-6.0206, abs=1e-4)
    assert [
        AudioChunk(0, 0, samples).carries_sound
        for samples in (silent, faint, fainter, square)
    ] == [False, True, False, True]


def test_wav_round_trip_keeps_every_sample():
    audio = _noise(1.25)
    assert np.array_equal(pcm16_wav_samples(wav_bytes(audio)), audio)


def test_a_wav_that_is_not_16k_mono_pcm16_is_refused():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(b"\x00" * 400)
    with pytest.raises(ValueError) as caught:
        pcm16_wav_samples(buffer.getvalue())
    assert str(caught.value) == (
        "expected 16 kHz mono PCM16 WAV, got 2 channel(s), 16-bit, 44100 Hz"
    )


def test_decode_resamples_a_container_to_16k_mono(tmp_path):
    source = tmp_path / "stereo.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2:sample_rate=44100",
            "-ac",
            "2",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    samples = decode_audio(source)

    assert samples.dtype == np.int16
    # AAC frames pad the stream; the decoded length is the 2 s plus at most one
    # 1024-sample frame of priming at the source rate.
    assert 32000 <= len(samples) <= 32000 + math.ceil(1024 * RATE / 44100) + 1
    # ffmpeg's sine is 1/8 full scale (-21.1 dBFS RMS per channel); the
    # stereo-to-mono downmix measures -23.95 dBFS.
    assert loudest_frame_dbfs(samples) == pytest.approx(-23.95, abs=0.05)


def test_a_file_without_audio_is_refused(tmp_path):
    video = tmp_path / "mute.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=5:d=1",
            "-c:v",
            "libx264",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError) as caught:
        decode_audio(video)
    assert str(caught.value) == f"{video}: no audio stream present"


def test_chunks_are_merged_in_order_with_the_first_chunks_language_and_offsets():
    audio = _noise(75.0)
    audio[472000:473600] = 0
    audio[939200:940800] = 0
    server = _Server({0: [" one"], 1: [" two  three"], 2: [" four"]})

    transcript = transcribe_in_chunks(
        audio, server, language=None, source="clip.wav", logger=LOGGER
    )

    # The server detects the language from the first chunk, as it does for a
    # whole file, and every later chunk is sent in that language.
    assert server.requests == [(0, None), (1, "en"), (2, "en")]
    assert transcript == {
        "full_text": "one  two  three  four",
        "language": "en",
        "duration": 75.0,
        "segments": [
            {"start": 0.5, "end": 1.5, "text": "one"},
            {"start": 29.5 + 0.5, "end": 29.5 + 1.5, "text": "two  three"},
            {"start": 58.7 + 0.5, "end": 58.7 + 1.5, "text": "four"},
        ],
    }


def test_a_named_language_is_sent_with_every_chunk_and_joins_without_spaces():
    audio = _noise(45.0)
    server = _Server({0: ["こんにちは"], 1: ["世界"]}, language="ja")

    transcript = transcribe_in_chunks(
        audio, server, language="ja", source="clip.wav", logger=LOGGER
    )

    assert server.requests == [(0, "ja"), (1, "ja")]
    assert transcript["full_text"] == "こんにちは世界"


def test_an_empty_answer_for_sound_is_asked_again_and_the_retry_is_kept(caplog):
    audio = _noise(45.0)
    server = _Server({0: [" first"], 1: ["", " second"]})

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        transcript = transcribe_in_chunks(
            audio, server, language="en", source="clip.wav", logger=LOGGER
        )

    assert server.requests == [(0, "en"), (1, "en"), (1, "en")]
    assert transcript["full_text"] == "first  second"
    first_cut = split_for_whisper(audio)[1].start_s
    assert [record.getMessage() for record in caplog.records] == [
        f"clip.wav: chunk 1 ({first_cut:.2f}-45.00s, loudest frame "
        f"{loudest_frame_dbfs(audio[int(first_cut * RATE) :]):.1f} dBFS) came back "
        f"with an empty transcript on attempt 1 of {TRANSCRIBE_ATTEMPTS}"
    ]


def test_an_empty_answer_for_sound_on_every_attempt_raises_naming_the_chunk():
    audio = _noise(45.0)
    server = _Server({0: [" first"], 1: [""] * TRANSCRIBE_ATTEMPTS})

    with pytest.raises(EmptyTranscriptError) as caught:
        transcribe_in_chunks(
            audio, server, language="en", source="clip.wav", logger=LOGGER
        )

    chunk = split_for_whisper(audio)[1]
    error = caught.value
    assert (
        error.source,
        error.chunk_index,
        error.start_s,
        error.end_s,
        error.loudest_frame_dbfs,
        error.attempts,
    ) == ("clip.wav", 1, chunk.start_s, 45.0, chunk.loudest_frame_dbfs, 3)
    assert str(error) == (
        f"clip.wav: chunk 1 ({chunk.start_s:.2f}-45.00s, loudest frame "
        f"{chunk.loudest_frame_dbfs:.1f} dBFS) carries sound but came back with "
        "an empty transcript on all 3 attempts"
    )
    assert server.requests == [(0, "en")] + [(1, "en")] * 3
    # The last attempt asks without timestamps.
    assert server.timestamps == [(0, True), (1, True), (1, True), (1, False)]


def test_an_empty_answer_for_silence_is_kept_without_asking_again():
    # The cut lands at 29.0 s, the first of the zero windows, so the second
    # chunk is all silence.
    audio = np.concatenate([_noise(29.0), np.zeros(16 * RATE, dtype=np.int16)])
    server = _Server({0: [" spoken"], 1: [""]})

    transcript = transcribe_in_chunks(
        audio, server, language="en", source="clip.wav", logger=LOGGER
    )

    assert server.requests == [(0, "en"), (1, "en")]
    assert transcript["full_text"] == "spoken"
    assert transcript["segments"] == [{"start": 0.5, "end": 1.5, "text": "spoken"}]


def test_a_failed_request_propagates_without_another_attempt():
    calls: list[int] = []

    def refuse(
        chunk: AudioChunk, language: str | None, timestamps: bool
    ) -> ChunkTranscript:
        calls.append(chunk.index)
        raise ConnectionError("asr unreachable")

    with pytest.raises(ConnectionError) as caught:
        transcribe_in_chunks(
            _noise(5.0), refuse, language="en", source="clip.wav", logger=LOGGER
        )
    assert str(caught.value) == "asr unreachable"
    assert calls == [0]


def test_the_split_mirrors_the_servers_chunk_bounds():
    assert (
        whisper_transcription.CHUNK_SECONDS,
        whisper_transcription.SPLIT_SEARCH_SECONDS,
        whisper_transcription.SPLIT_WINDOW_SAMPLES,
        whisper_transcription.WHISPER_SAMPLE_RATE,
    ) == (30, 1, 1600, 16000)


def test_path_type_is_accepted(tmp_path):
    wav = tmp_path / "tone.wav"
    wav.write_bytes(wav_bytes(_noise(0.5)))
    assert len(decode_audio(Path(wav))) == 8000


class _LoaderWhisper:
    """A real HTTP server for the remote loader: answers from a script and
    records each request's headers and form fields."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.requests: list[dict] = []
        whisper = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                whisper.requests.append(
                    {
                        "authorization": self.headers.get("Authorization"),
                        "format": re.search(
                            rb'name="response_format"\r\n\r\n([a-z_]+)\r\n', raw
                        )
                        .group(1)
                        .decode(),
                        "model": b'name="model"\r\n\r\nopenai/whisper-tiny' in raw,
                    }
                )
                text = whisper.texts.pop(0)
                body = json.dumps(
                    {
                        "text": text,
                        "language": "en",
                        "duration": "1.0",
                        "segments": (
                            [{"start": 0.0, "end": 1.0, "text": text}] if text else []
                        ),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _remote_loader_wrapper(url: str):
    loader = RemoteWhisperLoader(
        "openai/whisper-tiny",
        {"remote_inference_url": url, "remote_inference_api_key": "loader-key"},
        LOGGER,
    )
    wrapper, _ = loader.load_model()
    return wrapper


def test_the_remote_loader_asks_again_after_an_empty_answer_for_sound(tmp_path):
    clip = tmp_path / "tone.wav"
    clip.write_bytes(wav_bytes(_noise(1.0)))
    whisper = _LoaderWhisper(["", " hello"])

    with whisper as url:
        result = _remote_loader_wrapper(url).transcribe(str(clip), language="en")

    assert result == {
        "text": "hello",
        "language": "en",
        "duration": 1.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
    }
    assert (
        whisper.requests
        == [
            {
                "authorization": "Bearer loader-key",
                "format": "verbose_json",
                "model": True,
            }
        ]
        * 2
    )


def test_the_remote_loader_raises_when_sound_keeps_coming_back_empty(tmp_path):
    clip = tmp_path / "tone.wav"
    clip.write_bytes(wav_bytes(_noise(1.0)))
    whisper = _LoaderWhisper([""] * TRANSCRIBE_ATTEMPTS)

    with whisper as url:
        with pytest.raises(EmptyTranscriptError) as caught:
            _remote_loader_wrapper(url).transcribe(str(clip))

    assert (caught.value.source, caught.value.chunk_index, caught.value.attempts) == (
        str(clip),
        0,
        TRANSCRIBE_ATTEMPTS,
    )
    assert [request["format"] for request in whisper.requests] == [
        "verbose_json",
        "verbose_json",
        "json",
    ]


def test_a_chunk_empty_with_timestamps_every_time_keeps_the_untimed_text():
    """Some audio never decodes with timestamps; asked without them, its text
    is one segment spanning the chunk."""
    audio = _noise(45.0)
    audio[472000:473600] = 0
    server = _Server({0: [" first"], 1: ["", "", " *BANG* *BANG*"]})

    transcript = transcribe_in_chunks(
        audio, server, language=None, source="clip.wav", logger=LOGGER
    )

    assert server.timestamps == [(0, True), (1, True), (1, True), (1, False)]
    assert server.requests == [(0, None), (1, "en"), (1, "en"), (1, "en")]
    assert transcript == {
        "full_text": "first  *BANG* *BANG*",
        "language": "en",
        "duration": 45.0,
        "segments": [
            {"start": 0.5, "end": 1.5, "text": "first"},
            {"start": 29.5, "end": 45.0, "text": "*BANG* *BANG*"},
        ],
    }


def test_an_untimed_answer_keeps_the_language_its_empty_attempts_named():
    audio = _noise(45.0)
    audio[472000:473600] = 0
    server = _Server({0: ["", "", " untimed"], 1: [" second"], 2: []})

    transcript = transcribe_in_chunks(
        audio, server, language=None, source="clip.wav", logger=LOGGER
    )

    # The empty timestamped answers still named the language, and the next
    # chunk is sent in it.
    assert server.requests == [(0, None), (0, None), (0, None), (1, "en")]
    assert (transcript["language"], transcript["full_text"]) == (
        "en",
        "untimed  second",
    )
