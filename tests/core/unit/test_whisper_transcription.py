"""Chunked Whisper transcription: the server's cut points, each chunk's text from
a json answer timed by a verbose_json answer, and no empty or looping answer
accepted."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import subprocess
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.models import whisper_transcription
from cogniverse_core.common.models.model_loaders import RemoteWhisperLoader
from cogniverse_core.common.models.whisper_transcription import (
    FALLBACK_TEMPERATURES,
    GARBLED_COMPRESSION_RATIO,
    SAMPLING_SEED,
    TRANSCRIBE_ATTEMPTS,
    AudioChunk,
    ChunkTranscript,
    EmptyTranscriptError,
    GarbledTranscriptError,
    align_text,
    compression_ratio,
    decode_audio,
    lenient_chunk_answer,
    loudest_frame_dbfs,
    pcm16_wav_samples,
    response_format,
    sampling_fields,
    split_for_whisper,
    transcribe_in_chunks,
    wav_bytes,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

RATE = 16000
LOGGER = logging.getLogger("whisper-transcription-test")
RECORDED = Path(__file__).parent / "fixtures" / "vllm_whisper_chunk_answers.json"
# A json answer looping on one phrase, as the servers answer some chunks.
LOOP = " Go! Stop cooking in your safe!" * 30


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    return _noise_samples(round(seconds * RATE), seed)


def _noise_samples(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, count, dtype=np.int16)


def _segments(*spans: tuple[float, float, str]) -> list[dict]:
    return [{"start": start, "end": end, "text": text} for start, end, text in spans]


class _Server:
    """Answers each chunk's two requests from scripts and records every request.

    ``timed[i]`` lists chunk ``i``'s verbose_json answers, each a list of
    ``(start, end, text)`` segments; ``text[i]`` lists its json answers. A
    verbose_json answer names ``language``; a json answer names none.
    """

    def __init__(
        self,
        timed: dict[int, list[list[tuple[float, float, str]]]],
        text: dict[int, list[str]],
        language: str = "en",
    ):
        self.timed = {index: list(answers) for index, answers in timed.items()}
        self.text = {index: list(answers) for index, answers in text.items()}
        self.language = language
        self.requests: list[tuple[int, str | None, str, float]] = []

    def __call__(
        self,
        chunk: AudioChunk,
        language: str | None,
        timestamps: bool,
        temperature: float = 0.0,
    ) -> ChunkTranscript:
        self.requests.append(
            (chunk.index, language, response_format(timestamps), temperature)
        )
        if timestamps:
            spans = self.timed[chunk.index].pop(0)
            return ChunkTranscript(
                text=" ".join(text for _, _, text in spans),
                language=self.language,
                segments=_segments(*spans),
            )
        return ChunkTranscript(
            text=self.text[chunk.index].pop(0), language=None, segments=[]
        )


def _transcribe(audio: np.ndarray, server, language: str | None = "en") -> dict:
    return transcribe_in_chunks(
        audio, server, language=language, source="clip.wav", logger=LOGGER
    )


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


def test_each_chunk_is_asked_timed_then_untimed_and_keeps_the_json_text():
    audio = _noise(75.0)
    audio[472000:473600] = 0
    audio[939200:940800] = 0
    server = _Server(
        timed={
            0: [[(0.5, 1.5, " one")]],
            1: [[(0.5, 1.5, " two three")]],
            2: [[(0.5, 1.5, " four")]],
        },
        text={0: [" one"], 1: [" two  three"], 2: [" four"]},
    )

    transcript = _transcribe(audio, server, language=None)

    # The first chunk's timed answer names the language; its json request and
    # every later request are sent in it.
    assert server.requests == [
        (0, None, "verbose_json", 0.0),
        (0, "en", "json", 0.0),
        (1, "en", "verbose_json", 0.0),
        (1, "en", "json", 0.0),
        (2, "en", "verbose_json", 0.0),
        (2, "en", "json", 0.0),
    ]
    assert transcript == {
        "full_text": "one two three four",
        "language": "en",
        "duration": 75.0,
        "segments": [
            {"start": 0.5, "end": 1.5, "text": "one"},
            {"start": 29.5 + 0.5, "end": 29.5 + 1.5, "text": "two three"},
            {"start": 58.7 + 0.5, "end": 58.7 + 1.5, "text": "four"},
        ],
    }


def test_a_named_language_is_sent_with_every_request_and_joins_without_spaces():
    audio = _noise(45.0)
    server = _Server(
        timed={0: [[(0.0, 1.0, "こんにちは")]], 1: [[(0.0, 1.0, "世界")]]},
        text={0: ["こんにちは"], 1: ["世界"]},
        language="ja",
    )

    transcript = _transcribe(audio, server, language="ja")

    assert [(index, language) for index, language, _, _ in server.requests] == [
        (0, "ja"),
        (0, "ja"),
        (1, "ja"),
        (1, "ja"),
    ]
    assert transcript["full_text"] == "こんにちは世界"


def test_json_words_after_the_last_timed_segment_take_the_rest_of_the_chunk():
    assert align_text(" a b c d e", _segments((0.0, 9.56, " a b c")), 29.1) == (
        "a b c d e",
        _segments((0.0, 9.56, "a b c"), (9.56, 29.1, "d e")),
    )


def test_json_words_before_and_between_timed_segments_take_the_untimed_gaps():
    assert align_text("a b c d", _segments((2.0, 4.0, "b"), (6.0, 8.0, "d")), 10.0) == (
        "a b c d",
        _segments((0.0, 2.0, "a"), (2.0, 4.0, "b"), (4.0, 6.0, "c"), (6.0, 8.0, "d")),
    )


def test_json_words_the_timed_answer_lacks_in_timed_time_join_the_segment_before():
    assert align_text(
        "one two extra three", _segments((0.0, 5.0, "one two three")), 5.0
    ) == ("one two extra three", _segments((0.0, 5.0, "one two extra three")))
    assert align_text(
        "hypothermia. DR. DR. SOUDOS: Dress in layers.",
        _segments((7.0, 13.0, " hypothermia."), (13.0, 19.0, " Dress in layers.")),
        29.5,
    ) == (
        "hypothermia. DR. DR. SOUDOS: Dress in layers.",
        _segments(
            (7.0, 13.0, "hypothermia. DR. DR. SOUDOS:"),
            (13.0, 19.0, "Dress in layers."),
        ),
    )
    # At the start, with the first segment starting at 0, they join it.
    assert align_text(" When a big", _segments((0.0, 5.0, " a big")), 10.0) == (
        "When a big",
        _segments((0.0, 5.0, "When a big")),
    )
    # A gap shorter than one timestamp step is no untimed time.
    assert align_text("a b", _segments((0.0, 29.099999999999998, "a")), 29.1) == (
        "a b",
        _segments((0.0, 29.099999999999998, "a b")),
    )


def test_timed_words_the_json_answer_lacks_are_kept_where_they_were_timed():
    assert align_text(
        "one two", _segments((0.0, 5.0, "one two"), (5.0, 10.0, "three four.")), 10.0
    ) == (
        "one two three four.",
        _segments((0.0, 5.0, "one two"), (5.0, 10.0, "three four.")),
    )
    assert align_text("", _segments((0.0, 1.0, " hi")), 2.0) == (
        "hi",
        _segments((0.0, 1.0, "hi")),
    )


def test_where_the_answers_differ_the_json_words_take_the_timed_words_place():
    assert align_text(
        "Dr. Soto says take breaks.",
        _segments((0.0, 4.0, "Dr. Souto says"), (4.0, 8.0, "take breaks.")),
        8.0,
    ) == (
        "Dr. Soto says take breaks.",
        _segments((0.0, 4.0, "Dr. Soto says"), (4.0, 8.0, "take breaks.")),
    )
    # A longer json rendering spreads over the timed words' segments in order.
    assert align_text(
        "w1 w2 w3 w4", _segments((0.0, 1.0, "a"), (1.0, 2.0, "b")), 2.0
    ) == ("w1 w2 w3 w4", _segments((0.0, 1.0, "w1 w2"), (1.0, 2.0, "w3 w4")))
    # A shorter one, like a json decode that collapses, leaves the timed
    # words in place.
    assert align_text(
        "Dress in DR. S.:.....",
        _segments((0.0, 2.0, "Dress in layers."), (2.0, 5.0, "And as you start")),
        5.0,
    ) == (
        "Dress in layers. And as you start",
        _segments((0.0, 2.0, "Dress in layers."), (2.0, 5.0, "And as you start")),
    )


def test_a_no_space_language_aligns_by_character():
    assert align_text(
        "こんにちは世界", _segments((0.0, 1.0, "こんにちは")), 3.0, no_space=True
    ) == ("こんにちは世界", _segments((0.0, 1.0, "こんにちは"), (1.0, 3.0, "世界")))


def test_without_timed_segments_the_json_text_spans_the_chunk():
    assert align_text(" hello  world ", [], 7.5) == (
        "hello world",
        _segments((0.0, 7.5, "hello world")),
    )
    assert align_text("  ", [], 7.5) == ("", [])


def test_an_empty_timed_answer_is_asked_again_at_the_next_temperature(caplog):
    audio = _noise(10.0)
    server = _Server(
        timed={0: [[], [(0.0, 2.0, " hello")]]}, text={0: [" hello there"]}
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        transcript = _transcribe(audio, server)

    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
        (0, "en", "verbose_json", 0.2),
    ]
    assert transcript["full_text"] == "hello there"
    assert transcript["segments"] == _segments(
        (0.0, 2.0, "hello"), (2.0, 10.0, "there")
    )
    assert [record.getMessage() for record in caplog.records] == [
        f"clip.wav: chunk 0 (0.00-10.00s, loudest frame "
        f"{loudest_frame_dbfs(audio):.1f} dBFS) came back with an empty "
        f"verbose_json transcript on attempt 1 of {TRANSCRIBE_ATTEMPTS} at "
        "temperature 0.0"
    ]


def test_a_chunk_never_timed_keeps_its_json_text_as_one_segment():
    """Some audio never decodes with timestamps; its json text spans the chunk."""
    audio = _noise(10.0)
    server = _Server(timed={0: [[]] * 6}, text={0: [" *BANG* *BANG*"]})

    transcript = _transcribe(audio, server)

    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
    ] + [(0, "en", "verbose_json", t) for t in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert transcript["full_text"] == "*BANG* *BANG*"
    assert transcript["segments"] == _segments((0.0, 10.0, "*BANG* *BANG*"))


def test_a_looping_json_answer_is_asked_again_at_the_next_temperature(caplog):
    audio = _noise(10.0)
    server = _Server(
        timed={0: [[(0.0, 4.0, " Go! Stop cooking in your safe!")]]},
        text={0: [LOOP, " Go! Stop cooking in your safe!"]},
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        transcript = _transcribe(audio, server)

    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
        (0, "en", "json", 0.2),
    ]
    assert transcript["full_text"] == "Go! Stop cooking in your safe!"
    assert transcript["segments"] == _segments(
        (0.0, 4.0, "Go! Stop cooking in your safe!")
    )
    assert [record.getMessage() for record in caplog.records] == [
        "clip.wav: chunk 0 (0.00-10.00s) came back as a repetition loop in json "
        f"(compression ratio {compression_ratio(LOOP):.2f} > 2.4) on attempt 1 of "
        f"{TRANSCRIBE_ATTEMPTS} at temperature 0.0"
    ]


def test_a_json_answer_looping_on_every_attempt_raises_naming_the_chunk():
    audio = _noise(10.0)
    server = _Server(
        timed={0: [[(0.0, 4.0, " Go! Stop cooking in your safe!")]]},
        text={0: [LOOP] * TRANSCRIBE_ATTEMPTS},
    )

    with pytest.raises(GarbledTranscriptError) as caught:
        _transcribe(audio, server)

    ratio = compression_ratio(LOOP)
    error = caught.value
    assert (
        error.source,
        error.chunk_index,
        error.start_s,
        error.end_s,
        error.compression_ratios,
        error.attempts,
    ) == ("clip.wav", 0, 0.0, 10.0, (ratio,) * 6, 6)
    assert str(error) == (
        "clip.wav: chunk 0 (0.00-10.00s) came back unusable on all 6 attempts: "
        f"6 repetition loops (compression ratio {', '.join([f'{ratio:.2f}'] * 6)}"
        ", above 2.4) and 0 empty"
    )
    assert server.requests == [(0, "en", "verbose_json", 0.0)] + [
        (0, "en", "json", t) for t in FALLBACK_TEMPERATURES
    ]


def test_loops_and_empty_answers_together_raise_as_garbled():
    audio = _noise(10.0)
    server = _Server(timed={0: [[]] * 6}, text={0: [LOOP, "", LOOP, "", "", ""]})

    with pytest.raises(GarbledTranscriptError) as caught:
        _transcribe(audio, server)

    ratio = compression_ratio(LOOP)
    assert caught.value.compression_ratios == (ratio, ratio)
    assert str(caught.value).endswith(
        f"2 repetition loops (compression ratio {ratio:.2f}, {ratio:.2f}, above "
        "2.4) and 4 empty"
    )


def test_a_timed_answer_looping_on_every_attempt_leaves_the_json_text_untimed():
    audio = _noise(10.0)
    server = _Server(timed={0: [[(0.0, 1.0, LOOP)]] * 6}, text={0: [" real words"]})

    transcript = _transcribe(audio, server)

    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
    ] + [(0, "en", "verbose_json", t) for t in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert transcript["segments"] == _segments((0.0, 10.0, "real words"))


def test_an_empty_answer_for_sound_on_every_attempt_raises_naming_the_chunk():
    audio = _noise(45.0)
    server = _Server(
        timed={0: [[(0.5, 1.5, " first")]], 1: [[]] * 6},
        text={0: [" first"], 1: [""] * 6},
    )

    with pytest.raises(EmptyTranscriptError) as caught:
        _transcribe(audio, server)

    chunk = split_for_whisper(audio)[1]
    error = caught.value
    assert (
        error.source,
        error.chunk_index,
        error.start_s,
        error.end_s,
        error.loudest_frame_dbfs,
        error.attempts,
    ) == ("clip.wav", 1, chunk.start_s, 45.0, chunk.loudest_frame_dbfs, 6)
    assert str(error) == (
        f"clip.wav: chunk 1 ({chunk.start_s:.2f}-45.00s, loudest frame "
        f"{chunk.loudest_frame_dbfs:.1f} dBFS) carries sound but came back with "
        "an empty transcript on all 6 attempts"
    )
    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
    ] + [
        (1, "en", fmt, t)
        for t in FALLBACK_TEMPERATURES
        for fmt in ("verbose_json", "json")
    ]


def test_a_silent_chunk_may_come_back_empty_without_asking_again():
    # The cut lands at 29.0 s, the first of the zero windows, so the second
    # chunk is all silence.
    audio = np.concatenate([_noise(29.0), np.zeros(16 * RATE, dtype=np.int16)])
    server = _Server(
        timed={0: [[(0.5, 1.5, " spoken")]], 1: [[]]}, text={0: [" spoken"], 1: [""]}
    )

    transcript = _transcribe(audio, server)

    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
        (1, "en", "verbose_json", 0.0),
        (1, "en", "json", 0.0),
    ]
    assert transcript["full_text"] == "spoken"
    assert transcript["segments"] == _segments((0.5, 1.5, "spoken"))


def test_a_silent_chunk_keeps_its_first_answers_without_asking_again():
    audio = np.zeros(5 * RATE, dtype=np.int16)
    server = _Server(timed={0: [[]]}, text={0: [" Thank you."]})

    transcript = _transcribe(audio, server)

    assert server.requests == [
        (0, "en", "verbose_json", 0.0),
        (0, "en", "json", 0.0),
    ]
    assert transcript["segments"] == _segments((0.0, 5.0, "Thank you."))


def test_a_failed_request_propagates_without_another_attempt():
    calls: list[tuple[int, bool, float]] = []

    def refuse(
        chunk: AudioChunk, language: str | None, timestamps: bool, temperature: float
    ) -> ChunkTranscript:
        calls.append((chunk.index, timestamps, temperature))
        raise ConnectionError("asr unreachable")

    with pytest.raises(ConnectionError) as caught:
        _transcribe(_noise(5.0), refuse)
    assert str(caught.value) == "asr unreachable"
    assert calls == [(0, True, 0.0)]


def test_retries_follow_whispers_temperature_fallback_with_a_fixed_seed():
    assert (
        FALLBACK_TEMPERATURES,
        TRANSCRIBE_ATTEMPTS,
        SAMPLING_SEED,
        GARBLED_COMPRESSION_RATIO,
    ) == ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 6, 0, 2.4)
    assert [sampling_fields(t) for t in (0.0, 0.4)] == [
        {"temperature": "0.0", "seed": "0"},
        {"temperature": "0.4", "seed": "0"},
    ]


def _recording() -> dict:
    return json.loads(RECORDED.read_text())


RECORD_KEYS = {
    "server",
    "clip",
    "chunk",
    "chunk_start_s",
    "chunk_len_s",
    "response_format",
    "language",
    "temperature",
    "seed",
    "body",
}


def _check_recording(recording: dict) -> None:
    assert recording["servers"] == {
        "live": {
            "image": "vllm/vllm-openai-rocm:v0.23.0",
            "model": "openai/whisper-large-v3-turbo",
            "recorded": "2026-10-03",
        },
        "cpu": {
            "image": "vllm/vllm-openai-cpu:v0.23.0",
            "model": "openai/whisper-tiny",
            "recorded": "2026-10-03",
        },
    }
    answers = recording["answers"]
    assert len(answers) == 16
    body_keys = {
        response_format(True): {"duration", "language", "segments", "text", "words"},
        response_format(False): {"text", "usage"},
    }
    segment_keys = {
        "avg_logprob",
        "compression_ratio",
        "end",
        "id",
        "no_speech_prob",
        "seek",
        "start",
        "temperature",
        "text",
        "tokens",
    }
    for name, answer in answers.items():
        assert set(answer) == RECORD_KEYS, name
        assert name.split("/")[:3] == [
            answer["server"],
            answer["clip"][3:].split(".")[0],
            str(answer["chunk"]),
        ], name
        assert answer["response_format"] == name.split("/")[3], name
        assert set(answer["body"]) == body_keys[answer["response_format"]], name
        for segment in answer["body"].get("segments") or []:
            assert set(segment) == segment_keys, name
        assert 0 < answer["chunk_len_s"] <= whisper_transcription.CHUNK_SECONDS


def test_the_recorded_answers_keep_their_shape():
    _check_recording(_recording())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["answers"].pop("cpu/pkfcMUIEMo/2/json"),
        lambda r: r["answers"]["live/IMXSEIabMM/2/json"]["body"].pop("usage"),
        lambda r: r["answers"]["live/IMXSEIabMM/3/verbose_json"]["body"]["segments"][
            0
        ].pop("tokens"),
        lambda r: r["answers"]["live/IMXSEIabMM/0/verbose_json"].update(
            response_format="json"
        ),
        lambda r: r["answers"]["cpu/IMXSEIabMM/1/json"].update(chunk_len_s=30.5),
        lambda r: r["servers"]["cpu"].update(image="vllm/vllm-openai-cpu:v0.24.0"),
    ],
)
def test_the_recording_pins_go_red_when_the_recording_drifts(mutate):
    recording = _recording()
    mutate(recording)
    with pytest.raises(AssertionError):
        _check_recording(recording)


def test_the_compression_ratio_is_the_one_the_server_reports():
    answers = _recording()["answers"]
    segments = [
        segment
        for answer in answers.values()
        for segment in answer["body"].get("segments") or []
    ]
    assert len(segments) == 21
    assert [compression_ratio(s["text"]) for s in segments] == [
        s["compression_ratio"] for s in segments
    ]
    assert compression_ratio("") == 0.0


def test_only_the_recorded_loops_are_garbled():
    answers = _recording()["answers"]
    assert {
        name: compression_ratio(answer["body"]["text"]) > GARBLED_COMPRESSION_RATIO
        for name, answer in answers.items()
    } == {name: "/loop_" in name for name in answers}


class _Replay:
    """Answers one chunk's requests with recorded server bodies, read by the
    processor's and the loader's parser."""

    def __init__(self, verbose_json: list[str], json_: list[str]) -> None:
        answers = _recording()["answers"]
        self.scripts = {
            "verbose_json": [answers[name]["body"] for name in verbose_json],
            "json": [answers[name]["body"] for name in json_],
        }
        self.requests: list[tuple[str, float]] = []

    def __call__(self, chunk, language, timestamps, temperature=0.0):
        fmt = response_format(timestamps)
        self.requests.append((fmt, temperature))
        return lenient_chunk_answer(self.scripts[fmt].pop(0), chunk)


def _replay(name: str, replay: _Replay) -> dict:
    seconds = _recording()["answers"][name]["chunk_len_s"]
    transcript = _transcribe(_noise(seconds), replay)
    assert transcript["duration"] == seconds
    return transcript


def test_text_the_cpu_server_dropped_after_the_last_timestamp_is_kept():
    # whisper-tiny decoded 442 text tokens with two timestamps; the server kept
    # the 26 before them and dropped the rest.
    replay = _Replay(
        ["cpu/pkfcMUIEMo/2/verbose_json/tail_dropped"], ["cpu/pkfcMUIEMo/2/json"]
    )

    transcript = _replay("cpu/pkfcMUIEMo/2/json", replay)

    assert replay.requests == [("verbose_json", 0.0), ("json", 0.0)]
    assert transcript["segments"] == _segments(
        (
            0.0,
            9.56,
            "You're not twist, turn your body and throw the snow. If it's very "
            "heavy, do not even throw the snow.",
        ),
        (
            9.56,
            29.1,
            "Turn, walk, and dump the snow. Believe me, after a snow storm, I see "
            "plenty of people coming in here because they've shuffled them "
            "properly. Especially when you get tired, the tired of you are, and "
            "more likely you are to be sloppy and making mistakes. Then the knees,",
        ),
    )


def test_text_after_a_decode_that_stopped_early_is_kept():
    # whisper-tiny ended its timed decode with <|18.00|><|18.00|> and EOS.
    replay = _Replay(
        ["cpu/IMXSEIabMM/1/verbose_json/stopped_at_18s"], ["cpu/IMXSEIabMM/1/json"]
    )

    transcript = _replay("cpu/IMXSEIabMM/1/json", replay)

    assert transcript["segments"] == _segments(
        (
            0.0,
            18.0,
            "that's the most important. You know, sacrificing a limb hurting your "
            "hand and saving your head because having your head hit the ice, "
            "especially when it comes black ice and getting a subdued human toma, "
            "blood inside the brain can be devastating for a lot of people.",
        ),
        (
            18.0,
            29.0,
            "- Sprayne strains and fractures are always near the top of the list "
            "of snow shoveling related injuries. They're usually caused by slipping "
            "and twisting. Over exertion is also considered a common cause",
        ),
    )


def test_timed_text_a_json_answer_stopped_short_of_is_kept():
    replay = _Replay(
        ["live/IMXSEIabMM/1/verbose_json"], ["live/IMXSEIabMM/1/json/stopped_early"]
    )

    transcript = _replay("live/IMXSEIabMM/1/verbose_json", replay)

    assert transcript["segments"] == _segments(
        (
            0.0,
            7.0,
            "that's the most important. Sacrificing a limb, hurting your hand and "
            "saving your head because",
        ),
        (
            7.0,
            12.0,
            "having your head hit the ice, especially when it comes to black ice "
            "and getting a subdural",
        ),
        (
            12.0,
            17.0,
            "hematoma, blood inside the brain, can be devastating for a lot of people.",
        ),
        (
            17.0,
            23.0,
            "Sprains, strains and fractures are always near the top of the list of "
            "snow shoveling related injuries.",
        ),
        (
            23.0,
            29.0,
            "They're usually caused by slipping and twisting. Overexertion is also "
            "considered a common cause for",
        ),
    )


def test_a_json_answer_that_spent_its_tokens_elsewhere_keeps_the_timed_text():
    # The live json decode ran to its 444-token limit and answered " When".
    replay = _Replay(
        ["live/IMXSEIabMM/0/verbose_json"], ["live/IMXSEIabMM/0/json/token_limit"]
    )

    transcript = _replay("live/IMXSEIabMM/0/verbose_json", replay)

    timed = _recording()["answers"]["live/IMXSEIabMM/0/verbose_json"]["body"]
    assert transcript["full_text"] == " ".join(timed["text"].split())
    assert transcript["segments"] == _segments(
        (0.0, 29.8, " ".join(timed["text"].split()))
    )


def test_a_live_json_loop_is_asked_again_until_a_temperature_breaks_it():
    replay = _Replay(
        ["live/IMXSEIabMM/3/verbose_json"],
        [
            "live/IMXSEIabMM/3/json/loop_t0.0",
            "live/IMXSEIabMM/3/json/loop_t0.2",
            "live/IMXSEIabMM/3/json/loop_t0.4",
            "live/IMXSEIabMM/3/json/t0.6",
        ],
    )

    transcript = _replay("live/IMXSEIabMM/3/verbose_json", replay)

    assert replay.requests == [
        ("verbose_json", 0.0),
        ("json", 0.0),
        ("json", 0.2),
        ("json", 0.4),
        ("json", 0.6),
    ]
    assert transcript["segments"] == _segments(
        (
            0.0,
            7.0,
            "or you lift it improperly. And lots of people end up in the ER after "
            "being hit by a shovel, usually after slipping.",
        ),
        (
            7.0,
            13.0,
            "Dr. Souto says the other thing to remember is to dress properly to "
            "prevent hypothermia. DR. DR. DR. DR. SOUDOS, DR. DR. DR. DR. SOUDOS:",
        ),
        (
            13.0,
            19.0,
            "Dress in layers. And as you start getting heated up, you take a top "
            "layer off to cool off a little bit.",
        ),
        (
            19.0,
            23.0,
            "If you're going to be out there for a long time, you start to break a "
            "sweat, then you calm down.",
        ),
        (
            23.0,
            29.5,
            "The sweat wets your clothes, the clothes stay wet. Now they get cold, "
            "and that's how you become hypothermic.",
        ),
    )


def test_a_live_empty_timed_answer_is_asked_again_and_both_answers_combine():
    # The live server decoded 213 tokens with no timestamp pair and answered
    # empty; asked again it answered with eight segments.
    replay = _Replay(
        ["live/IMXSEIabMM/2/verbose_json/empty", "live/IMXSEIabMM/2/verbose_json"],
        ["live/IMXSEIabMM/2/json"],
    )

    transcript = _replay("live/IMXSEIabMM/2/verbose_json", replay)

    assert replay.requests == [
        ("verbose_json", 0.0),
        ("json", 0.0),
        ("verbose_json", 0.2),
    ]
    assert transcript["segments"] == _segments(
        (0.0, 3.0, "for emergency room visits after snow shoveling."),
        (
            3.0,
            7.76,
            "It can cause dehydration, fatigue and in some cases, heart attack.",
        ),
        (
            7.76,
            12.06,
            "If you could do something that's light, moderate activity, doesn't "
            "get yourself huffing and",
        ),
        (12.06, 14.06, "puffing too much, that's fine."),
        (
            14.06,
            18.78,
            "But if you have a heart problem, high blood pressure, chronic back "
            "problems, those are",
        ),
        (
            18.78,
            22.92,
            "things that it would be best deferred to somebody else to do for you.",
        ),
        (
            22.92,
            27.04,
            "You're also at an increased risk for lower back injuries when you're "
            "shoveling.",
        ),
        (27.04, 29.8, "They're usually caused when you try to lift too much snow."),
    )


def _form_field(raw: bytes, name: str) -> str | None:
    found = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n([^\r]*)\r\n', raw)
    return found.group(1).decode() if found else None


class _LoaderWhisper:
    """A real HTTP server for the remote loader: answers each response format
    from its own script, or ``answer(filename, format)`` when given, or HTTP
    ``status`` when it is not 200, and records each request's headers and form
    fields."""

    def __init__(
        self,
        timed: list[str] = (),
        untimed: list[str] = (),
        answer=None,
        status: int = 200,
    ) -> None:
        self.scripts = {"verbose_json": list(timed), "json": list(untimed)}
        self.answer = answer
        self.status = status
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        whisper = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                fmt = _form_field(raw, "response_format")
                filename = re.search(rb'filename="([^"]+)"', raw).group(1).decode()
                with whisper.lock:
                    whisper.requests.append(
                        {
                            "authorization": self.headers.get("Authorization"),
                            "format": fmt,
                            "model": _form_field(raw, "model"),
                            "language": _form_field(raw, "language"),
                            "temperature": _form_field(raw, "temperature"),
                            "seed": _form_field(raw, "seed"),
                            "filename": filename,
                        }
                    )
                    text = (
                        whisper.answer(filename, fmt)
                        if whisper.answer
                        else whisper.scripts[fmt].pop(0)
                    )
                if whisper.status != 200:
                    self.send_response(whisper.status)
                    self.end_headers()
                    return
                answer = {"text": text}
                if fmt == "verbose_json":
                    answer.update(
                        language="en",
                        duration="1.0",
                        segments=(
                            [{"start": 0.0, "end": 1.0, "text": text}] if text else []
                        ),
                    )
                body = json.dumps(answer).encode()
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


def _loader_request(
    fmt: str, temperature: str, language: str | None, filename: str = "tone.wav"
) -> dict:
    return {
        "authorization": "Bearer loader-key",
        "format": fmt,
        "model": "openai/whisper-tiny",
        "language": language,
        "temperature": temperature,
        "seed": "0",
        "filename": filename,
    }


def test_the_remote_loader_times_the_json_text_and_retries_an_empty_timed_answer(
    tmp_path,
):
    clip = tmp_path / "tone.wav"
    clip.write_bytes(wav_bytes(_noise(2.0)))
    whisper = _LoaderWhisper(timed=["", " hello"], untimed=[" hello there"])

    with whisper as url:
        result = _remote_loader_wrapper(url).transcribe(str(clip))

    assert result == {
        "text": "hello there",
        "language": "en",
        "duration": 2.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "there"},
        ],
    }
    assert whisper.requests == [
        _loader_request("verbose_json", "0.0", None),
        _loader_request("json", "0.0", "en"),
        _loader_request("verbose_json", "0.2", "en"),
    ]


def test_the_remote_loader_raises_when_sound_keeps_coming_back_empty(tmp_path):
    clip = tmp_path / "tone.wav"
    clip.write_bytes(wav_bytes(_noise(1.0)))
    whisper = _LoaderWhisper(
        timed=[""] * TRANSCRIBE_ATTEMPTS, untimed=[""] * TRANSCRIBE_ATTEMPTS
    )

    with whisper as url:
        with pytest.raises(EmptyTranscriptError) as caught:
            _remote_loader_wrapper(url).transcribe(str(clip), language="en")

    assert (caught.value.source, caught.value.chunk_index, caught.value.attempts) == (
        str(clip),
        0,
        TRANSCRIBE_ATTEMPTS,
    )
    assert whisper.requests == [
        _loader_request(fmt, str(t), "en")
        for t in FALLBACK_TEMPERATURES
        for fmt in ("verbose_json", "json")
    ]


def test_concurrent_transcriptions_through_one_loader_keep_their_own_text(tmp_path):
    names = [f"clip{index}.wav" for index in range(8)]
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(wav_bytes(_noise(2.0, seed=index)))
    ready = threading.Barrier(len(names))
    whisper = _LoaderWhisper(answer=lambda filename, fmt: f" {filename} {fmt}")

    with whisper as url:
        wrapper = _remote_loader_wrapper(url)

        def run(name: str) -> dict:
            ready.wait(timeout=10)
            return wrapper.transcribe(str(tmp_path / name), language="en")

        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            results = dict(zip(names, pool.map(run, names)))

    assert results == {
        name: {
            "text": f"{name} json",
            "language": "en",
            "duration": 2.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": f"{name} json"}],
        }
        for name in names
    }
    assert sorted(
        (request["filename"], request["format"]) for request in whisper.requests
    ) == sorted((name, fmt) for name in names for fmt in ("verbose_json", "json"))


def test_a_failing_server_raises_from_the_loader_without_another_attempt(tmp_path):
    clip = tmp_path / "tone.wav"
    clip.write_bytes(wav_bytes(_noise(1.0)))
    whisper = _LoaderWhisper(answer=lambda filename, fmt: " hello", status=503)

    with whisper as url:
        with pytest.raises(requests.HTTPError) as caught:
            _remote_loader_wrapper(url).transcribe(str(clip), language="en")

    assert str(caught.value) == (
        f"503 Server Error: Service Unavailable for url: {url}/v1/audio/transcriptions"
    )
    assert whisper.requests == [_loader_request("verbose_json", "0.0", "en")]
