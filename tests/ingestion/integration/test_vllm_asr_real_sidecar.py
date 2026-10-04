"""Remote vLLM ASR: end-to-end behaviour of the chunked Whisper client.

``AudioProcessor`` (remote endpoint mode) runs against the cluster's
``vllm_asr`` (whisper-large-v3-turbo on vLLM ROCm), resolved through
``remote_inference``; no model starts locally. That server's decode varies from
run to run (about one answer in ten differs from the most common one), so the
tests against it pin what holds on every run: the request contract, chunk
bounds, and the retry and failure paths a proxy forces by rewriting chosen
answers (HTTP 200 with an empty transcript, or a repetition loop). Exact
transcripts and request sequences are pinned on raw answers recorded from that
server and replayed by a test-owned HTTP server.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.models.whisper_transcription import (
    FALLBACK_TEMPERATURES,
    TRANSCRIBE_ATTEMPTS,
    compression_ratio,
    pcm16_wav_samples,
    reaches_chunk_end,
    response_format,
    split_for_whisper,
)
from cogniverse_foundation.inference_specs import get_inference_service_spec
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor

pytestmark = [
    pytest.mark.requires_docker,
    pytest.mark.requires_models,
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not installed",
    ),
]


@pytest.fixture(scope="module")
def vllm_asr_url(remote_inference):
    return remote_inference.resolve("vllm_asr").base_url


def _silent_wav(path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    samples = np.zeros(int(sample_rate * seconds), dtype=np.int16)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_audio_processor_remote_against_real_vllm(vllm_asr_url, tmp_path):
    audio_path = tmp_path / "silence.wav"
    _silent_wav(audio_path, seconds=1.0)

    processor = AudioProcessor(
        logging.getLogger("test"),
        model=MODEL,
        language="en",
        endpoint=vllm_asr_url,
    )
    transcript = processor.transcribe_audio(audio_path, output_dir=tmp_path)

    assert "error" not in transcript, (
        f"remote transcription must succeed; got error: {transcript.get('error')!r}"
    )
    assert transcript["video_id"] == "silence"
    assert isinstance(transcript.get("full_text"), str)
    assert isinstance(transcript.get("segments"), list)
    assert "whisper" in transcript.get("model", "").lower(), transcript.get("model")

    written = tmp_path / "transcripts" / "silence_transcript.json"
    assert written.exists(), "AudioProcessor must persist the transcript JSON"


# 126 s of English speech: five chunks.
SPEECH_CLIP = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "sample_videos"
    / "v_-IMXSEIabMM.mp4"
)
MODEL = get_inference_service_spec("vllm_asr").model_id
LOOP = " Over and over and over." * 30
V, J = "verbose_json", "json"
RECORDED = Path(__file__).parent / "fixtures" / "vllm_asr_cluster_speech_answers.json"


def _processor(endpoint: str) -> AudioProcessor:
    return AudioProcessor(
        logging.getLogger("test"), model=MODEL, language="auto", endpoint=endpoint
    )


@pytest.fixture(scope="module")
def speech_chunks():
    return split_for_whisper(
        pcm16_wav_samples(AudioProcessor._extract_audio_wav(SPEECH_CLIP))
    )


def _chunk_of(raw: bytes, chunks) -> int:
    audio = raw[raw.index(b"RIFF") : raw.rindex(b"\r\n--")]
    for chunk in chunks:
        if chunk.wav() == audio:
            return chunk.index
    raise AssertionError("a request carried audio that is no chunk of the clip")


_FIELD = re.compile(
    rb'name="(response_format|temperature|language)"\r\n\r\n([^\r]*)\r\n'
)


class _Server:
    """A test-owned HTTP server in front of the transcription endpoint.

    Each transcription request is recorded as ``(chunk, response_format,
    temperature)`` with its language and answer. ``answer(chunk, fields)``
    gives the body to return, or ``None`` to forward to ``upstream``;
    ``rewrite(chunk, fields)`` gives replacement text for a forwarded answer
    (its segments are emptied), or ``None`` to keep it.
    """

    def __init__(
        self,
        chunks,
        upstream: str | None = None,
        answer=lambda chunk, fields: None,
        rewrite=lambda chunk, fields: None,
    ) -> None:
        self.requests: list[tuple[int, str, float]] = []
        self.languages: list[str | None] = []
        self.answers: list[dict] = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status: int, content: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                if upstream is None:
                    self._send(200, json.dumps({"data": [{"id": MODEL}]}).encode())
                    return
                response = requests.get(upstream + self.path, timeout=600)
                self._send(response.status_code, response.content)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fields = {
                    key.decode(): value.decode() for key, value in _FIELD.findall(raw)
                }
                chunk = _chunk_of(raw, chunks)
                body = answer(chunk, fields)
                if body is None:
                    response = requests.post(
                        upstream + self.path,
                        data=raw,
                        headers={"Content-Type": self.headers["Content-Type"]},
                        timeout=600,
                    )
                    response.raise_for_status()
                    body = response.json()
                    text = rewrite(chunk, fields)
                    if text is not None:
                        body = dict(body, text=text)
                        if "segments" in body:
                            body["segments"] = []
                with server.lock:
                    server.requests.append(
                        (chunk, fields["response_format"], float(fields["temperature"]))
                    )
                    server.languages.append(fields.get("language"))
                    server.answers.append(body)
                self._send(200, json.dumps(body).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def of_chunk(self, index: int) -> list[tuple[str, float]]:
        return [(fmt, t) for chunk, fmt, t in self.requests if chunk == index]

    def answers_of(self, index: int, fmt: str) -> list[dict]:
        return [
            answer
            for (chunk, sent, _), answer in zip(self.requests, self.answers)
            if (chunk, sent) == (index, fmt)
        ]

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _recorded() -> dict:
    return json.loads(RECORDED.read_text())


def _replayed(chunk: int, fields: dict) -> dict:
    """The recorded cluster answer for a chunk, format and temperature."""
    for answer in _recorded()["answers"]:
        if (
            answer["chunk"],
            answer["response_format"],
            answer["temperature"],
        ) == (chunk, fields["response_format"], float(fields["temperature"])):
            return answer["body"]
    raise AssertionError(f"no recorded answer for chunk {chunk} {fields}")


def test_the_recorded_cluster_answers_keep_their_shape(speech_chunks):
    recorded = _recorded()
    assert (recorded["server"], recorded["clip"]) == (
        {
            "image": "vllm/vllm-openai-rocm:v0.23.0",
            "model": MODEL,
            "recorded": "2026-10-04",
        },
        SPEECH_CLIP.name,
    )
    answers = recorded["answers"]
    assert sorted(
        (a["chunk"], a["response_format"], a["temperature"]) for a in answers
    ) == sorted(
        (chunk.index, fmt, t)
        for chunk in speech_chunks
        for fmt in (response_format(True), response_format(False))
        for t in FALLBACK_TEMPERATURES
    )
    assert {(a["chunk"], a["chunk_start_s"], a["chunk_len_s"]) for a in answers} == {
        (chunk.index, chunk.start_s, round(chunk.end_s - chunk.start_s, 3))
        for chunk in speech_chunks
    }
    assert {(a["response_format"], frozenset(a["body"])) for a in answers} == {
        (
            response_format(True),
            frozenset({"duration", "language", "segments", "text", "words"}),
        ),
        (response_format(False), frozenset({"text", "usage"})),
    }


def test_a_partial_timed_answer_and_json_loops_replayed_keep_every_word(
    speech_chunks, tmp_path
):
    # Recorded at temperature 0 from the cluster: chunk 3's timed answer stops
    # at 23.0 of 29.5 s, and its json answer loops at 0.0, 0.2 and 0.4.
    with _Server(speech_chunks, answer=_replayed) as server:
        transcript = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert server.requests == [
        (0, V, 0.0),
        (1, V, 0.0),
        (2, V, 0.0),
        (3, V, 0.0),
        (3, J, 0.0),
        (3, J, 0.2),
        (3, J, 0.4),
        (3, J, 0.6),
        (4, V, 0.0),
        (4, J, 0.0),
    ]
    assert server.languages == [None] + ["en"] * 9
    loops = [answer["text"] for answer in server.answers_of(3, J)[:3]]
    assert [compression_ratio(text) > 2.4 for text in loops] == [True] * 3
    third = speech_chunks[3]
    chunk_3 = [
        segment
        for segment in transcript["segments"]
        if third.start_s <= segment["start"] < third.end_s
    ]
    # The json answer at 0.6 has the words the timed answer stopped before;
    # they take the rest of the chunk.
    assert [(round(s["start"], 2), round(s["end"], 2)) for s in chunk_3] == [
        (88.6, 95.6),
        (95.6, 101.6),
        (101.6, 107.6),
        (107.6, 111.6),
        (111.6, 118.1),
    ]
    assert chunk_3[-1]["text"] == (
        "The sweat wets your clothes, the clothes stay wet. Now they get cold, and "
        "that's how you become hypothermic."
    )
    assert "hypothermic" not in " ".join(a["text"] for a in server.answers_of(3, V))
    assert transcript["full_text"] == " ".join(
        s["text"] for s in transcript["segments"]
    )


def test_concurrent_transcriptions_through_one_processor_match_a_lone_one(
    speech_chunks, tmp_path
):
    with _Server(speech_chunks, answer=_replayed) as server:
        alone = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path / "alone")
        processor = _processor(server.url)
        ready = threading.Barrier(3)

        def run(index: int) -> dict:
            ready.wait(timeout=30)
            return processor.transcribe_audio(SPEECH_CLIP, tmp_path / f"run{index}")

        with ThreadPoolExecutor(max_workers=3) as pool:
            together = list(pool.map(run, range(3)))

    assert "error" not in alone, alone.get("error")
    assert sorted(server.requests) == sorted(server.requests[:10] * 4)
    for transcript in together:
        assert {
            key: transcript[key] for key in transcript if key != "transcription_time"
        } == {key: alone[key] for key in alone if key != "transcription_time"}


def test_every_chunk_follows_the_request_contract_on_the_cluster(
    vllm_asr_url, speech_chunks, tmp_path
):
    with _Server(speech_chunks, upstream=vllm_asr_url) as server:
        transcript = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert server.languages[0] is None
    assert set(server.languages[1:]) == {"en"}
    for chunk in speech_chunks:
        sent = server.of_chunk(chunk.index)
        timed = server.answers_of(chunk.index, V)
        duration = len(chunk.samples) / 16000
        usable_to_end = [
            compression_ratio(a["text"]) <= 2.4
            and bool(a["text"].strip())
            and reaches_chunk_end(a["segments"], duration)
            for a in timed
        ]
        # Timed first; json only while no usable timed answer runs to the end.
        assert sent[0] == (V, 0.0)
        assert ((J, 0.0) in sent) == (not usable_to_end[0])
        assert [t for fmt, t in sent if fmt == V] == list(
            FALLBACK_TEMPERATURES[: len(timed)]
        )
        inside = [
            s
            for s in transcript["segments"]
            if chunk.start_s <= s["start"] < chunk.end_s
        ]
        assert inside != []
        assert all(chunk.start_s <= s["end"] <= chunk.end_s + 1e-6 for s in inside)
    assert transcript["full_text"] == " ".join(
        s["text"] for s in transcript["segments"]
    )


def test_an_empty_timed_answer_is_asked_again_at_the_next_temperature(
    vllm_asr_url, speech_chunks, tmp_path
):
    first_timed = []

    def blank_chunk_1s_first_timed(chunk, fields):
        if (chunk, fields["response_format"]) == (1, V) and not first_timed:
            first_timed.append(fields["temperature"])
            return ""
        return None

    with _Server(
        speech_chunks, upstream=vllm_asr_url, rewrite=blank_chunk_1s_first_timed
    ) as server:
        transcript = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert first_timed == ["0.0"]
    assert server.of_chunk(1)[:3] == [(V, 0.0), (J, 0.0), (V, 0.2)]


def test_a_chunk_never_timed_keeps_its_untimed_text_as_one_segment(
    vllm_asr_url, speech_chunks, tmp_path
):
    chunk = speech_chunks[1]

    with _Server(
        speech_chunks,
        upstream=vllm_asr_url,
        rewrite=lambda index, fields: (
            "" if (index, fields["response_format"]) == (1, V) else None
        ),
    ) as server:
        transcript = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    untimed = server.answers_of(1, J)
    usable = [compression_ratio(a["text"]) <= 2.4 for a in untimed]
    assert usable == [False] * (len(untimed) - 1) + [True]
    assert server.of_chunk(1) == [
        (fmt, t)
        for i, t in enumerate(FALLBACK_TEMPERATURES)
        for fmt in (V, J)
        if fmt == V or i < len(untimed)
    ]
    assert [
        segment
        for segment in transcript["segments"]
        if chunk.start_s <= segment["start"] < chunk.end_s
    ] == [
        {
            "start": chunk.start_s,
            "end": chunk.start_s + len(chunk.samples) / 16000,
            "text": " ".join(untimed[-1]["text"].split()),
        }
    ]


def test_an_untimed_answer_looping_on_every_attempt_fails_the_transcription(
    vllm_asr_url, speech_chunks, tmp_path
):
    chunk = speech_chunks[1]

    def blank_and_loop_chunk_1(index, fields):
        if index != 1:
            return None
        return "" if fields["response_format"] == V else LOOP

    with _Server(
        speech_chunks, upstream=vllm_asr_url, rewrite=blank_and_loop_chunk_1
    ) as server:
        transcript = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    ratios = ", ".join([f"{compression_ratio(LOOP):.2f}"] * TRANSCRIBE_ATTEMPTS)
    assert transcript == {
        "video_id": SPEECH_CLIP.stem,
        "error": (
            f"{SPEECH_CLIP}: chunk 1 ({chunk.start_s:.2f}-{chunk.end_s:.2f}s) came "
            f"back unusable on all {TRANSCRIBE_ATTEMPTS} attempts: "
            f"{TRANSCRIBE_ATTEMPTS} repetition loops (compression ratio {ratios}, "
            "above 2.4) and 0 empty"
        ),
        "full_text": "",
        "segments": [],
    }
    assert server.of_chunk(1) == [
        (fmt, t) for t in FALLBACK_TEMPERATURES for fmt in (V, J)
    ]
    assert not (
        tmp_path / "transcripts" / f"{SPEECH_CLIP.stem}_transcript.json"
    ).exists()


def test_a_server_answering_empty_for_speech_fails_the_transcription(
    vllm_asr_url, speech_chunks, tmp_path
):
    with _Server(
        speech_chunks, upstream=vllm_asr_url, rewrite=lambda index, fields: ""
    ) as server:
        transcript = _processor(server.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert set(transcript) == {"video_id", "error", "full_text", "segments"}
    assert (transcript["full_text"], transcript["segments"]) == ("", [])
    assert transcript["error"].startswith(f"{SPEECH_CLIP}: chunk 0 (0.00-")
    assert transcript["error"].endswith(
        "carries sound but came back with an empty transcript on all "
        f"{TRANSCRIBE_ATTEMPTS} attempts"
    )
    assert server.of_chunk(0) == [
        (fmt, t) for t in FALLBACK_TEMPERATURES for fmt in (V, J)
    ]
