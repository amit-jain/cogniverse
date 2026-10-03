"""Real vLLM ASR sidecar — end-to-end behavior coverage.

Complements ``test_whisper_remote_roundtrip.py`` (which uses an HTTP
stub to capture request shape) by spawning an actual
``vllm/vllm-openai-cpu`` container serving ``openai/whisper-tiny`` and
driving a transcription through ``AudioProcessor`` (remote endpoint
mode) end-to-end. Catches contract drift between vLLM versions, model
loading regressions, and OpenAI-compat response-shape changes.

The processor asks each chunk with timestamps and without; the server's own
transcript of the whole file pins that the chunks are the ones it cuts. That
transcript also shows the text the server loses: whisper-tiny decodes the
clip's first chunk without timestamp tokens and stops its timed decode of the
second at 18 s, so the whole-file answer has nothing for the first chunk and
nothing after 18 s of the second; the processor keeps that text from the
untimed answers. A proxy in front of the server records every request and
rewrites chosen answers the way the cluster's ROCm server answers some (HTTP
200 with an empty transcript, or a repetition loop) to pin the retry and the
failure.
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
    split_for_whisper,
    wav_bytes,
)
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
def vllm_asr_url(vllm_sidecar):
    return vllm_sidecar.spawn(
        model="openai/whisper-tiny",
        extra_args=[
            "--runner",
            "generate",
            "--max-model-len",
            "448",
            "--gpu-memory-utilization",
            "0.05",
            # One sequence at a time, as the cluster's ROCm server runs, so
            # the server decodes each chunk of a whole file alone, as it
            # decodes each of the client's requests. One sequence caps the
            # step at max-model-len unless the budget is named, and the
            # encoder needs its 1500 audio tokens in one step.
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "2048",
        ],
    )


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
        model="openai/whisper-tiny",
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
# Bells and clicks; whisper-tiny answers its first chunk's untimed request with
# "[Bell]" repeated to the token limit.
BELLS_CLIP = SPEECH_CLIP.with_name("v_-vnSFKJNB94.mp4")
MODEL = "openai/whisper-tiny"
LOOP = " Over and over and over." * 30
V, J = "verbose_json", "json"


def _processor(endpoint: str) -> AudioProcessor:
    return AudioProcessor(
        logging.getLogger("test"), model=MODEL, language="auto", endpoint=endpoint
    )


@pytest.fixture(scope="module")
def whole_file_answer(vllm_asr_url):
    """The server's transcript of the whole clip, chunked by the server."""
    audio = AudioProcessor._extract_audio_wav(SPEECH_CLIP)
    response = requests.post(
        f"{vllm_asr_url}/v1/audio/transcriptions",
        data={"model": MODEL, "response_format": "verbose_json"},
        files={"file": ("whole.wav", audio, "audio/wav")},
        timeout=600,
    )
    response.raise_for_status()
    return audio, response.json()


_FIELD = re.compile(
    rb'name="(response_format|temperature|language)"\r\n\r\n([^\r]*)\r\n'
)


class _Proxy:
    """Forwards to the real server and records each transcription request's
    ``(response_format, temperature)`` and its answer; ``rewrite(number,
    fields)`` returns replacement text for chosen answers (their segments are
    emptied)."""

    def __init__(self, upstream: str, rewrite=lambda number, fields: None) -> None:
        self.requests: list[tuple[str, float]] = []
        self.languages: list[str | None] = []
        self.answers: list[dict] = []
        self.rewritten: list[int] = []
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _relay(self, method: str) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                response = requests.request(
                    method,
                    upstream + self.path,
                    data=body or None,
                    headers={"Content-Type": self.headers.get("Content-Type", "")},
                    timeout=600,
                )
                content = response.content
                if method == "POST":
                    fields = {
                        key.decode(): value.decode()
                        for key, value in _FIELD.findall(body)
                    }
                    number = len(proxy.requests)
                    proxy.requests.append(
                        (fields["response_format"], float(fields["temperature"]))
                    )
                    proxy.languages.append(fields.get("language"))
                    answer = response.json()
                    text = rewrite(number, fields)
                    if text is not None:
                        proxy.rewritten.append(number)
                        answer["text"] = text
                        if "segments" in answer:
                            answer["segments"] = []
                        content = json.dumps(answer).encode()
                    proxy.answers.append(answer)
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                self._relay("GET")

            def do_POST(self):
                self._relay("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "_Proxy":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


# whisper-tiny on the CPU server: chunk 0's timed decode has no timestamp
# pair until temperature 0.6; every other chunk answers both requests at 0.0.
SPEECH_REQUESTS = [(V, 0.0), (J, 0.0), (V, 0.2), (V, 0.4), (V, 0.6)] + [
    (V, 0.0),
    (J, 0.0),
] * 4


def test_each_chunk_keeps_its_untimed_text_timed_by_its_timed_answer(
    vllm_asr_url, whole_file_answer, tmp_path
):
    audio, whole = whole_file_answer
    chunks = split_for_whisper(pcm16_wav_samples(audio))
    assert len(chunks) == 5
    # The server times each chunk's segments from chunk index x 30 s: its
    # whole-file answer has no text for chunk 0, and none after 18 s of chunk 1.
    assert sorted({int(segment["start"] // 30) for segment in whole["segments"]}) == [
        1,
        2,
        3,
        4,
    ]
    assert [
        (segment["start"], segment["end"])
        for segment in whole["segments"]
        if 30.0 <= segment["start"] < 60.0
    ] == [(30.0, 48.0)]

    with _Proxy(vllm_asr_url) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert proxy.requests == SPEECH_REQUESTS
    assert proxy.languages == [None] + ["en"] * (len(SPEECH_REQUESTS) - 1)
    assert transcript["language"] == whole["language"] == "en"
    assert transcript["duration"] == float(whole["duration"])
    assert [
        (round(segment["start"], 2), round(segment["end"], 2))
        for segment in transcript["segments"]
    ] == [
        (0.0, 14.8),
        (14.8, 22.88),
        (22.88, 29.8),
        (29.8, 47.8),
        (47.8, 58.8),
        (58.8, 65.6),
        (66.24, 71.92),
        (71.92, 77.2),
        (77.2, 82.32),
        (82.32, 87.52),
        (87.52, 88.6),
        (88.6, 101.6),
        (101.6, 110.6),
        (110.6, 117.6),
        (118.1, 124.1),
    ]
    texts = [segment["text"] for segment in transcript["segments"]]
    assert transcript["full_text"] == " ".join(texts)
    assert texts[0] == (
        "When a big snowstorm strikes at some point, we're going to have to dig out "
        "from under it. But snow shoveling injuries land thousands of people in "
        "the emergency room each year. And 96% of them happen at home."
    )
    # Chunk 1's timed decode stopped at 18 s; the rest is its untimed text,
    # which the server's whole-file answer does not have.
    assert texts[4] == (
        "- Sprayne strains and fractures are always near the top of the list of "
        "snow shoveling related injuries. They're usually caused by slipping and "
        "twisting. Over exertion is also considered a common cause"
    )
    assert "Sprayne strains" not in whole["text"]


def test_an_empty_timed_answer_is_asked_again_at_the_next_temperature(
    vllm_asr_url, tmp_path
):
    # Request 5 is chunk 1's timed request.
    with _Proxy(
        vllm_asr_url, rewrite=lambda number, fields: "" if number == 5 else None
    ) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert proxy.rewritten == [5]
    assert proxy.requests == SPEECH_REQUESTS[:7] + [(V, 0.2)] + SPEECH_REQUESTS[7:]
    # The timed answer at 0.2 also stops at 18 s, so the untimed text after it
    # fills the rest of chunk 1 as before.
    assert [
        segment for segment in transcript["segments"] if 47.0 < segment["start"] < 58.8
    ] == [
        {
            "start": 29.8 + 18.0,
            "end": 29.8 + 29.0,
            "text": "- Sprayne strains and fractures are always near the top of the "
            "list of snow shoveling related injuries. They're usually caused by "
            "slipping and twisting. Over exertion is also considered a common cause",
        }
    ]


def test_a_chunk_never_timed_keeps_its_untimed_text_as_one_segment(
    vllm_asr_url, tmp_path
):
    chunk = split_for_whisper(
        pcm16_wav_samples(AudioProcessor._extract_audio_wav(SPEECH_CLIP))
    )[1]
    # Chunk 1's requests are 5 (timed), 6 (untimed) and 7-11 (timed again).
    timed_for_chunk_1 = {5, 7, 8, 9, 10, 11}

    def blank_chunk_1_timed(number, fields):
        if number in timed_for_chunk_1 and fields["response_format"] == V:
            return ""
        return None

    with _Proxy(vllm_asr_url, rewrite=blank_chunk_1_timed) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert proxy.requests[5:12] == [(V, 0.0), (J, 0.0)] + [
        (V, t) for t in FALLBACK_TEMPERATURES[1:]
    ]
    untimed = proxy.answers[6]["text"]
    assert [
        segment
        for segment in transcript["segments"]
        if chunk.start_s <= segment["start"] < chunk.end_s
    ] == [
        {
            "start": chunk.start_s,
            "end": chunk.start_s + len(chunk.samples) / 16000,
            "text": " ".join(untimed.split()),
        }
    ]


def test_an_untimed_answer_looping_on_every_attempt_fails_the_transcription(
    vllm_asr_url, tmp_path
):
    chunk = split_for_whisper(
        pcm16_wav_samples(AudioProcessor._extract_audio_wav(SPEECH_CLIP))
    )[1]

    def loop_after_chunk_0(number, fields):
        return LOOP if number >= 5 and fields["response_format"] == J else None

    with _Proxy(vllm_asr_url, rewrite=loop_after_chunk_0) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

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
    assert proxy.requests == SPEECH_REQUESTS[:5] + [(V, 0.0)] + [
        (J, t) for t in FALLBACK_TEMPERATURES
    ]
    assert not (
        tmp_path / "transcripts" / f"{SPEECH_CLIP.stem}_transcript.json"
    ).exists()


def test_a_server_answering_empty_for_speech_fails_the_transcription(
    vllm_asr_url, tmp_path
):
    with _Proxy(vllm_asr_url, rewrite=lambda number, fields: "") as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert set(transcript) == {"video_id", "error", "full_text", "segments"}
    assert (transcript["full_text"], transcript["segments"]) == ("", [])
    assert transcript["error"].startswith(f"{SPEECH_CLIP}: chunk 0 (0.00-")
    assert transcript["error"].endswith(
        "carries sound but came back with an empty transcript on all "
        f"{TRANSCRIBE_ATTEMPTS} attempts"
    )
    assert proxy.requests == [(fmt, t) for t in FALLBACK_TEMPERATURES for fmt in (V, J)]


def test_a_real_loop_is_asked_again_and_never_reaches_the_transcript(
    vllm_asr_url, tmp_path
):
    chunk = split_for_whisper(
        pcm16_wav_samples(AudioProcessor._extract_audio_wav(BELLS_CLIP))
    )[0]
    clip = tmp_path / "bells.wav"
    clip.write_bytes(wav_bytes(chunk.samples))

    with _Proxy(vllm_asr_url) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(clip, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert proxy.requests == [(V, 0.0), (J, 0.0), (J, 0.2)]
    looped = proxy.answers[1]["text"]
    assert compression_ratio(looped) > 2.4
    assert proxy.answers[2]["text"] == " [Bell]"
    assert (transcript["full_text"], transcript["segments"]) == (
        "[Bell]",
        [{"start": 0.0, "end": 2.0, "text": "[Bell]"}],
    )


def test_concurrent_transcriptions_through_one_processor_match_a_lone_one(
    vllm_asr_url, tmp_path
):
    alone = _processor(vllm_asr_url).transcribe_audio(SPEECH_CLIP, tmp_path / "alone")
    processor = _processor(vllm_asr_url)
    ready = threading.Barrier(3)

    def run(index: int) -> dict:
        ready.wait(timeout=30)
        return processor.transcribe_audio(SPEECH_CLIP, tmp_path / f"run{index}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        together = list(pool.map(run, range(3)))

    assert "error" not in alone, alone.get("error")
    for transcript in together:
        assert {
            key: transcript[key] for key in transcript if key != "transcription_time"
        } == {key: alone[key] for key in alone if key != "transcription_time"}
