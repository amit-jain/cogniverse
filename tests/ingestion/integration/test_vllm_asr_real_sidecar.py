"""Real vLLM ASR sidecar — end-to-end behavior coverage.

Complements ``test_whisper_remote_roundtrip.py`` (which uses an HTTP
stub to capture request shape) by spawning an actual
``vllm/vllm-openai-cpu`` container serving ``openai/whisper-tiny`` and
driving a transcription through ``AudioProcessor`` (remote endpoint
mode) end-to-end. Catches contract drift between vLLM versions, model
loading regressions, and OpenAI-compat response-shape changes.

The processor sends long audio one chunk per request; the server's own
transcript of the whole file pins that the chunks are the ones it cuts. That
transcript also shows the defect the chunking guards against: the server
decodes the clip's first chunk without timestamp tokens every time, so its
whole-file answer has no text for it, and the processor recovers that text
by asking for the chunk without timestamps. A proxy in front of the server
blanks chosen answers the way the cluster's ROCm server does (HTTP 200, empty
text, no segments) to pin the retry and the failure.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.models.whisper_transcription import (
    TRANSCRIBE_ATTEMPTS,
    pcm16_wav_samples,
    split_for_whisper,
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


# 126 s of English speech: five chunks, the first of which whisper-tiny on the
# CPU server decodes without timestamps.
SPEECH_CLIP = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "sample_videos"
    / "v_-IMXSEIabMM.mp4"
)
MODEL = "openai/whisper-tiny"


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


@pytest.fixture(scope="module")
def dropped_chunk_text(vllm_asr_url, whole_file_answer):
    """The server's answer without timestamps for chunk 0, asked as the
    processor's last attempt asks it."""
    audio, _ = whole_file_answer
    chunk = split_for_whisper(pcm16_wav_samples(audio))[0]
    response = requests.post(
        f"{vllm_asr_url}/v1/audio/transcriptions",
        data={"model": MODEL, "response_format": "json"},
        files={"file": ("chunk.wav", chunk.wav(), "audio/wav")},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["text"]


def _expected_text(dropped_chunk_text: str, whole: dict) -> str:
    """The whole-file transcript with chunk 0's text in front of it."""
    return " ".join([dropped_chunk_text, whole["text"]]).strip()


_RESPONSE_FORMAT = re.compile(rb'name="response_format"\r\n\r\n([a-z_]+)\r\n')


class _BlankingProxy:
    """Forwards to the real server, answering chosen transcriptions empty.

    ``formats`` records each transcription request's ``response_format``.
    """

    def __init__(self, upstream: str, blank) -> None:
        self.posts = 0
        self.blanked: list[int] = []
        self.formats: list[str] = []
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
                    number = proxy.posts
                    proxy.posts += 1
                    proxy.formats.append(
                        _RESPONSE_FORMAT.search(body).group(1).decode()
                    )
                    if blank(number):
                        proxy.blanked.append(number)
                        answer = response.json()
                        answer.update(text="", segments=[])
                        content = json.dumps(answer).encode()
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

    def __enter__(self) -> "_BlankingProxy":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def test_chunked_transcript_is_the_servers_whole_file_transcript_and_the_chunk_it_drops(
    vllm_asr_url, whole_file_answer, dropped_chunk_text, tmp_path
):
    audio, whole = whole_file_answer
    chunks = split_for_whisper(pcm16_wav_samples(audio))
    assert len(chunks) == 5
    # The server times each chunk's segments from chunk index x 30 s: its
    # whole-file answer has text for chunks 1-4 and none for chunk 0.
    answered = {int(segment["start"] // 30) for segment in whole["segments"]}
    assert sorted(answered) == [1, 2, 3, 4]

    transcript = _processor(vllm_asr_url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert transcript["full_text"] == _expected_text(dropped_chunk_text, whole)
    assert transcript["language"] == whole["language"]
    assert transcript["duration"] == float(whole["duration"])
    assert transcript["segments"][0] == {
        "start": 0.0,
        "end": chunks[0].end_s,
        "text": dropped_chunk_text.strip(),
    }
    assert [segment["text"] for segment in transcript["segments"][1:]] == [
        segment["text"].strip() for segment in whole["segments"]
    ]
    # Each segment sits inside the chunk that produced it, timed from where
    # that chunk starts in the clip.
    bounds = [(chunk.start_s, chunk.end_s) for chunk in chunks]
    for segment in transcript["segments"]:
        assert any(
            start <= segment["start"] <= segment["end"] <= end + 0.02
            for start, end in bounds
        ), segment


def test_an_empty_answer_from_the_server_is_asked_again_and_the_text_is_whole(
    vllm_asr_url, whole_file_answer, dropped_chunk_text, tmp_path
):
    _, whole = whole_file_answer
    # Requests 0-2 are chunk 0's; request 3 is chunk 1's first attempt.
    with _BlankingProxy(vllm_asr_url, blank=lambda number: number == 3) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert transcript["full_text"] == _expected_text(dropped_chunk_text, whole)
    assert (proxy.posts, proxy.blanked) == (8, [3])
    assert (
        proxy.formats == ["verbose_json", "verbose_json", "json"] + ["verbose_json"] * 5
    )


def test_a_server_answering_empty_for_speech_fails_the_transcription(
    vllm_asr_url, tmp_path
):
    with _BlankingProxy(vllm_asr_url, blank=lambda number: True) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert set(transcript) == {"video_id", "error", "full_text", "segments"}
    assert (transcript["full_text"], transcript["segments"]) == ("", [])
    assert transcript["error"].startswith(f"{SPEECH_CLIP}: chunk 0 (0.00-")
    assert transcript["error"].endswith(
        "carries sound but came back with an empty transcript on all "
        f"{TRANSCRIBE_ATTEMPTS} attempts"
    )
    assert proxy.posts == TRANSCRIBE_ATTEMPTS


def test_a_chunk_blank_whenever_timestamped_keeps_the_servers_untimed_text(
    vllm_asr_url, whole_file_answer, tmp_path
):
    audio, whole = whole_file_answer
    chunk = split_for_whisper(pcm16_wav_samples(audio))[1]
    untimed = requests.post(
        f"{vllm_asr_url}/v1/audio/transcriptions",
        data={"model": MODEL, "response_format": "json", "language": whole["language"]},
        files={"file": ("chunk.wav", chunk.wav(), "audio/wav")},
        timeout=600,
    ).json()["text"]

    # Requests 0-2 are chunk 0's; requests 3 and 4 are chunk 1's timestamped
    # attempts, and request 5 asks without timestamps and is answered by the
    # server.
    with _BlankingProxy(vllm_asr_url, blank=lambda number: number in (3, 4)) as proxy:
        transcript = _processor(proxy.url).transcribe_audio(SPEECH_CLIP, tmp_path)

    assert "error" not in transcript, transcript.get("error")
    assert [
        segment
        for segment in transcript["segments"]
        if chunk.start_s <= segment["start"] < chunk.end_s
    ] == [
        {
            "start": chunk.start_s,
            "end": chunk.start_s + (chunk.end_s - chunk.start_s),
            "text": untimed.strip(),
        }
    ]
    assert (proxy.posts, proxy.blanked) == (9, [3, 4])
    assert (
        proxy.formats
        == ["verbose_json", "verbose_json", "json"] * 2 + ["verbose_json"] * 3
    )
