"""Real vLLM ASR sidecar — end-to-end behavior coverage.

Complements ``test_whisper_remote_roundtrip.py`` (which uses an HTTP
stub to capture request shape) by spawning an actual
``vllm/vllm-openai-cpu`` container serving ``openai/whisper-tiny`` and
driving a transcription through ``AudioProcessor`` (remote endpoint
mode) end-to-end. Catches contract drift between vLLM versions, model
loading regressions, and OpenAI-compat response-shape changes.
"""

from __future__ import annotations

import logging
import shutil
import wave

import numpy as np
import pytest

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
