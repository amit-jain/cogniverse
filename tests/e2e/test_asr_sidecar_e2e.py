"""Real ASR sidecar e2e — drives ``AudioProcessor`` remote path against
the cluster's ``cogniverse-vllm-asr`` pod.

Replaces the in-tree ``ThreadingHTTPServer`` stub that mimicked vLLM's
contract: this test hits the actual vLLM whisper deployment so model
behavior, multipart parsing, and OpenAI-compatible response handling
are all exercised end-to-end.

Requires:
- ``cogniverse up`` running with ``inference.vllm_asr.enabled=true``.
- ``kubectl`` reachable in PATH.

Service exposure: the chart leaves ``cogniverse-vllm-asr`` as ClusterIP
(no NodePort) so the fixture below opens a session-scoped
``kubectl port-forward`` to a local high port.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from tests.e2e.conftest import skip_if_no_runtime

pytestmark = [pytest.mark.e2e, skip_if_no_runtime]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def asr_sidecar_url():
    """Open a kubectl port-forward to ``svc/cogniverse-vllm-asr`` so the
    test can POST audio multipart at it. Skips if kubectl is missing or
    the service isn't deployed."""
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not in PATH; cluster ASR e2e cannot reach the sidecar")

    probe = subprocess.run(
        [
            "kubectl",
            "-n",
            "cogniverse",
            "get",
            "svc",
            "cogniverse-vllm-asr",
            "-o",
            "name",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode != 0:
        pytest.skip(
            "cogniverse-vllm-asr service not present in 'cogniverse' namespace; "
            "deploy with `cogniverse up` first"
        )

    local_port = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            "cogniverse",
            "port-forward",
            "svc/cogniverse-vllm-asr",
            f"{local_port}:8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{local_port}"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{url}/v1/models", timeout=2.0)
                if resp.status_code == 200:
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                pass
            time.sleep(1)
        else:
            proc.terminate()
            pytest.fail(
                f"port-forward to cogniverse-vllm-asr did not become reachable "
                f"on {url} within 30s"
            )
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _silent_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    """Write a mono 16-bit WAV file of the requested duration. Whisper
    accepts silence and returns an empty transcript with no error,
    which is exactly what we want — the test asserts wiring + response
    shape, not transcription accuracy."""
    samples = np.zeros(int(sample_rate * seconds), dtype=np.int16)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_audio_processor_remote_against_cluster_sidecar(asr_sidecar_url, tmp_path):
    """End-to-end: AudioProcessor(endpoint=...) → cluster vLLM ASR pod."""
    audio_path = tmp_path / "silence.wav"
    _silent_wav(audio_path, seconds=1.0)

    processor = AudioProcessor(
        logging.getLogger("test"),
        model="openai/whisper-large-v3-turbo",
        language="en",
        endpoint=asr_sidecar_url,
    )
    transcript = processor.transcribe_audio(audio_path, output_dir=tmp_path)

    assert "error" not in transcript, (
        f"remote transcription must succeed; got error: {transcript.get('error')!r}"
    )
    assert transcript["video_id"] == "silence"
    assert "full_text" in transcript and isinstance(transcript["full_text"], str)
    assert "segments" in transcript and isinstance(transcript["segments"], list)
    assert transcript.get("language"), "language field must be populated"
    # vLLM whisper-large-v3-turbo returns its served model id in the body.
    assert transcript.get("model"), "model field must round-trip from server"

    written = tmp_path / "transcripts" / "silence_transcript.json"
    assert written.exists(), "AudioProcessor must persist the transcript JSON to disk"
