"""End-to-end test: AudioAnalysisAgent.transcribe_audio against the real
deploy/whisper sidecar.

Exercises the full chain that runs in production::

    audio_url
        → AudioAnalysisAgent._get_audio_path           (MediaLocator)
        → AudioAnalysisAgent._transcribe_via_sidecar   (HTTP POST)
        → deploy/whisper /v1/transcribe                (real faster-whisper)
        → TranscriptionResult

The fixture builds and runs the ``cogniverse/whisper:inttest`` image
(``deploy/whisper/Dockerfile``) on a free port with ``MODEL_NAME=tiny`` so
first-run download stays small. ``HF_HOME`` is mounted to a named docker
volume so subsequent test runs reuse the cached weights.

Companion to ``test_audio_agent_source_url.py`` (covers ``search_audio`` →
``AudioResult.audio_url`` round-trip via Vespa). Between the two, the
production chain ``search_audio → AudioResult.audio_url →
transcribe_audio(audio_url) → /v1/transcribe → TranscriptionResult`` is
fully covered without mocking any service boundary.

Mirrors the docker-fixture pattern in
``tests/ingestion/integration/test_pylate_sidecar_docker.py``.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SIDECAR_DIR = REPO_ROOT / "deploy" / "whisper"
IMAGE_TAG = "cogniverse/whisper-fw:dev"
CONTAINER_NAME = "cogniverse-whisper-agent-inttest"
HF_VOLUME = "cogniverse-whisper-inttest-hf-cache"

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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(
    cmd: list[str], *, timeout: int = 60, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=check, timeout=timeout
    )


def _wait_for_health(base_url: str, deadline_seconds: int = 300) -> None:
    """Poll ``GET /health`` until the sidecar reports ready or we time out.

    Whisper-tiny is ~75 MB; first cold pull lands inside the start-period
    of the Dockerfile's HEALTHCHECK (180s). 300s here gives slow networks
    headroom. On failure, dumps container logs so the cause is visible.
    """
    import requests

    end = time.monotonic() + deadline_seconds
    while time.monotonic() < end:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    logs = subprocess.run(
        ["docker", "logs", CONTAINER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    raise AssertionError(
        f"whisper sidecar at {base_url} did not become healthy within "
        f"{deadline_seconds}s\n--- container logs ---\n"
        f"{logs.stdout}\n{logs.stderr}"
    )


@pytest.fixture(scope="module")
def whisper_sidecar():
    """Build + run deploy/whisper, yield its base URL, tear down on exit."""
    _run(
        ["docker", "build", "-t", IMAGE_TAG, str(SIDECAR_DIR)],
        timeout=1800,
    )

    _run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=30)
    port = _free_port()
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{port}:7998",
            "-v",
            f"{HF_VOLUME}:/root/.cache/huggingface",
            "-e",
            "MODEL_NAME=tiny",
            "-e",
            "DEVICE=cpu",
            # No WHISPER_ENGINE env — engine is hardcoded by the image.
            # cogniverse/whisper-fw is the faster-whisper variant; sibling
            # images (cogniverse/whisper-wx, cogniverse/whisper-cpp) are
            # selected by changing the IMAGE_TAG, not an env var.
            IMAGE_TAG,
        ],
        timeout=30,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url)
        yield base_url
    finally:
        _run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=30)


@pytest.fixture
def transcribing_agent(whisper_sidecar, tmp_path):
    """Bare ``AudioAnalysisAgent`` wired to the running whisper sidecar.

    Constructs through ``__new__`` to skip the heavy A2A init path; sets
    only the attributes ``transcribe_audio`` reads. ``_whisper_endpoint``
    is the load-bearing knob — when set, ``transcribe_audio`` POSTs to
    the sidecar instead of falling back to the in-process AudioTranscriber.
    """
    from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent
    from cogniverse_core.common.media import MediaConfig, MediaLocator

    agent = AudioAnalysisAgent.__new__(AudioAnalysisAgent)
    agent._whisper_model_size = "tiny"
    agent._whisper_endpoint = whisper_sidecar
    agent._audio_transcriber = None
    agent._embedding_generator = None
    agent._locator = MediaLocator(
        tenant_id="test",
        config=MediaConfig(),
        cache_root=tmp_path / "audio-cache",
    )
    return agent


class TestTranscribeAudioE2E:
    @pytest.mark.asyncio
    async def test_file_uri_resolves_then_sidecar_transcribes(
        self, transcribing_agent, whisper_sidecar
    ):
        """``file://`` URI → locator returns the real path → agent POSTs
        bytes to the sidecar → faster-whisper transcribes → TranscriptionResult
        carries detected language and segments.

        Asserts:
        - ``language != "unknown"``: the sidecar's faster-whisper actually
          ran language detection. Every failure path (timeout, 5xx, parse
          error) would either raise or pass ``"unknown"`` through.
        - ``segments`` non-empty: real audio produced at least one segment
          with start/end/text. Empty segments would mean either a silent
          input or a sidecar that returned empty output.
        - ``text`` non-empty: transcription text was actually produced.
        """
        videos_dir = (
            Path(__file__).resolve().parents[2] / "system" / "resources" / "videos"
        )
        sources = sorted(videos_dir.glob("*.mp4"))
        if not sources:
            pytest.skip(f"No test videos with audio under {videos_dir}")

        audio_url = f"file://{sources[0].resolve()}"
        result = await transcribing_agent.transcribe_audio(audio_url)

        assert result.language and result.language != "unknown", (
            f"language detection failed; result.language={result.language!r}"
        )
        assert result.segments, (
            "no segments returned — sidecar produced no output for real audio"
        )
        assert result.text.strip(), (
            "result.text is empty — sidecar returned segments but no joined text"
        )
        # Each segment must carry the canonical fields the agent's HTTP
        # parser builds from the sidecar response.
        for seg in result.segments:
            assert "start" in seg and "end" in seg and "text" in seg, (
                f"segment missing canonical fields: {seg!r}"
            )
            assert seg["end"] >= seg["start"], (
                f"segment end before start: {seg!r}"
            )
