"""Remote transcription authenticates against the ASR endpoint.

``AudioProcessor`` posted the audio multipart (and the ``/v1/models`` probe)
with no Authorization header, so a Modal-hosted Whisper answered 401; the
``transcribe_audio`` catch-all logged it and returned an empty transcript.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import requests

from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

MODAL = "https://amit-jain--cogniverse-vllm-asr-inference.modal.run"
IN_CLUSTER = "http://cogniverse-vllm-asr:8000"


def _processor(endpoint: str) -> AudioProcessor:
    return AudioProcessor(logging.getLogger("test"), language="en", endpoint=endpoint)


def test_modal_endpoint_headers_carry_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    assert dict(_processor(MODAL).auth_headers()) == {
        "Authorization": "Bearer real-bearer"
    }


def test_in_cluster_endpoint_sends_no_credential(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    assert dict(_processor(IN_CLUSTER).auth_headers()) == {}


def test_modal_endpoint_without_a_bearer_fails_naming_the_variable(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        _processor(MODAL).auth_headers()


class _Response:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def test_both_remote_calls_carry_the_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    monkeypatch.setattr(
        AudioProcessor, "_extract_audio_wav", staticmethod(lambda p: b"RIFF")
    )
    calls: list[tuple] = []

    def _get(url, headers=None, timeout=None):
        calls.append(("GET", url, dict(headers)))
        return _Response({"data": [{"id": "openai/whisper-large-v3"}]})

    def _post(url, data=None, files=None, headers=None, timeout=None):
        calls.append(("POST", url, dict(headers), data, sorted(files)))
        return _Response(
            {
                "text": " hello world ",
                "language": "en",
                "duration": 1.5,
                "segments": [{"start": 0.0, "end": 1.5, "text": " hello world "}],
            }
        )

    monkeypatch.setattr(requests, "get", _get)
    monkeypatch.setattr(requests, "post", _post)

    transcript = _processor(MODAL)._transcribe_remote(Path("clip.mp4"), "clip")

    bearer = {"Authorization": "Bearer real-bearer"}
    assert calls == [
        ("GET", f"{MODAL}/v1/models", bearer),
        (
            "POST",
            f"{MODAL}/v1/audio/transcriptions",
            bearer,
            {
                "model": "openai/whisper-large-v3",
                "response_format": "verbose_json",
                "language": "en",
            },
            ["file"],
        ),
    ]
    assert transcript == {
        "video_id": "clip",
        "video_path": "clip.mp4",
        "model": "openai/whisper-large-v3",
        "language": "en",
        "duration": 1.5,
        "full_text": "hello world",
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello world"}],
    }
