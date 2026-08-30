"""
Unit tests for Audio Analysis Agent

Tests audio transcription with Whisper, audio search, and Vespa integration.
"""

import asyncio
import json
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import numpy as np
import pytest
import requests

from cogniverse_agents.audio_analysis_agent import (
    AudioAnalysisAgent,
    AudioAnalysisDeps,
    AudioResult,
    TranscriptionResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@contextmanager
def _transcription_server(
    response: object | Callable[[bytes], object],
    *,
    raw_body: bytes | None = None,
    arrival_barrier: threading.Barrier | None = None,
    release_response: threading.Event | None = None,
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    captured_requests: list[dict[str, object]] = []
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            request_body = self.rfile.read(content_length)
            with request_lock:
                captured_requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "content_type": self.headers.get("Content-Type"),
                        "body": request_body,
                    }
                )
            if arrival_barrier is not None:
                arrival_barrier.wait(timeout=5)
            if release_response is not None:
                assert release_response.wait(timeout=5)

            if raw_body is not None:
                response_body = raw_body
            else:
                payload = response(request_body) if callable(response) else response
                response_body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            try:
                self.wfile.write(response_body)
            except BrokenPipeError:
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", captured_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _agent_for_remote_transcription(base_url: str, authorization: str):
    return AudioAnalysisAgent(
        deps=AudioAnalysisDeps(
            tenant_id="test_tenant",
            whisper_endpoint=base_url,
            whisper_headers={"Authorization": authorization},
        )
    )


def _audio_search_hit(
    *,
    audio_id: str = "audio_001",
    source_url: str = "http://example.com/audio1.mp3",
    title: str = "Test Podcast",
    transcript: str = "This is a test podcast about AI",
    duration: float = 300.0,
    score: float = 0.91,
    speaker_labels: list[str] | None = None,
    detected_events: list[str] | None = None,
    language: str = "en",
):
    metadata = {
        "audio_id": audio_id,
        "source_url": source_url,
        "audio_title": title,
        "audio_transcript": transcript,
        "audio_duration": duration,
        "speaker_labels": speaker_labels if speaker_labels is not None else [],
        "detected_events": detected_events if detected_events is not None else [],
        "audio_language": language,
    }
    document = SimpleNamespace(id=audio_id, text_content=None, metadata=metadata)
    return SimpleNamespace(document=document, score=score)


class TestAudioAnalysisAgent:
    """Unit tests for AudioAnalysisAgent"""

    def setup_method(self):
        """Set up test fixtures"""
        self._schema_backend = MagicMock()
        self._schema_backend.schema_exists = MagicMock(return_value=True)
        self._schema_backend_patch = patch(
            "cogniverse_runtime.admin.tenant_manager.get_backend",
            return_value=self._schema_backend,
        )
        self._schema_backend_patch.start()
        self.agent = AudioAnalysisAgent(
            deps=AudioAnalysisDeps(
                tenant_id="test_tenant",
                vespa_endpoint="http://localhost:8080",
                whisper_model_size="base",
            ),
            port=8006,
        )

    def teardown_method(self):
        self._schema_backend_patch.stop()

    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent is not None
        assert self.agent._audio_transcriber is None  # Lazy loaded
        assert self.agent._whisper_model_size == "base"
        assert self.agent._vespa_endpoint == "http://localhost:8080"

    def test_whisper_headers_reject_non_bearer_credentials(self):
        with pytest.raises(ValueError, match="whisper_headers.*Authorization"):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="https://whisper.internal.example.com",
                whisper_headers={"Modal-Key": "wrong-auth-scheme"},
            )

    def test_whisper_headers_require_endpoint(self):
        with pytest.raises(
            ValueError, match="whisper_headers requires whisper_endpoint"
        ):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_headers={"Authorization": "Bearer custom-whisper-key"},
            )

    def test_custom_whisper_endpoint_accepts_one_canonical_mapping(self):
        agent = AudioAnalysisAgent(
            deps=AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="https://whisper.internal.example.com",
                whisper_headers={"Authorization": "Bearer custom-whisper-key"},
            )
        )

        assert agent._whisper_headers == {"Authorization": "Bearer custom-whisper-key"}
        with pytest.raises(TypeError):
            agent._whisper_headers["Authorization"] = "Bearer replacement"

    def test_modal_whisper_credentials_are_resolved_once_across_revalidation(
        self, monkeypatch
    ):
        initial_token = "initial-modal-whisper-key"
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", initial_token)
        deps = AudioAnalysisDeps(
            tenant_id="test_tenant",
            whisper_endpoint="https://whisper.modal.run",
        )

        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "rotated-modal-whisper-key")
        revalidated = AudioAnalysisDeps.model_validate(deps)
        agent = AudioAnalysisAgent(deps=revalidated)

        assert revalidated is deps
        assert deps.whisper_headers == {}
        assert agent._whisper_headers == {"Authorization": f"Bearer {initial_token}"}

    @pytest.mark.parametrize(
        "headers",
        [{"Authorization": "Bearer caller-specific-key"}, {}],
    )
    def test_modal_whisper_endpoint_rejects_dependency_headers(
        self, monkeypatch, headers
    ):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

        with pytest.raises(ValueError, match="whisper_headers.*Modal"):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="https://whisper.modal.run",
                whisper_headers=headers,
            )

    @patch("requests.post")
    def test_modal_whisper_endpoint_requires_https_before_request(
        self, mock_post, monkeypatch
    ):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

        with pytest.raises(ValueError, match="Modal inference endpoints require HTTPS"):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="http://whisper.modal.run",
            )

        mock_post.assert_not_called()

    @patch("requests.post")
    def test_modal_whisper_endpoint_requires_environment_credential_before_request(
        self, mock_post, monkeypatch
    ):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

        with pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="https://whisper.modal.run",
            )

        mock_post.assert_not_called()

    @patch("requests.post")
    def test_modal_whisper_endpoint_rejects_blank_credential_before_request(
        self, mock_post, monkeypatch
    ):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "   ")

        with pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="https://whisper.modal.run",
            )

        mock_post.assert_not_called()

    def test_clap_headers_reject_non_bearer_credentials(self):
        with pytest.raises(ValueError, match="clap_headers.*Authorization"):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                clap_endpoint="https://clap.modal.run",
                clap_headers={"Modal-Key": "wrong-auth-scheme"},
            )

    def test_clap_headers_require_endpoint(self):
        with pytest.raises(ValueError, match="clap_headers requires clap_endpoint"):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                clap_headers={"Authorization": "Bearer modal-clap-secret"},
            )

    def test_embedding_generator_receives_environment_clap_credentials(
        self, monkeypatch
    ):
        initial_token = "initial-production-key"
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", initial_token)
        deps = AudioAnalysisDeps(
            tenant_id="test_tenant",
            clap_endpoint="https://clap.modal.run",
        )
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY")

        revalidated = AudioAnalysisDeps.model_validate(deps)
        agent = AudioAnalysisAgent(deps=revalidated)

        generator = agent.embedding_generator

        assert revalidated is deps
        assert deps.clap_headers == {}
        assert generator._clap_endpoint_url == "https://clap.modal.run"
        assert generator._clap_headers == {"Authorization": f"Bearer {initial_token}"}
        with pytest.raises(TypeError):
            generator._clap_headers["Authorization"] = "Bearer replacement"
        assert initial_token not in repr(agent.deps)

    @pytest.mark.parametrize(
        "headers",
        [{"Authorization": "Bearer caller-specific-key"}, {}],
    )
    def test_modal_clap_endpoint_rejects_dependency_headers(self, monkeypatch, headers):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

        with pytest.raises(ValueError, match="clap_headers.*Modal"):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                clap_endpoint="https://clap.modal.run",
                clap_headers=headers,
            )

    def test_modal_clap_endpoint_requires_environment_credential(self, monkeypatch):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

        with pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ):
            AudioAnalysisDeps(
                tenant_id="test_tenant",
                clap_endpoint="https://clap.modal.run",
            )

    @patch("cogniverse_agents.audio_analysis_agent.AudioTranscriber")
    def test_audio_transcriber_lazy_loading(self, mock_transcriber_class):
        """Test AudioTranscriber is lazy loaded on first access"""
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        # Access audio_transcriber property
        transcriber = self.agent.audio_transcriber

        # Verify transcriber was loaded
        assert transcriber == mock_transcriber
        mock_transcriber_class.assert_called_once_with(model_size="base")

    def test_concurrent_first_transcriber_access_builds_one_instance(self, monkeypatch):
        callers = threading.Barrier(12)
        constructions = 0
        constructed = object()
        construction_changed = threading.Condition()
        release_construction = threading.Event()

        def build(*, model_size):
            nonlocal constructions
            assert model_size == "base"
            with construction_changed:
                constructions += 1
                construction_changed.notify_all()
            assert release_construction.wait(timeout=3)
            return constructed

        monkeypatch.setattr(
            "cogniverse_agents.audio_analysis_agent.AudioTranscriber",
            build,
        )

        def load(_):
            callers.wait(timeout=3)
            return self.agent.audio_transcriber

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(load, index) for index in range(12)]
            with construction_changed:
                construction_changed.wait_for(
                    lambda: constructions == 12,
                    timeout=0.5,
                )
            release_construction.set()
            loaded = [future.result(timeout=3) for future in futures]

        assert not callers.broken
        assert constructions == 1
        assert loaded == [constructed] * 12

    def test_concurrent_first_embedding_generator_access_builds_one_instance(
        self, monkeypatch
    ):
        callers = threading.Barrier(12)
        constructions = 0
        constructed = object()
        construction_changed = threading.Condition()
        release_construction = threading.Event()

        def build():
            nonlocal constructions
            with construction_changed:
                constructions += 1
                construction_changed.notify_all()
            assert release_construction.wait(timeout=3)
            return constructed

        monkeypatch.setattr(
            "cogniverse_agents.audio_analysis_agent.AudioEmbeddingGenerator",
            build,
        )

        def load(_):
            callers.wait(timeout=3)
            return self.agent.embedding_generator

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(load, index) for index in range(12)]
            with construction_changed:
                construction_changed.wait_for(
                    lambda: constructions == 12,
                    timeout=0.5,
                )
            release_construction.set()
            loaded = [future.result(timeout=3) for future in futures]

        assert not callers.broken
        assert constructions == 1
        assert loaded == [constructed] * 12

    @pytest.mark.asyncio
    async def test_search_audio_transcript_mode_uses_backend(self):
        backend = MagicMock()
        backend.search = MagicMock(return_value=[])
        self.agent._get_backend = MagicMock(return_value=backend)
        self.agent._embedding_generator = MagicMock(
            generate_acoustic_text_embedding=MagicMock(
                side_effect=AssertionError("acoustic embedding should not be used")
            )
        )

        results = await self.agent.search_audio(
            query="AI podcast", search_mode="transcript", limit=20
        )

        assert results == []
        backend.search.assert_called_once_with(
            {
                "query": "AI podcast",
                "type": "audio",
                "strategy": "transcript_search",
                "tenant_id": "test_tenant",
                "top_k": 20,
            }
        )
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_audio_default_mode_is_semantic(self):
        backend = MagicMock()
        backend.search = MagicMock(return_value=[])
        self.agent._get_backend = MagicMock(return_value=backend)
        self.agent._embedding_generator = MagicMock(
            generate_acoustic_text_embedding=MagicMock(
                side_effect=AssertionError("acoustic embedding should not be used")
            )
        )

        results = await self.agent.search_audio(query="machine learning", limit=10)

        assert results == []
        backend.search.assert_called_once_with(
            {
                "query": "machine learning",
                "type": "audio",
                "strategy": "phased_semantic",
                "tenant_id": "test_tenant",
                "top_k": 10,
            }
        )
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_audio_semantic_mode_maps_backend_hits_exactly(self):
        backend_hit = _audio_search_hit(
            speaker_labels=["host", "guest"],
            detected_events=["intro"],
        )
        backend = MagicMock()
        backend.search = MagicMock(return_value=[backend_hit])
        self.agent._get_backend = MagicMock(return_value=backend)
        self.agent._embedding_generator = MagicMock(
            generate_acoustic_text_embedding=MagicMock(
                side_effect=AssertionError("acoustic embedding should not be used")
            )
        )

        results = await self.agent.search_audio(
            query="what was said about AI", search_mode="semantic", limit=7
        )

        backend.search.assert_called_once_with(
            {
                "query": "what was said about AI",
                "type": "audio",
                "strategy": "phased_semantic",
                "tenant_id": "test_tenant",
                "top_k": 7,
            }
        )
        expected = AudioResult(
            audio_id="audio_001",
            audio_url="http://example.com/audio1.mp3",
            title="Test Podcast",
            transcript="This is a test podcast about AI",
            duration=300.0,
            relevance_score=0.91,
            speaker_labels=["host", "guest"],
            detected_events=["intro"],
            language="en",
            metadata=backend_hit.document.metadata,
        ).model_dump()
        assert [result.model_dump() for result in results] == [expected]
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_audio_acoustic_mode_still_uses_clap(self):
        query_embedding = np.ones(512, dtype=np.float32)
        self.agent._embedding_generator = MagicMock()
        self.agent._embedding_generator.generate_acoustic_text_embedding.return_value = query_embedding
        self.agent._search_by_acoustic_embedding = AsyncMock(return_value=[])
        self.agent._get_backend = MagicMock(
            side_effect=AssertionError("backend should not be used")
        )

        results = await self.agent.search_audio(
            query="sounds like thunder", search_mode="acoustic", limit=3
        )

        assert results == []
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_called_once_with(
            "sounds like thunder"
        )
        self.agent._search_by_acoustic_embedding.assert_awaited_once()
        call_args = self.agent._search_by_acoustic_embedding.await_args.args
        assert call_args[0].shape == (512,)
        assert call_args[1] == 3
        self.agent._get_backend.assert_not_called()
        assert self._schema_backend.schema_exists.call_args_list == [
            call("audio_content", tenant_id="test_tenant")
        ]

    @pytest.mark.asyncio
    async def test_search_audio_acoustic_mode_returns_empty_when_schema_missing(self):
        self._schema_backend.schema_exists.return_value = False
        self.agent._embedding_generator = MagicMock(
            generate_acoustic_text_embedding=MagicMock(
                side_effect=AssertionError("acoustic embedding should not be used")
            )
        )
        self.agent._search_by_acoustic_embedding = AsyncMock(
            side_effect=AssertionError("acoustic search must not run")
        )

        results = await self.agent.search_audio(
            query="sounds like thunder", search_mode="acoustic", limit=3
        )

        assert results == []
        assert self._schema_backend.schema_exists.call_args_list == [
            call("audio_content", tenant_id="test_tenant")
        ]
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()
        self.agent._search_by_acoustic_embedding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_audio_acoustic_mode_raises_when_schema_lookup_fails(self):
        self._schema_backend.schema_exists.side_effect = RuntimeError(
            "schema registry unavailable"
        )
        self.agent._embedding_generator = MagicMock(
            generate_acoustic_text_embedding=MagicMock(
                side_effect=AssertionError("acoustic embedding should not be used")
            )
        )

        with pytest.raises(RuntimeError, match="schema registry unavailable"):
            await self.agent.search_audio(
                query="sounds like thunder", search_mode="acoustic", limit=3
            )

        assert self._schema_backend.schema_exists.call_args_list == [
            call("audio_content", tenant_id="test_tenant")
        ]
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_audio_rejects_unknown_mode(self):
        backend = MagicMock()
        backend.search = MagicMock(
            side_effect=AssertionError("no backend search for an unknown mode")
        )
        self.agent._get_backend = MagicMock(return_value=backend)

        with pytest.raises(
            ValueError,
            match=(
                "Unknown audio search mode 'lexical'; expected one of "
                "acoustic, transcript, semantic, hybrid"
            ),
        ):
            await self.agent.search_audio(
                query="machine learning", search_mode="lexical", limit=10
            )

        backend.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_audio_backend_failure_raises(self):
        backend = MagicMock()
        backend.search.side_effect = RuntimeError("backend unavailable")
        self.agent._get_backend = MagicMock(return_value=backend)
        self.agent._embedding_generator = MagicMock(
            generate_acoustic_text_embedding=MagicMock(
                side_effect=AssertionError("acoustic embedding should not be used")
            )
        )

        with pytest.raises(RuntimeError, match="backend unavailable"):
            await self.agent.search_audio(
                query="machine learning", search_mode="semantic", limit=10
            )

        backend.search.assert_called_once_with(
            {
                "query": "machine learning",
                "type": "audio",
                "strategy": "phased_semantic",
                "tenant_id": "test_tenant",
                "top_k": 10,
            }
        )
        self.agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(AudioAnalysisAgent, "audio_transcriber", new_callable=PropertyMock)
    async def test_transcribe_audio(self, mock_transcriber, tmp_path):
        """Test audio transcription using Whisper"""
        # Mock AudioTranscriber
        mock_transcriber_obj = MagicMock()
        mock_transcriber.return_value = mock_transcriber_obj

        # Mock transcription result
        mock_transcriber_obj.transcribe_audio.return_value = {
            "full_text": "This is a test transcription",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "This is a test"},
                {"start": 5.0, "end": 10.0, "text": "transcription"},
            ],
            "language": "en",
            "duration": 10.0,
        }

        # Stub the locator: agent.transcribe_audio resolves audio_url via
        # MediaLocator before handing off to the transcriber. Returning a
        # known on-disk path keeps the unit test boundary at the agent's
        # own logic without performing real HTTP fetches.
        clip = tmp_path / "audio.mp3"
        clip.write_bytes(b"")
        self.agent._locator = MagicMock()
        self.agent._locator.localize.return_value = clip
        self.agent._locator.to_canonical_uri.return_value = f"file://{clip}"

        result = await self.agent.transcribe_audio("http://example.com/audio.mp3")

        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a test transcription"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_via_vllm_sidecar(self, tmp_path, monkeypatch):
        """Sidecar path POSTs multipart to vLLM /v1/audio/transcriptions.

        Pins the agent against the vLLM Whisper contract — the chart-deployed
        endpoint. The legacy deploy/whisper path served /v1/transcribe with
        a JSON ``audio_b64`` body and was removed when ASR migrated to
        vLLM; this test would have caught that the agent was still using
        the dead endpoint.
        """
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        response = {
            "text": "hello world",
            "language": "en",
            "duration": "1.23",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 1.23, "text": "world"},
            ],
        }

        token = "modal-whisper-secret"
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", token)
        agent = AudioAnalysisAgent(
            deps=AudioAnalysisDeps(
                tenant_id="test_tenant",
                whisper_endpoint="https://whisper.modal.run",
            )
        )
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "rotated-after-startup")
        monkeypatch.setattr(agent, "_get_audio_path", lambda _: str(clip))

        with _transcription_server(response) as (base_url, captured_requests):
            agent._whisper_endpoint = base_url
            result = await agent.transcribe_audio(f"file://{clip}")

        assert len(captured_requests) == 1
        request = captured_requests[0]
        assert request["path"] == "/v1/audio/transcriptions"
        assert request["authorization"] == f"Bearer {token}"
        assert str(request["content_type"]).startswith("multipart/form-data; boundary=")
        request_body = request["body"]
        assert isinstance(request_body, bytes)
        assert b'filename="audio.wav"' in request_body
        assert b"RIFF\x00\x00\x00\x00WAVE" in request_body
        assert b'name="model"\r\n\r\nopenai/whisper-large-v3-turbo' in request_body
        assert b'name="response_format"\r\n\r\nverbose_json' in request_body
        with pytest.raises(TypeError):
            agent._whisper_headers["Authorization"] = "Bearer replacement"
        assert token not in repr(agent.deps)

        assert result.text == "hello world"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_transcription_outage_propagates_without_leaking_key(
        self, tmp_path, monkeypatch
    ):
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        token = "remote-whisper-secret"
        unavailable = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        host, port = unavailable.server_address
        unavailable.server_close()
        agent = _agent_for_remote_transcription(
            f"http://{host}:{port}", f"Bearer {token}"
        )
        monkeypatch.setattr(agent, "_get_audio_path", lambda _: str(clip))

        with pytest.raises(requests.ConnectionError) as caught:
            await agent.transcribe_audio(f"file://{clip}")

        assert token not in str(caught.value)

    @pytest.mark.asyncio
    async def test_concurrent_modal_transcriptions_are_isolated_and_non_blocking(
        self, monkeypatch, tmp_path
    ):
        request_count = 8
        token = "remote-whisper-secret"
        clips = []
        for index in range(request_count):
            clip = tmp_path / f"clip-{index}.wav"
            clip.write_bytes(f"RIFF-audio-{index}".encode())
            clips.append(clip)

        all_requests_started = threading.Barrier(request_count)
        release = threading.Event()

        def response_for(request_body: bytes):
            match = re.search(rb'filename="clip-(\d+)\.wav"', request_body)
            assert match is not None
            index = int(match.group(1))
            return {
                "text": f"transcript-{index}",
                "language": "en",
                "duration": str(index + 0.5),
                "segments": [
                    {
                        "start": 0.0,
                        "end": index + 0.5,
                        "text": f"transcript-{index}",
                    }
                ],
            }

        async def prove_event_loop_progress():
            await asyncio.sleep(0.05)
            release.set()
            return "event-loop-progressed"

        with _transcription_server(
            response_for,
            arrival_barrier=all_requests_started,
            release_response=release,
        ) as (base_url, captured_requests):
            agent = _agent_for_remote_transcription(base_url, f"Bearer {token}")
            monkeypatch.setattr(agent, "_get_audio_path", lambda audio_url: audio_url)
            gathered = await asyncio.wait_for(
                asyncio.gather(
                    *(agent.transcribe_audio(str(clip)) for clip in clips),
                    prove_event_loop_progress(),
                ),
                timeout=6,
            )

        results = gathered[:-1]
        assert gathered[-1] == "event-loop-progressed"
        assert [result.text for result in results] == [
            f"transcript-{index}" for index in range(request_count)
        ]
        assert [result.language for result in results] == ["en"] * request_count
        assert [result.confidence for result in results] == [1.0] * request_count
        assert [result.segments for result in results] == [
            [
                {
                    "start": 0.0,
                    "end": index + 0.5,
                    "text": f"transcript-{index}",
                }
            ]
            for index in range(request_count)
        ]
        assert len(captured_requests) == request_count
        assert {request["path"] for request in captured_requests} == {
            "/v1/audio/transcriptions"
        }
        assert {request["authorization"] for request in captured_requests} == {
            f"Bearer {token}"
        }
        captured_bodies = [request["body"] for request in captured_requests]
        for index in range(request_count):
            assert any(
                f'filename="clip-{index}.wav"'.encode() in body
                and f"RIFF-audio-{index}".encode() in body
                for body in captured_bodies
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("response", "message"),
        [
            ([], "$: expected an object"),
            (
                {"language": "en", "duration": "1.0", "segments": []},
                "$.text: field is required",
            ),
            (
                {
                    "text": 7,
                    "language": "en",
                    "duration": "1.0",
                    "segments": [],
                },
                "$.text: expected a string",
            ),
            (
                {"text": "ok", "duration": "1.0", "segments": []},
                "$.language: field is required",
            ),
            (
                {
                    "text": "ok",
                    "language": 7,
                    "duration": "1.0",
                    "segments": [],
                },
                "$.language: expected a non-empty string",
            ),
            (
                {
                    "text": "ok",
                    "language": "   ",
                    "duration": "1.0",
                    "segments": [],
                },
                "$.language: expected a non-empty string",
            ),
            (
                {"text": "ok", "language": "en", "segments": []},
                "$.duration: field is required",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": 1.0,
                    "segments": [],
                },
                "$.duration: expected a finite non-negative decimal string",
            ),
            (
                {"text": "ok", "language": "en", "duration": False, "segments": []},
                "$.duration: expected a finite non-negative decimal string",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "-0.1",
                    "segments": [],
                },
                "$.duration: expected a finite non-negative decimal string",
            ),
            (
                {"text": "ok", "language": "en", "duration": "1.0"},
                "$.segments: field is required",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": {},
                },
                "$.segments: expected a list",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [7],
                },
                "$.segments[0]: expected an object",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"end": 1.0, "text": "ok"}],
                },
                "$.segments[0].start: field is required",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": "0.0", "end": 1.0, "text": "ok"}],
                },
                "$.segments[0].start: expected a finite non-negative number",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": -0.1, "end": 1.0, "text": "ok"}],
                },
                "$.segments[0].start: expected a finite non-negative number",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": 0.0, "text": "ok"}],
                },
                "$.segments[0].end: field is required",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": 0.0, "end": False, "text": "ok"}],
                },
                "$.segments[0].end: expected a finite non-negative number",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": 0.75, "end": 0.5, "text": "ok"}],
                },
                "$.segments[0].end: must be greater than or equal to start",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": 0.0, "end": 1.1, "text": "ok"}],
                },
                "$.segments[0].end: must not exceed $.duration",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": 0.0, "end": 1.0}],
                },
                "$.segments[0].text: field is required",
            ),
            (
                {
                    "text": "ok",
                    "language": "en",
                    "duration": "1.0",
                    "segments": [{"start": 0.0, "end": 1.0, "text": 7}],
                },
                "$.segments[0].text: expected a string",
            ),
        ],
    )
    async def test_remote_transcription_rejects_malformed_success_response(
        self, response, message, tmp_path, monkeypatch
    ):
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with _transcription_server(response) as (base_url, captured_requests):
            agent = _agent_for_remote_transcription(
                base_url, "Bearer remote-whisper-secret"
            )
            monkeypatch.setattr(agent, "_get_audio_path", lambda _: str(clip))

            with pytest.raises(ValueError, match=re.escape(message)) as caught:
                await agent.transcribe_audio(f"file://{clip}")

        assert len(captured_requests) == 1
        assert f"{base_url}/v1/audio/transcriptions" in str(caught.value)

    @pytest.mark.asyncio
    async def test_remote_transcription_rejects_non_json_success_response(
        self, tmp_path, monkeypatch
    ):
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with _transcription_server({}, raw_body=b"not-json") as (base_url, _):
            agent = _agent_for_remote_transcription(
                base_url, "Bearer remote-whisper-secret"
            )
            monkeypatch.setattr(agent, "_get_audio_path", lambda _: str(clip))

            with pytest.raises(
                ValueError, match="response body is not valid JSON"
            ) as caught:
                await agent.transcribe_audio(f"file://{clip}")

        assert f"{base_url}/v1/audio/transcriptions" in str(caught.value)

    @pytest.mark.asyncio
    async def test_transcribe_audio_sidecar_empty_segments_surfaced(
        self, tmp_path, monkeypatch
    ):
        """Empty segments from vLLM are returned as-is, not synthesised.

        vLLM may return ``segments=[]`` for very short audio. Synthesising a
        single segment to "always have something" hides the producer's
        actual output from downstream consumers; the test pins the surface-
        the-truth behaviour.
        """
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        response = {
            "text": "ok",
            "language": "en",
            "duration": "0.5",
            "segments": [],
        }

        with _transcription_server(response) as (base_url, _):
            agent = _agent_for_remote_transcription(
                base_url, "Bearer remote-whisper-secret"
            )
            monkeypatch.setattr(agent, "_get_audio_path", lambda _: str(clip))
            result = await agent.transcribe_audio(f"file://{clip}")

        assert result.text == "ok"
        assert result.segments == []

    @pytest.mark.asyncio
    @patch.object(AudioAnalysisAgent, "audio_transcriber", new_callable=PropertyMock)
    async def test_find_similar_audio_semantic(self, mock_transcriber, tmp_path):
        """Test finding similar audio using semantic similarity"""
        # Mock transcriber
        mock_transcriber_obj = MagicMock()
        mock_transcriber.return_value = mock_transcriber_obj
        mock_transcriber_obj.transcribe_audio.return_value = {
            "full_text": "Test transcription",
            "segments": [],
            "language": "en",
        }
        self.agent._search_transcript = AsyncMock(return_value=["stub"])

        clip = tmp_path / "ref.mp3"
        clip.write_bytes(b"")
        self.agent._locator = MagicMock()
        self.agent._locator.localize.return_value = clip
        self.agent._locator.to_canonical_uri.return_value = f"file://{clip}"

        results = await self.agent.find_similar_audio(
            reference_audio_url="http://example.com/ref.mp3",
            similarity_type="semantic",
            limit=20,
        )

        # Verify transcription was called
        mock_transcriber_obj.transcribe_audio.assert_called_once()
        self.agent._search_transcript.assert_awaited_once_with("Test transcription", 20)
        assert results == ["stub"]

    def test_get_audio_path_local(self, tmp_path):
        """Test getting local audio path"""
        clip = tmp_path / "audio.mp3"
        clip.write_bytes(b"")

        result = self.agent._get_audio_path(str(clip))

        # MediaLocator's file:// short-circuit returns the canonical path
        # unchanged — no copy, no cache entry.
        assert result == str(clip)

    def test_get_audio_path_url(self, tmp_path):
        """Test downloading audio from URL via MediaLocator"""
        # Stub the locator boundary: assert that bare-URL inputs are
        # canonicalized then handed to the locator's localize. The locator
        # itself is exercised end-to-end in the integration tests.
        clip = tmp_path / "audio.mp3"
        clip.write_bytes(b"fake audio data")

        self.agent._locator = MagicMock()
        self.agent._locator.localize.return_value = clip
        self.agent._locator.to_canonical_uri.return_value = (
            "http://example.com/audio.mp3"
        )

        result = self.agent._get_audio_path("http://example.com/audio.mp3")

        assert result == str(clip)
        self.agent._locator.to_canonical_uri.assert_called_once_with(
            "http://example.com/audio.mp3"
        )
        self.agent._locator.localize.assert_called_once_with(
            "http://example.com/audio.mp3"
        )

    @pytest.mark.asyncio
    async def test_dspy_to_a2a_output(self):
        """Test DSPy output to A2A format conversion"""
        # Create sample results
        results = [
            AudioResult(
                audio_id="audio_001",
                audio_url="http://example.com/audio1.mp3",
                title="Test Audio",
                transcript="This is a test",
                duration=120.0,
                relevance_score=0.85,
                speaker_labels=["Speaker A"],
                detected_events=["speech"],
                language="en",
            )
        ]

        # Convert to A2A output
        a2a_output = self.agent._dspy_to_a2a_output({"results": results, "count": 1})

        # Verify format
        assert a2a_output["status"] == "success"
        assert a2a_output["result_type"] == "audio_search_results"
        assert a2a_output["count"] == 1
        assert len(a2a_output["results"]) == 1
        assert a2a_output["results"][0]["audio_id"] == "audio_001"
        assert a2a_output["results"][0]["language"] == "en"

    def test_get_agent_skills(self):
        """Test agent skills definition"""
        skills = self.agent._get_agent_skills()

        # Verify skills — exactly the three real ones after removing the
        # dead stub skills (event detection / diarization / music classify).
        assert len(skills) == 3
        skill_names = [s["name"] for s in skills]
        assert "search_audio" in skill_names
        assert "transcribe_audio" in skill_names
        assert "find_similar_audio" in skill_names
        # Removed dead stub skills must no longer be advertised.
        assert "detect_audio_events" not in skill_names
        assert "identify_speakers" not in skill_names
        assert "classify_music" not in skill_names


class TestAudioSearchEventLoop:
    """Audio search/transcription must not block the event loop on HTTP."""

    def _bare_agent(self, **attrs):
        agent = object.__new__(AudioAnalysisAgent)
        agent._tenant_id = "acme:acme"
        agent._vespa_endpoint = "http://vespa:8080"
        agent._deployed_audio_schema = None
        agent._shared_backend = None
        agent._shared_backend_lock = threading.Lock()
        agent._backend_type = "vespa"
        agent._backend_config = {}
        agent.config_manager = None
        agent.schema_loader = None
        for k, v in attrs.items():
            setattr(agent, k, v)
        return agent

    @pytest.mark.asyncio
    async def test_search_transcript_offloads_blocking_backend_search(
        self, monkeypatch
    ):
        """Transcript search offloads the synchronous backend call off loop."""
        release = threading.Event()

        def blocking_search(query_dict):
            assert query_dict == {
                "query": "q",
                "type": "audio",
                "strategy": "transcript_search",
                "tenant_id": "acme:acme",
                "top_k": 5,
            }
            assert release.wait(timeout=5), "event loop was blocked by backend.search"
            return []

        agent = self._bare_agent()
        agent._get_backend = MagicMock(
            return_value=SimpleNamespace(search=blocking_search)
        )

        async def releaser():
            await asyncio.sleep(0.05)
            release.set()

        results, _ = await asyncio.wait_for(
            asyncio.gather(agent._search_transcript("q", 5), releaser()),
            timeout=5,
        )
        assert results == []
        agent._get_backend.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_via_sidecar_offloaded(self, monkeypatch, tmp_path):
        """transcribe_audio offloads the synchronous _transcribe_via_sidecar
        helper (blocking file read + POST) off the event loop."""
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"RIFFfake-wav-bytes")
        release = threading.Event()

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "text": "hello world",
                    "language": "en",
                    "segments": [],
                    "duration": "1.0",
                }

        def blocking_post(url, **kwargs):
            assert release.wait(timeout=5), "event loop was blocked by requests.post"
            return _Resp()

        monkeypatch.setattr("requests.post", blocking_post)
        agent = self._bare_agent(
            _whisper_endpoint="http://asr:8000",
            _whisper_headers={},
            _whisper_model="whisper-1",
        )
        monkeypatch.setattr(agent, "_get_audio_path", lambda url: str(audio))

        async def releaser():
            await asyncio.sleep(0.05)
            release.set()

        result, _ = await asyncio.wait_for(
            asyncio.gather(agent.transcribe_audio(str(audio)), releaser()),
            timeout=5,
        )
        assert result.text == "hello world"
        assert result.language == "en"


class _Vec:
    def tolist(self):
        return [0.0, 0.0]


class _EmptyVespaResp:
    status_code = 200

    def json(self):
        return {"root": {"children": []}}


def _bare_acoustic_agent(blocking_embed):
    agent = object.__new__(AudioAnalysisAgent)
    agent._tenant_id = "test:test"
    agent._vespa_endpoint = "http://fake-vespa:8080"
    agent._deployed_audio_schema = None
    agent._embedding_generator = SimpleNamespace(
        generate_acoustic_text_embedding=blocking_embed
    )
    return agent


@pytest.mark.asyncio
async def test_search_hybrid_offloads_blocking_backend_search(monkeypatch):
    """Hybrid text search offloads the synchronous backend call off loop."""
    release = threading.Event()
    blocking_embed = MagicMock(
        side_effect=AssertionError("acoustic embedding should not be used")
    )

    def blocking_search(query_dict):
        assert query_dict == {
            "query": "q",
            "type": "audio",
            "strategy": "hybrid_semantic_bm25",
            "tenant_id": "test:test",
            "top_k": 5,
        }
        assert release.wait(timeout=5), "event loop was blocked by backend.search"
        return []

    agent = _bare_acoustic_agent(blocking_embed)
    agent._get_backend = MagicMock(return_value=SimpleNamespace(search=blocking_search))

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    results, _ = await asyncio.wait_for(
        asyncio.gather(agent._search_hybrid("q", 5), releaser()), timeout=5
    )
    assert results == []
    agent._get_backend.assert_called_once()
    agent._embedding_generator.generate_acoustic_text_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_search_acoustic_offloads_blocking_clap_encode(monkeypatch):
    """Same contract for the pure-acoustic search mode."""
    release = threading.Event()

    def blocking_embed(query):
        assert release.wait(timeout=5), "event loop was blocked by CLAP encode"
        return _Vec()

    agent = _bare_acoustic_agent(blocking_embed)
    schema_backend = MagicMock()
    schema_backend.schema_exists = MagicMock(return_value=True)
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.get_backend",
        lambda: schema_backend,
    )
    monkeypatch.setattr(
        "cogniverse_agents.search.vespa_query.vespa_search_post",
        lambda endpoint, params, timeout: _EmptyVespaResp(),
    )

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    results, _ = await asyncio.wait_for(
        asyncio.gather(agent._search_acoustic("q", 5), releaser()), timeout=5
    )
    assert results == []
    assert schema_backend.schema_exists.call_args_list == [
        call("audio_content", tenant_id="test:test"),
        call("audio_content", tenant_id="test:test"),
    ]


@pytest.mark.asyncio
async def test_local_whisper_transcription_offloads_blocking_model():
    """The in-process Whisper fallback is a heavy blocking model forward —
    same offload contract as the sidecar branch above it."""
    release = threading.Event()

    class _BlockingTranscriber:
        def transcribe_audio(self, video_path, output_dir=None):
            assert release.wait(timeout=5), "event loop was blocked by whisper"
            return {
                "full_text": "hi",
                "segments": [],
                "language": "en",
                "duration": 1.0,
            }

    agent = object.__new__(AudioAnalysisAgent)
    agent._whisper_endpoint = None
    agent._deployed_audio_schema = None
    agent._audio_transcriber = _BlockingTranscriber()
    agent._get_audio_path = lambda url: "/tmp/x.wav"

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    result, _ = await asyncio.wait_for(
        asyncio.gather(agent.transcribe_audio("file:///x.wav"), releaser()),
        timeout=5,
    )
    assert result.text == "hi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
