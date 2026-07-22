"""
Unit tests for Audio Analysis Agent

Tests audio transcription with Whisper, audio search, and Vespa integration.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from cogniverse_agents.audio_analysis_agent import (
    AudioAnalysisAgent,
    AudioAnalysisDeps,
    AudioResult,
    TranscriptionResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class TestAudioAnalysisAgent:
    """Unit tests for AudioAnalysisAgent"""

    def setup_method(self):
        """Set up test fixtures"""
        self.agent = AudioAnalysisAgent(
            deps=AudioAnalysisDeps(
                tenant_id="test_tenant",
                vespa_endpoint="http://localhost:8080",
                whisper_model_size="base",
            ),
            port=8006,
        )

    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent is not None
        assert self.agent._audio_transcriber is None  # Lazy loaded
        assert self.agent._whisper_model_size == "base"
        assert self.agent._vespa_endpoint == "http://localhost:8080"

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

    @pytest.mark.asyncio
    @patch("requests.post")
    async def test_search_audio_transcript_mode(self, mock_post):
        """Test transcript-based audio search"""
        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "root": {
                "children": [
                    {
                        "relevance": 0.90,
                        # Field names are the deployed audio_content schema's --
                        # audio_transcript / audio_duration / audio_language, not
                        # bare transcript/duration/language (which never populate
                        # and previously shipped every hit empty).
                        "fields": {
                            "audio_id": "audio_001",
                            "source_url": "http://example.com/audio1.mp3",
                            "audio_title": "Test Podcast",
                            "audio_transcript": "This is a test podcast about AI",
                            "audio_duration": 300.0,
                            "audio_language": "en",
                        },
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        # Execute search
        results = await self.agent.search_audio(
            query="AI podcast", search_mode="transcript", limit=20
        )

        # Verify results -- the transcript/duration/language must round-trip from
        # the schema field names, not come back empty.
        assert len(results) == 1
        assert results[0].audio_id == "audio_001"
        assert results[0].title == "Test Podcast"
        assert results[0].transcript == "This is a test podcast about AI"
        assert results[0].duration == 300.0
        assert results[0].language == "en"

        # Verify Vespa was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "transcript_search" in str(call_args)

    @pytest.mark.asyncio
    @patch("requests.post")
    async def test_search_audio_hybrid_mode(self, mock_post):
        """Test hybrid audio search"""
        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        # Execute search
        await self.agent.search_audio(
            query="machine learning", search_mode="hybrid", limit=10
        )

        # Verify Vespa was called
        mock_post.assert_called_once()

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
    @patch("requests.post")
    async def test_transcribe_audio_via_vllm_sidecar(self, mock_post, tmp_path):
        """Sidecar path POSTs multipart to vLLM /v1/audio/transcriptions.

        Pins the agent against the vLLM Whisper contract — the chart-deployed
        endpoint. The legacy deploy/whisper path served /v1/transcribe with
        a JSON ``audio_b64`` body and was removed when ASR migrated to
        vLLM; this test would have caught that the agent was still using
        the dead endpoint.
        """
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "hello world",
            "language": "en",
            "duration": 1.23,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 1.23, "text": "world"},
            ],
        }
        mock_post.return_value = mock_response

        self.agent._whisper_endpoint = "http://vllm-asr:8000"
        self.agent._whisper_model = "openai/whisper-large-v3-turbo"
        self.agent._locator = MagicMock()
        self.agent._locator.localize.return_value = clip
        self.agent._locator.to_canonical_uri.return_value = f"file://{clip}"

        result = await self.agent.transcribe_audio(f"file://{clip}")

        assert mock_post.call_count == 1
        call = mock_post.call_args
        assert call.args[0] == "http://vllm-asr:8000/v1/audio/transcriptions"
        assert "files" in call.kwargs and "file" in call.kwargs["files"]
        assert call.kwargs["data"]["model"] == "openai/whisper-large-v3-turbo"

        assert result.text == "hello world"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["text"] == "hello"

    @pytest.mark.asyncio
    @patch("requests.post")
    async def test_transcribe_audio_sidecar_empty_segments_surfaced(
        self, mock_post, tmp_path
    ):
        """Empty segments from vLLM are returned as-is, not synthesised.

        vLLM may return ``segments=[]`` for very short audio. Synthesising a
        single segment to "always have something" hides the producer's
        actual output from downstream consumers; the test pins the surface-
        the-truth behaviour.
        """
        clip = tmp_path / "audio.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "ok",
            "language": "en",
            "duration": 0.5,
            "segments": [],
        }
        mock_post.return_value = mock_response

        self.agent._whisper_endpoint = "http://vllm-asr:8000"
        self.agent._whisper_model = "openai/whisper-large-v3-turbo"
        self.agent._locator = MagicMock()
        self.agent._locator.localize.return_value = clip
        self.agent._locator.to_canonical_uri.return_value = f"file://{clip}"

        result = await self.agent.transcribe_audio(f"file://{clip}")
        assert result.text == "ok"
        assert result.segments == []

    @pytest.mark.asyncio
    @patch.object(AudioAnalysisAgent, "audio_transcriber", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_find_similar_audio_semantic(
        self, mock_post, mock_transcriber, tmp_path
    ):
        """Test finding similar audio using semantic similarity"""
        # Mock transcriber
        mock_transcriber_obj = MagicMock()
        mock_transcriber.return_value = mock_transcriber_obj
        mock_transcriber_obj.transcribe_audio.return_value = {
            "full_text": "Test transcription",
            "segments": [],
            "language": "en",
        }

        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        clip = tmp_path / "ref.mp3"
        clip.write_bytes(b"")
        self.agent._locator = MagicMock()
        self.agent._locator.localize.return_value = clip
        self.agent._locator.to_canonical_uri.return_value = f"file://{clip}"

        await self.agent.find_similar_audio(
            reference_audio_url="http://example.com/ref.mp3",
            similarity_type="semantic",
            limit=20,
        )

        # Verify transcription was called
        mock_transcriber_obj.transcribe_audio.assert_called_once()
        # Verify Vespa search was called
        mock_post.assert_called_once()

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
        for k, v in attrs.items():
            setattr(agent, k, v)
        return agent

    @pytest.mark.asyncio
    async def test_search_transcript_offloads_blocking_post(self, monkeypatch):
        """Representative of the three async search methods, which share the
        same offload wrapper. If the post ran on the loop the releaser could
        never run and the gather would deadlock."""
        release = threading.Event()

        class _Resp:
            status_code = 200

            def json(self):
                return {"root": {"children": []}}

        def blocking_post(url, json=None, timeout=None):
            assert release.wait(timeout=5), "event loop was blocked by requests.post"
            return _Resp()

        monkeypatch.setattr("requests.post", blocking_post)
        agent = self._bare_agent()

        async def releaser():
            await asyncio.sleep(0.05)
            release.set()

        results, _ = await asyncio.wait_for(
            asyncio.gather(agent._search_transcript("q", 5), releaser()),
            timeout=5,
        )
        assert results == []

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
                    "duration": 1.0,
                }

        def blocking_post(url, **kwargs):
            assert release.wait(timeout=5), "event loop was blocked by requests.post"
            return _Resp()

        monkeypatch.setattr("requests.post", blocking_post)
        agent = self._bare_agent(
            _whisper_endpoint="http://asr:8000", _whisper_model="whisper-1"
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
    agent._embedding_generator = SimpleNamespace(
        generate_acoustic_text_embedding=blocking_embed
    )
    return agent


@pytest.mark.asyncio
async def test_search_hybrid_offloads_blocking_clap_encode(monkeypatch):
    """The CLAP text encode is a blocking HTTP call and must run in a worker
    thread like the Vespa post below it — inline on the loop it stalls every
    other coroutine for the whole encode round-trip."""
    release = threading.Event()

    def blocking_embed(query):
        assert release.wait(timeout=5), "event loop was blocked by CLAP encode"
        return _Vec()

    agent = _bare_acoustic_agent(blocking_embed)
    monkeypatch.setattr(
        "cogniverse_agents.search.vespa_query.vespa_search_post",
        lambda endpoint, params, timeout: _EmptyVespaResp(),
    )

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    results, _ = await asyncio.wait_for(
        asyncio.gather(agent._search_hybrid("q", 5), releaser()), timeout=5
    )
    assert results == []


@pytest.mark.asyncio
async def test_search_acoustic_offloads_blocking_clap_encode(monkeypatch):
    """Same contract for the pure-acoustic search mode."""
    release = threading.Event()

    def blocking_embed(query):
        assert release.wait(timeout=5), "event loop was blocked by CLAP encode"
        return _Vec()

    agent = _bare_acoustic_agent(blocking_embed)
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
