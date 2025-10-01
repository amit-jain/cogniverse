"""
Unit tests for Audio Analysis Agent

Tests audio transcription with Whisper, audio search, and Vespa integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path

from src.app.agents.audio_analysis_agent import (
    AudioAnalysisAgent,
    AudioResult,
    TranscriptionResult,
    AudioEvent,
    SpeakerSegment,
    MusicClassification,
)


class TestAudioAnalysisAgent:
    """Unit tests for AudioAnalysisAgent"""

    def setup_method(self):
        """Set up test fixtures"""
        self.agent = AudioAnalysisAgent(
            vespa_endpoint="http://localhost:8080",
            whisper_model_size="base",
            port=8006
        )

    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent is not None
        assert self.agent._audio_transcriber is None  # Lazy loaded
        assert self.agent._whisper_model_size == "base"
        assert self.agent._vespa_endpoint == "http://localhost:8080"

    @patch('src.app.agents.audio_analysis_agent.AudioTranscriber')
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
    @patch('requests.post')
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
                        "fields": {
                            "audio_id": "audio_001",
                            "source_url": "http://example.com/audio1.mp3",
                            "audio_title": "Test Podcast",
                            "transcript": "This is a test podcast about AI",
                            "duration": 300.0,
                            "speaker_labels": ["Speaker A"],
                            "detected_events": ["speech"],
                            "language": "en"
                        }
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        # Execute search
        results = await self.agent.search_audio(
            query="AI podcast",
            search_mode="transcript",
            limit=20
        )

        # Verify results
        assert len(results) == 1
        assert results[0].audio_id == "audio_001"
        assert results[0].title == "Test Podcast"
        assert results[0].duration == 300.0
        assert results[0].language == "en"

        # Verify Vespa was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "transcript_search" in str(call_args)

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_search_audio_hybrid_mode(self, mock_post):
        """Test hybrid audio search"""
        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        # Execute search
        results = await self.agent.search_audio(
            query="machine learning",
            search_mode="hybrid",
            limit=10
        )

        # Verify Vespa was called
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(AudioAnalysisAgent, 'audio_transcriber', new_callable=PropertyMock)
    async def test_transcribe_audio(self, mock_transcriber):
        """Test audio transcription using Whisper"""
        # Mock AudioTranscriber
        mock_transcriber_obj = MagicMock()
        mock_transcriber.return_value = mock_transcriber_obj

        # Mock transcription result
        mock_transcriber_obj.transcribe_audio.return_value = {
            "full_text": "This is a test transcription",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "This is a test"},
                {"start": 5.0, "end": 10.0, "text": "transcription"}
            ],
            "language": "en",
            "duration": 10.0
        }

        # Execute transcription
        result = await self.agent.transcribe_audio("http://example.com/audio.mp3")

        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a test transcription"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_detect_audio_events(self):
        """Test audio event detection (placeholder)"""
        # This is a placeholder method
        results = await self.agent.detect_audio_events("http://example.com/audio.mp3")

        # Should return empty list as it's not fully implemented
        assert results == []

    @pytest.mark.asyncio
    async def test_identify_speakers(self):
        """Test speaker diarization (placeholder)"""
        # This is a placeholder method
        results = await self.agent.identify_speakers("http://example.com/audio.mp3")

        # Should return empty list as it's not fully implemented
        assert results == []

    @pytest.mark.asyncio
    async def test_classify_music(self):
        """Test music classification (placeholder)"""
        # This is a placeholder method
        result = await self.agent.classify_music("http://example.com/music.mp3")

        # Should return default MusicClassification
        assert isinstance(result, MusicClassification)
        assert result.genre == "unknown"
        assert result.mood == "unknown"
        assert result.tempo == 0.0

    @pytest.mark.asyncio
    @patch.object(AudioAnalysisAgent, 'audio_transcriber', new_callable=PropertyMock)
    @patch('requests.post')
    async def test_find_similar_audio_semantic(self, mock_post, mock_transcriber):
        """Test finding similar audio using semantic similarity"""
        # Mock transcriber
        mock_transcriber_obj = MagicMock()
        mock_transcriber.return_value = mock_transcriber_obj
        mock_transcriber_obj.transcribe_audio.return_value = {
            "full_text": "Test transcription",
            "segments": [],
            "language": "en"
        }

        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        # Execute similar audio search
        results = await self.agent.find_similar_audio(
            reference_audio_url="http://example.com/ref.mp3",
            similarity_type="semantic",
            limit=20
        )

        # Verify transcription was called
        mock_transcriber_obj.transcribe_audio.assert_called_once()
        # Verify Vespa search was called
        mock_post.assert_called_once()

    def test_get_audio_path_local(self):
        """Test getting local audio path"""
        local_path = "/path/to/audio.mp3"
        result = self.agent._get_audio_path(local_path)

        # Should return the same path for local files
        assert result == local_path

    @patch('requests.get')
    def test_get_audio_path_url(self, mock_get):
        """Test downloading audio from URL"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.content = b"fake audio data"
        mock_get.return_value = mock_response

        # Execute download
        result = self.agent._get_audio_path("http://example.com/audio.mp3")

        # Verify file was downloaded to temp location
        assert result.endswith(".mp3")
        mock_get.assert_called_once_with("http://example.com/audio.mp3")

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
                language="en"
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

        # Verify skills
        assert len(skills) >= 4
        skill_names = [s["name"] for s in skills]
        assert "search_audio" in skill_names
        assert "transcribe_audio" in skill_names
        assert "detect_audio_events" in skill_names
        assert "identify_speakers" in skill_names
        assert "classify_music" in skill_names
        assert "find_similar_audio" in skill_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
