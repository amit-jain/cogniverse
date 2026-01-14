"""
Audio Analysis Agent using Whisper

Uses existing AudioTranscriber for transcription and connects to Vespa
for real audio search. Supports both transcript-based and acoustic similarity search.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from cogniverse_runtime.ingestion.processors.audio_transcriber import AudioTranscriber
from pydantic import Field as PydanticField

logger = logging.getLogger(__name__)


# =============================================================================
# Type-Safe Models
# =============================================================================


class AudioResult(AgentOutput):
    """Result from audio search"""

    audio_id: str = PydanticField(..., description="Audio identifier")
    audio_url: str = PydanticField(..., description="Audio URL")
    title: str = PydanticField("", description="Audio title")
    transcript: str = PydanticField("", description="Audio transcript")
    duration: float = PydanticField(0.0, description="Duration in seconds")
    relevance_score: float = PydanticField(0.0, description="Relevance score")
    speaker_labels: List[str] = PydanticField(default_factory=list, description="Speaker labels")
    detected_events: List[str] = PydanticField(default_factory=list, description="Detected events")
    language: str = PydanticField("unknown", description="Detected language")
    metadata: Dict[str, Any] = PydanticField(default_factory=dict, description="Additional metadata")


class AudioSearchInput(AgentInput):
    """Type-safe input for audio search"""

    query: str = PydanticField(..., description="Search query")
    search_mode: str = PydanticField("hybrid", description="Search mode: transcript, acoustic, hybrid")
    limit: int = PydanticField(20, description="Number of results")


class AudioSearchOutput(AgentOutput):
    """Type-safe output from audio search"""

    results: List[AudioResult] = PydanticField(default_factory=list, description="Search results")
    count: int = PydanticField(0, description="Number of results")


class AudioAnalysisDeps(AgentDeps):
    """Dependencies for audio analysis agent"""

    vespa_endpoint: str = PydanticField("http://localhost:8080", description="Vespa endpoint")
    whisper_model_size: str = PydanticField("base", description="Whisper model size")


@dataclass
class TranscriptionResult:
    """Result from audio transcription"""

    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float


@dataclass
class AudioEvent:
    """Detected audio event"""

    event_type: str  # speech, music, applause, etc.
    start_time: float
    end_time: float
    confidence: float


@dataclass
class SpeakerSegment:
    """Speaker diarization segment"""

    speaker_id: str
    start_time: float
    end_time: float
    text: Optional[str] = None


@dataclass
class MusicClassification:
    """Music classification result"""

    genre: str
    mood: str
    tempo: float  # BPM
    key: Optional[str] = None
    instruments: List[str] = field(default_factory=list)


class AudioAnalysisAgent(A2AAgent[AudioSearchInput, AudioSearchOutput, AudioAnalysisDeps]):
    """
    Type-safe audio content analysis using Whisper and Vespa.

    Capabilities:
    - Speech transcription using existing AudioTranscriber
    - Transcript-based semantic search
    - Acoustic similarity search
    - Hybrid search combining both approaches
    - Real Vespa backend integration
    """

    def __init__(self, deps: AudioAnalysisDeps, port: int = 8006):
        """
        Initialize Audio Analysis Agent with typed dependencies.

        Args:
            deps: Typed dependencies with tenant_id, vespa_endpoint, whisper_model_size
            port: A2A server port

        Raises:
            TypeError: If deps is not AudioAnalysisDeps
            ValidationError: If deps fails Pydantic validation
        """
        # Create DSPy module
        class AudioSearchSignature(dspy.Signature):
            query: str = dspy.InputField(desc="Audio search query")
            mode: str = dspy.InputField(
                desc="Search mode: transcript, acoustic, hybrid"
            )
            result: str = dspy.OutputField(desc="Search results")

        class AudioSearchModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, query: str, mode: str = "hybrid"):
                return dspy.Prediction(
                    result=f"Searching audio: {query} (mode: {mode})"
                )

        # Create A2A config
        config = A2AAgentConfig(
            agent_name="AudioAnalysisAgent",
            agent_description="Type-safe audio analysis using Whisper and acoustic models",
            capabilities=["audio_search", "transcription", "hybrid_search"],
            port=port,
            version="1.0.0",
        )

        # Initialize A2A base
        super().__init__(deps=deps, config=config, dspy_module=AudioSearchModule())

        self._vespa_endpoint = deps.vespa_endpoint
        self._whisper_model_size = deps.whisper_model_size

        # Initialize components (lazy loading)
        self._audio_transcriber = None
        self._embedding_generator = None

        logger.info(
            f"Initialized AudioAnalysisAgent for tenant: {deps.tenant_id}, "
            f"whisper: {deps.whisper_model_size}"
        )

    @property
    def audio_transcriber(self):
        """Lazy load AudioTranscriber"""
        if self._audio_transcriber is None:
            logger.info(f"Loading Whisper model: {self._whisper_model_size}")
            self._audio_transcriber = AudioTranscriber(
                model_size=self._whisper_model_size
            )
            logger.info("✅ AudioTranscriber loaded")
        return self._audio_transcriber

    @property
    def embedding_generator(self):
        """Lazy load AudioEmbeddingGenerator"""
        if self._embedding_generator is None:
            logger.info("Loading audio embedding models...")
            self._embedding_generator = AudioEmbeddingGenerator()
            logger.info("✅ AudioEmbeddingGenerator loaded")
        return self._embedding_generator

    async def search_audio(
        self,
        query: str,
        search_mode: str = "hybrid",
        limit: int = 20,
    ) -> List[AudioResult]:
        """
        Search audio content

        Args:
            query: Text query
            search_mode: "acoustic", "transcript", or "hybrid"
            limit: Number of results

        Returns:
            List of AudioResult with relevance scores
        """
        logger.info(f"🔍 Searching audio: query='{query}', mode={search_mode}")

        try:
            if search_mode == "acoustic":
                results = await self._search_acoustic(query, limit)
            elif search_mode == "transcript":
                results = await self._search_transcript(query, limit)
            elif search_mode == "hybrid":
                # Hybrid: BM25 on transcripts + semantic similarity
                results = await self._search_hybrid(query, limit)
            else:
                results = await self._search_transcript(query, limit)

            logger.info(f"✅ Found {len(results)} audio results")
            return results

        except Exception as e:
            logger.error(f"❌ Audio search failed: {e}")
            return []

    async def transcribe_audio(
        self,
        audio_url: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Whisper

        Args:
            audio_url: URL or path to audio file
            language: Language code (e.g., "en", "es"). Auto-detect if None

        Returns:
            TranscriptionResult with text, timestamps, and language
        """
        logger.info(f"🎤 Transcribing audio: {audio_url}")

        try:
            # Download audio if URL
            audio_path = self._get_audio_path(audio_url)

            # Use AudioTranscriber
            result = self.audio_transcriber.transcribe_audio(
                video_path=Path(audio_path), output_dir=None
            )

            transcription = TranscriptionResult(
                text=result.get("full_text", ""),
                segments=result.get("segments", []),
                language=result.get("language", "unknown"),
                confidence=1.0,  # Whisper doesn't provide confidence directly
            )

            logger.info(f"✅ Transcription complete: language={transcription.language}")
            return transcription

        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise

    async def _search_transcript(self, query: str, limit: int) -> List[AudioResult]:
        """Search by transcript text using BM25"""
        import requests

        # Build YQL query for transcript search
        yql = "select * from audio_content where userQuery()"

        params = {
            "yql": yql,
            "query": query,
            "hits": limit,
            "ranking.profile": "transcript_search",
        }

        try:
            response = requests.post(
                f"{self._vespa_endpoint}/search/", json=params, timeout=10
            )

            if response.status_code != 200:
                logger.error(
                    f"Vespa search failed: {response.status_code} - {response.text}"
                )
                return []

            # Parse results
            results = []
            data = response.json()

            for hit in data.get("root", {}).get("children", []):
                fields = hit.get("fields", {})
                results.append(
                    AudioResult(
                        audio_id=fields.get("audio_id", ""),
                        audio_url=fields.get("source_url", ""),
                        title=fields.get("audio_title", ""),
                        transcript=fields.get("transcript", ""),
                        duration=fields.get("duration", 0.0),
                        relevance_score=hit.get("relevance", 0.0),
                        speaker_labels=fields.get("speaker_labels", []),
                        detected_events=fields.get("detected_events", []),
                        language=fields.get("language", "unknown"),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"❌ Transcript search failed: {e}")
            return []

    async def _search_acoustic(self, query: str, limit: int) -> List[AudioResult]:
        """Search by acoustic similarity using CLAP embeddings"""
        import requests

        # Generate acoustic embedding from query text (CLAP supports text-to-audio retrieval)
        # For now, use semantic embedding as proxy - CLAP text encoder would be better
        logger.info("Generating query embedding for acoustic search...")
        query_embedding = self.embedding_generator.generate_semantic_embedding(query)

        # Pad/truncate to 512 dims for acoustic search
        if len(query_embedding) > 512:
            query_embedding = query_embedding[:512]
        else:
            padding = np.zeros(512 - len(query_embedding))
            query_embedding = np.concatenate([query_embedding, padding])

        # Build Vespa query for acoustic similarity
        yql = "select * from audio_content where true"

        params = {
            "yql": yql,
            "hits": limit,
            "ranking.profile": "acoustic_similarity",
            "input.query(q)": query_embedding.tolist(),
        }

        try:
            response = requests.post(
                f"{self._vespa_endpoint}/search/", json=params, timeout=10
            )

            if response.status_code != 200:
                logger.error(
                    f"Vespa search failed: {response.status_code} - {response.text}"
                )
                return []

            # Parse results
            results = []
            data = response.json()

            for hit in data.get("root", {}).get("children", []):
                fields = hit.get("fields", {})
                results.append(
                    AudioResult(
                        audio_id=fields.get("audio_id", ""),
                        audio_url=fields.get("source_url", ""),
                        title=fields.get("audio_title", ""),
                        transcript=fields.get("transcript", ""),
                        duration=fields.get("duration", 0.0),
                        relevance_score=hit.get("relevance", 0.0),
                        speaker_labels=fields.get("speaker_labels", []),
                        detected_events=fields.get("detected_events", []),
                        language=fields.get("language", "unknown"),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"❌ Acoustic search failed: {e}")
            return []

    async def _search_hybrid(self, query: str, limit: int) -> List[AudioResult]:
        """Search by hybrid (BM25 transcript + semantic embeddings)"""
        import requests

        # Generate semantic embedding for query
        logger.info("Generating query embedding for hybrid search...")
        query_embedding = self.embedding_generator.generate_semantic_embedding(query)

        # Build YQL query for hybrid search
        yql = "select * from audio_content where userQuery()"

        params = {
            "yql": yql,
            "query": query,
            "hits": limit,
            "ranking.profile": "hybrid_audio",
            "input.query(q)": query_embedding.tolist(),
        }

        try:
            response = requests.post(
                f"{self._vespa_endpoint}/search/", json=params, timeout=10
            )

            if response.status_code != 200:
                logger.error(
                    f"Vespa search failed: {response.status_code} - {response.text}"
                )
                return []

            # Parse results
            results = []
            data = response.json()

            for hit in data.get("root", {}).get("children", []):
                fields = hit.get("fields", {})
                results.append(
                    AudioResult(
                        audio_id=fields.get("audio_id", ""),
                        audio_url=fields.get("source_url", ""),
                        title=fields.get("audio_title", ""),
                        transcript=fields.get("transcript", ""),
                        duration=fields.get("duration", 0.0),
                        relevance_score=hit.get("relevance", 0.0),
                        speaker_labels=fields.get("speaker_labels", []),
                        detected_events=fields.get("detected_events", []),
                        language=fields.get("language", "unknown"),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []

    async def detect_audio_events(
        self, audio_url: str, event_types: Optional[List[str]] = None
    ) -> List[AudioEvent]:
        """
        Detect sounds, music, speech segments

        Args:
            audio_url: URL or path to audio file
            event_types: Specific event types to detect

        Returns:
            List of detected audio events

        Note: Placeholder implementation - would integrate with audio event detection model
        """
        logger.info(f"🔍 Detecting audio events: {audio_url}")
        logger.warning("⚠️  Audio event detection not yet fully implemented")
        return []

    async def identify_speakers(
        self, audio_url: str, num_speakers: Optional[int] = None
    ) -> List[SpeakerSegment]:
        """
        Speaker diarization - identify who spoke when

        Args:
            audio_url: URL or path to audio file
            num_speakers: Expected number of speakers (auto-detect if None)

        Returns:
            List of speaker segments

        Note: Placeholder implementation - would integrate with pyannote.audio or similar
        """
        logger.info(f"👥 Speaker diarization: {audio_url}")
        logger.warning("⚠️  Speaker diarization not yet fully implemented")
        return []

    async def classify_music(self, audio_url: str) -> MusicClassification:
        """
        Classify music genre, mood, tempo, key

        Args:
            audio_url: URL or path to audio file

        Returns:
            MusicClassification with genre, mood, tempo, etc.

        Note: Placeholder implementation - would integrate with music classification model
        """
        logger.info(f"🎵 Classifying music: {audio_url}")
        logger.warning("⚠️  Music classification not yet fully implemented")
        return MusicClassification(
            genre="unknown", mood="unknown", tempo=0.0, key=None, instruments=[]
        )

    async def find_similar_audio(
        self,
        reference_audio_url: str,
        similarity_type: str = "semantic",
        limit: int = 20,
    ) -> List[AudioResult]:
        """
        Find acoustically or semantically similar audio

        Args:
            reference_audio_url: Reference audio URL or path
            similarity_type: "acoustic" or "semantic"
            limit: Number of results

        Returns:
            List of similar audio results
        """
        logger.info(f"🔍 Finding similar audio to: {reference_audio_url}")

        try:
            if similarity_type == "semantic":
                # Transcribe and search semantically
                transcription = await self.transcribe_audio(reference_audio_url)
                results = await self._search_transcript(transcription.text, limit)
            else:
                # Acoustic similarity search
                logger.warning("⚠️  Acoustic similarity not yet fully implemented")
                results = []

            logger.info(f"✅ Found {len(results)} similar audio files")
            return results

        except Exception as e:
            logger.error(f"❌ Similar audio search failed: {e}")
            return []

    def _get_audio_path(self, audio_url: str) -> str:
        """
        Get audio file path (download if URL, or return path directly)

        Args:
            audio_url: URL or local path

        Returns:
            Local file path
        """
        if audio_url.startswith("http://") or audio_url.startswith("https://"):
            # Download audio
            import os
            import tempfile

            import requests

            response = requests.get(audio_url)
            suffix = os.path.splitext(audio_url)[1] or ".mp3"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)
                return f.name
        else:
            # Local path
            return audio_url

    # ==========================================================================
    # Type-safe process method (required by AgentBase)
    # ==========================================================================

    async def _process_impl(self, input: AudioSearchInput) -> AudioSearchOutput:
        """
        Process audio search request with typed input/output.

        Args:
            input: Typed input with query, search_mode, limit

        Returns:
            AudioSearchOutput with results and count
        """
        results = await self.search_audio(
            query=input.query,
            search_mode=input.search_mode,
            limit=input.limit,
        )

        return AudioSearchOutput(results=results, count=len(results))

    def _dspy_to_a2a_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DSPy result to A2A output format."""
        results = result.get("results", [])
        return {
            "status": "success",
            "agent": self.agent_name,
            "result_type": "audio_search_results",
            "count": result.get("count", len(results)),
            "results": [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "search_audio",
                "description": "Search audio content by transcript or acoustic features",
                "input_schema": {"query": "string", "search_mode": "string", "limit": "integer"},
                "output_schema": {"results": "list", "count": "integer"},
            },
            {
                "name": "transcribe_audio",
                "description": "Transcribe audio to text using Whisper",
                "input_schema": {"audio_url": "string"},
                "output_schema": {"text": "string", "language": "string", "segments": "list"},
            },
            {
                "name": "detect_audio_events",
                "description": "Detect audio events like speech, music, sounds",
                "input_schema": {"audio_url": "string", "event_types": "list"},
                "output_schema": {"events": "list"},
            },
            {
                "name": "identify_speakers",
                "description": "Identify and segment speakers in audio",
                "input_schema": {"audio_url": "string", "num_speakers": "integer"},
                "output_schema": {"segments": "list"},
            },
            {
                "name": "classify_music",
                "description": "Classify music genre, mood, tempo",
                "input_schema": {"audio_url": "string"},
                "output_schema": {"genre": "string", "mood": "string", "tempo": "float"},
            },
            {
                "name": "find_similar_audio",
                "description": "Find acoustically or semantically similar audio",
                "input_schema": {"reference_audio_url": "string", "similarity_type": "string", "limit": "integer"},
                "output_schema": {"results": "list", "count": "integer"},
            },
        ]
