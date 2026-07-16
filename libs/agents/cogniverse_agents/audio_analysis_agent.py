"""
Audio Analysis Agent using Whisper

Uses existing AudioTranscriber for transcription and connects to Vespa
for real audio search. Supports both transcript-based and acoustic similarity search.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from pydantic import Field as PydanticField

from cogniverse_agents.search.vespa_query import vespa_search_children
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from cogniverse_runtime.ingestion.processors.audio_transcriber import AudioTranscriber

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
    speaker_labels: List[str] = PydanticField(
        default_factory=list, description="Speaker labels"
    )
    detected_events: List[str] = PydanticField(
        default_factory=list, description="Detected events"
    )
    language: str = PydanticField("unknown", description="Detected language")
    metadata: Dict[str, Any] = PydanticField(
        default_factory=dict, description="Additional metadata"
    )


class AudioSearchInput(AgentInput):
    """Type-safe input for audio search"""

    query: str = PydanticField(..., description="Search query")
    search_mode: str = PydanticField(
        "hybrid", description="Search mode: transcript, acoustic, hybrid"
    )
    limit: int = PydanticField(20, description="Number of results")


class AudioSearchOutput(AgentOutput):
    """Type-safe output from audio search"""

    results: List[AudioResult] = PydanticField(
        default_factory=list, description="Search results"
    )
    count: int = PydanticField(0, description="Number of results")


class AudioAnalysisDeps(AgentDeps):
    """Dependencies for audio analysis agent"""

    vespa_endpoint: str = PydanticField(
        "http://localhost:8080", description="Vespa endpoint"
    )
    whisper_model_size: str = PydanticField(
        "base",
        description=(
            "Whisper model size for the in-process AudioTranscriber fallback. "
            "Ignored when whisper_endpoint is set."
        ),
    )
    whisper_endpoint: Optional[str] = PydanticField(
        None,
        description=(
            "Base URL of the vllm-asr inference pod (e.g. "
            "http://vllm-asr:8000). When set, transcribe_audio POSTs to "
            "{endpoint}/v1/audio/transcriptions instead of loading "
            "Whisper in-process — mirrors the AudioProcessor remote "
            "pattern from the ingestion pipeline. Read by the runtime "
            "from system_config.inference_service_urls['vllm_asr']."
        ),
    )
    whisper_model: str = PydanticField(
        "openai/whisper-large-v3-turbo",
        description=(
            "Model id sent in the /v1/audio/transcriptions request. Must "
            "match the model the vLLM ASR pod is serving — the chart's "
            "default is openai/whisper-large-v3-turbo."
        ),
    )
    clap_endpoint: Optional[str] = PydanticField(
        None,
        description=(
            "URL of the clap_embed sidecar for acoustic-mode query "
            "encoding. When unset, CLAP loads in-process (requires torch "
            "— unavailable in the deployed runtime image)."
        ),
    )


@dataclass
class TranscriptionResult:
    """Result from audio transcription"""

    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float


class AudioAnalysisAgent(
    A2AAgent[AudioSearchInput, AudioSearchOutput, AudioAnalysisDeps]
):
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

        self._tenant_id = deps.tenant_id
        self._vespa_endpoint = deps.vespa_endpoint
        self._whisper_model_size = deps.whisper_model_size
        self._whisper_endpoint = deps.whisper_endpoint
        self._whisper_model = deps.whisper_model

        # Initialize components (lazy loading)
        self._audio_transcriber = None
        self._embedding_generator = None

        from cogniverse_core.common.media import MediaConfig, MediaLocator

        self._locator = MediaLocator(tenant_id=deps.tenant_id, config=MediaConfig())

        logger.info(
            f"Initialized AudioAnalysisAgent for tenant: {deps.tenant_id}, "
            f"whisper: {deps.whisper_model_size}"
        )

    @property
    def _schema_name(self) -> str:
        """Tenant-scoped Vespa schema the audio ingestion pipeline feeds into."""
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        safe_tenant = canonical_tenant_id(self._tenant_id).replace(":", "_")
        return f"audio_content_{safe_tenant}"

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
            self._embedding_generator = AudioEmbeddingGenerator(
                clap_endpoint_url=self.deps.clap_endpoint
            )
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
            # Surface the failure (degraded Vespa, outage, sidecar error) —
            # returning [] here made every backend failure read as "no results".
            logger.error(f"❌ Audio search failed: {e}")
            raise

    async def transcribe_audio(
        self,
        audio_url: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.

        When ``whisper_endpoint`` is set on the agent's deps, the audio is
        POSTed multipart to ``{endpoint}/v1/audio/transcriptions`` (the
        OpenAI-compatible vLLM Whisper contract). Otherwise the in-process
        AudioTranscriber runs Whisper locally — fine for dev/test on hosts
        where the cluster ASR pod is unavailable.
        """
        logger.info(f"🎤 Transcribing audio: {audio_url}")

        audio_path = self._get_audio_path(audio_url)

        if self._whisper_endpoint:
            # _transcribe_via_sidecar is sync (blocking file read + POST) —
            # offload the whole helper off the event loop.
            result = await asyncio.to_thread(
                self._transcribe_via_sidecar, Path(audio_path), language
            )
        else:
            result = self.audio_transcriber.transcribe_audio(
                video_path=Path(audio_path), output_dir=None
            )

        transcription = TranscriptionResult(
            text=result.get("full_text", ""),
            segments=result.get("segments", []),
            language=result.get("language", "unknown"),
            confidence=1.0,
        )

        logger.info(f"✅ Transcription complete: language={transcription.language}")
        return transcription

    def _transcribe_via_sidecar(
        self, audio_path: Path, language: Optional[str]
    ) -> Dict[str, Any]:
        """POST audio multipart to vLLM ``/v1/audio/transcriptions``.

        Returns a dict in the same shape the in-process AudioTranscriber
        produces (``full_text``, ``segments``, ``language``) so the
        downstream TranscriptionResult mapping doesn't branch. Empty
        ``segments`` are passed through as-is — no synthesis — so callers
        can distinguish "no segments returned" from "single full-clip
        segment" if it matters.
        """
        import requests

        url = f"{self._whisper_endpoint.rstrip('/')}/v1/audio/transcriptions"
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data: Dict[str, Any] = {
                "model": self._whisper_model,
                "response_format": "verbose_json",
            }
            if language and language != "auto":
                data["language"] = language
            logger.info(
                f"🛰️  POST {url}  ({audio_path.stat().st_size / 1024:.1f} KiB audio)"
            )
            resp = requests.post(url, files=files, data=data, timeout=600.0)
        resp.raise_for_status()
        body = resp.json()

        return {
            "full_text": (body.get("text") or "").strip(),
            "language": body.get("language", "unknown"),
            "segments": [
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": (seg.get("text") or "").strip(),
                }
                for seg in body.get("segments") or []
            ],
            "duration": float(body.get("duration") or 0.0),
        }

    async def _search_transcript(self, query: str, limit: int) -> List[AudioResult]:
        """Search by transcript text using BM25"""

        # Build YQL query for transcript search
        yql = f"select * from {self._schema_name} where userQuery()"

        params = {
            "yql": yql,
            "query": query,
            "hits": limit,
            "ranking.profile": "transcript_search",
        }

        try:
            from cogniverse_agents.search.vespa_query import vespa_search_post

            response = await asyncio.to_thread(
                vespa_search_post, self._vespa_endpoint, params, 10
            )

            if response.status_code != 200:
                logger.error(
                    f"Vespa search failed: {response.status_code} - {response.text}"
                )
                return []

            # Parse results
            results = []
            data = response.json()

            for hit in vespa_search_children(
                data, correlation_id=f"audio_analysis_agent:{self._tenant_id}"
            ):
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
            raise

    async def _search_acoustic(self, query: str, limit: int) -> List[AudioResult]:
        """Search by acoustic similarity from a TEXT query via CLAP."""
        # CLAP text features land in the same 512-dim space as the stored audio
        # acoustic_embedding, so a text query is directly comparable to it.
        logger.info("Generating query embedding for acoustic search...")
        query_embedding = self.embedding_generator.generate_acoustic_text_embedding(
            query
        )
        return await self._search_by_acoustic_embedding(query_embedding, limit)

    async def _search_by_acoustic_embedding(
        self, query_embedding, limit: int
    ) -> List[AudioResult]:
        """Run an acoustic nearestNeighbor search from a 512-dim embedding.

        Shared by text-query acoustic search and reference-audio similarity —
        both compare a 512-dim CLAP vector against the stored
        ``acoustic_embedding`` field.
        """
        # acoustic_similarity ranks via closeness(field, acoustic_embedding),
        # which binds to a nearestNeighbor operator over the HNSW field; the
        # query tensor is the profile input query(acoustic_query).
        yql = (
            f"select * from {self._schema_name} where "
            f"{{targetHits:{limit}}}nearestNeighbor(acoustic_embedding, acoustic_query)"
        )

        params = {
            "yql": yql,
            "hits": limit,
            "ranking.profile": "acoustic_similarity",
            "input.query(acoustic_query)": query_embedding.tolist(),
        }

        try:
            from cogniverse_agents.search.vespa_query import vespa_search_post

            response = await asyncio.to_thread(
                vespa_search_post, self._vespa_endpoint, params, 10
            )

            if response.status_code != 200:
                logger.error(
                    f"Vespa search failed: {response.status_code} - {response.text}"
                )
                return []

            # Parse results
            results = []
            data = response.json()

            for hit in vespa_search_children(
                data, correlation_id=f"audio_analysis_agent:{self._tenant_id}"
            ):
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
            raise

    async def _search_hybrid(self, query: str, limit: int) -> List[AudioResult]:
        """Search by hybrid (BM25 transcript + semantic embeddings)"""

        # CLAP text features for the acoustic half (same 512-dim space as the
        # stored acoustic_embedding); the transcript half uses the raw query text.
        logger.info("Generating query embedding for hybrid search...")
        query_embedding = self.embedding_generator.generate_acoustic_text_embedding(
            query
        )

        # hybrid_acoustic_bm25 = 0.5*closeness(acoustic) + 0.5*bm25(text); the
        # acoustic half needs the nearestNeighbor operator, OR-ed with the text
        # match so either signal can surface a document.
        yql = (
            f"select * from {self._schema_name} where "
            f"({{targetHits:{limit}}}nearestNeighbor(acoustic_embedding, acoustic_query)) "
            "or userQuery()"
        )

        params = {
            "yql": yql,
            "query": query,
            "hits": limit,
            "ranking.profile": "hybrid_acoustic_bm25",
            "input.query(acoustic_query)": query_embedding.tolist(),
        }

        try:
            from cogniverse_agents.search.vespa_query import vespa_search_post

            response = await asyncio.to_thread(
                vespa_search_post, self._vespa_endpoint, params, 10
            )

            if response.status_code != 200:
                logger.error(
                    f"Vespa search failed: {response.status_code} - {response.text}"
                )
                return []

            # Parse results
            results = []
            data = response.json()

            for hit in vespa_search_children(
                data, correlation_id=f"audio_analysis_agent:{self._tenant_id}"
            ):
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
            raise

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
                # Acoustic similarity: encode the reference audio's CLAP
                # embedding and search the same acoustic_embedding space.
                from pathlib import Path as _Path

                audio_path = self._get_audio_path(reference_audio_url)
                reference_embedding = await asyncio.to_thread(
                    self.embedding_generator.generate_acoustic_embedding,
                    _Path(audio_path),
                )
                results = await self._search_by_acoustic_embedding(
                    reference_embedding, limit
                )

            logger.info(f"✅ Found {len(results)} similar audio files")
            return results

        except Exception as e:
            logger.error(f"❌ Similar audio search failed: {e}")
            raise

    def _get_audio_path(self, audio_url: str) -> str:
        """Resolve an audio URL or path to a local file via the MediaLocator.

        ``file://``, bare paths, and ``pvc://`` short-circuit to identity;
        ``http(s)://``, ``s3://``, etc. are fetched and cached locally.
        """
        return str(self._locator.localize(self._locator.to_canonical_uri(audio_url)))

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
        self.emit_progress("encoding", "Encoding audio query...")
        self.emit_progress("retrieval", "Searching audio content...")
        results = await self.search_audio(
            query=input.query,
            search_mode=input.search_mode,
            limit=input.limit,
        )

        self.emit_progress("complete", "Audio search complete.")
        return AudioSearchOutput(results=results, count=len(results))

    def _dspy_to_a2a_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DSPy result to A2A output format."""
        results = result.get("results", [])
        return {
            "status": "success",
            "agent": self.agent_name,
            "result_type": "audio_search_results",
            "count": result.get("count", len(results)),
            "results": [
                r.model_dump() if hasattr(r, "model_dump") else r for r in results
            ],
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "search_audio",
                "description": "Search audio content by transcript or acoustic features",
                "input_schema": {
                    "query": "string",
                    "search_mode": "string",
                    "limit": "integer",
                },
                "output_schema": {"results": "list", "count": "integer"},
            },
            {
                "name": "transcribe_audio",
                "description": "Transcribe audio to text using Whisper",
                "input_schema": {"audio_url": "string"},
                "output_schema": {
                    "text": "string",
                    "language": "string",
                    "segments": "list",
                },
            },
            {
                "name": "find_similar_audio",
                "description": "Find acoustically or semantically similar audio",
                "input_schema": {
                    "reference_audio_url": "string",
                    "similarity_type": "string",
                    "limit": "integer",
                },
                "output_schema": {"results": "list", "count": "integer"},
            },
        ]
