"""
Audio Analysis Agent using Whisper

Uses existing AudioTranscriber for transcription and connects to Vespa
for real audio search. Supports transcript, semantic, and acoustic search.
"""

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

import dspy
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, field_validator, model_validator

from cogniverse_agents.search.vespa_query import (
    VespaSearchError,
    vespa_search_children,
)
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.registries.backend_registry import get_backend_registry
from cogniverse_foundation.config.inference_auth import inference_headers
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from cogniverse_runtime.ingestion.processors.audio_transcriber import AudioTranscriber

logger = logging.getLogger(__name__)

AUDIO_SEARCH_PROFILE = "audio_clap_semantic"


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
        "hybrid", description="Search mode: transcript, semantic, acoustic, hybrid"
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

    _resolved_whisper_headers: Mapping[str, str] = PrivateAttr(
        default_factory=lambda: MappingProxyType({})
    )
    _whisper_auth_resolved: bool = PrivateAttr(default=False)
    _resolved_clap_headers: Mapping[str, str] = PrivateAttr(
        default_factory=lambda: MappingProxyType({})
    )
    _clap_auth_resolved: bool = PrivateAttr(default=False)

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
    whisper_headers: Dict[str, str] = PydanticField(
        default_factory=dict,
        repr=False,
        description=(
            "Validated request headers from the resolved Whisper endpoint. "
            "Authenticated endpoints accept only an Authorization bearer value."
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
    clap_headers: Dict[str, str] = PydanticField(
        default_factory=dict,
        repr=False,
        description=(
            "Validated request headers from the resolved CLAP endpoint. "
            "Authenticated endpoints accept only an Authorization bearer value."
        ),
    )
    backend_type: str = PydanticField("vespa", description="Backend type")
    backend_config: Dict[str, Any] = PydanticField(
        default_factory=dict,
        repr=False,
        description=(
            "Merged backend config, including backend.profiles, used to build "
            "the shared search backend."
        ),
    )
    config_manager: Any = PydanticField(
        None,
        repr=False,
        description="ConfigManager for search backend resolution",
    )
    schema_loader: Any = PydanticField(
        None,
        repr=False,
        description="SchemaLoader for search backend resolution",
    )

    @field_validator("whisper_headers")
    @classmethod
    def validate_whisper_headers(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not value:
            return {}
        if set(value) != {"Authorization"}:
            raise ValueError("whisper_headers must contain only Authorization")
        authorization = value["Authorization"]
        scheme, separator, token = authorization.partition(" ")
        if scheme != "Bearer" or not separator or not token or token != token.strip():
            raise ValueError(
                "whisper_headers Authorization must be a canonical bearer value"
            )
        return dict(value)

    @field_validator("clap_headers")
    @classmethod
    def validate_clap_headers(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not value:
            return {}
        if set(value) != {"Authorization"}:
            raise ValueError("clap_headers must contain only Authorization")
        authorization = value["Authorization"]
        scheme, separator, token = authorization.partition(" ")
        if scheme != "Bearer" or not separator or not token or token != token.strip():
            raise ValueError(
                "clap_headers Authorization must be a canonical bearer value"
            )
        return dict(value)

    @model_validator(mode="after")
    def validate_endpoint_auth(self) -> "AudioAnalysisDeps":
        if not self._whisper_auth_resolved:
            if self.whisper_headers and not self.whisper_endpoint:
                raise ValueError("whisper_headers requires whisper_endpoint")
            configured_headers = (
                inference_headers(self.whisper_endpoint)
                if self.whisper_endpoint
                else {}
            )
            if configured_headers and "whisper_headers" in self.model_fields_set:
                raise ValueError(
                    "whisper_headers must not be supplied for a Modal endpoint"
                )
            self._resolved_whisper_headers = MappingProxyType(
                dict(configured_headers or self.whisper_headers)
            )
            self._whisper_auth_resolved = True
        if not self._clap_auth_resolved:
            if self.clap_headers and not self.clap_endpoint:
                raise ValueError("clap_headers requires clap_endpoint")
            configured_headers = (
                inference_headers(self.clap_endpoint) if self.clap_endpoint else {}
            )
            if configured_headers and "clap_headers" in self.model_fields_set:
                raise ValueError(
                    "clap_headers must not be supplied for a Modal endpoint"
                )
            self._resolved_clap_headers = MappingProxyType(
                dict(configured_headers or self.clap_headers)
            )
            self._clap_auth_resolved = True
        return self


@dataclass
class TranscriptionResult:
    """Result from audio transcription"""

    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float


class RemoteTranscriptionContractError(ValueError):
    """The remote ASR endpoint returned a malformed success response."""


def _remote_contract_error(
    url: str, path: str, detail: str
) -> RemoteTranscriptionContractError:
    return RemoteTranscriptionContractError(
        f"Remote transcription response from {url} has invalid {path}: {detail}"
    )


def _required_remote_field(
    value: Mapping[str, Any], key: str, url: str, path: str
) -> Any:
    if key not in value:
        raise _remote_contract_error(url, f"{path}.{key}", "field is required")
    return value[key]


def _remote_non_negative_number(value: Any, url: str, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise _remote_contract_error(url, path, "expected a finite non-negative number")
    return float(value)


def _remote_duration_seconds(value: Any, url: str) -> float:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"(?:0|[1-9]\d*)(?:\.\d+)?", value) is None
    ):
        raise _remote_contract_error(
            url,
            "$.duration",
            "expected a finite non-negative decimal string",
        )
    duration = float(value)
    if not math.isfinite(duration):
        raise _remote_contract_error(
            url,
            "$.duration",
            "expected a finite non-negative decimal string",
        )
    return duration


def _parse_remote_transcription(body: Any, url: str) -> Dict[str, Any]:
    if not isinstance(body, Mapping):
        raise _remote_contract_error(url, "$", "expected an object")

    text = _required_remote_field(body, "text", url, "$")
    if not isinstance(text, str):
        raise _remote_contract_error(url, "$.text", "expected a string")

    language = _required_remote_field(body, "language", url, "$")
    if not isinstance(language, str) or not language.strip():
        raise _remote_contract_error(url, "$.language", "expected a non-empty string")

    duration = _remote_duration_seconds(
        _required_remote_field(body, "duration", url, "$"), url
    )
    raw_segments = _required_remote_field(body, "segments", url, "$")
    if not isinstance(raw_segments, list):
        raise _remote_contract_error(url, "$.segments", "expected a list")

    segments: List[Dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        segment_path = f"$.segments[{index}]"
        if not isinstance(raw_segment, Mapping):
            raise _remote_contract_error(url, segment_path, "expected an object")
        start = _remote_non_negative_number(
            _required_remote_field(raw_segment, "start", url, segment_path),
            url,
            f"{segment_path}.start",
        )
        end = _remote_non_negative_number(
            _required_remote_field(raw_segment, "end", url, segment_path),
            url,
            f"{segment_path}.end",
        )
        if end < start:
            raise _remote_contract_error(
                url,
                f"{segment_path}.end",
                "must be greater than or equal to start",
            )
        if end > duration:
            raise _remote_contract_error(
                url, f"{segment_path}.end", "must not exceed $.duration"
            )
        segment_text = _required_remote_field(raw_segment, "text", url, segment_path)
        if not isinstance(segment_text, str):
            raise _remote_contract_error(
                url, f"{segment_path}.text", "expected a string"
            )
        segments.append({"start": start, "end": end, "text": segment_text.strip()})

    return {
        "full_text": text.strip(),
        "language": language.strip(),
        "segments": segments,
        "duration": duration,
    }


class AudioAnalysisAgent(
    A2AAgent[AudioSearchInput, AudioSearchOutput, AudioAnalysisDeps]
):
    """
    Type-safe audio content analysis using Whisper and Vespa.

    Capabilities:
    - Speech transcription using existing AudioTranscriber
    - Transcript, semantic, and hybrid text search
    - Acoustic similarity search
    - Hybrid semantic + BM25 search
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
                desc="Search mode: transcript, semantic, acoustic, hybrid"
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
        self._whisper_headers = deps._resolved_whisper_headers
        self._whisper_model = deps.whisper_model
        self._clap_headers = deps._resolved_clap_headers
        self._backend_type = deps.backend_type
        self._backend_config = dict(deps.backend_config or {})
        self.config_manager = deps.config_manager
        self.schema_loader = deps.schema_loader

        # Initialize components (lazy loading)
        self._audio_transcriber = None
        self._embedding_generator = None
        self._shared_backend = None
        self._audio_transcriber_lock = Lock()
        self._embedding_generator_lock = Lock()
        self._shared_backend_lock = Lock()

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
            with self._audio_transcriber_lock:
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
            with self._embedding_generator_lock:
                if self._embedding_generator is None:
                    logger.info("Loading audio embedding models...")
                    if self.deps.clap_endpoint is None:
                        self._embedding_generator = AudioEmbeddingGenerator()
                    else:
                        self._embedding_generator = AudioEmbeddingGenerator(
                            clap_endpoint_url=self.deps.clap_endpoint,
                            _resolved_headers=self._clap_headers,
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
            search_mode: "acoustic", "transcript", "semantic", or "hybrid"
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
            elif search_mode == "semantic":
                results = await self._search_semantic(query, limit)
            elif search_mode == "hybrid":
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
            transcription = TranscriptionResult(
                text=result["full_text"],
                segments=result["segments"],
                language=result["language"],
                confidence=1.0,
            )
        else:
            # Local Whisper is a heavy blocking model forward — same
            # offload contract as the sidecar branch above.
            result = await asyncio.to_thread(
                self.audio_transcriber.transcribe_audio,
                video_path=Path(audio_path),
                output_dir=None,
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

        A successful response must include typed text, language, duration,
        and timestamped segments. Empty ``segments`` remain valid so callers
        can distinguish silence from a single full-clip segment.
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
            resp = requests.post(
                url,
                files=files,
                data=data,
                headers=self._whisper_headers,
                timeout=600.0,
            )
        resp.raise_for_status()
        try:
            body = resp.json()
        except ValueError as exc:
            raise _remote_contract_error(
                url, "$", "response body is not valid JSON"
            ) from exc

        return _parse_remote_transcription(body, url)

    def _get_backend(self):
        """Get or create the shared search backend (lazy initialization)."""
        if self._shared_backend is not None:
            return self._shared_backend
        with self._shared_backend_lock:
            if self._shared_backend is None:
                registry = get_backend_registry()
                self._shared_backend = registry.get_search_backend(
                    self._backend_type,
                    self._backend_config,
                    config_manager=self.config_manager,
                    schema_loader=self.schema_loader,
                )
                logger.info("Shared audio search backend initialized")
            return self._shared_backend

    def _build_backend_query(
        self, query: str, strategy: str, limit: int
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "type": "audio",
            "profile": AUDIO_SEARCH_PROFILE,
            "strategy": strategy,
            "tenant_id": self._tenant_id,
            "top_k": limit,
        }

    def _search_result_to_audio_result(self, search_result: Any) -> AudioResult:
        document = search_result.document
        metadata = dict(getattr(document, "metadata", {}) or {})
        transcript = document.text_content or metadata.get("audio_transcript", "")
        return AudioResult(
            audio_id=metadata.get("audio_id", document.id),
            audio_url=metadata.get("source_url", ""),
            title=metadata.get("audio_title", ""),
            transcript=transcript,
            duration=metadata.get("audio_duration", 0.0),
            relevance_score=search_result.score,
            speaker_labels=metadata.get("speaker_labels", []),
            detected_events=metadata.get("detected_events", []),
            language=metadata.get("audio_language", "unknown"),
            metadata=metadata,
        )

    async def _search_backend_mode(
        self, query: str, strategy: str, limit: int
    ) -> List[AudioResult]:
        backend = self._get_backend()
        query_dict = self._build_backend_query(query, strategy, limit)
        # backend.search is synchronous; keep the async audio API responsive.
        search_results = await asyncio.to_thread(backend.search, query_dict)
        return [self._search_result_to_audio_result(hit) for hit in search_results]

    async def _search_transcript(self, query: str, limit: int) -> List[AudioResult]:
        """Search by transcript text using backend BM25."""
        try:
            return await self._search_backend_mode(query, "transcript_search", limit)
        except Exception as e:
            logger.error(f"❌ Transcript search failed: {e}")
            raise

    async def _search_semantic(self, query: str, limit: int) -> List[AudioResult]:
        """Search by ColBERT semantic similarity through the backend."""
        try:
            return await self._search_backend_mode(query, "phased_semantic", limit)
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            raise

    async def _search_acoustic(self, query: str, limit: int) -> List[AudioResult]:
        """Search by acoustic similarity from a TEXT query via CLAP."""
        # CLAP text features land in the same 512-dim space as the stored audio
        # acoustic_embedding, so a text query is directly comparable to it.
        logger.info("Generating query embedding for acoustic search...")
        # Blocking CLAP HTTP call — off the loop, lazy generator build included.
        query_embedding = await asyncio.to_thread(
            lambda: self.embedding_generator.generate_acoustic_text_embedding(query)
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
                raise VespaSearchError(
                    f"Vespa search returned {response.status_code}: {response.text[:200]}"
                )

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
                        transcript=fields.get("audio_transcript", ""),
                        duration=fields.get("audio_duration", 0.0),
                        relevance_score=hit.get("relevance", 0.0),
                        speaker_labels=fields.get("speaker_labels", []),
                        detected_events=fields.get("detected_events", []),
                        language=fields.get("audio_language", "unknown"),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"❌ Acoustic search failed: {e}")
            raise

    async def _search_hybrid(self, query: str, limit: int) -> List[AudioResult]:
        """Search by semantic+BM25 hybrid text retrieval through the backend."""
        try:
            return await self._search_backend_mode(query, "hybrid_semantic_bm25", limit)
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
                "description": "Search audio content by transcript, semantic, or acoustic features",
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
