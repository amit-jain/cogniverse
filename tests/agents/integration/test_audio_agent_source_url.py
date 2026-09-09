"""Real-Vespa execution coverage for AudioAnalysisAgent search paths.

Deploys the tenant-scoped ``audio_content_<tenant>`` schema into the shared
Vespa and feeds the corpus through the production ingestion pipeline
(``EmbeddingGeneratorImpl`` -> ``VespaPyClient``), so every document carries
the CLAP ``acoustic_embedding`` and the LateOn ``semantic_embedding`` /
``semantic_embedding_binary`` the audio rank profiles score against. The
transcript encoder is the ColBERT-class model the shipped audio profile names
under ``inference_services.embedding``, served by the test-owned CPU PyLate
sidecar. ``_search_transcript`` / ``_search_semantic`` / ``_search_acoustic``
/ ``_search_hybrid`` then run against that corpus through the real backend.

Pins that the agent queries the same tenant-scoped schema the ingestion
pipeline feeds into: ``schema_full_name("audio_content", tenant)``. A bare
``audio_content`` query hits an undeployed doc type, so Vespa 400s and every
search silently returns ``[]``; ``test_schema_name_matches_agent_query_target``
locks that down.
"""

from __future__ import annotations

import socket
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from cogniverse_agents.audio_analysis_agent import (
    AudioAnalysisAgent,
    AudioAnalysisDeps,
)
from cogniverse_core.query.encoders import EncoderUnavailableError
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
)
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
    EmbeddingGeneratorImpl,
)
from tests.utils.vespa_test_helpers import (
    IngestionBackendAdapter,
    deploy_tenant_schema,
    load_raw_schema_json,
    make_config_manager,
    make_ingestion_client,
    schema_full_name,
    shipped_profile,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "audio_rt"

# Everything the corpus is fed with comes off the shipped audio profile, so a
# model/service/schema change in configs/config.json surfaces here instead of
# being absorbed by a literal.
AUDIO_PROFILE = shipped_profile(profile_type="audio", embedding_type="multi_vector")
BASE_SCHEMA = AUDIO_PROFILE.schema_name
SEMANTIC_MODEL = AUDIO_PROFILE.extra_config["semantic_model"]
ACOUSTIC_MODEL = AUDIO_PROFILE.embedding_model
EMBEDDING_SERVICE = AUDIO_PROFILE.extra_config["inference_services"]["embedding"]

SCHEMAS_DIR = Path("configs/schemas")

# Three clips: two English, one Dutch, at three distinct tones. "kestrel
# telemetry" appears three times in the short briefing and once in the long
# logbook, so BM25 orders those two deterministically; the Dutch interview
# carries neither term and is the negative control for every text query.
AUDIO_CLIPS = {
    "launch_briefing": {
        "filename": "launch_briefing.wav",
        "frequency": 440,
        "source_url": "s3://corpus/audio/launch_briefing.wav",
        "language": "en",
        "duration": 42.5,
        "transcript": (
            "Kestrel telemetry held nominal through staging. Kestrel telemetry "
            "dropped once and recovered within a second. Kestrel telemetry is "
            "green for launch."
        ),
    },
    "harbour_logbook": {
        "filename": "harbour_logbook.wav",
        "frequency": 880,
        "source_url": "s3://corpus/audio/harbour_logbook.wav",
        "language": "en",
        "duration": 128.0,
        "transcript": (
            "We noted the kestrel telemetry handover at dawn, then spent the "
            "rest of the watch reading tide tables, checking the mooring lines, "
            "recording the harbour weather and filing the shift report for the "
            "crew coming aboard on the morning relief."
        ),
    },
    "atrium_interview": {
        "filename": "atrium_interview.wav",
        "frequency": 220,
        "source_url": "s3://corpus/audio/atrium_interview.wav",
        "language": "nl",
        "duration": 61.25,
        "transcript": (
            "Het interview gaat over de akoestiek van het atrium, de nagalmtijd "
            "en hoe een glazen plafond de klank van de ruimte kleurt tijdens "
            "een opname."
        ),
    },
}


def _dead_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _write_tone_wav(path: Path, frequency: int) -> None:
    t = np.linspace(0, 1.0, 48000, dtype=np.float32)
    samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(samples.tobytes())


@pytest.fixture(scope="module")
def audio_wav_files(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("audio_agent_corpus")
    paths = {}
    for clip_id, clip in AUDIO_CLIPS.items():
        wav_path = tmp_dir / clip["filename"]
        _write_tone_wav(wav_path, clip["frequency"])
        paths[clip_id] = wav_path
    return paths


@pytest.fixture(scope="module")
def acoustic_embeddings(audio_wav_files):
    """CLAP vectors for the same WAVs the pipeline embedded.

    The model is deterministic, so these are the vectors sitting in
    ``acoustic_embedding`` — an acoustic query built from one of them is the
    clip searching for itself.
    """
    generator = AudioEmbeddingGenerator(clap_model=ACOUSTIC_MODEL)
    return {
        clip_id: generator.generate_acoustic_embedding(audio_path=wav_path)
        for clip_id, wav_path in audio_wav_files.items()
    }


@pytest.fixture(scope="module")
def audio_schema(shared_vespa, pylate_server, audio_wav_files):
    """Deploy the tenant audio schema and fill it through the ingest pipeline.

    Feed path (identical to production ingestion):
      EmbeddingGeneratorImpl._process_audio_segments()
        -> AudioEmbeddingGenerator.generate_acoustic_embedding()  (CLAP 512-d)
        -> colbert_model.encode(transcript, is_query=False)       (LateOn 128-d)
        -> VespaPyClient.process() + _feed_prepared_batch()
    """
    config_manager = make_config_manager(
        shared_vespa,
        inference_service_urls={EMBEDDING_SERVICE: pylate_server},
    )
    full = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT,
        base_schema_name=BASE_SCHEMA,
        config_manager=config_manager,
    )
    http_port = shared_vespa["http_port"]
    schema_loader = FilesystemSchemaLoader(SCHEMAS_DIR)

    fed_results = {}
    for clip_id, clip in AUDIO_CLIPS.items():
        client = make_ingestion_client(
            schema_name=full,
            base_schema_name=BASE_SCHEMA,
            http_port=http_port,
            schema_loader=schema_loader,
        )
        generator = EmbeddingGeneratorImpl(
            config={
                "embedding_model": ACOUSTIC_MODEL,
                "semantic_model": SEMANTIC_MODEL,
                "embedding_type": AUDIO_PROFILE.embedding_type,
                "model_loader": AUDIO_PROFILE.model_loader,
                "schema_name": BASE_SCHEMA,
                "inference_services": {"embedding": EMBEDDING_SERVICE},
                "remote_inference_url": pylate_server,
            },
            backend_client=IngestionBackendAdapter(client),
        )
        assert generator.colbert_model.endpoint_url == pylate_server
        assert generator.colbert_model.model_name == SEMANTIC_MODEL
        fed_results[clip_id] = generator.generate_embeddings(
            {
                "video_id": f"audio_{clip_id}",
                "source_url": clip["source_url"],
                "audio_files": [
                    {
                        "audio_id": clip_id,
                        "path": str(audio_wav_files[clip_id]),
                        "filename": clip["filename"],
                    }
                ],
                "transcript": {
                    "full_text": clip["transcript"],
                    "language": clip["language"],
                    "duration": clip["duration"],
                },
            },
            output_dir=Path("/tmp"),
        )

    time.sleep(3)
    yield {
        "full": full,
        "http_port": http_port,
        "config_port": shared_vespa["config_port"],
        "config_manager": config_manager,
        "pylate_server": pylate_server,
        "fed_results": fed_results,
    }

    import requests

    for clip_id in AUDIO_CLIPS:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/audio/{full}"
                f"/docid/audio_{clip_id}_{clip_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


def _build_agent(audio_schema, config_manager) -> AudioAnalysisAgent:
    return AudioAnalysisAgent(
        deps=AudioAnalysisDeps(
            tenant_id=TENANT,
            vespa_endpoint=f"http://localhost:{audio_schema['http_port']}",
            whisper_model_size="base",
            deployed_audio_schema=True,
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
            backend_config={
                "url": "http://localhost",
                "port": audio_schema["http_port"],
                "config_port": audio_schema["config_port"],
                "schema_name": BASE_SCHEMA,
                "backend": get_config(TENANT, config_manager).get("backend"),
            },
        )
    )


@pytest.fixture
def audio_agent(audio_schema):
    """The AudioAnalysisAgent production builds, wired to the test's own Vespa.

    Driven through the real constructor so every attribute the search path
    reads (``_deployed_audio_schema``, ``_shared_backend`` and its lock,
    ``_locator``) is the one ``__init__`` sets.
    """
    return _build_agent(audio_schema, audio_schema["config_manager"])


def test_schema_name_matches_agent_query_target(audio_schema, audio_agent):
    # The agent builds audio_content_<canonical_tenant>; the deployed schema
    # must carry the same name or every audio query 404s and returns [].
    assert audio_schema["full"] == schema_full_name(BASE_SCHEMA, TENANT)
    assert audio_agent._schema_name == audio_schema["full"]


def test_audio_profile_routes_its_semantic_encoder_to_the_pylate_sidecar():
    """The audio profile's transcript embedding is a ColBERT-class model served
    by ``colbert_pylate`` — the CPU sidecar this module provisions."""
    assert EMBEDDING_SERVICE == "colbert_pylate"
    assert SEMANTIC_MODEL == "lightonai/LateOn"
    assert AUDIO_PROFILE.model_loader == "colbert"
    assert AUDIO_PROFILE.schema_config["embedding_dim"] == 128
    assert AUDIO_PROFILE.schema_config["binary_dim"] == 16


def test_audio_schema_declares_the_rank_profiles_the_agent_requests():
    """Every strategy AudioAnalysisAgent names must exist in the shipped
    schema; the hybrid path requests ``hybrid_semantic_bm25``."""
    schema = load_raw_schema_json(BASE_SCHEMA)
    assert {p["name"] for p in schema["rank_profiles"]} == {
        "default",
        "transcript_search",
        "acoustic_similarity",
        "semantic_float",
        "semantic_binary",
        "phased_semantic",
        "hybrid_semantic_bm25",
        "hybrid_acoustic_bm25",
    }


def test_all_clips_fed_through_the_production_pipeline(audio_schema):
    for clip_id, result in audio_schema["fed_results"].items():
        assert result.documents_fed == 1, f"{clip_id}: {result.errors}"
        assert result.documents_processed == 1
        assert result.errors == []


@pytest.mark.requires_docker
@pytest.mark.integration
class TestAudioAgentSourceUrl:
    @pytest.mark.asyncio
    async def test_search_transcript_carries_source_url_into_audio_url(
        self, audio_agent
    ):
        results = await audio_agent._search_transcript("kestrel telemetry", limit=10)

        # BM25 over the two English clips: three occurrences in the short
        # briefing outranks one in the long logbook. The Dutch interview
        # carries neither term.
        assert [r.audio_id for r in results] == ["launch_briefing", "harbour_logbook"]
        assert {r.audio_id: r.audio_url for r in results} == {
            "launch_briefing": AUDIO_CLIPS["launch_briefing"]["source_url"],
            "harbour_logbook": AUDIO_CLIPS["harbour_logbook"]["source_url"],
        }
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_get_audio_path_resolves_via_locator(self, audio_agent, tmp_path):
        """The agent's _get_audio_path goes through the locator, so a
        file:// URI to a real on-disk audio resolves to the same path."""
        clip = tmp_path / "audio_e2e_clip.mp3"
        clip.write_bytes(b"fake audio bytes")

        local = audio_agent._get_audio_path(f"file://{clip}")
        assert local == str(clip)


@pytest.mark.requires_docker
@pytest.mark.integration
class TestAudioAcousticHybridSearch:
    """``_search_acoustic`` binds its CLAP query vector to the schema's
    ``acoustic_similarity`` profile; ``_search_hybrid`` requests
    ``hybrid_semantic_bm25``, whose first phase scores the ColBERT
    ``semantic_embedding_binary`` the ingest pipeline wrote and whose second
    phase reranks on ``bm25(audio_title) + bm25(audio_transcript)``.
    """

    @staticmethod
    def _bind_acoustic_query(agent, vec):
        # Acoustic search must encode the query with CLAP text features
        # (generate_acoustic_text_embedding), NOT the sentence-transformer
        # semantic embedder — providing only the former asserts the call site.
        # The value handed back is the real CLAP vector of a seeded clip, so
        # the query is that clip looking for itself.
        class _ClapTextOnly:
            @staticmethod
            def generate_acoustic_text_embedding(query):
                return np.asarray(vec, dtype=np.float32)

        agent._embedding_generator = _ClapTextOnly()

    @pytest.mark.asyncio
    async def test_search_acoustic_retrieves_nearest_by_embedding(
        self, audio_agent, acoustic_embeddings
    ):
        self._bind_acoustic_query(audio_agent, acoustic_embeddings["launch_briefing"])

        results = await audio_agent._search_acoustic("any spoken query", limit=5)

        assert results[0].audio_id == "launch_briefing", [r.audio_id for r in results]
        assert {r.audio_id for r in results} == set(AUDIO_CLIPS)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)
        # transcript/duration/language must round-trip from the deployed
        # schema's audio_transcript/audio_duration/audio_language fields,
        # which the ingest pipeline writes from the transcription result.
        assert results[0].transcript == AUDIO_CLIPS["launch_briefing"]["transcript"]
        assert results[0].duration == 42.5
        assert results[0].language == "en"

    @pytest.mark.asyncio
    async def test_search_semantic_ranks_by_transcript_embedding(self, audio_agent):
        """phased_semantic scores the fed ``semantic_embedding``; a query
        lifted from one clip's transcript must rank that clip first."""
        results = await audio_agent._search_semantic(
            "kestrel telemetry is green for launch", limit=5
        )

        assert {r.audio_id for r in results} == set(AUDIO_CLIPS)
        assert results[0].audio_id == "launch_briefing", [r.audio_id for r in results]
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_hybrid_ranks_by_semantic_first_phase_then_bm25(
        self, audio_agent
    ):
        results = await audio_agent._search_hybrid("kestrel telemetry", limit=5)

        assert [r.audio_id for r in results] == ["launch_briefing", "harbour_logbook"]
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].audio_url == AUDIO_CLIPS["launch_briefing"]["source_url"]

    @pytest.mark.asyncio
    async def test_search_hybrid_follows_the_query_to_another_clip(self, audio_agent):
        """The ordering above is the corpus answering this query, not a fixed
        arrangement: a query drawn from a different clip returns that clip."""
        results = await audio_agent._search_hybrid("atrium akoestiek", limit=5)

        assert [r.audio_id for r in results] == ["atrium_interview"]
        assert results[0].language == "nl"
        assert results[0].duration == 61.25


@pytest.mark.requires_docker
@pytest.mark.integration
class TestAudioSearchEncoderOutage:
    @pytest.fixture
    def unreachable_encoder_agent(self, audio_schema):
        """An agent whose audio profile resolves the ColBERT sidecar to a port
        nothing listens on. The search backend is a process-wide singleton, so
        the registry is cleared around it to make this agent build its own."""
        registry = BackendRegistry.get_instance()
        registry.clear_instances()
        dead_url = f"http://127.0.0.1:{_dead_port()}"
        config_manager = make_config_manager(
            {
                "http_port": audio_schema["http_port"],
                "config_port": audio_schema["config_port"],
            },
            inference_service_urls={EMBEDDING_SERVICE: dead_url},
        )
        try:
            yield _build_agent(audio_schema, config_manager), dead_url
        finally:
            registry.clear_instances()

    @pytest.mark.asyncio
    async def test_hybrid_search_raises_naming_the_unreachable_service(
        self, unreachable_encoder_agent
    ):
        """An encoder-sidecar outage must surface as a named failure, never as
        an empty result the caller reads as 'no matching audio'."""
        agent, dead_url = unreachable_encoder_agent

        with pytest.raises(EncoderUnavailableError) as excinfo:
            await agent._search_hybrid("kestrel telemetry", limit=5)

        error = excinfo.value
        assert error.service == EMBEDDING_SERVICE
        assert error.endpoint == dead_url
        assert error.profile == AUDIO_PROFILE.profile_name
        message = str(error)
        assert dead_url in message
        assert SEMANTIC_MODEL in message
        assert isinstance(error.__cause__, InferenceServiceUnavailableError)
        assert error.__cause__.service == "colbert_pooling"

    @pytest.mark.asyncio
    async def test_transcript_search_survives_the_encoder_outage(
        self, unreachable_encoder_agent
    ):
        """transcript_search needs no query embedding, so the same outage must
        leave the BM25 path serving its full, correctly ordered result."""
        agent, _ = unreachable_encoder_agent

        results = await agent._search_transcript("kestrel telemetry", limit=10)

        assert [r.audio_id for r in results] == ["launch_briefing", "harbour_logbook"]
