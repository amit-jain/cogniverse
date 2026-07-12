"""Real-Vespa execution coverage for AudioAnalysisAgent search paths.

Deploys the tenant-scoped ``audio_content_<tenant>`` schema into the shared
Vespa, feeds audio docs carrying known ``source_url`` and acoustic embeddings,
then drives ``_search_transcript`` / ``_search_acoustic`` / ``_search_hybrid``
against the real backend.

Pins that the agent queries the same tenant-scoped schema the ingestion
pipeline feeds into: ``schema_full_name("audio_content", tenant)``. A bare
``audio_content`` query — what the agent issued before this was tenant-aware —
hits an undeployed doc type, so Vespa 400s and every search silently returns
``[]``; ``test_schema_name_matches_agent_query_target`` locks that down.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent
from cogniverse_core.common.media import MediaConfig, MediaLocator
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "audio_rt"
BASE_SCHEMA = "audio_content"

# A 512-d acoustic query vector aligned with the "match" doc, orthogonal to
# the "other" doc.
_MATCH_VEC = [1.0] + [0.0] * 511
_OTHER_VEC = [0.0] * 511 + [1.0]


@pytest.fixture(scope="module")
def audio_schema(shared_vespa):
    full = deploy_tenant_schema(
        shared_vespa, tenant_id=TENANT, base_schema_name=BASE_SCHEMA
    )
    http_port = shared_vespa["http_port"]

    transcript_docs = [
        {
            "audio_id": f"audio_e2e_{i}",
            "audio_title": f"clip {i}",
            "source_url": f"s3://corpus/audio/clip_{i}.mp3",
            "audio_transcript": f"hello world this is clip number {i} talking",
        }
        for i in range(3)
    ]
    acoustic_docs = {
        "audio_match": {
            "audio_id": "audio_match",
            "audio_title": "matching clip",
            "source_url": "s3://corpus/audio/match.mp3",
            "audio_transcript": "a person describing the matching clip in detail",
            "acoustic_embedding": {"values": _MATCH_VEC},
        },
        "audio_other": {
            "audio_id": "audio_other",
            "audio_title": "other noise",
            "source_url": "s3://corpus/audio/other.mp3",
            "audio_transcript": "completely unrelated background sounds here",
            "acoustic_embedding": {"values": _OTHER_VEC},
        },
    }

    fed: list[str] = []
    for i, doc in enumerate(transcript_docs):
        doc_id = f"audio_e2e_doc_{i}"
        r = requests.post(
            f"http://localhost:{http_port}/document/v1/audio/{full}/docid/{doc_id}",
            json={"fields": doc},
            timeout=15,
        )
        assert r.status_code in (200, 201), r.text[:300]
        fed.append(doc_id)
    for doc_id, fields in acoustic_docs.items():
        r = requests.post(
            f"http://localhost:{http_port}/document/v1/audio/{full}/docid/{doc_id}",
            json={"fields": fields},
            timeout=15,
        )
        assert r.status_code in (200, 201), r.text[:300]
        fed.append(doc_id)

    time.sleep(2)
    yield {
        "full": full,
        "http_port": http_port,
        "transcript_docs": transcript_docs,
        "acoustic_docs": acoustic_docs,
    }

    for doc_id in fed:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/audio/{full}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


@pytest.fixture
def audio_agent(audio_schema, tmp_path):
    """Bare AudioAnalysisAgent with just the attributes the search path reads."""
    agent = AudioAnalysisAgent.__new__(AudioAnalysisAgent)
    agent._tenant_id = TENANT
    agent._vespa_endpoint = f"http://localhost:{audio_schema['http_port']}"
    agent._whisper_model_size = "base"
    agent._audio_transcriber = None
    agent._embedding_generator = None
    agent._locator = MediaLocator(
        tenant_id=TENANT,
        config=MediaConfig(),
        cache_root=tmp_path / "audio-cache",
    )
    return agent


def test_schema_name_matches_agent_query_target(audio_schema):
    # The agent builds audio_content_<canonical_tenant>; the deployed schema
    # must carry the same name or every audio query 404s and returns [].
    assert audio_schema["full"] == schema_full_name(BASE_SCHEMA, TENANT)
    agent = AudioAnalysisAgent.__new__(AudioAnalysisAgent)
    agent._tenant_id = TENANT
    assert agent._schema_name == audio_schema["full"]


@pytest.mark.requires_docker
@pytest.mark.integration
class TestAudioAgentSourceUrl:
    @pytest.mark.asyncio
    async def test_search_transcript_carries_source_url_into_audio_url(
        self, audio_agent, audio_schema
    ):
        results = await audio_agent._search_transcript("clip", limit=10)

        assert results, "search returned no results — BM25 should match 'clip'"

        # Every returned audio_url must be one of the canonical URIs we wrote.
        result_urls = {r.audio_url for r in results}
        expected_urls = {d["source_url"] for d in audio_schema["transcript_docs"]}
        assert result_urls.intersection(expected_urls), (
            f"audio_url did not carry source_url through; "
            f"got={result_urls!r} expected one of {expected_urls!r}"
        )
        # Pre-rollout corpora with empty source_url would have surfaced
        # empty strings; assert the field is fully populated.
        for r in results:
            assert r.audio_url.startswith("s3://"), (
                f"audio_url not a canonical s3 URI: {r.audio_url!r}"
            )

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
    """_search_acoustic / _search_hybrid must bind their query vector to the
    real acoustic_similarity / hybrid_acoustic_bm25 rank profiles against the
    tenant-scoped schema."""

    @staticmethod
    def _stub_embedding(agent, vec):
        # Acoustic search must encode the query with CLAP text features
        # (generate_acoustic_text_embedding), NOT the sentence-transformer
        # semantic embedder — providing only the former asserts the call site.
        agent._embedding_generator = SimpleNamespace(
            generate_acoustic_text_embedding=lambda q: np.array(vec, dtype=np.float32)
        )

    @pytest.mark.asyncio
    async def test_search_acoustic_retrieves_nearest_by_embedding(self, audio_agent):
        self._stub_embedding(audio_agent, _MATCH_VEC)

        results = await audio_agent._search_acoustic("any spoken query", limit=5)

        assert results, "acoustic search returned no results"
        assert results[0].audio_id == "audio_match", [r.audio_id for r in results]
        assert results[0].relevance_score > 0

    @pytest.mark.asyncio
    async def test_search_hybrid_retrieves_with_acoustic_and_text(self, audio_agent):
        self._stub_embedding(audio_agent, _MATCH_VEC)

        results = await audio_agent._search_hybrid("matching clip", limit=5)

        assert results, "hybrid search returned no results"
        ids = [r.audio_id for r in results]
        assert "audio_match" in ids
        assert results[0].audio_id == "audio_match", ids
