"""Acoustic audio similarity must actually search, not silently return [].

find_similar_audio(..., similarity_type="acoustic") logged "not yet
implemented" and returned an empty list, so a caller could not tell the feature
was missing from a genuine no-match. It now encodes the reference audio's CLAP
embedding and searches the acoustic_embedding space.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _agent():
    agent = object.__new__(AudioAnalysisAgent)
    agent._tenant_id = "acme:acme"
    agent._vespa_endpoint = "http://vespa:8080"
    agent._deployed_audio_schema = None
    # embedding_generator is a lazy property; seed its backing attr.
    gen = MagicMock()
    gen.generate_acoustic_embedding = MagicMock(
        return_value=np.ones(512, dtype=np.float32)
    )
    agent._embedding_generator = gen
    return agent


def _vespa_response(video_ids):
    children = [
        {
            "fields": {"audio_id": vid, "source_url": f"s3://a/{vid}.wav"},
            "relevance": 0.9,
        }
        for vid in video_ids
    ]
    return SimpleNamespace(
        status_code=200, text="", json=lambda: {"root": {"children": children}}
    )


@pytest.mark.asyncio
async def test_acoustic_similarity_searches_and_returns_results(monkeypatch):
    agent = _agent()
    backend = MagicMock()
    backend.schema_exists = MagicMock(return_value=True)
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
    )
    with (
        patch.object(agent, "_get_audio_path", return_value="/tmp/ref.wav"),
        patch(
            "cogniverse_agents.search.vespa_query.vespa_search_post",
            return_value=_vespa_response(["a1", "a2"]),
        ) as post,
    ):
        results = await agent.find_similar_audio(
            "s3://bucket/ref.wav", similarity_type="acoustic", limit=5
        )

    # The reference audio was encoded and a real acoustic search ran.
    agent.embedding_generator.generate_acoustic_embedding.assert_called_once()
    post.assert_called_once()
    assert [r.audio_id for r in results] == ["a1", "a2"]
    assert backend.schema_exists.call_args_list == [
        call("audio_content", tenant_id="acme:acme"),
        call("audio_content", tenant_id="acme:acme"),
    ]


@pytest.mark.asyncio
async def test_search_by_acoustic_embedding_binds_query_tensor(monkeypatch):
    agent = _agent()
    backend = MagicMock()
    backend.schema_exists = MagicMock(return_value=True)
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
    )
    captured = {}

    def fake_post(endpoint, params, timeout):
        captured.update(params)
        return _vespa_response(["x"])

    with patch(
        "cogniverse_agents.search.vespa_query.vespa_search_post", side_effect=fake_post
    ):
        await agent._search_by_acoustic_embedding(np.ones(512, dtype=np.float32), 3)

    assert captured["ranking.profile"] == "acoustic_similarity"
    assert "nearestNeighbor(acoustic_embedding, acoustic_query)" in captured["yql"]
    assert len(captured["input.query(acoustic_query)"]) == 512
    # The query must target the tenant-scoped schema ingestion feeds into,
    # not the bare base name (which is never deployed).
    assert "from audio_content_acme_acme where" in captured["yql"]
    assert backend.schema_exists.call_args_list == [
        call("audio_content", tenant_id="acme:acme")
    ]


@pytest.mark.asyncio
async def test_find_similar_audio_returns_empty_when_schema_missing(monkeypatch):
    agent = _agent()
    backend = MagicMock()
    backend.schema_exists = MagicMock(return_value=False)
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
    )

    results = await agent.find_similar_audio(
        "s3://bucket/ref.wav", similarity_type="acoustic", limit=5
    )

    assert results == []
    assert backend.schema_exists.call_args_list == [
        call("audio_content", tenant_id="acme:acme")
    ]


@pytest.mark.asyncio
async def test_search_by_acoustic_embedding_raises_when_schema_lookup_fails(
    monkeypatch,
):
    agent = _agent()
    backend = MagicMock()
    backend.schema_exists = MagicMock(
        side_effect=RuntimeError("schema registry unavailable")
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
    )

    with pytest.raises(RuntimeError, match="schema registry unavailable"):
        await agent._search_by_acoustic_embedding(np.ones(512, dtype=np.float32), 3)

    assert backend.schema_exists.call_args_list == [
        call("audio_content", tenant_id="acme:acme")
    ]


@pytest.mark.asyncio
async def test_semantic_similarity_still_uses_transcript_path():
    agent = _agent()
    agent.transcribe_audio = AsyncMock(
        return_value=SimpleNamespace(text="a dog barking")
    )
    agent._search_transcript = AsyncMock(return_value=["stub"])

    results = await agent.find_similar_audio(
        "s3://bucket/ref.wav", similarity_type="semantic"
    )
    agent._search_transcript.assert_awaited_once()
    assert results == ["stub"]
