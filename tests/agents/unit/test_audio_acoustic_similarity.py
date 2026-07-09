"""Acoustic audio similarity must actually search, not silently return [].

find_similar_audio(..., similarity_type="acoustic") logged "not yet
implemented" and returned an empty list, so a caller could not tell the feature
was missing from a genuine no-match. It now encodes the reference audio's CLAP
embedding and searches the acoustic_embedding space.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent


def _agent():
    agent = object.__new__(AudioAnalysisAgent)
    agent._vespa_endpoint = "http://vespa:8080"
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
async def test_acoustic_similarity_searches_and_returns_results():
    agent = _agent()
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


@pytest.mark.asyncio
async def test_search_by_acoustic_embedding_binds_query_tensor():
    agent = _agent()
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
