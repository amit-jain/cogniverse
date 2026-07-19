"""Media search agents must raise on a persistent Vespa 4xx, not return [].

``vespa_search_post`` retries transient failures and raises on a persistent
5xx; a 4xx (a malformed query — schema/dim drift, the "Expected X values, got
Y" class) comes back for the caller to surface. The image/document/audio agents
turned any non-200 into ``return []``, so a hard query error read as "no
results" — diverging from ``search_backend``, which raise_for_status()es on 4xx.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent
from cogniverse_agents.document_agent import DocumentAgent
from cogniverse_agents.image_search_agent import ImageSearchAgent
from cogniverse_agents.search.vespa_query import VespaSearchError

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Resp400:
    status_code = 400
    text = "Bad request: Expected 128 values, got 64"

    def json(self):  # pragma: no cover - never reached on a 4xx
        return {}


@pytest.fixture
def patch_vespa_400(monkeypatch):
    # The sub-methods do a call-time `from ...vespa_query import vespa_search_post`,
    # so patching the module attribute binds before each call.
    monkeypatch.setattr(
        "cogniverse_agents.search.vespa_query.vespa_search_post",
        lambda *a, **k: _Resp400(),
    )


class _Enc:
    @staticmethod
    def encode(_query):
        return np.zeros((2, 3), dtype=np.float32)


@pytest.mark.asyncio
async def test_image_search_raises_on_4xx(patch_vespa_400):
    agent = ImageSearchAgent.__new__(ImageSearchAgent)
    agent._tenant_id = "acme:acme"
    agent._vespa_endpoint = "http://vespa:8080"
    with pytest.raises(VespaSearchError, match="400"):
        await agent._search_vespa(
            query_embedding=np.zeros((2, 3), dtype=np.float32),
            query_text="q",
            search_mode="semantic",
            limit=5,
        )


@pytest.mark.asyncio
async def test_document_text_search_raises_on_4xx(patch_vespa_400, monkeypatch):
    # text_query_encoder is a read-only property; replace it at class level.
    monkeypatch.setattr(DocumentAgent, "text_query_encoder", _Enc())
    agent = DocumentAgent.__new__(DocumentAgent)
    agent._tenant_id = "acme:acme"
    agent._vespa_endpoint = "http://vespa:8080"
    with pytest.raises(VespaSearchError, match="400"):
        await agent._search_text("q", 5)


@pytest.mark.asyncio
async def test_audio_transcript_search_raises_on_4xx(patch_vespa_400, monkeypatch):
    # _schema_name is a read-only property; replace it at class level.
    monkeypatch.setattr(
        AudioAnalysisAgent, "_schema_name", "audio_transcript_acme_acme"
    )
    agent = AudioAnalysisAgent.__new__(AudioAnalysisAgent)
    agent._tenant_id = "acme:acme"
    agent._vespa_endpoint = "http://vespa:8080"
    with pytest.raises(VespaSearchError, match="400"):
        await agent._search_transcript("q", 5)
