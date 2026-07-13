"""Real search span -> relevance annotation -> triplet, end to end.

Drives the ACTUAL ``SearchAgent._process_impl`` (its real telemetry
recording), reads the span back out of a real Phoenix, writes a real
``result_relevance`` annotation through the dashboard's writer helper, and
mines a triplet with the real ``TripletExtractor``.

This is the test that was missing: every prior optimization/eval test that
touched a search span emitted the span by hand with the ideal shape, so none
noticed that a real search recorded ``input.value`` as a ``{query,top_k,
strategy}`` JSON blob, never recorded the result set, and carried no modality
— which left the extractor mining zero triplets in production. Here the span
is produced only by the real agent, so its recorded shape is the thing under
test.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps, SearchInput
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_dashboard.utils.annotations import persist_result_relevance
from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor

pytestmark = pytest.mark.integration

QUERY = "cat playing fetch in a park"
POS_CONTENT = "a tabby cat chasing a red ball across the grass"
NEG_CONTENT = "a dog sleeping on a leather couch"


class _StubEncoder:
    """Stands in for the remote query encoder (not the code under test)."""

    def encode(self, query: str):
        return np.zeros((1, 128), dtype=np.float32)


class _StubBackend:
    """Returns two deterministic SearchResult-shaped rows.

    Matches what ``_search_by_text`` reads: ``sr.document.id`` / ``sr.score`` /
    ``sr.document.metadata`` (spread into the result dict).
    """

    def search(self, query_dict):
        return [
            SimpleNamespace(
                document=SimpleNamespace(
                    id="vid_pos", metadata={"text_content": POS_CONTENT}
                ),
                score=0.91,
            ),
            SimpleNamespace(
                document=SimpleNamespace(
                    id="vid_neg", metadata={"text_content": NEG_CONTENT}
                ),
                score=0.82,
            ),
        ]


def _build_search_agent(tenant_id: str) -> SearchAgent:
    with patch(
        "cogniverse_agents.search_agent.QueryEncoderFactory.create_encoder",
        return_value=_StubEncoder(),
    ):
        agent = SearchAgent(
            deps=SearchAgentDeps(
                tenant_id=tenant_id,
                backend_url="http://localhost",
                backend_port=8080,
                auto_create_memory_schema=False,
            ),
            schema_loader=FilesystemSchemaLoader(
                base_path=Path("tests/system/resources/schemas")
            ),
            config_manager=None,
            port=8033,
        )
    agent.query_encoder = _StubEncoder()
    agent._get_backend = lambda: _StubBackend()
    # Memory would reach Mem0/Vespa; disable so the search stays deterministic.
    agent.is_memory_enabled = lambda: False
    return agent


@pytest.mark.asyncio
async def test_real_search_span_carries_io_and_yields_triplet(real_telemetry):
    tenant_id = f"trip{uuid4().hex[:8]}"
    agent = _build_search_agent(tenant_id)
    agent.set_telemetry_manager(real_telemetry)

    out = await agent.process(
        SearchInput(
            query=QUERY,
            tenant_id=tenant_id,
            modality="video",
            # Skip the internal DSPy rewrite (no LM in this test).
            enhanced_query=QUERY,
            top_k=5,
        )
    )

    assert out.span_id is not None
    assert len(out.span_id) == 16
    int(out.span_id, 16)  # 16-hex

    canonical = canonical_tenant_id(tenant_id)
    project = real_telemetry.config.get_project_name(canonical)
    provider = real_telemetry.get_provider(tenant_id=canonical, project_name=project)

    # 1. The span the agent actually emitted must carry the clean query, the
    #    modality, and the full result set — the shape the extractor reads.
    span_row = None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=1000,
        )
        if spans is not None and not spans.empty and "context.span_id" in spans.columns:
            hit = spans[spans["context.span_id"] == out.span_id]
            if not hit.empty:
                span_row = hit.iloc[0]
                break
        await asyncio.sleep(2)

    assert span_row is not None, f"span {out.span_id} not found in {project}"
    assert span_row["attributes.input.value"] == QUERY
    assert span_row["attributes.modality"] == "video"
    payload = json.loads(span_row["attributes.output.value"])
    assert {p["document_id"] for p in payload} == {"vid_pos", "vid_neg"}
    assert {p["content"] for p in payload} == {POS_CONTENT, NEG_CONTENT}

    # 2. Write the relevance annotation through the real dashboard helper.
    score = await persist_result_relevance(
        provider, project, out.span_id, "vid_pos", "Highly Relevant"
    )
    assert score == 1.0

    # 3. The real extractor must mine exactly the triplet those two rows imply.
    extractor = TripletExtractor(provider=provider)
    triplets = []
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        triplets = await extractor.extract(
            project=project,
            modality="video",
            strategy="top_k",
            min_triplets=1,
        )
        if triplets:
            break
        await asyncio.sleep(2)

    assert len(triplets) == 1
    t = triplets[0]
    assert t.anchor == QUERY
    assert t.positive == POS_CONTENT
    assert t.negative == NEG_CONTENT
    assert t.modality == "video"
    assert t.metadata["span_id"] == out.span_id
