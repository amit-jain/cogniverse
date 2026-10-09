"""Real search span -> relevance annotation -> triplet, end to end.

Drives the ACTUAL ``SearchAgent._process_impl`` (its real telemetry
recording), reads the span back out of a real Phoenix, writes a real
``result_relevance`` annotation through the shared relevance writer, and
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
from uuid import uuid4

import pytest

from cogniverse_agents.search_agent import SearchInput
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor
from cogniverse_foundation.telemetry.span_contract import (
    RESULT_RELEVANCE,
    SpanNotInProjectError,
    persist_result_relevance,
    span_readable_within_s,
)
from tests.utils.stub_search import build_stub_search_agent

pytestmark = pytest.mark.integration

QUERY = "cat playing fetch in a park"
POS_CONTENT = "a tabby cat chasing a red ball across the grass"
NEG_CONTENT = "a dog sleeping on a leather couch"


def _build_search_agent(tenant_id: str):
    return build_stub_search_agent(
        tenant_id,
        [
            ("vid_pos", 0.91, {"text_content": POS_CONTENT}),
            ("vid_neg", 0.82, {"text_content": NEG_CONTENT}),
        ],
    )


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

    # 2. Write the relevance annotation through the shared writer.
    #    Each result keeps its own rating: rating the negative afterwards
    #    must not replace the positive's.
    readable_within_s = span_readable_within_s(real_telemetry.config.batch_config)
    scores = [
        await persist_result_relevance(
            provider,
            project,
            out.span_id,
            "vid_pos",
            "Highly Relevant",
            readable_within_s=readable_within_s,
        ),
        await persist_result_relevance(
            provider,
            project,
            out.span_id,
            "vid_neg",
            "Not Relevant",
            readable_within_s=readable_within_s,
        ),
    ]
    assert scores == [1.0, 0.0]
    # The span is not one of another tenant's project, so it is not rated there.
    other_project = real_telemetry.config.get_project_name(
        canonical_tenant_id(f"other{uuid4().hex[:8]}")
    )
    with pytest.raises(
        SpanNotInProjectError,
        match=f"^span {out.span_id} is not in project {other_project}$",
    ):
        await persist_result_relevance(
            provider,
            other_project,
            out.span_id,
            "vid_neg",
            "Highly Relevant",
            readable_within_s=readable_within_s,
        )

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

    # Both ratings are stored, each under its own result.
    span_frame = span_row.to_frame().T
    ratings = set()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        annotations = await provider.annotations.get_annotations(
            spans_df=span_frame, project=project, annotation_names=[RESULT_RELEVANCE]
        )
        ratings = {
            (
                TripletExtractor._annotation_result_id(row),
                row["result.label"],
                row["result.score"],
            )
            for _, row in annotations.iterrows()
        }
        if len(ratings) == 2:
            break
        await asyncio.sleep(2)
    assert ratings == {
        ("vid_pos", "Highly Relevant", 1.0),
        ("vid_neg", "Not Relevant", 0.0),
    }
