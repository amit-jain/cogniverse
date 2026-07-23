"""Two-pass per-segment KG extraction preserves the serial coreference order.

The extraction parallelises pass 1 (entities) and pass 2 (claims). The
coreference prior pool each segment's claim pass sees MUST be exactly the entity
names from the EARLIER segments (0..N-1) — the same feed-forward the serial loop
provided, so a pronoun still binds to a name introduced in an earlier segment.
This drives the real ``_extract_graph_per_segment_inner`` and pins both that
reconstruction and the segment-ordered accumulation of nodes and back-refs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_runtime.routers import ingestion

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _record(i: int):
    return SimpleNamespace(
        text=f"segment {i} text",
        segment_anchor=SimpleNamespace(
            segment_id=f"s{i}",
            modality="document",
            ts_start=float(i),
            ts_end=float(i) + 1.0,
            source_doc_id="doc1",
        ),
    )


@pytest.mark.asyncio
async def test_claim_pass_prior_pool_is_earlier_segments_entities(monkeypatch):
    records = [_record(i) for i in range(3)]
    monkeypatch.setattr(
        ingestion, "_iter_segments_for_graph", lambda pr, sd: iter(records)
    )
    monkeypatch.setattr(ingestion, "_lookup_artifact_manager", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_resolve_tenant_llm_config", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_lookup_face_embed_endpoint", lambda cm: None)

    async def _no_backrefs(**kwargs):
        return None

    monkeypatch.setattr(ingestion, "_write_backrefs_to_content", _no_backrefs)

    recorded_priors: dict[str, list[str]] = {}

    class StubClaim:
        def __init__(self, **kwargs):
            pass

    class StubDoc:
        def __init__(self, **kwargs):
            pass

        def extract_entities_from_text(
            self, *, text, tenant_id, source_doc_id, segment_anchor
        ):
            name = f"Ent_{segment_anchor.segment_id}"
            node = SimpleNamespace(name=name, node_id=name.lower())
            return SimpleNamespace(nodes=[node], per_chunk_entity_names=[[name]])

        def extract_claims_from_text(
            self,
            *,
            text,
            segment_entities,
            prior_entities,
            tenant_id,
            source_doc_id,
            segment_anchor,
        ):
            recorded_priors[segment_anchor.segment_id] = list(prior_entities)
            return []

    class StubLinker:
        def link(self, combined):
            return combined

    class StubResult:
        def __init__(self, source_doc_id="", nodes=(), edges=(), file_sha256=None):
            self.source_doc_id = source_doc_id
            self.nodes = list(nodes)
            self.edges = list(edges)
            self.file_sha256 = file_sha256

    mgr = SimpleNamespace(
        upsert=lambda linked: {
            "nodes_upserted": len(linked.nodes),
            "edges_upserted": 0,
            "failed_ids": [],
        },
        _backend=SimpleNamespace(),
    )
    graph_router = SimpleNamespace(_graph_manager_factory=lambda t: mgr)

    result = await ingestion._extract_graph_per_segment_inner(
        processing_results={},
        source_doc_id="doc1",
        tenant_id="acme:acme",
        config_manager=SimpleNamespace(),
        DocExtractor=StubDoc,
        ClaimExtractor=StubClaim,
        CrossModalLinker=StubLinker,
        ExtractionResult=StubResult,
        graph_router=graph_router,
    )

    # The coreference prior each segment's claim pass saw = the earlier segments'
    # entity names, in order — identical to the serial entity_pool feed-forward.
    assert recorded_priors["s0"] == []
    assert recorded_priors["s1"] == ["Ent_s0"]
    assert recorded_priors["s2"] == ["Ent_s0", "Ent_s1"]

    # All three segments' nodes were accumulated and upserted.
    assert result["nodes_upserted"] == 3

    # Back-refs recorded per segment with each segment's own entity id.
    br = result["backrefs_by_segment"]
    assert br["s0"]["entity_ids"] == ["ent_s0"]
    assert br["s1"]["entity_ids"] == ["ent_s1"]
    assert br["s2"]["entity_ids"] == ["ent_s2"]
