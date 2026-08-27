"""Two-pass per-segment KG extraction preserves the serial coreference order.

The extraction parallelises pass 1 (entities) and pass 2 (claims). The
coreference prior pool each segment's claim pass sees MUST be exactly the entity
names from the EARLIER segments (0..N-1) — the same feed-forward the serial loop
provided, so a pronoun still binds to a name introduced in an earlier segment.
This drives the real ``_extract_graph_per_segment_inner`` and pins both that
reconstruction and the segment-ordered accumulation of nodes and back-refs.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from cogniverse_agents.graph.claim_extractor import ClaimExtractor
from cogniverse_agents.graph.doc_extractor import ClaimExtractionResult, DocExtractor
from cogniverse_agents.graph.graph_schema import (
    CLAIM_SEGMENT_MODALITIES,
    OCR_MODALITY,
    TRANSCRIPT_MODALITY,
    VLM_MODALITY,
    Edge,
    ExtractionResult,
    normalize_name,
)
from cogniverse_runtime.routers import ingestion

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def processing_results():
    return {
        "transcript": {
            "segments": [
                {
                    "start": 0.1,
                    "end": 0.9,
                    "text": "T0X transcript claim 0",
                },
                {
                    "start": 10.1,
                    "end": 10.9,
                    "text": "T1X transcript claim 1",
                },
                {
                    "start": 20.1,
                    "end": 20.9,
                    "text": "T2X transcript claim 2",
                },
            ]
        },
        "descriptions": {
            "descriptions": {
                "10": "V10X frame description 10",
                "11": "V11X frame description 11",
                "12": "V12X frame description 12",
                "13": "V13X frame description 13",
                "14": "V14X frame description 14",
            }
        },
        "keyframes": {
            "keyframes": [
                {
                    "frame_id": 0,
                    "timestamp": 0.0,
                    "ocr_text": "O0X OCR caption 0",
                },
                {
                    "frame_id": 1,
                    "timestamp": 10.0,
                    "caption": "O1X OCR caption 1",
                },
                {
                    "frame_id": 2,
                    "timestamp": 20.0,
                },
            ]
        },
    }


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
async def test_claims_skip_non_transcript_segments_and_keep_entities(
    processing_results, monkeypatch
):
    records = list(ingestion._iter_segments_for_graph(processing_results, "doc1"))
    assert [r.segment_anchor.modality for r in records] == [
        TRANSCRIPT_MODALITY,
        TRANSCRIPT_MODALITY,
        TRANSCRIPT_MODALITY,
        VLM_MODALITY,
        VLM_MODALITY,
        VLM_MODALITY,
        VLM_MODALITY,
        VLM_MODALITY,
        OCR_MODALITY,
        OCR_MODALITY,
    ]

    entity_calls: list[tuple[str, str, str]] = []
    claim_calls: list[tuple[str, str, str]] = []

    class _RecordingGliner:
        def predict_entities(self, chunk, labels, threshold):
            return [
                {
                    "text": chunk.split()[0],
                    "label": "Concept",
                    "score": 0.99,
                }
            ]

    class _ClaimsModule:
        def __call__(self, **kwargs):
            subject = kwargs["entity_hints"][0]
            return dspy.Prediction(
                claims=[
                    {
                        "subject": subject,
                        "predicate": "won",
                        "object": f"{subject}_claim",
                        "evidence_span": kwargs["text_segment"],
                        "confidence": 0.9,
                    }
                ]
            )

    class RecordingClaimExtractor(ClaimExtractor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._cot_module = _ClaimsModule()

    class RecordingDocExtractor(DocExtractor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._gliner = _RecordingGliner()

        def extract_entities_from_text(
            self, *, text, tenant_id, source_doc_id, segment_anchor
        ):
            entity_calls.append(
                (segment_anchor.segment_id, segment_anchor.modality, text)
            )
            return super().extract_entities_from_text(
                text=text,
                tenant_id=tenant_id,
                source_doc_id=source_doc_id,
                segment_anchor=segment_anchor,
            )

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
            claim_calls.append(
                (segment_anchor.segment_id, segment_anchor.modality, text)
            )
            return super().extract_claims_from_text(
                text=text,
                segment_entities=segment_entities,
                prior_entities=prior_entities,
                tenant_id=tenant_id,
                source_doc_id=source_doc_id,
                segment_anchor=segment_anchor,
            )

    class StubLinker:
        def link(self, combined):
            return combined

    entity_order = {
        record.segment_anchor.segment_id: idx for idx, record in enumerate(records)
    }
    expected_entity_calls = [
        (record.segment_anchor.segment_id, record.segment_anchor.modality, record.text)
        for record in records
    ]
    expected_claim_calls = [
        (record.segment_anchor.segment_id, record.segment_anchor.modality, record.text)
        for record in records
        if record.segment_anchor.modality in CLAIM_SEGMENT_MODALITIES
    ]
    expected_backrefs = {}
    for record in records:
        source_token = record.text.split()[0]
        expected_backrefs[record.segment_anchor.segment_id] = {
            "entity_ids": [normalize_name(source_token)],
            "relation_ids": [],
            "claim_ids": [],
        }
        if record.segment_anchor.modality not in CLAIM_SEGMENT_MODALITIES:
            continue
        edge = Edge(
            tenant_id="acme:acme",
            source=source_token,
            target=f"{source_token}_claim",
            relation="won",
            evidence_span=record.text,
            segment_id=record.segment_anchor.segment_id,
            ts_start=record.segment_anchor.ts_start,
            ts_end=record.segment_anchor.ts_end,
            modality=record.segment_anchor.modality,
            source_doc_id="doc1",
            confidence=0.9,
        )
        expected_backrefs[record.segment_anchor.segment_id]["relation_ids"] = [
            edge.edge_id
        ]
        expected_backrefs[record.segment_anchor.segment_id]["claim_ids"] = [
            edge.edge_id
        ]

    async def _no_backrefs(**kwargs):
        return None

    monkeypatch.setattr(ingestion, "_lookup_artifact_manager", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_resolve_tenant_llm_config", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_lookup_face_embed_endpoint", lambda cm: None)
    monkeypatch.setattr(ingestion, "_write_backrefs_to_content", _no_backrefs)

    mgr = SimpleNamespace(
        upsert=lambda linked: {
            "nodes_upserted": len(linked.nodes),
            "edges_upserted": len(linked.edges),
            "failed_ids": [],
        },
        _backend=SimpleNamespace(),
    )
    graph_router = SimpleNamespace(_graph_manager_factory=lambda t: mgr)

    result = await ingestion._extract_graph_per_segment_inner(
        processing_results=processing_results,
        source_doc_id="doc1",
        tenant_id="acme:acme",
        config_manager=SimpleNamespace(),
        DocExtractor=RecordingDocExtractor,
        ClaimExtractor=RecordingClaimExtractor,
        CrossModalLinker=StubLinker,
        ExtractionResult=ExtractionResult,
        graph_router=graph_router,
    )

    assert set(result) == {
        "nodes_upserted",
        "edges_upserted",
        "graph_failed",
        "backrefs_by_segment",
        "claim_segments_skipped_by_modality",
        "claim_segments_failed",
    }
    assert result["claim_segments_failed"] == 0
    assert result["nodes_upserted"] == 10
    assert result["edges_upserted"] == 3
    assert result["graph_failed"] == 0
    assert result["claim_segments_skipped_by_modality"] == 7
    assert (
        sorted(entity_calls, key=lambda call: entity_order[call[0]])
        == expected_entity_calls
    )
    assert sorted(claim_calls) == expected_claim_calls
    assert result["backrefs_by_segment"] == expected_backrefs


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
            return ClaimExtractionResult()

    class StubLinker:
        def link(self, combined):
            return combined

    StubResult = ExtractionResult

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


@pytest.mark.asyncio
async def test_claim_failure_count_surfaces_on_pipeline_result(monkeypatch):
    records = [_record(i) for i in range(2)]
    monkeypatch.setattr(
        ingestion, "_iter_segments_for_graph", lambda pr, sd: iter(records)
    )
    monkeypatch.setattr(ingestion, "_lookup_artifact_manager", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_resolve_tenant_llm_config", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_lookup_face_embed_endpoint", lambda cm: None)

    async def _no_backrefs(**kwargs):
        return None

    monkeypatch.setattr(ingestion, "_write_backrefs_to_content", _no_backrefs)

    claim_calls: list[str] = []

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
            del text, segment_entities, prior_entities, tenant_id, source_doc_id
            claim_calls.append(segment_anchor.segment_id)
            if segment_anchor.segment_id == "s0":
                return ClaimExtractionResult([], claim_segments_failed=1)
            return ClaimExtractionResult(
                [
                    Edge(
                        tenant_id="acme:acme",
                        source="Ent_s1",
                        target="Target",
                        relation="rel",
                        evidence_span="e",
                        segment_id=segment_anchor.segment_id,
                        ts_start=segment_anchor.ts_start,
                        ts_end=segment_anchor.ts_end,
                        modality=segment_anchor.modality,
                        source_doc_id=segment_anchor.source_doc_id,
                    )
                ],
                claim_segments_failed=0,
            )

    class StubClaim:
        def __init__(self, **kwargs):
            pass

    class StubLinker:
        def link(self, combined):
            return combined

    StubResult = ExtractionResult

    mgr = MagicMock()
    mgr.upsert.side_effect = lambda linked: {
        "nodes_upserted": len(linked.nodes),
        "edges_upserted": len(linked.edges),
        "failed_ids": [],
    }
    mgr._backend = SimpleNamespace()
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

    assert claim_calls == ["s0", "s1"]
    assert result["claim_segments_failed"] == 1
    assert result["nodes_upserted"] == 2
    assert result["edges_upserted"] == 1


@pytest.mark.asyncio
async def test_entity_pass_failure_settles_siblings(monkeypatch):
    """A segment's entity-extraction failure (e.g. a total GLiNER outage) must
    settle the sibling segments before propagating — a bare gather raises the
    first failure while sibling to_thread KG calls keep running detached in the
    shared executor, throttling unrelated offloads under a hung sidecar."""
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

    sibling_finished = threading.Event()

    class StubDoc:
        def __init__(self, **kwargs):
            pass

        def extract_entities_from_text(
            self, *, text, tenant_id, source_doc_id, segment_anchor
        ):
            if segment_anchor.segment_id == "s0":
                time.sleep(0.02)
                raise RuntimeError("total GLiNER outage")
            if segment_anchor.segment_id == "s2":
                time.sleep(0.2)  # outlives s0's fast failure
                sibling_finished.set()
            name = f"Ent_{segment_anchor.segment_id}"
            node = SimpleNamespace(name=name, node_id=name.lower())
            return SimpleNamespace(nodes=[node], per_chunk_entity_names=[[name]])

        def extract_claims_from_text(self, **kwargs):
            return ClaimExtractionResult()

    class StubClaim:
        def __init__(self, **kwargs):
            pass

    class StubLinker:
        def link(self, combined):
            return combined

    StubResult = ExtractionResult

    mgr = SimpleNamespace(
        upsert=lambda linked: {
            "nodes_upserted": 0,
            "edges_upserted": 0,
            "failed_ids": [],
        },
        _backend=SimpleNamespace(),
    )
    graph_router = SimpleNamespace(_graph_manager_factory=lambda t: mgr)

    with pytest.raises(RuntimeError, match="total GLiNER outage"):
        await ingestion._extract_graph_per_segment_inner(
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

    # If the failure orphaned the siblings, s2's 0.2s thread would still be
    # running here (s0 failed at ~0.02s). The event being set proves the
    # orchestration settled every segment before it raised.
    assert sibling_finished.is_set()


@pytest.mark.asyncio
async def test_all_claim_segments_failed_raise(monkeypatch):
    records = [_record(i) for i in range(2)]
    monkeypatch.setattr(
        ingestion, "_iter_segments_for_graph", lambda pr, sd: iter(records)
    )
    monkeypatch.setattr(ingestion, "_lookup_artifact_manager", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_resolve_tenant_llm_config", lambda t, cm: None)
    monkeypatch.setattr(ingestion, "_lookup_face_embed_endpoint", lambda cm: None)

    async def _no_backrefs(**kwargs):
        return None

    monkeypatch.setattr(ingestion, "_write_backrefs_to_content", _no_backrefs)

    claim_calls: list[str] = []

    class StubDoc:
        def __init__(self, **kwargs):
            pass

        def extract_entities_from_text(
            self, *, text, tenant_id, source_doc_id, segment_anchor
        ):
            name = f"Ent_{segment_anchor.segment_id}"
            node = SimpleNamespace(name=name, node_id=name.lower())
            return SimpleNamespace(nodes=[node], per_chunk_entity_names=[[name]])

        def extract_claims_from_text(self, **kwargs):
            claim_calls.append(kwargs["segment_anchor"].segment_id)
            return ClaimExtractionResult([], claim_segments_failed=1)

    class StubClaim:
        def __init__(self, **kwargs):
            pass

    class StubLinker:
        def link(self, combined):
            return combined

    StubResult = ExtractionResult

    mgr = MagicMock()
    mgr.upsert.return_value = {
        "nodes_upserted": 0,
        "edges_upserted": 0,
        "failed_ids": [],
    }
    mgr._backend = SimpleNamespace()
    graph_router = SimpleNamespace(_graph_manager_factory=lambda t: mgr)

    with pytest.raises(
        RuntimeError,
        match=r"^claim extraction failed for source 'doc1' across 2 segments$",
    ):
        await ingestion._extract_graph_per_segment_inner(
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

    assert claim_calls == ["s0", "s1"]
    mgr.upsert.assert_not_called()
