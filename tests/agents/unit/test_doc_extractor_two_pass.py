"""DocExtractor's two-pass split preserves the exact single-pass extraction.

The per-segment KG extraction is parallelised by running entity extraction for
every segment first (no coreference dependency) and claim extraction second with
a precomputed prior-entity pool. That split lives in DocExtractor as
``extract_entities_from_text`` (Pass 1) + ``extract_claims_from_text`` (Pass 2).
These pin that composing the two passes is byte-identical to the monolithic
``extract_from_text`` — same nodes, same edges, and the ClaimExtractor receives
the SAME per-chunk merged hints (chunk entities + prior, deduped case-insensitively
in order).
"""

from __future__ import annotations

from typing import List

import pytest

from cogniverse_agents.graph.doc_extractor import DocExtractor
from cogniverse_agents.graph.graph_schema import Edge, Mention

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubGliner:
    """Deterministic GLiNER: 'Alpha' in the chunk -> Alpha entity, 'Beta' ->
    Beta entity. Lets a two-chunk input carry distinct per-chunk entities."""

    def predict_entities(self, chunk: str, labels, threshold: float):
        out = []
        if "Alpha" in chunk:
            out.append({"text": "Alpha", "label": "Concept", "score": 0.9})
        if "Beta" in chunk:
            out.append({"text": "Beta", "label": "Concept", "score": 0.9})
        return out


class _RecordingClaims:
    """Records the hints each chunk's claim extraction is handed and returns one
    deterministic edge per call."""

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def extract(
        self,
        *,
        text: str,
        entity_hints: List[str],
        modality_hint: str,
        segment_anchor: Mention,
        tenant_id: str,
        source_doc_id: str,
    ) -> List[Edge]:
        self.calls.append({"text": text, "hints": list(entity_hints)})
        return [
            Edge(
                tenant_id=tenant_id,
                source=entity_hints[0],
                target="Target",
                relation="rel",
                evidence_span="e",
                segment_id=segment_anchor.segment_id,
                ts_start=segment_anchor.ts_start,
                ts_end=segment_anchor.ts_end,
                modality=segment_anchor.modality,
                source_doc_id=source_doc_id,
            )
        ]


def _anchor() -> Mention:
    return Mention(
        source_doc_id="doc1",
        segment_id="seg1",
        ts_start=0.0,
        ts_end=1.0,
        modality="document",
        evidence_span="stub",
    )


def _two_chunk_text() -> str:
    # Two paragraphs each ~1200 chars; combined > _MAX_CHARS_PER_CHUNK (2000),
    # so _chunk_text yields exactly two chunks, one carrying 'Alpha', one 'Beta'.
    para1 = "Alpha " + ("filler " * 200)
    para2 = "Beta " + ("padding " * 200)
    return f"{para1}\n\n{para2}"


def _extractor(claims) -> DocExtractor:
    ext = DocExtractor(claim_extractor=claims)
    ext._gliner = _StubGliner()  # inject; skip the sidecar/local load
    return ext


def test_entity_pass_yields_per_chunk_entities():
    ext = _extractor(_RecordingClaims())
    ents = ext.extract_entities_from_text(_two_chunk_text(), "t:t", "doc1", _anchor())

    assert [n.name for n in ents.nodes] == ["Alpha", "Beta"]
    assert [n.label for n in ents.nodes] == ["Concept", "Concept"]
    # One entry per chunk, each the chunk's raw entity names.
    assert ents.per_chunk_entity_names == [["Alpha"], ["Beta"]]


def test_claim_pass_merges_chunk_hints_with_prior_deduped_in_order():
    claims = _RecordingClaims()
    ext = _extractor(claims)
    text = _two_chunk_text()
    ents = ext.extract_entities_from_text(text, "t:t", "doc1", _anchor())

    edges = ext.extract_claims_from_text(
        text,
        ents,
        prior_entities=["Gamma", "alpha"],  # 'alpha' collides with chunk-1 'Alpha'
        tenant_id="t:t",
        source_doc_id="doc1",
        segment_anchor=_anchor(),
    )

    # One claim call per chunk, hints = chunk entities + prior, deduped
    # case-insensitively in order.
    assert len(claims.calls) == 2
    assert claims.calls[0]["hints"] == ["Alpha", "Gamma"]  # 'alpha' dropped as dup
    assert claims.calls[1]["hints"] == ["Beta", "Gamma", "alpha"]
    assert len(edges) == 2


def test_two_pass_is_identical_to_monolithic_extract_from_text():
    text = _two_chunk_text()
    prior = ["Gamma", "alpha"]

    # Monolithic path.
    mono_claims = _RecordingClaims()
    mono = _extractor(mono_claims)
    mono_result = mono.extract_from_text(
        text, "t:t", "doc1", _anchor(), prior_entities=prior
    )

    # Two-pass path.
    tp_claims = _RecordingClaims()
    tp = _extractor(tp_claims)
    ents = tp.extract_entities_from_text(text, "t:t", "doc1", _anchor())
    tp_edges = tp.extract_claims_from_text(text, ents, prior, "t:t", "doc1", _anchor())

    # Same nodes.
    assert [n.node_id for n in mono_result.nodes] == [n.node_id for n in ents.nodes]
    # Same edges.
    assert [e.edge_id for e in mono_result.edges] == [e.edge_id for e in tp_edges]
    # And the ClaimExtractor saw identical per-chunk hints on both paths.
    assert [c["hints"] for c in mono_claims.calls] == [
        c["hints"] for c in tp_claims.calls
    ]


def test_no_claim_extractor_yields_no_edges_but_still_entities():
    ext = DocExtractor()  # no claim extractor
    ext._gliner = _StubGliner()
    ents = ext.extract_entities_from_text(_two_chunk_text(), "t:t", "doc1", _anchor())
    edges = ext.extract_claims_from_text(
        _two_chunk_text(), ents, None, "t:t", "doc1", _anchor()
    )
    assert [n.name for n in ents.nodes] == ["Alpha", "Beta"]
    assert edges == []


class _OutageGliner:
    """Every predict_entities call raises — a sidecar outage."""

    def predict_entities(self, chunk, labels, threshold):
        raise ConnectionError("gliner sidecar down")


class _PartialGliner:
    """Raises on chunks containing 'Beta', succeeds on 'Alpha' — a transient
    per-chunk failure, not a total outage."""

    def predict_entities(self, chunk, labels, threshold):
        if "Beta" in chunk:
            raise ConnectionError("transient gliner error")
        return [{"text": "Alpha", "label": "Concept", "score": 0.9}]


def test_total_gliner_outage_raises_not_regex_noise():
    """A configured GLiNER failing on EVERY chunk is a sidecar outage — extract
    must raise, not silently return a regex-only knowledge graph reported as a
    successful ingest."""
    ext = DocExtractor()
    ext._gliner = _OutageGliner()
    with pytest.raises(RuntimeError, match="failed on all"):
        ext.extract_entities_from_text(_two_chunk_text(), "t:t", "doc1", _anchor())


def test_partial_gliner_failure_is_best_effort():
    """A GLiNER failure on SOME chunks falls back to regex for those and keeps
    the GLiNER entities for the rest — no raise (not a total outage)."""
    ext = DocExtractor()
    ext._gliner = _PartialGliner()
    ents = ext.extract_entities_from_text(_two_chunk_text(), "t:t", "doc1", _anchor())
    names = {n.name for n in ents.nodes}
    assert "Alpha" in names  # GLiNER succeeded on the Alpha chunk and was kept
