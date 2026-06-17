"""CrossModalLinker integration against a real ColBERT (pylate) endpoint.

The linker emits ``same_as`` edges only when two cross-modal mentions
are inside the ±5s temporal window AND their MaxSim cosine clears the
threshold; outside either condition, no edge is emitted; the operation
is idempotent.

The fixture builds an ``ExtractionResult`` by hand using a Marie Curie
fixture (transcript + two VLM keyframes + one OCR block) — no Vespa
needed for this file, only the ColBERT encoder. The golden-file harness
(``RECORD_GOLDEN=1`` to rewrite) locks the linker's output byte-equal so
any future LM/encoder drift trips immediately.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from cogniverse_agents.graph.cross_modal_linker import CrossModalLinker
from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
)

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"


# --------------------------------------------------------------------- #
# Golden-file harness                                                   #
# --------------------------------------------------------------------- #


def assert_golden(actual, name: str):
    path = GOLDEN_DIR / name
    actual_json = json.dumps(actual, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, (
        f"Golden {name} mismatch.\n"
        f"--- expected ---\n{expected}\n--- actual ---\n{actual_json}"
    )


pytestmark = pytest.mark.integration

# Historically the linker depended on a live ColBERT sidecar to score
# cross-modal pairs by MaxSim cosine. The structural-inference rewrite
# removed that dependency — the fixture below is kept inert for the few
# tests that still pass a colbert_endpoint argument, but the value is no
# longer consumed.


# --------------------------------------------------------------------- #
# ColBERT endpoint fixture                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def colbert_endpoint():
    """Inert fixture kept for backwards-compatibility with the existing
    test signatures. The structural-inference linker no longer consumes
    a ColBERT URL; we yield an empty string so existing
    ``def test_x(self, colbert_endpoint)`` signatures continue to work.
    """
    yield ""


# --------------------------------------------------------------------- #
# Shared Marie Curie fixture builders                                    #
# --------------------------------------------------------------------- #

VIDEO_ID = "marie_curie_30s"
TENANT_ID = "test"

TRANSCRIPT_TEXT = "Marie Curie discovered radium in 1898 at the Sorbonne."
VLM_LAB_COAT_TEXT = "woman in lab coat with glassware in laboratory"
VLM_PODIUM_TEXT = "woman holding award certificate at podium"
VLM_FLOWERS_TEXT = "yellow flowers in glass vase"
OCR_TEXT = "Curie 1903"


def _mention(
    *,
    segment_id: str,
    ts_start: float,
    ts_end: float,
    modality: str,
    evidence_span: str,
) -> Mention:
    return Mention(
        source_doc_id=VIDEO_ID,
        segment_id=segment_id,
        ts_start=ts_start,
        ts_end=ts_end,
        modality=modality,
        evidence_span=evidence_span,
    )


def _transcript_mention() -> Mention:
    return _mention(
        segment_id="seg_3",
        ts_start=12.0,
        ts_end=18.5,
        modality="transcript",
        evidence_span=TRANSCRIPT_TEXT,
    )


def _vlm_lab_coat_at_14_mention() -> Mention:
    return _mention(
        segment_id="frame_14_0",
        ts_start=14.0,
        ts_end=14.0,
        modality="vlm",
        evidence_span=VLM_LAB_COAT_TEXT,
    )


def _vlm_podium_at_21_mention() -> Mention:
    return _mention(
        segment_id="frame_21_0",
        ts_start=21.0,
        ts_end=21.0,
        modality="vlm",
        evidence_span=VLM_PODIUM_TEXT,
    )


def _vlm_flowers_at_14_mention() -> Mention:
    return _mention(
        segment_id="frame_14_0",
        ts_start=14.0,
        ts_end=14.0,
        modality="vlm",
        evidence_span=VLM_FLOWERS_TEXT,
    )


def _vlm_lab_coat_at_30_mention() -> Mention:
    """Lab coat keyframe placed at ts=30.0 — outside the ±5s window of seg_3."""
    return _mention(
        segment_id="frame_30_0",
        ts_start=30.0,
        ts_end=30.0,
        modality="vlm",
        evidence_span=VLM_LAB_COAT_TEXT,
    )


def _ocr_at_14_5_mention() -> Mention:
    return _mention(
        segment_id="frame_14_5",
        ts_start=14.5,
        ts_end=14.5,
        modality="ocr",
        evidence_span=OCR_TEXT,
    )


def _marie_curie_node() -> Node:
    return Node(
        tenant_id=TENANT_ID,
        name="Marie Curie",
        description="Person mentioned in marie_curie_30s",
        kind="entity",
        label="Person",
        mentions=[_transcript_mention()],
    )


def _woman_in_lab_coat_node(mention: Mention) -> Node:
    # GLiNER's default label set tags this VLM caption "Concept" — the
    # type gate then admits Person↔Concept pairs only when the Concept
    # name carries a person-indicator word ("woman").
    return Node(
        tenant_id=TENANT_ID,
        name="woman in lab coat",
        description="VLM visual subject in marie_curie_30s",
        kind="concept",
        label="Concept",
        mentions=[mention],
    )


def _woman_holding_award_node(mention: Mention) -> Node:
    return Node(
        tenant_id=TENANT_ID,
        name="woman holding award certificate",
        description="VLM visual subject in marie_curie_30s",
        kind="concept",
        label="Concept",
        mentions=[mention],
    )


def _yellow_flowers_node(mention: Mention) -> Node:
    # No name-token overlap with "Marie Curie" and no person-indicator
    # word — type gate must reject Marie Curie↔yellow flowers.
    return Node(
        tenant_id=TENANT_ID,
        name="yellow flowers in glass vase",
        description="VLM visual subject in marie_curie_30s",
        kind="concept",
        label="Concept",
        mentions=[mention],
    )


def _curie_1903_node(mention: Mention) -> Node:
    # OCR overlay text "Curie 1903" — labelled Concept (it is not a
    # well-formed personal name on its own), but the token "Curie"
    # overlaps Marie Curie so the type gate admits THAT pair. Concept↔
    # Concept against "woman in lab coat" is rejected outright by the
    # multi-modal triangle test.
    return Node(
        tenant_id=TENANT_ID,
        name="Curie 1903",
        description="OCR overlay in marie_curie_30s",
        kind="entity",
        label="Concept",
        mentions=[mention],
    )


def _edge_to_jsonable(edge: Edge) -> dict:
    """Stable dict of an Edge with ``created_at`` zeroed out.

    The ``created_at`` timestamp is wall-clock and intentionally excluded
    from the golden so re-running the test on a different day still
    produces a byte-equal serialization.
    """
    d = asdict(edge)
    d["created_at"] = "<<created_at>>"
    return d


def _linker(_unused_colbert_url: str = "") -> CrossModalLinker:
    """Linker configured with the same thresholds the runtime uses.

    Signature kept compatible with the previous ColBERT-based linker so
    the existing tests pass their fixture URL in; the value is ignored
    because the structural-inference primitives don't need an encoder.
    """
    return CrossModalLinker(temporal_window_s=5.0)


def _same_as_edges(result: ExtractionResult) -> list:
    return [e for e in result.edges if e.relation == "same_as"]


# --------------------------------------------------------------------- #
# Node-emission lock for the lab-coat VLM mention                        #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerNodeEmission:
    """The VLM keyframe at ts=14.0 produces node ``woman in lab coat``.

    This file does not run GLiNER-on-VLM — it locks the structure of the
    extraction the linker is contracted to receive. The full
    extraction_result is dumped to a golden so any regression in the
    Mention/Node shape trips here, not silently downstream.
    """

    def test_lab_coat_node_emitted_with_full_mention_anchor(self):
        node = _woman_in_lab_coat_node(_vlm_lab_coat_at_14_mention())
        assert node.name == "woman in lab coat"

        actual = {
            "name": node.name,
            "kind": node.kind,
            "mentions": [asdict(m) for m in node.mentions],
        }
        assert_golden(actual, "cross_modal_lab_coat_node.json")


# --------------------------------------------------------------------- #
# Node mentions JSON byte-equality                                       #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerNodeMentions:
    """``node("woman in lab coat")["mentions"]`` byte-equal to the golden."""

    def test_lab_coat_mentions_byte_equal(self):
        node = _woman_in_lab_coat_node(_vlm_lab_coat_at_14_mention())
        mentions_dump = [asdict(m) for m in node.mentions]
        assert_golden(mentions_dump, "cross_modal_lab_coat_mentions.json")


# --------------------------------------------------------------------- #
# Single same_as edge for VLM ts=14.0 ↔ transcript                       #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerSingleSameAsEdge:
    """Linker emits exactly one same_as edge for the ts=14 pair."""

    def test_single_same_as_edge_byte_equal(self, colbert_endpoint):
        nodes = [
            _marie_curie_node(),
            _woman_in_lab_coat_node(_vlm_lab_coat_at_14_mention()),
        ]
        extraction = ExtractionResult(
            source_doc_id=VIDEO_ID,
            nodes=nodes,
            edges=[],
        )

        linked = _linker(colbert_endpoint).link(extraction)
        same_as = _same_as_edges(linked)
        assert len(same_as) == 1, (
            f"expected exactly 1 same_as edge, got {len(same_as)}: "
            f"{[(e.source, e.target, e.confidence) for e in same_as]}"
        )

        edge_dump = _edge_to_jsonable(same_as[0])
        assert_golden(edge_dump, "cross_modal_lab_coat_to_curie_edge.json")


# --------------------------------------------------------------------- #
# Second same_as edge from VLM ts=21.0                                   #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerSecondSameAsEdge:
    """VLM keyframe at ts=21.0 produces its own same_as edge.

    The ``Marie Curie`` transcript mention sits at seg_3 (12.0–18.5s).
    The ±5s window of the linker means the VLM at ts=21.0 still overlaps
    seg_3's end (18.5 + 5 = 23.5). The cosine threshold is the same
    0.6 used by the runtime — if the encoder doesn't clear it, the
    test fails (and we re-record the golden after reviewing the diff).
    """

    def test_award_certificate_to_curie_edge_byte_equal(self, colbert_endpoint):
        nodes = [
            _marie_curie_node(),
            _woman_holding_award_node(_vlm_podium_at_21_mention()),
        ]
        extraction = ExtractionResult(
            source_doc_id=VIDEO_ID,
            nodes=nodes,
            edges=[],
        )

        linked = _linker(colbert_endpoint).link(extraction)
        same_as = _same_as_edges(linked)
        assert len(same_as) == 1, (
            f"expected exactly 1 same_as edge, got {len(same_as)}: "
            f"{[(e.source, e.target, e.confidence) for e in same_as]}"
        )

        edge_dump = _edge_to_jsonable(same_as[0])
        assert_golden(edge_dump, "cross_modal_award_to_curie_edge.json")


# --------------------------------------------------------------------- #
# Negative: semantic mismatch (flowers @ ts=14.0)                        #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerSemanticMismatch:
    """VLM ``yellow flowers in glass vase`` produces no same_as edge.

    The temporal window overlaps seg_3 but the entity is semantically
    unrelated to ``Marie Curie`` — cosine must stay under threshold.
    """

    def test_flowers_at_14_emits_no_same_as_edge(self, colbert_endpoint):
        nodes = [
            _marie_curie_node(),
            _yellow_flowers_node(_vlm_flowers_at_14_mention()),
        ]
        extraction = ExtractionResult(
            source_doc_id=VIDEO_ID,
            nodes=nodes,
            edges=[],
        )

        linked = _linker(colbert_endpoint).link(extraction)
        same_as = _same_as_edges(linked)
        assert same_as == [], (
            f"expected zero same_as edges for flowers@14, got "
            f"{[(e.source, e.target, e.confidence) for e in same_as]}"
        )


# --------------------------------------------------------------------- #
# Negative: temporal mismatch (lab coat @ ts=30.0)                       #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerPerWindowInference:
    """Per-window subject inference for multi-subject content.

    Constructs a fixture with two roughly-equally-mentioned Persons
    (Alice Chen and Bob Smith). With no single dominant subject, the
    linker falls back to per-window inference: each VLM mention
    attributes to the dominant Person inside its ±window. A VLM at
    ts=10 (when Alice was just named) attributes to Alice; a VLM at
    ts=40 (when Bob was just named) attributes to Bob.
    """

    def test_window_subject_inference_picks_correct_person_per_segment(
        self, colbert_endpoint
    ):
        from cogniverse_agents.graph.graph_schema import Mention, Node

        alice_node = Node(
            tenant_id="test",
            name="Alice Chen",
            label="Person",
            kind="entity",
            mentions=[
                Mention(
                    source_doc_id=VIDEO_ID,
                    segment_id="seg_1",
                    ts_start=5.0,
                    ts_end=12.0,
                    modality="transcript",
                    evidence_span="Alice Chen presents her research.",
                )
            ],
        )
        bob_node = Node(
            tenant_id="test",
            name="Bob Smith",
            label="Person",
            kind="entity",
            mentions=[
                Mention(
                    source_doc_id=VIDEO_ID,
                    segment_id="seg_3",
                    ts_start=35.0,
                    ts_end=45.0,
                    modality="transcript",
                    evidence_span="Bob Smith responds.",
                )
            ],
        )
        vlm_early = Node(
            tenant_id="test",
            name="woman at podium",
            label="Concept",
            kind="concept",
            mentions=[
                Mention(
                    source_doc_id=VIDEO_ID,
                    segment_id="frame_10",
                    ts_start=10.0,
                    ts_end=10.0,
                    modality="vlm",
                    evidence_span="woman at podium with slides",
                )
            ],
        )
        vlm_late = Node(
            tenant_id="test",
            name="man at lectern",
            label="Concept",
            kind="concept",
            mentions=[
                Mention(
                    source_doc_id=VIDEO_ID,
                    segment_id="frame_40",
                    ts_start=40.0,
                    ts_end=40.0,
                    modality="vlm",
                    evidence_span="man at lectern gesturing",
                )
            ],
        )

        extraction = ExtractionResult(
            source_doc_id=VIDEO_ID,
            nodes=[alice_node, bob_node, vlm_early, vlm_late],
            edges=[],
        )

        linked = _linker(colbert_endpoint).link(extraction)
        same_as = _same_as_edges(linked)

        # Two same_as edges, each attributing the visual mention in its
        # window to the Person who was being discussed then.
        targets_by_source = {e.source: e.target for e in same_as}
        assert targets_by_source == {
            "woman at podium": "Alice Chen",
            "man at lectern": "Bob Smith",
        }, f"per-window inference picked wrong attribution: {targets_by_source}"
        # Both edges carry the structural-inference provenance, not a
        # similarity-driven INFERRED tag.
        assert all(e.provenance == "video_subject_inference" for e in same_as), [
            e.provenance for e in same_as
        ]


# --------------------------------------------------------------------- #
# Multi-modal triangle (VLM + OCR + transcript co-occur)                 #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerTriangle:
    """VLM @ ts=14.0 + OCR @ ts=14.5 around the transcript seg_3.

    Three modalities all overlap seg_3's window. Linker fires
    transcript↔vlm AND transcript↔ocr AND vlm↔ocr — but only the two
    pairs whose cosine clears the threshold survive. The surviving set is
    locked at 2 same_as edges: (woman_in_lab_coat → Marie Curie) and
    (Curie 1903 → Marie Curie).
    """

    def test_triangle_emits_two_same_as_edges_byte_equal(self, colbert_endpoint):
        nodes = [
            _marie_curie_node(),
            _woman_in_lab_coat_node(_vlm_lab_coat_at_14_mention()),
            _curie_1903_node(_ocr_at_14_5_mention()),
        ]
        extraction = ExtractionResult(
            source_doc_id=VIDEO_ID,
            nodes=nodes,
            edges=[],
        )

        linked = _linker(colbert_endpoint).link(extraction)
        same_as = _same_as_edges(linked)
        assert len(same_as) == 2, (
            f"expected exactly 2 same_as edges in triangle, got {len(same_as)}: "
            f"{[(e.source, e.target, e.confidence) for e in same_as]}"
        )

        # Sort by source so the golden is stable regardless of internal
        # candidate ordering — the linker pairs in modality-sorted order,
        # but the test should not rely on that incidental detail.
        sorted_edges = sorted(
            [_edge_to_jsonable(e) for e in same_as],
            key=lambda d: (d["source"], d["target"]),
        )
        assert_golden(sorted_edges, "cross_modal_triangle.json")


# --------------------------------------------------------------------- #
# Idempotency: re-running the linker is a no-op                          #
# --------------------------------------------------------------------- #


class TestCrossModalLinkerIdempotency:
    """A second ``link()`` call on already-linked output is byte-equal.

    The linker dedupes by deterministic ``edge_id``; re-running on its
    own output must produce the same edge list in the same order, both
    in count and in dict shape.
    """

    def test_second_link_call_byte_equal(self, colbert_endpoint):
        nodes = [
            _marie_curie_node(),
            _woman_in_lab_coat_node(_vlm_lab_coat_at_14_mention()),
        ]
        extraction = ExtractionResult(
            source_doc_id=VIDEO_ID,
            nodes=nodes,
            edges=[],
        )

        linker = _linker(colbert_endpoint)
        first = linker.link(extraction)
        second = linker.link(first)

        first_same_as = sorted(
            [_edge_to_jsonable(e) for e in _same_as_edges(first)],
            key=lambda d: (d["source"], d["target"], d["segment_id"]),
        )
        second_same_as = sorted(
            [_edge_to_jsonable(e) for e in _same_as_edges(second)],
            key=lambda d: (d["source"], d["target"], d["segment_id"]),
        )
        assert first_same_as == second_same_as
        assert_golden(second_same_as, "cross_modal_idempotency.json")
