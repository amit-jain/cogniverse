"""Integration test for KG-overlap orphan attribution.

Drives ``attribute_orphans_by_kg_overlap`` + ``build_person_profile_bags``
over a rich-Person fixture (Marie Curie, Einstein, Newton) and locks
the KGO1–KGO8 contract from
``docs/plan/kg-overlap-attribution-assertions.md``: single-winner edge
shape byte-equal, dict equality, dominance-margin gate (close runners
suppress emission), zero overlap → no edge, empty inputs → empty,
profile-bag builder produces exact token sets, idempotency.
"""

import pytest

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    FaceCluster,
    FaceMention,
    Mention,
    Node,
)
from cogniverse_agents.graph.kg_overlap_attributor import (
    attribute_orphans_by_kg_overlap,
    build_person_profile_bags,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------- #
# Fixture builders                                                       #
# --------------------------------------------------------------------- #


MARIE_CURIE_BAG = {
    "sorbonne",
    "radium",
    "warsaw",
    "nobel",
    "prize",
    "physics",
    "chemistry",
    "pierre",
    "discovered",
    "worked_at",
    "won",
    "born_in",
    "field",
    "contemporary",
}
ALBERT_EINSTEIN_BAG = {
    "princeton",
    "relativity",
    "physics",
    "ulm",
    "patent",
    "office",
    "discovered",
    "wrote",
    "born_in",
    "field",
}
ISAAC_NEWTON_BAG = {
    "cambridge",
    "gravity",
    "calculus",
    "principia",
    "mathematics",
    "discovered",
    "wrote",
    "born_in",
    "field",
}

CANDIDATE_PERSONS = {
    "Marie Curie": MARIE_CURIE_BAG,
    "Albert Einstein": ALBERT_EINSTEIN_BAG,
    "Isaac Newton": ISAAC_NEWTON_BAG,
}


def _orphan_alice() -> FaceCluster:
    member = FaceMention(
        source_doc_id="science_panel_30s",
        segment_id="frame_42",
        ts_start=42.0,
        ts_end=42.0,
        bbox=(100, 40, 200, 140),
        vec=tuple([0.125] * 512),
        det_score=0.9,
    )
    return FaceCluster(
        cluster_id="face_cluster::frame_42::100_40",
        members=(member,),
        centroid_vec=member.vec,
    )


# --------------------------------------------------------------------- #
# KGO1 + KGO2 — Single winner + byte-equal edge dict                     #
# --------------------------------------------------------------------- #


def test_single_winner_emits_edge_byte_equal():
    caption_tokens = {
        "chemist",
        "demonstrating",
        "reaction",
        "periodic",
        "table",
        "radium",
        "sorbonne",
    }
    # Marie Curie ∩ = {radium, sorbonne} → 2; |∪| = 14 + 7 - 2 = 19 → ≈0.1053
    # Einstein, Newton both have 0 overlap.
    edges = attribute_orphans_by_kg_overlap(
        [(_orphan_alice(), caption_tokens)],
        CANDIDATE_PERSONS,
        source_doc_id="science_panel_30s",
        tenant_id="test",
    )
    assert len(edges) == 1
    edge_dict = {
        "source": edges[0].source,
        "target": edges[0].target,
        "relation": edges[0].relation,
        "evidence_span": edges[0].evidence_span,
        "segment_id": edges[0].segment_id,
        "ts_start": edges[0].ts_start,
        "ts_end": edges[0].ts_end,
        "modality": edges[0].modality,
        "provenance": edges[0].provenance,
        "source_doc_id": edges[0].source_doc_id,
        "confidence": edges[0].confidence,
    }
    assert edge_dict == {
        "source": "face_cluster::frame_42::100_40",
        "target": "Marie Curie",
        "relation": "same_as",
        "evidence_span": "kg_overlap",
        "segment_id": "frame_42",
        "ts_start": 42.0,
        "ts_end": 42.0,
        "modality": "vlm",
        "provenance": "kg_overlap",
        "source_doc_id": "science_panel_30s",
        "confidence": 0.1053,
    }


# --------------------------------------------------------------------- #
# KGO3 — Margin gate: decisive winner vs close runner-up                 #
# --------------------------------------------------------------------- #


def test_decisive_margin_emits_edge():
    """Top × 0.7 ≥ runner-up → decisive winner emits."""
    caption_tokens = {"physics", "discovered", "wrote"}
    # Marie ∩ = {physics, discovered} → 2; |∪| = 14+3-2 = 15 → 0.1333
    # Einstein ∩ = {physics, discovered, wrote} → 3; |∪| = 10+3-3 = 10 → 0.3000
    # Newton ∩ = {discovered, wrote} → 2; |∪| = 9+3-2 = 10 → 0.2000
    # Top: Einstein 0.30. Runner-up: Newton 0.20. 0.20 ≤ 0.30 × 0.7 = 0.21 ✓
    edges = attribute_orphans_by_kg_overlap(
        [(_orphan_alice(), caption_tokens)],
        CANDIDATE_PERSONS,
        source_doc_id="science_panel_30s",
        tenant_id="test",
    )
    assert len(edges) == 1
    assert edges[0].target == "Albert Einstein"
    assert edges[0].confidence == 0.3


def test_close_runner_up_suppresses_edge():
    """Top × 0.7 < runner-up → no decisive winner → empty list."""
    caption_tokens = {"physics", "discovered"}
    # Marie ∩ = {physics, discovered} → 2; |∪| = 14+2-2 = 14 → ≈0.1429
    # Einstein ∩ = {physics, discovered} → 2; |∪| = 10+2-2 = 10 → 0.2000
    # Newton ∩ = {discovered} → 1; |∪| = 9+2-1 = 10 → 0.1000
    # Top: Einstein 0.20. Runner-up: Marie 0.1429. 0.1429 > 0.20 × 0.7 = 0.14 → suppress.
    edges = attribute_orphans_by_kg_overlap(
        [(_orphan_alice(), caption_tokens)],
        CANDIDATE_PERSONS,
        source_doc_id="science_panel_30s",
        tenant_id="test",
    )
    assert edges == []


# --------------------------------------------------------------------- #
# KGO4 — Zero overlap → no edge                                          #
# --------------------------------------------------------------------- #


def test_zero_overlap_suppresses_edge():
    caption_tokens = {"weather", "umbrella", "raining"}
    edges = attribute_orphans_by_kg_overlap(
        [(_orphan_alice(), caption_tokens)],
        CANDIDATE_PERSONS,
        source_doc_id="science_panel_30s",
        tenant_id="test",
    )
    assert edges == []


# --------------------------------------------------------------------- #
# KGO5 + KGO6 — Empty inputs                                             #
# --------------------------------------------------------------------- #


def test_empty_orphan_list_yields_empty():
    assert (
        attribute_orphans_by_kg_overlap(
            [],
            CANDIDATE_PERSONS,
            source_doc_id="science_panel_30s",
            tenant_id="test",
        )
        == []
    )


def test_empty_person_dict_yields_empty():
    assert (
        attribute_orphans_by_kg_overlap(
            [(_orphan_alice(), {"radium"})],
            {},
            source_doc_id="science_panel_30s",
            tenant_id="test",
        )
        == []
    )


# --------------------------------------------------------------------- #
# KGO7 — Profile-bag builder produces exact token sets                    #
# --------------------------------------------------------------------- #


def _t_mention(seg_id="seg_1"):
    return Mention(
        source_doc_id="science_panel_30s",
        segment_id=seg_id,
        ts_start=0.0,
        ts_end=10.0,
        modality="transcript",
        evidence_span="...",
    )


def test_build_person_profile_bags_exact_tokens():
    result = ExtractionResult(
        source_doc_id="science_panel_30s",
        nodes=[
            Node(
                tenant_id="test",
                name="Marie Curie",
                label="Person",
                kind="entity",
                mentions=[_t_mention()],
            ),
            Node(
                tenant_id="test",
                name="Albert Einstein",
                label="Person",
                kind="entity",
                mentions=[_t_mention()],
            ),
        ],
        edges=[
            Edge(
                tenant_id="test",
                source="Marie Curie",
                target="radium",
                relation="discovered",
                evidence_span="...",
                segment_id="seg_1",
                ts_start=0.0,
                ts_end=10.0,
                modality="transcript",
            ),
            Edge(
                tenant_id="test",
                source="Marie Curie",
                target="Sorbonne",
                relation="worked_at",
                evidence_span="...",
                segment_id="seg_1",
                ts_start=0.0,
                ts_end=10.0,
                modality="transcript",
            ),
            Edge(
                tenant_id="test",
                source="Albert Einstein",
                target="relativity",
                relation="discovered",
                evidence_span="...",
                segment_id="seg_1",
                ts_start=0.0,
                ts_end=10.0,
                modality="transcript",
            ),
        ],
    )
    bags = build_person_profile_bags(result)
    assert bags == {
        "Marie Curie": {"radium", "sorbonne", "discovered", "worked_at"},
        "Albert Einstein": {"relativity", "discovered"},
    }


# --------------------------------------------------------------------- #
# KGO8 — Idempotency                                                     #
# --------------------------------------------------------------------- #


def test_idempotent():
    caption_tokens = {
        "chemist",
        "demonstrating",
        "reaction",
        "periodic",
        "table",
        "radium",
        "sorbonne",
    }
    first = attribute_orphans_by_kg_overlap(
        [(_orphan_alice(), caption_tokens)],
        CANDIDATE_PERSONS,
        source_doc_id="science_panel_30s",
        tenant_id="test",
    )
    second = attribute_orphans_by_kg_overlap(
        [(_orphan_alice(), caption_tokens)],
        CANDIDATE_PERSONS,
        source_doc_id="science_panel_30s",
        tenant_id="test",
    )
    assert [(e.source, e.target, e.confidence, e.provenance) for e in first] == [
        (e.source, e.target, e.confidence, e.provenance) for e in second
    ]
