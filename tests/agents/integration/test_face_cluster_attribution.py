"""Integration test for cluster ↔ named-entity temporal attribution.

Drives ``attribute_clusters_to_persons`` over the debate fixture and
locks the AT1–AT8 assertion contract in
``docs/plan/face-cluster-attribution-assertions.md``: edge count
exact, full edge dicts byte-equal, unambiguous dominance → confidence
1.0, tied overlap broken alphabetically, empty Person list / empty
cluster list / zero-overlap cluster → no edges, idempotent.
"""

import pytest

from cogniverse_agents.graph.face_cluster_attributor import (
    attribute_clusters_to_persons,
)
from cogniverse_agents.graph.graph_schema import (
    ExtractionResult,
    FaceCluster,
    FaceMention,
    Mention,
    Node,
)

pytestmark = pytest.mark.integration


ALICE_VEC = tuple([0.125] * 512)
BOB_VEC = tuple([-0.125] * 512)


def _fm(*, segment_id, ts_start, bbox, vec, det_score=0.9):
    return FaceMention(
        source_doc_id="debate_30s",
        segment_id=segment_id,
        ts_start=ts_start,
        ts_end=ts_start,
        bbox=bbox,
        vec=vec,
        det_score=det_score,
    )


def _person_node(name: str, transcript_mentions: list):
    return Node(
        tenant_id="test",
        name=name,
        mentions=transcript_mentions,
        label="Person",
        kind="entity",
    )


def _transcript_mention(seg_id: str, ts_start: float, ts_end: float, span: str):
    return Mention(
        source_doc_id="debate_30s",
        segment_id=seg_id,
        ts_start=ts_start,
        ts_end=ts_end,
        modality="transcript",
        evidence_span=span,
    )


def _debate_extraction_result():
    alice = _person_node(
        "Alice Chen",
        [
            _transcript_mention("seg_1", 0.0, 10.0, "Alice Chen presents..."),
            _transcript_mention("seg_3", 20.0, 25.0, "Alice Chen replies..."),
        ],
    )
    bob = _person_node(
        "Bob Smith",
        [_transcript_mention("seg_2", 10.0, 20.0, "Bob Smith responds...")],
    )
    return ExtractionResult(
        source_doc_id="debate_30s",
        nodes=[alice, bob],
        edges=[],
    )


def _alice_cluster():
    return FaceCluster(
        cluster_id="face_cluster::frame_5_0::100_40",
        members=(
            _fm(
                segment_id="frame_5_0",
                ts_start=5.0,
                bbox=(100, 40, 200, 140),
                vec=ALICE_VEC,
                det_score=0.987,
            ),
            _fm(
                segment_id="frame_22_5",
                ts_start=22.5,
                bbox=(20, 40, 120, 140),
                vec=ALICE_VEC,
                det_score=0.943,
            ),
        ),
        centroid_vec=ALICE_VEC,
    )


def _bob_cluster():
    return FaceCluster(
        cluster_id="face_cluster::frame_15_0::80_40",
        members=(
            _fm(
                segment_id="frame_15_0",
                ts_start=15.0,
                bbox=(80, 40, 180, 140),
                vec=BOB_VEC,
                det_score=0.964,
            ),
            _fm(
                segment_id="frame_22_5",
                ts_start=22.5,
                bbox=(300, 40, 400, 140),
                vec=BOB_VEC,
                det_score=0.928,
            ),
        ),
        centroid_vec=BOB_VEC,
    )


# --------------------------------------------------------------------- #
# AT1 — Edge count exact                                                 #
# --------------------------------------------------------------------- #


def test_edge_count_exact():
    edges = attribute_clusters_to_persons(
        [_alice_cluster(), _bob_cluster()],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    assert len(edges) == 2


# --------------------------------------------------------------------- #
# AT2 — Edges byte-equal sorted by source                                #
# --------------------------------------------------------------------- #


def test_edges_byte_equal_sorted_by_source():
    edges = attribute_clusters_to_persons(
        [_alice_cluster(), _bob_cluster()],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    edges_sorted = sorted(edges, key=lambda e: e.source)
    edge_dicts = [
        {
            "source": e.source,
            "target": e.target,
            "relation": e.relation,
            "evidence_span": e.evidence_span,
            "segment_id": e.segment_id,
            "ts_start": e.ts_start,
            "ts_end": e.ts_end,
            "modality": e.modality,
            "provenance": e.provenance,
            "source_doc_id": e.source_doc_id,
            "confidence": e.confidence,
        }
        for e in edges_sorted
    ]
    assert edge_dicts == [
        {
            "source": "face_cluster::frame_15_0::80_40",
            "target": "Alice Chen",
            "relation": "same_as",
            "evidence_span": "face_cluster_temporal",
            "segment_id": "frame_15_0",
            "ts_start": 15.0,
            "ts_end": 15.0,
            "modality": "vlm",
            "provenance": "face_cluster_temporal",
            "source_doc_id": "debate_30s",
            "confidence": 0.5,
        },
        {
            "source": "face_cluster::frame_5_0::100_40",
            "target": "Alice Chen",
            "relation": "same_as",
            "evidence_span": "face_cluster_temporal",
            "segment_id": "frame_5_0",
            "ts_start": 5.0,
            "ts_end": 5.0,
            "modality": "vlm",
            "provenance": "face_cluster_temporal",
            "source_doc_id": "debate_30s",
            "confidence": 1.0,
        },
    ]


# --------------------------------------------------------------------- #
# AT3 — Unambiguous dominance → confidence 1.0                           #
# --------------------------------------------------------------------- #


def test_unambiguous_dominance_yields_confidence_one():
    edges = attribute_clusters_to_persons(
        [_alice_cluster()],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    assert len(edges) == 1
    assert edges[0].confidence == 1.0
    assert edges[0].target == "Alice Chen"


# --------------------------------------------------------------------- #
# AT4 — Tied overlap broken alphabetically                               #
# --------------------------------------------------------------------- #


def test_tied_overlap_broken_alphabetically():
    edges = attribute_clusters_to_persons(
        [_bob_cluster()],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    assert len(edges) == 1
    # Bob cluster scores 1/2 vs Alice Chen and 1/2 vs Bob Smith.
    # "Alice Chen" < "Bob Smith" lexicographically → Alice wins.
    assert edges[0].target == "Alice Chen"
    assert edges[0].confidence == 0.5


# --------------------------------------------------------------------- #
# AT5 — Empty Person list → empty edges                                  #
# --------------------------------------------------------------------- #


def test_empty_person_list_yields_no_edges():
    no_persons = ExtractionResult(
        source_doc_id="debate_30s",
        nodes=[],
        edges=[],
    )
    edges = attribute_clusters_to_persons(
        [_alice_cluster(), _bob_cluster()],
        no_persons,
        source_doc_id="debate_30s",
    )
    assert edges == []


# --------------------------------------------------------------------- #
# AT6 — Empty cluster list → empty edges                                 #
# --------------------------------------------------------------------- #


def test_empty_cluster_list_yields_no_edges():
    edges = attribute_clusters_to_persons(
        [],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    assert edges == []


# --------------------------------------------------------------------- #
# AT7 — Zero-overlap cluster emits NO edge                               #
# --------------------------------------------------------------------- #


def test_zero_overlap_cluster_emits_no_edge():
    far_future = FaceCluster(
        cluster_id="face_cluster::frame_far::0_0",
        members=(
            _fm(
                segment_id="frame_far",
                ts_start=200.0,
                bbox=(0, 0, 100, 100),
                vec=ALICE_VEC,
                det_score=0.9,
            ),
        ),
        centroid_vec=ALICE_VEC,
    )
    edges = attribute_clusters_to_persons(
        [far_future],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    assert edges == []


# --------------------------------------------------------------------- #
# AT8 — Idempotency                                                      #
# --------------------------------------------------------------------- #


def test_attribution_is_idempotent():
    first = attribute_clusters_to_persons(
        [_alice_cluster(), _bob_cluster()],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    second = attribute_clusters_to_persons(
        [_alice_cluster(), _bob_cluster()],
        _debate_extraction_result(),
        source_doc_id="debate_30s",
    )
    assert [
        (e.source, e.target, e.relation, e.confidence, e.provenance)
        for e in sorted(first, key=lambda e: e.source)
    ] == [
        (e.source, e.target, e.relation, e.confidence, e.provenance)
        for e in sorted(second, key=lambda e: e.source)
    ]
