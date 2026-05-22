"""Integration test for emit_anonymous_face_nodes.

Locks the AN1–AN6 contract for the anonymous-face Node builder:
 * AN1 — orphan cluster list yields one Node per cluster.
 * AN2 — Node fields byte-equal: deterministic name (cluster_id),
   kind="anonymous_face", label="Person", per-member Mention list.
 * AN3 — empty orphan list yields empty list.
 * AN4 — multi-member cluster produces multi-mention Node.
 * AN5 — idempotency: same input → same Node identities byte-equal.
 * AN6 — Node.node_id matches cluster_id slugified (so re-ingest
   upserts the same row in GraphManager).
"""

import dataclasses

import pytest

from cogniverse_agents.graph.anonymous_face_node_builder import (
    emit_anonymous_face_nodes,
)
from cogniverse_agents.graph.graph_schema import (
    FaceCluster,
    FaceMention,
    normalize_name,
)

pytestmark = pytest.mark.integration


ALICE_VEC = tuple([0.125] * 512)


def _fm(*, segment_id, ts_start, bbox, det_score):
    return FaceMention(
        source_doc_id="debate_30s",
        segment_id=segment_id,
        ts_start=ts_start,
        ts_end=ts_start,
        bbox=bbox,
        vec=ALICE_VEC,
        det_score=det_score,
    )


def _orphan_alice_two_keyframes():
    return FaceCluster(
        cluster_id="face_cluster::frame_5_0::100_40",
        members=(
            _fm(
                segment_id="frame_5_0",
                ts_start=5.0,
                bbox=(100, 40, 200, 140),
                det_score=0.987,
            ),
            _fm(
                segment_id="frame_22_5",
                ts_start=22.5,
                bbox=(20, 40, 120, 140),
                det_score=0.943,
            ),
        ),
        centroid_vec=ALICE_VEC,
    )


def _orphan_single_keyframe():
    return FaceCluster(
        cluster_id="face_cluster::frame_15_0::80_40",
        members=(
            _fm(
                segment_id="frame_15_0",
                ts_start=15.0,
                bbox=(80, 40, 180, 140),
                det_score=0.964,
            ),
        ),
        centroid_vec=ALICE_VEC,
    )


# --------------------------------------------------------------------- #
# AN1 — One Node per orphan cluster                                       #
# --------------------------------------------------------------------- #


def test_one_node_per_orphan_cluster():
    nodes = emit_anonymous_face_nodes(
        [_orphan_alice_two_keyframes(), _orphan_single_keyframe()],
        source_doc_id="debate_30s",
        tenant_id="test",
    )
    assert len(nodes) == 2
    assert sorted(n.name for n in nodes) == [
        "face_cluster::frame_15_0::80_40",
        "face_cluster::frame_5_0::100_40",
    ]


# --------------------------------------------------------------------- #
# AN2 — Node fields byte-equal (kind, label, name, mention shape)         #
# --------------------------------------------------------------------- #


def test_node_fields_byte_equal():
    nodes = emit_anonymous_face_nodes(
        [_orphan_alice_two_keyframes()],
        source_doc_id="debate_30s",
        tenant_id="test",
    )
    assert len(nodes) == 1
    n = nodes[0]
    assert n.name == "face_cluster::frame_5_0::100_40"
    assert n.kind == "anonymous_face"
    assert n.label == "Person"
    assert n.tenant_id == "test"
    assert len(n.mentions) == 2
    m0 = n.mentions[0]
    assert m0.segment_id == "frame_5_0"
    assert m0.ts_start == 5.0
    assert m0.ts_end == 5.0
    assert m0.modality == "vlm"
    assert m0.source_doc_id == "debate_30s"
    assert m0.evidence_span == (
        "anonymous face bbox (100,40)-(200,140) det_score=0.9870"
    )
    m1 = n.mentions[1]
    assert m1.segment_id == "frame_22_5"
    assert m1.ts_start == 22.5
    assert m1.evidence_span == (
        "anonymous face bbox (20,40)-(120,140) det_score=0.9430"
    )


# --------------------------------------------------------------------- #
# AN3 — Empty orphan list → empty list                                   #
# --------------------------------------------------------------------- #


def test_empty_orphan_list_yields_empty():
    assert (
        emit_anonymous_face_nodes(
            [],
            source_doc_id="debate_30s",
            tenant_id="test",
        )
        == []
    )


# --------------------------------------------------------------------- #
# AN4 — Multi-member cluster → multi-mention Node                         #
# --------------------------------------------------------------------- #


def test_multi_member_cluster_produces_multi_mention_node():
    nodes = emit_anonymous_face_nodes(
        [_orphan_alice_two_keyframes()],
        source_doc_id="debate_30s",
        tenant_id="test",
    )
    assert len(nodes[0].mentions) == 2
    segs = {m.segment_id for m in nodes[0].mentions}
    assert segs == {"frame_5_0", "frame_22_5"}


# --------------------------------------------------------------------- #
# AN5 — Idempotency                                                       #
# --------------------------------------------------------------------- #


def test_emission_is_idempotent():
    first = emit_anonymous_face_nodes(
        [_orphan_alice_two_keyframes(), _orphan_single_keyframe()],
        source_doc_id="debate_30s",
        tenant_id="test",
    )
    second = emit_anonymous_face_nodes(
        [_orphan_alice_two_keyframes(), _orphan_single_keyframe()],
        source_doc_id="debate_30s",
        tenant_id="test",
    )
    assert [n.name for n in first] == [n.name for n in second]
    assert [[dataclasses.asdict(m) for m in n.mentions] for n in first] == [
        [dataclasses.asdict(m) for m in n.mentions] for n in second
    ]
    # node_id is derived from name → byte-equal across runs.
    assert [n.node_id for n in first] == [n.node_id for n in second]


# --------------------------------------------------------------------- #
# AN6 — Node.node_id derives from cluster_id deterministically            #
# --------------------------------------------------------------------- #


def test_node_id_derives_from_cluster_id():
    nodes = emit_anonymous_face_nodes(
        [_orphan_alice_two_keyframes()],
        source_doc_id="debate_30s",
        tenant_id="test",
    )
    n = nodes[0]
    # normalize_name lowercases + replaces non-alphanumeric with _.
    expected_node_id = normalize_name("face_cluster::frame_5_0::100_40")
    assert n.node_id == expected_node_id
    # Also confirm doc_id pattern (tenant + node_id slug).
    assert n.doc_id == f"kg_node_test_{expected_node_id}"
