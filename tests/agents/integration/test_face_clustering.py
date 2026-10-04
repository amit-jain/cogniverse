"""Integration test for per-video face clustering.

Drives ``cluster_faces`` over a deterministic 4-FaceMention debate
fixture and locks the CL1–CL9 assertion contract from
``docs/plan/face-clustering-assertions.md``. Clusters pinned
float32-exact vectors
(``0.125`` and ``-0.125`` per 512 dims) so cosine distance between
identities is exactly 2.0 and within identity exactly 0.0 — clustering
is deterministic without any LM / encoder dependency.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cogniverse_agents.graph.face_clusterer import cluster_faces
from cogniverse_agents.graph.face_extractor import face_mention_as_jsonable
from cogniverse_agents.graph.graph_schema import FaceCluster, FaceMention

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------- #
# Fixture builders                                                       #
# --------------------------------------------------------------------- #


ALICE_VEC = tuple([0.125] * 512)
BOB_VEC = tuple([-0.125] * 512)


def _mention(
    *,
    segment_id: str,
    ts_start: float,
    bbox: tuple,
    vec: tuple,
    det_score: float,
) -> FaceMention:
    return FaceMention(
        source_doc_id="debate_30s",
        segment_id=segment_id,
        ts_start=ts_start,
        ts_end=ts_start,
        bbox=bbox,
        vec=vec,
        det_score=det_score,
    )


def _debate_mentions() -> list:
    return [
        _mention(
            segment_id="frame_5_0",
            ts_start=5.0,
            bbox=(100, 40, 200, 140),
            vec=ALICE_VEC,
            det_score=0.987,
        ),
        _mention(
            segment_id="frame_15_0",
            ts_start=15.0,
            bbox=(80, 40, 180, 140),
            vec=BOB_VEC,
            det_score=0.964,
        ),
        _mention(
            segment_id="frame_22_5",
            ts_start=22.5,
            bbox=(20, 40, 120, 140),
            vec=ALICE_VEC,
            det_score=0.943,
        ),
        _mention(
            segment_id="frame_22_5",
            ts_start=22.5,
            bbox=(300, 40, 400, 140),
            vec=BOB_VEC,
            det_score=0.928,
        ),
    ]


# --------------------------------------------------------------------- #
# CL1 — Cluster count exact                                              #
# --------------------------------------------------------------------- #


def test_cluster_count_for_debate_is_exactly_two():
    clusters = cluster_faces(_debate_mentions())
    assert len(clusters) == 2


# --------------------------------------------------------------------- #
# CL2 — Members byte-equal per cluster                                   #
# --------------------------------------------------------------------- #


def test_cluster_members_byte_equal_sorted():
    clusters = sorted(cluster_faces(_debate_mentions()), key=lambda c: c.cluster_id)
    # Bob's anchor (frame_15_0::80_40) sorts alphabetically before
    # Alice's (frame_5_0::100_40) — "1" < "5" in lexicographic order.
    bob, alice = clusters

    assert [face_mention_as_jsonable(m) for m in bob.members] == [
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_15_0",
            "ts_start": 15.0,
            "ts_end": 15.0,
            "bbox": [80, 40, 180, 140],
            "vec": list(BOB_VEC),
            "det_score": 0.964,
        },
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_22_5",
            "ts_start": 22.5,
            "ts_end": 22.5,
            "bbox": [300, 40, 400, 140],
            "vec": list(BOB_VEC),
            "det_score": 0.928,
        },
    ]
    assert [face_mention_as_jsonable(m) for m in alice.members] == [
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_5_0",
            "ts_start": 5.0,
            "ts_end": 5.0,
            "bbox": [100, 40, 200, 140],
            "vec": list(ALICE_VEC),
            "det_score": 0.987,
        },
        {
            "source_doc_id": "debate_30s",
            "segment_id": "frame_22_5",
            "ts_start": 22.5,
            "ts_end": 22.5,
            "bbox": [20, 40, 120, 140],
            "vec": list(ALICE_VEC),
            "det_score": 0.943,
        },
    ]


# --------------------------------------------------------------------- #
# CL3 — Cluster IDs locked                                               #
# --------------------------------------------------------------------- #


def test_cluster_ids_locked():
    clusters = cluster_faces(_debate_mentions())
    assert sorted(c.cluster_id for c in clusters) == [
        "face_cluster::frame_15_0::80_40",
        "face_cluster::frame_5_0::100_40",
    ]


# --------------------------------------------------------------------- #
# CL4 — Centroid vector matches expectation                              #
# --------------------------------------------------------------------- #


def test_centroid_vectors_match_normalised_mean():
    clusters = sorted(cluster_faces(_debate_mentions()), key=lambda c: c.cluster_id)
    bob, alice = clusters

    expected_alice = 0.125 / math.sqrt(0.125 * 0.125 * 512)
    expected_bob = -expected_alice

    assert alice.centroid_vec == pytest.approx(tuple([expected_alice] * 512), abs=1e-9)
    assert bob.centroid_vec == pytest.approx(tuple([expected_bob] * 512), abs=1e-9)


# --------------------------------------------------------------------- #
# CL5 — Single-input single cluster                                      #
# --------------------------------------------------------------------- #


def test_single_input_yields_single_cluster_with_member_vec_as_centroid():
    only = _mention(
        segment_id="frame_5_0",
        ts_start=5.0,
        bbox=(100, 40, 200, 140),
        vec=ALICE_VEC,
        det_score=0.987,
    )
    clusters = cluster_faces([only])
    assert len(clusters) == 1
    assert clusters[0].members == (only,)
    assert clusters[0].centroid_vec == only.vec


# --------------------------------------------------------------------- #
# CL6 — Empty input → empty list                                         #
# --------------------------------------------------------------------- #


def test_empty_input_yields_empty_list():
    assert cluster_faces([]) == []


# --------------------------------------------------------------------- #
# CL7 — All-distinct vectors → N clusters                                #
# --------------------------------------------------------------------- #


def test_orthogonal_vectors_produce_n_singleton_clusters():
    def basis_vec(dim_idx: int) -> tuple:
        v = [0.0] * 512
        v[dim_idx] = 1.0
        return tuple(v)

    mentions = [
        _mention(
            segment_id=f"frame_{i}_0",
            ts_start=float(i),
            bbox=(0, 0, 100, 100),
            vec=basis_vec(i),
            det_score=0.9,
        )
        for i in range(3)
    ]
    clusters = cluster_faces(mentions)
    assert len(clusters) == 3
    assert {len(c.members) for c in clusters} == {1}


# --------------------------------------------------------------------- #
# CL8 — Idempotency                                                      #
# --------------------------------------------------------------------- #


def test_clustering_is_idempotent():
    first = cluster_faces(_debate_mentions())
    second = cluster_faces(_debate_mentions())
    assert [
        (c.cluster_id, c.centroid_vec, tuple(c.members))
        for c in sorted(first, key=lambda c: c.cluster_id)
    ] == [
        (c.cluster_id, c.centroid_vec, tuple(c.members))
        for c in sorted(second, key=lambda c: c.cluster_id)
    ]


# --------------------------------------------------------------------- #
# CL9 — FaceCluster dataclass shape locked                               #
# --------------------------------------------------------------------- #


def test_facecluster_shape_locked():
    import dataclasses

    fields = dataclasses.fields(FaceCluster)
    assert len(fields) == 3
    assert [f.name for f in fields] == ["cluster_id", "members", "centroid_vec"]


# --------------------------------------------------------------------- #
# CL10 — Average-linkage partitions, without loading scikit-learn       #
# --------------------------------------------------------------------- #


def _partition(labels) -> list:
    groups: dict = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(tuple(g) for g in groups.values())


def test_partitions_match_sklearn_average_linkage():
    """The clusterer's partitions equal scikit-learn's cosine average-linkage
    cut at the same threshold, over identities with varied spread."""
    from sklearn.cluster import AgglomerativeClustering

    from cogniverse_agents.graph.face_clusterer import _agglomerative_cosine

    rng = np.random.default_rng(7)
    compared = []
    for _ in range(60):
        identities = int(rng.integers(1, 10))
        faces = int(rng.integers(2, 60))
        centres = rng.normal(size=(identities, 512))
        spread = float(rng.uniform(0.2, 1.2))
        vecs = centres[rng.integers(0, identities, faces)] + spread * rng.normal(
            size=(faces, 512)
        )
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        for threshold in (0.3, 0.4, 0.5):
            expected = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=threshold,
            ).fit_predict(vecs)
            got = _agglomerative_cosine(vecs, threshold)
            assert _partition(got) == _partition(expected), (threshold, faces)
            compared.append(len(_partition(expected)))
    # The draws cut into anything from one cluster to 59, at 41 distinct counts.
    assert min(compared) == 1
    assert max(compared) == 59
    assert len(set(compared)) == 41


def test_a_pair_exactly_at_the_threshold_stays_apart():
    from sklearn.cluster import AgglomerativeClustering

    from cogniverse_agents.graph.face_clusterer import _agglomerative_cosine

    # Orthogonal faces sit at a cosine distance of exactly 1.0.
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    for threshold, expected in ((1.0, [(0,), (1,)]), (1.0000001, [(0, 1)])):
        reference = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold,
        ).fit_predict(vecs)
        assert _partition(reference) == expected
        assert _partition(_agglomerative_cosine(vecs, threshold)) == expected


def test_clustering_does_not_load_scikit_learn():
    """scikit-learn costs an ingestion worker about 70 MiB of resident
    memory for the rest of its life; clustering a video's faces must not
    import it."""
    script = (
        "import sys\n"
        "from tests.agents.integration.test_face_clustering import _debate_mentions\n"
        "from cogniverse_agents.graph.face_clusterer import cluster_faces\n"
        "clusters = cluster_faces(_debate_mentions())\n"
        "print(len(clusters), 'sklearn' in sys.modules, 'scipy' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert done.stdout.strip() == "2 False False"
