"""Per-video face clustering.

Groups a ``list[FaceMention]`` produced by ``extract_faces_per_keyframe``
into anonymous-identity clusters via cosine-distance agglomerative
clustering. Output is a deterministic ``list[FaceCluster]`` where each
cluster's members share the same person across keyframes — without
naming the person. The named-entity attribution step downstream
attributes each cluster to a transcript Person by temporal overlap.

``cluster_id`` derivation is deterministic: each cluster's id is
anchored on the lowest-``(ts_start, segment_id, bbox)`` member of that
cluster, formatted as ``f"face_cluster::{segment_id}::{x1}_{y1}"``.
Identical input → identical IDs across runs, enabling golden replay.
"""

from __future__ import annotations

from typing import List

import numpy as np

from cogniverse_agents.graph.graph_schema import FaceCluster, FaceMention

# Cosine-distance threshold separating "same identity" from "different
# identity" pairs. 0.4 is the standard ArcFace-Buffalo_L cut quoted in
# the InsightFace docs; same-person within-cluster pairs typically
# score below 0.3 in real footage while different-person pairs sit
# above 0.5. Anything in [0.3, 0.5] is ambiguous and the threshold can
# be tuned per-tenant later if the operating point needs shifting.
DEFAULT_DISTANCE_THRESHOLD = 0.4


def _anchor_key(m: FaceMention) -> tuple:
    """Stable ordering key for picking the cluster-id anchor member."""
    return (m.ts_start, m.segment_id, m.bbox)


def _cluster_id_for(members: List[FaceMention]) -> str:
    """Derive a deterministic ID from a cluster's lowest-anchor member."""
    anchor = min(members, key=_anchor_key)
    return f"face_cluster::{anchor.segment_id}::{anchor.bbox[0]}_{anchor.bbox[1]}"


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D ndarray in-place-friendly form."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec  # zero vector — leave as-is
    return vec / norm


def _agglomerative_cosine(vecs: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Run cosine-distance average-linkage agglomerative clustering.

    Returns an int ndarray of cluster labels parallel to ``vecs``.
    Deterministic given identical input (sklearn's implementation is).
    Imported inside the function so the import cost isn't paid by
    consumers that never reach the clustering path.
    """
    from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(vecs)


def cluster_faces(
    face_mentions: List[FaceMention],
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> List[FaceCluster]:
    """Cluster face mentions by ArcFace cosine distance.

    * Empty input → empty list.
    * Single input → single cluster of length 1 whose ``centroid_vec``
      equals the lone member's ``vec``.
    * Otherwise → agglomerative clustering at ``distance_threshold``,
      sorted by ``cluster_id`` so the output is replay-stable.

    Members within each cluster are themselves sorted by
    ``(ts_start, segment_id, bbox)`` so byte-equal goldens hold across
    re-runs without depending on sklearn's internal label assignment.
    """
    if not face_mentions:
        return []
    if len(face_mentions) == 1:
        only = face_mentions[0]
        return [
            FaceCluster(
                cluster_id=_cluster_id_for([only]),
                members=(only,),
                centroid_vec=only.vec,
            )
        ]

    vecs = np.array([list(m.vec) for m in face_mentions], dtype=np.float64)
    labels = _agglomerative_cosine(vecs, distance_threshold)

    by_label: dict[int, List[FaceMention]] = {}
    for label, mention in zip(labels, face_mentions, strict=True):
        by_label.setdefault(int(label), []).append(mention)

    clusters: List[FaceCluster] = []
    for label, members in by_label.items():
        members_sorted = sorted(members, key=_anchor_key)
        member_vecs = np.array([list(m.vec) for m in members_sorted], dtype=np.float64)
        centroid = _l2_normalise(member_vecs.mean(axis=0))
        clusters.append(
            FaceCluster(
                cluster_id=_cluster_id_for(members_sorted),
                members=tuple(members_sorted),
                centroid_vec=tuple(float(c) for c in centroid),
            )
        )

    clusters.sort(key=lambda c: c.cluster_id)
    return clusters
