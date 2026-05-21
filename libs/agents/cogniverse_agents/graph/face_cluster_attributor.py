"""Cluster ↔ named-entity temporal attribution.

For each anonymous ``FaceCluster``, find the named transcript Person
whose mention windows overlap most heavily with the cluster's keyframe
timestamps. Emit a ``same_as`` Edge from the cluster's ``cluster_id``
to the Person's name with ``provenance="face_cluster_temporal"`` and
``confidence`` equal to the fraction of cluster face mentions that
fell within any of the winning Person's transcript windows.

Resolves the multi-subject case (interviews, debates, compilations)
that single-video subject inference cannot handle on its own — when no
Person dominates the whole transcript, per-cluster temporal alignment
attributes each anonymous face group to the right named entity.

Determinism:
* Clusters processed in ``cluster_id`` order.
* Ties on overlap broken alphabetically by Person name.
* Zero-overlap clusters emit NO edge (silence, not confidence=0.0).
* Same input → same output edge list, byte-equal.
"""

from __future__ import annotations

from typing import List

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    FaceCluster,
    FaceMention,
    Mention,
)

_TRANSCRIPT_MODALITY = "transcript"
_VLM_MODALITY = "vlm"
_RELATION = "same_as"
_PROVENANCE = "face_cluster_temporal"
_EVIDENCE_SPAN = "face_cluster_temporal"


def _person_transcript_windows(
    extraction_result: ExtractionResult,
) -> dict:
    """Return mapping of Person name → list of transcript Mention windows.

    Only ``label == "Person"`` nodes with at least one transcript-modality
    mention are returned. Empty mapping when no Person is present.
    """
    out: dict = {}
    for node in extraction_result.nodes:
        if (node.label or "").strip() != "Person":
            continue
        windows = [m for m in node.mentions if m.modality == _TRANSCRIPT_MODALITY]
        if windows:
            out[node.name] = windows
    return out


def _overlap_count(face_mentions: tuple, person_windows: List[Mention]) -> int:
    """Count face mentions whose ts_start lies within any Person window."""
    count = 0
    for fm in face_mentions:
        for pw in person_windows:
            if pw.ts_start <= fm.ts_start <= pw.ts_end:
                count += 1
                break  # one window match is enough; don't double-count
    return count


def _anchor_member(cluster: FaceCluster) -> FaceMention:
    """Lowest-(ts_start, segment_id) member — the edge's anchor."""
    return min(cluster.members, key=lambda m: (m.ts_start, m.segment_id))


def attribute_clusters_to_persons(
    clusters: List[FaceCluster],
    extraction_result: ExtractionResult,
    *,
    source_doc_id: str,
) -> List[Edge]:
    """Emit one ``same_as`` Edge per cluster that has a temporally-aligned Person.

    Empty cluster list → empty edge list. Empty Person list (or no
    Persons with transcript windows) → empty edge list. Clusters whose
    face mentions never fall inside any Person's transcript windows
    are skipped (no zero-confidence edges).
    """
    if not clusters:
        return []
    person_windows = _person_transcript_windows(extraction_result)
    if not person_windows:
        return []

    edges: List[Edge] = []
    tenant_id = extraction_result.nodes[0].tenant_id if extraction_result.nodes else ""

    for cluster in sorted(clusters, key=lambda c: c.cluster_id):
        scores: List[tuple] = []
        for person_name in sorted(person_windows.keys()):
            count = _overlap_count(cluster.members, person_windows[person_name])
            if count == 0:
                continue
            scores.append((count, person_name))
        if not scores:
            continue

        # Best overlap wins; ties broken alphabetically (the sort above
        # places earlier-alphabet Persons first, so a max() with a stable
        # key on ``-count`` then ``name`` picks the right one).
        scores.sort(key=lambda s: (-s[0], s[1]))
        winning_count, winning_name = scores[0]
        confidence = winning_count / len(cluster.members)

        anchor = _anchor_member(cluster)
        edges.append(
            Edge(
                tenant_id=tenant_id,
                source=cluster.cluster_id,
                target=winning_name,
                relation=_RELATION,
                evidence_span=_EVIDENCE_SPAN,
                segment_id=anchor.segment_id,
                ts_start=anchor.ts_start,
                ts_end=anchor.ts_end,
                modality=_VLM_MODALITY,
                provenance=_PROVENANCE,
                source_doc_id=source_doc_id,
                confidence=round(confidence, 4),
            )
        )
    return edges
