"""Anonymous-face Node builder.

For each ``FaceCluster`` left orphan by ``attribute_clusters_to_persons``
(no transcript Person temporally aligned with the cluster's keyframes),
emit a first-class ``Node`` with ``kind="anonymous_face"`` so the
face-detection + clustering work isn't thrown away. The Node's
``name`` is the deterministic cluster_id, so re-ingesting the same
video upserts the same Node (idempotent at the GraphManager layer).

Downstream consumers (``KnowledgeGraphTraversalAgent``,
``TemporalReasoningAgent``, ``CitationTracingAgent``, etc.) treat the
anonymous Node like any other Person-labelled Node: it appears in
temporal timelines, citation chains, and traversal results. A future
operator (or a manual-labelling UI) can attach a real ``same_as``
edge linking the anonymous cluster_id to a real Person Node — the
algorithmic guess-from-KG-token-bag path is intentionally absent
because sparse-KG environments produce more false positives than
correct attributions.
"""

from __future__ import annotations

from typing import List

from cogniverse_agents.graph.graph_schema import FaceCluster, Mention, Node


def emit_anonymous_face_nodes(
    orphan_clusters: List[FaceCluster],
    *,
    source_doc_id: str,
    tenant_id: str,
) -> List[Node]:
    """Build anonymous-face ``Node`` instances for orphan clusters.

    ``orphan_clusters`` is the subset of ``cluster_faces`` output that
    received no ``face_cluster_temporal`` ``same_as`` edge — typically
    because no transcript ``Person`` mention overlapped the cluster's
    keyframe windows. Returns one ``Node`` per orphan, with one
    ``Mention`` per cluster member (modality="vlm"). Empty input →
    empty list.

    Determinism:
    * ``Node.name = cluster.cluster_id`` is derived from the cluster's
      lowest-``(ts_start, segment_id, bbox)`` anchor member, so the
      identical input always produces identical Node ids — re-ingest
      is a no-op via GraphManager's upsert dedup by ``node_id``.
    * ``Mention`` entries are emitted in the cluster's member order
      (already ``(ts_start, segment_id, bbox)``-sorted by
      ``cluster_faces``).
    """
    nodes: List[Node] = []
    for cluster in orphan_clusters:
        mentions = [
            Mention(
                source_doc_id=source_doc_id,
                segment_id=fm.segment_id,
                ts_start=fm.ts_start,
                ts_end=fm.ts_end,
                modality="vlm",
                evidence_span=(
                    f"anonymous face bbox "
                    f"({fm.bbox[0]},{fm.bbox[1]})-({fm.bbox[2]},{fm.bbox[3]}) "
                    f"det_score={fm.det_score:.4f}"
                ),
            )
            for fm in cluster.members
        ]
        nodes.append(
            Node(
                tenant_id=tenant_id,
                name=cluster.cluster_id,
                description=(
                    "Anonymous face — face-embed sidecar detected this "
                    "identity across one or more keyframes but no "
                    "transcript Person temporally aligned. Upsert a "
                    "same_as edge from this Node to a real Person Node "
                    "to label it."
                ),
                kind="anonymous_face",
                # Person label so downstream Person-aware agents
                # (CitationTracing, TemporalReasoning, KG traversal) see
                # anonymous faces in their results without special-casing.
                label="Person",
                mentions=mentions,
            )
        )
    return nodes
