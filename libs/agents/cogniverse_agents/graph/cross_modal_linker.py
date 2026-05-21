"""Cross-modal entity linker.

Run once per ``source_doc_id`` after all per-segment extractions complete.
Emits ``same_as`` edges via three structural-inference primitives — no
pairwise text-similarity scoring:

  * **Shared-name-token coreference** — two cross-modal mentions whose
    Node names share a substantive token (length ≥ 3, case-insensitive,
    e.g. "Marie Curie" vs "Curie 1903" share ``curie``) link via
    ``provenance="INFERRED"``, ``evidence_span="shared_token:<token>"``.
  * **Video-subject inference** — when one Person dominates the
    transcript Person-mentions of a ``source_doc_id``
    (share ≥ ``SUBJECT_DOMINANCE_THRESHOLD``), every generic
    VLM/OCR description in that doc links to the subject via
    ``provenance="video_subject_inference"``, confidence = the
    subject's dominance share.
  * **Per-window subject inference** — when no single Person dominates
    the whole video (interviews / debates / compilations), each VLM/OCR
    mention falls back to the dominant Person inside a
    ``±window_s`` window around its timestamp. Same provenance tag,
    typically lower confidence.

The linker is deterministic: pair ordering is set by sorted
``(source_doc_id, modality, segment_id, ts_start)``. Re-running on its
own output is a no-op because every emitted edge has a deterministic
``edge_id`` and duplicates are filtered.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
)

logger = logging.getLogger(__name__)

# Share of all transcript Person-mentions one Person must hold to be
# treated as the video's subject. 0.6 = 60% dominance; below that the
# content is multi-subject and per-window inference takes over.
SUBJECT_DOMINANCE_THRESHOLD = 0.6

# ± seconds around a VLM/OCR mention's timestamp when running per-window
# subject inference. Wide enough to overlap a few transcript segments,
# narrow enough that the "dominant person" in a compilation video shifts
# scene-to-scene.
DEFAULT_WINDOW_S = 15.0

# Labels that describe a real-world subject the visual modality may
# also describe — used by the (intentionally still-supported) Person↔peer
# token-overlap path.
_PERSON_LIKE_PEER_LABELS = frozenset(
    {
        "Concept",
        "Location",
        "Organization",
        "Field",
        "Substance",
        "Event",
        "",
    }
)

# Words that flag a non-Person caption as describing a human subject.
# Used by per-video and per-window subject inference to decide whether
# a generic visual caption is a candidate for subject attribution.
_PERSON_INDICATOR_WORDS = frozenset(
    {
        "woman",
        "women",
        "man",
        "men",
        "person",
        "people",
        "child",
        "children",
        "boy",
        "girl",
        "scientist",
        "researcher",
        "doctor",
        "professor",
        "teacher",
        "student",
        "engineer",
        "speaker",
        "guest",
        "host",
        "presenter",
        "individual",
        "human",
        "lady",
        "gentleman",
    }
)

_TRANSCRIPT_MODALITY = "transcript"


def _tokenize_for_match(name: str) -> set[str]:
    """Lowercased alphanumeric tokens of length ≥ 3 from a name string."""
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", name or "") if len(t) >= 3}


def _shared_substantive_token(name_a: str, name_b: str) -> Optional[str]:
    """Return the first shared token (length ≥ 3) between two names or None."""
    overlap = _tokenize_for_match(name_a) & _tokenize_for_match(name_b)
    if not overlap:
        return None
    return sorted(overlap)[0]


def _caption_looks_like_subject(node: Node) -> bool:
    """A generic VLM/OCR caption that's a candidate for subject attribution.

    True when the node's tokens contain a person-indicator word AND the
    node isn't already a Person (Person nodes self-attribute by name).
    """
    if (node.label or "").strip() == "Person":
        return False
    tokens = _tokenize_for_match(node.name)
    return bool(tokens & _PERSON_INDICATOR_WORDS)


def infer_video_subject(
    extraction_result: ExtractionResult,
    threshold: float = SUBJECT_DOMINANCE_THRESHOLD,
) -> Optional[Node]:
    """Return the dominant Person if one carries ≥ threshold of all
    transcript Person-mentions in ``extraction_result``. None when no
    Person dominates (multi-subject content).
    """
    counts: Dict[str, int] = {}
    person_nodes: Dict[str, Node] = {}
    total = 0
    for node in extraction_result.nodes:
        if (node.label or "").strip() != "Person":
            continue
        transcript_mentions = sum(
            1 for m in node.mentions if m.modality == _TRANSCRIPT_MODALITY
        )
        if transcript_mentions == 0:
            continue
        counts[node.name] = transcript_mentions
        person_nodes[node.name] = node
        total += transcript_mentions

    if total == 0:
        return None
    top_name = max(counts, key=counts.get)
    if counts[top_name] / total < threshold:
        return None
    return person_nodes[top_name]


def infer_subject_per_window(
    extraction_result: ExtractionResult,
    window_s: float = DEFAULT_WINDOW_S,
) -> Dict[Tuple[float, float], Node]:
    """For each VLM/OCR mention, return the dominant transcript Person
    in its (ts - window_s, ts + window_s) window. Mapping keyed by the
    (ts_start, ts_end) of the mention being attributed.
    """
    transcript_person_mentions: List[Tuple[Node, Mention]] = []
    for node in extraction_result.nodes:
        if (node.label or "").strip() != "Person":
            continue
        for m in node.mentions:
            if m.modality == _TRANSCRIPT_MODALITY:
                transcript_person_mentions.append((node, m))

    if not transcript_person_mentions:
        return {}

    visual_mentions: List[Tuple[Node, Mention]] = []
    for node in extraction_result.nodes:
        if not _caption_looks_like_subject(node):
            continue
        for m in node.mentions:
            if m.modality != _TRANSCRIPT_MODALITY:
                visual_mentions.append((node, m))

    out: Dict[Tuple[float, float], Node] = {}
    for _, vm in visual_mentions:
        window_start = vm.ts_start - window_s
        window_end = vm.ts_end + window_s
        window_counts: Dict[str, int] = {}
        window_nodes: Dict[str, Node] = {}
        for person_node, pm in transcript_person_mentions:
            if pm.ts_end < window_start or pm.ts_start > window_end:
                continue
            window_counts[person_node.name] = window_counts.get(person_node.name, 0) + 1
            window_nodes[person_node.name] = person_node
        if not window_counts:
            continue
        top_name = max(window_counts, key=window_counts.get)
        out[(vm.ts_start, vm.ts_end)] = window_nodes[top_name]
    return out


class CrossModalLinker:
    """Emit ``same_as`` edges via structural-inference primitives only.

    No similarity scoring. No encoder dependency. The three primitives
    (shared-token coreference, video-subject inference, per-window
    subject inference) are documented at module level.
    """

    def __init__(
        self,
        temporal_window_s: float = DEFAULT_WINDOW_S,
        subject_dominance_threshold: float = SUBJECT_DOMINANCE_THRESHOLD,
    ) -> None:
        self._temporal_window_s = float(temporal_window_s)
        self._subject_dominance_threshold = float(subject_dominance_threshold)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def link(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """Return a new ExtractionResult with cross-modal ``same_as`` edges.

        Preserves all original nodes and edges; appends inferred edges.
        Duplicates by ``edge_id`` are filtered, so re-running the linker
        on its own output is a no-op.
        """
        nodes = list(extraction_result.nodes)
        edges = list(extraction_result.edges)
        existing_edge_ids = {edge.edge_id for edge in edges}
        tenant_id = nodes[0].tenant_id if nodes else ""

        new_edges: List[Edge] = []

        # ── Shared-name-token coreference ─────────────────────────────────
        new_edges.extend(
            self._link_by_shared_token(extraction_result, tenant_id, existing_edge_ids)
        )

        # ── Video-subject inference (single-subject content) ─────────────
        subject = infer_video_subject(
            extraction_result, threshold=self._subject_dominance_threshold
        )
        if subject is not None:
            new_edges.extend(
                self._link_by_video_subject(
                    extraction_result,
                    subject,
                    tenant_id,
                    existing_edge_ids,
                )
            )
        else:
            # ── Per-window subject inference (multi-subject fallback) ────
            window_subjects = infer_subject_per_window(
                extraction_result, window_s=self._temporal_window_s
            )
            new_edges.extend(
                self._link_by_window_subjects(
                    extraction_result,
                    window_subjects,
                    tenant_id,
                    existing_edge_ids,
                )
            )

        return ExtractionResult(
            source_doc_id=extraction_result.source_doc_id,
            nodes=nodes,
            edges=edges + new_edges,
            file_sha256=extraction_result.file_sha256,
        )

    # ------------------------------------------------------------------ #
    # Primitive 1 — shared-name-token coreference                        #
    # ------------------------------------------------------------------ #

    def _link_by_shared_token(
        self,
        extraction_result: ExtractionResult,
        tenant_id: str,
        existing_edge_ids: set,
    ) -> List[Edge]:
        """Pair cross-modal mentions whose Node names share a substantive
        token. Emits one ``same_as`` edge per qualifying pair, anchored
        on the lexicographically-first side's mention for determinism.
        """
        nodes = list(extraction_result.nodes)
        # Sort for deterministic pair enumeration.
        nodes_sorted = sorted(nodes, key=lambda n: n.name)
        out: List[Edge] = []
        for i, node_a in enumerate(nodes_sorted):
            for node_b in nodes_sorted[i + 1 :]:
                token = _shared_substantive_token(node_a.name, node_b.name)
                if token is None:
                    continue
                # Find one cross-modal mention pair to anchor the edge.
                for ma in node_a.mentions:
                    for mb in node_b.mentions:
                        if ma.modality == mb.modality:
                            continue
                        edge = Edge(
                            tenant_id=tenant_id,
                            source=node_a.name,
                            target=node_b.name,
                            relation="same_as",
                            evidence_span=f"shared_token:{token}",
                            segment_id=ma.segment_id,
                            ts_start=ma.ts_start,
                            ts_end=ma.ts_end,
                            modality=ma.modality,
                            provenance="INFERRED",
                            source_doc_id=ma.source_doc_id,
                            confidence=1.0,
                        )
                        if edge.edge_id in existing_edge_ids:
                            continue
                        existing_edge_ids.add(edge.edge_id)
                        out.append(edge)
                        break  # one anchor per (node_a, node_b) is enough
                    else:
                        continue
                    break
        return out

    # ------------------------------------------------------------------ #
    # Primitive 2 — video-subject inference (single-subject content)      #
    # ------------------------------------------------------------------ #

    def _link_by_video_subject(
        self,
        extraction_result: ExtractionResult,
        subject: Node,
        tenant_id: str,
        existing_edge_ids: set,
    ) -> List[Edge]:
        """For every generic VLM/OCR mention in this video, emit a
        same_as edge to the dominant subject Person.
        """
        confidence = self._subject_dominance_for(extraction_result, subject)
        out: List[Edge] = []
        for node in extraction_result.nodes:
            if not _caption_looks_like_subject(node):
                continue
            for m in node.mentions:
                if m.modality == _TRANSCRIPT_MODALITY:
                    continue
                edge = Edge(
                    tenant_id=tenant_id,
                    source=node.name,
                    target=subject.name,
                    relation="same_as",
                    evidence_span="video_subject_inference",
                    segment_id=m.segment_id,
                    ts_start=m.ts_start,
                    ts_end=m.ts_end,
                    modality=m.modality,
                    provenance="video_subject_inference",
                    source_doc_id=m.source_doc_id,
                    confidence=confidence,
                )
                if edge.edge_id in existing_edge_ids:
                    continue
                existing_edge_ids.add(edge.edge_id)
                out.append(edge)
        return out

    @staticmethod
    def _subject_dominance_for(
        extraction_result: ExtractionResult, subject: Node
    ) -> float:
        """Share of transcript Person-mentions held by ``subject``."""
        subject_count = sum(
            1 for m in subject.mentions if m.modality == _TRANSCRIPT_MODALITY
        )
        total = 0
        for node in extraction_result.nodes:
            if (node.label or "").strip() != "Person":
                continue
            total += sum(1 for m in node.mentions if m.modality == _TRANSCRIPT_MODALITY)
        if total == 0:
            return 0.0
        return round(subject_count / total, 4)

    # ------------------------------------------------------------------ #
    # Primitive 3 — per-window subject inference (multi-subject content)  #
    # ------------------------------------------------------------------ #

    def _link_by_window_subjects(
        self,
        extraction_result: ExtractionResult,
        window_subjects: Dict[Tuple[float, float], Node],
        tenant_id: str,
        existing_edge_ids: set,
    ) -> List[Edge]:
        """Attribute each generic VLM/OCR mention to its per-window
        dominant Person. Confidence is the per-window dominance share.
        """
        if not window_subjects:
            return []
        out: List[Edge] = []
        for node in extraction_result.nodes:
            if not _caption_looks_like_subject(node):
                continue
            for m in node.mentions:
                if m.modality == _TRANSCRIPT_MODALITY:
                    continue
                subject = window_subjects.get((m.ts_start, m.ts_end))
                if subject is None:
                    continue
                confidence = self._window_dominance_for(
                    extraction_result, subject, m, self._temporal_window_s
                )
                edge = Edge(
                    tenant_id=tenant_id,
                    source=node.name,
                    target=subject.name,
                    relation="same_as",
                    evidence_span="video_subject_inference",
                    segment_id=m.segment_id,
                    ts_start=m.ts_start,
                    ts_end=m.ts_end,
                    modality=m.modality,
                    provenance="video_subject_inference",
                    source_doc_id=m.source_doc_id,
                    confidence=confidence,
                )
                if edge.edge_id in existing_edge_ids:
                    continue
                existing_edge_ids.add(edge.edge_id)
                out.append(edge)
        return out

    @staticmethod
    def _window_dominance_for(
        extraction_result: ExtractionResult,
        subject: Node,
        mention: Mention,
        window_s: float,
    ) -> float:
        """Share of in-window transcript Person-mentions held by ``subject``."""
        window_start = mention.ts_start - window_s
        window_end = mention.ts_end + window_s
        subject_count = 0
        total = 0
        for node in extraction_result.nodes:
            if (node.label or "").strip() != "Person":
                continue
            for m in node.mentions:
                if m.modality != _TRANSCRIPT_MODALITY:
                    continue
                if m.ts_end < window_start or m.ts_start > window_end:
                    continue
                total += 1
                if node.name == subject.name:
                    subject_count += 1
        if total == 0:
            return 0.0
        return round(subject_count / total, 4)
