"""Cross-modal entity linker.

Run once per ``source_doc_id`` after all per-segment extractions complete.
For each pair of ``Mention`` objects on *different* modalities whose
temporal windows overlap within ``±temporal_window_s``, encode
``entity_name + " " + evidence_span`` for each side via the
``colbert_pylate`` sidecar (multi-vector ColBERT), compute MaxSim
cosine, and emit an ``Edge(relation="same_as", provenance="INFERRED")``
when the score exceeds ``cosine_threshold``.

The linker is deterministic: it sorts mentions by
``(source_doc_id, modality, segment_id, ts_start)`` before pairing, so a
repeat call on the same input produces identical ordering. Duplicate
``same_as`` edges (by ``edge_id``) are skipped so re-running the linker
on its own output is a no-op.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
)
from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

logger = logging.getLogger(__name__)

# Hard cap so a single noisy edge_span doesn't blow up the encode payload.
_MAX_TEXT_CHARS = 512

# Labels that describe a real-world subject the visual-VLM modality may
# also describe — a Person-named transcript mention can co-refer with a
# Concept/Location/Organization VLM caption when the caption contains a
# person-shaped indicator word OR shares a name token with the Person.
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

# Generic words a VLM caption uses to describe a human subject — when one
# of these appears on the non-Person side of a Person/Peer pair, the
# pair is allowed through the type gate even if no name token is shared.
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


def _tokenize_for_match(name: str) -> set:
    """Lowercased alphanumeric tokens of length >= 3 from a name string."""
    import re as _re

    return {t.lower() for t in _re.findall(r"[A-Za-z0-9]+", name or "") if len(t) >= 3}


def _accepts_type_gate(label_a: str, name_a: str, label_b: str, name_b: str) -> bool:
    """Return True iff the two sides may be considered ``same_as`` candidates.

    Type-gated linking — implements Option 2 from
    ``docs/knowledge/CROSS_MODAL_PRECISION_TODO.md``:

      * ``Person`` <-> ``Person``: accept (entity ↔ entity).
      * ``Person`` <-> peer (``Concept``/``Location``/``Organization``/...):
        accept when the non-Person side either shares a name token with
        the Person side (length >= 3, case-insensitive) OR contains a
        generic person-indicator word ("woman", "scientist", ...).
      * Otherwise: reject — including ``Concept`` <-> ``Concept`` pairs,
        which the ColBERT-LateOn cosine cannot reliably discriminate.

    The reject path is what stops false positives like Person
    "Marie Curie" being fused with Concept "yellow flowers in glass
    vase" (no shared token, no person word) and stops the bogus
    Concept-↔-Concept third edge in triangle scenarios where neither
    side carries a Person tag.
    """
    a_label = (label_a or "").strip()
    b_label = (label_b or "").strip()

    if a_label == "Person" and b_label == "Person":
        return True

    if a_label == "Person" and b_label in _PERSON_LIKE_PEER_LABELS:
        person_tokens = _tokenize_for_match(name_a)
        peer_tokens = _tokenize_for_match(name_b)
        if person_tokens & peer_tokens:
            return True
        if peer_tokens & _PERSON_INDICATOR_WORDS:
            return True
        return False

    if b_label == "Person" and a_label in _PERSON_LIKE_PEER_LABELS:
        person_tokens = _tokenize_for_match(name_b)
        peer_tokens = _tokenize_for_match(name_a)
        if person_tokens & peer_tokens:
            return True
        if peer_tokens & _PERSON_INDICATOR_WORDS:
            return True
        return False

    return False


class CrossModalLinker:
    """Emit ``same_as`` edges between co-temporal cross-modal mentions."""

    def __init__(
        self,
        colbert_endpoint_url: str,
        temporal_window_s: float = 5.0,
        cosine_threshold: float = 0.6,
        colbert_model: str = "lightonai/LateOn",
    ) -> None:
        if not colbert_endpoint_url:
            raise ValueError(
                "CrossModalLinker requires a colbert_endpoint_url — same_as "
                "linking uses the colbert_pylate sidecar for multi-vector "
                "encoding."
            )
        self._temporal_window_s = float(temporal_window_s)
        self._cosine_threshold = float(cosine_threshold)
        loader = RemoteColBERTLoader(
            model_name=colbert_model,
            config={"remote_inference_url": colbert_endpoint_url},
            logger=logger,
        )
        self._encoder, _ = loader.load_model()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def link(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """Return a new ExtractionResult with cross-modal ``same_as`` edges.

        The returned result preserves all original nodes and edges and
        appends inferred edges. No duplicates by ``edge_id``.
        """
        nodes = list(extraction_result.nodes)
        edges = list(extraction_result.edges)
        existing_edge_ids = {edge.edge_id for edge in edges}

        node_by_name = {node.name: node for node in nodes}

        # Build (name, mention) pairs grouped by source_doc_id then modality.
        # A single node can carry multiple mentions and contribute to many
        # cross-modal candidates.
        per_doc: Dict[str, Dict[str, List[Tuple[Node, Mention]]]] = {}
        for node in nodes:
            for mention in node.mentions:
                doc_bucket = per_doc.setdefault(mention.source_doc_id, {})
                modality_bucket = doc_bucket.setdefault(mention.modality, [])
                modality_bucket.append((node, mention))

        new_edges: List[Edge] = []

        for source_doc_id, modality_buckets in per_doc.items():
            modalities = sorted(modality_buckets.keys())
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    side_a = sorted(
                        modality_buckets[modalities[i]],
                        key=lambda nm: (nm[1].segment_id, nm[1].ts_start),
                    )
                    side_b = sorted(
                        modality_buckets[modalities[j]],
                        key=lambda nm: (nm[1].segment_id, nm[1].ts_start),
                    )
                    candidates = self._collect_temporal_pairs(side_a, side_b)
                    new_edges.extend(
                        self._score_and_emit(
                            candidates,
                            tenant_id=self._tenant_id_for(nodes),
                            node_by_name=node_by_name,
                            existing_edge_ids=existing_edge_ids,
                            source_doc_id=source_doc_id,
                        )
                    )

        return ExtractionResult(
            source_doc_id=extraction_result.source_doc_id,
            nodes=nodes,
            edges=edges + new_edges,
            file_sha256=extraction_result.file_sha256,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tenant_id_for(nodes: List[Node]) -> str:
        """All nodes in an ExtractionResult share a tenant_id."""
        if not nodes:
            return ""
        return nodes[0].tenant_id

    def _collect_temporal_pairs(
        self,
        side_a: List[Tuple[Node, Mention]],
        side_b: List[Tuple[Node, Mention]],
    ) -> List[Tuple[Tuple[Node, Mention], Tuple[Node, Mention]]]:
        """Return all (a, b) pairs whose mention windows overlap within ±window."""
        pairs: List[Tuple[Tuple[Node, Mention], Tuple[Node, Mention]]] = []
        window = self._temporal_window_s
        for node_a, mention_a in side_a:
            for node_b, mention_b in side_b:
                if self._windows_overlap(mention_a, mention_b, window):
                    pairs.append(((node_a, mention_a), (node_b, mention_b)))
        return pairs

    @staticmethod
    def _windows_overlap(a: Mention, b: Mention, window: float) -> bool:
        """True if [a.ts_start - w, a.ts_end + w] intersects [b.ts_start, b.ts_end]."""
        return (a.ts_start - window) <= b.ts_end and (b.ts_start - window) <= a.ts_end

    def _score_and_emit(
        self,
        candidates: List[Tuple[Tuple[Node, Mention], Tuple[Node, Mention]]],
        tenant_id: str,
        node_by_name: Dict[str, Node],
        existing_edge_ids: set,
        source_doc_id: str,
    ) -> List[Edge]:
        # Type-gate first — pairs whose labels are incompatible never
        # touch the encoder. ColBERT-LateOn's cosine is too lexical to
        # discriminate Person↔unrelated-Concept; gating up front keeps
        # the encode payload small AND prevents false positives no
        # threshold could filter post-hoc.
        candidates = [
            ((node_a, mention_a), (node_b, mention_b))
            for (node_a, mention_a), (node_b, mention_b) in candidates
            if _accepts_type_gate(node_a.label, node_a.name, node_b.label, node_b.name)
        ]
        if not candidates:
            return []

        # Build the encode payload deterministically.
        texts: List[str] = []
        for (node_a, mention_a), (node_b, mention_b) in candidates:
            texts.append(self._format_for_encoding(node_a.name, mention_a))
            texts.append(self._format_for_encoding(node_b.name, mention_b))

        encodings = self._encoder.encode(texts, is_query=False)

        out: List[Edge] = []
        for idx, ((node_a, mention_a), (node_b, mention_b)) in enumerate(candidates):
            vec_a = np.asarray(encodings[2 * idx], dtype=np.float32)
            vec_b = np.asarray(encodings[2 * idx + 1], dtype=np.float32)
            cosine = self._maxsim_cosine(vec_a, vec_b)
            if cosine <= self._cosine_threshold:
                continue

            # Anchor the inferred edge on side_a's mention (deterministic
            # because candidates were built from sorted sides).
            edge = Edge(
                tenant_id=tenant_id,
                source=node_a.name,
                target=node_b.name,
                relation="same_as",
                evidence_span="cross_modal_temporal",
                segment_id=mention_a.segment_id,
                ts_start=mention_a.ts_start,
                ts_end=mention_a.ts_end,
                modality=mention_a.modality,
                provenance="INFERRED",
                source_doc_id=source_doc_id,
                confidence=float(cosine),
            )
            if edge.edge_id in existing_edge_ids:
                continue
            existing_edge_ids.add(edge.edge_id)
            out.append(edge)
        return out

    @staticmethod
    def _format_for_encoding(entity_name: str, mention: Mention) -> str:
        """Build the encode input: entity name + verbatim evidence span."""
        merged = f"{entity_name} {mention.evidence_span}".strip()
        if len(merged) > _MAX_TEXT_CHARS:
            return merged[:_MAX_TEXT_CHARS]
        return merged

    @staticmethod
    def _maxsim_cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Mean over rows in A of the max cosine to any row in B.

        Deterministic given identical inputs (no random sampling, no
        floating-point reductions whose order depends on input length).
        """
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
            return 0.0

        a_norms = np.linalg.norm(a, axis=1, keepdims=True)
        b_norms = np.linalg.norm(b, axis=1, keepdims=True)
        # Guard against zero-norm rows producing NaN; treat them as zero
        # contribution rather than crashing.
        a_safe = np.where(a_norms == 0.0, 1.0, a_norms)
        b_safe = np.where(b_norms == 0.0, 1.0, b_norms)
        a_normalized = a / a_safe
        b_normalized = b / b_safe

        # (N, M) cosine matrix; a row is zero if the source vector was zero
        # because the corresponding row in `a` is the zero vector.
        sim = a_normalized @ b_normalized.T
        per_row_max = sim.max(axis=1)
        return float(per_row_max.mean())
