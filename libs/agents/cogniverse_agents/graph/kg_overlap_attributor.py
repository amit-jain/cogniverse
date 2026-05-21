"""KG-attribute overlap orphan attribution.

For face clusters that the structural primitives (shared-name-token
coreference, video-subject inference, face-cluster temporal attribution)
left unattributed, score them against each ``Person`` node's one-hop
KG profile bag and emit a ``same_as`` edge when one Person dominates
the score margin.

Reasons about structured-claim tokens (predicate + object) rather than
raw text similarity. This is the *third-chance* path — designed to
catch ambiguous cases where a multi-subject video's transcript doesn't
place a named person near the orphan face but the KG already has rich
profile claims for the right Person from past ingest runs.

Scoring: weighted Jaccard between the orphan's caption-token bag and
each candidate Person's profile bag. The top Person wins iff the
runner-up's score is ``<= top × score_margin`` (default 0.7) — close
runner-ups produce silence rather than guesses.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    FaceCluster,
)

_RELATION = "same_as"
_PROVENANCE = "kg_overlap"
_EVIDENCE_SPAN = "kg_overlap"
_VLM_MODALITY = "vlm"

# Common English stopwords that should never make it into a profile bag
# or a caption-token bag. Length-3 minimum already filters most of the
# noise; this set kills the residue (``the``, ``and``, ...).
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "this",
        "that",
        "these",
        "those",
        "are",
        "was",
        "were",
        "has",
        "had",
        "have",
        "its",
        "his",
        "her",
        "their",
        "our",
        "any",
        "all",
        "but",
        "not",
        "you",
        "your",
        "they",
        "them",
    }
)


def _tokenise(text: str) -> Set[str]:
    """Lowercased alphanumeric tokens of length ≥ 3, minus stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    return {t.lower() for t in tokens if len(t) >= 3 and t.lower() not in _STOPWORDS}


def build_person_profile_bags(
    extraction_result: ExtractionResult,
) -> Dict[str, Set[str]]:
    """Build a token bag per Person node from the result's edges.

    Walks every Edge whose ``source`` matches a Person node's name.
    Predicate (``relation``) and target name both contribute tokens.
    Empty bag is returned for Persons that have zero outgoing edges
    (the caller may decide whether to filter those out).
    """
    person_names = {
        node.name
        for node in extraction_result.nodes
        if (node.label or "").strip() == "Person"
    }
    bags: Dict[str, Set[str]] = {name: set() for name in person_names}
    for edge in extraction_result.edges:
        if edge.source not in bags:
            continue
        bags[edge.source].update(_tokenise(edge.relation))
        bags[edge.source].update(_tokenise(edge.target))
    return bags


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Standard Jaccard score; safe on empty inputs."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _anchor_member(cluster: FaceCluster):
    """Lowest-(ts_start, segment_id) member of the cluster."""
    return min(cluster.members, key=lambda m: (m.ts_start, m.segment_id))


def attribute_orphans_by_kg_overlap(
    orphans: List[Tuple[FaceCluster, Set[str]]],
    candidate_persons: Dict[str, Set[str]],
    *,
    source_doc_id: str,
    tenant_id: str,
    score_margin: float = 0.7,
) -> List[Edge]:
    """Emit ``same_as`` Edges for orphan clusters that map cleanly to a Person.

    Each item in ``orphans`` is a ``(cluster, caption_tokens)`` tuple:
    the caller is responsible for deriving caption tokens from whatever
    source the cluster came from (a VLM caption Node, accumulated
    evidence_span text, OCR output, etc.).

    ``candidate_persons`` is the output of ``build_person_profile_bags``.

    Returns an empty list when there are no orphans, no candidates, no
    Person scores above zero, or no Person passes the margin gate.
    """
    if not orphans or not candidate_persons:
        return []

    edges: List[Edge] = []
    for cluster, caption_tokens in orphans:
        scores: List[Tuple[float, str]] = []
        for person_name, profile_bag in candidate_persons.items():
            score = _jaccard(caption_tokens, profile_bag)
            if score > 0:
                scores.append((score, person_name))
        if not scores:
            continue

        # Sort by (-score, name) so highest wins; alphabetical
        # tiebreak below the dominance gate.
        scores.sort(key=lambda s: (-s[0], s[1]))
        top_score, top_name = scores[0]
        if len(scores) >= 2:
            runner_up_score = scores[1][0]
            if runner_up_score > top_score * score_margin:
                # Runner-up is too close — no decisive winner.
                continue

        anchor = _anchor_member(cluster)
        edges.append(
            Edge(
                tenant_id=tenant_id,
                source=cluster.cluster_id,
                target=top_name,
                relation=_RELATION,
                evidence_span=_EVIDENCE_SPAN,
                segment_id=anchor.segment_id,
                ts_start=anchor.ts_start,
                ts_end=anchor.ts_end,
                modality=_VLM_MODALITY,
                provenance=_PROVENANCE,
                source_doc_id=source_doc_id,
                confidence=round(top_score, 4),
            )
        )
    return edges
