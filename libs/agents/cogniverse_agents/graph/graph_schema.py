"""Unified Node, Edge, and Mention dataclasses for the knowledge graph."""

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_tenant(tenant_id: str) -> str:
    return tenant_id.replace(":", "_")


def normalize_name(name: str) -> str:
    """Stable node identifier from a name string.

    Lowercases, decomposes unicode, replaces non-alphanumerics with
    underscores. Two extractors that find "SearchAgent" and "searchagent"
    in different files both map to the same node_id so they're merged.
    """
    normalized = unicodedata.normalize("NFD", name)
    ascii_str = normalized.encode("ascii", errors="ignore").decode("ascii")
    slugged = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_str.strip())
    return slugged.strip("_").lower()


@dataclass
class Mention:
    """A single grounded occurrence of a node within a source segment.

    Carries the temporal/positional anchor so KG consumers (CitationTracing,
    KnowledgeGraphTraversal, TemporalReasoning, etc.) can surface where
    in a video/document/code-file the entity was extracted from.
    """

    source_doc_id: str
    segment_id: str
    ts_start: float
    ts_end: float
    modality: str
    evidence_span: str


@dataclass
class Node:
    """A graph node — a concept or entity extracted from source content.

    ``label`` carries the entity-type tag from the upstream extractor
    (GLiNER labels: ``Person``, ``Location``, ``Organization``,
    ``Substance``, ``Concept``, ...). Cross-modal linkers consult this
    tag to gate ``same_as`` candidates so two mentions of incompatible
    types do not get fused on a borderline cosine score. Default
    ``"Concept"`` matches the doc-extractor fallback path.
    """

    tenant_id: str
    name: str
    mentions: List[Mention]
    description: str = ""
    kind: str = "concept"  # "concept" | "entity"
    label: str = "Concept"  # GLiNER tag (Person, Location, Substance, ...)
    degree: int = 0
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    @property
    def node_id(self) -> str:
        return normalize_name(self.name)

    @property
    def doc_id(self) -> str:
        return f"kg_node_{_safe_tenant(self.tenant_id)}_{self.node_id}"

    def to_vespa_document(self) -> dict:
        return {
            "fields": {
                "doc_id": self.doc_id,
                "tenant_id": self.tenant_id,
                "doc_type": "node",
                "name": self.name,
                "description": self.description,
                "kind": self.kind,
                "label": self.label,
                "mentions": json.dumps([asdict(m) for m in self.mentions]),
                "degree": self.degree,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        }


@dataclass
class Edge:
    """A directed graph edge between two nodes, grounded in a specific segment."""

    tenant_id: str
    source: str  # node name (will be normalized to node_id)
    target: str  # node name (will be normalized to node_id)
    relation: str
    evidence_span: str
    segment_id: str
    ts_start: float
    ts_end: float
    modality: str
    provenance: str = "EXTRACTED"  # "EXTRACTED" | "INFERRED"
    source_doc_id: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=_utcnow_iso)

    @property
    def source_node_id(self) -> str:
        return normalize_name(self.source)

    @property
    def target_node_id(self) -> str:
        return normalize_name(self.target)

    @property
    def edge_id(self) -> str:
        """Deterministic edge id — same (source, relation, target, segment) → same id."""
        raw = (
            f"{self.source_node_id}|{self.relation}|{self.target_node_id}"
            f"|{self.segment_id}|{self.ts_start}|{self.ts_end}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def doc_id(self) -> str:
        return f"kg_edge_{_safe_tenant(self.tenant_id)}_{self.edge_id}"

    def to_vespa_document(self) -> dict:
        return {
            "fields": {
                "doc_id": self.doc_id,
                "tenant_id": self.tenant_id,
                "doc_type": "edge",
                "source_node_id": self.source_node_id,
                "target_node_id": self.target_node_id,
                "relation": self.relation,
                "evidence_span": self.evidence_span,
                "segment_id": self.segment_id,
                "ts_start": self.ts_start,
                "ts_end": self.ts_end,
                "modality": self.modality,
                "provenance": self.provenance,
                "source_doc_id": self.source_doc_id,
                "confidence": self.confidence,
                "created_at": self.created_at,
            }
        }


@dataclass
class ExtractionResult:
    """Nodes and edges extracted from a single source (file or segment batch)."""

    source_doc_id: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    file_sha256: Optional[str] = None
