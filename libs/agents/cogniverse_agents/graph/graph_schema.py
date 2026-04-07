"""Unified Node and Edge dataclasses for the knowledge graph."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
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
class Node:
    """A graph node — a concept or entity extracted from source content."""

    tenant_id: str
    name: str
    description: str = ""
    kind: str = "concept"  # "concept" | "entity"
    mentions: List[str] = field(default_factory=list)
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
                "mentions": json.dumps(self.mentions),
                "degree": self.degree,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        }


@dataclass
class Edge:
    """A directed graph edge between two nodes."""

    tenant_id: str
    source: str  # node name (will be normalized to node_id)
    target: str  # node name (will be normalized to node_id)
    relation: str
    provenance: str = "INFERRED"  # "EXTRACTED" | "INFERRED"
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
        """Deterministic edge id — same (source, relation, target) produces same id."""
        raw = f"{self.source_node_id}|{self.relation}|{self.target_node_id}"
        hashed = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return hashed

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
                "provenance": self.provenance,
                "source_doc_id": self.source_doc_id,
                "confidence": self.confidence,
                "created_at": self.created_at,
            }
        }


@dataclass
class ExtractionResult:
    """Nodes and edges extracted from a single file."""

    source_doc_id: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    file_sha256: Optional[str] = None
