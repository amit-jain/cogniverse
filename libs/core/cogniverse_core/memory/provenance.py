"""Provenance + citation graph data model and walker.

Every knowledge write carries a ``Provenance`` record describing where the
content came from: the writing agent, the time, the source citations
(``derived_from``), the derivation kind, a confidence score, and an optional
trace id. The schema's ``provenance_required`` flag gates writes that
omit provenance.

Provenance is persisted in a per-tenant ``provenance`` Vespa schema (see
``provenance_store.ProvenanceStore``) so citation chains can be walked
with one Vespa query per BFS level instead of one Mem0 fetch per node.
A copy is also written into the memory's own metadata under the
``provenance`` key for direct inspection of a single memory dict.

The ``ProvenanceWalker`` is the read API: walk a knowledge node back to its
primary sources, return the full chain plus a structured graph view for
downstream auditing or citation rendering. The walker requires the
indexed store; without it, ``walk()`` raises.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_core.memory.manager import Mem0MemoryManager

logger = logging.getLogger(__name__)


class DerivationKind(str, Enum):
    """How the memory came into existence.

    * ``direct_ingest`` — the content is the source itself (a doc, a fact
      asserted by a human). ``derived_from`` may reference an external URL.
    * ``synthesis`` — composed from multiple memories (citation chain shows
      the inputs).
    * ``summarization`` — distilled from one or more memories.
    * ``extraction`` — pulled out of a larger memory (entity from a doc).
    * ``user_assert`` — a user-supplied fact, possibly without external source.
    * ``agent_inference`` — agent reasoned a new claim from memories.
    """

    DIRECT_INGEST = "direct_ingest"
    SYNTHESIS = "synthesis"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    USER_ASSERT = "user_assert"
    AGENT_INFERENCE = "agent_inference"


@dataclass(frozen=True)
class CitationRef:
    """Reference to a single source.

    The ``ref_kind`` distinguishes between a memory in this tenant's store
    vs. an external URL or document id. Walkers stop following at non-memory
    refs (they're terminal sources).
    """

    ref_kind: str  # "memory" | "external_doc" | "url" | "trace" | other
    ref_id: str
    label: Optional[str] = None  # human-readable annotation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CitationRef":
        return cls(
            ref_kind=str(d["ref_kind"]),
            ref_id=str(d["ref_id"]),
            label=d.get("label"),
        )

    @classmethod
    def memory(cls, memory_id: str, label: Optional[str] = None) -> "CitationRef":
        return cls(ref_kind="memory", ref_id=memory_id, label=label)

    @classmethod
    def external(cls, url: str, label: Optional[str] = None) -> "CitationRef":
        return cls(ref_kind="url", ref_id=url, label=label)


@dataclass(frozen=True)
class Provenance:
    """Provenance record attached to every memory write.

    Fields are intentionally compact so the JSON-encoded record stays small
    (the citation chain may itself be long; we don't want each node to
    bloat). Add fields conservatively.
    """

    written_by: str  # "agent:<name>" or "user:<id>"
    written_at: str  # ISO-8601 UTC
    derivation_kind: DerivationKind
    confidence: float  # 0.0–1.0
    derived_from: List[CitationRef] = field(default_factory=list)
    trace_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0]; got {self.confidence}")
        if not self.written_by:
            raise ValueError("written_by must be a non-empty string")
        if not self.written_at:
            raise ValueError("written_at must be a non-empty ISO-8601 string")

    def to_metadata_payload(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict suitable for memory metadata."""
        return {
            "written_by": self.written_by,
            "written_at": self.written_at,
            "derivation_kind": self.derivation_kind.value,
            "confidence": self.confidence,
            "derived_from": [r.to_dict() for r in self.derived_from],
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_metadata_payload(cls, payload: Dict[str, Any]) -> "Provenance":
        return cls(
            written_by=str(payload["written_by"]),
            written_at=str(payload["written_at"]),
            derivation_kind=DerivationKind(payload["derivation_kind"]),
            confidence=float(payload["confidence"]),
            derived_from=[
                CitationRef.from_dict(r) for r in payload.get("derived_from") or []
            ],
            trace_id=payload.get("trace_id"),
        )


def attach_to_metadata(
    metadata: Optional[Dict[str, Any]],
    provenance: Provenance,
) -> Dict[str, Any]:
    """Return a new metadata dict with the provenance payload merged in.

    Stored under the ``provenance`` key as a nested dict (NOT JSON-encoded,
    so downstream code can read it without parsing). The Vespa backend's
    ``metadata_`` field will JSON-encode the whole metadata dict on write.
    """
    out = dict(metadata or {})
    out["provenance"] = provenance.to_metadata_payload()
    return out


def extract_from_memory(memory: Dict[str, Any]) -> Optional[Provenance]:
    """Extract a ``Provenance`` from a memory dict, or ``None`` if absent.

    Tolerates both shapes: metadata as a dict (in-memory tests) and metadata
    as a JSON-encoded string (Vespa round-trip).
    """
    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return None
    if not isinstance(meta, dict):
        return None
    payload = meta.get("provenance")
    if not isinstance(payload, dict):
        return None
    try:
        return Provenance.from_metadata_payload(payload)
    except (KeyError, ValueError) as exc:
        logger.debug("Malformed provenance on memory %s: %s", memory.get("id"), exc)
        return None


def make_provenance(
    *,
    written_by: str,
    derivation_kind: DerivationKind,
    confidence: float,
    derived_from: Optional[List[CitationRef]] = None,
    trace_id: Optional[str] = None,
) -> Provenance:
    """Convenience constructor: stamps ``written_at`` to now (UTC, ISO-8601)."""
    return Provenance(
        written_by=written_by,
        written_at=datetime.now(timezone.utc).isoformat(),
        derivation_kind=derivation_kind,
        confidence=confidence,
        derived_from=derived_from or [],
        trace_id=trace_id,
    )


@dataclass
class CitationNode:
    """A single node in a walked citation chain."""

    memory_id: str
    provenance: Optional[Provenance]
    content_excerpt: str  # first ~200 chars; full content via re-fetch
    depth: int  # 0 = the original node walked from


@dataclass
class CitationGraph:
    """Result of walking a memory's citation chain.

    ``primary_sources`` lists the leaf citations (memory refs whose target
    has no further provenance, plus all non-memory refs). Useful for "show
    the user where this answer came from" rendering.
    """

    root_memory_id: str
    nodes: List[CitationNode]
    primary_sources: List[CitationRef]
    truncated_at_max_depth: bool = False


class ProvenanceWalker:
    """Walks the citation graph backwards from a root memory.

    Walking is performed against the indexed
    :class:`provenance_store.ProvenanceStore` (one Vespa query per BFS
    level). Memory content is fetched per visited node so each
    ``CitationNode`` carries an excerpt; the indexed store does the
    graph traversal in constant queries per level.

    Args:
        memory_manager: Live Mem0 manager for the tenant. Must have its
            ``provenance_store`` initialised (it is, by default, on any
            manager constructed via ``Mem0MemoryManager.initialize``).
        max_depth: Stop walking past this depth (cycle / runaway protection).
        max_nodes: Stop walking after visiting this many nodes total.

    Raises:
        ValueError: When ``memory_manager.provenance_store`` is missing —
            the walker has no fallback path. Operators must run with a
            backend that has the per-tenant ``provenance`` schema deployed.
    """

    def __init__(
        self,
        memory_manager: "Mem0MemoryManager",
        max_depth: int = 10,
        max_nodes: int = 100,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        store = getattr(memory_manager, "provenance_store", None)
        if store is None:
            raise ValueError(
                "ProvenanceWalker requires memory_manager.provenance_store; "
                "the per-tenant provenance Vespa schema is missing or the "
                "manager was constructed without a backend"
            )
        self._mm = memory_manager
        self._store = store
        self._max_depth = max_depth
        self._max_nodes = max_nodes

    def walk(self, root_memory_id: str, tenant_id: str) -> CitationGraph:
        """Build a ``CitationGraph`` for ``root_memory_id``.

        Delegates the BFS traversal to the indexed
        :class:`ProvenanceStore`, then fetches each visited memory's
        content for the per-node excerpt. ``tenant_id`` is accepted for
        signature compatibility with older callers but the store is
        already tenant-scoped.
        """
        del tenant_id  # tenant scoping lives on the store
        ordered, primary_sources, truncated, records_by_id = self._store.walk(
            root_memory_id,
            max_depth=self._max_depth,
            max_nodes=self._max_nodes,
        )

        nodes: List[CitationNode] = []
        for memory_id, depth in ordered:
            memory = self._fetch_memory(memory_id)
            if memory is None:
                content_excerpt = ""
            else:
                content = memory.get("memory") or memory.get("content") or ""
                content_excerpt = str(content)[:200]

            rec = records_by_id.get(memory_id)
            prov = rec.to_provenance() if rec is not None else None

            nodes.append(
                CitationNode(
                    memory_id=memory_id,
                    provenance=prov,
                    content_excerpt=content_excerpt,
                    depth=depth,
                )
            )

        return CitationGraph(
            root_memory_id=root_memory_id,
            nodes=nodes,
            primary_sources=primary_sources,
            truncated_at_max_depth=truncated,
        )

    def _fetch_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a memory by id for excerpt rendering; None when missing."""
        try:
            mem_obj = self._mm.memory.get(memory_id)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("ProvenanceWalker: get(%s) failed: %r", memory_id, exc)
            return None
        if mem_obj is None:
            return None
        if isinstance(mem_obj, dict):
            return mem_obj
        if isinstance(mem_obj, list) and mem_obj:
            return mem_obj[0] if isinstance(mem_obj[0], dict) else None
        return None
