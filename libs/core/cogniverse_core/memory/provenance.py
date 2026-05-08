"""Provenance + citation graph (A.2).

Every knowledge write carries a ``Provenance`` record describing where the
content came from: the writing agent, the time, the source citations
(``derived_from``), the derivation kind, a confidence score, and an optional
trace id. The schema's ``provenance_required`` flag (A.1) gates writes that
omit provenance.

Design decision (V1): provenance lives **in the memory's own metadata** under
the ``provenance`` key, JSON-encoded. A separate Vespa schema for the
citation graph is deferred to A.3/A.4 (contradiction + trust ranking) where
the indexable graph buys real query value. For V1, citation walks happen by
fetching memories by id and following the in-band ``derived_from`` list —
O(N) for chain length N, which is acceptable given typical chains are short
(~3–5 hops).

The ``ProvenanceWalker`` is the read API: walk a knowledge node back to its
primary sources, return the full chain plus a structured graph view for
downstream auditing or citation rendering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

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

    Args:
        memory_manager: Live Mem0 manager for the tenant.
        max_depth: Stop walking past this depth (cycle / runaway protection).
        max_nodes: Stop walking after visiting this many nodes total.
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
        self._mm = memory_manager
        self._max_depth = max_depth
        self._max_nodes = max_nodes

    def walk(self, root_memory_id: str, tenant_id: str) -> CitationGraph:
        """Build a ``CitationGraph`` for ``root_memory_id``.

        Performs a BFS over ``derived_from`` references of kind
        ``memory``. Non-memory refs (URLs, external docs) are surfaced as
        ``primary_sources`` without further traversal. Cycles are detected
        via a visited-set.
        """
        visited: Set[str] = set()
        nodes: List[CitationNode] = []
        primary_sources: List[CitationRef] = []
        truncated = False

        # BFS frontier: list of (memory_id, depth)
        frontier: List[tuple] = [(root_memory_id, 0)]

        while frontier:
            if len(nodes) >= self._max_nodes:
                truncated = True
                break
            memory_id, depth = frontier.pop(0)
            if memory_id in visited:
                continue
            visited.add(memory_id)

            memory = self._fetch_memory(memory_id, tenant_id)
            if memory is None:
                # Memory referenced but not in the store — record as a
                # primary source (the chain terminates at an unknown).
                primary_sources.append(
                    CitationRef.memory(memory_id, label="unknown_or_deleted")
                )
                continue

            prov = extract_from_memory(memory)
            content = memory.get("memory") or memory.get("content") or ""
            excerpt = str(content)[:200]
            nodes.append(
                CitationNode(
                    memory_id=memory_id,
                    provenance=prov,
                    content_excerpt=excerpt,
                    depth=depth,
                )
            )

            if prov is None or not prov.derived_from:
                # No further refs — this node is itself a primary source.
                primary_sources.append(CitationRef.memory(memory_id))
                continue

            if depth + 1 > self._max_depth:
                truncated = True
                # Surface remaining refs as primary sources rather than
                # silently dropping them.
                for ref in prov.derived_from:
                    primary_sources.append(ref)
                continue

            for ref in prov.derived_from:
                if ref.ref_kind != "memory":
                    primary_sources.append(ref)
                    continue
                if ref.ref_id not in visited:
                    frontier.append((ref.ref_id, depth + 1))

        # Dedup primary sources by (kind, id) preserving first-seen order.
        seen_ps: Set[tuple] = set()
        deduped_ps: List[CitationRef] = []
        for r in primary_sources:
            key = (r.ref_kind, r.ref_id)
            if key in seen_ps:
                continue
            seen_ps.add(key)
            deduped_ps.append(r)

        return CitationGraph(
            root_memory_id=root_memory_id,
            nodes=nodes,
            primary_sources=deduped_ps,
            truncated_at_max_depth=truncated,
        )

    def _fetch_memory(self, memory_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a memory by id, tolerating Mem0's varying return shapes."""
        try:
            # Mem0.get returns a dict or None; we go through the manager so
            # tenant scoping stays explicit.
            mem_obj = self._mm.memory.get(memory_id)  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("ProvenanceWalker: get(%s) failed: %s", memory_id, exc)
            return None
        if mem_obj is None:
            return None
        if isinstance(mem_obj, dict):
            return mem_obj
        # Some Mem0 backends return list[result]; take first.
        if isinstance(mem_obj, list) and mem_obj:
            return mem_obj[0] if isinstance(mem_obj[0], dict) else None
        return None
