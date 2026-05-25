"""Indexed Vespa store for memory provenance + citation graph walks.

Each ``add_memory`` write that carries a ``Provenance`` block produces
one document in a per-tenant ``provenance`` Vespa schema. Walking the
citation chain from a root memory issues one Vespa query per BFS level
(``select * from provenance where memory_id in (...)``) instead of one
Mem0 fetch per node. Chain depth N → N round trips, regardless of
fan-out at each level.

In-band provenance in ``metadata.provenance`` is still attached so
existing reads (and tests) that pull provenance out of a memory dict
keep working. The indexed store is the source of truth for graph
traversal.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    Provenance,
)

if TYPE_CHECKING:
    from cogniverse_sdk.interfaces.backend import IngestionBackend

logger = logging.getLogger(__name__)

PROVENANCE_BASE_SCHEMA = "provenance"


@dataclass(frozen=True)
class ProvenanceRecord:
    """One indexed provenance row.

    ``memory_id`` is the memory this provenance describes (the unique
    lookup key for graph walks); ``derived_from_memory_ids`` is the
    list of memory ids this memory cites; ``derived_from_other`` holds
    non-memory citations (URLs, external doc ids) as a JSON-encoded
    list so the schema doesn't need a polymorphic field type.
    """

    memory_id: str
    tenant_id: str
    written_by: str
    written_at_unix: int
    derivation_kind: str
    confidence: float
    derived_from_memory_ids: List[str]
    derived_from_other: List[Dict[str, Any]]
    trace_id: Optional[str] = None

    @classmethod
    def from_provenance(
        cls,
        memory_id: str,
        tenant_id: str,
        provenance: Provenance,
    ) -> "ProvenanceRecord":
        from datetime import datetime

        try:
            written_at_unix = int(
                datetime.fromisoformat(
                    provenance.written_at.replace("Z", "+00:00")
                ).timestamp()
            )
        except (ValueError, AttributeError):
            written_at_unix = int(time.time())

        memory_refs = [r for r in provenance.derived_from if r.ref_kind == "memory"]
        other_refs = [r for r in provenance.derived_from if r.ref_kind != "memory"]

        return cls(
            memory_id=memory_id,
            tenant_id=tenant_id,
            written_by=provenance.written_by,
            written_at_unix=written_at_unix,
            derivation_kind=provenance.derivation_kind.value,
            confidence=provenance.confidence,
            derived_from_memory_ids=[r.ref_id for r in memory_refs],
            derived_from_other=[r.to_dict() for r in other_refs],
            trace_id=provenance.trace_id,
        )

    def to_provenance(self) -> Provenance:
        """Reconstruct a Provenance from the indexed record."""
        from datetime import datetime, timezone

        derived_from = [
            CitationRef(ref_kind="memory", ref_id=mid)
            for mid in self.derived_from_memory_ids
        ] + [CitationRef.from_dict(d) for d in self.derived_from_other]
        return Provenance(
            written_by=self.written_by,
            written_at=datetime.fromtimestamp(
                self.written_at_unix, tz=timezone.utc
            ).isoformat(),
            derivation_kind=DerivationKind(self.derivation_kind),
            confidence=self.confidence,
            derived_from=derived_from,
            trace_id=self.trace_id,
        )


class ProvenanceStore:
    """Per-tenant indexed provenance store backed by Vespa.

    Args:
        backend: Live Vespa backend bound to the tenant.
        tenant_id: Tenant id — written into every record's ``tenant_id``
            field and used to resolve the per-tenant schema name.
        base_schema_name: Defaults to ``provenance``. Combined with the
            backend's tenant scoping to produce ``provenance_<tenant>``.
    """

    def __init__(
        self,
        backend: "IngestionBackend",
        tenant_id: str,
        base_schema_name: str = PROVENANCE_BASE_SCHEMA,
    ) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self._backend = backend
        self._tenant_id = tenant_id
        self._base_schema = base_schema_name

    @property
    def schema_name(self) -> str:
        """Resolve the per-tenant Vespa schema name."""
        get_name = getattr(self._backend, "get_tenant_schema_name", None)
        if callable(get_name):
            return get_name(self._tenant_id, self._base_schema)
        return f"{self._base_schema}_{self._tenant_id}"

    def attach(self, memory_id: str, provenance: Provenance) -> str:
        """Persist a provenance record for ``memory_id``. Returns the row id.

        Idempotent on (memory_id, tenant_id): subsequent writes for the
        same memory overwrite the existing row (Vespa upsert semantics
        on the document id).
        """
        record = ProvenanceRecord.from_provenance(
            memory_id, self._tenant_id, provenance
        )
        row_id = self._row_id(memory_id)
        from cogniverse_sdk.document import Document

        doc = Document(
            id=row_id,
            text_content="",
            metadata={
                "id": row_id,
                "memory_id": record.memory_id,
                "tenant_id": record.tenant_id,
                "written_by": record.written_by,
                "written_at": record.written_at_unix,
                "derivation_kind": record.derivation_kind,
                "confidence": record.confidence,
                "derived_from_ids": record.derived_from_memory_ids,
                "derived_from_other": json.dumps(record.derived_from_other),
                "trace_id": record.trace_id or "",
            },
        )
        self._backend.ingest_documents([doc], schema_name=self._base_schema)
        return row_id

    def fetch(self, memory_ids: List[str]) -> Dict[str, ProvenanceRecord]:
        """Batch-fetch provenance records for a list of memory ids.

        Returns a dict keyed by ``memory_id``. Memories that were
        written without a ``Provenance`` block (legitimate for kinds
        whose schema sets ``provenance_required=False``) simply have no
        record and appear as terminal nodes in any walk.
        """
        if not memory_ids:
            return {}
        # YQL `in` clause on the indexed memory_id attribute.
        quoted = ", ".join(f'"{_escape(mid)}"' for mid in memory_ids)
        yql = (
            f"select * from {self.schema_name} where memory_id in ({quoted}) "
            f'and tenant_id contains "{_escape(self._tenant_id)}" '
            f"limit {max(len(memory_ids), 100)}"
        )
        try:
            rows = self._backend.query_metadata_documents(
                schema=self.schema_name,
                yql=yql,
                hits=max(len(memory_ids), 100),
            )
        except Exception as exc:
            logger.debug("ProvenanceStore.fetch query failed: %s", exc)
            return {}
        out: Dict[str, ProvenanceRecord] = {}
        for row in rows:
            try:
                rec = self._row_to_record(row)
            except Exception as exc:
                logger.debug(
                    "Skipping malformed provenance row id=%s: %s", row.get("id"), exc
                )
                continue
            out[rec.memory_id] = rec
        return out

    def get(self, memory_id: str) -> Optional[ProvenanceRecord]:
        """Single-id convenience wrapper around :meth:`fetch`."""
        return self.fetch([memory_id]).get(memory_id)

    def walk(
        self,
        root_memory_id: str,
        max_depth: int = 10,
        max_nodes: int = 100,
    ) -> Tuple[List[Tuple[str, int]], List[CitationRef], bool]:
        """BFS-walk the citation chain from ``root_memory_id``.

        Each BFS level batches into ONE Vespa query (the whole frontier
        in a single ``memory_id in (...)`` lookup). That's the O(1)-per-level
        property the plan called out: chain depth N → N round trips, vs.
        the previous N×fanout per-memory fetches.

        Returns:
            Tuple of ``(visited_with_depth, primary_sources, truncated)``.
            ``visited_with_depth`` is the BFS-ordered list of
            ``(memory_id, depth)`` for every node reached;
            ``primary_sources`` is the deduplicated list of leaf citation
            refs (terminal memory nodes + every non-memory ref).
        """
        if max_depth < 1 or max_nodes < 1:
            raise ValueError("max_depth and max_nodes must be >= 1")
        visited: Set[str] = set()
        ordered: List[Tuple[str, int]] = []
        primary_sources: List[CitationRef] = []
        truncated = False

        frontier: List[Tuple[str, int]] = [(root_memory_id, 0)]
        while frontier:
            if len(ordered) >= max_nodes:
                truncated = True
                break
            # Slice the level (all entries at the same depth) for batched fetch.
            current_depth = frontier[0][1]
            level_ids: List[str] = []
            level_pairs: List[Tuple[str, int]] = []
            i = 0
            while i < len(frontier) and frontier[i][1] == current_depth:
                mid, depth = frontier[i]
                if mid not in visited:
                    level_ids.append(mid)
                    level_pairs.append((mid, depth))
                i += 1
            frontier = frontier[i:]

            if not level_pairs:
                continue

            records = self.fetch(level_ids)

            for mid, depth in level_pairs:
                if mid in visited:
                    continue
                visited.add(mid)
                ordered.append((mid, depth))
                if len(ordered) >= max_nodes:
                    truncated = True
                    break

                rec = records.get(mid)
                if rec is None:
                    # No indexed provenance for this memory — leaf.
                    primary_sources.append(CitationRef.memory(mid))
                    continue

                # Surface non-memory refs immediately (they're terminal).
                for other in rec.derived_from_other:
                    primary_sources.append(CitationRef.from_dict(other))

                if not rec.derived_from_memory_ids:
                    # No memory children — this node is itself a primary.
                    primary_sources.append(CitationRef.memory(mid))
                    continue

                if depth + 1 > max_depth:
                    truncated = True
                    for child in rec.derived_from_memory_ids:
                        primary_sources.append(CitationRef.memory(child))
                    continue

                for child in rec.derived_from_memory_ids:
                    if child not in visited:
                        frontier.append((child, depth + 1))

        # Deduplicate primary sources by (kind, id) preserving order.
        seen: Set[Tuple[str, str]] = set()
        deduped: List[CitationRef] = []
        for ref in primary_sources:
            key = (ref.ref_kind, ref.ref_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)

        return ordered, deduped, truncated

    def _row_id(self, memory_id: str) -> str:
        # Stable per-(tenant, memory_id) document id so re-attach upserts.
        return f"prov-{self._tenant_id}-{memory_id}"

    def _row_to_record(self, row: Dict[str, Any]) -> ProvenanceRecord:
        derived_other_raw = row.get("derived_from_other") or "[]"
        try:
            derived_other = json.loads(derived_other_raw)
        except (ValueError, TypeError):
            derived_other = []
        derived_ids = row.get("derived_from_ids") or []
        if isinstance(derived_ids, str):
            # Some backends serialize array<string> as a single string.
            try:
                derived_ids = json.loads(derived_ids)
            except (ValueError, TypeError):
                derived_ids = []
        return ProvenanceRecord(
            memory_id=str(row["memory_id"]),
            tenant_id=str(row.get("tenant_id") or self._tenant_id),
            written_by=str(row.get("written_by") or ""),
            written_at_unix=int(row.get("written_at") or 0),
            derivation_kind=str(row.get("derivation_kind") or "direct_ingest"),
            confidence=float(row.get("confidence") or 0.0),
            derived_from_memory_ids=[str(x) for x in derived_ids],
            derived_from_other=[d for d in derived_other if isinstance(d, dict)],
            trace_id=row.get("trace_id") or None,
        )


def _escape(value: str) -> str:
    """Minimal YQL string escape (quotes only)."""
    return str(value).replace('"', '\\"')
