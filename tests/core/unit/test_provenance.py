"""Unit tests for the Provenance + ProvenanceWalker."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    Provenance,
    ProvenanceWalker,
    attach_to_metadata,
    extract_from_memory,
    make_provenance,
)


class TestCitationRef:
    def test_memory_helper(self):
        r = CitationRef.memory("m1", label="seed")
        assert r.ref_kind == "memory"
        assert r.ref_id == "m1"
        assert r.label == "seed"

    def test_external_helper(self):
        r = CitationRef.external("https://en.wikipedia.org/wiki/X")
        assert r.ref_kind == "url"
        assert r.ref_id == "https://en.wikipedia.org/wiki/X"

    def test_to_from_dict_round_trip(self):
        original = CitationRef(ref_kind="memory", ref_id="m1", label="seed")
        rebuilt = CitationRef.from_dict(original.to_dict())
        assert rebuilt == original


@pytest.mark.unit
class TestProvenanceConstruction:
    @pytest.mark.ci_fast
    def test_confidence_in_unit_interval(self):
        with pytest.raises(ValueError):
            Provenance(
                written_by="agent:x",
                written_at="2026-05-08T00:00:00+00:00",
                derivation_kind=DerivationKind.SYNTHESIS,
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            Provenance(
                written_by="agent:x",
                written_at="2026-05-08T00:00:00+00:00",
                derivation_kind=DerivationKind.SYNTHESIS,
                confidence=-0.1,
            )

    def test_written_by_required(self):
        with pytest.raises(ValueError):
            Provenance(
                written_by="",
                written_at="2026-05-08T00:00:00+00:00",
                derivation_kind=DerivationKind.SYNTHESIS,
                confidence=0.5,
            )

    @pytest.mark.ci_fast
    def test_to_from_metadata_payload_round_trip(self):
        original = Provenance(
            written_by="agent:search_agent",
            written_at="2026-05-08T01:23:45+00:00",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.75,
            derived_from=[
                CitationRef.memory("m1", label="alpha"),
                CitationRef.external("https://wiki/x"),
            ],
            trace_id="trace-abc",
        )
        rebuilt = Provenance.from_metadata_payload(original.to_metadata_payload())
        assert rebuilt == original

    def test_make_provenance_stamps_timestamp(self):
        p = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.AGENT_INFERENCE,
            confidence=0.8,
            derived_from=[CitationRef.memory("m1")],
        )
        assert p.written_at  # ISO-8601 string set
        # Round-trip survives.
        rebuilt = Provenance.from_metadata_payload(p.to_metadata_payload())
        assert rebuilt == p


class TestAttachExtract:
    def test_attach_preserves_other_metadata(self):
        meta = {"kind": "entity_fact", "tenant_id": "acme"}
        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external("https://wiki/x")],
        )
        out = attach_to_metadata(meta, prov)
        assert out["kind"] == "entity_fact"
        assert out["tenant_id"] == "acme"
        assert "provenance" in out
        assert out["provenance"]["confidence"] == 0.9

    def test_attach_to_none_metadata(self):
        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
        )
        out = attach_to_metadata(None, prov)
        assert "provenance" in out

    def test_extract_from_memory_dict_metadata(self):
        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.6,
            derived_from=[CitationRef.memory("m1")],
        )
        memory = {"id": "m99", "metadata": attach_to_metadata({}, prov)}
        rebuilt = extract_from_memory(memory)
        assert rebuilt is not None
        assert rebuilt.confidence == 0.6
        assert rebuilt.derived_from[0].ref_id == "m1"

    def test_extract_from_memory_json_string_metadata(self):
        """Vespa round-trips metadata as a JSON-encoded string."""
        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.6,
        )
        meta = attach_to_metadata({}, prov)
        memory = {"id": "m99", "metadata": json.dumps(meta)}
        rebuilt = extract_from_memory(memory)
        assert rebuilt is not None
        assert rebuilt.derivation_kind is DerivationKind.SYNTHESIS

    def test_extract_returns_none_for_missing_provenance(self):
        memory = {"id": "m1", "metadata": {"kind": "x"}}
        assert extract_from_memory(memory) is None

    def test_extract_returns_none_for_malformed_metadata(self):
        memory = {"id": "m1", "metadata": "not valid json"}
        assert extract_from_memory(memory) is None


class _StubProvenanceStore:
    """In-memory ProvenanceStore that walks memories[mid].metadata.provenance."""

    def __init__(self, memories: Dict[str, Dict[str, Any]]):
        self._memories = memories

    def _record(self, mid: str):
        from cogniverse_core.memory.provenance import (
            DerivationKind,
            Provenance,
            primary_provenance_digest,
        )
        from cogniverse_core.memory.provenance_store import ProvenanceRecord

        row = self._memories.get(mid)
        if row is None:
            return None
        prov_meta = (row.get("metadata") or {}).get("provenance")
        if not prov_meta:
            return None
        derived_from = []
        for d in prov_meta.get("derived_from") or []:
            if d.get("ref_kind") == "memory":
                derived_from.append(CitationRef.memory(d["ref_id"]))
            else:
                derived_from.append(CitationRef.external(d["ref_id"]))
        provenance = Provenance(
            written_by=prov_meta["written_by"],
            written_at=prov_meta["written_at"],
            derived_from=derived_from,
            derivation_kind=DerivationKind(prov_meta["derivation_kind"]),
            confidence=prov_meta.get("confidence", 0.5),
            trace_id=prov_meta.get("trace_id"),
        )
        return ProvenanceRecord.from_provenance(
            mid,
            "t1",
            provenance,
            primary_digest=primary_provenance_digest(row, provenance),
        )

    def get(self, memory_id: str):
        return self._record(memory_id)

    def walk(self, root: str, *, max_depth: int = 10, max_nodes: int = 100):
        ordered = []
        visited = set()
        primary = []
        records_by_id = {}
        truncated = False
        frontier = [(root, 0)]
        while frontier:
            mid, depth = frontier.pop(0)
            if mid in visited:
                continue
            if len(ordered) >= max_nodes:
                truncated = True
                break
            visited.add(mid)
            ordered.append((mid, depth))
            rec = self._record(mid)
            if rec is None:
                # Unknown memory_id: surface as a primary memory ref.
                primary.append(CitationRef.memory(mid))
                continue
            records_by_id[mid] = rec
            # External refs surface as primary sources.
            for ref_dict in rec.derived_from_other:
                primary.append(CitationRef.from_dict(ref_dict))
            if not rec.derived_from_memory_ids:
                # Terminal memory node — also a primary source.
                primary.append(CitationRef.memory(mid))
                continue
            if depth >= max_depth:
                truncated = True
                primary.append(CitationRef.memory(mid))
                continue
            for child_id in rec.derived_from_memory_ids:
                frontier.append((child_id, depth + 1))
        return ordered, primary, truncated, records_by_id


class FakeManager:
    """In-memory Mem0 stub for walker tests."""

    def __init__(self, memories: Dict[str, Dict[str, Any]]):
        self.memory = MagicMock()
        # Configure get() to return from the dict.
        self.memory.get.side_effect = lambda mid: memories.get(mid)
        self.provenance_store = _StubProvenanceStore(memories)


def _seed(
    mid: str,
    content: str,
    derived_from: list,
    *,
    derivation_kind: DerivationKind = DerivationKind.SYNTHESIS,
) -> Dict[str, Any]:
    prov = make_provenance(
        written_by="agent:test",
        derivation_kind=derivation_kind,
        confidence=0.7,
        derived_from=derived_from,
    )
    return {
        "id": mid,
        "memory": content,
        "metadata": attach_to_metadata({"kind": "synthesis_fact"}, prov),
    }


@pytest.mark.unit
class TestProvenanceWalker:
    @pytest.mark.ci_fast
    def test_walks_two_level_chain(self):
        memories = {
            "m_root": _seed(
                "m_root",
                "synthesis at root",
                [CitationRef.memory("m_a"), CitationRef.memory("m_b")],
            ),
            "m_a": _seed(
                "m_a",
                "intermediate a",
                [CitationRef.external("https://source/a")],
            ),
            "m_b": {
                # m_b has no provenance — terminal source.
                "id": "m_b",
                "memory": "primary doc b",
                "metadata": {"kind": "external_doc"},
            },
        }
        mm = FakeManager(memories)
        walker = ProvenanceWalker(mm)

        graph = walker.walk("m_root", tenant_id="t1")

        ids_in_chain = [n.memory_id for n in graph.nodes]
        assert "m_root" in ids_in_chain
        assert "m_a" in ids_in_chain
        assert "m_b" in ids_in_chain

        # Primary sources: m_a's external URL + m_b (terminal memory).
        ps_keys = {(r.ref_kind, r.ref_id) for r in graph.primary_sources}
        assert ("url", "https://source/a") in ps_keys
        assert ("memory", "m_b") in ps_keys

    @pytest.mark.ci_fast
    def test_handles_cycles_without_infinite_loop(self):
        memories = {
            "m1": _seed("m1", "a", [CitationRef.memory("m2")]),
            "m2": _seed("m2", "b", [CitationRef.memory("m1")]),  # back-ref
        }
        mm = FakeManager(memories)
        walker = ProvenanceWalker(mm)

        graph = walker.walk("m1", tenant_id="t1")
        # Both nodes visited exactly once despite the cycle.
        assert sorted(n.memory_id for n in graph.nodes) == ["m1", "m2"]

    def test_max_depth_truncates_chain_and_marks_truncated(self):
        # Linear 5-node chain m1 -> m2 -> ... -> m5
        memories = {}
        for i in range(1, 5):
            memories[f"m{i}"] = _seed(
                f"m{i}", f"node {i}", [CitationRef.memory(f"m{i + 1}")]
            )
        memories["m5"] = _seed("m5", "leaf", [])
        mm = FakeManager(memories)
        walker = ProvenanceWalker(mm, max_depth=2)

        graph = walker.walk("m1", tenant_id="t1")
        # Depths visited: 0 (m1), 1 (m2), 2 (m3). m4 / m5 not walked.
        assert graph.truncated_at_max_depth is True
        ids = {n.memory_id for n in graph.nodes}
        assert ids == {"m1", "m2", "m3"}

    def test_max_nodes_caps_traversal(self):
        # Wide fan-out: m_root -> 10 children
        children = [f"c{i}" for i in range(10)]
        memories = {
            "m_root": _seed("m_root", "root", [CitationRef.memory(c) for c in children])
        }
        for c in children:
            memories[c] = _seed(c, f"child {c}", [])
        mm = FakeManager(memories)
        walker = ProvenanceWalker(mm, max_nodes=5)

        graph = walker.walk("m_root", tenant_id="t1")
        assert len(graph.nodes) <= 5
        assert graph.truncated_at_max_depth is True

    def test_unknown_memory_id_listed_as_primary(self):
        """Reference to a deleted/unknown memory becomes a primary source."""
        memories = {
            "m_root": _seed(
                "m_root",
                "x",
                [CitationRef.memory("m_missing")],
            )
        }
        mm = FakeManager(memories)
        walker = ProvenanceWalker(mm)
        graph = walker.walk("m_root", tenant_id="t1")
        keys = {(r.ref_kind, r.ref_id) for r in graph.primary_sources}
        assert ("memory", "m_missing") in keys

    def test_walker_validates_constructor_args(self):
        with pytest.raises(ValueError):
            ProvenanceWalker(FakeManager({}), max_depth=0)
        with pytest.raises(ValueError):
            ProvenanceWalker(FakeManager({}), max_nodes=0)


class TestSchemaIntegration:
    """Provenance objects must satisfy the schema's validate_provenance."""

    def test_provenance_with_derived_from_passes_schema_check(self):
        from cogniverse_core.memory.schema import (
            KnowledgeSchema,
        )

        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.8,
            derived_from=[CitationRef.memory("m1")],
        )
        schema = KnowledgeSchema(kind="entity_fact", provenance_required=True)
        # Must NOT raise.
        schema.validate_provenance(prov)

    def test_provenance_with_no_derived_from_fails_schema_check(self):
        from cogniverse_core.memory.schema import (
            KnowledgeSchema,
            SchemaViolationError,
        )

        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.USER_ASSERT,
            confidence=0.8,
            derived_from=[],
        )
        schema = KnowledgeSchema(kind="entity_fact", provenance_required=True)
        with pytest.raises(SchemaViolationError):
            schema.validate_provenance(prov)


class _TenantScopedQueryBackend:
    """Backend stub that resolves tenant schemas like production."""

    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
        if not tenant_id:
            return base_schema_name
        return f"{base_schema_name}_{tenant_id.replace(':', '_')}"

    @staticmethod
    def _resolve_schema(schema: str, tenant_id: str | None) -> str:
        if not tenant_id:
            return schema
        return f"{schema}_{tenant_id.replace(':', '_')}"


class _FailingQueryBackend(_TenantScopedQueryBackend):
    """Backend stub whose metadata query raises — the rejected-query seam."""

    def __init__(self) -> None:
        self.queries: list[dict] = []

    def query_metadata_documents(self, schema, **kwargs):
        resolved_schema = self._resolve_schema(schema, kwargs.get("tenant_id"))
        self.queries.append(
            {"schema": schema, "resolved_schema": resolved_schema, **kwargs}
        )
        raise RuntimeError(f"Could not resolve source ref '{resolved_schema}'")


class _EmptyQueryBackend(_TenantScopedQueryBackend):
    """Backend stub whose metadata query succeeds with zero rows."""

    def __init__(self) -> None:
        self.queries: list[dict] = []

    def query_metadata_documents(self, schema, **kwargs):
        resolved_schema = self._resolve_schema(schema, kwargs.get("tenant_id"))
        self.queries.append(
            {"schema": schema, "resolved_schema": resolved_schema, **kwargs}
        )
        return []


class TestProvenanceStoreFaultContract:
    """A failed provenance query must raise — never read as "no provenance".

    The BFS walk treats a missing record as a legitimate leaf, so a
    swallowed query failure truncates every citation graph to a single
    node and the trace is served as success.
    """

    def test_fetch_raises_with_schema_context_on_query_failure(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _FailingQueryBackend()
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")
        with pytest.raises(RuntimeError) as excinfo:
            store.fetch(["m_child"])
        assert "provenance_t1" in str(excinfo.value)
        assert "Could not resolve source ref" in str(excinfo.value.__cause__)
        assert backend.queries == [
            {
                "schema": "provenance",
                "resolved_schema": "provenance_t1",
                "tenant_id": "t1",
                "yql": (
                    'select * from provenance where memory_id in ("m_child") '
                    'and tenant_id contains "t1" limit 100'
                ),
                "hits": 100,
            }
        ]

    def test_get_raises_on_query_failure(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _FailingQueryBackend()
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")
        with pytest.raises(RuntimeError):
            store.get("m_child")
        assert backend.queries[0]["schema"] == "provenance"
        assert backend.queries[0]["resolved_schema"] == "provenance_t1"

    def test_walk_propagates_query_failure_instead_of_single_node_graph(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        store = ProvenanceStore(
            backend_resolver=lambda: _FailingQueryBackend(), tenant_id="t1"
        )
        with pytest.raises(RuntimeError):
            store.walk("m_child", max_depth=5, max_nodes=10)

    def test_clean_empty_result_keeps_leaf_semantics(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _EmptyQueryBackend()
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")
        assert store.fetch(["m_child"]) == {}
        ordered, primary, truncated, records = store.walk(
            "m_child", max_depth=5, max_nodes=10
        )
        assert ordered == [("m_child", 0)]
        assert [(r.ref_kind, r.ref_id) for r in primary] == [("memory", "m_child")]
        assert truncated is False
        assert records == {}
        # One query from the explicit fetch above, one from the walk's
        # level-0 batch fetch.
        assert len(backend.queries) == 2
        assert backend.queries[0]["tenant_id"] == "t1"
        assert backend.queries[1]["tenant_id"] == "t1"
        assert [q["schema"] for q in backend.queries] == [
            "provenance",
            "provenance",
        ]
        assert [q["resolved_schema"] for q in backend.queries] == [
            "provenance_t1",
            "provenance_t1",
        ]


class _SchemaAwareDeleteBackend(_TenantScopedQueryBackend):
    """Backend stub distinguishing "schema not live" from a delete call.

    ``delete_live_document`` blows up if invoked while ``schema_live`` is
    False, so a test using this stub fails loudly if ``delete()`` still
    tries to delete (which would force a schema deploy in production)
    instead of returning idempotently.
    """

    def __init__(
        self, *, schema_live: bool, exists_raises: BaseException | None = None
    ) -> None:
        self.schema_live = schema_live
        self.exists_raises = exists_raises
        self.schema_exists_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def schema_exists(self, schema_name, tenant_id=None):
        self.schema_exists_calls.append(
            {"schema_name": schema_name, "tenant_id": tenant_id}
        )
        if self.exists_raises is not None:
            raise self.exists_raises
        return self.schema_live

    def delete_live_document(self, document_id, schema_name=None):
        self.delete_calls.append(
            {"document_id": document_id, "schema_name": schema_name}
        )
        if not self.schema_live:
            raise AssertionError(
                "delete_live_document called without a live tenant schema"
            )
        return True


class TestProvenanceStoreDeleteDoesNotForceDeploy:
    """A tenant clear must not redeploy the provenance schema to delete from it.

    When the tenant's provenance schema was never deployed there can be
    no indexed row for it, so ``delete`` returns idempotently instead of
    reaching ``delete_document`` (which would trigger an
    application-package deploy on a cache miss in the real backend).
    """

    def test_delete_is_idempotent_when_tenant_schema_never_deployed(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _SchemaAwareDeleteBackend(schema_live=False)
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        assert store.delete("m1") is True

        assert backend.delete_calls == []
        assert backend.schema_exists_calls == [
            {"schema_name": "provenance", "tenant_id": "t1"}
        ]

    def test_delete_still_removes_the_row_when_schema_is_live(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _SchemaAwareDeleteBackend(schema_live=True)
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        assert store.delete("m1") is True

        assert backend.delete_calls == [
            {"document_id": store._row_id("m1"), "schema_name": "provenance"}
        ]

    def test_delete_propagates_a_schema_lookup_failure(self):
        """A briefly-unreadable registry must not be read as "no schema"."""
        from cogniverse_core.memory.provenance_store import (
            ProvenanceStore,
            ProvenanceWriteError,
        )

        backend = _SchemaAwareDeleteBackend(
            schema_live=False, exists_raises=RuntimeError("registry outage")
        )
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        with pytest.raises(ProvenanceWriteError) as excinfo:
            store.delete("m1")

        assert backend.delete_calls == []
        assert isinstance(excinfo.value.cause, RuntimeError)


class _DeleteAnswerBackend(_TenantScopedQueryBackend):
    """Live-schema backend whose delete answers or raises as configured."""

    def __init__(self, *, answer: Any = True, raises: BaseException | None = None):
        self.answer = answer
        self.raises = raises
        self.delete_calls: list[str] = []

    def schema_exists(self, schema_name, tenant_id=None):
        return True

    def delete_live_document(self, document_id, schema_name=None):
        self.delete_calls.append(document_id)
        if self.raises is not None:
            raise self.raises
        return self.answer


class TestProvenanceStoreDeleteFaultContract:
    """Neither delete raise branch may be read as a completed deletion."""

    def test_a_backend_resolution_failure_raises_a_write_error(self):
        from cogniverse_core.memory.provenance_store import (
            ProvenanceStore,
            ProvenanceWriteError,
        )

        outage = RuntimeError("backend registry closed the instance")

        def resolver():
            raise outage

        store = ProvenanceStore(backend_resolver=resolver, tenant_id="t1")

        with pytest.raises(ProvenanceWriteError) as excinfo:
            store.delete("m1")

        assert excinfo.value.memory_id == "m1"
        assert excinfo.value.row_id == "prov-t1-m1"
        assert excinfo.value.cause is outage
        assert excinfo.value.result is None

    def test_a_delete_transport_failure_raises_a_write_error(self):
        from cogniverse_core.memory.provenance_store import (
            ProvenanceStore,
            ProvenanceWriteError,
        )

        refused = RuntimeError("Vespa returned HTTP 400")
        backend = _DeleteAnswerBackend(raises=refused)
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        with pytest.raises(ProvenanceWriteError) as excinfo:
            store.delete("m1")

        assert backend.delete_calls == ["prov-t1-m1"]
        assert excinfo.value.cause is refused
        assert excinfo.value.result is None

    @pytest.mark.parametrize("answer", [False, None, "deleted"])
    def test_an_unconfirmed_delete_raises_with_the_backend_answer(self, answer):
        from cogniverse_core.memory.provenance_store import (
            ProvenanceStore,
            ProvenanceWriteError,
        )

        backend = _DeleteAnswerBackend(answer=answer)
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        with pytest.raises(ProvenanceWriteError) as excinfo:
            store.delete("m1")

        assert backend.delete_calls == ["prov-t1-m1"]
        assert excinfo.value.memory_id == "m1"
        assert excinfo.value.row_id == "prov-t1-m1"
        assert excinfo.value.result == {"deleted": answer}
        assert excinfo.value.cause is None


class _RecordingIngestBackend(_TenantScopedQueryBackend):
    def __init__(self) -> None:
        self.fed: list[dict] = []

    def ingest_documents(self, documents, schema_name=None):
        self.fed.extend(document.metadata for document in documents)
        return {
            "success_count": len(documents),
            "failed_count": 0,
            "failed_documents": [],
            "total_documents": len(documents),
        }


class TestAttachRequiresAPrimaryDigest:
    """A row without a digest is read as legacy and never digest-checked, so
    no in-contract writer may create one."""

    @staticmethod
    def _provenance():
        return make_provenance(
            written_by="agent:digest",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.5,
            derived_from=[CitationRef.external("https://source/digest")],
        )

    def test_attach_without_a_digest_is_refused_before_any_write(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _RecordingIngestBackend()
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        with pytest.raises(TypeError, match="primary_digest"):
            store.attach("m1", self._provenance())
        assert backend.fed == []

    def test_attach_with_an_empty_digest_is_refused_before_any_write(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _RecordingIngestBackend()
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        with pytest.raises(ValueError, match="primary_digest"):
            store.attach("m1", self._provenance(), primary_digest="")
        assert backend.fed == []

    def test_attach_writes_the_given_digest(self):
        from cogniverse_core.memory.provenance_store import ProvenanceStore

        backend = _RecordingIngestBackend()
        store = ProvenanceStore(backend_resolver=lambda: backend, tenant_id="t1")

        assert store.attach("m1", self._provenance(), primary_digest="a" * 64) == (
            "prov-t1-m1"
        )
        assert [row["primary_digest"] for row in backend.fed] == ["a" * 64]


@pytest.mark.unit
class TestLegacyIndexedRows:
    """Rows indexed before the digest field exists stay readable.

    ``_row_to_record`` reads a missing ``primary_digest`` back as ``""``,
    which can never equal a SHA-256, so every citation walk over a memory
    indexed before this field shipped reported torn provenance for data that
    is in fact intact. ``primary_provenance_digest`` always returns 64 hex
    characters, so ``""`` means "legacy row" and nothing else.
    """

    @staticmethod
    def _legacy_store(memories):
        class _LegacyDigestStore(_StubProvenanceStore):
            def _record(self, mid: str):
                record = super()._record(mid)
                if record is None:
                    return None
                return dataclasses.replace(record, primary_digest="")

        return _LegacyDigestStore(memories)

    def test_a_legacy_row_without_a_digest_still_walks(self):
        memories = {
            "m_root": _seed(
                "m_root",
                "synthesis at root",
                [CitationRef.external("https://source/legacy")],
            ),
        }
        mm = FakeManager(memories)
        mm.provenance_store = self._legacy_store(memories)

        graph = ProvenanceWalker(mm).walk("m_root", tenant_id="t1")

        assert [n.memory_id for n in graph.nodes] == ["m_root"]
        assert [(r.ref_kind, r.ref_id) for r in graph.primary_sources] == [
            ("url", "https://source/legacy"),
            ("memory", "m_root"),
        ]

    def test_a_legacy_row_still_fails_every_other_consistency_check(self):
        """Only the digest comparison is skipped; a genuine mismatch between
        the primary's declared provenance and the indexed row still raises."""
        from cogniverse_core.memory.provenance import ProvenanceConsistencyError

        memories = {
            "m_root": _seed(
                "m_root",
                "synthesis at root",
                [CitationRef.external("https://source/legacy")],
            ),
        }
        mm = FakeManager(memories)
        store = self._legacy_store(memories)
        disagreeing = store._record
        store._record = lambda mid: dataclasses.replace(
            disagreeing(mid), confidence=0.123
        )
        mm.provenance_store = store

        with pytest.raises(
            ProvenanceConsistencyError,
            match="indexed provenance does not match primary provenance",
        ):
            ProvenanceWalker(mm).walk("m_root", tenant_id="t1")

    def test_a_present_digest_that_disagrees_still_raises(self):
        from cogniverse_core.memory.provenance import ProvenanceConsistencyError

        memories = {
            "m_root": _seed(
                "m_root",
                "synthesis at root",
                [CitationRef.external("https://source/current")],
            ),
        }
        mm = FakeManager(memories)
        original = mm.provenance_store._record

        def tampered(mid: str):
            record = original(mid)
            if record is None:
                return None
            return dataclasses.replace(record, primary_digest="0" * 64)

        mm.provenance_store._record = tampered

        with pytest.raises(
            ProvenanceConsistencyError, match="primary digest does not match"
        ):
            ProvenanceWalker(mm).walk("m_root", tenant_id="t1")
