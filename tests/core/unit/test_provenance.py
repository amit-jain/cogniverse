"""Unit tests for the Provenance + ProvenanceWalker."""

from __future__ import annotations

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


class TestProvenanceConstruction:
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


class FakeManager:
    """In-memory Mem0 stub for walker tests."""

    def __init__(self, memories: Dict[str, Dict[str, Any]]):
        self.memory = MagicMock()
        # Configure get() to return from the dict.
        self.memory.get.side_effect = lambda mid: memories.get(mid)


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


class TestProvenanceWalker:
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
    """Provenance objects must satisfy A.1 schema's validate_provenance."""

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
