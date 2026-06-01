"""Unit tests for AuditExplanationAgent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.audit_explanation_agent import (
    AuditExplanationAgent,
    AuditExplanationDeps,
    AuditExplanationInput,
    ContradictionOut,
    SourceExplanationOut,
    _format_explanation,
)


def _row(
    mid: str,
    content: str,
    *,
    subject_key: str = "policy:refunds",
    derived_from: Optional[List[str]] = None,
    derivation_kind: str = "synthesis",
    written_by: str = "search_agent",
    confidence: float = 0.8,
    trust_score: Optional[float] = None,
    written_at: str = "2026-01-15T00:00:00Z",
):
    meta: Dict[str, Any] = {
        "kind": "external_doc",
        "subject_key": subject_key,
        "written_at": written_at,
    }
    if derived_from is not None:
        meta["provenance"] = {
            "written_by": written_by,
            "written_at": written_at,
            "derived_from": [
                {"ref_kind": "memory", "ref_id": d, "label": None} for d in derived_from
            ],
            "derivation_kind": derivation_kind,
            "confidence": confidence,
            "trace_id": None,
        }
    if trust_score is not None:
        meta["trust"] = {
            "score": trust_score,
            "initial_score": trust_score,
            "decayed_at": "2026-01-15T00:00:00Z",
            "endorsements": 0,
        }
    return {"id": mid, "memory": content, "metadata": meta}


class _StubProvenanceStore:
    """In-memory ProvenanceStore that walks rows_by_id's provenance chains.

    Implements the contract ProvenanceWalker depends on:
      ``walk(root, max_depth, max_nodes) → (ordered, primary_sources, truncated)``
      ``get(memory_id) → ProvenanceRecord | None``
    """

    def __init__(self, rows_by_id: Dict[str, Dict[str, Any]]):
        self._rows = rows_by_id

    def _record(self, mid: str):
        from cogniverse_core.memory.provenance import (
            CitationRef,
            DerivationKind,
            Provenance,
        )
        from cogniverse_core.memory.provenance_store import ProvenanceRecord

        row = self._rows.get(mid)
        if row is None:
            return None
        prov_meta = (row.get("metadata") or {}).get("provenance")
        if not prov_meta:
            return None
        derived_from = [
            CitationRef.memory(d["ref_id"]) for d in prov_meta.get("derived_from") or []
        ]
        provenance = Provenance(
            written_by=prov_meta["written_by"],
            written_at=prov_meta["written_at"],
            derived_from=derived_from,
            derivation_kind=DerivationKind(prov_meta["derivation_kind"]),
            confidence=prov_meta.get("confidence", 0.5),
            trace_id=prov_meta.get("trace_id"),
        )
        return ProvenanceRecord.from_provenance(mid, "acme", provenance)

    def get(self, memory_id: str):
        return self._record(memory_id)

    def walk(self, root: str, *, max_depth: int = 10, max_nodes: int = 100):
        from cogniverse_core.memory.provenance import CitationRef

        ordered: List = []
        visited = set()
        primary: List = []
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
            if rec is None or not rec.derived_from_memory_ids:
                primary.append(CitationRef.memory(mid))
                continue
            if depth >= max_depth:
                truncated = True
                # Children not walked, but the current node still in
                # primary list since its children were not explored.
                primary.append(CitationRef.memory(mid))
                continue
            for child_id in rec.derived_from_memory_ids:
                frontier.append((child_id, depth + 1))
        return ordered, primary, truncated


def _factory_for(rows_by_id: Dict[str, Dict[str, Any]]):
    def _factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()
        mm.memory.get = lambda memory_id: rows_by_id.get(memory_id)
        mm.provenance_store = _StubProvenanceStore(rows_by_id)
        return mm

    return _factory


def _build(rows_by_id: Dict[str, Dict[str, Any]]):
    return AuditExplanationAgent(
        deps=AuditExplanationDeps(tenant_id="acme"),
        memory_manager_factory=_factory_for(rows_by_id),
    )


@pytest.mark.asyncio
class TestExplanationFlow:
    async def test_single_node_no_provenance_yields_one_source(self):
        # Answer with no provenance — chain is just the answer node itself.
        rows = {
            "answer": _row("answer", "the refund policy is X"),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                include_trust=False,
                include_contradictions=False,
            )
        )
        assert out.answer_memory_id == "answer"
        assert len(out.sources) == 1
        assert out.sources[0].memory_id == "answer"
        assert out.sources[0].depth == 0
        # Without provenance, the node is itself a primary source.
        assert any(p["ref_id"] == "answer" for p in out.primary_sources)

    async def test_provenance_chain_walked_breadth_first(self):
        # answer derived_from [src_a, src_b]; src_a derived_from [leaf_l].
        rows = {
            "answer": _row("answer", "synthesised", derived_from=["src_a", "src_b"]),
            "src_a": _row("src_a", "doc A", derived_from=["leaf_l"]),
            "src_b": _row("src_b", "doc B"),
            "leaf_l": _row("leaf_l", "primary source L"),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                include_trust=False,
                include_contradictions=False,
            )
        )
        ids = [s.memory_id for s in out.sources]
        # BFS: answer (d=0), then src_a, src_b (d=1), then leaf_l (d=2).
        assert ids[0] == "answer"
        assert set(ids[1:3]) == {"src_a", "src_b"}
        assert "leaf_l" in ids
        # Depth assignment correct.
        depth_by_id = {s.memory_id: s.depth for s in out.sources}
        assert depth_by_id["answer"] == 0
        assert depth_by_id["src_a"] == 1
        assert depth_by_id["src_b"] == 1
        assert depth_by_id["leaf_l"] == 2

    async def test_truncation_flag_set_at_depth_cap(self):
        # 4-deep chain; cap at depth 1 forces truncation.
        rows = {
            "answer": _row("answer", "x", derived_from=["a"]),
            "a": _row("a", "x", derived_from=["b"]),
            "b": _row("b", "x", derived_from=["c"]),
            "c": _row("c", "x"),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                max_chain_depth=1,
                include_trust=False,
                include_contradictions=False,
            )
        )
        assert out.truncated_chain is True

    async def test_trust_scoring_returned_when_present(self):
        rows = {
            "answer": _row(
                "answer",
                "synth",
                derived_from=["src_a"],
                trust_score=0.9,
            ),
            "src_a": _row("src_a", "doc", trust_score=0.6),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                include_trust=True,
                include_contradictions=False,
            )
        )
        ts_by_id = {s.memory_id: s.trust_score for s in out.sources}
        # apply_decay: with same-day timestamps decay is ~0; trust unchanged.
        assert ts_by_id["answer"] == pytest.approx(0.9, abs=0.05)
        assert ts_by_id["src_a"] == pytest.approx(0.6, abs=0.05)

    async def test_trust_skipped_when_disabled(self):
        rows = {
            "answer": _row(
                "answer",
                "synth",
                derived_from=["src_a"],
                trust_score=0.9,
            ),
            "src_a": _row("src_a", "doc", trust_score=0.6),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                include_trust=False,
                include_contradictions=False,
            )
        )
        for s in out.sources:
            assert s.trust_score is None

    async def test_contradiction_detected_among_sources(self):
        # Two sources on the same subject_key with different content → conflict.
        rows = {
            "answer": _row("answer", "synth", derived_from=["src_a", "src_b"]),
            "src_a": _row(
                "src_a", "Paris is the capital", subject_key="france:capital"
            ),
            "src_b": _row("src_b", "Lyon is the capital", subject_key="france:capital"),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                include_trust=False,
                include_contradictions=True,
            )
        )
        assert len(out.contradictions_touched) == 1
        c = out.contradictions_touched[0]
        assert c.subject_key == "france:capital"
        assert set(c.conflicting_memory_ids) == {"src_a", "src_b"}

    async def test_contradictions_skipped_when_disabled(self):
        rows = {
            "answer": _row("answer", "synth", derived_from=["src_a", "src_b"]),
            "src_a": _row(
                "src_a", "Paris is the capital", subject_key="france:capital"
            ),
            "src_b": _row("src_b", "Lyon is the capital", subject_key="france:capital"),
        }
        agent = _build(rows)
        out = await agent._process_impl(
            AuditExplanationInput(
                tenant_id="acme",
                answer_memory_id="answer",
                include_trust=False,
                include_contradictions=False,
            )
        )
        assert out.contradictions_touched == []


class TestFormatExplanation:
    def test_no_sources(self):
        text = _format_explanation(
            answer_memory_id="x", sources=[], contradictions=[], truncated=False
        )
        assert "Answer memory: x" in text
        assert "no derivation" in text

    def test_with_sources_and_contradictions(self):
        sources = [
            SourceExplanationOut(
                memory_id="a",
                depth=0,
                excerpt="root",
                derivation_kind="synthesis",
                trust_score=0.9,
            ),
            SourceExplanationOut(
                memory_id="b",
                depth=1,
                excerpt="child",
            ),
        ]
        contradictions = [
            ContradictionOut(subject_key="x", conflicting_memory_ids=["a", "b"])
        ]
        text = _format_explanation(
            answer_memory_id="root",
            sources=sources,
            contradictions=contradictions,
            truncated=True,
        )
        assert "depth=0" in text and "depth=1" in text
        assert "synthesis" in text
        assert "Contradictions touched (1)" in text
        assert "[chain truncated" in text


@pytest.mark.asyncio
async def test_missing_tenant_raises():
    agent = AuditExplanationAgent(
        deps=AuditExplanationDeps(tenant_id=None),
        memory_manager_factory=_factory_for({}),
    )
    with pytest.raises(ValueError, match="no tenant_id"):
        await agent._process_impl(
            AuditExplanationInput(
                answer_memory_id="x",
            )
        )


def test_input_requires_answer_memory_id():
    with pytest.raises(Exception):
        AuditExplanationInput(answer_memory_id="")


def test_agent_capabilities_advertised():
    agent = AuditExplanationAgent(deps=AuditExplanationDeps(tenant_id="acme"))
    assert agent.agent_name == "audit_explanation_agent"
    assert "audit_explanation" in agent.capabilities
    assert agent.port == 8027


async def _fetch_counts(*, include_contradictions: bool):
    import collections

    rows = {
        "answer": _row("answer", "X synthesised", derived_from=["src_a", "src_b"]),
        "src_a": _row("src_a", "doc A says X"),
        "src_b": _row("src_b", "doc B says not X"),
    }
    calls: collections.Counter = collections.Counter()

    def factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()

        def _get(memory_id: str):
            calls[memory_id] += 1
            return rows.get(memory_id)

        mm.memory.get = _get
        mm.provenance_store = _StubProvenanceStore(rows)
        return mm

    agent = AuditExplanationAgent(
        deps=AuditExplanationDeps(tenant_id="acme"),
        memory_manager_factory=factory,
    )
    await agent._process_impl(
        AuditExplanationInput(
            tenant_id="acme",
            answer_memory_id="answer",
            include_trust=True,
            include_contradictions=include_contradictions,
        )
    )
    return calls


@pytest.mark.asyncio
async def test_contradiction_pass_adds_no_extra_memory_fetches():
    """Enabling contradiction detection must reuse the already-fetched source
    memories, not issue a second backend fetch per source."""
    without = await _fetch_counts(include_contradictions=False)
    with_contra = await _fetch_counts(include_contradictions=True)

    # The contradiction pass adds zero fetches — counts are identical.
    assert with_contra == without
    assert with_contra["src_a"] == without["src_a"]
    assert with_contra["src_b"] == without["src_b"]
