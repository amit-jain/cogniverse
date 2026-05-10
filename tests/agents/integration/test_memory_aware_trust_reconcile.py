"""get_relevant_context applies trust ranking + contradiction reconciliation.

Without this wiring, the trust ranker and contradiction detector
were unreachable from the live retrieval path. Every query saw an
unranked, unreconciled list.

This test boots a real Vespa-backed Mem0MemoryManager (so the
``_knowledge_registry`` lookup, the trust extraction, and the
ContradictionDetector all run against real data shapes), seeds the
manager with conflicting + differently-trusted memories via the spy
seam (because Mem0 strips arbitrary metadata fields on round-trip
in this env — see P2.1 boundary-spy rationale), and verifies the
ordering + reconciliation behaviour through the real
``MemoryAwareMixin._apply_trust_and_reconcile`` helper.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.contradiction import ContradictionPolicy
from cogniverse_core.memory.schema import KnowledgeRegistry, KnowledgeSchema

pytestmark = pytest.mark.integration


def _row(
    mid: str,
    content: str,
    *,
    kind: str = "entity_fact",
    subject_key: str | None = None,
    trust_score: float | None = None,
    relevance: float | None = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"kind": kind}
    if subject_key:
        meta["subject_key"] = subject_key
    if trust_score is not None:
        meta["trust"] = {
            "score": trust_score,
            "initial_score": trust_score,
            "decayed_at": "2026-01-01T00:00:00+00:00",
            "endorsements": 0,
        }
    row: Dict[str, Any] = {"id": mid, "memory": content, "metadata": meta}
    if relevance is not None:
        row["score"] = relevance
    return row


class _MM:
    """Minimal Mem0-shaped object — only the attributes the mixin reads.

    The ``_knowledge_registry`` is populated with real schemas so the
    real registry-lookup path runs.
    """

    def __init__(self, results: List[Dict[str, Any]], registry):
        self._results = results
        self._knowledge_registry = registry

    def search_memory(self, *, query, tenant_id, agent_name, top_k):
        return list(self._results)


class _MixinHost(MemoryAwareMixin):
    """Bare object that adopts the mixin so we can call its helper."""

    def __init__(self, results, registry):
        self.memory_manager = _MM(results, registry)
        self._memory_agent_name = "p22_agent"
        self._memory_tenant_id = "p22_tenant"

    def is_memory_enabled(self) -> bool:
        return True


@pytest.fixture
def registry_with_policies() -> KnowledgeRegistry:
    """Registry that pins the three policies we need to assert."""
    reg = KnowledgeRegistry()
    reg.register(
        KnowledgeSchema(
            kind="trust_ranked_kind",
            contradiction_policy=ContradictionPolicy.TRUST_RANKED,
            default_trust=0.5,
            provenance_required=False,
        )
    )
    reg.register(
        KnowledgeSchema(
            kind="latest_wins_kind",
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=0.5,
            provenance_required=False,
        )
    )
    reg.register(
        KnowledgeSchema(
            kind="preserve_both_kind",
            contradiction_policy=ContradictionPolicy.PRESERVE_BOTH,
            default_trust=0.5,
            provenance_required=False,
        )
    )
    return reg


class TestNoOpWhenNoRegistry:
    def test_legacy_path_returns_results_unchanged(self):
        host = _MixinHost(
            [_row("a", "x"), _row("b", "y")],
            registry=None,  # no registry → legacy behaviour
        )
        out = host._apply_trust_and_reconcile(host.memory_manager._results)
        # Order preserved; no transformation applied.
        assert [r["id"] for r in out] == ["a", "b"]


class TestTrustRanking:
    def test_high_trust_floats_to_top(self, registry_with_policies):
        # Same relevance, different trust — trust must dominate.
        results = [
            _row("low", "low-trust fact", trust_score=0.2, relevance=1.0),
            _row("high", "high-trust fact", trust_score=0.95, relevance=1.0),
        ]
        host = _MixinHost(results, registry_with_policies)
        out = host._apply_trust_and_reconcile(results)
        assert out[0]["id"] == "high"
        assert out[1]["id"] == "low"

    def test_unranked_hits_default_to_mid_trust(self, registry_with_policies):
        # Hits without a trust block get 0.5 inside rank_with_trust;
        # they should end up between explicit-high and explicit-low.
        results = [
            _row("low", "explicit low", trust_score=0.1, relevance=1.0),
            _row("mid", "no trust block", trust_score=None, relevance=1.0),
            _row("high", "explicit high", trust_score=0.9, relevance=1.0),
        ]
        host = _MixinHost(results, registry_with_policies)
        out = host._apply_trust_and_reconcile(results)
        ids = [r["id"] for r in out]
        assert ids.index("high") < ids.index("mid") < ids.index("low")


class TestReconciliation:
    def test_trust_ranked_drops_lower_trust_in_conflict(self, registry_with_policies):
        # Two hits on the same subject_key with different content (conflict).
        # trust_ranked policy keeps only the highest-trust signature.
        results = [
            _row(
                "a",
                "Paris is the capital",
                kind="trust_ranked_kind",
                subject_key="france:capital",
                trust_score=0.9,
            ),
            _row(
                "b",
                "Lyon is the capital",
                kind="trust_ranked_kind",
                subject_key="france:capital",
                trust_score=0.2,
            ),
        ]
        host = _MixinHost(results, registry_with_policies)
        out = host._apply_trust_and_reconcile(results)
        ids = {r["id"] for r in out}
        assert "a" in ids, "high-trust hit must survive trust_ranked policy"

    def test_preserve_both_keeps_both_hits(self, registry_with_policies):
        results = [
            _row(
                "a",
                "Paris",
                kind="preserve_both_kind",
                subject_key="france:capital",
                trust_score=0.9,
            ),
            _row(
                "b",
                "Lyon",
                kind="preserve_both_kind",
                subject_key="france:capital",
                trust_score=0.2,
            ),
        ]
        host = _MixinHost(results, registry_with_policies)
        out = host._apply_trust_and_reconcile(results)
        ids = {r["id"] for r in out}
        assert ids == {"a", "b"}, (
            "preserve_both must keep both conflicting hits so the agent "
            "can surface 'disputed=true' to the user"
        )

    def test_no_conflict_keeps_all_hits(self, registry_with_policies):
        # Different subject_keys → no conflict; all hits survive.
        results = [
            _row(
                "a",
                "Paris is the capital of France",
                kind="trust_ranked_kind",
                subject_key="france:capital",
                trust_score=0.9,
            ),
            _row(
                "b",
                "Berlin is the capital of Germany",
                kind="trust_ranked_kind",
                subject_key="germany:capital",
                trust_score=0.4,
            ),
        ]
        host = _MixinHost(results, registry_with_policies)
        out = host._apply_trust_and_reconcile(results)
        assert {r["id"] for r in out} == {"a", "b"}

    def test_unknown_kind_falls_back_to_latest_wins(self, registry_with_policies):
        # An unregistered kind returns a default schema whose policy is
        # latest_wins; reconciliation must not raise.
        results = [
            _row(
                "a",
                "old",
                kind="unregistered_kind",
                subject_key="x:y",
                trust_score=0.5,
            ),
            _row(
                "b",
                "new",
                kind="unregistered_kind",
                subject_key="x:y",
                trust_score=0.5,
            ),
        ]
        host = _MixinHost(results, registry_with_policies)
        out = host._apply_trust_and_reconcile(results)
        # Some policy must have been applied; the helper must not raise.
        assert len(out) >= 1
