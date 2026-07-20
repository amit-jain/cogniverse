"""Unit tests for ContradictionReconciliationAgent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.contradiction_reconciliation_agent import (
    ContradictionReconciliationAgent,
    ContradictionReconciliationDeps,
    ContradictionReconciliationInput,
)
from cogniverse_core.memory.schema import (
    ContradictionPolicy,
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
)
from cogniverse_core.memory.trust import TrustRecord, attach_trust_to_metadata


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(
    mid: str,
    content: str,
    subject_key: str,
    created_at: str = None,
    trust_score: float = 0.5,
):
    meta: Dict[str, Any] = {
        "kind": "entity_fact",
        "subject_key": subject_key,
        "created_at": created_at or _now_iso(),
        "provenance": {
            "written_by": "agent:test",
            "written_at": _now_iso(),
            "derivation_kind": "synthesis",
            "confidence": 1.0,
            "derived_from": [],
        },
    }
    meta = attach_trust_to_metadata(
        meta,
        TrustRecord(
            score=trust_score,
            initial_score=trust_score,
            decayed_at=_now_iso(),
            endorsements=0,
        ),
    )
    return {"id": mid, "memory": content, "metadata": meta}


def _build_agent(
    memories: List[Dict[str, Any]],
    *,
    policy: ContradictionPolicy = ContradictionPolicy.LATEST_WINS,
    target_kind: str = "entity_fact",
) -> ContradictionReconciliationAgent:
    """Wire an agent that resolves the supplied memories under ``policy``."""
    registry = KnowledgeRegistry()
    registry.register(
        KnowledgeSchema(
            kind=target_kind,
            contradiction_policy=policy,
            provenance_required=False,
            pinnable_by=Pinnable.USER,
        ),
        replace=True,
    )
    agent = ContradictionReconciliationAgent(
        deps=ContradictionReconciliationDeps(tenant_id="acme"),
        registry=registry,
    )
    # Stub memory_manager.memory.get(mid) to return the seeded dicts.
    by_id = {m["id"]: m for m in memories}
    fake_mm = MagicMock()
    fake_mm.memory = MagicMock()
    fake_mm.memory.get.side_effect = lambda mid: by_id.get(mid)
    agent.memory_manager = fake_mm
    agent._memory_tenant_id = "acme"
    agent._memory_agent_name = "test_agent"
    # Force is_memory_enabled() True without touching the real mixin path.
    agent.is_memory_enabled = lambda: True  # type: ignore[assignment]
    return agent


@pytest.mark.asyncio
class TestLatestWinsPath:
    async def test_latest_wins_picks_most_recent(self):
        old = _seed(
            "m_old",
            "Lyon",
            "france:capital",
            created_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        )
        new = _seed("m_new", "Paris", "france:capital")
        agent = _build_agent([old, new], policy=ContradictionPolicy.LATEST_WINS)

        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_old", "m_new"],
                tenant_id="acme",
            )
        )
        assert out.policy_used == "latest_wins"
        assert out.survivors == ["m_new"]
        assert out.metadata["survivor_count"] == 1
        assert out.metadata["input_count"] == 2

        survived_flags = {r.memory_id: r.survived for r in out.resolved}
        assert survived_flags == {"m_old": False, "m_new": True}


@pytest.mark.asyncio
class TestTrustRankedPath:
    async def test_higher_trust_wins(self):
        low = _seed("m_low", "Lyon", "france:capital", trust_score=0.3)
        high = _seed("m_high", "Paris", "france:capital", trust_score=0.95)
        agent = _build_agent([low, high], policy=ContradictionPolicy.TRUST_RANKED)
        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_low", "m_high"],
            )
        )
        assert out.policy_used == "trust_ranked"
        assert out.survivors == ["m_high"]


@pytest.mark.asyncio
class TestPreserveBothPath:
    async def test_preserve_both_keeps_all_with_disputed_flag(self):
        a = _seed("m_a", "Lyon", "france:capital")
        b = _seed("m_b", "Paris", "france:capital")
        agent = _build_agent([a, b], policy=ContradictionPolicy.PRESERVE_BOTH)
        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_a", "m_b"],
            )
        )
        assert out.policy_used == "preserve_both"
        assert sorted(out.survivors) == ["m_a", "m_b"]
        for r in out.resolved:
            assert r.survived is True
            assert r.disputed is True


@pytest.mark.asyncio
class TestPolicyOverride:
    async def test_explicit_override_wins(self):
        a = _seed(
            "m_old",
            "Lyon",
            "france:capital",
            created_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        )
        b = _seed("m_new", "Paris", "france:capital", trust_score=0.1)
        # Schema default is LATEST_WINS, but caller forces TRUST_RANKED.
        agent = _build_agent([a, b], policy=ContradictionPolicy.LATEST_WINS)
        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_old", "m_new"],
                policy_override="trust_ranked",
            )
        )
        # m_old has default 0.5 trust; m_new explicitly 0.1.
        assert out.policy_used == "trust_ranked"
        assert out.metadata["policy_overridden"] is True
        assert out.survivors == ["m_old"]

    async def test_invalid_override_raises(self):
        a = _seed("m_a", "Lyon", "france:capital")
        b = _seed("m_b", "Paris", "france:capital")
        agent = _build_agent([a, b])
        with pytest.raises(ValueError, match="unknown policy_override"):
            await agent._process_impl(
                ContradictionReconciliationInput(
                    target_kind="entity_fact",
                    conflict_member_ids=["m_a", "m_b"],
                    policy_override="bogus",
                )
            )


@pytest.mark.asyncio
class TestMissingMembers:
    async def test_some_missing_some_resolved(self):
        a = _seed("m_a", "Lyon", "france:capital")
        b = _seed("m_b", "Paris", "france:capital")
        # Stub the manager so 'm_missing' returns None.
        agent = _build_agent([a, b])

        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_a", "m_b", "m_missing"],
            )
        )
        assert out.metadata["missing_count"] == 1
        assert "m_missing" in out.metadata["missing"]
        assert out.metadata["fetched_count"] == 2

    async def test_member_outage_propagates_not_recorded_missing(self):
        """A backend outage fetching a member must propagate — not be recorded
        as 'missing', which would no-op the reconciliation as if every
        conflicting memory had been deleted."""
        a = _seed("m_a", "Lyon", "france:capital")
        agent = _build_agent([a])

        def _get(mid):
            if mid == "m_a":
                return a
            raise ConnectionError("mem0 backend unreachable")

        agent.memory_manager.memory.get.side_effect = _get

        with pytest.raises(ConnectionError):
            await agent._process_impl(
                ContradictionReconciliationInput(
                    target_kind="entity_fact",
                    conflict_member_ids=["m_a", "m_down"],
                )
            )

    async def test_all_missing_returns_empty(self):
        agent = _build_agent([])
        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_x", "m_y"],
            )
        )
        assert out.survivors == []
        assert out.metadata["missing_count"] == 2


@pytest.mark.asyncio
class TestNoMemoryManager:
    async def test_returns_empty_when_memory_disabled(self):
        agent = ContradictionReconciliationAgent(
            deps=ContradictionReconciliationDeps(tenant_id="acme")
        )
        # is_memory_enabled defaults to False without a wired memory manager.
        out = await agent._process_impl(
            ContradictionReconciliationInput(
                target_kind="entity_fact",
                conflict_member_ids=["m_a", "m_b"],
            )
        )
        assert out.survivors == []
        assert out.metadata["reason"] == "memory_manager_unavailable"


def test_input_validation_requires_at_least_two_members():
    with pytest.raises(Exception):  # pydantic ValidationError
        ContradictionReconciliationInput(
            target_kind="entity_fact",
            conflict_member_ids=["only_one"],
        )


def test_agent_capabilities_advertised():
    agent = ContradictionReconciliationAgent(
        deps=ContradictionReconciliationDeps(tenant_id="acme")
    )
    assert agent.agent_name == "contradiction_reconciliation_agent"
    assert "contradiction_reconciliation" in agent.capabilities
    assert agent.port == 8020
