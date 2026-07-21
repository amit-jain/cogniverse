"""Unit tests for FederatedQueryAgent."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.federated_query_agent import (
    FederatedQueryAgent,
    FederatedQueryDeps,
    FederatedQueryInput,
    _matches_query,
)
from cogniverse_core.memory.federation import ACLRejected
from cogniverse_core.memory.schema import build_default_registry


def _row(mid: str, content: str, *, subject_key: str = "", kind: str = "external_doc"):
    meta: Dict[str, Any] = {"kind": kind}
    if subject_key:
        meta["subject_key"] = subject_key
    return {"id": mid, "memory": content, "metadata": meta}


def _factory_for(per_tenant: Dict[str, List[Dict[str, Any]]]):
    """Return a memory_manager factory mirroring the federation test pattern."""

    def _factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()
        rows = list(per_tenant.get(tenant_id, []))
        mm.get_all_memories = lambda *, tenant_id=tenant_id, agent_name: list(rows)
        return mm

    return _factory


def _build(per_tenant: Dict[str, List[Dict[str, Any]]]):
    factory = _factory_for(per_tenant)
    return FederatedQueryAgent(
        deps=FederatedQueryDeps(tenant_id="acme:production"),
        memory_manager_factory=factory,
        registry=build_default_registry(),
    )


class TestMatchesQuery:
    def test_substring_case_insensitive(self):
        assert _matches_query({"memory": "Paris is the Capital"}, "paris") is True
        assert _matches_query({"memory": "Paris is the Capital"}, "CAPITAL") is True

    def test_no_match(self):
        assert _matches_query({"memory": "Berlin"}, "paris") is False

    def test_empty_query_matches_anything(self):
        assert _matches_query({"memory": "anything"}, "") is True

    def test_handles_content_field_alias(self):
        assert _matches_query({"content": "fallback field"}, "fallback") is True

    def test_missing_content_does_not_match(self):
        assert _matches_query({}, "anything") is False


@pytest.mark.asyncio
class TestQueryHappyPath:
    async def test_single_tenant_returns_matches_only(self):
        per_tenant = {
            "acme:alpha": [
                _row("m1", "Paris is the capital of France"),
                _row("m2", "Berlin is the capital of Germany"),
            ],
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="Paris",
                tenant_ids=["acme:alpha"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        assert out.query == "Paris"
        assert len(out.hits) == 1
        assert out.hits[0].memory_id == "m1"
        assert out.hits[0].tenant_id == "acme:alpha"
        assert out.metadata["per_tenant_counts"]["acme:alpha"] == 1
        assert out.used_rlm is False
        assert out.summary is None

    async def test_two_tenants_aggregated(self):
        per_tenant = {
            "acme:alpha": [_row("a", "Paris is the capital")],
            "acme:beta": [_row("b", "Paris bistros are great")],
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="Paris",
                tenant_ids=["acme:alpha", "acme:beta"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        assert {h.memory_id for h in out.hits} == {"a", "b"}
        assert out.metadata["tenants_queried"] == 2
        assert out.metadata["hit_count"] == 2

    async def test_org_trunk_visible_per_tenant(self):
        per_tenant = {
            "acme:alpha": [],
            "acme:beta": [],
            "acme:_org_trunk": [_row("trunk", "Paris is the capital")],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="Paris",
                tenant_ids=["acme:alpha", "acme:beta"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        # Both tenants see the trunk row through their federated reads.
        assert len(out.hits) == 2
        for h in out.hits:
            assert h.memory_id == "trunk"
            assert h.origin == "org_trunk"

    async def test_top_k_per_tenant_caps_results(self):
        per_tenant = {
            "acme:alpha": [_row(f"m{i}", "Paris match") for i in range(50)],
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="Paris",
                tenant_ids=["acme:alpha"],
                actor_role="org_admin",
                actor_id="oadm",
                top_k_per_tenant=10,
            )
        )
        assert len(out.hits) == 10
        assert out.metadata["per_tenant_counts"]["acme:alpha"] == 10

    async def test_no_matches_returns_empty(self):
        per_tenant = {
            "acme:alpha": [_row("m", "Berlin only")],
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="Paris",
                tenant_ids=["acme:alpha"],
                actor_role="tenant_admin",
                actor_id="tadm",
            )
        )
        assert out.hits == []
        assert out.used_rlm is False


@pytest.mark.asyncio
class TestACLs:
    async def test_user_role_rejected(self):
        agent = _build({})
        with pytest.raises(ACLRejected, match="tenant_admin or org_admin"):
            await agent._process_impl(
                FederatedQueryInput(
                    tenant_id="acme:production",
                    query="x",
                    tenant_ids=["acme:alpha"],
                    actor_role="user",
                    actor_id="alice",
                )
            )

    async def test_unknown_role_rejected(self):
        agent = _build({})
        with pytest.raises(ACLRejected, match="unknown actor_role"):
            await agent._process_impl(
                FederatedQueryInput(
                    tenant_id="acme:production",
                    query="x",
                    tenant_ids=["acme:alpha"],
                    actor_role="superuser",
                    actor_id="alice",
                )
            )

    async def test_cross_org_request_rejected(self):
        agent = _build({})
        with pytest.raises(ACLRejected, match="cross-org query"):
            await agent._process_impl(
                FederatedQueryInput(
                    tenant_id="acme:production",
                    query="x",
                    tenant_ids=["acme:alpha", "globex:production"],
                    actor_role="org_admin",
                    actor_id="oadm",
                )
            )

    async def test_no_caller_tenant_is_rejected(self):
        # Without a caller tenant_id there is no org to scope to; the agent
        # must reject rather than read across every listed org.
        per_tenant = {
            "acme:alpha": [_row("a", "Paris")],
            "globex:beta": [_row("b", "Paris")],
            "acme:_org_trunk": [],
            "globex:_org_trunk": [],
        }
        agent = _build(per_tenant)
        with pytest.raises(ACLRejected, match="tenant_id is required") as exc:
            await agent._process_impl(
                FederatedQueryInput(
                    tenant_id=None,
                    query="Paris",
                    tenant_ids=["acme:alpha", "globex:beta"],
                    actor_role="org_admin",
                    actor_id="oadm",
                )
            )
        assert "cannot be verified" in str(exc.value)


@pytest.mark.asyncio
class TestAgentNameFilter:
    async def test_default_filter_is_promoted(self):
        captured: list[str] = []

        def _factory(tenant_id):
            mm = MagicMock()
            mm.memory = MagicMock()

            def _get(*, tenant_id, agent_name):
                captured.append(agent_name)
                return []

            mm.get_all_memories = _get
            return mm

        agent = FederatedQueryAgent(
            deps=FederatedQueryDeps(tenant_id="acme:production"),
            memory_manager_factory=_factory,
            registry=build_default_registry(),
        )
        await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="x",
                tenant_ids=["acme:alpha"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        # Two reads (tenant + org-trunk) — both with default agent_name.
        assert "_promoted" in captured

    async def test_custom_filter_passed_through(self):
        captured: list[str] = []

        def _factory(tenant_id):
            mm = MagicMock()
            mm.memory = MagicMock()

            def _get(*, tenant_id, agent_name):
                captured.append(agent_name)
                return []

            mm.get_all_memories = _get
            return mm

        agent = FederatedQueryAgent(
            deps=FederatedQueryDeps(tenant_id="acme:production"),
            memory_manager_factory=_factory,
            registry=build_default_registry(),
        )
        await agent._process_impl(
            FederatedQueryInput(
                tenant_id="acme:production",
                query="x",
                tenant_ids=["acme:alpha"],
                actor_role="org_admin",
                actor_id="oadm",
                agent_name_filter="search_agent",
            )
        )
        # Only "search_agent" — never "_promoted".
        assert all(name == "search_agent" for name in captured)


def test_input_validation_requires_at_least_one_tenant():
    with pytest.raises(Exception):  # pydantic ValidationError
        FederatedQueryInput(
            query="x",
            tenant_ids=[],
            actor_role="org_admin",
            actor_id="oadm",
        )


def test_top_k_bounds_enforced():
    with pytest.raises(Exception):
        FederatedQueryInput(
            query="x",
            tenant_ids=["acme:alpha"],
            actor_role="org_admin",
            actor_id="oadm",
            top_k_per_tenant=0,
        )
    with pytest.raises(Exception):
        FederatedQueryInput(
            query="x",
            tenant_ids=["acme:alpha"],
            actor_role="org_admin",
            actor_id="oadm",
            top_k_per_tenant=10_000,
        )


def test_agent_capabilities_advertised():
    agent = FederatedQueryAgent(deps=FederatedQueryDeps(tenant_id="acme:production"))
    assert agent.agent_name == "federated_query_agent"
    assert "federated_query" in agent.capabilities
    assert agent.port == 8024
