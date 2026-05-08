"""Unit tests for CrossTenantComparisonAgent (C3.3)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.cross_tenant_comparison_agent import (
    CrossTenantComparisonAgent,
    CrossTenantComparisonDeps,
    CrossTenantComparisonInput,
    _ACLRejected,
)
from cogniverse_core.memory.schema import build_default_registry


def _row(mid: str, content: str, *, subject_key: str = "", kind: str = "external_doc"):
    meta: Dict[str, Any] = {"kind": kind}
    if subject_key:
        meta["subject_key"] = subject_key
    return {"id": mid, "memory": content, "metadata": meta}


def _factory_for(per_tenant: Dict[str, List[Dict[str, Any]]]):
    """Return a memory_manager factory matching the federation test pattern."""

    def _factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()
        rows = list(per_tenant.get(tenant_id, []))
        mm.get_all_memories = lambda *, tenant_id=tenant_id, agent_name: list(rows)
        return mm

    return _factory


def _build(per_tenant: Dict[str, List[Dict[str, Any]]]):
    factory = _factory_for(per_tenant)
    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id="acme:production"),
        memory_manager_factory=factory,
        registry=build_default_registry(),
    )
    return agent


@pytest.mark.asyncio
class TestComparisonHappyPath:
    async def test_two_tenants_agree_one_signature(self):
        per_tenant = {
            "acme:alpha": [
                _row("m_a", "Paris is the capital", subject_key="france:capital"),
            ],
            "acme:beta": [
                _row("m_b", "Paris is the capital", subject_key="france:capital"),
            ],
            # Org trunks empty for both.
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            CrossTenantComparisonInput(
                tenant_id="acme:production",
                subject_key="france:capital",
                tenant_ids=["acme:alpha", "acme:beta"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        assert out.subject_key == "france:capital"
        assert len(out.tenant_views) == 2
        assert out.distinct_signatures_count == 1  # both tenants agree
        assert out.metadata["tenants_compared"] == 2

    async def test_two_tenants_disagree_two_signatures(self):
        per_tenant = {
            "acme:alpha": [
                _row("m_a", "Paris is the capital", subject_key="france:capital"),
            ],
            "acme:beta": [
                _row("m_b", "Lyon is the capital", subject_key="france:capital"),
            ],
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            CrossTenantComparisonInput(
                tenant_id="acme:production",
                subject_key="france:capital",
                tenant_ids=["acme:alpha", "acme:beta"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        assert out.distinct_signatures_count == 2

    async def test_org_trunk_visible_to_each_tenant(self):
        per_tenant = {
            "acme:alpha": [],
            "acme:beta": [],
            "acme:_org_trunk": [
                _row("m_org", "Paris", subject_key="france:capital"),
            ],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            CrossTenantComparisonInput(
                tenant_id="acme:production",
                subject_key="france:capital",
                tenant_ids=["acme:alpha", "acme:beta"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        # Each tenant sees the same org-trunk record via federated read.
        for v in out.tenant_views:
            assert v.matching_memory_ids == ["m_org"]
            assert v.origin_tags == ["org_trunk"]


@pytest.mark.asyncio
class TestACLs:
    async def test_user_role_rejected(self):
        agent = _build({})
        with pytest.raises(_ACLRejected, match="tenant_admin or org_admin"):
            await agent._process_impl(
                CrossTenantComparisonInput(
                    tenant_id="acme:production",
                    subject_key="x",
                    tenant_ids=["acme:alpha", "acme:beta"],
                    actor_role="user",
                    actor_id="alice",
                )
            )

    async def test_unknown_role_rejected(self):
        agent = _build({})
        with pytest.raises(_ACLRejected, match="unknown actor_role"):
            await agent._process_impl(
                CrossTenantComparisonInput(
                    tenant_id="acme:production",
                    subject_key="x",
                    tenant_ids=["acme:alpha", "acme:beta"],
                    actor_role="superuser",
                    actor_id="alice",
                )
            )

    async def test_cross_org_request_rejected(self):
        agent = _build({})
        with pytest.raises(_ACLRejected, match="cross-org comparison"):
            await agent._process_impl(
                CrossTenantComparisonInput(
                    tenant_id="acme:production",
                    subject_key="x",
                    tenant_ids=["acme:alpha", "globex:production"],  # different org
                    actor_role="org_admin",
                    actor_id="oadm",
                )
            )


@pytest.mark.asyncio
class TestMissingTenantData:
    async def test_tenant_with_no_matching_subject_returns_empty_view(self):
        per_tenant = {
            "acme:alpha": [
                _row("m_a", "Paris", subject_key="france:capital"),
            ],
            "acme:beta": [
                _row("m_b", "unrelated content", subject_key="other:thing"),
            ],
            "acme:_org_trunk": [],
        }
        agent = _build(per_tenant)
        out = await agent._process_impl(
            CrossTenantComparisonInput(
                tenant_id="acme:production",
                subject_key="france:capital",
                tenant_ids=["acme:alpha", "acme:beta"],
                actor_role="org_admin",
                actor_id="oadm",
            )
        )
        beta_view = next(v for v in out.tenant_views if v.tenant_id == "acme:beta")
        assert beta_view.matching_memory_ids == []
        alpha_view = next(v for v in out.tenant_views if v.tenant_id == "acme:alpha")
        assert alpha_view.matching_memory_ids == ["m_a"]


def test_input_validation_requires_two_tenants():
    with pytest.raises(Exception):  # pydantic validation
        CrossTenantComparisonInput(
            subject_key="x",
            tenant_ids=["only_one"],
            actor_role="org_admin",
            actor_id="oadm",
        )


def test_agent_capabilities_advertised():
    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id="acme:production")
    )
    assert agent.agent_name == "cross_tenant_comparison_agent"
    assert "cross_tenant_comparison" in agent.capabilities
    assert agent.port == 8023
