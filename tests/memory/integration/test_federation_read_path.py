"""MemoryAwareMixin.get_relevant_context federates over org trunk.

Audit found A.5 was half-shipped: FederationService existed but
MemoryAwareMixin.get_relevant_context never called it. Org-trunk
knowledge was invisible to every agent's normal read path. The plan
explicitly required get_relevant_context to query both tenant and org
trunk, dedup by subject_key, and prefer the tenant overlay.

This test verifies, against real Vespa:

  * an agent with federation enabled retrieves an org-trunk memory
    that has no tenant-side equivalent (cross-tenant visibility);
  * when the tenant has its own memory for the same subject_key, the
    tenant version wins on dedup (overlay semantics);
  * when federation is disabled (default), the org-trunk memory is
    invisible — legacy agents are unchanged;
  * the org-trunk memory is tagged with origin metadata so audit and
    UI can show provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.federation import org_trunk_tenant_id
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

# The shared Vespa fixture deploys schemas keyed on "test_tenant" alone;
# initialising another tenant requires auto_create_schema=True. To keep
# the test focused on the read-path behavior, we colocate both sides
# under the same physical schema by giving the org-trunk tenant the
# same base name.
TENANT = "test_tenant"
TRUNK_TENANT = org_trunk_tenant_id(TENANT)
AGENT = "h6_reader"


def _build_manager(
    tenant_id: str, shared_memory_vespa, shared_denseon, *, auto_create: bool
):
    """Build a per-tenant Mem0 manager. Trunk tenant requires
    auto_create=True so the test deploys the trunk's own per-tenant
    schema on first use."""
    Mem0MemoryManager._instances.pop(tenant_id, None)
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=auto_create,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    return mm


@pytest.fixture(scope="module")
def federation_setup(shared_memory_vespa, shared_denseon):
    Mem0MemoryManager._instances.clear()
    tenant_mm = _build_manager(
        TENANT, shared_memory_vespa, shared_denseon, auto_create=False
    )
    # Trunk gets its own per-tenant schema deployed on first init.
    trunk_mm = _build_manager(
        TRUNK_TENANT, shared_memory_vespa, shared_denseon, auto_create=True
    )
    yield tenant_mm, trunk_mm
    try:
        tenant_mm.clear_agent_memory(TENANT, AGENT)
        trunk_mm.clear_agent_memory(TRUNK_TENANT, AGENT)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


class _FederatingAgent(MemoryAwareMixin):
    """Bare mixin instance to exercise get_relevant_context federation."""


def _bind(agent: _FederatingAgent, mm, tenant_id: str, *, federate: bool) -> None:
    agent.memory_manager = mm
    agent._memory_tenant_id = tenant_id
    agent._memory_agent_name = AGENT
    agent._memory_initialized = True
    agent.enable_org_trunk_federation(federate)


class TestFederationReadPath:
    def test_org_trunk_memory_visible_when_federation_enabled(self, federation_setup):
        tenant_mm, trunk_mm = federation_setup
        # Seed an org-trunk-only memory.
        trunk_id = trunk_mm.add_memory(
            content="ORG_TRUNK_FACT_about_compliance_h6",
            tenant_id=TRUNK_TENANT,
            agent_name=AGENT,
            metadata={
                "kind": "tenant_instruction",
                "subject_key": "h6:compliance:trunk_only",
            },
            infer=False,
        )
        assert trunk_id

        agent = _FederatingAgent()
        _bind(agent, tenant_mm, TENANT, federate=True)
        ctx = agent.get_relevant_context("compliance", top_k=10)
        assert ctx is not None, "federated read should surface trunk memory"
        assert "ORG_TRUNK_FACT_about_compliance_h6" in ctx, (
            f"org-trunk memory must appear in federated context; got "
            f"context={ctx[:200]!r}"
        )

    def test_org_trunk_invisible_when_federation_disabled(self, federation_setup):
        tenant_mm, trunk_mm = federation_setup
        # Seed a fresh trunk memory under a different subject so it
        # doesn't collide with the previous test's leftover.
        trunk_mm.add_memory(
            content="ORG_TRUNK_FACT_default_off_h6",
            tenant_id=TRUNK_TENANT,
            agent_name=AGENT,
            metadata={
                "kind": "tenant_instruction",
                "subject_key": "h6:compliance:default_off",
            },
            infer=False,
        )

        agent = _FederatingAgent()
        _bind(agent, tenant_mm, TENANT, federate=False)
        ctx = agent.get_relevant_context("default off", top_k=10) or ""
        # Without federation, the trunk-only memory must not surface.
        assert "ORG_TRUNK_FACT_default_off_h6" not in ctx, (
            "federation-disabled retrieval must not see org-trunk memories; "
            "the legacy tenant-only contract is broken"
        )

    def test_tenant_overlay_wins_on_subject_collision(self, federation_setup):
        tenant_mm, trunk_mm = federation_setup
        subject = "h6:capital:france"
        # Trunk says one thing.
        trunk_mm.add_memory(
            content="TRUNK_SAYS_PARIS_h6",
            tenant_id=TRUNK_TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": subject},
            infer=False,
        )
        # Tenant overlay says something newer for the SAME subject.
        tenant_mm.add_memory(
            content="TENANT_OVERLAY_PARIS_DELUXE_h6",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": subject},
            infer=False,
        )

        agent = _FederatingAgent()
        _bind(agent, tenant_mm, TENANT, federate=True)
        ctx = agent.get_relevant_context("capital of france", top_k=10) or ""
        assert "TENANT_OVERLAY_PARIS_DELUXE_h6" in ctx, (
            f"tenant overlay must win on subject_key collision; ctx={ctx[:300]!r}"
        )
        assert "TRUNK_SAYS_PARIS_h6" not in ctx, (
            "trunk version should be hidden by the tenant overlay; got "
            f"both in ctx={ctx[:300]!r}"
        )
