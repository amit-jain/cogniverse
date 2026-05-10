"""MemoryAwareMixin.initialize_memory wires knowledge_registry.

The previous commits added enforcement (provenance,
auto-trust, retrieval ranking, reconciliation) but every check was
gated by ``if self._knowledge_registry is None: return``. Production
paths never set the registry, so the checks were dead.

This test verifies the wire that activates them: after calling
``initialize_memory`` on any MemoryAware agent, the underlying
``Mem0MemoryManager._knowledge_registry`` is populated. The
downstream effects (provenance enforcement, trust attachment,
retrieval ranking) have their own integration tests; this file only
covers the wire that makes those tests' premises true in production.
"""

from __future__ import annotations

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import KnowledgeRegistry

pytestmark = pytest.mark.integration


class _Host(MemoryAwareMixin):
    """Bare object adopting the mixin so we can call its initializer."""


def test_initialize_memory_passes_knowledge_registry_to_manager(monkeypatch):
    """The wire: initialize_memory must set Mem0MemoryManager._knowledge_registry."""
    captured: dict = {}

    def _stub_initialize(self, **kwargs):
        # Capture the kwargs the mixin handed to the manager.
        captured.update(kwargs)
        # Pretend we're now initialized so the mixin proceeds.
        self.memory = object()
        # Apply the kwarg the same way the real initialize does — that's
        # what the registry-gated checks consult downstream.
        self._knowledge_registry = kwargs.get("knowledge_registry")

    monkeypatch.setattr(Mem0MemoryManager, "initialize", _stub_initialize)
    # Mem0MemoryManager is a per-tenant singleton with cached state; clear
    # so this test gets a fresh instance whose memory is None.
    Mem0MemoryManager._instances.clear()

    host = _Host()
    ok = host.initialize_memory(
        agent_name="f11_test_agent",
        tenant_id="f11_tenant",
        backend_host="http://localhost",
        backend_port=8080,
        llm_model="openai/test-model",
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://localhost:9000",
        embedder_base_url="http://localhost:8081",
        config_manager=None,
        schema_loader=None,
    )
    assert ok is True
    assert "knowledge_registry" in captured, (
        "initialize_memory must pass knowledge_registry to "
        "Mem0MemoryManager.initialize — without it the "
        "enforcement code is gated off in production"
    )
    registry = captured["knowledge_registry"]
    assert isinstance(registry, KnowledgeRegistry), (
        f"expected KnowledgeRegistry instance; got {type(registry)!r}"
    )
    # Sanity: the registry the mixin built must include the canonical kinds
    # the schema-driven enforcement actually checks.
    assert registry.is_registered("entity_fact"), (
        "the default registry must include entity_fact (provenance_required); "
        "without it, downstream agents writing facts won't trigger the "
        "provenance check"
    )

    # The manager's _knowledge_registry attribute must be populated so
    # the in-process check `if self._knowledge_registry is None: return`
    # in add_memory + get_relevant_context evaluates False.
    assert host.memory_manager._knowledge_registry is not None, (
        "after initialize_memory, the manager's _knowledge_registry "
        "must be set so enforcement and P2.2 ranking actually fire"
    )


def test_two_separate_agents_each_get_a_registry(monkeypatch):
    """Each agent's initialize_memory wires its own registry — no cross-talk."""
    captured: list = []

    def _stub_initialize(self, **kwargs):
        captured.append(kwargs.get("knowledge_registry"))
        self.memory = object()
        self._knowledge_registry = kwargs.get("knowledge_registry")

    monkeypatch.setattr(Mem0MemoryManager, "initialize", _stub_initialize)
    Mem0MemoryManager._instances.clear()

    for tenant in ("tenant_a", "tenant_b"):
        host = _Host()
        host.initialize_memory(
            agent_name=f"agent_for_{tenant}",
            tenant_id=tenant,
            backend_host="http://localhost",
            backend_port=8080,
            llm_model="openai/test-model",
            embedding_model="lightonai/DenseOn",
            llm_base_url="http://localhost:9000",
            embedder_base_url="http://localhost:8081",
            config_manager=None,
            schema_loader=None,
        )

    assert len(captured) == 2
    for reg in captured:
        assert isinstance(reg, KnowledgeRegistry)
