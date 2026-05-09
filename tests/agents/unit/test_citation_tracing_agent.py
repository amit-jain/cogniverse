"""Unit tests for CitationTracingAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_agents.citation_tracing_agent import (
    CitationTracingAgent,
    CitationTracingDeps,
    CitationTracingInput,
    CitationTracingOutput,
)
from cogniverse_core.memory.provenance import (
    CitationGraph,
    CitationNode,
    CitationRef,
    DerivationKind,
    Provenance,
)


def _node(memory_id: str, depth: int, with_prov: bool = True) -> CitationNode:
    prov = (
        Provenance(
            written_by="agent:test",
            written_at="2026-05-08T00:00:00+00:00",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.7,
            derived_from=[],
        )
        if with_prov
        else None
    )
    return CitationNode(
        memory_id=memory_id,
        provenance=prov,
        content_excerpt=f"excerpt of {memory_id}",
        depth=depth,
    )


@pytest.mark.asyncio
async def test_returns_empty_chain_when_memory_disabled():
    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id="acme"))
    # No memory_manager wired → is_memory_enabled() → False.
    assert agent.is_memory_enabled() is False

    out = await agent._process_impl(
        CitationTracingInput(memory_id="m_root", tenant_id="acme")
    )
    assert isinstance(out, CitationTracingOutput)
    assert out.root_memory_id == "m_root"
    assert out.nodes == []
    assert out.primary_sources == []
    assert out.metadata.get("reason") == "memory_manager_unavailable"


@pytest.mark.asyncio
async def test_walker_results_serialised_to_typed_output(monkeypatch):
    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id="acme"))

    # Force the mixin's enable check to True so the walker path runs.
    monkeypatch.setattr(agent, "is_memory_enabled", lambda: True)
    fake_mm = MagicMock()
    agent.memory_manager = fake_mm
    agent._memory_tenant_id = "acme"

    fake_graph = CitationGraph(
        root_memory_id="m_root",
        nodes=[
            _node("m_root", depth=0),
            _node("m_a", depth=1),
            _node("m_leaf", depth=2, with_prov=False),
        ],
        primary_sources=[
            CitationRef.external("https://wiki/x"),
            CitationRef.memory("m_leaf"),
        ],
        truncated_at_max_depth=False,
    )

    # Patch the walker class used inside _process_impl.
    import cogniverse_core.memory.provenance as prov_mod

    class FakeWalker:
        def __init__(self, *a, **kw):
            self.args = a
            self.kwargs = kw

        def walk(self, memory_id, tenant_id):
            assert memory_id == "m_root"
            assert tenant_id == "acme"
            return fake_graph

    monkeypatch.setattr(prov_mod, "ProvenanceWalker", FakeWalker)

    out = await agent._process_impl(
        CitationTracingInput(
            memory_id="m_root", tenant_id="acme", max_depth=4, max_nodes=20
        )
    )

    assert out.root_memory_id == "m_root"
    assert [n.memory_id for n in out.nodes] == ["m_root", "m_a", "m_leaf"]
    assert out.nodes[0].written_by == "agent:test"
    assert out.nodes[0].derivation_kind == "synthesis"
    assert out.nodes[2].written_by is None  # leaf had no provenance
    keys = {(r.ref_kind, r.ref_id) for r in out.primary_sources}
    assert ("url", "https://wiki/x") in keys
    assert ("memory", "m_leaf") in keys
    assert out.truncated is False
    assert out.metadata == {"nodes_visited": 3, "primary_source_count": 2}


@pytest.mark.asyncio
async def test_truncated_flag_propagated(monkeypatch):
    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id="acme"))
    monkeypatch.setattr(agent, "is_memory_enabled", lambda: True)
    agent.memory_manager = MagicMock()
    agent._memory_tenant_id = "acme"

    truncated_graph = CitationGraph(
        root_memory_id="m_root",
        nodes=[_node("m_root", depth=0)],
        primary_sources=[],
        truncated_at_max_depth=True,
    )

    import cogniverse_core.memory.provenance as prov_mod

    class FakeWalker:
        def __init__(self, *a, **kw):
            pass

        def walk(self, memory_id, tenant_id):
            return truncated_graph

    monkeypatch.setattr(prov_mod, "ProvenanceWalker", FakeWalker)

    out = await agent._process_impl(
        CitationTracingInput(memory_id="m_root", tenant_id="acme")
    )
    assert out.truncated is True


def test_input_validation_bounds():
    """Walker bounds must be honoured by Pydantic on input construction."""
    # Below floor
    with pytest.raises(Exception):  # pydantic ValidationError
        CitationTracingInput(memory_id="m", max_depth=0)
    # Above ceiling
    with pytest.raises(Exception):
        CitationTracingInput(memory_id="m", max_depth=999)
    with pytest.raises(Exception):
        CitationTracingInput(memory_id="m", max_nodes=0)
    with pytest.raises(Exception):
        CitationTracingInput(memory_id="m", max_nodes=10000)


def test_agent_capabilities_advertised():
    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id="acme"))
    assert agent.agent_name == "citation_tracing_agent"
    assert "citation_tracing" in agent.capabilities
    assert "provenance_walk" in agent.capabilities
    assert agent.port == 8019
