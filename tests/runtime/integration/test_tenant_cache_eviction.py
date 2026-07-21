"""Eviction of a cached orchestrator releases its pooled http client.

Each cached per-tenant OrchestratorAgent owns a pooled ``httpx.AsyncClient``
(the policy-enforcing A2A client, built via
``SandboxManager.make_http_client``). When the LRU cache evicts a tenant, the
cache's ``on_evict`` must ``aclose`` that client, or its connection pool and
file descriptors leak for the pod's lifetime.

Also pins the graph-bind failure log at WARNING so a persistent graph-backend
outage that silently drops KG enrichment stays visible in the logs.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest

pytestmark = pytest.mark.integration


def _build_dispatcher(monkeypatch, capacity):
    """Construct an AgentDispatcher whose orchestrator cache is capped at
    ``capacity``, so a couple of distinct tenants force a real LRU eviction."""
    import cogniverse_foundation.telemetry.manager as tm_mod
    import cogniverse_runtime.agent_dispatcher as ad
    from cogniverse_runtime.agent_dispatcher import AgentDispatcher

    monkeypatch.setattr(
        tm_mod,
        "get_telemetry_manager",
        lambda *a, **k: SimpleNamespace(name="tm"),
        raising=True,
    )
    monkeypatch.setattr(ad, "ORCHESTRATOR_AGENT_CACHE_CAPACITY", capacity)
    return AgentDispatcher(
        agent_registry=SimpleNamespace(),
        config_manager=SimpleNamespace(),
        schema_loader=SimpleNamespace(),
    )


class _FakeOrchestrator:
    """Orchestrator stand-in holding a real pooled httpx client under the same
    attribute the served OrchestratorAgent stores it in."""

    def __init__(self):
        self._http_client_override = httpx.AsyncClient()

    def _load_artifact(self):
        return None


async def _poll_until_closed(client, timeout_s=1.0):
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if client.is_closed:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_eviction_closes_evicted_orchestrator_http_client(monkeypatch):
    d = _build_dispatcher(monkeypatch, capacity=1)

    built: dict[str, _FakeOrchestrator] = {}

    async def _fake_build(tenant_id):
        agent = _FakeOrchestrator()
        built[tenant_id] = agent
        return agent

    monkeypatch.setattr(d, "_build_orchestrator_agent", _fake_build)

    agent_a = await d._get_or_build_orchestrator("orchevict:a")
    # Capacity 1: building b evicts a; on_evict must close a's client.
    agent_b = await d._get_or_build_orchestrator("orchevict:b")

    assert agent_a is built["orchevict:a"]
    assert agent_b is built["orchevict:b"]

    await _poll_until_closed(agent_a._http_client_override)

    assert agent_a._http_client_override.is_closed is True
    assert agent_b._http_client_override.is_closed is False

    # The resident tenant keeps serving the same instance, client still open.
    assert await d._get_or_build_orchestrator("orchevict:b") is agent_b
    assert agent_b._http_client_override.is_closed is False

    await agent_b._http_client_override.aclose()


@pytest.mark.asyncio
async def test_set_displace_closes_prior_orchestrator_http_client(monkeypatch):
    d = _build_dispatcher(monkeypatch, capacity=8)

    first = _FakeOrchestrator()
    second = _FakeOrchestrator()

    from cogniverse_runtime.agent_dispatcher import _OrchestratorAgentEntry

    d._orchestrator_agents.set(
        "orchdisp:a", _OrchestratorAgentEntry(agent=first, loaded_at=0.0)
    )
    # Displacing the same key must release the prior entry's client.
    d._orchestrator_agents.set(
        "orchdisp:a", _OrchestratorAgentEntry(agent=second, loaded_at=0.0)
    )

    await _poll_until_closed(first._http_client_override)

    assert first._http_client_override.is_closed is True
    assert second._http_client_override.is_closed is False

    await second._http_client_override.aclose()


def test_graph_bind_failure_logs_at_warning(monkeypatch, caplog):
    import cogniverse_runtime.routers.graph as graph_mod

    d = _build_dispatcher(monkeypatch, capacity=64)

    def _boom(tenant_id, deploy=False):
        raise RuntimeError("graph backend unavailable")

    monkeypatch.setattr(graph_mod, "get_graph_manager", _boom)

    agent = SimpleNamespace(set_graph_manager=lambda gm: None)

    with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.agent_dispatcher"):
        d._bind_graph_manager(agent, "graphwarn:a")

    recs = [r for r in caplog.records if "Graph manager bind skipped" in r.getMessage()]
    assert len(recs) == 1
    assert recs[0].levelname == "WARNING"
    assert "graphwarn:a" in recs[0].getMessage()
    assert "graph backend unavailable" in recs[0].getMessage()
