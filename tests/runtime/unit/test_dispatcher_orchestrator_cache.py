"""Orchestration resolves through a per-tenant cache (dispatch + streaming).

The orchestration path built a fresh OrchestratorAgent per complex query and
re-read the workflow corpus (4+ Phoenix reads) each time; the streaming path
built a bare one that never loaded templates at all. Both now resolve through
``_get_or_build_orchestrator``, which builds once per tenant, loads the artifact
on build, TTL-reloads it on cache hits, and funnels concurrent cold builds
through a single in-flight build.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def dispatcher(monkeypatch):
    import cogniverse_agents.orchestrator_agent as orch_mod
    import cogniverse_foundation.telemetry.manager as tm_mod

    monkeypatch.setattr(tm_mod, "get_telemetry_manager", lambda *a, **k: None)

    class _StubOrch:
        def __init__(self, **k):
            self.telemetry_manager = None
            self._artifact_tenant_id = None
            self.loaded = 0

        def _load_artifact(self):
            self.loaded += 1

    monkeypatch.setattr(orch_mod, "OrchestratorAgent", _StubOrch)

    d = object.__new__(AgentDispatcher)
    d._registry = SimpleNamespace(
        get_agent=lambda name: SimpleNamespace(capabilities=["orchestration"])
    )
    d._config_manager = SimpleNamespace()
    d._sandbox_manager = None
    d._init_agent_memory = lambda *a, **k: None
    return d


@pytest.mark.asyncio
async def test_build_loads_the_artifact_once(dispatcher):
    agent = await dispatcher._get_or_build_orchestrator("acme:acme")
    assert agent._artifact_tenant_id == "acme:acme"
    assert agent.loaded == 1  # workflow templates loaded on build


@pytest.mark.asyncio
async def test_same_tenant_reuses_cached_orchestrator(dispatcher):
    first = await dispatcher._get_or_build_orchestrator("acme:acme")
    second = await dispatcher._get_or_build_orchestrator("acme:acme")
    assert first is second
    assert first.loaded == 1  # no rebuild, no re-read within the TTL


@pytest.mark.asyncio
async def test_streaming_loads_templates_and_shares_instance(dispatcher):
    dispatched = await dispatcher._get_or_build_orchestrator("acme:acme")
    streamed, typed_input = await dispatcher.create_streaming_agent(
        "orchestrator_agent", "plan this", "acme:acme"
    )
    # Pre-fix the streaming path built a bare agent that never loaded templates.
    assert streamed is dispatched
    assert streamed.loaded == 1
    assert typed_input.tenant_id == "acme:acme"


@pytest.mark.asyncio
async def test_cached_orchestrator_reloads_after_ttl(dispatcher):
    dispatcher._orchestrator_artifact_ttl_s = 0.0
    first = await dispatcher._get_or_build_orchestrator("acme:acme")
    assert first.loaded == 1
    second = await dispatcher._get_or_build_orchestrator("acme:acme")
    assert second is first
    assert first.loaded == 2  # TTL elapsed → re-read on the same instance


@pytest.mark.asyncio
async def test_concurrent_cold_builds_run_once(dispatcher):
    calls = {"n": 0}
    real_build = dispatcher._build_orchestrator_agent

    async def _counting(tenant_id):
        calls["n"] += 1
        await asyncio.sleep(0.05)  # widen the window so first-touches overlap
        return await real_build(tenant_id)

    dispatcher._build_orchestrator_agent = _counting
    results = await asyncio.gather(
        dispatcher._get_or_build_orchestrator("acme:acme"),
        dispatcher._get_or_build_orchestrator("acme:acme"),
        dispatcher._get_or_build_orchestrator("acme:acme"),
    )
    assert calls["n"] == 1, (
        "concurrent cold dispatches ran duplicate orchestrator builds"
    )
    assert results[0] is results[1] is results[2]
