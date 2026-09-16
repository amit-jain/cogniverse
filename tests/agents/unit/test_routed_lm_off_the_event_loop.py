"""Answer agents must bind their tenant-routed LM off the serving loop.

``routed_lm_context_for`` resolves the tenant's router tier through a
TTL-expiring ConfigStore read — a backend round trip with retries and backoff.
Called inline inside a coroutine it freezes every request, stream and health
probe on the replica for the length of that read. Each agent below sits on a
serving path: the summarizer and the detailed-report agent carry the answer
traffic, deep research and the orchestrator carry the multi-step paths.

The resolution runs in a worker thread and the resulting ``dspy.context`` is
entered on the request task, so the bind is still that request's.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import dspy
import pytest

from cogniverse_foundation.config.semantic_router import SemanticRouterConfig

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

BLOCK_S = 0.3


class _Stop(Exception):
    """Ends the agent call once the bound LM has been read."""


def _router_enabled(monkeypatch, tiers):
    """Enable the semantic router and give each tenant its own tier."""
    cfg = MagicMock()
    cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=True)
    cfg.get_llm_config.return_value.resolve.return_value = MagicMock(name="endpoint")
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda tenant_id=None, config_manager=None, **kw: cfg,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.tenant_tiers.resolve_tenant_tier",
        lambda config, tenant_id: tiers[tenant_id],
    )
    return cfg


def _blocking_routed_lm(monkeypatch, sink, block_s=BLOCK_S):
    """Replace the LM build with a blocking one that records its thread."""
    sentinels: dict = {}

    def _create(endpoint, router, tenant_id, tier, call_site):
        sink.append((threading.get_ident(), call_site, tenant_id, tier))
        time.sleep(block_s)
        return sentinels.setdefault(
            tenant_id, MagicMock(name=f"routed_lm::{tenant_id}")
        )

    monkeypatch.setattr(
        "cogniverse_foundation.config.semantic_router.create_routed_lm", _create
    )
    return sentinels


async def _ticks_during(awaitable):
    """Run ``awaitable``, returning the event-loop turns it allowed."""
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    try:
        await awaitable
    finally:
        stop.set()
        await ticker_task
    return ticks


def _summarizer(tenant_id):
    from cogniverse_agents.summarizer_agent import SummarizerAgent, SummarizerDeps
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    with (
        patch("cogniverse_agents.summarizer_agent.VLMInterface"),
        patch.object(SummarizerAgent, "_initialize_vlm_client"),
    ):
        agent = SummarizerAgent(deps=SummarizerDeps(), config_manager=Mock())
    agent._llm_config = LLMEndpointConfig(
        model="openai/test-model", api_base="http://localhost:11434/v1"
    )
    agent._memory_tenant_id = tenant_id
    return agent


async def _run_summarizer(agent, bound):
    from cogniverse_agents.summarizer_agent import SummaryRequest

    async def _record(request):
        bound["lm"] = dspy.settings.lm
        raise _Stop()

    agent._thinking_phase = _record
    with pytest.raises(_Stop):
        await agent._summarize(SummaryRequest(query="q", search_results=[]))


def _detailed_report(tenant_id):
    from cogniverse_agents.detailed_report_agent import (
        DetailedReportAgent,
        DetailedReportDeps,
    )
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    with (
        patch("cogniverse_agents.detailed_report_agent.VLMInterface"),
        patch.object(DetailedReportAgent, "_initialize_vlm_client"),
    ):
        agent = DetailedReportAgent(deps=DetailedReportDeps(), config_manager=Mock())
    agent._llm_config = LLMEndpointConfig(
        model="openai/test-model", api_base="http://localhost:11434/v1"
    )
    agent._memory_tenant_id = tenant_id
    return agent


async def _run_detailed_report(agent, bound):
    from cogniverse_agents.detailed_report_agent import ReportRequest

    async def _record(request):
        bound["lm"] = dspy.settings.lm
        raise _Stop()

    agent._thinking_phase = _record
    with pytest.raises(_Stop):
        await agent._generate_report(ReportRequest(query="q", search_results=[]))


def _deep_research(tenant_id):
    from cogniverse_agents.deep_research_agent import DeepResearchAgent

    agent = object.__new__(DeepResearchAgent)
    agent._config_manager = object()
    agent._research_tenant_id = tenant_id
    return agent


async def _run_deep_research(agent, bound):
    from cogniverse_agents.deep_research_agent import DeepResearchInput

    async def _record(_input):
        bound["lm"] = dspy.settings.lm
        raise _Stop()

    agent.validate_attachments = lambda _input: None
    agent._research = _record
    with pytest.raises(_Stop):
        await agent._process_impl(
            DeepResearchInput(query="q", tenant_id=agent._research_tenant_id)
        )


def _orchestrator(tenant_id):
    from cogniverse_agents.orchestrator_agent import OrchestratorAgent

    agent = object.__new__(OrchestratorAgent)
    agent._config_manager = object()
    agent._orchestrator_tenant_id = tenant_id
    return agent


async def _run_orchestrator(agent, bound):
    context = await agent._semantic_router_lm_context(agent._orchestrator_tenant_id)
    with context:
        bound["lm"] = dspy.settings.lm


AGENTS = {
    "summarizer_agent": (_summarizer, _run_summarizer),
    "detailed_report_agent": (_detailed_report, _run_detailed_report),
    "deep_research_agent": (_deep_research, _run_deep_research),
    "orchestrator_agent": (_orchestrator, _run_orchestrator),
}


@pytest.mark.parametrize("call_site", sorted(AGENTS))
async def test_agent_resolves_the_tenant_tier_off_the_event_loop(
    call_site, monkeypatch
):
    build, run = AGENTS[call_site]
    _router_enabled(monkeypatch, {"acme:acme": "pro"})
    calls: list = []
    sentinels = _blocking_routed_lm(monkeypatch, calls)
    bound: dict = {}

    agent = build("acme:acme")
    ambient = MagicMock(name="ambient_global_lm")
    with dspy.context(lm=ambient):
        ticks = await _ticks_during(run(agent, bound))

    assert [(entry[1], entry[2], entry[3]) for entry in calls] == [
        (call_site, "acme:acme", "pro")
    ]
    assert calls[0][0] != threading.get_ident()
    assert bound["lm"] is sentinels["acme:acme"]
    assert ticks >= 10, (
        f"only {ticks} event-loop turns during a {BLOCK_S}s tier resolution — "
        f"{call_site} resolved the tenant tier on the loop"
    )


@pytest.mark.parametrize("call_site", sorted(AGENTS))
async def test_concurrent_tenants_each_bind_their_own_routed_lm(call_site, monkeypatch):
    """Two requests overlapping on one loop: the tier resolution runs in a
    worker thread, but ``dspy.context`` binds through a ContextVar entered on
    each request's own task, so neither request sees the other's LM."""
    build, run = AGENTS[call_site]
    _router_enabled(monkeypatch, {"acme:acme": "pro", "globex:globex": "free"})
    calls: list = []
    sentinels = _blocking_routed_lm(monkeypatch, calls)
    seen: dict = {}

    async def _one(tenant_id):
        bound: dict = {}
        await run(build(tenant_id), bound)
        # Read after a further suspension: a bind that escaped to the loop's
        # own context would have been overwritten by the sibling request.
        await asyncio.sleep(0.05)
        seen[tenant_id] = (bound["lm"], dspy.settings.lm)

    ambient = MagicMock(name="ambient_global_lm")
    with dspy.context(lm=ambient):
        await asyncio.gather(_one("acme:acme"), _one("globex:globex"))

    assert sorted(entry[2] for entry in calls) == ["acme:acme", "globex:globex"]
    assert sorted(entry[3] for entry in calls) == ["free", "pro"]
    assert seen["acme:acme"] == (sentinels["acme:acme"], ambient)
    assert seen["globex:globex"] == (sentinels["globex:globex"], ambient)
