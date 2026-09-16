"""DeepResearchAgent evidence-gathering contract and tenant LM routing.

A failed sub-question search fails the research run, the evidence summary's
result count must not misreport non-list payloads, and the whole research run
must bind the REQUEST tenant's LM — every sibling answer agent routes through
``routed_lm_context_for``; deep research silently ran on the process-global
default LM for every tenant.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import dspy
import pytest

from cogniverse_agents.deep_research_agent import DeepResearchAgent, DeepResearchInput
from tests.utils.tenant_helpers import config_manager_with_tiers

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _isolated_agent(search_fn, sub_questions=("q1", "q2")):
    """A DeepResearchAgent with only the search leg live.

    The decomposer, the context stack and the LM bind are stubbed so the
    assertions read the search leg's own outcome.
    """
    agent = object.__new__(DeepResearchAgent)
    agent._search_fn = search_fn
    agent.set_tenant_for_context = lambda t: None
    agent.emit_progress = lambda *a, **k: None

    async def _enrich(a, b):
        return a

    agent.inject_context_into_prompt_async = _enrich

    async def _decompose(q):
        return list(sub_questions)

    agent._decompose = _decompose

    async def _must_not_run(*a, **k):
        raise AssertionError("evaluated or synthesized over failed retrieval")

    agent._evaluate_evidence = _must_not_run
    agent._synthesize = _must_not_run
    return agent


@pytest.mark.asyncio
async def test_one_failed_subsearch_fails_the_search_leg():
    """A sub-question search that fails is a retrieval outage, not empty
    evidence: it propagates with its own identity and message rather than
    becoming ``{"results": [], "error": ...}`` the synthesis reads as "nothing
    found"."""
    agent = object.__new__(DeepResearchAgent)
    outage = RuntimeError("backend down")

    async def search_fn(query, tenant_id):
        if query == "boom":
            raise outage
        return [{"document_id": "d1"}]

    agent._search_fn = search_fn

    with pytest.raises(RuntimeError) as failure:
        await agent._search_parallel(["ok", "boom"], tenant_id="t")

    assert failure.value is outage
    assert str(failure.value) == "backend down"


@pytest.mark.asyncio
async def test_failed_subsearch_cancels_its_siblings():
    """The failure ends the leg: a sibling search still in flight against the
    same backend is cancelled, not left running past the failed turn."""
    agent = object.__new__(DeepResearchAgent)
    started = asyncio.Event()
    sibling_outcome: dict = {}

    async def search_fn(query, tenant_id):
        if query == "boom":
            await started.wait()
            raise RuntimeError("backend down")
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            sibling_outcome["cancelled"] = True
            raise
        sibling_outcome["cancelled"] = False
        return []

    agent._search_fn = search_fn

    with pytest.raises(RuntimeError, match="^backend down$"):
        await agent._search_parallel(["slow", "boom"], tenant_id="t")

    assert sibling_outcome == {"cancelled": True}


@pytest.mark.asyncio
async def test_total_search_outage_raises_not_synthesizes():
    """When the search backend is down the run must raise — not synthesize a
    confident summary over zero evidence, which reads as a genuine answer."""
    outage = RuntimeError("vespa down")

    async def failing_search(query, tenant_id):
        raise outage

    agent = _isolated_agent(failing_search)

    inp = DeepResearchInput(
        query="what happened?", max_iterations=2, tenant_id="acme:acme"
    )
    with pytest.raises(RuntimeError) as failure:
        await agent._research(inp)

    assert failure.value is outage


@pytest.mark.asyncio
async def test_a_failing_tenant_does_not_fail_a_concurrent_healthy_one():
    """Two research runs overlapping on the same event loop: the one whose
    backend is down fails with its own error and the healthy one returns its
    own evidence."""
    both_in_search = asyncio.Barrier(2)

    async def failing_search(query, tenant_id):
        await both_in_search.wait()
        raise RuntimeError("vespa down for acme")

    async def healthy_search(query, tenant_id):
        await both_in_search.wait()
        return [{"document_id": f"{tenant_id}-{query}"}]

    failing = _isolated_agent(failing_search, sub_questions=("q1",))
    healthy = object.__new__(DeepResearchAgent)
    healthy._search_fn = healthy_search

    failed, evidence = await asyncio.gather(
        failing._research(
            DeepResearchInput(query="q", max_iterations=1, tenant_id="acme:acme")
        ),
        healthy._search_parallel(["q1"], tenant_id="globex:globex"),
        return_exceptions=True,
    )

    assert isinstance(failed, RuntimeError)
    assert str(failed) == "vespa down for acme"
    assert evidence == [
        {
            "question": "q1",
            "results": [{"document_id": "globex:globex-q1"}],
            "source": "search",
        }
    ]


def test_result_count_handles_non_list_shapes():
    assert DeepResearchAgent._result_count([{"a": 1}, {"b": 2}]) == 2
    assert DeepResearchAgent._result_count([]) == 0
    assert DeepResearchAgent._result_count("some text") == 1
    assert DeepResearchAgent._result_count({"results": []}) == 1
    assert DeepResearchAgent._result_count(None) == 0


@pytest.mark.asyncio
async def test_research_runs_under_request_tenant_routed_lm(monkeypatch):
    """With the semantic router enabled, the entire research run must see the
    LM routed for the REQUEST tenant — not the ambient process-global LM."""
    from cogniverse_foundation.config.semantic_router import SemanticRouterConfig

    agent = object.__new__(DeepResearchAgent)
    agent._config_manager = object()

    sentinel_lm = MagicMock(name="tenant_routed_lm")
    ambient_lm = MagicMock(name="ambient_global_lm")
    seen: dict = {}

    async def fake_research(inp):
        seen["lm"] = dspy.settings.lm
        return "RESEARCH_DONE"

    agent._research = fake_research

    cfg = MagicMock()
    cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=True)
    cfg.config_manager = config_manager_with_tiers({"acme:acme": "pro"})
    endpoint = MagicMock(name="deep_research_endpoint")
    cfg.get_llm_config.return_value.resolve.return_value = endpoint

    captured: dict = {}

    def fake_create_routed_lm(ep, router, tenant_id, tier, call_site):
        captured["endpoint"] = ep
        captured["tenant_id"] = tenant_id
        captured["tier"] = tier
        return sentinel_lm

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda tenant_id, config_manager: cfg,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.semantic_router.create_routed_lm",
        fake_create_routed_lm,
    )

    with dspy.context(lm=ambient_lm):
        result = await agent._process_impl(
            DeepResearchInput(query="q", tenant_id="acme:acme")
        )

    assert result == "RESEARCH_DONE"
    assert seen["lm"] is sentinel_lm, (
        f"research ran on {seen['lm']!r}, not the tenant-routed LM"
    )
    assert captured["tier"] == "pro"
    assert captured["tenant_id"] == "acme:acme"
    assert captured["endpoint"] is endpoint
    assert cfg.get_llm_config.return_value.resolve.call_args[0][0] == (
        "deep_research_agent"
    )


@pytest.mark.asyncio
async def test_research_keeps_ambient_lm_without_config_manager():
    """Standalone process with no config store: routing cannot be enabled, so
    the ambient LM is the defined behaviour — unchanged by the wrap."""
    agent = object.__new__(DeepResearchAgent)
    agent._config_manager = None
    seen: dict = {}

    async def fake_research(inp):
        seen["lm"] = dspy.settings.lm
        return "OK"

    agent._research = fake_research

    ambient_lm = MagicMock(name="ambient_global_lm")
    with dspy.context(lm=ambient_lm):
        result = await agent._process_impl(
            DeepResearchInput(query="q", tenant_id="t:t")
        )

    assert result == "OK"
    assert seen["lm"] is ambient_lm
