"""DeepResearchAgent evidence-gathering resilience and tenant LM routing.

One failing sub-question search must not abort the whole research run, the
evidence summary's result count must not misreport non-list payloads, and the
whole research run must bind the REQUEST tenant's LM — every sibling answer
agent routes through ``routed_lm_context_for``; deep research silently ran on
the process-global default LM for every tenant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import dspy
import pytest

from cogniverse_agents.deep_research_agent import DeepResearchAgent, DeepResearchInput

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.asyncio
async def test_one_failed_subsearch_does_not_abort_the_rest():
    agent = object.__new__(DeepResearchAgent)

    async def search_fn(query, tenant_id):
        if query == "boom":
            raise RuntimeError("backend down")
        return [{"document_id": "d1"}]

    agent._search_fn = search_fn

    evidence = await agent._search_parallel(["ok", "boom"], tenant_id="t")

    by_q = {e["question"]: e for e in evidence}
    assert by_q["ok"]["results"] == [{"document_id": "d1"}]
    assert by_q["boom"]["results"] == []
    assert "backend down" in by_q["boom"]["error"]


@pytest.mark.asyncio
async def test_total_search_outage_raises_not_synthesizes():
    """When EVERY sub-question search errors (search backend down), the run
    must raise — not synthesize a confident summary over zero evidence, which
    reads as a genuine answer. A partial failure still proceeds (above)."""
    agent = object.__new__(DeepResearchAgent)

    async def failing_search(query, tenant_id):
        raise RuntimeError("vespa down")

    agent._search_fn = failing_search

    # Isolate the search-outage guard from the pre-search context stack + LM.
    agent.set_tenant_for_context = lambda t: None

    async def _enrich(a, b):
        return a

    agent.inject_context_into_prompt_async = _enrich
    agent.emit_progress = lambda *a, **k: None

    async def _decompose(q):
        return ["q1", "q2"]

    agent._decompose = _decompose

    async def _must_not_run(*a, **k):
        raise AssertionError("synthesized/evaluated over zero evidence")

    agent._evaluate_evidence = _must_not_run
    agent._synthesize = _must_not_run

    inp = DeepResearchInput(
        query="what happened?", max_iterations=2, tenant_id="acme:acme"
    )
    with pytest.raises(RuntimeError, match="search backend unavailable"):
        await agent._research(inp)


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
    endpoint = MagicMock(name="deep_research_endpoint")
    cfg.get_llm_config.return_value.resolve.return_value = endpoint

    captured: dict = {}

    def fake_create_routed_lm(ep, router, tenant_id):
        captured["endpoint"] = ep
        captured["tenant_id"] = tenant_id
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
