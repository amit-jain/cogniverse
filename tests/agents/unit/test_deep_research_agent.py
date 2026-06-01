"""DeepResearchAgent evidence-gathering resilience.

One failing sub-question search must not abort the whole research run, and the
evidence summary's result count must not misreport non-list payloads.
"""

from __future__ import annotations

import pytest

from cogniverse_agents.deep_research_agent import DeepResearchAgent


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


def test_result_count_handles_non_list_shapes():
    assert DeepResearchAgent._result_count([{"a": 1}, {"b": 2}]) == 2
    assert DeepResearchAgent._result_count([]) == 0
    assert DeepResearchAgent._result_count("some text") == 1
    assert DeepResearchAgent._result_count({"results": []}) == 1
    assert DeepResearchAgent._result_count(None) == 0
