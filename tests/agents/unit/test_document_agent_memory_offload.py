"""Document search must not run Mem0's LLM fact extraction on the event loop,
nor wait for it before answering.

``remember_success`` / ``remember_failure`` reach
``core/memory/manager.add(..., infer=True)``, a blocking LLM chat-completion
round trip. ``search_documents`` hands them to the shared background memory
writer, which runs them on its own threads after the response returns.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from cogniverse_agents.background_memory_writes import drain_background_memory_writes
from cogniverse_agents.document_agent import DocumentAgent, DocumentResult

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _hit() -> DocumentResult:
    return DocumentResult(
        document_id="doc-7",
        document_url="s3://docs/doc-7.pdf",
        title="Quarterly filing",
        relevance_score=0.91,
        strategy_used="text",
    )


def _agent(**overrides):
    agent = object.__new__(DocumentAgent)
    agent.is_memory_enabled = lambda: True
    agent._memory_agent_name = "document_agent"
    agent._memory_tenant_id = "acme:acme"
    agent.get_relevant_context = lambda query, top_k=3: ""
    agent._deployed_strategy = lambda strategy: strategy
    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent


@pytest.mark.asyncio
async def test_success_memory_write_runs_off_the_event_loop():
    recorded: dict = {}

    def remember_success(**kwargs):
        recorded["thread"] = threading.get_ident()
        recorded["kwargs"] = kwargs

    async def search_text(query, limit):
        return [_hit()]

    agent = _agent(remember_success=remember_success, _search_text=search_text)

    loop_thread = threading.get_ident()
    results = await agent.search_documents("filing", strategy="text", limit=4)
    assert await drain_background_memory_writes(5.0) is True

    assert [r.document_id for r in results] == ["doc-7"]
    assert recorded["thread"] != loop_thread
    assert recorded["kwargs"] == {
        "query": "filing",
        "result": {
            "result_count": 1,
            "strategy": "text",
            "top_result": "Quarterly filing",
        },
        "metadata": {"search_strategy": "text", "limit": 4},
    }


@pytest.mark.asyncio
async def test_failure_memory_write_runs_off_the_event_loop():
    recorded: dict = {}

    def remember_failure(**kwargs):
        recorded["thread"] = threading.get_ident()
        recorded["kwargs"] = kwargs

    async def search_text(query, limit):
        raise ConnectionError("vespa document backend refused the query")

    agent = _agent(remember_failure=remember_failure, _search_text=search_text)

    loop_thread = threading.get_ident()
    with pytest.raises(ConnectionError) as raised:
        await agent.search_documents("filing", strategy="text", limit=4)
    assert await drain_background_memory_writes(5.0) is True

    assert str(raised.value) == "vespa document backend refused the query"
    assert recorded["thread"] != loop_thread
    assert recorded["kwargs"] == {
        "query": "filing",
        "error": "vespa document backend refused the query",
        "metadata": {"search_strategy": "text", "limit": 4},
    }


@pytest.mark.asyncio
async def test_searches_answer_while_their_memory_writes_are_still_held():
    """Two searches answer while both memory writes are held; the writes land
    once released."""
    started: list[str] = []
    landed: list[str] = []
    release = threading.Event()
    lock = threading.Lock()

    def remember_success(**kwargs):
        with lock:
            started.append(kwargs["query"])
        assert release.wait(5) is True
        with lock:
            landed.append(kwargs["query"])

    async def search_text(query, limit):
        return [_hit()]

    agent = _agent(remember_success=remember_success, _search_text=search_text)

    try:
        first, second = await asyncio.wait_for(
            asyncio.gather(
                agent.search_documents("alpha", strategy="text", limit=1),
                agent.search_documents("beta", strategy="text", limit=1),
            ),
            timeout=2.0,
        )
        landed_when_answered = list(landed)
    finally:
        release.set()
    assert await drain_background_memory_writes(5.0) is True

    assert [r.document_id for r in first + second] == ["doc-7", "doc-7"]
    assert landed_when_answered == []
    assert sorted(started) == ["alpha", "beta"]
    assert sorted(landed) == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_a_memory_backend_failure_is_logged_and_the_search_still_answers(
    caplog,
):
    """A memory outage never fails the search: the write's error is logged
    with the tenant and agent, and no failure memory is recorded for a search
    that succeeded."""
    failures: list[dict] = []

    def remember_success(**kwargs):
        raise RuntimeError("mem0 vector store unreachable")

    def remember_failure(**kwargs):
        failures.append(kwargs)

    async def search_text(query, limit):
        return [_hit()]

    agent = _agent(
        remember_success=remember_success,
        remember_failure=remember_failure,
        _search_text=search_text,
    )

    with caplog.at_level(logging.ERROR):
        results = await agent.search_documents("filing", strategy="text", limit=4)
        assert await drain_background_memory_writes(5.0) is True

    assert [r.document_id for r in results] == ["doc-7"]
    assert failures == []
    logged = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]
    assert len(logged) == 1
    assert "acme:acme" in logged[0]
    assert "document_agent" in logged[0]
    assert "mem0 vector store unreachable" in logged[0]
