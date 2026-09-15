"""Document search must not run Mem0's LLM fact extraction on the event loop.

``remember_success`` / ``remember_failure`` reach
``core/memory/manager.add(..., infer=True)``, a blocking LLM chat-completion
round trip. ``search_documents`` is ``async``, so calling them inline stalled
every concurrent request on the replica for the duration of that call. They
now run via ``asyncio.to_thread``, as the orchestrator and search agents
already do.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

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

    assert str(raised.value) == "vespa document backend refused the query"
    assert recorded["thread"] != loop_thread
    assert recorded["kwargs"] == {
        "query": "filing",
        "error": "vespa document backend refused the query",
        "metadata": {"search_strategy": "text", "limit": 4},
    }


@pytest.mark.asyncio
async def test_concurrent_requests_are_served_while_memory_writes_are_in_flight():
    """Two searches hold their memory writes; a third request still completes."""
    started: list[str] = []
    order: list[str] = []
    release = threading.Event()
    lock = threading.Lock()

    def remember_success(**kwargs):
        with lock:
            started.append(kwargs["query"])
        assert release.wait(5) is True
        order.append(kwargs["query"])

    async def search_text(query, limit):
        return [_hit()]

    agent = _agent(remember_success=remember_success, _search_text=search_text)

    async def unrelated_request():
        for _ in range(500):
            if len(started) == 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("the event loop never ran while memory writes were pending")
        order.append("health")
        release.set()
        return {"status": "healthy"}

    first, second, health = await asyncio.gather(
        agent.search_documents("alpha", strategy="text", limit=1),
        agent.search_documents("beta", strategy="text", limit=1),
        unrelated_request(),
    )

    assert health == {"status": "healthy"}
    assert order[0] == "health"
    assert sorted(order[1:]) == ["alpha", "beta"]
    assert [r.document_id for r in first + second] == ["doc-7", "doc-7"]


@pytest.mark.asyncio
async def test_memory_backend_failure_surfaces_instead_of_a_silent_result():
    """A memory outage raises with its own detail and is itself recorded."""
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

    with pytest.raises(RuntimeError) as raised:
        await agent.search_documents("filing", strategy="text", limit=4)

    assert str(raised.value) == "mem0 vector store unreachable"
    assert failures == [
        {
            "query": "filing",
            "error": "mem0 vector store unreachable",
            "metadata": {"search_strategy": "text", "limit": 4},
        }
    ]
