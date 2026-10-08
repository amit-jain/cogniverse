"""Search paths must not run Mem0's LLM fact-extraction on the event loop,
nor wait for it before answering.

``remember_success`` -> Mem0 ``add(infer=True)`` is a blocking LLM
chat-completion round trip. The ensemble and the single-modality search paths
hand it to the shared background memory writer, which runs it on its own
threads after the search returns. The proof is deterministic: the write
records ``threading.get_ident()``, which must differ from the event loop's
thread, and a held write does not hold the search.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cogniverse_agents.background_memory_writes import drain_background_memory_writes
from cogniverse_agents.search_agent import EnsembleOutcome, SearchAgent

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _FakeDoc:
    def __init__(self, doc_id):
        self.id = doc_id
        self.metadata = {"title": f"doc-{doc_id}"}


class _FakeSearchResult:
    def __init__(self, doc_id, score):
        self.document = _FakeDoc(doc_id)
        self.score = score


@pytest.mark.asyncio
async def test_ensemble_remember_success_runs_off_the_event_loop():
    agent = object.__new__(SearchAgent)
    agent._backend_type = "vespa"
    agent.search_config = {"backend": {"profiles": {}}}
    agent.active_profile = "p1"
    agent.query_encoder = SimpleNamespace(
        encode=lambda q: np.zeros((1, 4), dtype=np.float32)
    )

    recorded: dict = {}

    def _rec_remember_success(**kwargs):
        recorded["thread"] = threading.get_ident()
        return True

    agent.is_memory_enabled = lambda: True
    agent._memory_agent_name = "search_agent"
    agent._memory_tenant_id = "acme:acme"
    agent.remember_success = _rec_remember_success
    agent._get_backend = lambda: SimpleNamespace(
        search=lambda query_dict: [_FakeSearchResult("d1", 0.9)]
    )
    agent._build_date_filter = lambda *a, **k: None
    agent._fuse_results_rrf = lambda profile_results, k, top_k: [
        {"id": "d1", "score": 0.9}
    ]

    loop_thread = threading.get_ident()
    outcome = await agent._search_ensemble(
        "robot dancing", tenant_id="acme:acme", profiles=["p1"], top_k=5
    )
    assert await drain_background_memory_writes(5.0) is True

    assert outcome == EnsembleOutcome(
        results=[{"id": "d1", "score": 0.9}], searched=("p1",), degraded=()
    )
    assert set(recorded) == {"thread"}
    # to_thread offload => the blocking Mem0 add ran on a worker thread.
    assert recorded["thread"] != loop_thread


def _single_modality_agent(**overrides):
    agent = object.__new__(SearchAgent)
    agent._backend_type = "vespa"
    agent.search_config = {"backend": {"profiles": {}}}
    agent.active_profile = "p1"
    agent.is_memory_enabled = lambda: True
    agent._memory_agent_name = "search_agent"
    agent._memory_tenant_id = "acme:acme"
    agent._build_date_filter = lambda *a, **k: None
    agent.get_relevant_context = lambda query, top_k=3: None
    agent.query_encoder = SimpleNamespace(
        encode=lambda q: np.zeros((1, 4), dtype=np.float32)
    )
    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent


@pytest.mark.asyncio
async def test_ensemble_answers_while_its_memory_write_is_held():
    release = threading.Event()
    landed: list[str] = []

    def remember_success(**kwargs):
        assert release.wait(5) is True
        landed.append(kwargs["query"])

    agent = _single_modality_agent(
        search_config={"backend": {"profiles": {}}},
        remember_success=remember_success,
        _get_backend=lambda: SimpleNamespace(
            search=lambda query_dict: [_FakeSearchResult("d1", 0.9)]
        ),
        _fuse_results_rrf=lambda profile_results, k, top_k: [
            {"id": "d1", "score": 0.9}
        ],
    )

    try:
        outcome = await asyncio.wait_for(
            agent._search_ensemble(
                "robot dancing", tenant_id="acme:acme", profiles=["p1"], top_k=5
            ),
            timeout=2.0,
        )
        landed_when_answered = list(landed)
    finally:
        release.set()
    assert await drain_background_memory_writes(5.0) is True

    assert outcome.results == [{"id": "d1", "score": 0.9}]
    assert landed_when_answered == []
    assert landed == ["robot dancing"]


@pytest.mark.asyncio
async def test_text_search_answers_while_its_memory_write_is_held():
    """``_search_by_text`` runs inside ``asyncio.to_thread``; its success
    memory is queued from that worker thread, not written on it."""
    release = threading.Event()
    recorded: dict = {}

    def remember_success(**kwargs):
        assert release.wait(5) is True
        recorded["thread"] = threading.get_ident()
        recorded["query"] = kwargs["query"]

    agent = _single_modality_agent(
        remember_success=remember_success,
        _search_backend=lambda query_dict: [_FakeSearchResult("d1", 0.9)],
    )

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                agent._search_by_text,
                query="robot dancing",
                tenant_id="acme:acme",
                modality="video",
                top_k=5,
            ),
            timeout=2.0,
        )
        recorded_when_answered = dict(recorded)
    finally:
        release.set()
    assert await drain_background_memory_writes(5.0) is True

    assert [r["id"] for r in results] == ["d1"]
    assert recorded_when_answered == {}
    assert recorded["query"] == "robot dancing"
    assert recorded["thread"] != threading.get_ident()


@pytest.mark.asyncio
async def test_a_failed_text_search_raises_without_waiting_for_its_failure_memory():
    release = threading.Event()
    recorded: list[dict] = []

    def remember_failure(**kwargs):
        assert release.wait(5) is True
        recorded.append(kwargs)

    def backend_down(query_dict):
        raise ConnectionError("vespa refused the query")

    agent = _single_modality_agent(
        remember_failure=remember_failure, _search_backend=backend_down
    )

    try:
        with pytest.raises(ConnectionError) as raised:
            await asyncio.wait_for(
                asyncio.to_thread(
                    agent._search_by_text,
                    query="robot dancing",
                    tenant_id="acme:acme",
                    modality="video",
                    top_k=5,
                ),
                timeout=2.0,
            )
        recorded_when_raised = list(recorded)
    finally:
        release.set()
    assert await drain_background_memory_writes(5.0) is True

    assert str(raised.value) == "vespa refused the query"
    assert recorded_when_raised == []
    assert recorded == [
        {
            "query": "robot dancing",
            "error": "vespa refused the query",
            "metadata": {"search_type": "text", "modality": "video", "top_k": 5},
        }
    ]


@pytest.mark.asyncio
async def test_a_failing_search_memory_write_is_logged_and_the_search_answers(
    caplog,
):
    def remember_success(**kwargs):
        raise RuntimeError("mem0 vector store unreachable")

    agent = _single_modality_agent(
        remember_success=remember_success,
        _search_backend=lambda query_dict: [_FakeSearchResult("d1", 0.9)],
    )

    with caplog.at_level(logging.ERROR):
        results = await asyncio.to_thread(
            agent._search_by_text,
            query="robot dancing",
            tenant_id="acme:acme",
            modality="video",
            top_k=5,
        )
        assert await drain_background_memory_writes(5.0) is True

    assert [r["id"] for r in results] == ["d1"]
    logged = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]
    assert len(logged) == 1
    assert "acme:acme" in logged[0]
    assert "search_agent" in logged[0]
    assert "mem0 vector store unreachable" in logged[0]
