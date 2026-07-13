"""The ensemble search path must not run Mem0's LLM fact-extraction on the
event loop.

``remember_success`` -> Mem0 ``add(infer=True)`` is a blocking LLM
chat-completion round trip. ``_search_ensemble`` is ``async``, so calling it
directly stalled every concurrent request (including /health/live) for the
duration of the LLM call. It now runs via ``asyncio.to_thread``, mirroring
orchestrator_agent. The proof is deterministic: the offloaded write records
``threading.get_ident()``, which must differ from the event loop's thread.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cogniverse_agents.search_agent import SearchAgent


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
    agent.remember_success = _rec_remember_success
    agent._get_backend = lambda: SimpleNamespace(
        search=lambda query_dict: [_FakeSearchResult("d1", 0.9)]
    )
    agent._build_date_filter = lambda *a, **k: None
    agent._fuse_results_rrf = lambda profile_results, k, top_k: [
        {"id": "d1", "score": 0.9}
    ]

    loop_thread = threading.get_ident()
    results = await agent._search_ensemble(
        "robot dancing", tenant_id="acme:acme", profiles=["p1"], top_k=5
    )

    assert results == [{"id": "d1", "score": 0.9}]
    assert recorded.get("thread") is not None
    # to_thread offload => the blocking Mem0 add ran on a worker thread.
    assert recorded["thread"] != loop_thread
