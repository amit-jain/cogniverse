"""DocumentAgent search must not block the event loop.

Regression (PERF): _search_visual / _search_text were ``async def`` but issued
a blocking ``requests.post`` directly on the loop, stalling every other
coroutine for the full Vespa round-trip; _search_hybrid then awaited them
sequentially despite a 'parallel' comment. These prove the HTTP call is
offloaded and the hybrid runs both concurrently.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cogniverse_agents.document_agent import DocumentAgent

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _FakeResp:
    status_code = 200

    def json(self):
        return {"root": {"children": []}}


def _bare_agent() -> DocumentAgent:
    """Construct without the heavy __init__; set only what _search_* read."""
    agent = DocumentAgent.__new__(DocumentAgent)
    agent._vespa_endpoint = "http://fake-vespa:8080"
    agent._tenant_id = "test:test"
    # query_encoder / text_query_encoder are lazy properties; set the backing
    # attrs so the real (heavy) encoders are never constructed.
    agent._query_encoder = SimpleNamespace(encode=lambda q: np.zeros((2, 3)))
    agent._text_query_encoder = SimpleNamespace(encode=lambda q: np.zeros((2, 3)))
    return agent


@pytest.mark.asyncio
async def test_search_visual_offloads_blocking_post(monkeypatch):
    """The blocking post must run in a worker thread: a coroutine scheduled
    alongside it must get to run (and release it) while it is in-flight. If the
    post ran on the loop, the releaser could never run and this would deadlock.
    """
    release = threading.Event()

    def blocking_post(url, json=None, timeout=None):
        # Only completes once the concurrent coroutine sets the event — which
        # can only happen if this call is OFF the event loop.
        assert release.wait(timeout=5), "event loop was blocked by requests.post"
        return _FakeResp()

    # document_agent does a local ``import requests`` inside the method, so
    # patch the real module's post (what that import resolves to).
    monkeypatch.setattr("requests.post", blocking_post)

    agent = _bare_agent()

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    # Will hang (then TimeoutError) on the old blocking code.
    results, _ = await asyncio.wait_for(
        asyncio.gather(agent._search_visual("q", 5), releaser()), timeout=5
    )
    assert results == []


@pytest.mark.asyncio
async def test_search_hybrid_runs_both_concurrently(monkeypatch):
    order: list[str] = []

    async def fake_visual(query, limit):
        order.append("v_start")
        await asyncio.sleep(0.1)
        order.append("v_end")
        return []

    async def fake_text(query, limit):
        order.append("t_start")
        await asyncio.sleep(0.1)
        order.append("t_end")
        return []

    agent = _bare_agent()
    monkeypatch.setattr(agent, "_search_visual", fake_visual)
    monkeypatch.setattr(agent, "_search_text", fake_text)

    await agent._search_hybrid("q", 5)

    # Concurrent execution: each starts before the other finishes. Sequential
    # awaits (the old code) would give [v_start, v_end, t_start, t_end].
    assert order.index("v_start") < order.index("t_end")
    assert order.index("t_start") < order.index("v_end")


def _patch_vespa_seam(monkeypatch):
    monkeypatch.setattr(
        "cogniverse_agents.search.vespa_query.vespa_search_post",
        lambda endpoint, params, timeout: _FakeResp(),
    )


@pytest.mark.asyncio
async def test_search_visual_offloads_blocking_encode(monkeypatch):
    """The ColPali query encode is a blocking HTTP/model call and must run in a
    worker thread like the Vespa post below it — inline on the loop it stalls
    every other coroutine for the whole encode round-trip."""
    release = threading.Event()

    def blocking_encode(query):
        assert release.wait(timeout=5), "event loop was blocked by query encode"
        return np.zeros((2, 3))

    agent = _bare_agent()
    agent._query_encoder = SimpleNamespace(encode=blocking_encode)
    _patch_vespa_seam(monkeypatch)

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    results, _ = await asyncio.wait_for(
        asyncio.gather(agent._search_visual("q", 5), releaser()), timeout=5
    )
    assert results == []


@pytest.mark.asyncio
async def test_search_text_offloads_blocking_encode(monkeypatch):
    """Same contract for the ColBERT text-query encode in _search_text."""
    release = threading.Event()

    def blocking_encode(query):
        assert release.wait(timeout=5), "event loop was blocked by query encode"
        return np.zeros((2, 3))

    agent = _bare_agent()
    agent._text_query_encoder = SimpleNamespace(encode=blocking_encode)
    _patch_vespa_seam(monkeypatch)

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    results, _ = await asyncio.wait_for(
        asyncio.gather(agent._search_text("q", 5), releaser()), timeout=5
    )
    assert results == []
