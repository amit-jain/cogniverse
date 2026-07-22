"""Search-agent FastAPI routes must not block the event loop.

Regression (PERF): the standalone ``/process``, ``/upload/*`` and
``/search/enhanced`` routes are ``async def`` but call synchronous,
network-heavy agent methods. Run directly on the loop they stall every other
coroutine for the whole search. These prove the blocking call is offloaded to
a worker thread — a coroutine scheduled alongside the request still runs and
releases it while the search is in flight.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from httpx import ASGITransport

import cogniverse_agents.search_agent as sa

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.asyncio
async def test_process_route_offloads_blocking_call(monkeypatch):
    release = threading.Event()

    class _Stub:
        def process_enhanced_task(self, task):
            # Off-loop: the releaser sets the event within 50ms. On-loop: the
            # loop is frozen here so wait() times out and the route 500s.
            assert release.wait(timeout=5), "event loop blocked by sync call"
            return {"ok": True, "task_id": task.get("id")}

    monkeypatch.setattr(sa, "search_agent", _Stub())

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    transport = ASGITransport(app=sa.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp, _ = await asyncio.wait_for(
            asyncio.gather(client.post("/process", json={"id": "abc"}), releaser()),
            timeout=5,
        )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "task_id": "abc"}


@pytest.mark.asyncio
async def test_enhanced_search_route_offloads_blocking_call(monkeypatch):
    release = threading.Event()

    class _Stub:
        def search_with_relationship_context(self, search_context, **kwargs):
            assert release.wait(timeout=5), "event loop blocked by sync call"
            return {"results": [], "tenant_id": kwargs.get("tenant_id")}

    monkeypatch.setattr(sa, "search_agent", _Stub())

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    transport = ASGITransport(app=sa.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp, _ = await asyncio.wait_for(
            asyncio.gather(
                client.post(
                    "/search/enhanced",
                    json={"query": "cats", "tenant_id": "acme"},
                ),
                releaser(),
            ),
            timeout=5,
        )

    assert resp.status_code == 200
    assert resp.json() == {"results": [], "tenant_id": "acme"}
