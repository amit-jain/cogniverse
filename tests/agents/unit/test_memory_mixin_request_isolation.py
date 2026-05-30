"""Per-request state on a SHARED MemoryAwareMixin agent must not bleed.

Regression (request-bleed): the dispatcher caches and shares one agent instance
across requests (_get_search_agent, _gateway_agent). The artefact overlay +
session id used to be instance attributes, so two concurrent requests racing on
the same cached agent overwrote each other. They now live in ContextVars
(per-asyncio-task), so each in-flight request sees only its own value.
"""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin


class _Agent(MemoryAwareMixin):
    pass


@pytest.mark.asyncio
async def test_overlay_does_not_bleed_across_concurrent_requests():
    """One shared agent, two interleaved requests: each must read back its own
    overlay even though the other set a different one in between."""
    shared = _Agent()  # the single cached instance both requests use
    seen: dict[str, dict] = {}
    barrier = asyncio.Event()

    async def request(name: str, overlay: dict, wait_for_other: bool):
        # Set this request's overlay on the SHARED agent.
        shared.set_dispatched_artefact(overlay)
        if wait_for_other:
            # Yield so the other request runs and sets ITS overlay in between.
            await barrier.wait()
        else:
            await asyncio.sleep(0)
            barrier.set()
        # Read back AFTER the other request has set a different overlay.
        seen[name] = shared.get_dispatched_artefact()

    await asyncio.gather(
        request("A", {"served_from": "canary", "version": 1}, wait_for_other=True),
        request("B", {"served_from": "active", "version": 2}, wait_for_other=False),
    )

    # No bleed: each request reads back exactly what IT set.
    assert seen["A"] == {"served_from": "canary", "version": 1}
    assert seen["B"] == {"served_from": "active", "version": 2}


@pytest.mark.asyncio
async def test_session_id_does_not_bleed_across_concurrent_requests():
    shared = _Agent()
    seen: dict[str, str | None] = {}
    barrier = asyncio.Event()

    async def request(name: str, sid: str, wait_for_other: bool):
        shared.set_session_id(sid)
        if wait_for_other:
            await barrier.wait()
        else:
            await asyncio.sleep(0)
            barrier.set()
        seen[name] = shared.get_session_id()

    await asyncio.gather(
        request("A", "session_A", wait_for_other=True),
        request("B", "session_B", wait_for_other=False),
    )

    assert seen["A"] == "session_A"
    assert seen["B"] == "session_B"
