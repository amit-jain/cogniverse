"""LM calls run on a dedicated executor and keep their dspy context.

Every dspy agent call parks a thread for the whole LM round-trip (up to the
request timeout). On the shared default executor — bounded small for ordinary
offloads — a burst of slow LM calls occupies every worker, so unrelated
``asyncio.to_thread`` users (search, admin, keyframes) queue behind LM traffic
runtime-wide. The dedicated pool confines that blast radius; the contextvars
snapshot must still carry the per-tenant ``dspy.context`` LM binding into the
worker thread.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import dspy
import pytest

from cogniverse_core.agents.base import _call_in_lm_executor

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.asyncio
async def test_slow_lm_calls_do_not_starve_default_executor():
    """20 concurrent 0.3s LM calls through a 2-worker default executor: on the
    old ``to_thread`` path they serialize (~3s) and a bystander offload waits
    behind them; on the dedicated pool they run wide and the bystander stays
    fast."""
    loop = asyncio.get_running_loop()
    tiny_default = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(tiny_default)
    try:

        def slow_lm_call():
            time.sleep(0.3)
            return "done"

        start = time.monotonic()
        lm_tasks = [
            asyncio.create_task(_call_in_lm_executor(slow_lm_call)) for _ in range(20)
        ]
        await asyncio.sleep(0.05)  # LM calls are in flight and hold threads

        bystander_start = time.monotonic()
        assert await asyncio.to_thread(lambda: "quick") == "quick"
        bystander_elapsed = time.monotonic() - bystander_start

        results = await asyncio.gather(*lm_tasks)
        total_elapsed = time.monotonic() - start

        assert results == ["done"] * 20
        assert bystander_elapsed < 0.5, (
            f"bystander offload waited {bystander_elapsed:.2f}s behind LM calls"
        )
        assert total_elapsed < 1.5, (
            f"20x0.3s LM calls took {total_elapsed:.2f}s — serialized on a "
            "shared pool instead of running on the dedicated one"
        )
    finally:
        tiny_default.shutdown(wait=False)


@pytest.mark.asyncio
async def test_dspy_context_propagates_into_lm_executor():
    """Two concurrent requests with different ``dspy.context`` LMs must each
    see their own LM inside the executor thread — losing the snapshot would
    silently serve one tenant with another tenant's model."""
    lm_a, lm_b = MagicMock(name="lm_a"), MagicMock(name="lm_b")

    def read_active_lm():
        return dspy.settings.lm

    async def request(lm):
        with dspy.context(lm=lm):
            return await _call_in_lm_executor(read_active_lm)

    seen_a, seen_b = await asyncio.gather(request(lm_a), request(lm_b))
    assert seen_a is lm_a
    assert seen_b is lm_b
