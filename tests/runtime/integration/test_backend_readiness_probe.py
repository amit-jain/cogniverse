"""Backend-readiness probe in the runtime lifespan.

``_wait_for_backend_ready`` runs at startup inside the async lifespan, so it
must use async HTTP + ``asyncio.sleep`` — a blocking ``httpx.get`` /
``time.sleep`` would freeze the event loop for up to five minutes while Vespa
converges. These tests prove it returns True against a real Vespa and keeps
the loop responsive while retrying an unreachable backend.
"""

import asyncio

import pytest

from cogniverse_runtime.main import _wait_for_backend_ready

pytestmark = pytest.mark.integration


async def test_wait_for_backend_ready_against_real_vespa(vespa_instance):
    vespa_base = f"http://localhost:{vespa_instance['http_port']}"
    ready = await _wait_for_backend_ready(
        vespa_base, max_attempts=12, retry_interval=2.0, timeout=5.0
    )
    assert ready is True


async def test_wait_for_backend_ready_returns_false_when_unreachable():
    ready = await _wait_for_backend_ready(
        "http://127.0.0.1:1", max_attempts=3, retry_interval=0.05, timeout=0.5
    )
    assert ready is False


async def test_wait_for_backend_ready_does_not_block_event_loop():
    stop = asyncio.Event()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    ready = await _wait_for_backend_ready(
        "http://127.0.0.1:1", max_attempts=3, retry_interval=0.05, timeout=0.5
    )
    stop.set()
    await ticker_task

    assert ready is False
    # A blocking time.sleep across the three retries would freeze the loop so
    # the concurrent ticker never advances; async sleep lets it keep ticking.
    assert ticks >= 5
