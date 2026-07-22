"""Backend-readiness probe in the runtime lifespan.

``_wait_for_backend_ready`` runs at startup inside the async lifespan, so it
must use async HTTP + ``asyncio.sleep`` — a blocking ``httpx.get`` /
``time.sleep`` would freeze the event loop for up to five minutes while Vespa
converges. These tests prove it returns True against a real Vespa and keeps
the loop responsive while retrying an unreachable backend.
"""

import asyncio
import socket

import pytest

from cogniverse_runtime.main import (
    _wait_for_backend_ready,
    _wait_for_config_server,
)

pytestmark = pytest.mark.integration


def test_wait_for_config_server_true_when_port_accepts():
    """A cold Vespa opens its query port before its config/deploy server, so
    the metadata bootstrap waits for the config server to accept connections
    rather than deploying blind and crash-looping the whole runtime."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        assert _wait_for_config_server("127.0.0.1", port, max_attempts=1) is True
    finally:
        listener.close()


def test_wait_for_config_server_false_when_refused():
    # Bind then close so the port is definitely free (connection refused),
    # and cap attempts so the bounded wait returns quickly.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert (
        _wait_for_config_server("127.0.0.1", port, max_attempts=3, interval=0.05)
        is False
    )


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
