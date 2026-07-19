"""Gateway outbound drain loop + its runtime_client call.

The gateway polls the runtime's outbound queue and delivers each message via
its bot. A drain failure (runtime blip) is survived and retried; a per-message
send failure is isolated so one bad chat never stops the others; and the loop
task is cancelled cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cogniverse_messaging.gateway import MessagingGateway
from cogniverse_messaging.runtime_client import RuntimeClient

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _response(json_body):
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=json_body)
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_drain_outbound_gets_route_and_returns_messages():
    rc = RuntimeClient("http://runtime")
    http = AsyncMock()
    http.is_closed = False
    rc._client = http
    msgs = [{"chat_id": "1", "text": "a"}, {"chat_id": "2", "text": "b"}]
    http.get = AsyncMock(return_value=_response({"messages": msgs}))

    out = await rc.drain_outbound()

    http.get.assert_awaited_once_with("/admin/messaging/outbound/drain")
    assert out == msgs


def _gateway_with_bot(sends):
    gw = MessagingGateway.__new__(MessagingGateway)
    gw._outbound_poll_seconds = 0  # no real delay between ticks

    class _Bot:
        async def send_message(self, chat_id, text):
            sends.append((chat_id, text))
            if chat_id == "bad":
                raise RuntimeError("telegram rejected this chat")

    gw._app = SimpleNamespace(bot=_Bot())
    return gw


@pytest.mark.asyncio
async def test_drain_loop_delivers_each_message_and_isolates_a_failure():
    sends: list = []
    gw = _gateway_with_bot(sends)

    batches = [[{"chat_id": "bad", "text": "m1"}, {"chat_id": "good", "text": "m2"}]]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            i = calls["n"]
            calls["n"] += 1
            if i < len(batches):
                return batches[i]
            raise asyncio.CancelledError  # stop the loop after the batch

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    # Both attempted, in order; the failing 'bad' send did not stop 'good'.
    assert sends == [("bad", "m1"), ("good", "m2")]


@pytest.mark.asyncio
async def test_drain_loop_survives_a_drain_failure_and_keeps_delivering():
    sends: list = []
    gw = _gateway_with_bot(sends)

    seq = ["fail", [{"chat_id": "good", "text": "hi"}], "stop"]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            step = seq[calls["n"]]
            calls["n"] += 1
            if step == "fail":
                raise RuntimeError("runtime unreachable")
            if step == "stop":
                raise asyncio.CancelledError
            return step

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    # The drain failure was logged and survived; the next tick still delivered.
    assert sends == [("good", "hi")]


@pytest.mark.asyncio
async def test_run_polling_starts_and_cancels_the_drain_task():
    gw = MessagingGateway.__new__(MessagingGateway)
    gw._outbound_poll_seconds = 0

    drain_started = asyncio.Event()
    drain_cancelled = {"v": False}

    async def fake_loop():
        drain_started.set()
        try:
            await asyncio.Event().wait()  # block until cancelled on shutdown
        except asyncio.CancelledError:
            drain_cancelled["v"] = True
            raise

    gw._outbound_drain_loop = fake_loop
    app = SimpleNamespace(
        initialize=AsyncMock(),
        start=AsyncMock(),
        updater=SimpleNamespace(start_polling=AsyncMock(), stop=AsyncMock()),
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    gw.build_app = lambda: app
    gw.runtime_client = SimpleNamespace(close=AsyncMock())

    task = asyncio.create_task(gw.run_polling())
    await asyncio.wait_for(drain_started.wait(), timeout=2)  # loop launched
    task.cancel()  # simulate a shutdown signal
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert drain_cancelled["v"] is True  # the finally cancelled the drain task
    app.updater.stop.assert_awaited_once()
    gw.runtime_client.close.assert_awaited_once()
