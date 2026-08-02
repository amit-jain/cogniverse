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


def _gateway_with_bot(sends, fail_chats=("bad",), fail_times=None):
    """Gateway with a scripted bot. ``fail_chats`` always fail; ``fail_times``
    maps a chat_id to how many leading sends fail before succeeding."""
    gw = MessagingGateway.__new__(MessagingGateway)
    gw._outbound_poll_seconds = 0  # no real delay between ticks
    gw._outbound_retry = []
    remaining = dict(fail_times or {})

    class _Bot:
        async def send_message(self, chat_id, text):
            sends.append((chat_id, text))
            if chat_id in fail_chats:
                raise RuntimeError("telegram rejected this chat")
            if remaining.get(chat_id, 0) > 0:
                remaining[chat_id] -= 1
                raise RuntimeError("transient telegram failure")

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
    # The failed send is buffered for the next tick, not lost.
    assert gw._outbound_retry == [({"chat_id": "bad", "text": "m1"}, 1)]


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


@pytest.mark.asyncio
async def test_drain_loop_skips_malformed_entries_and_keeps_going():
    """A non-dict entry in one batch must not kill the delivery loop — the
    exception previously ended the task silently while the process stayed
    up, so runtime→user delivery was dead until a restart."""
    sends: list = []
    gw = _gateway_with_bot(sends)

    batches = [
        ["junk-string", {"chat_id": "good", "text": "first"}],
        [{"chat_id": "good", "text": "second"}],
    ]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            i = calls["n"]
            calls["n"] += 1
            if i < len(batches):
                return batches[i]
            raise asyncio.CancelledError

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    assert sends == [("good", "first"), ("good", "second")]


@pytest.mark.asyncio
async def test_failed_send_retries_next_tick_and_delivers_exactly_once():
    """The runtime clears a message on drain, so a transient Telegram failure
    must keep it in the gateway's retry buffer and deliver it on a later
    tick — exactly once, never dropped, never duplicated after success."""
    sends: list = []
    gw = _gateway_with_bot(sends, fail_chats=(), fail_times={"flaky": 1})

    batches = [
        [{"chat_id": "flaky", "text": "m1"}, {"chat_id": "good", "text": "m2"}],
        [{"chat_id": "good", "text": "m3"}],
        [],
    ]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            i = calls["n"]
            calls["n"] += 1
            if i < len(batches):
                return batches[i]
            raise asyncio.CancelledError

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    # Tick 1: flaky fails, good delivers. Tick 2: flaky retries FIRST
    # (submission order), then the new message. Tick 3+: nothing re-sent.
    assert sends == [
        ("flaky", "m1"),
        ("good", "m2"),
        ("flaky", "m1"),
        ("good", "m3"),
    ]
    assert gw._outbound_retry == []


@pytest.mark.asyncio
async def test_dead_chat_dropped_after_max_attempts_without_blocking_others():
    """A permanently failing chat is retried OUTBOUND_SEND_MAX_ATTEMPTS times
    total, then dropped so the buffer cannot grow forever; healthy chats
    deliver on every tick throughout."""
    from cogniverse_messaging.gateway import OUTBOUND_SEND_MAX_ATTEMPTS

    sends: list = []
    gw = _gateway_with_bot(sends, fail_chats=("dead",))

    batches = [[{"chat_id": "dead", "text": "m1"}], [], [], [], []]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            i = calls["n"]
            calls["n"] += 1
            if i < len(batches):
                return batches[i]
            raise asyncio.CancelledError

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    assert sends == [("dead", "m1")] * OUTBOUND_SEND_MAX_ATTEMPTS
    assert gw._outbound_retry == []


@pytest.mark.asyncio
async def test_retry_buffer_still_flushes_while_the_runtime_is_down():
    """A drain outage must not stall the retry buffer: messages already
    drained (which the runtime has forgotten) keep retrying each tick even
    when drain_outbound itself fails."""
    sends: list = []
    gw = _gateway_with_bot(sends, fail_chats=(), fail_times={"flaky": 1})

    seq = [[{"chat_id": "flaky", "text": "m1"}], "fail", "stop"]
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

    # Tick 1: send fails, buffered. Tick 2: drain is down but the buffered
    # message is still retried and delivered.
    assert sends == [("flaky", "m1"), ("flaky", "m1")]
    assert gw._outbound_retry == []


@pytest.mark.asyncio
async def test_message_missing_chat_or_text_is_dropped_not_retried():
    """A message that can never send (no chat_id or no text) is dropped with
    an error log — buffering it would retry a permanently unsendable item."""
    sends: list = []
    gw = _gateway_with_bot(sends, fail_chats=())

    batches = [
        [{"chat_id": "", "text": "x"}, {"text": "no-chat"}, {"chat_id": "g"}],
        [{"chat_id": "g", "text": "ok"}],
    ]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            i = calls["n"]
            calls["n"] += 1
            if i < len(batches):
                return batches[i]
            raise asyncio.CancelledError

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    assert sends == [("g", "ok")]
    assert gw._outbound_retry == []


@pytest.mark.asyncio
async def test_drain_loop_survives_a_none_batch():
    """messages=None (a regressed client) must be treated as an empty tick,
    not a TypeError that kills the loop."""
    sends: list = []
    gw = _gateway_with_bot(sends)

    seq = [None, [{"chat_id": "good", "text": "hi"}]]
    calls = {"n": 0}

    class _RC:
        async def drain_outbound(self):
            i = calls["n"]
            calls["n"] += 1
            if i < len(seq):
                return seq[i]
            raise asyncio.CancelledError

    gw.runtime_client = _RC()

    with pytest.raises(asyncio.CancelledError):
        await gw._outbound_drain_loop()

    assert sends == [("good", "hi")]


@pytest.mark.asyncio
async def test_drain_outbound_normalizes_shapes():
    """The client never hands the loop a non-list, and non-dict entries are
    dropped with an error log — {"messages": null} previously came back as
    None and the loop died iterating it."""
    rc = RuntimeClient("http://runtime")
    http = AsyncMock()
    http.is_closed = False
    rc._client = http

    http.get = AsyncMock(return_value=_response({"messages": None}))
    assert await rc.drain_outbound() == []

    http.get = AsyncMock(
        return_value=_response({"messages": ["garbage", {"chat_id": "1", "text": "a"}]})
    )
    assert await rc.drain_outbound() == [{"chat_id": "1", "text": "a"}]

    http.get = AsyncMock(return_value=_response(["not", "a", "dict"]))
    assert await rc.drain_outbound() == []
