"""OutboundQueue — FIFO enqueue/drain with an atomic clear, safe under load.

The runtime fans a `/messaging/send` request into one OutboundMessage per
target chat and enqueues them; the gateway drains and delivers. drain() must
return every buffered message and atomically clear the buffer, and concurrent
enqueues racing a drain must neither lose nor duplicate a message.
"""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_runtime.messaging import (
    OutboundMessage,
    OutboundQueue,
    get_outbound_queue,
    reset_outbound_queue_for_testing,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _msg(i: int) -> OutboundMessage:
    return OutboundMessage(
        tenant_id="acme:acme",
        chat_id=str(i),
        text=f"m{i}",
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_enqueue_then_drain_returns_and_clears():
    q = OutboundQueue()
    await q.enqueue(_msg(1))
    await q.enqueue(_msg(2))

    batch = await q.drain()
    assert [m.chat_id for m in batch] == ["1", "2"]  # FIFO order preserved
    assert await q.drain() == []  # drained → empty


@pytest.mark.asyncio
async def test_concurrent_enqueue_and_drain_neither_loses_nor_duplicates():
    q = OutboundQueue()
    n = 200
    start = asyncio.Barrier(n + 1)
    drained: list[OutboundMessage] = []

    async def producer(i: int):
        await start.wait()
        await q.enqueue(_msg(i))

    async def draining():
        await start.wait()
        for _ in range(50):
            drained.extend(await q.drain())
            await asyncio.sleep(0)

    producers = [asyncio.create_task(producer(i)) for i in range(n)]
    drainer = asyncio.create_task(draining())
    await asyncio.gather(*producers, drainer)
    drained.extend(await q.drain())  # final sweep for anything enqueued last

    ids = sorted(int(m.chat_id) for m in drained)
    assert ids == list(range(n))  # every message exactly once — no loss, no dup


@pytest.mark.asyncio
async def test_get_outbound_queue_is_process_singleton():
    reset_outbound_queue_for_testing()
    first = get_outbound_queue()
    assert get_outbound_queue() is first
    reset_outbound_queue_for_testing()
    assert get_outbound_queue() is not first
