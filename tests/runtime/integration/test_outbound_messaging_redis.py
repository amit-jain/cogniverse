"""Redis-backed cross-pod outbound delivery queue contract.

Tests against a real Redis the suite provisions itself (a throwaway
``redis:7.4-alpine`` container on a free port; ``COGNIVERSE_TEST_REDIS_URL``
overrides for operators with their own instance). No mocks. Two
``RedisOutboundQueue`` instances bound to the same Redis URL simulate a runtime
pod and the gateway; the assertions verify cross-pod routing, durability across
a singleton reset, and that the atomic drain neither loses nor duplicates a
message under concurrent enqueues.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time

import pytest
import redis.asyncio as aioredis

from cogniverse_runtime.messaging import OutboundMessage
from cogniverse_runtime.messaging_redis import (
    _OUTBOUND_KEY,
    RedisOutboundQueue,
    reset_redis_outbound_queue_for_testing,
)

CONTAINER_NAME = "redis-outbound-messaging-tests"

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_url():
    override = os.environ.get("COGNIVERSE_TEST_REDIS_URL")
    if override:
        yield override
        return

    port = _free_port()
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis: {result.stderr}")

    deadline = time.time() + 30
    while time.time() < deadline:
        ping = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        pytest.fail("Redis did not become ready within 30s")

    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


@pytest.fixture
async def redis_client(redis_url):
    r = aioredis.from_url(redis_url, decode_responses=True)
    await r.delete(_OUTBOUND_KEY)
    yield r
    await r.delete(_OUTBOUND_KEY)
    await r.close()


def _msg(i: int) -> OutboundMessage:
    return OutboundMessage(
        tenant_id="acme:acme",
        chat_id=str(i),
        text=f"m{i}",
        created_at="2026-01-01T00:00:00+00:00",
    )


async def test_runtime_enqueues_gateway_drains_same_message(redis_client):
    runtime_pod = RedisOutboundQueue(redis_client)
    gateway = RedisOutboundQueue(redis_client)  # second handle, same Redis

    await runtime_pod.enqueue(_msg(1))
    await runtime_pod.enqueue(_msg(2))

    drained = await gateway.drain()
    assert [(m.chat_id, m.text) for m in drained] == [("1", "m1"), ("2", "m2")]
    # Drained by the gateway → the runtime's next drain is empty (delivered once).
    assert await runtime_pod.drain() == []


async def test_pending_messages_survive_a_singleton_reset(redis_client):
    q = RedisOutboundQueue(redis_client)
    await q.enqueue(_msg(7))

    # Simulate a pod restart: drop the Python singleton, rebuild a fresh handle.
    reset_redis_outbound_queue_for_testing()
    fresh = RedisOutboundQueue(redis_client)

    drained = await fresh.drain()
    assert [m.chat_id for m in drained] == ["7"]


async def test_atomic_drain_under_concurrent_enqueues(redis_client):
    q = RedisOutboundQueue(redis_client)
    n = 100
    start = asyncio.Barrier(n + 1)
    drained: list[OutboundMessage] = []

    async def producer(i: int):
        await start.wait()
        await q.enqueue(_msg(i))

    async def draining():
        await start.wait()
        for _ in range(40):
            drained.extend(await q.drain())
            await asyncio.sleep(0)

    producers = [asyncio.create_task(producer(i)) for i in range(n)]
    drainer = asyncio.create_task(draining())
    await asyncio.gather(*producers, drainer)
    drained.extend(await q.drain())  # final sweep

    ids = sorted(int(m.chat_id) for m in drained)
    assert ids == list(range(n))  # every message exactly once — no loss, no dup
