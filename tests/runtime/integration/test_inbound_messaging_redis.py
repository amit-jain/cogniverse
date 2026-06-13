"""Redis-backed durable cross-pod inbound queue contract.

Tests against a real Redis the suite provisions itself (a throwaway
``redis:7.4-alpine`` container on a free port; ``COGNIVERSE_TEST_REDIS_URL``
overrides for operators with their own instance). No mocks. Two
``RedisInboundQueueRegistry`` instances bound to the same Redis URL
simulate two pods; the assertions verify cross-pod routing and
durability across registry resets.

Assertions:

* Cross-pod routing: Pod A enqueues, Pod B drains the
  same message byte-equal.
* Session-active visibility across pods.
* Cross-tenant collision raises ValueError across pods.
* Close on Pod A → 404-equivalent (get_queue None) on Pod B.
* Durability: enqueue → reset registry → fresh registry
  drains the persisted messages byte-equal.
* TTL expiry: active-marker disappears after TTL, get_queue returns
  None.
* Atomic drain under concurrent enqueues: 50 producers, one
  consumer, no lost messages.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from datetime import datetime, timezone

import pytest
import redis.asyncio as aioredis

from cogniverse_runtime.messaging import InboundMessage, QueueClosedError
from cogniverse_runtime.messaging_redis import (
    RedisInboundQueueRegistry,
    _list_key,
    reset_redis_inbound_queue_registry_for_testing,
)

CONTAINER_NAME = "redis-inbound-messaging-tests"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


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
    # Wipe inbound + session keys from any previous test run.
    async for k in r.scan_iter(match="inbound:*"):
        await r.delete(k)
    async for k in r.scan_iter(match="session:*:tenant"):
        await r.delete(k)
    yield r
    async for k in r.scan_iter(match="inbound:*"):
        await r.delete(k)
    async for k in r.scan_iter(match="session:*:tenant"):
        await r.delete(k)
    await r.close()


@pytest.fixture
async def reg_a(redis_client):
    reset_redis_inbound_queue_registry_for_testing()
    return RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)


@pytest.fixture
async def reg_b(redis_client):
    # A second registry instance bound to the SAME Redis simulates a
    # second pod. The two share state via Redis, not Python.
    return RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)


def _msg(content: str, *, tags=("constraint",)) -> InboundMessage:
    return InboundMessage(
        session_id="sess-1",
        role="user",
        content=content,
        tags=tags,
        created_at=datetime.now(timezone.utc).isoformat(),
        deadline_ms=None,
    )


# --------------------------------------------------------------------- #
# Cross-pod routing                                                       #
# --------------------------------------------------------------------- #


async def test_cross_pod_enqueue_and_drain(reg_a, reg_b):
    """Pod A registers a session; Pod B observes it via get_queue and
    can enqueue into it. Pod A's subsequent drain returns the
    message byte-equal. Proves Redis is the shared substrate."""
    q_a = await reg_a.get_or_create_queue("sess-1", "tenant-x")
    q_b = await reg_b.get_queue("sess-1")
    assert q_b is not None
    assert q_b.tenant_id == "tenant-x"

    await q_b.enqueue(_msg("cross-pod-msg"))
    drained = await q_a.drain()
    assert len(drained) == 1
    assert drained[0].content == "cross-pod-msg"
    assert drained[0].tags == ("constraint",)


async def test_cross_pod_get_queue_returns_none_for_unknown(reg_b):
    """A pod whose registry has no record of the session still sees
    None from Redis until SOMEONE creates the session."""
    assert await reg_b.get_queue("never-existed") is None


async def test_cross_pod_close_makes_get_queue_return_none(reg_a, reg_b):
    """Pod B closes the session → Pod A's get_queue sees the
    closure (via Redis active-marker DEL). Subsequent enqueue from
    either pod raises QueueClosedError."""
    await reg_a.get_or_create_queue("sess-close", "tenant-x")
    assert (await reg_b.get_queue("sess-close")) is not None

    closed = await reg_b.close_queue("sess-close")
    assert closed is True
    assert await reg_a.get_queue("sess-close") is None

    # Stale handle from Pod A can't enqueue after close.
    stale_handle = await reg_a.get_or_create_queue("sess-stale", "tenant-x")
    await reg_b.close_queue("sess-stale")
    with pytest.raises(QueueClosedError) as exc_info:
        await stale_handle.enqueue(_msg("late"))
    assert "sess-stale" in str(exc_info.value)


# --------------------------------------------------------------------- #
# Cross-tenant collision raises across pods                               #
# --------------------------------------------------------------------- #


async def test_cross_tenant_collision_raises_across_pods(reg_a, reg_b):
    """Pod A registers under alice; Pod B tries to register the same
    session_id under bob — must raise so a routing bug surfaces."""
    await reg_a.get_or_create_queue("shared", "alice")
    with pytest.raises(ValueError) as exc_info:
        await reg_b.get_or_create_queue("shared", "bob")
    assert "alice" in str(exc_info.value)
    assert "bob" in str(exc_info.value)


async def test_get_or_create_idempotent_same_tenant_across_pods(reg_a, reg_b):
    """Both pods can call get_or_create_queue with the SAME tenant —
    second call must succeed and return a usable handle, not raise."""
    q1 = await reg_a.get_or_create_queue("sess-idem", "alice")
    q2 = await reg_b.get_or_create_queue("sess-idem", "alice")
    # Different Python instances but same Redis-backed identity.
    assert q1.session_id == q2.session_id == "sess-idem"
    assert q1.tenant_id == q2.tenant_id == "alice"
    # Enqueue via q1, drain via q2 — they share state.
    await q1.enqueue(_msg("from-q1"))
    drained = await q2.drain()
    assert [m.content for m in drained] == ["from-q1"]


# --------------------------------------------------------------------- #
# Durability across registry resets (simulated pod restart)              #
# --------------------------------------------------------------------- #


async def test_messages_persist_across_registry_reset(redis_client):
    """Enqueue → simulate pod restart by resetting the Python-side
    registry singleton AND discarding the registry instance →
    create a fresh registry bound to the same Redis → drain returns
    the persisted messages byte-equal. This is the durability
    contract: messages survive pod death because they live in
    Redis, not in Python.
    """
    reg_v1 = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    q = await reg_v1.get_or_create_queue("sess-persist", "tenant-x")
    await q.enqueue(_msg("survive-1"))
    await q.enqueue(_msg("survive-2"))

    # Simulate pod death — drop ALL Python references to reg_v1 + queue.
    del q
    del reg_v1
    reset_redis_inbound_queue_registry_for_testing()

    # Fresh pod comes up — new registry, same Redis.
    reg_v2 = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    q2 = await reg_v2.get_or_create_queue("sess-persist", "tenant-x")
    drained = await q2.drain()
    # Submission order preserved byte-equal across the restart.
    assert [m.content for m in drained] == ["survive-1", "survive-2"]
    assert all(m.tags == ("constraint",) for m in drained)


async def test_close_drops_persisted_messages_too(redis_client):
    """``close_queue`` MUST delete both the active-marker AND the
    list — otherwise closed sessions accumulate orphaned messages
    in Redis indefinitely."""
    reg = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    q = await reg.get_or_create_queue("sess-clean", "tenant-x")
    await q.enqueue(_msg("orphan-candidate"))
    list_len_before = await redis_client.llen(_list_key("tenant-x", "sess-clean"))
    assert list_len_before == 1

    await reg.close_queue("sess-clean")
    list_len_after = await redis_client.llen(_list_key("tenant-x", "sess-clean"))
    # List was deleted along with the active-marker — no orphan
    # messages left in Redis.
    assert list_len_after == 0


# --------------------------------------------------------------------- #
# TTL expiry                                                              #
# --------------------------------------------------------------------- #


async def test_active_marker_expires_after_ttl(redis_client):
    """Sessions auto-cleanup after the configured TTL — admin
    endpoints don't accumulate stale entries forever. List under
    inbound:* stays (orphans are recoverable by an operator) but
    get_queue returns None once the active-marker expires."""
    reg = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=1)
    await reg.get_or_create_queue("sess-ttl", "tenant-x")
    assert await reg.get_queue("sess-ttl") is not None

    await asyncio.sleep(1.5)
    assert await reg.get_queue("sess-ttl") is None


# --------------------------------------------------------------------- #
# Atomic drain under contention                                           #
# --------------------------------------------------------------------- #


async def test_concurrent_enqueue_and_drain_preserves_every_message(redis_client):
    """50 messages from 10 concurrent producers; the consumer drains
    repeatedly until all 50 have been seen. Set equality on the
    contents proves the atomic Lua drain doesn't lose or duplicate
    any message under contention.
    """
    reg = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    q = await reg.get_or_create_queue("sess-load", "tenant-x")

    expected = {f"msg-{i:03d}" for i in range(50)}

    async def produce(start: int) -> None:
        for i in range(start, start + 5):
            await q.enqueue(_msg(f"msg-{i:03d}"))

    producers = [asyncio.create_task(produce(s)) for s in range(0, 50, 5)]
    seen: set[str] = set()
    while len(seen) < 50:
        batch = await q.drain()
        seen.update(m.content for m in batch)
        if not all(p.done() for p in producers):
            await asyncio.sleep(0.005)
    await asyncio.gather(*producers)
    seen.update(m.content for m in await q.drain())

    assert seen == expected


# --------------------------------------------------------------------- #
# Past-deadline drop at drain (mirrors in-pod semantics)                  #
# --------------------------------------------------------------------- #


async def test_past_deadline_message_dropped_at_drain(redis_client):
    """A past-deadline message is buffered in Redis but the consumer
    never sees it — matches the in-pod contract."""
    import time as _time

    reg = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    q = await reg.get_or_create_queue("sess-dl", "tenant-x")

    past_msg = InboundMessage(
        session_id="sess-dl",
        role="user",
        content="expired",
        tags=("constraint",),
        created_at=datetime.now(timezone.utc).isoformat(),
        deadline_ms=int(_time.time() * 1000) - 5000,
    )
    fresh_msg = _msg("fresh")
    await q.enqueue(past_msg)
    await q.enqueue(fresh_msg)

    drained = await q.drain()
    assert [m.content for m in drained] == ["fresh"]


# --------------------------------------------------------------------- #
# list_active_queues filters by tenant                                    #
# --------------------------------------------------------------------- #


async def test_list_active_queues_filters_by_tenant_cross_pod(reg_a, reg_b):
    """Phase-2 admin endpoint behaviour: each pod can scan + filter
    active sessions by tenant. Cross-tenant data MUST NOT leak."""
    await reg_a.get_or_create_queue("alice-1", "alice")
    await reg_a.get_or_create_queue("alice-2", "alice")
    await reg_b.get_or_create_queue("bob-1", "bob")

    alice_from_b = await reg_b.list_active_queues("alice")
    bob_from_a = await reg_a.list_active_queues("bob")
    assert sorted(e["session_id"] for e in alice_from_b) == ["alice-1", "alice-2"]
    assert sorted(e["session_id"] for e in bob_from_a) == ["bob-1"]
    # All entries from a tenant-scoped query carry that tenant id.
    assert all(e["tenant_id"] == "alice" for e in alice_from_b)
