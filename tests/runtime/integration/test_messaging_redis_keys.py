"""Redis key-handling contracts for the cross-pod inbound queue.

Spins its own Redis container so the assertions run against a real
Redis (no mocks, no external dependency).

Assertions:

* list_active_queues returns the original session_id even when it
  contains colons (the prefix/suffix strip must not truncate it).
* The inbound message list carries a TTL bounded by the active-marker
  TTL, so an abandoned (never-closed) session self-expires.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import time
from datetime import datetime, timezone

import pytest
import redis.asyncio as aioredis

from cogniverse_runtime.messaging import InboundMessage
from cogniverse_runtime.messaging_redis import (
    RedisInboundQueueRegistry,
    _list_key,
    _session_id_from_active_key,
)

CONTAINER_NAME = "cogniverse-test-messaging-redis"


def test_session_id_from_active_key_round_trips_colon():
    """A colon-bearing session_id survives the active-key strip."""
    sid = "acme:team:sess-1"
    key = f"session:{sid}:tenant"
    assert _session_id_from_active_key(key) == sid


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_url():
    port = _free_port()
    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"
    )

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
            "--platform",
            docker_platform,
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
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()


def _msg(content: str) -> InboundMessage:
    return InboundMessage(
        session_id="sess",
        role="user",
        content=content,
        tags=("constraint",),
        created_at=datetime.now(timezone.utc).isoformat(),
        deadline_ms=None,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_active_queues_preserves_colon_session_id(redis_client):
    reg = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    sid = "acme:team:sess-1"
    await reg.get_or_create_queue(sid, "acme:team")

    listed = await reg.list_active_queues()
    assert [q["session_id"] for q in listed] == [sid]
    assert listed[0]["tenant_id"] == "acme:team"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_inbound_list_carries_ttl(redis_client):
    reg = RedisInboundQueueRegistry(redis_client, active_ttl_seconds=60)
    q = await reg.get_or_create_queue("sess-ttl", "tenant-x")
    await q.enqueue(_msg("hi"))

    ttl = await redis_client.ttl(_list_key("tenant-x", "sess-ttl"))
    assert 0 < ttl <= 60
