"""Integration tests for the ingestion_worker Redis primitives.

Spins up a real Redis 7.4 container, exercises the full
submit → claim → ack round-trip + idempotency + status streams + the
per-tenant active counter that backpressure reads.

Mocking Redis would prove only that our calls match the API shape
``redis.asyncio`` expects; it wouldn't catch consumer-group semantics
(exclusive claim across consumers, BUSYGROUP on re-create, XACK
removing entries from PEL) or the JSON round-trip through XADD/XREAD
fields. Real container costs ~5s startup, runs every assertion against
the real server.
"""

from __future__ import annotations

import asyncio
import platform
import socket
import subprocess
import time
import uuid

import pytest

from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

CONTAINER_NAME = "redis-ingestion-v2-tests"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_container():
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
async def redis(redis_container):
    """Fresh redis client per test, FLUSHDB to isolate state."""
    await close_redis()  # drop any pool from a previous test
    client = await get_redis(redis_container)
    await client.flushdb()
    yield client
    await close_redis()


class TestIdempotency:
    def test_compute_sha_is_deterministic(self):
        a = idempotency.compute_sha("s3://b/k", "video", "acme")
        b = idempotency.compute_sha("s3://b/k", "video", "acme")
        assert a == b
        assert len(a) == 16

    def test_compute_sha_changes_with_inputs(self):
        base = idempotency.compute_sha("s3://b/k", "video", "acme")
        assert idempotency.compute_sha("s3://b/k2", "video", "acme") != base
        assert idempotency.compute_sha("s3://b/k", "audio", "acme") != base
        assert idempotency.compute_sha("s3://b/k", "video", "other") != base

    def test_compute_sha_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="non-empty"):
            idempotency.compute_sha("", "video", "acme")
        with pytest.raises(ValueError, match="non-empty"):
            idempotency.compute_sha("s3://b/k", "", "acme")
        with pytest.raises(ValueError, match="non-empty"):
            idempotency.compute_sha("s3://b/k", "video", "")

    @pytest.mark.asyncio
    async def test_done_record_round_trip(self, redis):
        sha = idempotency.compute_sha("s3://b/k", "video", "acme")
        assert await idempotency.get_existing_ingest_id(redis, sha) is None

        await idempotency.mark_done(redis, sha, "ingest_42", ttl_seconds=60)
        assert await idempotency.get_existing_ingest_id(redis, sha) == "ingest_42"

        await idempotency.clear_done(redis, sha)
        assert await idempotency.get_existing_ingest_id(redis, sha) is None

    @pytest.mark.asyncio
    async def test_inflight_takes_priority_over_done(self, redis):
        """An old ``done`` record from a prior submission must not mask a
        newly-inflight ingest_id — the running job's id is the right
        answer for re-submitters polling status."""
        sha = idempotency.compute_sha("s3://b/k", "video", "acme")
        await idempotency.mark_done(redis, sha, "old_done_id", ttl_seconds=60)
        await idempotency.mark_inflight(redis, sha, "new_inflight_id", ttl_seconds=60)
        assert await idempotency.get_existing_ingest_id(redis, sha) == "new_inflight_id"

        await idempotency.clear_inflight(redis, sha)
        assert await idempotency.get_existing_ingest_id(redis, sha) == "old_done_id"

    @pytest.mark.asyncio
    async def test_inflight_ttl_self_heals_after_crash(self, redis):
        """A worker that dies between mark_inflight and clear_inflight must
        not poison the sha forever. With a TTL, the stale inflight key
        expires and a re-submitter gets None (re-enqueue), instead of the
        dead job's id."""
        sha = idempotency.compute_sha("s3://crash/k", "video", "acme")
        await idempotency.mark_inflight(redis, sha, "dead_id", ttl_seconds=1)
        assert await idempotency.get_existing_ingest_id(redis, sha) == "dead_id"
        # The worker never reaches clear_inflight (simulated crash). The TTL
        # must reap the key on its own.
        assert await redis.ttl(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") > 0
        await asyncio.sleep(1.5)
        assert await idempotency.get_existing_ingest_id(redis, sha) is None


class TestQueue:
    @pytest.mark.asyncio
    async def test_submit_then_claim_returns_the_job(self, redis):
        await queue.ensure_consumer_group(redis, "g1")

        message_id = await queue.submit(
            redis,
            ingest_id="ingest_1",
            source_url="s3://b/k",
            profile="video",
            tenant_id="acme",
            sha="abc123",
        )
        assert message_id

        jobs = await queue.claim(redis, "g1", "consumer_a", block_ms=1000)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.ingest_id == "ingest_1"
        assert job.source_url == "s3://b/k"
        assert job.profile == "video"
        assert job.tenant_id == "acme"
        assert job.sha == "abc123"
        assert job.message_id == message_id

    @pytest.mark.asyncio
    async def test_claim_returns_empty_after_block_when_no_work(self, redis):
        await queue.ensure_consumer_group(redis, "g1")
        jobs = await queue.claim(redis, "g1", "consumer_a", block_ms=200)
        assert jobs == []

    @pytest.mark.asyncio
    async def test_consumer_group_delivers_each_entry_to_one_consumer(self, redis):
        """Two consumers in the same group must split the work, not
        both receive the same entry. This is the property that makes
        horizontal scale safe."""
        await queue.ensure_consumer_group(redis, "g1")

        for i in range(4):
            await queue.submit(
                redis,
                ingest_id=f"ingest_{i}",
                source_url=f"s3://b/{i}",
                profile="video",
                tenant_id="acme",
                sha=f"sha_{i}",
            )

        seen_a = await queue.claim(redis, "g1", "consumer_a", block_ms=500, count=4)
        seen_b = await queue.claim(redis, "g1", "consumer_b", block_ms=500, count=4)
        seen_total = seen_a + seen_b
        assert {j.ingest_id for j in seen_total} == {f"ingest_{i}" for i in range(4)}
        # No overlap — each entry delivered to exactly one consumer.
        assert {j.message_id for j in seen_a}.isdisjoint({j.message_id for j in seen_b})

    @pytest.mark.asyncio
    async def test_ack_removes_from_pending(self, redis):
        """After ack, the same consumer asking for ``>`` shouldn't see
        the entry again — neither should reaper logic that scans the
        pending entries list (PEL)."""
        await queue.ensure_consumer_group(redis, "g1")
        await queue.submit(redis, "i1", "s3://b/k", "video", "acme", "sha_1")
        jobs = await queue.claim(redis, "g1", "consumer_a", block_ms=500)
        assert len(jobs) == 1

        # PEL has one entry before ack
        pending_before = await redis.xpending(queue.QUEUE_STREAM, "g1")
        assert pending_before["pending"] == 1

        ack_result = await queue.ack(redis, "g1", jobs[0].message_id)
        assert ack_result == 1

        pending_after = await redis.xpending(queue.QUEUE_STREAM, "g1")
        assert pending_after["pending"] == 0

    @pytest.mark.asyncio
    async def test_ensure_consumer_group_is_idempotent(self, redis):
        """Restarting an ingestor pod must not crash on BUSYGROUP."""
        await queue.ensure_consumer_group(redis, "g1")
        await queue.ensure_consumer_group(redis, "g1")  # must not raise

    @pytest.mark.asyncio
    async def test_queue_depth_grows_and_shrinks(self, redis):
        await queue.ensure_consumer_group(redis, "g1")
        assert await queue.queue_depth(redis) == 0

        for i in range(3):
            await queue.submit(
                redis, f"i_{i}", f"s3://b/{i}", "video", "acme", f"sha_{i}"
            )
        assert await queue.queue_depth(redis) == 3
        # XLEN counts every stream entry, including consumed-but-unacked ones;
        # only ack() (XACK + XDEL) removes an entry. These 3 were never claimed
        # or acked, so all 3 remain counted.

    @pytest.mark.asyncio
    async def test_ack_drains_stream_and_clears_backpressure(self, redis):
        """Fully processing N jobs must drain the stream to 0 and leave the
        cluster backpressure axis below its limit. The old ack() only XACKed,
        so completed jobs lingered in XLEN and every submit past
        queue_depth_limit lifetime jobs 429ed forever."""
        from cogniverse_runtime.ingestion_worker import backpressure

        await queue.ensure_consumer_group(redis, "g1")
        n = 5
        for i in range(n):
            await queue.submit(
                redis, f"i_{i}", f"s3://b/{i}", "video", "acme", f"sha_{i}"
            )
            jobs = await queue.claim(redis, "g1", "consumer_a", block_ms=500)
            assert len(jobs) == 1
            assert await queue.ack(redis, "g1", jobs[0].message_id) == 1

        pending = await redis.xpending(queue.QUEUE_STREAM, "g1")
        assert pending["pending"] == 0  # proves the acks really happened
        assert await queue.queue_depth(redis) == 0

        rejection = await backpressure.check(
            redis, "acme", queue_depth_limit=n, per_tenant_concurrency=100
        )
        assert rejection is None


class TestActiveCounter:
    @pytest.mark.asyncio
    async def test_increment_decrement_round_trip(self, redis):
        assert await queue.get_active(redis, "acme") == 0
        assert await queue.increment_active(redis, "acme") == 1
        assert await queue.increment_active(redis, "acme") == 2
        assert await queue.get_active(redis, "acme") == 2
        assert await queue.decrement_active(redis, "acme") == 1
        assert await queue.decrement_active(redis, "acme") == 0

    @pytest.mark.asyncio
    async def test_decrement_floors_at_zero(self, redis):
        """A double-decrement under a duplicate terminal event must not
        leave the counter negative; otherwise backpressure undercounts
        in-flight jobs forever."""
        await queue.increment_active(redis, "acme")
        await queue.decrement_active(redis, "acme")
        # Stray decrement
        assert await queue.decrement_active(redis, "acme") == 0
        assert await queue.get_active(redis, "acme") == 0

    @pytest.mark.asyncio
    async def test_per_tenant_isolation(self, redis):
        await queue.increment_active(redis, "acme")
        await queue.increment_active(redis, "acme")
        await queue.increment_active(redis, "other")
        assert await queue.get_active(redis, "acme") == 2
        assert await queue.get_active(redis, "other") == 1

    @pytest.mark.asyncio
    async def test_increment_sets_self_healing_ttl(self, redis):
        """A leaked counter (incremented but never decremented) must expire so
        backpressure self-heals instead of wedging the tenant forever."""
        key = f"{queue.ACTIVE_KEY_PREFIX}acme"
        await queue.increment_active(redis, "acme")
        # A TTL is set (not -1 = no-expiry), bounded by the configured ceiling.
        assert 0 < await redis.ttl(key) <= queue.ACTIVE_TTL_SECONDS
        # Re-incrementing refreshes it — an active tenant keeps the key alive.
        await queue.increment_active(redis, "acme")
        assert 0 < await redis.ttl(key) <= queue.ACTIVE_TTL_SECONDS


class TestStatusStream:
    @pytest.mark.asyncio
    async def test_publish_then_read_returns_events_in_order(self, redis):
        ingest_id = f"ing_{uuid.uuid4().hex[:8]}"
        await queue.publish_status(redis, ingest_id, {"state": "queued"})
        await queue.publish_status(
            redis, ingest_id, {"state": "running", "progress": 0.5}
        )
        await queue.publish_status(redis, ingest_id, {"state": "complete"})

        events = await queue.read_status_since(redis, ingest_id)
        assert [e[1]["state"] for e in events] == ["queued", "running", "complete"]
        # Nested fields survive JSON round-trip
        assert events[1][1]["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_status_stream_is_capped_and_expires(self, redis):
        """A chatty job must not grow its status stream unbounded, and the
        stream must carry a TTL so a finished job's stream is reclaimed instead
        of leaking one immortal ``ingest:status:<id>`` stream per ingest."""
        ingest_id = f"ing_{uuid.uuid4().hex[:8]}"
        key = queue._status_stream_key(ingest_id)
        for i in range(queue.STATUS_STREAM_MAXLEN + 25):
            await queue.publish_status(redis, ingest_id, {"state": "running", "i": i})

        # XADD MAXLEN trims the oldest, so the count is capped and the newest
        # events (current state) are the ones retained.
        assert await redis.xlen(key) == queue.STATUS_STREAM_MAXLEN
        ttl = await redis.ttl(key)
        assert 0 < ttl <= queue.STATUS_STREAM_TTL_SECONDS, (
            f"status stream must carry a positive TTL, got {ttl}"
        )
        # FIFO trim drops the oldest 25 (0..24); the oldest retained is now 25.
        events = await queue.read_status_since(redis, ingest_id)
        assert events[0][1]["i"] == 25

    @pytest.mark.asyncio
    async def test_read_status_resumes_from_last_id(self, redis):
        """Late-connecting SSE clients must be able to replay only the
        events they haven't seen — not the whole history every poll."""
        ingest_id = f"ing_{uuid.uuid4().hex[:8]}"
        await queue.publish_status(redis, ingest_id, {"state": "queued"})
        await queue.publish_status(redis, ingest_id, {"state": "running"})

        first = await queue.read_status_since(redis, ingest_id)
        assert len(first) == 2
        last_seen = first[-1][0]

        await queue.publish_status(redis, ingest_id, {"state": "complete"})

        second = await queue.read_status_since(redis, ingest_id, last_id=last_seen)
        assert [e[1]["state"] for e in second] == ["complete"]
