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
import functools
import os
import platform
import socket
import subprocess
import time
import uuid

import pytest

from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

CONTAINER_NAME = "redis-ingestion-v2-tests"


@pytest.fixture(autouse=True)
def _disable_worker_telemetry_export():
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    TelemetryManager.reset()
    telemetry_manager_module._telemetry_manager = TelemetryManager(
        TelemetryConfig(enabled=False, otlp_enabled=False)
    )
    yield
    TelemetryManager.reset()


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
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
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
    async def test_refresh_claim_resets_idle_without_bumping_deliveries(self, redis):
        """The worker's processing heartbeat resets the PEL idle clock (so
        XAUTOCLAIM's min-idle never fires on a live job) WITHOUT advancing
        the delivery counter (so heartbeats can't push a healthy long job
        over the poison-message dead-letter cap)."""
        await queue.ensure_consumer_group(redis, "g1")
        await queue.submit(redis, "i_hb", "s3://b/hb", "video", "acme", "sha_hb")
        jobs = await queue.claim(redis, "g1", "consumer_a", block_ms=500)
        mid = jobs[0].message_id

        await asyncio.sleep(0.3)
        detail = await redis.xpending_range(
            queue.QUEUE_STREAM, "g1", min=mid, max=mid, count=1
        )
        assert detail[0]["time_since_delivered"] >= 250

        assert await queue.refresh_claim(redis, "g1", "consumer_a", mid) is True
        detail = await redis.xpending_range(
            queue.QUEUE_STREAM, "g1", min=mid, max=mid, count=1
        )
        assert detail[0]["time_since_delivered"] < 150
        assert detail[0]["times_delivered"] == 1

        await queue.ack(redis, "g1", mid)
        assert await queue.refresh_claim(redis, "g1", "consumer_a", mid) is False

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
    async def test_decrement_atomic_against_concurrent_increment(self, redis):
        """A clamping decrement must not clobber a concurrent increment.

        The counter is at logical zero, so a decrement floors. The old
        two-step DECR-then-SET(0) left a window: an increment landing between
        the DECR and the SET was overwritten back to 0. The single atomic
        ``eval`` closes it. We gate the client's ``set`` to force exactly that
        window — the atomic path performs its clamp server-side inside the Lua,
        so it never calls the client's ``set`` and the gate simply never fires.
        """
        assert await queue.get_active(redis, "acme") == 0

        release = asyncio.Event()
        orig_set = redis.set

        async def gated_set(*args, **kwargs):
            await release.wait()
            return await orig_set(*args, **kwargs)

        redis.set = gated_set
        try:
            dec_task = asyncio.create_task(queue.decrement_active(redis, "acme"))
            # Atomic decrement returns now; the racy two-step version blocks in
            # gated_set until released.
            await asyncio.wait({dec_task}, timeout=0.5)
            # A real job starts concurrently, inside the decrement's set window.
            assert await queue.increment_active(redis, "acme") == 1
            release.set()
            await dec_task
        finally:
            redis.set = orig_set

        # The floored spurious decrement is absorbed and the new active job is
        # counted; the two-step version overwrites this back to 0.
        assert await queue.get_active(redis, "acme") == 1

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


class TestEnqueueCompensation:
    @pytest.mark.asyncio
    async def test_submit_failure_clears_inflight_so_retry_re_enqueues(
        self, redis, monkeypatch
    ):
        """mark_inflight runs BEFORE the work-stream submit. If submit fails,
        the idempotency key must be cleared — otherwise a non-force retry
        returns the orphaned 'in_flight' id for a job that never reached the
        stream and will never run, silently dropping the upload until the 6h
        TTL lapses. Verified end-to-end against real Redis: fault -> cleared ->
        retry actually lands a job on the work stream."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = "s3://bucket/video.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)
        real_submit = queue.submit

        async def _boom(*args, **kwargs):
            raise ConnectionError("redis reset on xadd")

        monkeypatch.setattr(queue, "submit", _boom)
        with pytest.raises(ConnectionError, match="reset on xadd"):
            await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )

        # The orphaned inflight key must be gone — a retry must not see it.
        assert await idempotency.get_existing_ingest_id(redis, sha) is None

        # A retry (submit restored) actually enqueues and lands on the stream.
        monkeypatch.setattr(queue, "submit", real_submit)
        result = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert result.existing is False
        assert result.state == "queued"
        jobs = await queue.claim(redis, "ingestors", "consumer_retry", block_ms=1000)
        assert [j.ingest_id for j in jobs] == [result.ingest_id]

    @pytest.mark.asyncio
    async def test_submit_reply_lost_after_commit_is_treated_as_success(
        self, redis, monkeypatch
    ):
        """A connection dropped AFTER the submit MULTI/EXEC commits — but before
        the client reads the reply — must NOT trigger compensation. The job is
        durably enqueued with its markers set; clearing them lets a resubmit
        duplicate the job and makes the worker's terminal decrement undercount.
        Injected at the lowest seam: the submit pipeline's execute runs the real
        EXEC (committing the XADD + committed marker) and then raises."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = "s3://bucket/lost-reply.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)

        real_pipeline = redis.pipeline

        def _pipeline_dropping_reply(*args, **kwargs):
            pipe = real_pipeline(*args, **kwargs)
            real_execute = pipe.execute

            async def _execute_then_drop(*a, **k):
                await real_execute(*a, **k)  # real EXEC commits both writes
                raise ConnectionError("connection reset after EXEC committed")

            pipe.execute = _execute_then_drop
            return pipe

        monkeypatch.setattr(redis, "pipeline", _pipeline_dropping_reply)
        result = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        monkeypatch.setattr(redis, "pipeline", real_pipeline)

        # Submit is reported successful — the job is durably queued.
        assert result.existing is False
        assert result.state == "queued"

        # Both markers survived (compensation did not destroy them).
        assert await idempotency.is_submitted(redis, sha) is True
        assert await idempotency.get_existing_ingest_id(redis, sha) == result.ingest_id

        # The active counter was incremented once and NOT decremented.
        assert await queue.get_active(redis, tenant) == 1

        # The job landed on the work stream exactly once.
        assert await queue.queue_depth(redis) == 1

        # A resubmit finds the real in-flight run and enqueues no duplicate.
        resubmit = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert resubmit.existing is True
        assert resubmit.ingest_id == result.ingest_id
        assert await queue.queue_depth(redis) == 1
        assert await queue.get_active(redis, tenant) == 1

    @pytest.mark.asyncio
    async def test_submit_failure_before_commit_compensates_fully(
        self, redis, monkeypatch
    ):
        """The genuine-failure companion at the same seam: the submit pipeline's
        execute raises BEFORE the real EXEC, so nothing commits. The committed
        marker is absent, so compensation must run fully — clear the inflight
        marker, restore the active counter, and re-raise — and a retry must
        re-enqueue a real job."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = "s3://bucket/precommit-fail.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)

        real_pipeline = redis.pipeline

        def _pipeline_never_commits(*args, **kwargs):
            pipe = real_pipeline(*args, **kwargs)

            async def _execute_raises(*a, **k):
                raise ConnectionError("connection reset before EXEC")

            pipe.execute = _execute_raises
            return pipe

        monkeypatch.setattr(redis, "pipeline", _pipeline_never_commits)
        with pytest.raises(ConnectionError, match="before EXEC"):
            await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )
        monkeypatch.setattr(redis, "pipeline", real_pipeline)

        # Nothing committed: no marker, no stream entry, counter restored.
        assert await idempotency.is_submitted(redis, sha) is False
        assert await idempotency.get_existing_ingest_id(redis, sha) is None
        assert await queue.get_active(redis, tenant) == 0
        assert await queue.queue_depth(redis) == 0

        # A retry re-enqueues a real job that lands on the stream.
        result = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert result.existing is False
        assert result.state == "queued"
        assert await queue.queue_depth(redis) == 1

    @pytest.mark.asyncio
    async def test_happy_path_marks_submitted(self, redis):
        """A successful enqueue records the submitted marker so a resubmit can
        tell a real in-flight run from a crash-orphaned phantom."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = "s3://bucket/submitted.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)
        await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert await idempotency.is_submitted(redis, sha) is True

    @pytest.mark.asyncio
    async def test_committed_marker_is_atomic_with_submit(self, redis, monkeypatch):
        """A crash between XADD and the committed marker must be impossible.

        The old submit-then-mark sequence left a window: a hard crash after the
        job reached the work stream but before the committed marker was written
        produced a real queued job with no marker, which a resubmit past the
        commit grace read as a phantom and duplicated. Writing the marker in the
        same transaction as the XADD closes the window.

        To prove the atomic marker — not the commit grace — is what prevents the
        dupe, we drop the commit grace to zero (disabling phantom detection) and
        assert the marker is still present and the resubmit returns the same id
        without enqueuing a duplicate. The marker has no writer other than the
        submit transaction, so if that stopped writing it this would dupe.
        """
        import cogniverse_runtime.ingestion_worker.submit_api as submit_api_mod
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        monkeypatch.setattr(submit_api_mod, "_COMMIT_GRACE_SECONDS", 0.0)

        src, profile, tenant = "s3://bucket/atomic.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)

        first = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert first.existing is False
        # Written inside the submit transaction, atomically with the XADD.
        assert await idempotency.is_submitted(redis, sha) is True

        depth_after_first = await queue.queue_depth(redis)

        second = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert second.existing is True
        assert second.ingest_id == first.ingest_id
        assert await queue.queue_depth(redis) == depth_after_first

    @pytest.mark.asyncio
    async def test_aged_phantom_marker_is_cleared_and_re_enqueued(self, redis):
        """A hard crash (SIGKILL/OOM) between claim_inflight and submit runs no
        compensation, orphaning the inflight marker with no submitted marker.
        A resubmit must not hand back that phantom id for the 6h TTL — it must
        re-enqueue a real job that lands on the work stream."""
        from cogniverse_runtime.ingestion_worker.submit_api import (
            _inflight_ttl_seconds,
            enqueue_ingestion,
        )

        src, profile, tenant = "s3://bucket/phantom.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)

        # Plant an aged inflight marker (claimed ~60s ago, past the commit
        # grace) with NO submitted marker — a phantom from a pre-submit crash.
        await redis.set(
            f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}",
            "ingest_phantom",
            ex=_inflight_ttl_seconds() - 60,
        )

        result = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )

        assert result.existing is False, "phantom returned instead of re-enqueue"
        assert result.ingest_id != "ingest_phantom"
        assert result.state == "queued"
        jobs = await queue.claim(redis, "ingestors", "consumer_phantom", block_ms=1000)
        assert [j.ingest_id for j in jobs] == [result.ingest_id]

    @pytest.mark.asyncio
    async def test_fresh_inflight_without_submitted_is_treated_as_live(self, redis):
        """An inflight marker claimed within the commit grace and not yet
        submitted is a concurrent winner mid-submit — a resubmit returns its id
        rather than racing it into a duplicate enqueue."""
        from cogniverse_runtime.ingestion_worker.submit_api import (
            _inflight_ttl_seconds,
            enqueue_ingestion,
        )

        src, profile, tenant = "s3://bucket/fresh.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)

        # Freshly claimed (full TTL remaining ⇒ age ~0 < grace), no submitted.
        await redis.set(
            f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}",
            "ingest_winner",
            ex=_inflight_ttl_seconds(),
        )

        result = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )

        assert result.existing is True
        assert result.ingest_id == "ingest_winner"
        assert await queue.queue_depth(redis) == 0

    @pytest.mark.asyncio
    async def test_increment_fault_clears_inflight_without_bogus_decrement(
        self, redis, monkeypatch
    ):
        """Redis dropping the connection between mark_inflight and
        increment_active must still clear the inflight marker — and must NOT
        run decrement_active for a counter that was never incremented (that
        would free a slot a different running job of the same tenant holds)."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = "s3://bucket/incr.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)
        # Another job of the same tenant is mid-flight and owns one slot.
        await queue.increment_active(redis, tenant)
        decrements: list = []
        real_decrement = queue.decrement_active

        async def _boom_increment(*args, **kwargs):
            raise ConnectionError("redis reset on incr")

        async def _recording_decrement(r, t):
            decrements.append(t)
            return await real_decrement(r, t)

        monkeypatch.setattr(queue, "increment_active", _boom_increment)
        monkeypatch.setattr(queue, "decrement_active", _recording_decrement)
        with pytest.raises(ConnectionError, match="reset on incr"):
            await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )

        assert await idempotency.get_existing_ingest_id(redis, sha) is None
        assert decrements == [], "compensated a counter that was never incremented"
        assert await queue.get_active(redis, tenant) == 1
        assert await queue.queue_depth(redis) == 0

    @pytest.mark.asyncio
    async def test_decrement_fault_in_compensation_still_clears_inflight(
        self, redis, monkeypatch
    ):
        """When Redis is still failing during compensation, a decrement_active
        failure must not skip clear_inflight, and the ORIGINAL submit error
        must propagate — not the compensation's. The leaked counter is the
        accepted residual (self-heals via ACTIVE_TTL); a surviving inflight
        marker would poison every non-force resubmit for its whole TTL."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = "s3://bucket/comp.mp4", "video", "acme:acme"
        sha = idempotency.compute_sha(src, profile, tenant)

        async def _boom_submit(*args, **kwargs):
            raise ConnectionError("xadd reset")

        async def _boom_decrement(*args, **kwargs):
            raise TimeoutError("decr timed out")

        monkeypatch.setattr(queue, "submit", _boom_submit)
        monkeypatch.setattr(queue, "decrement_active", _boom_decrement)
        with pytest.raises(ConnectionError, match="xadd reset"):
            await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )

        assert await idempotency.get_existing_ingest_id(redis, sha) is None
        # The failed decrement leaves the counter at 1 — bounded by the
        # self-healing TTL set at increment time.
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.ttl(f"{queue.ACTIVE_KEY_PREFIX}{tenant}") > 0


class TestDedupeStatusSurface:
    @pytest.mark.asyncio
    async def test_done_dedupe_restores_expired_status_stream(self, redis):
        """A duplicate submit whose done run outlived its status stream must
        hand back a pollable id: the status read yields exactly one terminal
        snapshot, not the empty history /status turns into a permanent 404.

        The done marker lives for the idempotency TTL (days); the status
        stream expires after STATUS_STREAM_TTL_SECONDS (hours). In that
        window the dedupe hit used to return a dead handle."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = (
            "s3://bucket/dedupe-expired.md",
            "image_colpali_mv",
            "acme:acme",
        )
        sha = idempotency.compute_sha(src, profile, tenant)

        first = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        assert first.existing is False
        depth_after_first = await queue.queue_depth(redis)

        await idempotency.mark_done(
            redis, sha, first.ingest_id, ttl_seconds=7 * 24 * 3600
        )
        await idempotency.clear_inflight(redis, sha)
        assert await redis.delete(f"ingest:status:{first.ingest_id}") == 1

        second = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )

        assert second.existing is True
        assert second.ingest_id == first.ingest_id
        assert await queue.queue_depth(redis) == depth_after_first
        events = await queue.read_status_since(redis, second.ingest_id)
        assert [event for _, event in events] == [
            {
                "state": "complete",
                "ingest_id": first.ingest_id,
                "source_url": src,
                "profile": profile,
                "tenant_id": tenant,
                "existing": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_done_dedupe_with_live_stream_appends_nothing(self, redis):
        """While the done run's stream is still readable, a dedupe hit returns
        the id without appending a synthetic event over the real history."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = (
            "s3://bucket/dedupe-live.md",
            "image_colpali_mv",
            "acme:acme",
        )
        sha = idempotency.compute_sha(src, profile, tenant)

        first = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        await idempotency.mark_done(
            redis, sha, first.ingest_id, ttl_seconds=7 * 24 * 3600
        )
        await idempotency.clear_inflight(redis, sha)

        second = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )

        assert second.ingest_id == first.ingest_id
        events = await queue.read_status_since(redis, second.ingest_id)
        assert [event for _, event in events] == [
            {
                "state": "queued",
                "ingest_id": first.ingest_id,
                "source_url": src,
                "profile": profile,
                "tenant_id": tenant,
            }
        ]

    @pytest.mark.asyncio
    async def test_concurrent_dedupes_restore_exactly_one_event(self, redis):
        """Concurrency invariant: N simultaneous resubmits against a reclaimed
        stream seed it once, not once per caller. Every caller still gets the
        same pollable id and nothing is re-enqueued."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = (
            "s3://bucket/dedupe-race.md",
            "image_colpali_mv",
            "acme:acme",
        )
        sha = idempotency.compute_sha(src, profile, tenant)

        first = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        depth_after_first = await queue.queue_depth(redis)
        await idempotency.mark_done(
            redis, sha, first.ingest_id, ttl_seconds=7 * 24 * 3600
        )
        await idempotency.clear_inflight(redis, sha)
        assert await redis.delete(f"ingest:status:{first.ingest_id}") == 1

        racers = 8
        rendezvous = asyncio.Barrier(racers)

        async def _resubmit():
            await rendezvous.wait()
            return await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )

        results = await asyncio.gather(*(_resubmit() for _ in range(racers)))

        assert [r.ingest_id for r in results] == [first.ingest_id] * racers
        assert [r.existing for r in results] == [True] * racers
        assert [r.state for r in results] == ["complete"] * racers
        assert await queue.queue_depth(redis) == depth_after_first
        events = await queue.read_status_since(redis, first.ingest_id)
        assert [event for _, event in events] == [
            {
                "state": "complete",
                "ingest_id": first.ingest_id,
                "source_url": src,
                "profile": profile,
                "tenant_id": tenant,
                "existing": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_status_restore_failure_surfaces(self, redis, monkeypatch):
        """Fault contract: a status store that fails the restore must raise, so
        the caller never receives an accepted-looking id whose every poll 404s.
        The idempotency record survives for the retry."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        src, profile, tenant = (
            "s3://bucket/dedupe-fault.md",
            "image_colpali_mv",
            "acme:acme",
        )
        sha = idempotency.compute_sha(src, profile, tenant)

        first = await enqueue_ingestion(
            redis, source_url=src, profile=profile, tenant_id=tenant
        )
        await idempotency.mark_done(
            redis, sha, first.ingest_id, ttl_seconds=7 * 24 * 3600
        )
        await idempotency.clear_inflight(redis, sha)
        assert await redis.delete(f"ingest:status:{first.ingest_id}") == 1

        async def _boom_restore(*args, **kwargs):
            raise ConnectionError("status stream eval reset")

        monkeypatch.setattr(queue, "publish_status_if_absent", _boom_restore)
        with pytest.raises(ConnectionError, match="status stream eval reset"):
            await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )

        assert await idempotency.get_done_ingest_id(redis, sha) == first.ingest_id


class TestSubmitActiveCounterOrdering:
    @pytest.mark.asyncio
    async def test_worker_decrement_in_submit_window_does_not_stick_counter(
        self, redis, monkeypatch
    ):
        """The active counter must be incremented BEFORE the job becomes
        claimable. If the increment runs after ``queue.submit`` (which makes
        the job claimable), a fast worker can claim + process + decrement
        (floored at 0) in the window before the increment, leaving the counter
        stuck at 1 for the whole ACTIVE_TTL and permanently undercounting one
        slot of tenant capacity."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        tenant = "acme:acme"
        real_submit = queue.submit

        async def submit_then_worker_decrement(*args, **kwargs):
            msg_id = await real_submit(*args, **kwargs)
            # A fast worker claims the just-submitted job and reaches terminal
            # cleanup (decrement_active) before control returns to the API.
            await queue.decrement_active(redis, tenant)
            return msg_id

        monkeypatch.setattr(queue, "submit", submit_then_worker_decrement)
        await enqueue_ingestion(
            redis, source_url="s3://b/v.mp4", profile="video", tenant_id=tenant
        )
        # increment(->1) must precede submit's decrement(->0): net 0, not 1.
        assert await queue.get_active(redis, tenant) == 0

    @pytest.mark.asyncio
    async def test_submit_fault_restores_active_counter_and_inflight(
        self, redis, monkeypatch
    ):
        """A submit failure compensates BOTH the inflight marker and the active
        counter — the counter is incremented before the guarded submit, so the
        except path must decrement it back to its prior value."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        tenant, src, profile = "acme:acme", "s3://b/v.mp4", "video"
        sha = idempotency.compute_sha(src, profile, tenant)

        async def _boom(*args, **kwargs):
            raise ConnectionError("xadd reset")

        monkeypatch.setattr(queue, "submit", _boom)
        with pytest.raises(ConnectionError, match="xadd reset"):
            await enqueue_ingestion(
                redis, source_url=src, profile=profile, tenant_id=tenant
            )
        assert await queue.get_active(redis, tenant) == 0
        assert await idempotency.get_existing_ingest_id(redis, sha) is None

    @pytest.mark.asyncio
    async def test_successful_submit_increments_once_worker_returns_to_zero(
        self, redis, redis_container, monkeypatch
    ):
        """A successful submit increments the counter exactly once; the
        worker's terminal decrement returns it to 0 with no underflow."""
        from cogniverse_runtime.ingestion_worker import worker
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        tenant = "acme:acme"
        result = await enqueue_ingestion(
            redis, source_url="s3://b/v.mp4", profile="video", tenant_id=tenant
        )
        assert await queue.get_active(redis, tenant) == 1

        monkeypatch.setenv("REDIS_URL", redis_container)
        config = worker.WorkerConfig()
        await queue.ensure_consumer_group(redis, config.consumer_group)
        jobs = await queue.claim(redis, config.consumer_group, "c1", block_ms=1000)
        assert [j.ingest_id for j in jobs] == [result.ingest_id]

        async def _trivial(job):
            return {}

        await worker._process_job(redis, jobs[0], config, processor=_trivial)
        assert await queue.get_active(redis, tenant) == 0


class TestTerminalCleanupBestEffort:
    async def _submit_and_claim(self, redis, config, tenant, sha, ingest_id):
        await queue.ensure_consumer_group(redis, config.consumer_group)
        await queue.submit(
            redis,
            ingest_id=ingest_id,
            source_url="s3://b/v.mp4",
            profile="video",
            tenant_id=tenant,
            sha=sha,
        )
        jobs = await queue.claim(
            redis, config.consumer_group, config.consumer_id, block_ms=1000
        )
        assert len(jobs) == 1
        return jobs[0]

    @staticmethod
    def _terminal_event(events):
        terminal = [
            event for _, event in events if event.get("state") in ("complete", "failed")
        ]
        assert terminal, "no terminal event was published"
        return terminal[-1]

    @pytest.mark.asyncio
    async def test_ack_failure_flags_cleanup_error_and_leaves_entry_pending(
        self, redis, redis_container, monkeypatch
    ):
        """When ack fails, the earlier steps still ran, the terminal event
        carries a cleanup-error signal (not a fully-clean complete), and the
        entry stays in the PEL so the reaper can recover it."""
        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.setenv("REDIS_URL", redis_container)
        config = worker.WorkerConfig()
        tenant, sha, ingest_id = "acme:acme", "sha_ack", "ing_ack"
        await queue.increment_active(redis, tenant)
        job = await self._submit_and_claim(redis, config, tenant, sha, ingest_id)

        async def _ack_boom(*args, **kwargs):
            raise ConnectionError("ack reset")

        monkeypatch.setattr(queue, "ack", _ack_boom)

        async def _trivial(job):
            return {}

        await worker._process_job(redis, job, config, processor=_trivial)

        assert await redis.get(f"{idempotency.DONE_KEY_PREFIX}{sha}") == ingest_id
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") is None
        assert await queue.get_active(redis, tenant) == 0
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1  # ack never landed -> reaper can recover
        terminal = self._terminal_event(await queue.read_status_since(redis, ingest_id))
        assert terminal["state"] == "complete"
        assert "ack" in terminal["cleanup_error"]

    @pytest.mark.asyncio
    async def test_early_cleanup_failure_does_not_skip_decrement_or_ack(
        self, redis, redis_container, monkeypatch
    ):
        """A failure in an EARLY cleanup step (clear_inflight) must not skip the
        later steps. Each step is best-effort, so decrement_active and ack still
        run: the tenant slot frees and the PEL drains, with the failed step
        named on the terminal event."""
        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.setenv("REDIS_URL", redis_container)
        config = worker.WorkerConfig()
        tenant, sha, ingest_id = "acme:acme", "sha_early", "ing_early"
        await queue.increment_active(redis, tenant)
        job = await self._submit_and_claim(redis, config, tenant, sha, ingest_id)

        async def _clear_boom(_redis, _sha):
            raise ConnectionError("clear_inflight reset")

        monkeypatch.setattr(idempotency, "clear_inflight", _clear_boom)

        async def _trivial(job):
            return {}

        await worker._process_job(redis, job, config, processor=_trivial)

        # decrement_active ran despite the earlier failure -> slot freed.
        assert await queue.get_active(redis, tenant) == 0
        # ack ran -> PEL drained, stream emptied.
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0
        terminal = self._terminal_event(await queue.read_status_since(redis, ingest_id))
        assert "clear_inflight" in terminal["cleanup_error"]

    @pytest.mark.asyncio
    async def test_pipeline_exception_with_raising_str_still_publishes_terminal(
        self, redis, redis_container, monkeypatch
    ):
        """A pipeline exception whose __str__ itself raises must not stop the
        failed terminal from being built and published: otherwise no terminal
        is emitted, the client sees 'running' forever, and the tenant slot leaks
        until the active-counter TTL. The error text falls back to the type."""
        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.setenv("REDIS_URL", redis_container)
        config = worker.WorkerConfig()
        tenant, sha, ingest_id = "acme:acme", "sha_rstr", "ing_rstr"
        await queue.increment_active(redis, tenant)
        job = await self._submit_and_claim(redis, config, tenant, sha, ingest_id)

        class _RaisingStr(Exception):
            def __str__(self):
                raise RuntimeError("boom in __str__")

        async def _explode(job):
            raise _RaisingStr()

        await worker._process_job(redis, job, config, processor=_explode)

        terminal = self._terminal_event(await queue.read_status_since(redis, ingest_id))
        assert terminal["state"] == "failed"
        assert terminal["error_type"] == "_RaisingStr"
        assert "unprintable" in terminal["error"]
        # Cleanup still ran: slot freed, inflight cleared, entry acked.
        assert await queue.get_active(redis, tenant) == 0
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") is None


class TestColdBuildOffload:
    @pytest.mark.asyncio
    async def test_prepare_job_context_runs_off_the_event_loop(self, monkeypatch):
        """The cold builds (config manager + graph-factory install, with their
        Vespa reads and deploy convergence waits) must run via ``to_thread``.
        Inline on the loop they block the SIGTERM handler, deferring graceful
        shutdown into a SIGKILL. Proof: the builder records its thread id and
        blocks until the LOOP releases it — release can only arrive while the
        builder is blocked if the loop stayed responsive, and the recorded id
        must differ from the loop thread's."""
        import threading

        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.setenv("REDIS_URL", "redis://worker:6379/0")
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            '{"denseon":"http://denseon-at-startup:8000"}',
        )
        config = worker.WorkerConfig()
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            '{"denseon":"http://changed-after-startup:9000"}',
        )
        loop_thread = threading.get_ident()
        seen: dict = {}
        release = threading.Event()

        def _blocking_context(service_urls):
            seen["thread"] = threading.get_ident()
            seen["urls"] = dict(service_urls)
            seen["released"] = release.wait(timeout=5.0)
            return (object(), object())

        monkeypatch.setattr(worker, "_prepare_job_context", _blocking_context)

        class _Pipeline:
            def __init__(self, **kwargs):
                pass

            async def process_video_async(self, path, source_uri=None):
                return {"status": "success", "video_id": "v1", "results": {}}

        async def _no_graph(**kwargs):
            return {"nodes_upserted": 0, "edges_upserted": 0, "graph_failed": 0}

        class _Locator:
            def __init__(self, tenant_id, config):
                pass

            def localize(self, url):
                return "/tmp/never-read.mp4"

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", _Pipeline
        )
        monkeypatch.setattr(
            "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
            _no_graph,
        )
        monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", _Locator)

        job = queue.IngestJob(
            message_id="0-1",
            ingest_id="ing_offload",
            source_url="file:///tmp/never-read.mp4",
            profile="video",
            tenant_id="acme:acme",
            sha="sha_offload",
        )

        async def _noop_marker(pending_job):
            return None

        task = asyncio.create_task(
            worker._default_processor(
                job,
                service_urls=config.inference_service_urls,
                mark_graph_pending=_noop_marker,
                graph_deadline_s=config.graph_deadline_s,
            )
        )
        deadline = time.time() + 5.0
        while "thread" not in seen and not task.done() and time.time() < deadline:
            await asyncio.sleep(0.01)
        release.set()
        result = await asyncio.wait_for(task, timeout=10.0)

        assert result["status"] == "success" and result["video_id"] == "v1"
        assert seen["urls"] == {"denseon": "http://denseon-at-startup:8000"}
        assert seen["urls"] == config.inference_service_urls
        assert seen["thread"] != loop_thread, "cold build ran ON the event loop"
        assert seen["released"] is True, (
            "loop never serviced the release while the builder blocked — "
            "the build was not offloaded"
        )


class TestSubmitIdempotencyRace:
    @pytest.mark.asyncio
    async def test_concurrent_identical_submits_enqueue_once(self, redis, monkeypatch):
        """Identical submissions racing through the check-then-mark window
        must yield ONE queued job and one shared ingest_id. The old
        get-then-SET let every racer enqueue: the same video ingested
        twice and the duplicates burned the tenant's concurrency budget,
        429ing a different legitimate upload."""
        from cogniverse_runtime.ingestion_worker import submit_api

        results = await asyncio.gather(
            *[
                submit_api.enqueue_ingestion(
                    redis,
                    source_url="s3://b/dup.mp4",
                    profile="video",
                    tenant_id="acme:acme",
                )
                for _ in range(4)
            ]
        )

        assert len({r.ingest_id for r in results}) == 1
        assert sum(1 for r in results if not r.existing) == 1
        assert await queue.queue_depth(redis) == 1
        assert await queue.get_active(redis, "acme:acme") == 1


class TestJobDeadline:
    @pytest.mark.asyncio
    async def test_hung_processor_hits_deadline_and_settles(
        self, redis, redis_container, monkeypatch
    ):
        """A processor stuck on a timeoutless await must not stay 'running'
        forever behind its own heartbeat: the deadline fails the job,
        publishes a terminal event, frees the tenant slot, and acks the
        entry."""
        monkeypatch.setenv("REDIS_URL", redis_container)
        monkeypatch.setenv("INGEST_JOB_DEADLINE_SECONDS", "1")
        from cogniverse_runtime.ingestion_worker import worker

        config = worker.WorkerConfig()
        await queue.ensure_consumer_group(redis, config.consumer_group)
        await queue.increment_active(redis, "acme:acme")
        await idempotency.mark_inflight(redis, "sha_hang", "ing_hang", ttl_seconds=600)
        await queue.submit(
            redis,
            ingest_id="ing_hang",
            source_url="s3://b/h.mp4",
            profile="video",
            tenant_id="acme:acme",
            sha="sha_hang",
        )
        jobs = await queue.claim(
            redis, config.consumer_group, config.consumer_id, block_ms=1000
        )

        async def hung(job):
            await asyncio.sleep(600)

        await worker._process_job(redis, jobs[0], config, processor=hung)

        events = [e for _, e in await queue.read_status_since(redis, "ing_hang")]
        terminal = events[-1]
        assert terminal["state"] == "failed"
        assert terminal["error_type"] == "JobDeadlineExceeded"
        assert await queue.get_active(redis, "acme:acme") == 0
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0


class TestMalformedEntryTolerance:
    @pytest.mark.asyncio
    async def test_malformed_entry_dropped_good_sibling_claimed(self, redis):
        """A stream entry missing required fields (external XADD, version
        skew) can never be processed — it must be logged and acked away,
        not crash the claim batch and starve everything behind it."""
        await queue.ensure_consumer_group(redis, "g1")
        await redis.xadd(queue.QUEUE_STREAM, {"ingest_id": "only_field"})
        await queue.submit(
            redis,
            ingest_id="ok1",
            source_url="s3://b/k",
            profile="video",
            tenant_id="t",
            sha="s1",
        )

        jobs = await queue.claim(redis, "g1", "c1", block_ms=1000, count=10)

        assert [j.ingest_id for j in jobs] == ["ok1"]
        pending = await redis.xpending(queue.QUEUE_STREAM, "g1")
        assert pending["pending"] == 1


class TestClaimLoopSurvivesPublishFault:
    @pytest.mark.asyncio
    async def test_publish_fault_leaves_job_pending_and_loop_alive(
        self, redis, redis_container, monkeypatch
    ):
        """A transient Redis error while publishing job status must not crash
        the consumer loop: the running-publish fires before the per-job error
        guard, so its failure escapes _process_job — the loop has to absorb
        it, leave the entry in the PEL for the reaper, and keep claiming and
        processing later jobs. A crash here takes down the whole worker pod
        on any Redis blip."""
        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.setenv("REDIS_URL", redis_container)
        monkeypatch.setenv("INGEST_CLAIM_BLOCK_MS", "200")
        config = worker.WorkerConfig()

        real_publish = queue.publish_status

        async def flaky_publish(r, ingest_id, event):
            if ingest_id == "ing_blip" and event.get("state") == "running":
                raise ConnectionError("redis blip during status publish")
            return await real_publish(r, ingest_id, event)

        monkeypatch.setattr(queue, "publish_status", flaky_publish)

        processed: list[str] = []

        async def processor(job):
            processed.append(job.ingest_id)
            return {"status": "success", "video_id": job.ingest_id}

        await queue.ensure_consumer_group(redis, config.consumer_group)
        await queue.submit(
            redis, "ing_blip", "s3://b/k1", "video", "acme:acme", "sha_b1"
        )

        stop = asyncio.Event()
        loop_task = asyncio.create_task(
            worker._claim_loop(redis, config, stop, processor=processor)
        )
        try:
            deadline = time.time() + 10
            while time.time() < deadline:
                pending = await redis.xpending(
                    queue.QUEUE_STREAM, config.consumer_group
                )
                if pending["pending"] == 1:
                    break
                await asyncio.sleep(0.05)

            await queue.submit(
                redis, "ing_ok", "s3://b/k2", "video", "acme:acme", "sha_b2"
            )
            deadline = time.time() + 10
            while "ing_ok" not in processed and time.time() < deadline:
                await asyncio.sleep(0.05)

            assert not loop_task.done(), "claim loop crashed on a status-publish blip"
            # The blipped job never reached the processor (its publish failed
            # first) and stays pending for the reaper; the later job ran.
            assert processed == ["ing_ok"], processed
            # ing_ok's ack runs after the processor returns; wait for it to
            # land so only the blipped entry remains for the reaper.
            deadline = time.time() + 10
            while time.time() < deadline:
                pending = await redis.xpending(
                    queue.QUEUE_STREAM, config.consumer_group
                )
                if pending["pending"] == 1:
                    break
                await asyncio.sleep(0.05)
            assert pending["pending"] == 1
        finally:
            stop.set()
            await asyncio.wait_for(loop_task, timeout=5)


class TestMalformedEntrySettlement:
    @pytest.mark.asyncio
    async def test_malformed_entry_settles_the_client(self, redis):
        """A version-skew entry (missing a required field but carrying its
        ingest_id + sha) is dropped, but the client polling that ingest_id must
        not wait forever: a failed terminal is published and the inflight marker
        released so a resubmit is not blocked for the 6h TTL."""
        await queue.ensure_consumer_group(redis, "ingestors")
        sha = idempotency.compute_sha("s3://b/skew.mp4", "video", "acme:acme")
        await idempotency.mark_inflight(redis, sha, "ingest_skew", ttl_seconds=3600)

        # An entry missing source_url + profile (producer wrote a newer shape).
        await redis.xadd(
            queue.QUEUE_STREAM,
            {"ingest_id": "ingest_skew", "tenant_id": "acme:acme", "sha": sha},
        )

        jobs = await queue.claim(redis, "ingestors", "consumer_skew", block_ms=1000)
        assert jobs == [], "malformed entry must not be returned as a job"

        # The client's status resolved to failed.
        events = await queue.read_status_since(
            redis, "ingest_skew", last_id="0-0", block_ms=500
        )
        states = [e[1]["state"] for e in events]
        assert "failed" in states, f"no failed terminal published: {states}"

        # The inflight marker was released so a resubmit is not blocked.
        assert await idempotency.get_existing_ingest_id(redis, sha) is None


class TestGraphStageDurability:
    @pytest.mark.no_shared_vespa
    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
    def test_graph_deadline_must_be_positive_and_finite(
        self,
        value,
        redis_container,
        monkeypatch,
    ):
        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.setenv("REDIS_URL", redis_container)
        monkeypatch.setenv("INGEST_GRAPH_DEADLINE_SECONDS", value)

        with pytest.raises(
            RuntimeError,
            match="INGEST_GRAPH_DEADLINE_SECONDS must be positive and finite",
        ):
            worker.WorkerConfig()

    @pytest.mark.asyncio
    @pytest.mark.no_shared_vespa
    async def test_concurrent_graph_failures_retain_each_jobs_queue_state(
        self,
        redis,
        redis_container,
        telemetry_manager_without_phoenix,
        monkeypatch,
    ):
        from cogniverse_foundation.telemetry.config import TelemetryLevel
        from cogniverse_runtime.ingestion_worker import worker

        telemetry_manager_without_phoenix.config.level = TelemetryLevel.BASIC
        monkeypatch.setenv("REDIS_URL", redis_container)
        config = worker.WorkerConfig()
        await queue.ensure_consumer_group(redis, config.consumer_group)
        jobs = []
        for suffix in ("alpha", "beta"):
            tenant = f"acme:{suffix}"
            sha = f"sha-{suffix}"
            ingest_id = f"ing-{suffix}"
            await queue.increment_active(redis, tenant)
            await idempotency.mark_inflight(redis, sha, ingest_id, ttl_seconds=600)
            await queue.submit(
                redis,
                ingest_id=ingest_id,
                source_url=f"s3://media/{suffix}.mp4",
                profile="video",
                tenant_id=tenant,
                sha=sha,
            )
        jobs = await queue.claim(
            redis,
            config.consumer_group,
            config.consumer_id,
            block_ms=1000,
            count=2,
        )
        assert {job.ingest_id for job in jobs} == {"ing-alpha", "ing-beta"}

        rendezvous = asyncio.Barrier(2)
        simultaneous = 0
        peak_simultaneous = 0
        counter_lock = asyncio.Lock()

        async def graph_failed(job):
            nonlocal simultaneous, peak_simultaneous
            async with counter_lock:
                simultaneous += 1
                peak_simultaneous = max(peak_simultaneous, simultaneous)
            await rendezvous.wait()
            async with counter_lock:
                simultaneous -= 1
            raise worker.GraphStageIncomplete(
                f"graph boundary unavailable for {job.ingest_id}"
            )

        await asyncio.gather(
            *[
                worker._process_job(redis, job, config, processor=graph_failed)
                for job in jobs
            ]
        )

        assert peak_simultaneous == 2
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 2
        for job in jobs:
            events = [
                event
                for _, event in await queue.read_status_since(redis, job.ingest_id)
            ]
            assert [event["state"] for event in events] == ["running", "retrying"]
            assert events[-1]["error_type"] == "GraphStageIncomplete"
            assert await queue.get_active(redis, job.tenant_id) == 1
            assert (
                await redis.get(f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}")
                == job.ingest_id
            )
            assert await idempotency.get_done_ingest_id(redis, job.sha) is None

    @pytest.mark.asyncio
    @pytest.mark.no_shared_vespa
    async def test_hung_graph_is_cancelled_after_its_deadline(self, monkeypatch):
        from cogniverse_runtime.ingestion_worker import worker

        class Pipeline:
            def __init__(self, **kwargs):
                pass

            async def process_video_async(self, path, source_uri=None):
                return {
                    "status": "success",
                    "video_id": "video-hung-graph",
                    "results": {
                        "video_id": "video-hung-graph",
                        "embeddings": {"documents_fed": 1},
                    },
                }

        class Locator:
            def __init__(self, tenant_id, config):
                pass

            def localize(self, url):
                return "/tmp/video-hung-graph.mp4"

        graph_started = asyncio.Event()
        graph_cancelled = asyncio.Event()
        marked = []

        async def hung_graph(**kwargs):
            assert marked == [("0-graph-hung", "ing-graph-hung")]
            graph_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                graph_cancelled.set()

        async def mark_graph_pending(job):
            marked.append((job.message_id, job.ingest_id))

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", Pipeline
        )
        monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", Locator)
        monkeypatch.setattr(
            "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
            hung_graph,
        )
        monkeypatch.setattr(
            worker,
            "_prepare_job_context",
            lambda service_urls: (object(), object()),
        )
        job = queue.IngestJob(
            message_id="0-graph-hung",
            ingest_id="ing-graph-hung",
            source_url="s3://media/video-hung-graph.mp4",
            profile="video",
            tenant_id="acme:hung",
            sha="sha-graph-hung",
        )

        with pytest.raises(
            worker.GraphStageIncomplete,
            match="graph extraction exceeded the 0.05s deadline",
        ):
            await asyncio.wait_for(
                worker._default_processor(
                    job,
                    service_urls=None,
                    mark_graph_pending=mark_graph_pending,
                    graph_deadline_s=0.05,
                ),
                timeout=1,
            )

        assert graph_started.is_set()
        assert graph_cancelled.is_set()
        assert marked == [("0-graph-hung", "ing-graph-hung")]

    @pytest.mark.asyncio
    async def test_real_vespa_content_survives_retry_and_completes_with_graph(
        self,
        redis,
        redis_container,
        vespa_instance,
        telemetry_manager_without_phoenix,
        monkeypatch,
    ):
        from cogniverse_foundation.telemetry.config import TelemetryLevel
        from cogniverse_runtime.ingestion_worker import reaper, worker
        from cogniverse_vespa._vespa_factory import make_vespa_app

        telemetry_manager_without_phoenix.config.level = TelemetryLevel.BASIC
        monkeypatch.setenv("REDIS_URL", redis_container)
        config = worker.WorkerConfig()
        config.reaper_max_deliveries = 0
        tenant = "test:unit"
        ingest_id = f"ing-graph-{uuid.uuid4().hex[:8]}"
        sha = f"sha-{uuid.uuid4().hex[:8]}"
        doc_id = f"curie-{uuid.uuid4().hex[:8]}_seg_0"
        schema = "video_colpali_smol500_mv_frame_test_unit"
        app = make_vespa_app(
            url="http://localhost",
            port=vespa_instance["http_port"],
        )
        fields = {
            "video_id": doc_id.removesuffix("_seg_0"),
            "video_title": "Marie Curie and radium",
            "source_url": "s3://media/curie-radium.mp4",
            "creation_timestamp": 1785859200,
            "segment_id": 0,
            "start_time": 12.0,
            "end_time": 18.0,
            "segment_description": "Marie Curie discovered radium in Paris.",
            "audio_transcript": "Marie Curie discovered radium.",
            "entity_ids": [],
            "relation_ids": [],
            "claim_ids": [],
        }

        class Pipeline:
            feed_count = 0

            def __init__(self, **kwargs):
                assert kwargs["tenant_id"] == tenant

            async def process_video_async(self, path, source_uri=None):
                response = await asyncio.to_thread(
                    app.feed_data_point,
                    schema=schema,
                    namespace="content",
                    data_id=doc_id,
                    fields=fields,
                )
                assert response.is_successful(), response.json
                Pipeline.feed_count += 1
                return {
                    "status": "success",
                    "video_id": fields["video_id"],
                    "results": {
                        "video_id": fields["video_id"],
                        "embeddings": {"documents_fed": 1},
                    },
                }

        class Locator:
            def __init__(self, tenant_id, config):
                pass

            def localize(self, url):
                return "/tmp/curie-radium.mp4"

        graph_attempts = 0

        async def graph_once_unavailable(**kwargs):
            nonlocal graph_attempts
            graph_attempts += 1
            if graph_attempts == 1:
                raise ConnectionError("knowledge-graph Vespa connection reset")
            return {"nodes_upserted": 2, "edges_upserted": 1, "graph_failed": 0}

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", Pipeline
        )
        monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", Locator)
        monkeypatch.setattr(
            "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
            graph_once_unavailable,
        )
        monkeypatch.setattr(
            worker,
            "_prepare_job_context",
            lambda service_urls: (object(), object()),
        )

        await queue.ensure_consumer_group(redis, config.consumer_group)
        await queue.increment_active(redis, tenant)
        await idempotency.mark_inflight(redis, sha, ingest_id, ttl_seconds=600)
        await queue.submit(
            redis,
            ingest_id=ingest_id,
            source_url=fields["source_url"],
            profile="video_colpali_smol500_mv_frame",
            tenant_id=tenant,
            sha=sha,
        )
        [job] = await queue.claim(
            redis,
            config.consumer_group,
            config.consumer_id,
            block_ms=1000,
        )
        processor = functools.partial(
            worker._default_processor,
            service_urls=None,
            mark_graph_pending=functools.partial(worker._mark_graph_pending, redis),
            graph_deadline_s=1,
        )

        try:
            await worker._process_job(redis, job, config, processor=processor)

            first_read = await asyncio.to_thread(
                app.get_data,
                schema=schema,
                namespace="content",
                data_id=doc_id,
            )
            assert first_read.status_code == 200
            assert (
                first_read.json["fields"]["segment_description"]
                == fields["segment_description"]
            )
            events = [
                event for _, event in await queue.read_status_since(redis, ingest_id)
            ]
            assert [event["state"] for event in events] == ["running", "retrying"]
            assert not any(event["state"] in {"complete", "failed"} for event in events)
            pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
            assert pending["pending"] == 1
            assert await queue.get_active(redis, tenant) == 1
            assert await idempotency.get_existing_ingest_id(redis, sha) == ingest_id
            assert (
                await redis.get(f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}")
                == ingest_id
            )

            recovered = await reaper.run_reaper_once(
                redis,
                config,
                min_idle_ms=0,
                processor=processor,
            )

            assert recovered == 1
            assert Pipeline.feed_count == 2
            assert graph_attempts == 2
            final_read = await asyncio.to_thread(
                app.get_data,
                schema=schema,
                namespace="content",
                data_id=doc_id,
            )
            assert final_read.status_code == 200
            assert final_read.json["fields"] == {
                key: value
                for key, value in fields.items()
                if key not in {"entity_ids", "relation_ids", "claim_ids"}
            }
            assert {
                key
                for key in ("entity_ids", "relation_ids", "claim_ids")
                if key in final_read.json["fields"]
            } == set()
            events = [
                event for _, event in await queue.read_status_since(redis, ingest_id)
            ]
            assert [event["state"] for event in events] == [
                "running",
                "retrying",
                "running",
                "complete",
            ]
            assert events[-1]["result"]["graph_nodes"] == 2
            assert events[-1]["result"]["graph_edges"] == 1
            assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
            assert await queue.get_active(redis, tenant) == 0
            pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
            assert pending["pending"] == 0
            assert (
                await redis.get(f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}")
                is None
            )
        finally:
            await asyncio.to_thread(
                app.delete_data,
                schema=schema,
                namespace="content",
                data_id=doc_id,
            )
