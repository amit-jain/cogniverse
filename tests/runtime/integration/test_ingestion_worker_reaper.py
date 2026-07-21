"""Crash-recovery reaper for orphaned ingestion PEL entries, against real Redis.

A worker that dies between claim and ack leaves its entry in a dead
consumer's PEL — invisible to ``claim()`` (which reads only new entries)
forever. These tests prove the reaper's contract on a real server:
orphans idle past the threshold are reclaimed and re-driven exactly once,
finished-but-unacked jobs are settled without duplicate side effects,
live recent claims are never stolen, and two concurrent sweeps never
process one entry twice (XAUTOCLAIM claims each entry for exactly one
caller).
"""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import subprocess
import time

import pytest

from cogniverse_runtime.ingestion_worker import idempotency, queue, reaper, worker
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

CONTAINER_NAME = "redis-ingestion-reaper-tests"


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
async def redis(redis_container, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_container)
    await close_redis()
    client = await get_redis(redis_container)
    await client.flushdb()
    yield client
    await close_redis()


async def _orphan_job(redis, config, *, ingest_id: str, sha: str, tenant: str):
    """Enqueue one job with full submit-side state (inflight marker + active
    counter) and claim it as a consumer that then 'dies' without acking."""
    await queue.ensure_consumer_group(redis, config.consumer_group)
    await idempotency.mark_inflight(redis, sha, ingest_id, ttl_seconds=600)
    await queue.increment_active(redis, tenant)
    await queue.submit(
        redis,
        ingest_id=ingest_id,
        source_url=f"s3://b/{ingest_id}.mp4",
        profile="video",
        tenant_id=tenant,
        sha=sha,
    )
    jobs = await queue.claim(redis, config.consumer_group, "dead-1", block_ms=1000)
    assert [j.ingest_id for j in jobs] == [ingest_id]
    return jobs[0]


class TestReaperRecovery:
    @pytest.mark.asyncio
    async def test_reaper_redrives_orphaned_unprocessed_job(self, redis):
        """A job claimed by a dead consumer and never processed is reclaimed,
        processed exactly once through the normal path, and fully settled:
        done marker set, inflight cleared, tenant slot freed, PEL and stream
        drained, clean ``complete`` terminal event published."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-1"
        tenant, sha, ingest_id = "acme:acme", "sha_orphan", "ing_orphan"
        await _orphan_job(redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant)

        processed: list = []

        async def _proc(job):
            processed.append(job.ingest_id)
            return {"status": "success", "video_id": "v1", "results": {}}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_proc
        )

        assert recovered == 1
        assert processed == [ingest_id]
        assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
        assert await idempotency.get_existing_ingest_id(redis, sha) == ingest_id
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") is None
        assert await queue.get_active(redis, tenant) == 0
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert events[-1]["state"] == "complete"
        assert "cleanup_error" not in events[-1]

    @pytest.mark.asyncio
    async def test_reaper_settles_finished_but_unacked_job_without_reprocess(
        self, redis
    ):
        """The dead worker completed everything except the ack (the ack-blip
        case). The reaper must ONLY settle: ack + clear the stale inflight
        marker — no reprocess, no second decrement that would free a slot a
        different still-running job of the same tenant holds."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-1"
        tenant, sha, ingest_id = "acme:acme", "sha_done", "ing_done"
        await _orphan_job(redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant)
        # Dead worker's completed run: mark_done + clear_inflight +
        # decrement ran; only the ack failed.
        await idempotency.mark_done(redis, sha, ingest_id, ttl_seconds=600)
        await idempotency.clear_inflight(redis, sha)
        await queue.decrement_active(redis, tenant)
        # A DIFFERENT job of the same tenant is still running and holds
        # one slot; the reaper must not free it.
        await queue.increment_active(redis, tenant)

        processed: list = []

        async def _proc(job):
            processed.append(job.ingest_id)
            return {}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_proc
        )

        assert recovered == 1
        assert processed == []
        assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
        assert await queue.get_active(redis, tenant) == 1
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0

    @pytest.mark.asyncio
    async def test_reaper_leaves_live_recent_claims_alone(self, redis):
        """An entry a live worker claimed moments ago is below the idle
        threshold and must never be stolen mid-processing."""
        config = worker.WorkerConfig()
        config.consumer_id = "reaper-1"
        tenant, sha, ingest_id = "acme:acme", "sha_live", "ing_live"
        await _orphan_job(redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant)

        processed: list = []

        async def _proc(job):
            processed.append(job.ingest_id)
            return {}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=60000, processor=_proc
        )

        assert recovered == 0
        assert processed == []
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1
        assert await queue.get_active(redis, tenant) == 1

    @pytest.mark.asyncio
    async def test_reaper_never_reclaims_heartbeating_live_job(self, redis):
        """A live worker mid-pipeline heartbeats its claim, so reaper sweeps
        whose min-idle threshold is far below the pipeline duration must
        never steal the entry — without the heartbeat, XAUTOCLAIM (idle-based
        only) reclaims the still-processing job, runs the pipeline a second
        time concurrently, double-decrements the tenant counter, and after
        enough reclaims dead-letters a legitimately long video as poison."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-1"
        config.heartbeat_interval_s = 0.05
        tenant, sha, ingest_id = "acme:acme", "sha_hb", "ing_hb"
        await queue.ensure_consumer_group(redis, config.consumer_group)
        await idempotency.mark_inflight(redis, sha, ingest_id, ttl_seconds=600)
        await queue.increment_active(redis, tenant)
        await queue.submit(
            redis,
            ingest_id=ingest_id,
            source_url="s3://b/hb.mp4",
            profile="video",
            tenant_id=tenant,
            sha=sha,
        )
        jobs = await queue.claim(
            redis, config.consumer_group, config.consumer_id, block_ms=1000
        )
        assert [j.ingest_id for j in jobs] == [ingest_id]

        # Warm the telemetry singleton so its one-time cold init (a sync,
        # loop-blocking setup) doesn't consume the idle budget before the
        # first heartbeat can run.
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        get_telemetry_manager()

        runs: list = []

        async def _slow(job):
            runs.append(job.ingest_id)
            await asyncio.sleep(0.8)
            return {"status": "success", "video_id": "v1", "results": {}}

        async def _sweeps():
            recovered = 0
            for _ in range(3):
                await asyncio.sleep(0.2)
                recovered += await reaper.run_reaper_once(
                    redis, config, min_idle_ms=200, processor=_slow
                )
            return recovered

        _, recovered = await asyncio.gather(
            worker._process_job(redis, jobs[0], config, processor=_slow),
            _sweeps(),
        )

        assert runs == [ingest_id], f"pipeline ran {len(runs)}x for one job"
        assert recovered == 0, "a sweep reclaimed a live, heartbeating claim"
        assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
        assert await queue.get_active(redis, tenant) == 0
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert events[-1]["state"] == "complete"
        assert "cleanup_error" not in events[-1]

    @pytest.mark.asyncio
    async def test_reaper_dead_letters_poison_job_after_max_deliveries(self, redis):
        """A job that keeps getting redelivered without ever completing (a
        pod-killing poison message — e.g. an OOM-sized video) must not be
        re-driven forever: past the delivery cap the reaper abandons it to the
        dead stream, publishes an observable ``failed`` terminal, settles the
        submit-side state (slot freed, inflight cleared), and drains the PEL.
        No done marker is written, so a corrected re-submission re-enqueues."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-1"
        config.reaper_max_deliveries = 3
        tenant, sha, ingest_id = "acme:acme", "sha_poison", "ing_poison"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        # The dead consumer's claim was delivery 1; two crash-redeliveries
        # bump it to 3, and the reaper's own XAUTOCLAIM makes it 4 > cap.
        for consumer in ("dead-2", "dead-3"):
            await redis.xclaim(
                queue.QUEUE_STREAM,
                config.consumer_group,
                consumer,
                min_idle_time=0,
                message_ids=[job.message_id],
            )

        processed: list = []

        async def _proc(j):
            processed.append(j.ingest_id)
            return {}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_proc
        )

        assert recovered == 1
        assert processed == [], "poison job was re-driven past the delivery cap"
        dead = await redis.xrange(reaper.DEAD_STREAM)
        assert len(dead) == 1
        fields = dead[0][1]
        assert fields["ingest_id"] == ingest_id
        assert fields["tenant_id"] == tenant
        assert fields["sha"] == sha
        assert int(fields["times_delivered"]) == 4
        assert await idempotency.get_done_ingest_id(redis, sha) is None
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") is None
        assert await queue.get_active(redis, tenant) == 0
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert events[-1]["state"] == "failed"
        assert events[-1]["error_type"] == "MaxDeliveriesExceeded"
        assert "abandoned after 4 deliveries" in events[-1]["error"]

    @pytest.mark.asyncio
    async def test_dead_letter_crash_before_ack_never_double_settles(
        self, redis, monkeypatch
    ):
        """A crash after the dead-letter settle but before the ack redelivers
        the poison entry; the next sweep must NOT settle again — a second
        decrement would free a slot a DIFFERENT still-running job of the same
        tenant holds, and the dead stream would gain a duplicate entry. Only
        the terminal publish + ack are repeated (at-least-once)."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-1"
        config.reaper_max_deliveries = 3
        tenant, sha, ingest_id = "acme:acme", "sha_crash", "ing_crash"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        # A DIFFERENT job of the same tenant is running and holds one slot.
        await queue.increment_active(redis, tenant)
        for consumer in ("dead-2", "dead-3"):
            await redis.xclaim(
                queue.QUEUE_STREAM,
                config.consumer_group,
                consumer,
                min_idle_time=0,
                message_ids=[job.message_id],
            )

        real_publish = queue.publish_status

        async def _crash(*args, **kwargs):
            raise ConnectionError("pod killed before terminal publish")

        monkeypatch.setattr(queue, "publish_status", _crash)
        with pytest.raises(ConnectionError):
            await reaper._dead_letter(redis, config, job, 4)
        monkeypatch.setattr(queue, "publish_status", real_publish)

        # Settle committed exactly once; entry still pending (no ack ran).
        assert len(await redis.xrange(reaper.DEAD_STREAM)) == 1
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") is None
        assert await redis.get(f"{reaper.DEAD_MARKER_PREFIX}{job.message_id}")
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1

        # Next sweep redelivers (bumping to 4 > cap): it must skip the settle
        # and only re-publish the terminal + ack.
        recovered = await reaper.run_reaper_once(redis, config, min_idle_ms=0)
        assert recovered == 1
        dead = await redis.xrange(reaper.DEAD_STREAM)
        assert len(dead) == 1, "crash-redelivery duplicated the dead entry"
        assert dead[0][1]["times_delivered"] == "4"
        assert await queue.get_active(redis, tenant) == 1, (
            "second settle freed a slot the other running job holds"
        )
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert events[-1]["state"] == "failed"
        assert events[-1]["error_type"] == "MaxDeliveriesExceeded"
        assert "abandoned after 4 deliveries" in events[-1]["error"]

    @pytest.mark.asyncio
    async def test_dead_letter_settle_fault_leaves_state_retryable(
        self, redis, monkeypatch
    ):
        """Redis failing the settle call itself must tear nothing: no marker,
        no dead entry, counter and inflight untouched, entry still pending —
        the next sweep retries the full settle successfully."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-1"
        config.reaper_max_deliveries = 3
        tenant, sha, ingest_id = "acme:acme", "sha_efault", "ing_efault"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        await queue.increment_active(redis, tenant)

        with monkeypatch.context() as m:

            async def _down(*args, **kwargs):
                raise ConnectionError("redis reset on eval")

            m.setattr(redis, "eval", _down)
            with pytest.raises(ConnectionError, match="reset on eval"):
                await reaper._dead_letter(redis, config, job, 4)

        assert await redis.get(f"{reaper.DEAD_MARKER_PREFIX}{job.message_id}") is None
        assert len(await redis.xrange(reaper.DEAD_STREAM)) == 0
        assert await queue.get_active(redis, tenant) == 2
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") == ingest_id
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1

        await reaper._dead_letter(redis, config, job, 4)
        assert len(await redis.xrange(reaper.DEAD_STREAM)) == 1
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") is None
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_reapers_process_each_orphan_exactly_once(self, redis):
        """Two sweeps running concurrently over two orphans: XAUTOCLAIM hands
        each entry to exactly one caller, so no orphan is processed twice and
        both end fully settled."""
        config_a = worker.WorkerConfig()
        config_a.consumer_id = "live-a"
        config_b = worker.WorkerConfig()
        config_b.consumer_id = "live-b"
        tenant = "acme:acme"
        await _orphan_job(
            redis, config_a, ingest_id="ing_c1", sha="sha_c1", tenant=tenant
        )
        # claim() above consumed the group cursor; enqueue+claim the second
        # orphan the same way.
        await _orphan_job(
            redis, config_a, ingest_id="ing_c2", sha="sha_c2", tenant=tenant
        )
        # Let both orphans age past the threshold; a just-reclaimed entry
        # (idle ~0) then stays below it while its new owner processes.
        await asyncio.sleep(0.3)

        processed: list = []
        lock = asyncio.Lock()

        async def _proc(job):
            async with lock:
                processed.append(job.ingest_id)
            await asyncio.sleep(0.02)
            return {"status": "success", "results": {}}

        recovered = await asyncio.gather(
            reaper.run_reaper_once(
                redis, config_a, min_idle_ms=150, processor=_proc, count=1
            ),
            reaper.run_reaper_once(
                redis, config_b, min_idle_ms=150, processor=_proc, count=1
            ),
        )

        assert sum(recovered) == 2
        assert sorted(processed) == ["ing_c1", "ing_c2"], (
            f"orphans processed {processed} — an entry ran twice or was lost"
        )
        pending = await redis.xpending(queue.QUEUE_STREAM, "ingestors")
        assert pending["pending"] == 0
        assert await queue.queue_depth(redis) == 0
        assert await queue.get_active(redis, tenant) == 0


class TestReaperWiredIntoWorkerRun:
    @pytest.mark.asyncio
    async def test_run_reaper_recovers_orphan_end_to_end(
        self, redis, redis_container, monkeypatch
    ):
        """worker.run() with the reaper enabled recovers a dead consumer's
        orphan without any claim() ever seeing it: XREADGROUP('>') skips
        already-delivered entries, so only the reaper task can re-drive it."""
        monkeypatch.setenv("REDIS_URL", redis_container)
        monkeypatch.setenv("INGEST_REAPER_INTERVAL_SECONDS", "1")
        monkeypatch.setenv("INGEST_REAPER_MIN_IDLE_MS", "0")
        monkeypatch.setenv("INGEST_CLAIM_BLOCK_MS", "200")

        seed_config = worker.WorkerConfig()
        tenant, sha, ingest_id = "acme:acme", "sha_wire", "ing_wire"
        await _orphan_job(
            redis, seed_config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )

        processed: list = []

        async def _proc(job):
            processed.append(job.ingest_id)
            return {"status": "success", "video_id": "v1", "results": {}}

        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop=stop, processor=_proc))
        deadline = time.time() + 10
        while not processed and time.time() < deadline:
            await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=10)

        assert processed == [ingest_id]
        # run() closed the shared client; reopen for the assertions.
        client = await get_redis(redis_container)
        assert await idempotency.get_done_ingest_id(client, sha) == ingest_id
        pending = await client.xpending(queue.QUEUE_STREAM, seed_config.consumer_group)
        assert pending["pending"] == 0
        assert await queue.queue_depth(client) == 0
        assert await queue.get_active(client, tenant) == 0
