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
import logging
import os
import platform
import socket
import subprocess
import time

import pytest

from cogniverse_runtime.ingestion_worker import idempotency, queue, reaper, worker
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

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


@pytest.fixture(autouse=True)
def _minio_outage_env(monkeypatch):
    """MinIO endpoint that refuses connections, single boto3 attempt: the
    worker's original-filename lookup exercises its real outage fallback
    ("") instead of failing the job on unset env."""
    monkeypatch.setenv("MINIO_ENDPOINT", "http://127.0.0.1:29071")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("AWS_RETRY_MODE", "standard")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")


@pytest.fixture(autouse=True)
def worker_telemetry(telemetry_manager_without_phoenix):
    from cogniverse_foundation.telemetry.config import TelemetryLevel

    telemetry_manager_without_phoenix.config.level = TelemetryLevel.BASIC
    return telemetry_manager_without_phoenix


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


REAPER_LOGGER = "cogniverse_runtime.ingestion_worker.reaper"


def _reaper_lines(caplog, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == REAPER_LOGGER and r.levelno == level
    ]


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
    async def test_orphans_are_claimed_one_at_a_time(self, redis):
        """While the reaper re-drives one orphan, the remaining orphans must
        still belong to the dead consumer — claiming a batch up front parks
        the tail with its idle clock running, so past the min-idle window a
        second replica's reaper reclaims a tail entry and both re-drive it
        concurrently (duplicate ingestion, double-decremented counter)."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-reaper"
        tenant = "acme:acme"
        for i in range(3):
            await _orphan_job(
                redis,
                config,
                ingest_id=f"ing_tail_{i}",
                sha=f"sha_tail_{i}",
                tenant=tenant,
            )

        owners_during_first: list = []
        gate = asyncio.Event()

        async def _proc(job):
            if not gate.is_set():
                entries = await redis.xpending_range(
                    queue.QUEUE_STREAM,
                    config.consumer_group,
                    min="-",
                    max="+",
                    count=10,
                )
                owners_during_first.extend(e["consumer"] for e in entries)
                gate.set()
            return {"status": "success", "video_id": job.ingest_id, "results": {}}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_proc
        )

        assert recovered == 3
        # During the first re-drive: exactly one entry claimed by the reaper,
        # the other two still owned by the dead consumer (idle only while
        # actually about to be processed).
        owners = [
            o.decode() if isinstance(o, bytes) else o for o in owners_during_first
        ]
        assert owners.count("live-reaper") == 1, owners
        assert owners.count("dead-1") == 2, owners
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0

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
    async def test_reaper_redrives_marked_graph_stage_past_delivery_cap(
        self,
        redis,
        telemetry_manager_without_phoenix,
    ):
        from cogniverse_foundation.telemetry.config import TelemetryLevel

        telemetry_manager_without_phoenix.config.level = TelemetryLevel.BASIC
        config = worker.WorkerConfig()
        config.consumer_id = "live-graph"
        config.reaper_max_deliveries = 0
        tenant, sha, ingest_id = (
            "acme:graph",
            "sha_graph_resume",
            "ing_graph_resume",
        )
        job = await _orphan_job(
            redis,
            config,
            ingest_id=ingest_id,
            sha=sha,
            tenant=tenant,
        )
        await worker._mark_graph_pending(redis, job)
        processed: list[str] = []

        async def _complete(j):
            processed.append(j.ingest_id)
            return {
                "status": "success",
                "video_id": "video-graph-resumed",
                "graph_nodes": 2,
                "graph_edges": 1,
            }

        recovered = await reaper.run_reaper_once(
            redis,
            config,
            min_idle_ms=0,
            processor=_complete,
        )

        assert recovered == 1
        assert processed == [ingest_id]
        assert await redis.xrange(reaper.DEAD_STREAM) == []
        assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
        assert await queue.get_active(redis, tenant) == 0
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0
        events = [event for _, event in await queue.read_status_since(redis, ingest_id)]
        assert events[-1]["state"] == "complete"
        assert events[-1]["result"] == {
            "video_id": "video-graph-resumed",
            "keyframes": 0,
            "documents_fed": 0,
            "chunks": 0,
            "graph_nodes": 2,
            "graph_edges": 1,
        }
        assert (
            await redis.get(f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}")
            is None
        )

    @pytest.mark.asyncio
    async def test_reaper_redrive_keeps_graph_stage_failure_retrying(
        self, redis, telemetry_manager_without_phoenix
    ):
        """A re-driven job whose graph stage fails again stays nonterminal:
        the sweep publishes ``retrying`` with the graph error, retains the
        PEL entry and graph-pending marker, and leaves the tenant slot,
        in-flight marker and done marker untouched for the next sweep."""
        from cogniverse_foundation.telemetry.config import TelemetryLevel

        telemetry_manager_without_phoenix.config.level = TelemetryLevel.BASIC
        config = worker.WorkerConfig()
        config.consumer_id = "live-graph-retry"
        tenant, sha, ingest_id = "acme:graph", "sha_graph_retry", "ing_graph_retry"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        await worker._mark_graph_pending(redis, job)

        async def _graph_fails(j):
            raise worker.GraphStageIncomplete(
                f"graph extraction failed for ingest {j.ingest_id}"
            ) from RuntimeError(
                "claim extraction failed for source 'doc1' across 1 segments"
            )

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_graph_fails
        )

        assert recovered == 1
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert [e["state"] for e in events] == ["running", "retrying"]
        assert events[-1]["error_type"] == "GraphStageIncomplete"
        assert events[-1]["error"] == (
            f"graph extraction failed for ingest {ingest_id}"
        )
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1
        assert (
            await redis.get(f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}")
            == ingest_id
        )
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") == ingest_id
        assert await idempotency.get_done_ingest_id(redis, sha) is None
        assert await redis.xrange(reaper.DEAD_STREAM) == []

    @pytest.mark.asyncio
    async def test_graph_pending_redrive_is_held_until_its_backoff_elapses(
        self, redis, caplog
    ):
        """Each graph-stage failure is recorded with its cause; the reaper
        then waits ``graph_redrive_hold_ms(redrives, base_ms=min_idle_ms)``
        from that failure before re-running the pipeline. Inside the hold
        the entry is claimed but not processed and keeps every nonterminal
        invariant; the first sweep past the hold re-drives it and names
        the re-drive number, the last cause and the next hold."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-graph-hold"
        tenant, sha, ingest_id = "acme:graph", "sha_graph_hold", "ing_graph_hold"
        source = f"s3://b/{ingest_id}.mp4"
        cause = f"graph extraction failed for ingest {ingest_id}"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        await worker._mark_graph_pending(redis, job)
        marker_key = f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}"
        redrive_key = f"{worker.GRAPH_REDRIVE_KEY_PREFIX}{job.message_id}"
        processed: list[str] = []

        async def _graph_fails(j):
            processed.append(j.ingest_id)
            raise worker.GraphStageIncomplete(cause)

        with caplog.at_level(logging.INFO, logger=REAPER_LOGGER):
            for _ in range(2):
                recovered = await reaper.run_reaper_once(
                    redis, config, min_idle_ms=0, processor=_graph_fails
                )
                assert recovered == 1
        assert processed == [ingest_id, ingest_id]
        assert _reaper_lines(caplog, logging.WARNING) == [
            f"Reaper re-driving graph-pending ingest {ingest_id} (tenant={tenant}, "
            f"source={source}): re-drive 1, last outcome: none recorded; another "
            "failure holds re-drive 2 for 0s",
            f"Reaper re-driving graph-pending ingest {ingest_id} (tenant={tenant}, "
            f"source={source}): re-drive 2, last outcome: {cause}; another failure "
            "holds re-drive 3 for 0s",
        ]
        # The first sweep moved the entry onto the live consumer, so the dead
        # owner owns nothing and is dropped once; the second finds no name idle.
        assert _reaper_lines(caplog, logging.INFO) == [
            "Reaper dropped 1 idle consumer name(s) owning nothing: ['dead-1']"
        ]
        state = await redis.hgetall(redrive_key)
        assert set(state) == {"redrives", "at_ms", "cause"}
        assert state["redrives"] == "2"
        assert state["cause"] == cause
        failed_at_ms = int(state["at_ms"])

        # Hold before re-drive 3 is 200ms << 2 = 800ms from that failure. The
        # entry is claimable once idle 200ms but must not be processed.
        await asyncio.sleep(0.25)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=REAPER_LOGGER):
            held = await reaper.run_reaper_once(
                redis, config, min_idle_ms=200, processor=_graph_fails
            )
        assert held == 0
        assert processed == [ingest_id, ingest_id]
        assert _reaper_lines(caplog, logging.WARNING) == []
        [hold_line] = _reaper_lines(caplog, logging.INFO)
        hold_prefix = (
            f"Reaper holding graph-pending ingest {ingest_id} (tenant={tenant}, "
            f"source={source}): 2 re-drives so far, last outcome: {cause}; "
            "re-drive 3 due in "
        )
        assert hold_line.startswith(hold_prefix), hold_line
        remaining_s = float(hold_line.removeprefix(hold_prefix).removesuffix("s"))
        assert 0 < remaining_s <= 0.8
        assert await redis.hgetall(redrive_key) == {
            "redrives": "2",
            "at_ms": str(failed_at_ms),
            "cause": cause,
        }
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert [e["state"] for e in events] == ["running", "retrying"] * 2
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1
        assert await redis.get(marker_key) == ingest_id
        assert await redis.xrange(reaper.DEAD_STREAM) == []
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") == ingest_id

        # Past the hold: re-driven, outcome recorded as re-drive 3.
        await asyncio.sleep(0.7)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=REAPER_LOGGER):
            recovered = await reaper.run_reaper_once(
                redis, config, min_idle_ms=200, processor=_graph_fails
            )
        assert recovered == 1
        assert processed == [ingest_id] * 3
        assert _reaper_lines(caplog, logging.INFO) == []
        assert _reaper_lines(caplog, logging.WARNING) == [
            f"Reaper re-driving graph-pending ingest {ingest_id} (tenant={tenant}, "
            f"source={source}): re-drive 3, last outcome: {cause}; another failure "
            "holds re-drive 4 for 1.6s",
        ]
        state = await redis.hgetall(redrive_key)
        assert state["redrives"] == "3"
        assert state["cause"] == cause
        assert int(state["at_ms"]) > failed_at_ms
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert [e["state"] for e in events] == ["running", "retrying"] * 3
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1
        assert await redis.get(marker_key) == ingest_id
        assert await redis.xrange(reaper.DEAD_STREAM) == []

    @pytest.mark.asyncio
    async def test_graph_pending_entry_is_never_acked_or_dead_lettered(self, redis):
        """The delivery cap never reaches a graph-pending entry. With the PEL
        delivery count already past ``reaper_max_deliveries``, re-drive 1,
        re-drive 5 and re-drive 40 (far past where the hold clamps at its
        cap) each re-run the pipeline and leave the entry pending, un-acked,
        off the dead stream, with its marker, tenant slot and in-flight
        record intact and no done marker."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-graph-forever"
        config.reaper_max_deliveries = 3
        tenant, sha, ingest_id = (
            "acme:graph",
            "sha_graph_forever",
            "ing_graph_forever",
        )
        cause = f"graph extraction failed for ingest {ingest_id}"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        await worker._mark_graph_pending(redis, job)
        # dead-1's claim was delivery 1; three crash-redeliveries make it 4,
        # past the cap before the reaper's own claim bumps it further.
        for consumer in ("dead-2", "dead-3", "dead-4"):
            await redis.xclaim(
                queue.QUEUE_STREAM,
                config.consumer_group,
                consumer,
                min_idle_time=0,
                message_ids=[job.message_id],
            )
        marker_key = f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}"
        redrive_key = f"{worker.GRAPH_REDRIVE_KEY_PREFIX}{job.message_id}"
        processed: list[str] = []

        async def _graph_fails(j):
            processed.append(j.ingest_id)
            raise worker.GraphStageIncomplete(cause)

        async def _assert_still_pending(redrives: int) -> None:
            assert await redis.xrange(reaper.DEAD_STREAM) == []
            pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
            assert pending["pending"] == 1
            assert await queue.queue_depth(redis) == 1
            assert (
                await queue.times_delivered(
                    redis, config.consumer_group, job.message_id
                )
                == 4 + redrives
            )
            assert await redis.get(marker_key) == ingest_id
            assert (await redis.hgetall(redrive_key))["redrives"] == str(redrives)
            assert await queue.get_active(redis, tenant) == 1
            assert (
                await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") == ingest_id
            )
            assert await idempotency.get_done_ingest_id(redis, sha) is None
            events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
            assert [e["state"] for e in events] == ["running", "retrying"] * redrives
            assert events[-1] == {
                "state": "retrying",
                "ingest_id": ingest_id,
                "error": cause,
                "error_type": "GraphStageIncomplete",
            }

        for redrive in range(1, 41):
            recovered = await reaper.run_reaper_once(
                redis, config, min_idle_ms=0, processor=_graph_fails
            )
            assert recovered == 1
            if redrive in (1, 5, 40):
                await _assert_still_pending(redrive)
        assert processed == [ingest_id] * 40

    @pytest.mark.asyncio
    async def test_graph_redrive_record_fault_raises_before_the_pipeline_runs(
        self, redis, monkeypatch
    ):
        """Redis failing the re-drive record write raises out of the sweep —
        never a silent hold or an unrecorded run: the pipeline does not run,
        the entry stays pending with its marker, and the next sweep re-drives
        it as re-drive 1 and settles it."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-graph-fault"
        tenant, sha, ingest_id = "acme:graph", "sha_graph_fault", "ing_graph_fault"
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        await worker._mark_graph_pending(redis, job)
        marker_key = f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}"
        redrive_key = f"{worker.GRAPH_REDRIVE_KEY_PREFIX}{job.message_id}"
        processed: list[str] = []

        async def _complete(j):
            processed.append(j.ingest_id)
            return {"status": "success", "video_id": "v-graph-fault", "results": {}}

        with monkeypatch.context() as m:

            async def _down(*args, **kwargs):
                raise ConnectionError("redis reset on hset")

            m.setattr(redis, "hset", _down)
            with pytest.raises(ConnectionError, match="reset on hset"):
                await reaper.run_reaper_once(
                    redis, config, min_idle_ms=0, processor=_complete
                )

        assert processed == []
        assert await redis.hgetall(redrive_key) == {}
        assert await redis.get(marker_key) == ingest_id
        assert await redis.xrange(reaper.DEAD_STREAM) == []
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") == ingest_id

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_complete
        )
        assert recovered == 1
        assert processed == [ingest_id]
        assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
        assert await redis.get(marker_key) is None
        assert await redis.hgetall(redrive_key) == {}
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 0

    @pytest.mark.asyncio
    async def test_graph_failure_note_fault_raises_and_keeps_the_entry_pending(
        self, redis, monkeypatch
    ):
        """Redis failing the failure-note write raises out of ``_process_job``
        with the marker already set and nothing acked or published as
        terminal, so the reaper's next sweep re-drives the entry."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-graph-note-fault"
        tenant, sha, ingest_id = (
            "acme:graph",
            "sha_graph_note_fault",
            "ing_graph_note_fault",
        )
        job = await _orphan_job(
            redis, config, ingest_id=ingest_id, sha=sha, tenant=tenant
        )
        marker_key = f"{worker.GRAPH_PENDING_KEY_PREFIX}{job.message_id}"
        redrive_key = f"{worker.GRAPH_REDRIVE_KEY_PREFIX}{job.message_id}"

        async def _graph_fails(j):
            raise worker.GraphStageIncomplete(
                f"graph extraction failed for ingest {j.ingest_id}"
            )

        with monkeypatch.context() as m:

            async def _down(*args, **kwargs):
                raise ConnectionError("redis reset on hset")

            m.setattr(redis, "hset", _down)
            with pytest.raises(ConnectionError, match="reset on hset"):
                await worker._process_job(redis, job, config, processor=_graph_fails)

        assert await redis.get(marker_key) == ingest_id
        assert await redis.hgetall(redrive_key) == {}
        events = [e for _, e in await queue.read_status_since(redis, ingest_id)]
        assert [e["state"] for e in events] == ["running"]
        pending = await redis.xpending(queue.QUEUE_STREAM, config.consumer_group)
        assert pending["pending"] == 1
        assert await queue.get_active(redis, tenant) == 1
        assert await redis.get(f"{idempotency.INFLIGHT_KEY_PREFIX}{sha}") == ingest_id
        assert await idempotency.get_done_ingest_id(redis, sha) is None
        assert await redis.xrange(reaper.DEAD_STREAM) == []

        processed: list[str] = []

        async def _complete(j):
            processed.append(j.ingest_id)
            return {"status": "success", "video_id": "v-note-fault", "results": {}}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=0, processor=_complete
        )
        assert recovered == 1
        assert processed == [ingest_id]
        assert await idempotency.get_done_ingest_id(redis, sha) == ingest_id
        assert await redis.get(marker_key) is None
        assert await redis.hgetall(redrive_key) == {}

    @pytest.mark.asyncio
    async def test_concurrent_reapers_redrive_each_graph_pending_entry_once(
        self, redis
    ):
        """Two sweeps over two graph-pending orphans: each entry is re-driven
        by exactly one caller and its record counts exactly one re-drive."""
        config_a = worker.WorkerConfig()
        config_a.consumer_id = "live-graph-a"
        config_b = worker.WorkerConfig()
        config_b.consumer_id = "live-graph-b"
        tenant = "acme:graph"
        jobs = []
        for i in (1, 2):
            job = await _orphan_job(
                redis,
                config_a,
                ingest_id=f"ing_gc{i}",
                sha=f"sha_gc{i}",
                tenant=tenant,
            )
            await worker._mark_graph_pending(redis, job)
            jobs.append(job)
        await asyncio.sleep(0.3)

        processed: list[str] = []
        lock = asyncio.Lock()

        async def _graph_fails(j):
            async with lock:
                processed.append(j.ingest_id)
            await asyncio.sleep(0.02)
            raise worker.GraphStageIncomplete(
                f"graph extraction failed for ingest {j.ingest_id}"
            )

        recovered = await asyncio.gather(
            reaper.run_reaper_once(
                redis, config_a, min_idle_ms=150, processor=_graph_fails, count=1
            ),
            reaper.run_reaper_once(
                redis, config_b, min_idle_ms=150, processor=_graph_fails, count=1
            ),
        )

        assert sum(recovered) == 2
        assert sorted(processed) == ["ing_gc1", "ing_gc2"], (
            f"graph-pending entries processed {processed} — one ran twice or was lost"
        )
        for job in jobs:
            state = await redis.hgetall(
                f"{worker.GRAPH_REDRIVE_KEY_PREFIX}{job.message_id}"
            )
            assert state["redrives"] == "1"
            assert state["cause"] == (
                f"graph extraction failed for ingest {job.ingest_id}"
            )
        pending = await redis.xpending(queue.QUEUE_STREAM, config_a.consumer_group)
        assert pending["pending"] == 2
        assert await redis.xrange(reaper.DEAD_STREAM) == []
        assert await queue.get_active(redis, tenant) == 2

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
        # run()'s pipeline-cache fail-fast reads config from the backend; this
        # stack has only Redis, and that gate is pinned by the entrypoint
        # bootstrap tests. The subject here is the reaper wiring.
        monkeypatch.setattr(worker, "_validate_pipeline_cache_defaults", lambda: None)

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


async def _consumer_names(redis, group: str) -> dict:
    return {
        entry["name"]: entry
        for entry in await redis.xinfo_consumers(queue.QUEUE_STREAM, group)
    }


async def _dead_pods(redis, group: str, count: int) -> list[str]:
    """Consumer names that issued one claim and vanished, one per pod
    incarnation: the group remembers every one of them forever."""
    names = [f"cogniverse-ingestor-{i:02d}" for i in range(count)]
    for name in names:
        await queue.claim(redis, group, name, block_ms=1)
    return names


class TestConsumerGroupHygiene:
    @pytest.mark.asyncio
    async def test_prune_removes_idle_consumers_that_own_nothing(self, redis):
        config = worker.WorkerConfig()
        config.consumer_id = "live-self"
        group = config.consumer_group
        await queue.ensure_consumer_group(redis, group)
        dead = await _dead_pods(redis, group, 65)
        await _orphan_job(
            redis, config, ingest_id="ing_held", sha="sha_held", tenant="acme"
        )
        await queue.claim(redis, group, config.consumer_id, block_ms=1)
        assert len(await _consumer_names(redis, group)) == 67
        await asyncio.sleep(0.3)

        removed = await queue.prune_consumers(
            redis, group, keep=config.consumer_id, min_idle_ms=200
        )

        assert sorted(removed) == dead
        remaining = await _consumer_names(redis, group)
        assert set(remaining) == {"live-self", "dead-1"}
        assert remaining["dead-1"]["pending"] == 1
        pending = await redis.xpending_range(queue.QUEUE_STREAM, group, "-", "+", 10)
        assert [(entry["consumer"], entry["times_delivered"]) for entry in pending] == [
            ("dead-1", 1)
        ]

    @pytest.mark.asyncio
    async def test_prune_leaves_recently_seen_consumers_alone(self, redis):
        config = worker.WorkerConfig()
        config.consumer_id = "live-self"
        group = config.consumer_group
        await queue.ensure_consumer_group(redis, group)
        await _dead_pods(redis, group, 3)
        await queue.claim(redis, group, config.consumer_id, block_ms=1)

        removed = await queue.prune_consumers(
            redis, group, keep=config.consumer_id, min_idle_ms=60_000
        )

        assert removed == []
        assert set(await _consumer_names(redis, group)) == {
            "cogniverse-ingestor-00",
            "cogniverse-ingestor-01",
            "cogniverse-ingestor-02",
            "live-self",
        }

    @pytest.mark.asyncio
    async def test_sweep_reclaims_then_forgets_the_dead_owner(self, redis):
        """One sweep: the dead owner's entry is re-driven onto the live
        consumer, and once it owns nothing the dead name is dropped from the
        group along with every other idle empty consumer."""
        config = worker.WorkerConfig()
        config.consumer_id = "live-self"
        group = config.consumer_group
        await queue.ensure_consumer_group(redis, group)
        await _dead_pods(redis, group, 65)
        await _orphan_job(
            redis, config, ingest_id="ing_dead", sha="sha_dead", tenant="acme"
        )
        await asyncio.sleep(0.3)
        processed: list = []

        async def _proc(job):
            processed.append(job.ingest_id)
            return {"status": "success", "video_id": "v1", "results": {}}

        recovered = await reaper.run_reaper_once(
            redis, config, min_idle_ms=200, processor=_proc
        )

        assert (recovered, processed) == (1, ["ing_dead"])
        assert set(await _consumer_names(redis, group)) == {"live-self"}
        assert await redis.xpending(queue.QUEUE_STREAM, group) == {
            "pending": 0,
            "min": None,
            "max": None,
            "consumers": [],
        }

    @pytest.mark.asyncio
    async def test_concurrent_prunes_remove_each_dead_consumer_exactly_once(
        self, redis
    ):
        config = worker.WorkerConfig()
        group = config.consumer_group
        await queue.ensure_consumer_group(redis, group)
        dead = await _dead_pods(redis, group, 65)
        await asyncio.sleep(0.3)
        barrier = asyncio.Barrier(2)

        async def prune(keep: str) -> list[str]:
            # Each replica's own name was seen just now: idle well under the
            # threshold, so neither replica drops the other.
            await queue.claim(redis, group, keep, block_ms=1)
            await barrier.wait()
            return await queue.prune_consumers(redis, group, keep=keep, min_idle_ms=200)

        removed_a, removed_b = await asyncio.gather(
            prune("reaper-a"), prune("reaper-b")
        )

        assert sorted(removed_a + removed_b) == dead
        assert set(await _consumer_names(redis, group)) == {"reaper-a", "reaper-b"}

    @pytest.mark.asyncio
    async def test_prune_against_a_dead_redis_raises(self):
        import redis.asyncio as aioredis
        from redis.exceptions import ConnectionError as RedisConnectionError

        client = aioredis.from_url(
            "redis://127.0.0.1:29071/0", socket_connect_timeout=1
        )
        try:
            with pytest.raises(RedisConnectionError):
                await queue.prune_consumers(
                    client, "ingestors", keep="live-self", min_idle_ms=0
                )
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_autoclaim_cost_is_independent_of_dead_consumer_count(
        self, redis, capsys
    ):
        """XAUTOCLAIM scans the PEL, not the consumer list; measured here so the
        pruning is justified by group hygiene, not by a claim-latency claim."""
        config = worker.WorkerConfig()
        group = config.consumer_group
        await queue.ensure_consumer_group(redis, group)
        await _dead_pods(redis, group, 65)
        await queue.claim(redis, group, "live-self", block_ms=1)

        async def measure(iterations: int) -> float:
            started = time.perf_counter()
            for _ in range(iterations):
                await queue.autoclaim(
                    redis, group, "live-self", min_idle_ms=60_000, count=10
                )
            return (time.perf_counter() - started) * 1000 / iterations

        with_dead = await measure(200)
        await asyncio.sleep(0.3)
        removed = await queue.prune_consumers(
            redis, group, keep="live-self", min_idle_ms=200
        )
        after = await measure(200)
        print(
            f"xautoclaim per call: {with_dead:.3f}ms with 66 consumers, "
            f"{after:.3f}ms with 1 (pruned {len(removed)})"
        )
        assert len(removed) == 65
        assert set(await _consumer_names(redis, group)) == {"live-self"}
