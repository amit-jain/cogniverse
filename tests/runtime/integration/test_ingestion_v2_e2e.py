"""End-to-end test for the ingestion_v2 path.

Real Redis 7.4 container, real ``status_api`` SSE + snapshot routes
(via httpx.ASGITransport), real worker async-task in the same process
calling ``enqueue_ingestion`` to exercise the full submit → claim →
ack chain. The pipeline itself is stubbed via the injectable
``processor`` argument since the existing pipeline integration tests
cover that side; this file is about the queue + worker + status-stream
contract.

Catches the bug where any one of {idempotency, backpressure, queue,
status stream, SSE replay, terminal cleanup} drifts away from what
the others expect — none reachable by mocking a single layer.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import subprocess
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_runtime.ingestion_v2 import queue
from cogniverse_runtime.ingestion_v2 import status_api as ingest_status
from cogniverse_runtime.ingestion_v2.queue import IngestJob
from cogniverse_runtime.ingestion_v2.redis_client import close_redis, get_redis
from cogniverse_runtime.ingestion_v2.submit_api import (
    BackpressureError,
    enqueue_ingestion,
)
from cogniverse_runtime.ingestion_v2.worker import WorkerConfig, _claim_loop

CONTAINER_NAME = "redis-ingestion-v2-e2e"


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
async def env_redis(redis_container, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_container)
    monkeypatch.setenv("INGEST_QUEUE_DEPTH_LIMIT", "100")
    monkeypatch.setenv("INGEST_PER_TENANT_CONCURRENCY", "5")
    monkeypatch.setenv("INGEST_IDEMPOTENCY_TTL_SECONDS", "60")
    await close_redis()
    redis = await get_redis(redis_container)
    await redis.flushdb()
    yield redis
    await close_redis()


@pytest.fixture
def app(env_redis):
    """FastAPI app with the status_api router mounted at the production
    prefix (``/ingestion``) so URLs match what main.py wires."""
    application = FastAPI()
    application.include_router(ingest_status.router, prefix="/ingestion")
    return application


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _stub_processor(job: IngestJob) -> dict:
    """Fake pipeline that just succeeds with a small result. Adds a
    short await so the queue claim path actually has to wait for the
    coroutine to finish (regression guard for cleanup ordering)."""
    await asyncio.sleep(0.05)
    return {
        "video_id": job.ingest_id,
        "schema_name": job.profile,
        "tenant_id": job.tenant_id,
        "results": {"keyframes": {"keyframes": [{"id": 0}, {"id": 1}]}},
    }


async def _failing_processor(job: IngestJob) -> dict:
    raise RuntimeError(f"intentional failure for {job.ingest_id}")


async def _spawn_worker(
    redis_url: str, processor
) -> tuple[asyncio.Task, asyncio.Event]:
    """Start the claim loop as a background task. Returns the task +
    the stop-event the test should set in cleanup."""
    stop = asyncio.Event()
    config = WorkerConfig()
    config.claim_block_ms = 200  # short block so shutdown is snappy
    redis = await get_redis(redis_url)
    task = asyncio.create_task(_claim_loop(redis, config, stop, processor=processor))
    return task, stop


async def _wait_for_state(
    redis, ingest_id: str, target_state: str, timeout_s: float = 5.0
) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        events = await queue.read_status_since(redis, ingest_id)
        for _, ev in events:
            if ev.get("state") == target_state:
                return ev
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Did not see state={target_state!r} for {ingest_id} within {timeout_s}s; "
        f"events: {[ev for _, ev in events]}"
    )


class TestEnqueueAndWorker:
    """Full path: enqueue_ingestion → queue → worker → status events."""

    @pytest.mark.asyncio
    async def test_submit_runs_through_worker_to_complete(
        self, env_redis, redis_container
    ):
        worker_task, stop = await _spawn_worker(redis_container, _stub_processor)
        try:
            result = await enqueue_ingestion(
                env_redis,
                source_url="s3://bucket/fake.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
            )
            assert result.state == "queued"
            assert result.existing is False

            final = await _wait_for_state(env_redis, result.ingest_id, "complete")
            assert final["result"]["video_id"] == result.ingest_id
            assert final["result"]["keyframes"] == 2

            done = await env_redis.get(f"ingest:done:{result.sha}")
            assert done == result.ingest_id
            inflight = await env_redis.get(f"ingest:by_sha:{result.sha}")
            assert inflight is None
            assert await queue.get_active(env_redis, "acme") == 0
        finally:
            stop.set()
            await asyncio.wait_for(worker_task, timeout=2)

    @pytest.mark.asyncio
    async def test_resubmit_returns_existing_ingest_id(
        self, env_redis, redis_container
    ):
        worker_task, stop = await _spawn_worker(redis_container, _stub_processor)
        try:
            kwargs = dict(
                source_url="s3://bucket/idem.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
            )
            first = await enqueue_ingestion(env_redis, **kwargs)
            await _wait_for_state(env_redis, first.ingest_id, "complete")

            second = await enqueue_ingestion(env_redis, **kwargs)
            assert second.existing is True
            assert second.ingest_id == first.ingest_id
            assert second.sha == first.sha

            assert await queue.queue_depth(env_redis) == 1
        finally:
            stop.set()
            await asyncio.wait_for(worker_task, timeout=2)

    @pytest.mark.asyncio
    async def test_force_bypasses_idempotency(self, env_redis, redis_container):
        worker_task, stop = await _spawn_worker(redis_container, _stub_processor)
        try:
            kwargs = dict(
                source_url="s3://bucket/force.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
            )
            first = await enqueue_ingestion(env_redis, **kwargs)
            await _wait_for_state(env_redis, first.ingest_id, "complete")

            second = await enqueue_ingestion(env_redis, **kwargs, force=True)
            assert second.existing is False
            assert second.ingest_id != first.ingest_id
        finally:
            stop.set()
            await asyncio.wait_for(worker_task, timeout=2)

    @pytest.mark.asyncio
    async def test_failing_processor_publishes_failed_and_acks(
        self, env_redis, redis_container
    ):
        worker_task, stop = await _spawn_worker(redis_container, _failing_processor)
        try:
            result = await enqueue_ingestion(
                env_redis,
                source_url="s3://bucket/bad.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
            )
            final = await _wait_for_state(env_redis, result.ingest_id, "failed")
            assert "intentional failure" in final["error"]
            assert final["error_type"] == "RuntimeError"

            assert await env_redis.get(f"ingest:by_sha:{result.sha}") is None
            assert await queue.get_active(env_redis, "acme") == 0
            pending = await env_redis.xpending(queue.QUEUE_STREAM, "ingestors")
            assert pending["pending"] == 0
            # Failed jobs do NOT mark done — re-submit (without force)
            # should re-enqueue, not return existing.
            assert await env_redis.get(f"ingest:done:{result.sha}") is None
        finally:
            stop.set()
            await asyncio.wait_for(worker_task, timeout=2)

    @pytest.mark.asyncio
    async def test_wait_returns_terminal_event_synchronously(
        self, env_redis, redis_container
    ):
        worker_task, stop = await _spawn_worker(redis_container, _stub_processor)
        try:
            result = await enqueue_ingestion(
                env_redis,
                source_url="s3://bucket/sync.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
                wait=True,
                wait_timeout=10,
            )
            assert result.state == "complete"
            assert result.final_event is not None
            assert result.final_event["state"] == "complete"
            assert result.final_event["result"]["keyframes"] == 2
        finally:
            stop.set()
            await asyncio.wait_for(worker_task, timeout=2)


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_per_tenant_concurrency_raises(self, env_redis, monkeypatch):
        """No worker running — submissions accumulate against the
        per-tenant counter. The 6th submission to ``acme`` (limit=5)
        must raise BackpressureError."""
        monkeypatch.setenv("INGEST_PER_TENANT_CONCURRENCY", "5")

        for i in range(5):
            await enqueue_ingestion(
                env_redis,
                source_url=f"s3://bucket/{i}.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
            )

        with pytest.raises(BackpressureError) as exc:
            await enqueue_ingestion(
                env_redis,
                source_url="s3://bucket/over.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="acme",
            )
        assert exc.value.rejection.axis == "tenant"
        assert exc.value.rejection.current == 5
        assert exc.value.rejection.limit == 5

        # Other tenant unaffected
        other = await enqueue_ingestion(
            env_redis,
            source_url="s3://bucket/other.mp4",
            profile="video_colpali_smol500_mv_frame",
            tenant_id="other",
        )
        assert other.state == "queued"

    @pytest.mark.asyncio
    async def test_cluster_queue_depth_raises(self, env_redis, monkeypatch):
        monkeypatch.setenv("INGEST_QUEUE_DEPTH_LIMIT", "3")
        for i in range(3):
            await enqueue_ingestion(
                env_redis,
                source_url=f"s3://bucket/{i}.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id=f"t{i}",
            )

        with pytest.raises(BackpressureError) as exc:
            await enqueue_ingestion(
                env_redis,
                source_url="s3://bucket/over.mp4",
                profile="video_colpali_smol500_mv_frame",
                tenant_id="t4",
            )
        assert exc.value.rejection.axis == "cluster"


class TestSseEndpoints:
    """The SSE + snapshot endpoints mounted at /ingestion/{id}/..."""

    @pytest.mark.asyncio
    async def test_sse_replays_history_then_streams_to_terminal(
        self, client, env_redis
    ):
        ingest_id = f"ing_{uuid.uuid4().hex[:8]}"
        await queue.publish_status(env_redis, ingest_id, {"state": "queued"})
        await queue.publish_status(env_redis, ingest_id, {"state": "running"})

        async def emit_complete():
            await asyncio.sleep(0.2)
            await queue.publish_status(
                env_redis, ingest_id, {"state": "complete", "result": {"ok": True}}
            )

        emitter = asyncio.create_task(emit_complete())

        async with client.stream(
            "GET",
            f"/ingestion/{ingest_id}/events?timeout_seconds=30",
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            seen_states: list[str] = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    seen_states.append(payload.get("state"))
                    if payload.get("state") == "complete":
                        break

        await emitter
        assert seen_states == ["queued", "running", "complete"]

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_history_snapshot(self, client, env_redis):
        ingest_id = f"ing_{uuid.uuid4().hex[:8]}"
        await queue.publish_status(env_redis, ingest_id, {"state": "queued"})
        await queue.publish_status(
            env_redis, ingest_id, {"state": "complete", "result": {"x": 1}}
        )

        r = await client.get(f"/ingestion/{ingest_id}/status")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "complete"
        assert body["events_count"] == 2
        assert body["latest"]["result"]["x"] == 1
        assert [e["state"] for e in body["history"]] == ["queued", "complete"]

    @pytest.mark.asyncio
    async def test_status_endpoint_404_for_unknown_ingest(self, client):
        r = await client.get(f"/ingestion/never_existed_{os.getpid()}/status")
        assert r.status_code == 404
