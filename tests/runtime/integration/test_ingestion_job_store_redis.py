"""``/ingestion/start`` job status shared through a real Redis.

A second ``IngestionJobStore`` on its own client stands in for another
runtime process: it shares nothing with the job's owner but Redis.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import MagicMock

import pytest
from redis.asyncio import Redis

from cogniverse_runtime.ingestion_jobs import (
    ABANDONED_ERROR,
    IngestionJobStore,
    IngestionJobStoreUnavailableError,
)
from cogniverse_runtime.shared_state import connect_shared_state_redis

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
    pytest.mark.asyncio,
]


def _started(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "status": "started",
        "videos_processed": 0,
        "videos_total": 0,
        "errors": [],
    }


@pytest.fixture
def prefix():
    return f"test:ingestion-job:{uuid.uuid4().hex}"


@pytest.fixture
def owner(shared_state_redis, prefix):
    return IngestionJobStore(shared_state_redis, owner="owner", key_prefix=prefix)


@pytest.fixture
async def peer(shared_state_redis_url, prefix):
    client = await connect_shared_state_redis(shared_state_redis_url)
    yield IngestionJobStore(client, owner="peer", key_prefix=prefix)
    await client.aclose()


class TestJobRecord:
    async def test_a_started_job_reads_the_same_on_another_process(self, owner, peer):
        assert await owner.create("job-1") == _started("job-1")

        assert await peer.get("job-1") == _started("job-1")
        assert await peer.get("job-unknown") is None

    async def test_progress_and_outcome_round_trip(self, owner, peer):
        await owner.create("job-1")

        assert await owner.update("job-1", videos_total=3, status="processing")
        assert await peer.get("job-1") == {
            **_started("job-1"),
            "status": "processing",
            "videos_total": 3,
        }
        assert await owner.finish(
            "job-1",
            status="completed_with_errors",
            videos_processed=2,
            errors=["/videos/c.mp4: decode failed [no stream]"],
        )

        assert await peer.get("job-1") == {
            "job_id": "job-1",
            "status": "completed_with_errors",
            "videos_processed": 2,
            "videos_total": 3,
            "errors": ["/videos/c.mp4: decode failed [no stream]"],
        }

    async def test_a_finished_job_is_final(self, owner, peer):
        await owner.create("job-1")
        await owner.finish("job-1", status="failed", errors=["boom"])

        assert await owner.update("job-1", status="processing") is False
        assert await owner.finish("job-1", status="completed", errors=[]) is False
        assert await owner.renew("job-1") is False
        assert await peer.get("job-1") == {
            **_started("job-1"),
            "status": "failed",
            "errors": ["boom"],
        }

    async def test_a_running_status_is_not_an_outcome(self, owner):
        await owner.create("job-1")

        with pytest.raises(ValueError) as refused:
            await owner.finish("job-1", status="processing", errors=[])

        assert str(refused.value) == "'processing' is not a finished job status"

    async def test_records_expire_after_retention(self, shared_state_redis, prefix):
        store = IngestionJobStore(
            shared_state_redis, owner="owner", key_prefix=prefix, retention_seconds=60
        )
        await store.create("job-1")
        await store.finish("job-1", status="completed", errors=[])

        ttl_ms = await shared_state_redis.pttl(f"{prefix}:job-1")
        lease_left = await shared_state_redis.exists(f"{prefix}:job-1:lease")

        assert 59_000 < ttl_ms <= 60_000
        assert lease_left == 0


class TestOwnerLease:
    async def test_a_job_whose_owner_stopped_reads_as_failed_once(
        self, shared_state_redis, peer, prefix
    ):
        owner = IngestionJobStore(
            shared_state_redis,
            owner="owner",
            key_prefix=prefix,
            lease_seconds=0.5,
            heartbeat_seconds=0.1,
        )
        await owner.create("job-1")
        await owner.update("job-1", status="processing", videos_total=4)
        await asyncio.sleep(0.6)

        abandoned = {
            **_started("job-1"),
            "status": "failed",
            "videos_total": 4,
            "errors": [ABANDONED_ERROR],
        }
        assert await peer.get("job-1") == abandoned
        assert await peer.get("job-1") == abandoned
        assert await owner.finish("job-1", status="completed", errors=[]) is False
        assert await owner.get("job-1") == abandoned

    async def test_the_lease_holds_while_the_owner_runs(
        self, shared_state_redis, peer, prefix
    ):
        owner = IngestionJobStore(
            shared_state_redis,
            owner="owner",
            key_prefix=prefix,
            lease_seconds=0.5,
            heartbeat_seconds=0.1,
        )
        await owner.create("job-1")

        async with owner.lease("job-1"):
            await asyncio.sleep(1.5)
            during = await peer.get("job-1")
        await owner.finish("job-1", status="completed", videos_processed=1, errors=[])

        assert during == _started("job-1")
        assert await peer.get("job-1") == {
            **_started("job-1"),
            "status": "completed",
            "videos_processed": 1,
        }

    async def test_a_running_job_outlives_retention_while_its_lease_holds(
        self, shared_state_redis, peer, prefix
    ):
        owner = IngestionJobStore(
            shared_state_redis,
            owner="owner",
            key_prefix=prefix,
            retention_seconds=1,
            lease_seconds=0.5,
            heartbeat_seconds=0.1,
        )
        await owner.create("job-1")

        async with owner.lease("job-1"):
            await asyncio.sleep(1.5)
            during = await peer.get("job-1")

        assert during == _started("job-1")

    async def test_concurrent_readers_mark_an_abandoned_job_once(
        self, shared_state_redis_url, shared_state_redis, prefix
    ):
        owner = IngestionJobStore(
            shared_state_redis,
            owner="owner",
            key_prefix=prefix,
            lease_seconds=0.2,
            heartbeat_seconds=0.1,
        )
        await owner.create("job-1")
        await asyncio.sleep(0.3)
        clients = [
            await connect_shared_state_redis(shared_state_redis_url) for _ in range(16)
        ]
        readers = [
            IngestionJobStore(client, owner=f"reader-{i}", key_prefix=prefix)
            for i, client in enumerate(clients)
        ]
        barrier = asyncio.Barrier(len(readers))

        async def read(store):
            await barrier.wait()
            return await store.get("job-1")

        try:
            records = await asyncio.gather(*(read(r) for r in readers))
        finally:
            for client in clients:
                await client.aclose()

        assert (
            records
            == [{**_started("job-1"), "status": "failed", "errors": [ABANDONED_ERROR]}]
            * 16
        )


class TestRedisFailures:
    async def test_every_operation_raises_when_redis_is_unreachable(
        self, dead_redis_url, prefix
    ):
        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        store = IngestionJobStore(client, owner="owner", key_prefix=prefix)
        operations = {
            "create job job-1": lambda: store.create("job-1"),
            "read job job-1": lambda: store.get("job-1"),
            "update job job-1": lambda: store.update("job-1", status="processing"),
            "renew job job-1": lambda: store.renew("job-1"),
        }
        try:
            for operation, call in operations.items():
                with pytest.raises(IngestionJobStoreUnavailableError) as failed:
                    await call()
                assert str(failed.value) == (
                    f"ingestion job store unavailable: {operation}"
                )
        finally:
            await client.aclose()

    async def test_a_hung_redis_fails_within_the_command_timeout(
        self, own_redis, prefix
    ):
        url, pause, _ = own_redis
        client = await connect_shared_state_redis(url, timeout_seconds=1.5)
        store = IngestionJobStore(client, owner="owner", key_prefix=prefix)
        await store.create("job-1")
        pause()
        started = time.monotonic()
        try:
            with pytest.raises(IngestionJobStoreUnavailableError) as failed:
                await store.get("job-1")
            elapsed = time.monotonic() - started
        finally:
            await client.aclose()

        assert str(failed.value) == "ingestion job store unavailable: read job job-1"
        assert 1.5 <= elapsed < 4.0, elapsed


class TestOutcomeWrite:
    async def test_an_outcome_redis_refuses_lands_on_a_later_attempt(
        self, own_redis, prefix, tmp_path, monkeypatch
    ):
        """Redis refuses every write while the job finishes (out of memory);
        the outcome write is retried and lands once writes are accepted."""
        from cogniverse_runtime.routers import ingestion as ing

        url, _, _ = own_redis
        client = await connect_shared_state_redis(url)
        operator = await connect_shared_state_redis(url)
        store = IngestionJobStore(client, owner="owner", key_prefix=prefix)
        await store.create("job-1")
        accepting = asyncio.Event()

        async def accept_writes_again():
            await asyncio.sleep(2.0)
            await operator.config_set("maxmemory", "0")
            accepting.set()

        class _Pipeline:
            def __init__(self, **kwargs):
                pass

            async def process_videos_concurrent(self, video_files, max_concurrent):
                await operator.config_set("maxmemory", "1")
                self.reopen = asyncio.create_task(accept_writes_again())
                return {"status": "completed", "successful": 1, "results": []}

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", _Pipeline
        )
        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.strategies.discover_ingestible_files",
            lambda video_dir, content_type: ["a.mp4"],
        )
        request = ing.IngestionRequest(
            video_dir=str(tmp_path),
            profile="video_colpali_smol500_mv_frame",
            tenant_id="acme:acme",
            content_type="video",
        )
        try:
            started = asyncio.get_running_loop().time()
            await ing.run_ingestion(
                "job-1",
                request,
                config_manager=MagicMock(),
                schema_loader=MagicMock(),
                job_store=store,
            )
            elapsed = asyncio.get_running_loop().time() - started
            record = await store.get("job-1")
        finally:
            await operator.config_set("maxmemory", "0")
            await client.aclose()
            await operator.aclose()

        assert accepting.is_set()
        # Refused at once and after one second; accepted after three.
        assert 3.0 <= elapsed < 5.0, elapsed
        assert record == {
            "job_id": "job-1",
            "status": "completed",
            "videos_processed": 1,
            "videos_total": 1,
            "errors": [],
        }
