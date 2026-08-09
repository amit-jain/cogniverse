"""Renewable approval dataset locking against an isolated Redis process."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.approval.replacement_store import RedisReplacementRecordStore
from cogniverse_core.approval.interfaces import ReviewDecision, ReviewItem

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.no_shared_memory_vespa,
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def redis_process():
    port = _free_port()
    container_name = f"cogniverse-approval-lock-{os.getpid()}-{port}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
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

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", container_name, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        pytest.fail("Redis did not become ready within 30 seconds")

    try:
        yield SimpleNamespace(
            url=f"redis://127.0.0.1:{port}/0",
            container_name=container_name,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def _storage(redis_url: str, tenant_id: str) -> ApprovalStorageImpl:
    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = tenant_id
    storage.redis_url = redis_url
    storage._replacement_records = RedisReplacementRecordStore(redis_url)
    storage._DATASET_LOCK_LEASE_MS = 120
    storage._DATASET_LOCK_WAIT_SECONDS = 3.0
    storage._DATASET_LOCK_POLL_SECONDS = 0.01
    storage._DATASET_LOCK_SOCKET_TIMEOUT_SECONDS = 0.25
    return storage


async def test_wrong_owner_cannot_renew_or_release_lock(redis_process):
    storage = _storage(redis_process.url, "acme:ownership")
    lock_key = (
        "cogniverse:approval:dataset-lock:acme:ownership:"
        "approved_synthetic_data-acme:ownership"
    )
    redis = aioredis.from_url(
        redis_process.url,
        decode_responses=True,
        socket_connect_timeout=0.25,
        socket_timeout=0.25,
        retry_on_timeout=False,
    )
    try:
        assert await redis.set(lock_key, "owner-token", nx=True, px=2_000) is True

        wrong_renewal = await redis.eval(
            storage._DATASET_LOCK_RENEW_SCRIPT,
            1,
            lock_key,
            "wrong-token",
            5_000,
        )
        wrong_release = await redis.eval(
            storage._DATASET_LOCK_RELEASE_SCRIPT,
            1,
            lock_key,
            "wrong-token",
        )

        assert (wrong_renewal, wrong_release) == (0, 0)
        assert await redis.get(lock_key) == "owner-token"
        assert 0 < await redis.pttl(lock_key) <= 2_000
        assert (
            await redis.eval(
                storage._DATASET_LOCK_RENEW_SCRIPT,
                1,
                lock_key,
                "owner-token",
                5_000,
            )
            == 1
        )
        assert 4_000 < await redis.pttl(lock_key) <= 5_000
        assert (
            await redis.eval(
                storage._DATASET_LOCK_RELEASE_SCRIPT,
                1,
                lock_key,
                "owner-token",
            )
            == 1
        )
        assert await redis.exists(lock_key) == 0
    finally:
        await redis.aclose()


async def test_independent_storages_never_overlap_across_multiple_leases(
    redis_process,
):
    tenant_id = "acme:exclusive"
    dataset_name = "approved_synthetic_data-acme:exclusive"
    first = _storage(redis_process.url, tenant_id)
    second = _storage(redis_process.url, tenant_id)
    first_entered = asyncio.Event()
    events = []
    active = 0
    maximum_active = 0

    async def hold(storage, name, duration):
        nonlocal active, maximum_active
        async with storage._approval_dataset_lock(dataset_name):
            active += 1
            maximum_active = max(maximum_active, active)
            events.append(f"{name}_entered")
            if name == "first":
                first_entered.set()
            await asyncio.sleep(duration)
            events.append(f"{name}_leaving")
            active -= 1

    first_task = asyncio.create_task(hold(first, "first", 0.65))
    await asyncio.wait_for(first_entered.wait(), timeout=2)
    second_task = asyncio.create_task(hold(second, "second", 0.05))
    await asyncio.gather(first_task, second_task)

    assert maximum_active == 1
    assert events == [
        "first_entered",
        "first_leaving",
        "second_entered",
        "second_leaving",
    ]


async def test_redis_termination_aborts_approval_before_dataset_write_or_status(
    redis_process,
):
    from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

    tenant_id = "acme:redis-loss"
    dataset_name = "approved_synthetic_data-acme:redis-loss"
    batch_id = "approval-batch"
    item = ReviewItem(
        item_id="approval-item",
        data={
            "query": "find Marie Curie laboratory footage",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.875,
        metadata={"agent_type": "routing"},
    )
    decision = ReviewDecision(
        item_id=item.item_id,
        approved=True,
        feedback="The routing decision is exact.",
        reviewer="reviewer@example.com",
        timestamp=datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc),
    )
    dataset_read_started = asyncio.Event()
    events = []

    class Datasets:
        async def get_dataset(self, name):
            assert name == dataset_name
            events.append("dataset_read_started")
            dataset_read_started.set()
            await asyncio.Event().wait()
            events.append("dataset_read_continued")
            raise DatasetNotFoundError(name)

        async def create_dataset(self, name, data):
            events.append(("dataset_created", name, len(data)))

        async def append_to_dataset(self, name, data):
            events.append(("dataset_appended", name, len(data)))

    storage = _storage(redis_process.url, tenant_id)
    storage.provider = SimpleNamespace(datasets=Datasets())

    async def get_item_span_id(_item_id, batch_id=None):
        events.append(("span_lookup", batch_id))
        return "approval-span"

    async def log_approval_decision(**_kwargs):
        events.append("decision_annotation")

    async def update_item(_item, batch_id=None):
        events.append(("status_annotation", batch_id))

    storage.get_item_span_id = get_item_span_id
    storage.log_approval_decision = log_approval_decision
    storage.update_item = update_item

    approval = asyncio.create_task(
        storage.persist_approved_item(
            batch_id=batch_id,
            dataset_name=dataset_name,
            item=item,
            decision=decision,
            project_context={"tenant_id": tenant_id, "optimizer": "routing"},
        )
    )
    await asyncio.wait_for(dataset_read_started.wait(), timeout=2)
    stopped = await asyncio.to_thread(
        subprocess.run,
        ["docker", "stop", redis_process.container_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert stopped.returncode == 0, stopped.stderr

    with pytest.raises(RuntimeError) as error:
        await asyncio.wait_for(approval, timeout=5)

    assert str(error.value) == (
        "Failed to persist approved item: "
        f"tenant={tenant_id} dataset={dataset_name} "
        f"batch={batch_id} item={item.item_id}"
    )
    renewal_error = error.value.__cause__
    assert isinstance(renewal_error, RuntimeError)
    assert str(renewal_error) == (
        "Failed to renew approved dataset lock: "
        f"tenant={tenant_id} dataset={dataset_name}"
    )
    assert isinstance(renewal_error.__cause__, RedisError)
    assert str(renewal_error.__cause__)
    assert events == ["dataset_read_started"]
