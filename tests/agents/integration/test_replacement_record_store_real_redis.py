"""Canonical approval replacements shared through a real Redis process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.approval.replacement_store import RedisReplacementRecordStore
from cogniverse_core.approval.interfaces import (
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.no_shared_memory_vespa,
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _redis_container_name() -> str:
    return f"cogniverse-replacement-record-{os.getpid()}"


@pytest.fixture(scope="module")
def redis_url():
    port = _free_port()
    container_name = _redis_container_name()
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
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def _store(redis_url: str):
    return RedisReplacementRecordStore(redis_url)


def _candidate(index: int) -> dict:
    return {
        "item_id": "routing_17_regen_0",
        "data": {
            "query": f"find exact candidate {index}",
            "entities": [{"text": f"candidate-{index}", "type": "TOPIC"}],
        },
        "confidence": 0.61 + index / 1000,
        "status": "regenerated",
        "metadata": {
            "original_item_id": "routing_17",
            "decision": {"reviewer": f"reviewer-{index}"},
        },
        "created_at": f"2026-08-03T01:02:{index:02d}+00:00",
        "reviewed_at": None,
    }


def _decision_candidate(timestamp: str, **changes) -> dict:
    candidate = {
        "item_id": "routing_17",
        "approved": True,
        "reviewer": "reviewer@example.com",
        "feedback": "The route is exact.",
        "corrections": {"query": "find the exact launch recording"},
        "timestamp": timestamp,
    }
    candidate.update(changes)
    return candidate


def _approval_batch_candidate(batch_timestamp: str, item_timestamp: str) -> dict:
    return {
        "batch_id": "synthetic_routing_17",
        "context": {
            "tenant_id": "acme:batch-selection",
            "agent_type": "routing",
        },
        "created_at": batch_timestamp,
        "items": [
            {
                "item_id": "synthetic_routing_item_17",
                "data": {
                    "query": "find the exact launch recording",
                    "chosen_agent": "video_search_agent",
                },
                "confidence": 0.93,
                "status": "auto_approved",
                "metadata": {"agent_type": "routing"},
                "created_at": item_timestamp,
                "reviewed_at": None,
            }
        ],
    }


async def test_concurrent_replicas_return_one_exact_persisted_record(redis_url):
    tenant_id = "acme:production"
    batch_id = "batch:quotes/'/spaces"
    original_item_id = "routing_17"
    candidates = [_candidate(index) for index in range(12)]
    start = asyncio.Barrier(len(candidates))

    async def select(index: int):
        await start.wait()
        return await _store(redis_url).select_canonical(
            tenant_id=tenant_id,
            batch_id=batch_id,
            original_item_id=original_item_id,
            candidate=candidates[index],
        )

    selected = await asyncio.gather(*(select(index) for index in range(12)))

    assert selected == [selected[0]] * 12
    assert selected[0].payload in candidates
    assert (
        selected[0].sha256
        == hashlib.sha256(selected[0].json.encode("utf-8")).hexdigest()
    )

    redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        keys = await redis.keys("cogniverse:approval:replacement:*")
        assert len(keys) == 1
        assert await redis.get(keys[0]) == selected[0].json
        assert json.loads(await redis.get(keys[0])) == selected[0].payload
        assert await redis.ttl(keys[0]) == -1
    finally:
        await redis.aclose()


async def test_new_store_instance_replays_the_original_record(redis_url):
    first = _candidate(21)
    second = _candidate(22)

    selected_first = await _store(redis_url).select_canonical(
        tenant_id="acme:replay",
        batch_id="batch-replay",
        original_item_id="routing_17",
        candidate=first,
    )
    selected_after_restart = await _store(redis_url).select_canonical(
        tenant_id="acme:replay",
        batch_id="batch-replay",
        original_item_id="routing_17",
        candidate=second,
    )

    assert selected_first.payload == first
    assert selected_after_restart == selected_first


async def test_selected_record_has_exact_canonical_bytes_and_digest(redis_url):
    selected = await _store(redis_url).select_canonical(
        tenant_id="acme:canonical",
        batch_id="batch-canonical",
        original_item_id="routing_17",
        candidate=_candidate(7),
    )

    expected_json = (
        '{"confidence":0.617,"created_at":"2026-08-03T01:02:07+00:00",'
        '"data":{"entities":[{"text":"candidate-7","type":"TOPIC"}],'
        '"query":"find exact candidate 7"},"item_id":"routing_17_regen_0",'
        '"metadata":{"decision":{"reviewer":"reviewer-7"},'
        '"original_item_id":"routing_17"},"reviewed_at":null,'
        '"status":"regenerated"}'
    )
    assert selected.json == expected_json
    assert (
        selected.sha256
        == "c552afe67ce0c250d27b25499f63beb835a2ce08a6294a6f3552fc8e12fbda13"
    )


async def test_review_decision_retry_reuses_first_canonical_timestamp(redis_url):
    first_timestamp = "2026-08-05T03:04:05+00:00"
    retry_timestamp = "2026-08-05T03:09:10+00:00"
    store = _store(redis_url)

    first = await store.select_review_decision(
        tenant_id="acme:decision-retry",
        batch_id="batch-decision-retry",
        original_item_id="routing_17",
        candidate=_decision_candidate(first_timestamp),
    )
    retried = await _store(redis_url).select_review_decision(
        tenant_id="acme:decision-retry",
        batch_id="batch-decision-retry",
        original_item_id="routing_17",
        candidate=_decision_candidate(retry_timestamp),
    )

    assert retried == first
    assert retried.payload["timestamp"] == first_timestamp
    assert retried.json == (
        '{"approved":true,"corrections":{"query":"find the exact launch '
        'recording"},"feedback":"The route is exact.","item_id":"routing_17",'
        '"reviewer":"reviewer@example.com","timestamp":'
        '"2026-08-05T03:04:05+00:00"}'
    )

    redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        key = RedisReplacementRecordStore._decision_key(
            "acme:decision-retry", "batch-decision-retry", "routing_17"
        )
        assert await redis.get(key) == first.json
        assert await redis.ttl(key) == -1
    finally:
        await redis.aclose()


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        pytest.param("reviewer", "other@example.com", id="reviewer"),
        pytest.param("approved", False, id="approved"),
        pytest.param("feedback", "Use a different route.", id="feedback"),
        pytest.param(
            "corrections",
            {"query": "find an unrelated recording"},
            id="corrections",
        ),
    ],
)
async def test_review_decision_retry_rejects_changed_intent(
    redis_url, field, changed_value
):
    tenant_id = f"acme:decision-conflict:{field}"
    batch_id = "batch-decision-conflict"
    store = _store(redis_url)
    first = await store.select_review_decision(
        tenant_id=tenant_id,
        batch_id=batch_id,
        original_item_id="routing_17",
        candidate=_decision_candidate("2026-08-05T03:04:05+00:00"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Review decision conflicts with canonical review decision for "
            f"tenant={tenant_id} batch={batch_id} original=routing_17"
        ),
    ):
        await _store(redis_url).select_review_decision(
            tenant_id=tenant_id,
            batch_id=batch_id,
            original_item_id="routing_17",
            candidate=_decision_candidate(
                "2026-08-05T03:09:10+00:00", **{field: changed_value}
            ),
        )

    redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        key = RedisReplacementRecordStore._decision_key(
            tenant_id, batch_id, "routing_17"
        )
        assert await redis.get(key) == first.json
    finally:
        await redis.aclose()


async def test_concurrent_identical_decisions_select_one_first_timestamp(redis_url):
    timestamps = [f"2026-08-05T03:04:{second:02d}+00:00" for second in range(12)]
    start = asyncio.Barrier(len(timestamps))

    async def select(timestamp: str):
        await start.wait()
        return await _store(redis_url).select_review_decision(
            tenant_id="acme:decision-concurrent",
            batch_id="batch-decision-concurrent",
            original_item_id="routing_17",
            candidate=_decision_candidate(timestamp),
        )

    selected = await asyncio.gather(*(select(timestamp) for timestamp in timestamps))

    assert selected == [selected[0]] * len(timestamps)
    assert selected[0].payload["timestamp"] in timestamps


async def test_approval_batch_retry_reuses_first_batch_and_item_timestamps(redis_url):
    first = await _store(redis_url).select_approval_batch(
        tenant_id="acme:batch-selection",
        batch_id="synthetic_routing_17",
        candidate=_approval_batch_candidate(
            "2026-08-05T04:00:00+00:00", "2026-08-05T04:00:01+00:00"
        ),
    )
    retried = await _store(redis_url).select_approval_batch(
        tenant_id="acme:batch-selection",
        batch_id="synthetic_routing_17",
        candidate=_approval_batch_candidate(
            "2026-08-05T04:10:00+00:00", "2026-08-05T04:10:01+00:00"
        ),
    )

    assert retried == first
    assert retried.payload["created_at"] == "2026-08-05T04:00:00+00:00"
    assert retried.payload["items"][0]["created_at"] == ("2026-08-05T04:00:01+00:00")

    conflicting = _approval_batch_candidate(
        "2026-08-05T04:20:00+00:00", "2026-08-05T04:20:01+00:00"
    )
    conflicting["items"][0]["data"]["chosen_agent"] = "summarizer_agent"
    with pytest.raises(
        RuntimeError,
        match=(
            "Approval batch conflicts with canonical approval batch for "
            "tenant=acme:batch-selection batch=synthetic_routing_17"
        ),
    ):
        await _store(redis_url).select_approval_batch(
            tenant_id="acme:batch-selection",
            batch_id="synthetic_routing_17",
            candidate=conflicting,
        )


async def test_concurrent_approval_batch_retries_share_first_timestamps(redis_url):
    timestamps = [f"2026-08-05T05:{minute:02d}:00+00:00" for minute in range(12)]
    start = asyncio.Barrier(len(timestamps))

    async def select(timestamp: str):
        await start.wait()
        return await _store(redis_url).select_approval_batch(
            tenant_id="acme:batch-concurrent",
            batch_id="synthetic_routing_concurrent",
            candidate={
                **_approval_batch_candidate(timestamp, timestamp),
                "batch_id": "synthetic_routing_concurrent",
                "context": {
                    "tenant_id": "acme:batch-concurrent",
                    "agent_type": "routing",
                },
            },
        )

    selected = await asyncio.gather(*(select(timestamp) for timestamp in timestamps))

    assert selected == [selected[0]] * len(timestamps)
    assert selected[0].payload["created_at"] in timestamps
    assert (
        selected[0].payload["items"][0]["created_at"]
        == (selected[0].payload["created_at"])
    )


async def test_dataset_commit_then_annotation_retry_reuses_first_timestamp(redis_url):
    class Datasets:
        def __init__(self):
            self.frame = None
            self.write_count = 0

        async def get_dataset(self, name):
            assert name == dataset_name
            if self.frame is None:
                raise DatasetNotFoundError(name)
            return self.frame.copy(deep=True)

        async def create_dataset(self, name, data):
            assert name == dataset_name
            self.write_count += 1
            self.frame = pd.DataFrame([{"input": data.iloc[0].to_dict()}])
            return "dataset-id"

        async def append_to_dataset(self, name, data):
            raise AssertionError(
                f"retry appended another dataset row to {name}: {data}"
            )

    tenant_id = "acme:dataset-retry"
    batch_id = "batch-dataset-retry"
    dataset_name = approved_synthetic_dataset_name(tenant_id)
    datasets = Datasets()
    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = tenant_id
    storage.redis_url = redis_url
    storage.full_project_name = f"cogniverse-{tenant_id}-synthetic_data"
    storage.provider = SimpleNamespace(datasets=datasets)
    storage._replacement_records = _store(redis_url)
    item = ReviewItem(
        item_id="routing_17",
        data={
            "query": "find the recorded product launch",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.875,
        metadata={"agent_type": "routing"},
    )
    first_timestamp = datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc)
    retry_timestamp = datetime(2026, 8, 5, 3, 9, 10, tzinfo=timezone.utc)
    events = []

    async def get_item_span_id(item_id, batch_id=None):
        assert (item_id, batch_id) == (item.item_id, "batch-dataset-retry")
        return "approval-span"

    async def annotation_fails(**_kwargs):
        events.append("annotation-failed")
        raise ConnectionError("Phoenix annotation unavailable")

    async def annotation_succeeds(**kwargs):
        events.append("annotation-succeeded")
        assert kwargs["decision_timestamp"] == first_timestamp
        return True

    async def update_item(updated, batch_id=None):
        events.append("status-updated")
        assert updated.status is ApprovalStatus.APPROVED
        assert updated.reviewed_at == first_timestamp
        assert batch_id == "batch-dataset-retry"

    storage.get_item_span_id = get_item_span_id
    storage.log_approval_decision = annotation_fails
    storage.update_item = update_item
    first_decision = ReviewDecision(
        item_id=item.item_id,
        approved=True,
        reviewer="reviewer@example.com",
        feedback="The route is exact.",
        corrections={"query": "find the recorded product launch"},
        timestamp=first_timestamp,
    )
    with pytest.raises(RuntimeError) as first_error:
        await storage.persist_approved_item(
            batch_id=batch_id,
            dataset_name=dataset_name,
            item=item,
            decision=first_decision,
            project_context={"tenant_id": tenant_id, "optimizer": "routing"},
        )
    assert isinstance(first_error.value.__cause__, ConnectionError)

    storage.log_approval_decision = annotation_succeeds
    approved = await storage.persist_approved_item(
        batch_id=batch_id,
        dataset_name=dataset_name,
        item=item,
        decision=ReviewDecision(
            item_id=item.item_id,
            approved=True,
            reviewer="reviewer@example.com",
            feedback="The route is exact.",
            corrections={"query": "find the recorded product launch"},
            timestamp=retry_timestamp,
        ),
        project_context={"tenant_id": tenant_id, "optimizer": "routing"},
    )

    assert datasets.write_count == 1
    assert len(datasets.frame) == 1
    record = datasets.frame.iloc[0]["input"]
    assert record["reviewed_at"] == first_timestamp.isoformat()
    assert record["metadata.decision"]["timestamp"] == first_timestamp.isoformat()
    assert approved.reviewed_at == first_timestamp
    assert approved.metadata["decision"]["timestamp"] == first_timestamp.isoformat()
    assert events == ["annotation-failed", "annotation-succeeded", "status-updated"]


@pytest.mark.parametrize(
    "confidence",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
    ],
)
async def test_non_finite_candidate_is_rejected_before_redis_write(
    redis_url, confidence
):
    candidate = _candidate(31)
    candidate["confidence"] = confidence

    with pytest.raises(
        ValueError,
        match=(
            "Replacement candidate is not strict JSON for tenant=acme:non-finite "
            "batch=batch-non-finite original=routing_17"
        ),
    ):
        await _store(redis_url).select_canonical(
            tenant_id="acme:non-finite",
            batch_id="batch-non-finite",
            original_item_id="routing_17",
            candidate=candidate,
        )

    redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        assert (
            await redis.exists(
                RedisReplacementRecordStore._key(
                    "acme:non-finite", "batch-non-finite", "routing_17"
                )
            )
            == 0
        )
    finally:
        await redis.aclose()


@pytest.mark.parametrize(
    "stored_payload",
    [
        pytest.param("[]", id="non-object"),
        pytest.param('{"confidence":NaN}', id="non-finite-json"),
        pytest.param('{"z":1, "a":2}', id="non-canonical-json"),
    ],
)
async def test_corrupt_persisted_record_is_rejected(redis_url, stored_payload):
    tenant_id = f"acme:corrupt:{hashlib.sha256(stored_payload.encode()).hexdigest()}"
    batch_id = "batch-corrupt"
    original_item_id = "routing_17"
    key = RedisReplacementRecordStore._key(tenant_id, batch_id, original_item_id)
    redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await redis.set(key, stored_payload)
    finally:
        await redis.aclose()

    with pytest.raises(
        RuntimeError,
        match=(
            "Stored canonical replacement is invalid for "
            f"tenant={tenant_id} batch={batch_id} original={original_item_id}"
        ),
    ):
        await _store(redis_url).select_canonical(
            tenant_id=tenant_id,
            batch_id=batch_id,
            original_item_id=original_item_id,
            candidate=_candidate(32),
        )


async def test_unavailable_redis_raises_with_record_context():
    unavailable_url = f"redis://127.0.0.1:{_free_port()}/0"

    with pytest.raises(RuntimeError) as error:
        await _store(unavailable_url).select_canonical(
            tenant_id="acme:failure",
            batch_id="batch-failure",
            original_item_id="routing_17",
            candidate=_candidate(30),
        )

    assert str(error.value) == (
        "Failed to select canonical replacement for tenant=acme:failure "
        "batch=batch-failure original=routing_17"
    )
    assert isinstance(error.value.__cause__, RedisError)


async def test_redis_loss_aborts_event_owner_before_replacement_export(redis_url):
    store = _store(redis_url)
    store._EVENT_LOCK_LEASE_MS = 300
    critical_section_entered = asyncio.Event()
    events = []

    async def export_replacement() -> None:
        async with store.replacement_event_lock(
            tenant_id="acme:lease-loss",
            batch_id="batch-lease-loss",
            original_item_id="routing_17",
        ):
            events.append("critical_section_entered")
            critical_section_entered.set()
            await asyncio.Event().wait()
            events.append("phoenix_replacement_written")

    export_task = asyncio.create_task(export_replacement())
    await asyncio.wait_for(critical_section_entered.wait(), timeout=2)
    container_name = _redis_container_name()
    try:
        stopped = await asyncio.to_thread(
            subprocess.run,
            ["docker", "stop", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert stopped.returncode == 0, stopped.stderr

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to renew replacement event lock for "
                "tenant=acme:lease-loss batch=batch-lease-loss original=routing_17"
            ),
        ) as exc_info:
            await asyncio.wait_for(export_task, timeout=4)

        assert isinstance(exc_info.value.__cause__, RedisError)
        assert events == ["critical_section_entered"]
    finally:
        started = await asyncio.to_thread(
            subprocess.run,
            ["docker", "start", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert started.returncode == 0, started.stderr
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ping = await asyncio.to_thread(
                subprocess.run,
                ["docker", "exec", container_name, "redis-cli", "ping"],
                capture_output=True,
                text=True,
            )
            if ping.stdout.strip() == "PONG":
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("Redis did not recover within 30 seconds")
