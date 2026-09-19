"""Real-Redis contract for the shared A2A task store."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import subprocess
import time
import uuid

import pytest
import redis.asyncio as aioredis
from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from cogniverse_runtime.a2a_task_store import (
    A2ATaskCapacityError,
    A2ATaskStoreError,
    RedisTaskStore,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def redis_url():
    override = os.environ.get("COGNIVERSE_TEST_REDIS_URL")
    if override:
        yield override
        return

    port = _free_port()
    container_name = f"redis-a2a-task-store-{os.getpid()}"
    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"
    )
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
            "--platform",
            docker_platform,
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


@pytest.fixture
async def redis_client(redis_url):
    client = aioredis.from_url(redis_url, decode_responses=True)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


def _task(task_id: str, state: TaskState = TaskState.input_required) -> Task:
    context_id = f"context-{task_id}"
    user = Message(
        message_id=f"message-user-{task_id}",
        context_id=context_id,
        task_id=task_id,
        role=Role.user,
        parts=[
            Part(root=TextPart(text=f"question-{task_id}", metadata={"turn": 1})),
            Part(root=DataPart(data={"filters": ["video", "document"]})),
        ],
        metadata={"request": {"tenant": "test:tenant"}},
        reference_task_ids=["source-task"],
        extensions=["urn:cogniverse:test"],
    )
    agent = Message(
        message_id=f"message-agent-{task_id}",
        context_id=context_id,
        task_id=task_id,
        role=Role.agent,
        parts=[Part(root=TextPart(text=f"answer-{task_id}"))],
    )
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(
            state=state,
            message=agent,
            timestamp="2026-09-19T10:11:12Z",
        ),
        history=[user, agent],
        artifacts=[
            Artifact(
                artifact_id=f"artifact-{task_id}",
                name="evidence",
                description="exact round-trip fixture",
                parts=[
                    Part(root=TextPart(text="artifact text")),
                    Part(root=FilePart(file=FileWithBytes(bytes="Ynl0ZXM="))),
                ],
                metadata={"rank": 1},
                extensions=["urn:cogniverse:artifact"],
            )
        ],
        metadata={"nested": {"score": 0.75}, "labels": ["a", "b"]},
    )


async def test_exact_task_round_trip_across_clients(redis_client, redis_url):
    writer = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    peer_client = aioredis.from_url(redis_url, decode_responses=True)
    reader = RedisTaskStore(peer_client, max_tasks=10, key_prefix="test:a2a")
    expected = _task("task-round-trip")
    try:
        await writer.save(expected)
        actual = await reader.get(expected.id)
    finally:
        await peer_client.aclose()

    assert actual.model_dump(mode="json") == expected.model_dump(mode="json")


async def test_owned_store_connects_deletes_and_closes(redis_url):
    store = await RedisTaskStore.from_url(
        redis_url, max_tasks=10, key_prefix="test:a2a-owned"
    )
    task = _task("task-delete")
    await store.save(task)

    await store.delete(task.id)
    deleted = await store.get(task.id)
    await store.close()

    assert deleted is None


async def test_inactive_lru_eviction_preserves_active_tasks(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=3, key_prefix="test:a2a")
    await store.save(_task("inactive-old"))
    await store.save(_task("active", TaskState.working))
    await store.save(_task("inactive-hot"))
    assert (await store.get("inactive-old")).id == "inactive-old"
    assert (await store.get("inactive-hot")).id == "inactive-hot"

    await store.save(_task("incoming"))

    assert await store.get("inactive-old") is None
    assert (await store.get("active")).id == "active"
    assert (await store.get("inactive-hot")).id == "inactive-hot"
    assert (await store.get("incoming")).id == "incoming"


async def test_capacity_rejects_when_only_active_tasks_can_be_retained(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    await store.save(_task("active-a", TaskState.submitted))
    await store.save(_task("active-b", TaskState.auth_required))

    with pytest.raises(
        A2ATaskCapacityError,
        match="capacity 2 is full of active tasks; rejected task incoming",
    ):
        await store.save(_task("incoming"))

    retained = await asyncio.gather(
        store.get("active-a"), store.get("active-b"), store.get("incoming")
    )
    assert [task.id if task else None for task in retained] == [
        "active-a",
        "active-b",
        None,
    ]


async def test_concurrent_saves_never_exceed_capacity(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=4, key_prefix="test:a2a")
    tasks = [_task(f"concurrent-{index}") for index in range(12)]

    await asyncio.gather(*(store.save(task) for task in tasks))

    loaded = await asyncio.gather(*(store.get(task.id) for task in tasks))
    retained_ids = [task.id for task in loaded if task]
    assert len(retained_ids) == 4
    assert set(retained_ids).issubset({task.id for task in tasks})


async def test_redis_outage_is_an_explicit_store_error():
    closed_port = _free_port()
    client = aioredis.from_url(
        f"redis://127.0.0.1:{closed_port}/0",
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )
    store = RedisTaskStore(client, max_tasks=2, key_prefix=f"test:{uuid.uuid4().hex}")
    try:
        with pytest.raises(A2ATaskStoreError, match="get task outage-task"):
            await store.get("outage-task")
        with pytest.raises(A2ATaskStoreError, match="save task outage-task"):
            await store.save(_task("outage-task"))
    finally:
        await client.aclose()
