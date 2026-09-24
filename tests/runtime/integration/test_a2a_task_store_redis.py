"""Real-Redis contract for the shared A2A task store."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import subprocess
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest
import redis.asyncio as aioredis
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    InvalidParamsError,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils.errors import ServerError

from cogniverse_runtime.a2a_request_handler import RedisRequestHandler
from cogniverse_runtime.a2a_task_store import (
    A2ATaskCapacityError,
    A2ATaskConflictError,
    A2ATaskOwnershipLostError,
    A2ATaskStoreError,
    RedisTaskStore,
)
from cogniverse_runtime.main import _build_shared_a2a_protocol

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


def _owned_context(store: RedisTaskStore, lease) -> ServerCallContext:
    """Bind a lease the way the request handler binds one for the SDK."""
    context = ServerCallContext()
    store.attach_execution(context, lease)
    return context


def _seed_store(redis, **kwargs) -> RedisTaskStore:
    """Write a precondition with no execution behind it; serving never does."""
    return RedisTaskStore(redis, enforce_leases=False, **kwargs)


async def _save_owned(store: RedisTaskStore, task: Task) -> None:
    """Save one task the way a served execution does: acquire, save, release."""
    lease = await store.acquire_execution(
        task.id, replica_id="replica-writer", lease_seconds=2
    )
    try:
        await store.save(task, _owned_context(store, lease))
    finally:
        await store.release_execution(lease)


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
    writer = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
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
    await _save_owned(store, task)

    await store.delete(task.id)
    deleted = await store.get(task.id)
    await store.close()

    assert deleted is None


async def test_inactive_lru_eviction_preserves_active_tasks(redis_client):
    store = _seed_store(redis_client, max_tasks=3, key_prefix="test:a2a")
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
    store = _seed_store(redis_client, max_tasks=2, key_prefix="test:a2a")
    await store.save(_task("active-a", TaskState.submitted))
    await store.save(_task("active-b", TaskState.auth_required))

    with pytest.raises(
        A2ATaskCapacityError,
        match="capacity 2 is full of active or leased tasks; rejected task incoming",
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
    store = _seed_store(redis_client, max_tasks=4, key_prefix="test:a2a")
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
        with pytest.raises(
            A2ATaskStoreError,
            match="shared A2A task store unavailable: get task outage-task",
        ):
            await store.get("outage-task")
        with pytest.raises(
            A2ATaskStoreError,
            match="shared A2A task store unavailable: save task outage-task",
        ):
            await store.save(_task("outage-task"))
    finally:
        await client.aclose()


async def test_same_task_conflicts_while_distinct_tasks_acquire(redis_client):
    store = RedisTaskStore(
        redis_client,
        max_tasks=10,
        key_prefix="test:a2a",
        enforce_leases=True,
    )
    lease_a = await store.acquire_execution(
        "task-shared", replica_id="replica-a", lease_seconds=2
    )

    with pytest.raises(
        A2ATaskConflictError,
        match="task task-shared is owned by replica-a; retry after",
    ):
        await store.acquire_execution(
            "task-shared", replica_id="replica-b", lease_seconds=2
        )

    lease_b, lease_c = await asyncio.gather(
        store.acquire_execution(
            "task-distinct-b", replica_id="replica-b", lease_seconds=2
        ),
        store.acquire_execution(
            "task-distinct-c", replica_id="replica-c", lease_seconds=2
        ),
    )
    # One monotonic sequence issues every generation, so no two leases share
    # one even across tasks, and a recycled task id can never reissue one.
    assert lease_a.generation == 1
    assert sorted((lease_b.generation, lease_c.generation)) == [2, 3]
    await asyncio.gather(
        store.release_execution(lease_a),
        store.release_execution(lease_b),
        store.release_execution(lease_c),
    )


async def test_lease_renewal_prevents_takeover(redis_client):
    store = RedisTaskStore(
        redis_client,
        max_tasks=10,
        key_prefix="test:a2a",
        enforce_leases=True,
    )
    lease = await store.acquire_execution(
        "task-renew", replica_id="replica-a", lease_seconds=0.2
    )
    await asyncio.sleep(0.12)
    await store.renew_execution(lease, lease_seconds=0.3)
    await asyncio.sleep(0.12)

    with pytest.raises(A2ATaskConflictError, match="owned by replica-a"):
        await store.acquire_execution(
            "task-renew", replica_id="replica-b", lease_seconds=0.2
        )

    await store.release_execution(lease)


async def test_cancel_generation_fences_late_owner_save(redis_client):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(
        redis_client,
        max_tasks=10,
        key_prefix="test:a2a",
        enforce_leases=True,
    )
    await seed.save(_task("task-cancel", TaskState.working))
    executing = await store.acquire_execution(
        "task-cancel", replica_id="replica-a", lease_seconds=2
    )
    canceling = await store.begin_cancel(
        "task-cancel", replica_id="replica-a", lease_seconds=2
    )

    with pytest.raises(
        A2ATaskOwnershipLostError,
        match="save task task-cancel with stale ownership generation 1",
    ):
        await store.save(
            _task("task-cancel", TaskState.input_required),
            _owned_context(store, executing),
        )

    canceled = _task("task-cancel", TaskState.canceled)
    await store.save(canceled, _owned_context(store, canceling))
    await store.release_execution(canceling)

    actual = await store.get("task-cancel")
    assert actual.model_dump(mode="json") == canceled.model_dump(mode="json")


async def test_released_owner_save_lands_until_a_new_generation_fences_it(
    redis_client,
):
    """A turn's last save can reach Redis after its lease was released.

    On the non-blocking path the SDK keeps persisting events in a background
    task that nothing sequences against producer cleanup, so the terminal save
    can arrive after the lease is gone. Fencing is on the generation, so that
    write still lands, and only a newer owner's generation rejects it.
    """
    store = RedisTaskStore(
        redis_client,
        max_tasks=10,
        key_prefix="test:a2a",
        enforce_leases=True,
    )
    owner = await store.acquire_execution(
        "task-late-save", replica_id="replica-a", lease_seconds=2
    )
    owner_context = _owned_context(store, owner)
    await store.save(_task("task-late-save", TaskState.working), owner_context)
    assert await store.release_execution(owner) is True

    terminal = _task("task-late-save", TaskState.input_required)
    await store.save(terminal, owner_context)
    landed = await store.get("task-late-save")
    assert landed.model_dump(mode="json") == terminal.model_dump(mode="json")

    await store.acquire_execution(
        "task-late-save", replica_id="replica-b", lease_seconds=2
    )
    with pytest.raises(
        A2ATaskOwnershipLostError,
        match="save task task-late-save with stale ownership generation 1",
    ):
        await store.save(_task("task-late-save", TaskState.working), owner_context)


async def test_retention_bookkeeping_is_dropped_with_the_task(redis_client):
    """Eviction and delete must not leave per-task bookkeeping behind forever.

    The task hash is capped by ``max_tasks``; the lease, generation and event
    keys are per task id, so leaving them behind grows without bound. The one
    key that outlives every task is the single generation sequence.
    """
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    working = TaskStatusUpdateEvent(
        task_id="deleted",
        context_id="context-deleted",
        final=False,
        status=TaskStatus(state=TaskState.working),
    )
    await _save_owned(store, _task("deleted"))
    await store.publish_event("deleted", working)
    await store.delete("deleted")

    await _save_owned(store, _task("evicted"))
    await store.publish_event(
        "evicted", working.model_copy(update={"task_id": "evicted"})
    )
    await _save_owned(store, _task("kept"))
    await _save_owned(store, _task("incoming"))

    assert await store.get("evicted") is None
    assert await store.get("kept") is not None
    assert sorted(await redis_client.hkeys("test:a2a:generations")) == [
        "incoming",
        "kept",
    ]
    assert await redis_client.hkeys("test:a2a:leases") == []
    assert await redis_client.exists("test:a2a:events:deleted") == 0
    assert await redis_client.exists("test:a2a:events:evicted") == 0
    assert sorted(await redis_client.keys("test:a2a:*")) == [
        "test:a2a:generation-seq",
        "test:a2a:generations",
        "test:a2a:inactive-lru",
        "test:a2a:tasks",
    ]


async def test_an_id_acquired_but_never_saved_leaves_no_bookkeeping(redis_client):
    """An execution that never lands a task must not leave a generation behind.

    ``message/send`` with an unknown task id acquires before it reads, then
    fails with task-not-found; so does an owner that dies before its first
    save. Neither creates a task, so neither may grow the bookkeeping.
    """
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    released = await store.acquire_execution(
        "never-saved", replica_id="replica-a", lease_seconds=0.2
    )
    await store.release_execution(released)
    await store.acquire_execution(
        "owner-died", replica_id="replica-a", lease_seconds=0.2
    )
    assert sorted(await redis_client.hkeys("test:a2a:generations")) == [
        "never-saved",
        "owner-died",
    ]

    await asyncio.sleep(0.3)

    assert await redis_client.hkeys("test:a2a:generations") == []
    assert await redis_client.hkeys("test:a2a:leases") == []
    with pytest.raises(
        A2ATaskOwnershipLostError,
        match=f"save task never-saved with stale ownership generation "
        f"{released.generation}",
    ):
        await store.save(_task("never-saved"), _owned_context(store, released))
    assert await store.get("never-saved") is None


async def test_a_new_task_saved_within_its_lease_keeps_its_generation(redis_client):
    """The first save of a task makes its bookkeeping permanent.

    The non-blocking path can land a task's first save just after its lease
    was released; that save must still land and must not be expired later.
    """
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    owner = await store.acquire_execution(
        "late-first-save", replica_id="replica-a", lease_seconds=0.2
    )
    await store.release_execution(owner)
    first = _task("late-first-save", TaskState.input_required)
    await store.save(first, _owned_context(store, owner))

    await asyncio.sleep(0.3)

    assert await redis_client.hget("test:a2a:generations", "late-first-save") == str(
        owner.generation
    )
    landed = await store.get("late-first-save")
    assert landed.model_dump(mode="json") == first.model_dump(mode="json")


async def test_a_straggler_cannot_resurrect_an_evicted_task(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    owner = await store.acquire_execution(
        "victim", replica_id="replica-a", lease_seconds=2
    )
    owner_context = _owned_context(store, owner)
    await store.save(_task("victim"), owner_context)
    assert await store.release_execution(owner) is True
    await _save_owned(store, _task("kept"))
    await _save_owned(store, _task("incoming"))
    assert await store.get("victim") is None

    with pytest.raises(
        A2ATaskOwnershipLostError,
        match=f"save task victim with stale ownership generation {owner.generation}",
    ):
        await store.save(_task("victim", TaskState.working), owner_context)

    assert await store.get("victim") is None
    assert (await store.get("kept")).id == "kept"
    assert (await store.get("incoming")).id == "incoming"


async def test_a_straggler_cannot_resurrect_or_overwrite_a_deleted_task(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner = await store.acquire_execution(
        "recycled", replica_id="replica-a", lease_seconds=2
    )
    owner_context = _owned_context(store, owner)
    await store.save(_task("recycled"), owner_context)
    await store.release_execution(owner)
    await store.delete("recycled")

    with pytest.raises(
        A2ATaskOwnershipLostError,
        match=f"save task recycled with stale ownership generation {owner.generation}",
    ):
        await store.save(_task("recycled", TaskState.working), owner_context)
    assert await store.get("recycled") is None

    reborn = await store.acquire_execution(
        "recycled", replica_id="replica-b", lease_seconds=2
    )
    assert reborn.generation > owner.generation
    current = _task("recycled", TaskState.working)
    await store.save(current, _owned_context(store, reborn))
    with pytest.raises(
        A2ATaskOwnershipLostError,
        match=f"save task recycled with stale ownership generation {owner.generation}",
    ):
        await store.save(_task("recycled", TaskState.completed), owner_context)

    stored = await store.get("recycled")
    assert stored.model_dump(mode="json") == current.model_dump(mode="json")


async def test_a_task_with_a_live_lease_is_never_evicted(redis_client):
    """A continuation owns its task before it saves the next state.

    Until that save the task is still recorded in ``input_required``, which is
    inactive, so an LRU-only victim choice evicts it from under its owner.
    """
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    await _save_owned(store, _task("continuing"))
    continuation = await store.acquire_execution(
        "continuing", replica_id="replica-a", lease_seconds=5
    )
    await _save_owned(store, _task("idle"))

    await _save_owned(store, _task("incoming"))

    assert await store.get("idle") is None
    assert (await store.get("continuing")).id == "continuing"
    assert (await store.get("incoming")).id == "incoming"

    # With every retained task leased, capacity refuses rather than evicts.
    await store.acquire_execution("incoming", replica_id="replica-b", lease_seconds=5)
    with pytest.raises(
        A2ATaskCapacityError,
        match="capacity 2 is full of active or leased tasks; rejected task overflow",
    ):
        await _save_owned(store, _task("overflow"))
    assert await store.get("overflow") is None

    continued = _task("continuing", TaskState.working)
    await store.save(continued, _owned_context(store, continuation))
    stored = await store.get("continuing")
    assert stored.model_dump(mode="json") == continued.model_dump(mode="json")


async def test_stragglers_racing_eviction_never_resurrect_or_overflow(redis_client):
    """Released owners' late saves race capacity eviction of their tasks.

    Whatever interleaving Redis serializes, retention never exceeds capacity,
    a straggler that was refused leaves no task behind, and bookkeeping names
    only retained tasks.
    """
    store = RedisTaskStore(redis_client, max_tasks=4, key_prefix="test:a2a")
    owners = []
    for index in range(8):
        task_id = f"old-{index}"
        lease = await store.acquire_execution(
            task_id, replica_id="replica-old", lease_seconds=2
        )
        context = _owned_context(store, lease)
        await store.save(_task(task_id), context)
        await store.release_execution(lease)
        owners.append((task_id, context))

    outcomes = await asyncio.gather(
        *(
            store.save(_task(task_id, TaskState.working), context)
            for task_id, context in owners
        ),
        *(_save_owned(store, _task(f"new-{index}")) for index in range(8)),
        return_exceptions=True,
    )

    stragglers, writers = outcomes[:8], outcomes[8:]
    assert [
        outcome
        for outcome in stragglers
        if outcome is not None and not isinstance(outcome, A2ATaskOwnershipLostError)
    ] == []
    assert [
        outcome
        for outcome in writers
        if outcome is not None and not isinstance(outcome, A2ATaskCapacityError)
    ] == []
    refused = [
        task_id
        for (task_id, _), outcome in zip(owners, stragglers, strict=True)
        if outcome is not None
    ]
    retained = await redis_client.hkeys("test:a2a:tasks")
    assert len(retained) <= 4
    assert set(refused).isdisjoint(retained)
    assert await redis_client.hkeys("test:a2a:leases") == []

    # A writer refused at capacity never created its task; its generation
    # lapses with the lease it was issued under.
    await asyncio.sleep(2.1)
    assert sorted(await redis_client.hkeys("test:a2a:generations")) == sorted(retained)


async def test_a_default_store_fences_every_save(redis_client, redis_url):
    """Fencing is what serving relies on, so it is not an opt-in."""
    constructed = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    connected = await RedisTaskStore.from_url(
        redis_url, max_tasks=2, key_prefix="test:a2a"
    )
    try:
        for store in (constructed, connected):
            with pytest.raises(
                A2ATaskOwnershipLostError,
                match="save task unowned with stale ownership generation 0",
            ):
                await store.save(_task("unowned"))
    finally:
        await connected.close()
    assert await redis_client.hkeys("test:a2a:tasks") == []


async def test_a_dead_owners_relay_expires_even_after_its_terminal_save(redis_client):
    """An owner can die between saving its terminal status and closing its relay.

    The task is then inactive, so the interruption is declined; the relay it
    left behind still has no expiry and must get the drain window anyway. A
    live owner's relay is never expired.
    """
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner = await store.acquire_execution(
        "task-dead-owner", replica_id="replica-dead", lease_seconds=0.3
    )
    owner_context = _owned_context(store, owner)
    await store.save(_task("task-dead-owner", TaskState.working), owner_context)
    await store.publish_event(
        "task-dead-owner",
        TaskStatusUpdateEvent(
            task_id="task-dead-owner",
            context_id="context-task-dead-owner",
            final=False,
            status=TaskStatus(state=TaskState.working),
        ),
    )
    await store.save(_task("task-dead-owner", TaskState.input_required), owner_context)

    assert await store.mark_owner_lost("task-dead-owner") is False
    assert await redis_client.ttl("test:a2a:events:task-dead-owner") == -1

    await asyncio.sleep(0.4)

    assert await store.mark_owner_lost("task-dead-owner") is False
    ttl = await redis_client.ttl("test:a2a:events:task-dead-owner")
    assert 0 < ttl <= 60
    assert (await store.get("task-dead-owner")).status.state == TaskState.input_required


async def test_expired_owner_is_marked_interrupted_without_reexecution(redis_client):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(
        redis_client,
        max_tasks=10,
        key_prefix="test:a2a",
        enforce_leases=True,
    )
    await seed.save(_task("task-orphan", TaskState.working))
    await store.acquire_execution(
        "task-orphan", replica_id="replica-lost", lease_seconds=0.1
    )
    await asyncio.sleep(0.15)

    with pytest.raises(
        A2ATaskOwnershipLostError,
        match="task task-orphan lost owner replica-lost during active execution",
    ):
        await store.acquire_execution(
            "task-orphan", replica_id="replica-next", lease_seconds=1
        )

    interrupted = await store.mark_owner_lost("task-orphan")
    actual = await store.get("task-orphan")
    assert interrupted is True
    assert actual.status.state == TaskState.failed
    assert actual.status.message.parts[0].root.text == (
        "Execution interrupted because its owning runtime stopped before completion."
    )


async def test_runtime_protocol_factory_owns_shared_store_and_handler(redis_url):
    agent = SimpleNamespace(capabilities=["search", "video_search"])

    class Registry:
        def list_agents(self):
            return ["search_agent"]

        def get_agent(self, name):
            return agent if name == "search_agent" else None

    protocol = await _build_shared_a2a_protocol(
        agent_registry=Registry(),
        dispatcher=SimpleNamespace(),
        redis_url=redis_url,
        replica_id="factory-replica",
        max_tasks=7,
        lease_seconds=1,
        cancel_timeout_seconds=2,
        drain_timeout_seconds=1,
        max_concurrent_cancels=5,
    )
    try:
        assert protocol.skill_ids == ("search_agent",)
        assert protocol.handler.task_store is protocol.task_store
        assert protocol.handler._max_concurrent_cancels == 5
        assert [route.path for route in protocol.app.routes] == [
            "/",
            "/.well-known/agent-card.json",
            "/.well-known/agent.json",
        ]
    finally:
        await protocol.close()


async def test_runtime_protocol_factory_validates_redis_before_registry_access():
    class Registry:
        def list_agents(self):
            raise AssertionError("registry accessed before Redis validation")

    closed_port = _free_port()
    with pytest.raises(
        A2ATaskStoreError, match="shared A2A task store unavailable: connect to"
    ):
        await _build_shared_a2a_protocol(
            agent_registry=Registry(),
            dispatcher=SimpleNamespace(),
            redis_url=f"redis://127.0.0.1:{closed_port}/0",
            replica_id="factory-replica",
            max_tasks=7,
            lease_seconds=1,
            cancel_timeout_seconds=2,
            drain_timeout_seconds=1,
        )


async def test_runtime_protocol_close_cancels_and_drains_active_execution(redis_url):
    canceled = asyncio.Event()

    class BlockingExecutor(AgentExecutor):
        async def execute(
            self, context: RequestContext, event_queue: EventQueue
        ) -> None:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=False,
                    status=TaskStatus(state=TaskState.working),
                )
            )
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                canceled.set()
                raise

        async def cancel(
            self, context: RequestContext, event_queue: EventQueue
        ) -> None:
            raise AssertionError("shutdown must cancel the active producer task")

    class Registry:
        def list_agents(self):
            return []

        def get_agent(self, name):
            return None

    protocol = await _build_shared_a2a_protocol(
        agent_registry=Registry(),
        dispatcher=SimpleNamespace(),
        redis_url=redis_url,
        replica_id="shutdown-replica",
        max_tasks=7,
        lease_seconds=0.2,
        cancel_timeout_seconds=1,
        drain_timeout_seconds=0.2,
    )
    protocol.handler.agent_executor = BlockingExecutor()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=protocol.app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "shutdown-start",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": "shutdown-message",
                        "contextId": "shutdown-context",
                        "parts": [{"kind": "text", "text": "block"}],
                    },
                    "configuration": {
                        "acceptedOutputModes": ["text"],
                        "blocking": False,
                    },
                    "metadata": {
                        "agent_name": "search_agent",
                        "tenant_id": "test:shutdown",
                    },
                },
            },
        )
    assert response.json()["result"]["status"]["state"] == "working"

    await asyncio.wait_for(protocol.close(), timeout=1)

    assert canceled.is_set() is True


async def test_served_store_outage_is_a_dependency_error_not_task_not_found():
    """A caller must be able to tell a broken store from a missing task.

    ``tasks/get`` is the method a peer uses to find a task another replica
    created, so a Redis outage answered as ``-32001 Task not found`` would
    read exactly like the cross-replica bug this store exists to fix.
    """

    class UnusedExecutor(AgentExecutor):
        async def execute(self, context, event_queue) -> None:
            raise AssertionError("no execution is expected on a broken store")

        async def cancel(self, context, event_queue) -> None:
            raise AssertionError("no cancellation is expected on a broken store")

    closed_port = _free_port()
    client = aioredis.from_url(
        f"redis://127.0.0.1:{closed_port}/0",
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )
    store = RedisTaskStore(client, max_tasks=2, key_prefix=f"test:{uuid.uuid4().hex}")
    handler = RedisRequestHandler(
        agent_executor=UnusedExecutor(),
        task_store=store,
        replica_id="outage-replica",
    )
    card = AgentCard(
        name="Cogniverse Runtime",
        description="store outage test",
        url="http://127.0.0.1:1/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="search_agent",
                name="search_agent",
                description="store outage test",
                tags=["search"],
            )
        ],
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            response = await http_client.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": "outage-get",
                    "method": "tasks/get",
                    "params": {"id": "outage-task"},
                },
            )
    finally:
        await client.aclose()

    error = response.json()["error"]
    assert error["code"] == -32603, error
    assert error["message"] == (
        "shared A2A task store unavailable: get task outage-task"
    )


def _cancel_handler(store: RedisTaskStore, replica_id: str) -> RedisRequestHandler:
    class UnusedExecutor(AgentExecutor):
        async def execute(self, context, event_queue) -> None:
            raise AssertionError("no execution is expected")

        async def cancel(self, context, event_queue) -> None:
            raise AssertionError("no cancellation is expected")

    return RedisRequestHandler(
        agent_executor=UnusedExecutor(),
        task_store=store,
        replica_id=replica_id,
    )


async def test_cancel_uses_redis_time_not_the_replica_wall_clock(redis_client):
    """A pod whose clock runs ahead of Redis must not declare live owners dead.

    Every other liveness check reads ``redis.time()``. Comparing a
    Redis-derived ``expires_at_ms`` against the replica's own ``time.time()``
    made a skewed replica report "owner expired; task is interrupted" — which
    is false, because the interrupt script re-checks against Redis and
    declines, so the task keeps running and the cancel just fails.
    """
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("task-skew", TaskState.working))
    lease = await store.acquire_execution(
        "task-skew", replica_id="replica-owner", lease_seconds=30
    )
    # A replica a day ahead of Redis: the wall-clock comparison this path used
    # to make would have called this live owner expired.
    assert lease.expires_at_ms <= int((time.time() + 86_400) * 1000)

    handler = _cancel_handler(store, "replica-owner")
    consulted = []
    real_has_live_owner = store.has_live_owner

    async def spy_has_live_owner(task_id: str):
        consulted.append(task_id)
        return await real_has_live_owner(task_id)

    store.has_live_owner = spy_has_live_owner
    owned = []

    async def record_cancel(task_id: str):
        owned.append(task_id)
        return _task(task_id, TaskState.canceled)

    handler._cancel_owned = record_cancel

    result = await handler.on_cancel_task(TaskIdParams(id="task-skew"))

    assert consulted == ["task-skew"]
    assert owned == ["task-skew"]
    assert result.status.state == TaskState.canceled
    assert (await store.get("task-skew")).status.state == TaskState.working


async def test_cancel_honors_a_declined_interruption_instead_of_raising(redis_client):
    """``mark_owner_lost`` declining means the task was NOT interrupted.

    An idle task carrying a stale lease record is not active, so the interrupt
    script refuses it. Raising "owner expired; task is interrupted" there
    reports an interruption that did not happen and refuses the cancel.
    """
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("task-idle-stale", TaskState.working))
    await store.acquire_execution(
        "task-idle-stale", replica_id="replica-gone", lease_seconds=0.05
    )
    await asyncio.sleep(0.1)
    await seed.save(_task("task-idle-stale", TaskState.input_required))
    assert await store.get_execution_lease("task-idle-stale") is not None
    assert await store.has_live_owner("task-idle-stale") is False
    assert await store.mark_owner_lost("task-idle-stale") is False

    handler = _cancel_handler(store, "replica-self")
    owned = []

    async def record_cancel(task_id: str):
        owned.append(task_id)
        return _task(task_id, TaskState.canceled)

    handler._cancel_owned = record_cancel

    result = await handler.on_cancel_task(TaskIdParams(id="task-idle-stale"))

    assert owned == ["task-idle-stale"]
    assert result.status.state == TaskState.canceled
    assert (await store.get("task-idle-stale")).status.state != TaskState.failed


async def test_idle_cancel_losing_the_race_is_a_conflict_not_an_internal_error(
    redis_client,
):
    """A lost cancel race is a conflict the client can act on.

    ``message/send`` already answers a lost ownership race with an explicit
    conflict; the idle cancel path let ``A2ATaskOwnershipLostError`` escape
    untranslated, so the same race read as a generic internal error on the
    most commonly cancelled path.
    """
    from a2a.types import TaskIdParams
    from a2a.utils.errors import ServerError

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("task-race", TaskState.input_required))
    handler = _cancel_handler(store, "replica-self")

    async def lost_race(*args, **kwargs):
        raise A2ATaskOwnershipLostError(
            "cancel task task-race is owned by another replica "
            "(replica-winner), not replica-self"
        )

    store.begin_cancel = lost_race

    with pytest.raises(ServerError) as caught:
        await handler.on_cancel_task(TaskIdParams(id="task-race"))

    assert caught.value.error.message == (
        "Task task-race is active on another replica; retry"
    )


async def test_a_stale_save_of_the_stored_terminal_state_is_acknowledged_unwritten(
    redis_client,
):
    """A superseded owner repeating the terminal state already stored is a no-op.

    The owner's own consumers persist the cancel event a canceller already
    persisted under the newer generation. That write can change nothing, so
    it is acknowledged without being written; any other stale write is still
    refused.
    """
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner = await store.acquire_execution(
        "task-stale-terminal", replica_id="replica-a", lease_seconds=2
    )
    owner_context = _owned_context(store, owner)
    await store.save(_task("task-stale-terminal", TaskState.working), owner_context)
    canceling = await store.begin_cancel(
        "task-stale-terminal", replica_id="replica-a", lease_seconds=2
    )
    canceled = _task("task-stale-terminal", TaskState.canceled)
    await store.save(canceled, _owned_context(store, canceling))

    repeated = _task("task-stale-terminal", TaskState.canceled)
    repeated.metadata = {"written_by": "superseded owner"}
    await store.save(repeated, owner_context)
    stored = await store.get("task-stale-terminal")
    assert stored.model_dump(mode="json") == canceled.model_dump(mode="json")

    for state in (TaskState.completed, TaskState.working):
        with pytest.raises(
            A2ATaskOwnershipLostError,
            match=(
                "save task task-stale-terminal with stale ownership generation "
                f"{owner.generation}"
            ),
        ):
            await store.save(_task("task-stale-terminal", state), owner_context)
    stored = await store.get("task-stale-terminal")
    assert stored.model_dump(mode="json") == canceled.model_dump(mode="json")


def _send_params(text: str, *, task_id: str | None = None, blocking: bool = True):
    from a2a.types import MessageSendConfiguration, MessageSendParams

    return MessageSendParams(
        message=Message(
            message_id=f"message-{uuid.uuid4().hex}",
            context_id="context-cancel",
            task_id=task_id,
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
        ),
        configuration=MessageSendConfiguration(
            accepted_output_modes=["text"], blocking=blocking
        ),
        metadata={"agent_name": "search_agent", "tenant_id": "test:cancel"},
    )


class _CancellableExecutor(AgentExecutor):
    """Runs until cancelled; its cancel answers with ``cancel_state``."""

    def __init__(self, cancel_state: TaskState = TaskState.canceled) -> None:
        self.cancel_state = cancel_state
        self.wedged_cancels: set[str] = set()
        self.wedge_every_cancel = False
        # A stubborn wedged cancel swallows cancellation until unwedged.
        self.stubborn = False
        self.unwedge = asyncio.Event()
        self.working = asyncio.Event()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=False,
                status=TaskStatus(state=TaskState.working),
            )
        )
        self.working.set()
        await asyncio.Event().wait()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if self.wedge_every_cancel or context.task_id in self.wedged_cancels:
            while not self.unwedge.is_set():
                try:
                    await self.unwedge.wait()
                except asyncio.CancelledError:
                    if not self.stubborn:
                        raise
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=True,
                status=TaskStatus(state=self.cancel_state),
            )
        )


def _relay_states(entries) -> list[str]:
    return [
        "closed"
        if "closed" in fields
        else TaskStatusUpdateEvent.model_validate_json(fields["payload"]).status.state
        for _, fields in entries
    ]


async def test_a_local_stream_consumer_observes_its_task_being_canceled(redis_client):
    """The owner's own ``message/stream`` consumer sees the cancel and ends.

    The cancel event used to go to a fresh queue nobody local read, so the
    stream stayed open waiting on a producer that had been cancelled.
    """
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler = RedisRequestHandler(
        agent_executor=executor, task_store=store, replica_id="replica-a"
    )
    events = []
    first_event = asyncio.Event()

    async def consume() -> None:
        async for event in handler.on_message_send_stream(_send_params("stream")):
            events.append(event)
            first_event.set()

    stream = asyncio.create_task(consume())
    await asyncio.wait_for(first_event.wait(), timeout=5)
    task_id = events[0].task_id

    result = await handler.on_cancel_task(TaskIdParams(id=task_id))
    await asyncio.wait_for(stream, timeout=5)
    await handler.close()

    assert result.status.state == TaskState.canceled
    assert [event.status.state for event in events] == [
        TaskState.working,
        TaskState.canceled,
    ]
    assert (await store.get(task_id)).status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{task_id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]


async def test_a_local_blocking_send_returns_its_task_canceled(redis_client):
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler = RedisRequestHandler(
        agent_executor=executor, task_store=store, replica_id="replica-a"
    )
    send = asyncio.create_task(handler.on_message_send(_send_params("blocking")))
    await asyncio.wait_for(executor.working.wait(), timeout=5)
    task_id = next(iter(handler._running_agents))
    async with asyncio.timeout(5):
        while await store.get(task_id) is None:
            await asyncio.sleep(0.01)

    await handler.on_cancel_task(TaskIdParams(id=task_id))
    sent = await asyncio.wait_for(send, timeout=5)
    await handler.close()

    assert sent.id == task_id
    assert sent.status.state == TaskState.canceled
    assert (await store.get(task_id)).status.state == TaskState.canceled


async def test_an_idle_cancel_closes_the_relay_it_publishes_on(redis_client):
    """With no live queue the cancel publishes on a relay it must also close."""
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("task-idle-relay", TaskState.input_required))
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(),
        task_store=store,
        replica_id="replica-a",
    )

    result = await handler.on_cancel_task(TaskIdParams(id="task-idle-relay"))

    assert result.status.state == TaskState.canceled
    relay = await redis_client.xrange("test:a2a:events:task-idle-relay")
    assert _relay_states(relay) == [TaskState.canceled, "closed"]
    ttl = await redis_client.ttl("test:a2a:events:task-idle-relay")
    assert 0 < ttl <= 60


async def test_a_cancel_that_does_not_end_canceled_is_not_cancelable(redis_client):
    """The stock handler refuses a cancel whose result is not ``canceled``."""
    from a2a.types import TaskIdParams, TaskNotCancelableError
    from a2a.utils.errors import ServerError

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("task-refuses-cancel", TaskState.input_required))
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(cancel_state=TaskState.completed),
        task_store=store,
        replica_id="replica-a",
    )

    with pytest.raises(ServerError) as caught:
        await handler.on_cancel_task(TaskIdParams(id="task-refuses-cancel"))

    assert isinstance(caught.value.error, TaskNotCancelableError)
    assert caught.value.error.message == (
        f"Task cannot be canceled - current state: {TaskState.completed}"
    )


async def test_a_wedged_cancel_does_not_stop_the_owner_serving_later_cancels(
    redis_client,
):
    """One executor cancel that never returns must not starve the listener."""
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("task-wedged", TaskState.input_required))
    await seed.save(_task("task-healthy", TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("task-wedged")
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        cancel_timeout_seconds=2,
    )
    await owner.start()
    # A requesting replica waits the same configured cancel timeout, so the
    # owner has to give up first for its refusal to be read at all.
    try:
        wedged = asyncio.create_task(
            requester.request_cancel(
                owner_replica_id="replica-owner",
                task_id="task-wedged",
                timeout_seconds=2,
            )
        )
        await asyncio.sleep(0.1)
        healthy = await requester.request_cancel(
            owner_replica_id="replica-owner",
            task_id="task-healthy",
            timeout_seconds=2,
        )
        with pytest.raises(A2ATaskStoreError) as refused:
            await wedged
    finally:
        await owner.close()

    assert healthy.status.state == TaskState.canceled
    assert (await seed.get("task-healthy")).status.state == TaskState.canceled
    assert str(refused.value) == (
        "owner replica-owner rejected cancel for task task-wedged: "
        "A2ACancelTimeoutError: cancel of task task-wedged did not finish "
        "within 1s"
    )
    assert (await seed.get("task-wedged")).status.state == TaskState.input_required
    assert await owner_store.get_execution_lease("task-wedged") is None


async def test_a_stale_failed_copy_is_refused_even_when_it_matches(redis_client):
    """Only a repeated ``canceled`` is acknowledged unwritten.

    An interrupted owner's own ``failed`` must not be answered as if it were
    the stored interruption the client would read.
    """
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner = await store.acquire_execution(
        "task-stale-failed", replica_id="replica-lost", lease_seconds=0.1
    )
    owner_context = _owned_context(store, owner)
    await store.save(_task("task-stale-failed", TaskState.working), owner_context)
    await asyncio.sleep(0.15)
    assert await store.mark_owner_lost("task-stale-failed") is True
    interrupted = await store.get("task-stale-failed")

    with pytest.raises(
        A2ATaskOwnershipLostError,
        match=(
            "save task task-stale-failed with stale ownership generation "
            f"{owner.generation}"
        ),
    ):
        await store.save(_task("task-stale-failed", TaskState.failed), owner_context)
    stored = await store.get("task-stale-failed")
    assert stored.model_dump(mode="json") == interrupted.model_dump(mode="json")


async def test_cancelling_a_non_blocking_send_leaves_a_closed_expiring_relay(
    redis_client,
):
    """The SDK closes a non-blocking send's relay once its producer stops.

    The cancel event has to be published before that close, never after it:
    an event appended after ``closed`` also PERSISTs the stream, so the relay
    of the commonest cancel would never expire.
    """
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler = RedisRequestHandler(
        agent_executor=executor, task_store=store, replica_id="replica-a"
    )
    sent = await handler.on_message_send(_send_params("non-blocking", blocking=False))
    assert sent.status.state == TaskState.working

    result = await handler.on_cancel_task(TaskIdParams(id=sent.id))
    await handler.close()

    assert result.status.state == TaskState.canceled
    assert (await store.get(sent.id)).status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]
    ttl = await redis_client.ttl(f"test:a2a:events:{sent.id}")
    assert 0 < ttl <= 60


@pytest.fixture(scope="module")
def redis_without_hash_field_expiry_url():
    """An owned Redis older than 7.4, which has no hash-field expiry."""
    port = _free_port()
    container_name = f"redis-a2a-7-2-{os.getpid()}-{uuid.uuid4().hex}"
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
            "redis:7.2-alpine",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis 7.2: {result.stderr}")
    try:
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
            pytest.fail("Redis 7.2 did not become ready within 30 seconds")
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


async def test_a_redis_without_hash_field_expiry_is_refused_at_connect(
    redis_without_hash_field_expiry_url,
):
    url = redis_without_hash_field_expiry_url
    with pytest.raises(A2ATaskStoreError) as refused:
        await RedisTaskStore.from_url(url, key_prefix="test:a2a")
    assert str(refused.value) == (
        f"shared A2A task store requires Redis >= 7.4 (HPEXPIRE); {url} does not "
        "support hash-field expiry"
    )

    probe = aioredis.from_url(url, decode_responses=True)
    try:
        assert await probe.keys("*") == []
    finally:
        await probe.aclose()


class _FinishesDuringCancelExecutor(AgentExecutor):
    """A non-cooperative executor whose turn completes while it is cancelled.

    ``cancel`` releases the running turn, which then publishes ``completed``
    and returns, and only after that turn has finished (or cannot finish)
    does ``cancel`` answer ``canceled``.
    """

    def __init__(self) -> None:
        self.handler: RedisRequestHandler | None = None
        self.working = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=False,
                status=TaskStatus(state=TaskState.working),
            )
        )
        self.working.set()
        await self.release.wait()
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=True,
                status=TaskStatus(state=TaskState.completed),
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        producer = self.handler._running_agents.get(context.task_id)
        self.release.set()
        if producer is not None:
            await asyncio.wait({producer}, timeout=1)
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=True,
                status=TaskStatus(state=TaskState.canceled),
            )
        )


async def test_a_turn_finishing_inside_the_cancel_leaves_the_relay_canceled(
    redis_client,
):
    """Once a cancel is committed the relay ends with it, not the late turn."""
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _FinishesDuringCancelExecutor()
    handler = RedisRequestHandler(
        agent_executor=executor, task_store=store, replica_id="replica-a"
    )
    executor.handler = handler
    sent = await handler.on_message_send(_send_params("finishes", blocking=False))
    assert sent.status.state == TaskState.working

    result = await handler.on_cancel_task(TaskIdParams(id=sent.id))
    await handler.close()

    assert result.status.state == TaskState.canceled
    assert (await store.get(sent.id)).status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]
    ttl = await redis_client.ttl(f"test:a2a:events:{sent.id}")
    assert 0 < ttl <= 60


async def test_wedged_cancels_do_not_delay_a_healthy_one_past_its_deadline(
    redis_client,
):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    for task_id in ("task-wedged-1", "task-wedged-2", "task-healthy"):
        await seed.save(_task(task_id, TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.update({"task-wedged-1", "task-wedged-2"})
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        cancel_timeout_seconds=2,
        max_concurrent_cancels=3,
    )
    await owner.start()
    try:
        wedged = [
            asyncio.create_task(
                requester.request_cancel(
                    owner_replica_id="replica-owner",
                    task_id=task_id,
                    timeout_seconds=2,
                )
            )
            for task_id in ("task-wedged-1", "task-wedged-2")
        ]
        await asyncio.sleep(0.1)
        started = time.monotonic()
        healthy = await requester.request_cancel(
            owner_replica_id="replica-owner",
            task_id="task-healthy",
            timeout_seconds=2,
        )
        healthy_seconds = time.monotonic() - started
        refusals = await asyncio.gather(*wedged, return_exceptions=True)
    finally:
        await owner.close()

    assert healthy.status.state == TaskState.canceled
    assert healthy_seconds < 0.5
    assert [str(refusal) for refusal in refusals] == [
        f"owner replica-owner rejected cancel for task {task_id}: "
        f"A2ACancelTimeoutError: cancel of task {task_id} did not finish within 1s"
        for task_id in ("task-wedged-1", "task-wedged-2")
    ]


async def test_a_full_cancel_listener_refuses_instead_of_queueing(redis_client, caplog):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    for task_id in ("task-wedged-1", "task-wedged-2", "task-overflow"):
        await seed.save(_task(task_id, TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.update({"task-wedged-1", "task-wedged-2"})
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        cancel_timeout_seconds=2,
        max_concurrent_cancels=2,
    )
    await owner.start()
    try:
        wedged = [
            asyncio.create_task(
                requester.request_cancel(
                    owner_replica_id="replica-owner",
                    task_id=task_id,
                    timeout_seconds=2,
                )
            )
            for task_id in ("task-wedged-1", "task-wedged-2")
        ]
        await asyncio.sleep(0.1)
        started = time.monotonic()
        with pytest.raises(A2ATaskStoreError) as refused:
            await requester.request_cancel(
                owner_replica_id="replica-owner",
                task_id="task-overflow",
                timeout_seconds=2,
            )
        refused_seconds = time.monotonic() - started
        await asyncio.gather(*wedged, return_exceptions=True)
    finally:
        await owner.close()

    assert refused_seconds < 0.5
    refusal_logs = [
        record
        for record in caplog.records
        if record.name == "cogniverse_runtime.a2a_request_handler"
        and "task-overflow" in record.getMessage()
    ]
    assert [(r.levelname, r.exc_info) for r in refusal_logs] == [("WARNING", None)]
    assert str(refused.value) == (
        "owner replica-owner rejected cancel for task task-overflow: "
        "A2ACancelCapacityError: replica replica-owner is already running 2 "
        "cancels; retry"
    )
    assert (await seed.get("task-overflow")).status.state == TaskState.input_required


@pytest.fixture
def restricted_redis_url():
    """An owned Redis 7.4 a test may restrict (ACL user, read-only replica)."""
    port = _free_port()
    container_name = f"redis-a2a-restricted-{os.getpid()}-{uuid.uuid4().hex}"
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
    try:
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
            pytest.fail("Redis did not become ready within 30 seconds")
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


async def test_a_user_not_permitted_hash_field_expiry_is_refused_at_connect(
    restricted_redis_url,
):
    admin = aioredis.from_url(restricted_redis_url, decode_responses=True)
    try:
        await admin.execute_command(
            "ACL",
            "SETUSER",
            "a2a-limited",
            "on",
            "nopass",
            "~*",
            "&*",
            "+@all",
            "-hpexpire",
        )
    finally:
        await admin.aclose()
    url = restricted_redis_url.replace("redis://", "redis://a2a-limited:unused@")

    with pytest.raises(A2ATaskStoreError) as refused:
        await RedisTaskStore.from_url(url, key_prefix="test:a2a")

    assert str(refused.value) == (
        f"shared A2A task store cannot use {url}: its Redis user is not "
        "permitted HPEXPIRE"
    )


async def test_a_read_only_replica_is_refused_at_connect(restricted_redis_url):
    admin = aioredis.from_url(restricted_redis_url, decode_responses=True)
    try:
        await admin.execute_command("REPLICAOF", "127.0.0.1", "1")
    finally:
        await admin.aclose()

    with pytest.raises(A2ATaskStoreError) as refused:
        await RedisTaskStore.from_url(restricted_redis_url, key_prefix="test:a2a")

    assert str(refused.value) == (
        f"shared A2A task store cannot use {restricted_redis_url}: it is a "
        "read-only replica"
    )


async def _running_blocking_send(handler, store, executor):
    send = asyncio.create_task(handler.on_message_send(_send_params("dbl")))
    await asyncio.wait_for(executor.working.wait(), timeout=5)
    task_id = next(iter(handler._running_agents))
    async with asyncio.timeout(5):
        while await store.get(task_id) is None:
            await asyncio.sleep(0.01)
    return send, task_id


def _states(replies) -> list:
    return [getattr(getattr(r, "status", None), "state", r) for r in replies]


async def test_two_peers_cancelling_one_task_at_once_publish_one_cancel(
    redis_client,
):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    peers = [
        RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
        for _ in range(2)
    ]
    executor = _FinishesDuringCancelExecutor()
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-owner",
        cancel_timeout_seconds=4,
    )
    executor.handler = owner
    await owner.start()
    send, task_id = await _running_blocking_send(owner, store, executor)

    replies = await asyncio.gather(
        *(
            peer.request_cancel(
                owner_replica_id="replica-owner", task_id=task_id, timeout_seconds=4
            )
            for peer in peers
        ),
        return_exceptions=True,
    )
    sent = await asyncio.wait_for(send, timeout=5)
    await asyncio.wait_for(owner.close(), timeout=10)

    assert _states(replies) == [TaskState.canceled, TaskState.canceled]
    assert sent.status.state == TaskState.canceled
    assert (await store.get(task_id)).status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{task_id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]


async def test_a_cancel_arriving_after_the_first_is_saved_is_not_republished(
    redis_client,
):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    peers = [
        RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
        for _ in range(2)
    ]
    executor = _FinishesDuringCancelExecutor()
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-owner",
        cancel_timeout_seconds=4,
    )
    executor.handler = owner
    await owner.start()
    send, task_id = await _running_blocking_send(owner, store, executor)

    first = asyncio.create_task(
        peers[0].request_cancel(
            owner_replica_id="replica-owner", task_id=task_id, timeout_seconds=4
        )
    )
    async with asyncio.timeout(5):
        while (await store.get(task_id)).status.state != TaskState.canceled:
            await asyncio.sleep(0.005)
    second = await asyncio.gather(
        peers[1].request_cancel(
            owner_replica_id="replica-owner", task_id=task_id, timeout_seconds=4
        ),
        return_exceptions=True,
    )
    replies = [await asyncio.gather(first, return_exceptions=True), second]
    sent = await asyncio.wait_for(send, timeout=5)
    await asyncio.wait_for(owner.close(), timeout=10)

    assert _states([reply[0] for reply in replies]) == [
        TaskState.canceled,
        TaskState.canceled,
    ]
    assert sent.status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{task_id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]


async def test_a_local_and_a_routed_cancel_of_one_task_share_one_cancel(
    redis_client,
):
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    peer = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _FinishesDuringCancelExecutor()
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-owner",
        cancel_timeout_seconds=4,
    )
    executor.handler = owner
    await owner.start()
    send, task_id = await _running_blocking_send(owner, store, executor)

    replies = await asyncio.gather(
        owner.on_cancel_task(TaskIdParams(id=task_id)),
        peer.request_cancel(
            owner_replica_id="replica-owner", task_id=task_id, timeout_seconds=4
        ),
        return_exceptions=True,
    )
    sent = await asyncio.wait_for(send, timeout=5)
    await asyncio.wait_for(owner.close(), timeout=10)

    assert _states(replies) == [TaskState.canceled, TaskState.canceled]
    assert sent.status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{task_id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]


async def test_a_wedged_local_cancel_is_bounded_and_does_not_hang_close(
    redis_client,
):
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    executor.wedge_every_cancel = True
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        cancel_timeout_seconds=1,
        drain_timeout_seconds=1,
    )
    sent = await handler.on_message_send(_send_params("wedged", blocking=False))

    started = time.monotonic()
    with pytest.raises(ServerError) as refused:
        await asyncio.wait_for(
            handler.on_cancel_task(TaskIdParams(id=sent.id)), timeout=5
        )
    cancel_seconds = time.monotonic() - started
    await asyncio.wait_for(handler.close(), timeout=5)

    # The retryable conflict the routed path answers, not a JSON-RPC
    # internal error.
    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        f"cancel of task {sent.id} did not finish within 0.5s; retry"
    )
    assert cancel_seconds < 1.5
    assert (await store.get(sent.id)).status.state == TaskState.working
    relay = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    assert _relay_states(relay) == [TaskState.working, "closed"]


async def test_a_producer_event_during_begin_cancel_never_reaches_the_relay(
    redis_client,
):
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler = RedisRequestHandler(
        agent_executor=executor, task_store=store, replica_id="replica-a"
    )
    sent = await handler.on_message_send(_send_params("window", blocking=False))
    real_begin_cancel = store.begin_cancel

    async def begin_cancel_then_producer_emits(task_id, **kwargs):
        lease = await real_begin_cancel(task_id, **kwargs)
        # The producer emits between the cancel taking its generation and the
        # cancel running.
        live = await handler._queue_manager.get(task_id)
        await live.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id="context-cancel",
                final=False,
                status=TaskStatus(state=TaskState.working, message=None),
                metadata={"emitted": "inside-begin-cancel"},
            )
        )
        return lease

    store.begin_cancel = begin_cancel_then_producer_emits

    result = await asyncio.wait_for(
        handler.on_cancel_task(TaskIdParams(id=sent.id)), timeout=10
    )
    # The cancel is published before on_cancel_task returns; whether the
    # consumer has closed the relay yet is a race, so check what precedes it.
    published = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    assert _relay_states(published)[:2] == [TaskState.working, TaskState.canceled]
    assert "inside-begin-cancel" not in str(published)
    await asyncio.wait_for(handler.close(), timeout=10)

    assert result.status.state == TaskState.canceled
    relay = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]


async def test_abandoned_cancels_count_against_the_cancel_limit(redis_client):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    for task_id in ("task-stubborn-1", "task-stubborn-2", "task-next"):
        await seed.save(_task(task_id, TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.update({"task-stubborn-1", "task-stubborn-2"})
    executor.stubborn = True
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        cancel_timeout_seconds=1,
        max_concurrent_cancels=2,
    )
    await owner.start()
    try:
        abandoned = await asyncio.gather(
            *(
                requester.request_cancel(
                    owner_replica_id="replica-owner",
                    task_id=task_id,
                    timeout_seconds=1,
                )
                for task_id in ("task-stubborn-1", "task-stubborn-2")
            ),
            return_exceptions=True,
        )
        assert len(owner._abandoned_cancels) == 2
        with pytest.raises(A2ATaskStoreError) as refused:
            await requester.request_cancel(
                owner_replica_id="replica-owner", task_id="task-next", timeout_seconds=1
            )
    finally:
        executor.unwedge.set()
        await owner.close()

    assert [type(reply).__name__ for reply in abandoned] == [
        "A2ATaskStoreError",
        "A2ATaskStoreError",
    ]
    assert str(refused.value) == (
        "owner replica-owner rejected cancel for task task-next: "
        "A2ACancelCapacityError: replica replica-owner is already running 2 "
        "cancels; retry"
    )
    assert (await seed.get("task-next")).status.state == TaskState.input_required


def _owned_container(image: str, name_prefix: str):
    port = _free_port()
    container_name = f"{name_prefix}-{os.getpid()}-{uuid.uuid4().hex}"
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
            image,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start {image}: {result.stderr}")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", container_name, "valkey-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            return container_name, f"redis://127.0.0.1:{port}/0"
        time.sleep(0.25)
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    pytest.fail(f"{image} did not become ready within 30 seconds")


@pytest.fixture
def valkey_url(request):
    container_name, url = _owned_container(request.param, "valkey-a2a")
    try:
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


@pytest.mark.parametrize("valkey_url", ["valkey/valkey:9.0-alpine"], indirect=True)
async def test_valkey_with_hash_field_expiry_is_admitted(valkey_url):
    store = await RedisTaskStore.from_url(valkey_url, key_prefix="test:a2a")
    try:
        lease = await store.acquire_execution(
            "valkey-task", replica_id="replica-a", lease_seconds=2
        )
        saved = _task("valkey-task")
        await store.save(saved, _owned_context(store, lease))
        await store.release_execution(lease)
        loaded = await store.get("valkey-task")
    finally:
        await store.close()
    assert loaded.model_dump(mode="json") == saved.model_dump(mode="json")


@pytest.mark.parametrize("valkey_url", ["valkey/valkey:8.1-alpine"], indirect=True)
async def test_valkey_without_hash_field_expiry_is_refused(valkey_url):
    with pytest.raises(A2ATaskStoreError) as refused:
        await RedisTaskStore.from_url(valkey_url, key_prefix="test:a2a")
    assert str(refused.value) == (
        "shared A2A task store requires Redis >= 7.4 (HPEXPIRE); "
        f"{valkey_url} does not support hash-field expiry"
    )


async def test_a_cancel_that_ignores_cancellation_holds_the_relay_close_boundedly(
    redis_client, caplog
):
    """An abandoned cancel that never ends must not hold the relay's close."""
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    executor.wedge_every_cancel = True
    executor.stubborn = True
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        cancel_timeout_seconds=1,
        drain_timeout_seconds=1,
    )
    sent = await handler.on_message_send(_send_params("stubborn", blocking=False))
    try:
        with pytest.raises(ServerError):
            await asyncio.wait_for(
                handler.on_cancel_task(TaskIdParams(id=sent.id)), timeout=5
            )
        started = time.monotonic()
        await asyncio.wait_for(handler.close(), timeout=10)
        close_seconds = time.monotonic() - started
    finally:
        executor.unwedge.set()

    assert close_seconds < 4
    relay = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    assert _relay_states(relay) == [TaskState.working, "closed"]
    assert 0 < await redis_client.ttl(f"test:a2a:events:{sent.id}") <= 60
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "committed cancel" in record.getMessage()
    ] == [
        f"A2A task {sent.id}: closing its relay after 0.5s without the committed cancel"
    ]


async def _handler_with_live_relay(redis_client, task_id: str):
    """A handler whose live relay for ``task_id`` has no consumer closing it.

    The relay stays open across cancels, the state a non-cooperative
    producer keeps publishing into.
    """
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task(task_id, TaskState.working))
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(),
        task_store=store,
        replica_id="replica-a",
    )
    relay = await handler._queue_manager.create_or_tap(task_id)
    await relay.enqueue_event(_status_event(task_id, TaskState.working))
    return store, handler, relay


def _status_event(task_id: str, state: TaskState, mark: str | None = None):
    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=f"context-{task_id}",
        final=state != TaskState.working,
        status=TaskStatus(state=state),
        metadata={"mark": mark} if mark else None,
    )


async def test_a_failed_later_cancel_does_not_reopen_a_canceled_relay(redis_client):
    store, handler, relay = await _handler_with_live_relay(redis_client, "reopen")
    canceled = await handler._cancel_owned("reopen")
    assert canceled.status.state == TaskState.canceled

    async def lost(*args, **kwargs):
        raise A2ATaskOwnershipLostError("cancel task reopen lost its owner")

    store.begin_cancel = lost
    with pytest.raises(A2ATaskOwnershipLostError):
        await handler._cancel_owned("reopen")
    # A non-cooperative producer still emitting after the published cancel.
    await relay.enqueue_event(_status_event("reopen", TaskState.completed, "late"))
    # Nothing consumes this relay locally, so drop its local backlog on close.
    await asyncio.wait_for(relay.close(immediate=True), timeout=5)

    entries = await redis_client.xrange("test:a2a:events:reopen")
    assert "late" not in str(entries)
    assert _relay_states(entries) == [TaskState.working, TaskState.canceled, "closed"]


async def test_events_during_a_failed_begin_cancel_are_delivered_not_lost(
    redis_client,
):
    store, handler, relay = await _handler_with_live_relay(redis_client, "kept")
    consumer = relay.tap()
    real_begin_cancel = store.begin_cancel

    async def producer_emits_then_begin_cancel_fails(task_id, **kwargs):
        await relay.enqueue_event(
            _status_event(task_id, TaskState.working, "during-begin-cancel")
        )
        raise A2ATaskOwnershipLostError(f"cancel task {task_id} lost its owner")

    store.begin_cancel = producer_emits_then_begin_cancel_fails
    with pytest.raises(A2ATaskOwnershipLostError):
        await handler._cancel_owned("kept")
    store.begin_cancel = real_begin_cancel
    # The task was not cancelled, so the producer's next event flows as usual.
    await relay.enqueue_event(_status_event("kept", TaskState.working, "after"))

    delivered = [
        (await consumer.dequeue_event(no_wait=True)).metadata for _ in range(2)
    ]
    entries = await redis_client.xrange("test:a2a:events:kept")
    marks = [
        TaskStatusUpdateEvent.model_validate_json(fields["payload"]).metadata
        for _, fields in entries
    ]
    await relay.close(immediate=True)

    assert delivered == [{"mark": "during-begin-cancel"}, {"mark": "after"}]
    assert marks == [None, {"mark": "during-begin-cancel"}, {"mark": "after"}]


async def test_a_full_local_cancel_is_a_retryable_conflict(redis_client):
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    for task_id in ("local-stubborn", "local-next"):
        await seed.save(_task(task_id, TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("local-stubborn")
    executor.stubborn = True
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        cancel_timeout_seconds=1,
        drain_timeout_seconds=1,
        max_concurrent_cancels=1,
    )
    try:
        with pytest.raises(ServerError):
            await handler.on_cancel_task(TaskIdParams(id="local-stubborn"))
        with pytest.raises(ServerError) as refused:
            await handler.on_cancel_task(TaskIdParams(id="local-next"))
    finally:
        executor.unwedge.set()
        await asyncio.wait_for(handler.close(), timeout=10)

    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        "replica replica-a is already running 1 cancels; retry"
    )
    assert (await seed.get("local-next")).status.state == TaskState.input_required


async def test_close_cancels_cancel_runs_that_outlast_the_drain(redis_client):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("drain-wedged", TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("drain-wedged")
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        cancel_timeout_seconds=4,
        drain_timeout_seconds=0.2,
    )
    await owner.start()
    routed = asyncio.create_task(
        requester.request_cancel(
            owner_replica_id="replica-owner",
            task_id="drain-wedged",
            timeout_seconds=4,
        )
    )
    async with asyncio.timeout(5):
        while "drain-wedged" not in owner._inflight_cancels:
            await asyncio.sleep(0.01)
    run = owner._inflight_cancels["drain-wedged"]

    started = time.monotonic()
    await asyncio.wait_for(owner.close(), timeout=5)
    close_seconds = time.monotonic() - started
    with pytest.raises(A2ATaskStoreError) as refused:
        await routed

    assert close_seconds < 1.5
    assert run.cancelled()
    assert owner._inflight_cancels == {}
    assert str(refused.value) == (
        "owner replica-owner rejected cancel for task drain-wedged: "
        "A2ACancelTimeoutError: replica replica-owner shut down before the "
        "cancel of task drain-wedged finished"
    )


async def test_a_cancel_limit_below_one_is_refused_before_redis_is_touched():
    class Registry:
        def list_agents(self):
            raise AssertionError("registry accessed before the limit was validated")

    closed_port = _free_port()
    with pytest.raises(ValueError) as refused:
        await _build_shared_a2a_protocol(
            agent_registry=Registry(),
            dispatcher=SimpleNamespace(),
            redis_url=f"redis://127.0.0.1:{closed_port}/0",
            replica_id="factory-replica",
            max_tasks=7,
            lease_seconds=1,
            cancel_timeout_seconds=2,
            drain_timeout_seconds=1,
            max_concurrent_cancels=0,
        )
    assert str(refused.value) == (
        "A2A_MAX_CONCURRENT_CANCELS (max_concurrent_cancels) must be >= 1, got 0"
    )


async def _abort_whose_flush_fails_to_publish(store, handler, relay, task_id):
    """Hold two producer events, then fail begin_cancel while Redis refuses
    to publish, so the abort's flush of those events fails."""
    real_publish = store.publish_event
    refusing = {"on": False}

    async def publish_event(task, event):
        if refusing["on"]:
            raise A2ATaskStoreError(
                f"shared A2A task store unavailable: publish event for task {task}"
            )
        await real_publish(task, event)

    store.publish_event = publish_event
    real_begin_cancel = store.begin_cancel

    async def producer_emits_then_begin_cancel_fails(task, **kwargs):
        for mark in ("held-1", "held-2"):
            await relay.enqueue_event(_status_event(task, TaskState.working, mark))
        refusing["on"] = True
        raise A2ATaskOwnershipLostError(f"cancel task {task} lost its owner")

    store.begin_cancel = producer_emits_then_begin_cancel_fails
    try:
        with pytest.raises(A2ATaskStoreError) as refused:
            await handler._cancel_owned(task_id)
    finally:
        refusing["on"] = False
        store.begin_cancel = real_begin_cancel
    return refused.value


async def test_a_failed_abort_flush_leaves_the_relay_open_and_loud(
    redis_client, caplog
):
    store, handler, relay = await _handler_with_live_relay(redis_client, "flush")
    consumer = relay.tap()

    refused = await _abort_whose_flush_fails_to_publish(store, handler, relay, "flush")
    await relay.enqueue_event(_status_event("flush", TaskState.working, "later"))

    delivered = [
        (await consumer.dequeue_event(no_wait=True)).metadata for _ in range(3)
    ]
    started = time.monotonic()
    await asyncio.wait_for(relay.close(immediate=True), timeout=10)
    close_seconds = time.monotonic() - started
    entries = await redis_client.xrange("test:a2a:events:flush")

    # The failure reaches the canceller; nothing held is lost locally.
    assert str(refused) == (
        "shared A2A task store unavailable: publish event for task flush"
    )
    assert delivered == [{"mark": "held-1"}, {"mark": "held-2"}, {"mark": "later"}]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and "relay publish" in record.getMessage()
    ] == [
        "A2A task flush: 2 events held during a failed cancel reached local "
        "consumers but not the relay publish"
    ]
    # The relay is open again: later events publish and the close is not held.
    assert close_seconds < 1
    assert "later" in str(entries)
    assert _relay_states(entries) == [TaskState.working, TaskState.working, "closed"]


async def test_a_cancel_after_a_failed_abort_flush_holds_and_publishes(
    redis_client,
):
    store, handler, relay = await _handler_with_live_relay(redis_client, "again")
    await _abort_whose_flush_fails_to_publish(store, handler, relay, "again")

    canceled = await asyncio.wait_for(handler._cancel_owned("again"), timeout=10)
    # A cancel that could hold the relay releases it when done; a relay the
    # failed abort left holding would make this close wait out the 5s hold.
    started = time.monotonic()
    await asyncio.wait_for(relay.close(immediate=True), timeout=10)
    close_seconds = time.monotonic() - started

    assert canceled.status.state == TaskState.canceled
    assert close_seconds < 1
    entries = await redis_client.xrange("test:a2a:events:again")
    assert _relay_states(entries) == [TaskState.working, TaskState.canceled, "closed"]
