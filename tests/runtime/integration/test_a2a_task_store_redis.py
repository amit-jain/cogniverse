"""Real-Redis contract for the shared A2A task store."""

from __future__ import annotations

import asyncio
import json
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
    A2ACancelTimeoutError,
    A2ATaskCapacityError,
    A2ATaskConflictError,
    A2ATaskOwnershipLostError,
    A2ATaskStoreError,
    RedisTaskStore,
)
from cogniverse_runtime.main import _a2a_settings_from_env, _build_shared_a2a_protocol

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _a2a_settings(**overrides):
    """Production A2A settings with a test's overrides."""
    return {**_a2a_settings_from_env({}), **overrides}


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


def _rendered_chain(exc: BaseException) -> str:
    """Every message and traceback line a logger prints for ``exc``."""
    import traceback

    return "".join(traceback.format_exception(exc))


@pytest.mark.parametrize(
    "credentials,query",
    [
        ("a2a-user:a2a-secret-pw@", ""),
        (":a2a-secret-pw@", ""),
        ("", "?password=a2a-secret-pw"),
    ],
)
async def test_an_unreachable_redis_is_named_without_its_credentials(
    credentials, query
):
    closed_port = _free_port()
    url = f"redis://{credentials}127.0.0.1:{closed_port}/0{query}"

    with pytest.raises(A2ATaskStoreError) as refused:
        await RedisTaskStore.from_url(url, key_prefix="test:a2a")

    assert str(refused.value) == (
        "shared A2A task store unavailable: connect to "
        f"redis://127.0.0.1:{closed_port}/0"
    )
    assert "a2a-secret-pw" not in _rendered_chain(refused.value)


async def test_a_wrong_password_is_refused_without_echoing_it(redis_url):
    admin = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await admin.execute_command(
            "ACL", "SETUSER", "a2a-authed", "on", ">right-pw", "~*", "&*", "+@all"
        )
    finally:
        await admin.aclose()
    url = redis_url.replace("redis://", "redis://a2a-authed:a2a-secret-pw@")

    try:
        with pytest.raises(A2ATaskStoreError) as refused:
            await RedisTaskStore.from_url(url, key_prefix="test:a2a")
    finally:
        admin = aioredis.from_url(redis_url, decode_responses=True)
        try:
            await admin.execute_command("ACL", "DELUSER", "a2a-authed")
        finally:
            await admin.aclose()

    assert str(refused.value) == (
        f"shared A2A task store unavailable: connect to {redis_url}"
    )
    assert "a2a-secret-pw" not in _rendered_chain(refused.value)


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
        **_a2a_settings(
            max_tasks=7,
            lease_seconds=1,
            cancel_timeout_seconds=2,
            drain_timeout_seconds=1,
            max_concurrent_cancels=5,
            max_concurrent_resubscriptions=3,
            redis_timeout_seconds=2.5,
            redis_max_connections=9,
        ),
    )
    try:
        assert protocol.skill_ids == ("search_agent",)
        assert protocol.handler.task_store is protocol.task_store
        assert protocol.handler._max_concurrent_cancels == 5
        assert protocol.handler._max_concurrent_resubscriptions == 3
        pool = protocol.task_store._redis.connection_pool
        assert (pool.max_connections, pool.timeout) == (9, 2.5)
        assert {
            key: pool.connection_kwargs[key]
            for key in (
                "socket_timeout",
                "socket_connect_timeout",
                "socket_keepalive",
                "health_check_interval",
            )
        } == {
            "socket_timeout": 2.5,
            "socket_connect_timeout": 2.5,
            "socket_keepalive": True,
            "health_check_interval": 30,
        }
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
            **_a2a_settings(
                max_tasks=7,
                lease_seconds=1,
                cancel_timeout_seconds=2,
                drain_timeout_seconds=1,
            ),
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
        **_a2a_settings(
            max_tasks=7,
            lease_seconds=0.2,
            cancel_timeout_seconds=1,
            drain_timeout_seconds=0.2,
        ),
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
        # A wedged cancel that is cancelled takes this long to stop.
        self.stop_seconds = 0.0
        self.unwedge = asyncio.Event()
        self.working = asyncio.Event()
        self.cancelling = asyncio.Event()

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
            self.cancelling.set()
            while not self.unwedge.is_set():
                try:
                    await self.unwedge.wait()
                except asyncio.CancelledError:
                    if not self.stubborn:
                        await asyncio.sleep(self.stop_seconds)
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
        else f"missing:{fields['missing']}"
        if "missing" in fields
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
    url = restricted_redis_url.replace("redis://", "redis://a2a-limited:a2a-secret-pw@")

    with pytest.raises(A2ATaskStoreError) as refused:
        await RedisTaskStore.from_url(url, key_prefix="test:a2a")

    assert str(refused.value) == (
        f"shared A2A task store cannot use {restricted_redis_url}: its Redis "
        "user is not permitted HPEXPIRE"
    )
    assert "a2a-secret-pw" not in _rendered_chain(refused.value)


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
    producer keeps publishing into. It belongs to the handler's execution
    lease, as a served execution's relay does.
    """
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task(task_id, TaskState.working))
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(),
        task_store=store,
        replica_id="replica-a",
    )
    lease = await store.acquire_execution(
        task_id, replica_id="replica-a", lease_seconds=30
    )
    relay = await handler._queue_manager.create_or_tap(
        task_id, generation=lease.generation
    )
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
            **_a2a_settings(
                max_tasks=7,
                lease_seconds=1,
                cancel_timeout_seconds=2,
                drain_timeout_seconds=1,
                max_concurrent_cancels=0,
            ),
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

    # The canceller gets the cancel's own failure, not the flush's; nothing
    # held is lost locally.
    assert type(refused) is A2ATaskOwnershipLostError
    assert str(refused) == "cancel task flush lost its owner"
    assert delivered == [{"mark": "held-1"}, {"mark": "held-2"}, {"mark": "later"}]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and "relay publish" in record.getMessage()
    ] == [
        "A2A task flush: of the events held during a failed cancel, 2 reached "
        "local consumers but not the relay publish and 0 were dropped by an "
        "interrupted release (first publish failure: shared A2A task store "
        "unavailable: publish event for task flush)"
    ]
    # The relay records the gap, then is open again: later events publish and
    # the close is not held.
    assert close_seconds < 1
    assert "later" in str(entries)
    assert _relay_states(entries) == [
        TaskState.working,
        "missing:2",
        TaskState.working,
        "closed",
    ]


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
    assert _relay_states(entries) == [
        TaskState.working,
        "missing:2",
        TaskState.canceled,
        "closed",
    ]


def _cancel_run_named(name: str) -> asyncio.Task:
    [run] = [task for task in asyncio.all_tasks() if task.get_name() == name]
    return run


async def test_close_waits_for_the_cleanup_of_a_cancel_it_cut_short(redis_client):
    """A cancel run that close() cuts short at the drain budget still releases
    its cancel lease before close() returns."""
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("cut-short", TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("cut-short")
    executor.stop_seconds = 0.3
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        cancel_timeout_seconds=4,
        drain_timeout_seconds=1,
    )
    cancel = asyncio.create_task(handler.on_cancel_task(TaskIdParams(id="cut-short")))
    await asyncio.wait_for(executor.cancelling.wait(), timeout=5)
    owner_cancel = _cancel_run_named("a2a-owner-cancel:cut-short")
    assert await store.get_execution_lease("cut-short") is not None

    started = time.monotonic()
    await asyncio.wait_for(handler.close(), timeout=5)
    close_seconds = time.monotonic() - started
    await asyncio.gather(cancel, return_exceptions=True)

    assert owner_cancel.done()
    assert handler._abandoned_cancels == set()
    assert await store.get_execution_lease("cut-short") is None
    assert close_seconds < 1.5


async def test_close_bounds_its_wait_for_a_cut_short_cancel_that_never_stops(
    redis_client,
):
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("never-stops", TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("never-stops")
    executor.stubborn = True
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        cancel_timeout_seconds=4,
        drain_timeout_seconds=0.3,
    )
    cancel = asyncio.create_task(handler.on_cancel_task(TaskIdParams(id="never-stops")))
    await asyncio.wait_for(executor.cancelling.wait(), timeout=5)
    owner_cancel = _cancel_run_named("a2a-owner-cancel:never-stops")
    try:
        started = time.monotonic()
        await asyncio.wait_for(handler.close(), timeout=5)
        close_seconds = time.monotonic() - started
        tracked = set(handler._abandoned_cancels)
    finally:
        executor.unwedge.set()
        await asyncio.gather(cancel, owner_cancel, return_exceptions=True)

    assert tracked == {owner_cancel}
    assert close_seconds < 1.5


async def test_a_local_cancel_cut_short_by_shutdown_is_refused_not_cancelled(
    redis_client,
):
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("shutdown", TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("shutdown")
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        cancel_timeout_seconds=4,
        drain_timeout_seconds=0.2,
    )
    cancel = asyncio.create_task(handler.on_cancel_task(TaskIdParams(id="shutdown")))
    await asyncio.wait_for(executor.cancelling.wait(), timeout=5)

    await asyncio.wait_for(handler.close(), timeout=5)
    with pytest.raises(ServerError) as refused:
        await asyncio.wait_for(cancel, timeout=5)

    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        "replica replica-a shut down before the cancel of task shutdown finished; retry"
    )
    assert (await seed.get("shutdown")).status.state == TaskState.input_required


async def test_a_routed_cancel_the_owner_never_answers_is_a_retryable_conflict(
    redis_client,
):
    from a2a.types import TaskIdParams

    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        drain_timeout_seconds=1,
    )
    requester = RedisRequestHandler(
        agent_executor=_CancellableExecutor(),
        task_store=requester_store,
        replica_id="replica-requester",
        cancel_timeout_seconds=1,
    )
    # The owner runs the task but never listens for routed cancels.
    sent = await owner.on_message_send(_send_params("unanswered", blocking=False))
    await asyncio.wait_for(executor.working.wait(), timeout=5)
    try:
        with pytest.raises(ServerError) as refused:
            await asyncio.wait_for(
                requester.on_cancel_task(TaskIdParams(id=sent.id)), timeout=5
            )
    finally:
        await asyncio.wait_for(owner.close(), timeout=10)

    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        f"owner replica-owner did not acknowledge cancel for task {sent.id} "
        "within 1s; retry"
    )


async def test_a_routed_cancel_the_owner_refuses_is_a_retryable_conflict(
    redis_client,
):
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    owner_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    requester_store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("occupying", TaskState.input_required))
    executor = _CancellableExecutor()
    executor.wedged_cancels.add("occupying")
    executor.stubborn = True
    owner = RedisRequestHandler(
        agent_executor=executor,
        task_store=owner_store,
        replica_id="replica-owner",
        cancel_timeout_seconds=1,
        drain_timeout_seconds=1,
        max_concurrent_cancels=1,
    )
    requester = RedisRequestHandler(
        agent_executor=_CancellableExecutor(),
        task_store=requester_store,
        replica_id="replica-requester",
        cancel_timeout_seconds=4,
    )
    await owner.start()
    sent = await owner.on_message_send(_send_params("refused", blocking=False))
    await asyncio.wait_for(executor.working.wait(), timeout=5)
    try:
        # A stubborn local cancel outlives its deadline and holds the only slot.
        with pytest.raises(ServerError):
            await owner.on_cancel_task(TaskIdParams(id="occupying"))
        assert len(owner._abandoned_cancels) == 1
        with pytest.raises(ServerError) as refused:
            await asyncio.wait_for(
                requester.on_cancel_task(TaskIdParams(id=sent.id)), timeout=5
            )
    finally:
        executor.unwedge.set()
        await asyncio.wait_for(owner.close(), timeout=10)

    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        f"owner replica-owner rejected cancel for task {sent.id}: "
        "A2ACancelCapacityError: replica replica-owner is already running 1 "
        "cancels; retry"
    )


async def test_a_failed_flush_does_not_replace_the_cancels_own_conflict(
    redis_client,
):
    from a2a.types import TaskIdParams

    store, handler, relay = await _handler_with_live_relay(redis_client, "own")
    real_publish = store.publish_event
    refusing = {"on": False}

    async def publish_event(task, event):
        if refusing["on"]:
            raise A2ATaskStoreError(
                f"shared A2A task store unavailable: publish event for task {task}"
            )
        await real_publish(task, event)

    async def producer_emits_then_begin_cancel_fails(task, **kwargs):
        await relay.enqueue_event(_status_event(task, TaskState.working, "held"))
        refusing["on"] = True
        raise A2ATaskOwnershipLostError(f"cancel task {task} lost its owner")

    store.publish_event = publish_event
    store.begin_cancel = producer_emits_then_begin_cancel_fails
    try:
        with pytest.raises(ServerError) as refused:
            await handler.on_cancel_task(TaskIdParams(id="own"))
    finally:
        refusing["on"] = False
        await relay.close(immediate=True)

    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        "Task own is active on another replica; retry"
    )


async def test_a_store_outage_during_a_local_cancel_is_a_retryable_conflict(
    redis_client,
):
    from a2a.types import TaskIdParams

    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("outage", TaskState.input_required))
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(), task_store=store, replica_id="replica-a"
    )

    async def unavailable(task_id, **kwargs):
        raise A2ATaskStoreError(
            f"shared A2A task store unavailable: begin cancel for task {task_id}"
        )

    store.begin_cancel = unavailable
    with pytest.raises(ServerError) as refused:
        await handler.on_cancel_task(TaskIdParams(id="outage"))

    assert isinstance(refused.value.error, InvalidParamsError)
    assert refused.value.error.message == (
        "shared A2A task store unavailable: begin cancel for task outage; retry"
    )


@pytest.mark.parametrize("cancel_begins", [True, False])
async def test_a_chatty_producer_waits_while_a_cancel_holds_its_relay(
    redis_client, monkeypatch, cancel_begins
):
    """The relay holds at most its limit of producer events while a cancel
    takes its generation; a producer emitting more waits, then its events are
    dropped behind a committed cancel or delivered in order after an aborted
    one."""
    from cogniverse_runtime import a2a_request_handler

    monkeypatch.setattr(a2a_request_handler, "_MAX_HELD_EVENTS", 3)
    store, handler, relay = await _handler_with_live_relay(redis_client, "chatty")
    consumer = relay.tap()
    real_begin_cancel = store.begin_cancel
    emitted: list[int] = []
    observed: dict = {}

    async def chatty():
        for index in range(10):
            await relay.enqueue_event(
                _status_event("chatty", TaskState.working, f"chatty-{index}")
            )
            emitted.append(index)

    async def slow_begin_cancel(task_id, **kwargs):
        producer = asyncio.create_task(chatty())
        observed["producer"] = producer
        await asyncio.sleep(0.3)
        observed["held"] = len(relay._held)
        observed["emitted"] = list(emitted)
        if not cancel_begins:
            raise A2ATaskOwnershipLostError(f"cancel task {task_id} lost its owner")
        return await real_begin_cancel(task_id, **kwargs)

    store.begin_cancel = slow_begin_cancel
    if cancel_begins:
        await handler._cancel_owned("chatty")
    else:
        with pytest.raises(A2ATaskOwnershipLostError):
            await handler._cancel_owned("chatty")
    await asyncio.wait_for(observed["producer"], timeout=5)
    delivered = []
    while not consumer.queue.empty():
        delivered.append(await consumer.dequeue_event(no_wait=True))
    entries = await redis_client.xrange("test:a2a:events:chatty")
    await relay.close(immediate=True)

    assert observed["held"] == 3
    assert observed["emitted"] == [0, 1, 2]
    assert emitted == list(range(10))
    marks = [f"chatty-{index}" for index in range(10)]
    if cancel_begins:
        assert _relay_states(entries) == [TaskState.working, TaskState.canceled]
        assert not any(mark in str(entries) for mark in marks)
        assert [event.status.state for event in delivered] == [TaskState.canceled]
    else:
        assert [
            (TaskStatusUpdateEvent.model_validate_json(fields["payload"]).metadata)
            for _, fields in entries
        ] == [None, *({"mark": mark} for mark in marks)]
        assert [event.metadata for event in delivered] == [
            {"mark": mark} for mark in marks
        ]


async def _live_relay_on(client, redis_client, task_id: str):
    """A live relay for ``task_id`` on ``client``, with a live execution lease
    so a resubscriber waits on it instead of ending."""
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task(task_id, TaskState.working))
    lease = await store.acquire_execution(
        task_id, replica_id="replica-a", lease_seconds=30
    )
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(), task_store=store, replica_id="replica-a"
    )
    relay = await handler._queue_manager.create_or_tap(
        task_id, generation=lease.generation
    )
    await relay.enqueue_event(_status_event(task_id, TaskState.working))
    return store, handler, relay, lease


def _producer_emits_then_redis_pauses(relay, redis_client, count: int, pause_ms: int):
    async def begin_cancel(task_id, **kwargs):
        for index in range(count):
            await relay.enqueue_event(
                _status_event(task_id, TaskState.working, f"held-{index}")
            )
        await redis_client.execute_command("CLIENT", "PAUSE", str(pause_ms), "WRITE")
        raise A2ATaskOwnershipLostError(f"cancel task {task_id} lost its owner")

    return begin_cancel


async def test_a_flush_into_a_hung_redis_stops_publishing_and_marks_the_gap(
    redis_url, redis_client, caplog
):
    """Redis hangs while a failed cancel releases eight held events: the
    release gives up on the relay after one publish timeout, every event still
    reaches local consumers, and a resubscriber reading past the gap fails on
    the marker the next publish writes."""
    client = aioredis.from_url(redis_url, decode_responses=True, socket_timeout=0.25)
    store, handler, relay, _lease = await _live_relay_on(client, redis_client, "gap")
    consumer = relay.tap()
    resubscriber = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    received = []

    async def resubscribe():
        async for event in resubscriber.subscribe_events("gap"):
            received.append(event.metadata)

    subscription = asyncio.create_task(resubscribe())
    # Emit until the resubscription, positioned after its start, reads one.
    readies = 0
    async with asyncio.timeout(5):
        while not received:
            await relay.enqueue_event(_status_event("gap", TaskState.working, "ready"))
            readies += 1
            await asyncio.sleep(0.05)
    seen_readies = len(received)
    store.begin_cancel = _producer_emits_then_redis_pauses(
        relay, redis_client, count=8, pause_ms=2000
    )
    try:
        started = time.monotonic()
        with pytest.raises(A2ATaskOwnershipLostError):
            await handler._cancel_owned("gap")
        release_seconds = time.monotonic() - started
    finally:
        async with asyncio.timeout(5):
            while True:
                try:
                    await client.set("test:a2a:probe", "1")
                    break
                except aioredis.RedisError:
                    await asyncio.sleep(0.1)
    await relay.enqueue_event(_status_event("gap", TaskState.working, "later"))
    with pytest.raises(A2ATaskStoreError) as gap:
        await asyncio.wait_for(subscription, timeout=5)
    delivered = []
    while not consumer.queue.empty():
        delivered.append((await consumer.dequeue_event(no_wait=True)).metadata)
    entries = await redis_client.xrange("test:a2a:events:gap")
    await relay.close(immediate=True)
    await client.aclose()

    assert release_seconds < 1.2
    assert delivered == [
        *({"mark": "ready"} for _ in range(readies)),
        *({"mark": f"held-{index}"} for index in range(8)),
        {"mark": "later"},
    ]
    assert str(gap.value) == (
        "event relay for task gap is missing 8 events its owner could not publish"
    )
    assert received == [{"mark": "ready"}] * seen_readies
    assert _relay_states(entries)[-2:] == ["missing:8", TaskState.working]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and "gap" in record.getMessage()
    ] == [
        "A2A task gap: of the events held during a failed cancel, 8 reached "
        "local consumers but not the relay publish and 0 were dropped by an "
        "interrupted release (first publish failure: shared A2A task store "
        "unavailable: publish event for task gap)",
        "A2A task gap: could not mark its relay as missing 8 events; nothing "
        "more is published until it is: shared A2A task store unavailable: "
        "mark event stream incomplete for task gap",
    ]


@pytest.mark.parametrize("publish_after", [False, True])
async def test_an_interrupted_release_logs_and_marks_what_it_did_not_publish(
    redis_url, redis_client, caplog, publish_after
):
    """A release cut short while its first publish hangs still records the
    gap: the next publish, or else the close marker, carries the count, and a
    resubscriber fails on it instead of reading a clean stream."""
    client = aioredis.from_url(redis_url, decode_responses=True)
    store, handler, relay, _lease = await _live_relay_on(
        client, redis_client, "interrupted"
    )
    consumer = relay.tap()
    resubscriber = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    received = []

    async def resubscribe():
        async for event in resubscriber.subscribe_events("interrupted"):
            received.append(event.metadata)

    subscription = asyncio.create_task(resubscribe())
    await asyncio.sleep(0.3)
    store.begin_cancel = _producer_emits_then_redis_pauses(
        relay, redis_client, count=8, pause_ms=1500
    )
    cancel = asyncio.create_task(handler._cancel_owned("interrupted"))
    # The first held event reaches the consumer, then its publish hangs.
    first = await asyncio.wait_for(consumer.dequeue_event(), timeout=5)
    await asyncio.sleep(0.1)
    cancel.cancel()
    await asyncio.gather(cancel, return_exceptions=True)
    await asyncio.sleep(1.5)
    if publish_after:
        await relay.enqueue_event(
            _status_event("interrupted", TaskState.working, "next")
        )
        following = await asyncio.wait_for(consumer.dequeue_event(), timeout=5)
        assert following.metadata == {"mark": "next"}
    await relay.close(immediate=True)
    with pytest.raises(A2ATaskStoreError) as gap:
        await asyncio.wait_for(subscription, timeout=5)
    entries = await redis_client.xrange("test:a2a:events:interrupted")
    await client.aclose()

    assert cancel.cancelled()
    assert first.metadata == {"mark": "held-0"}
    assert str(gap.value) == (
        "event relay for task interrupted is missing 8 events its owner could "
        "not publish"
    )
    # The cut-short publish of held-0 may still land once Redis resumes.
    assert received in ([], [{"mark": "held-0"}])
    tail = _relay_states(entries)[-3:]
    if publish_after:
        assert tail[-3:] == ["missing:8", TaskState.working, "closed"]
        assert entries[-1][1] == {"closed": "1"}
    else:
        assert entries[-1][1] == {"closed": "1", "missing": "8"}
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and "interrupted" in record.getMessage()
    ] == [
        "A2A task interrupted: of the events held during a failed cancel, 1 "
        "reached local consumers but not the relay publish and 7 were dropped "
        "by an interrupted release"
    ]


async def test_a_gap_counts_each_unpublished_event_once(redis_client):
    """A release whose first publish finds an earlier gap still unmarked, and
    cannot mark it, counts its own event once."""
    store, handler, relay, _lease = await _live_relay_on(
        redis_client, redis_client, "once"
    )
    real_publish, real_mark = store.publish_event, store.mark_event_stream_incomplete
    refusing = {"on": False}

    async def publish_event(task, event):
        if refusing["on"]:
            raise A2ATaskStoreError(f"publish event for task {task} refused")
        await real_publish(task, event)

    async def mark_event_stream_incomplete(task, *, missing):
        if refusing["on"]:
            raise A2ATaskStoreError(f"mark event stream for task {task} refused")
        await real_mark(task, missing=missing)

    def emits_then_fails(count):
        async def begin_cancel(task, **kwargs):
            for index in range(count):
                await relay.enqueue_event(
                    _status_event(task, TaskState.working, f"{count}-{index}")
                )
            refusing["on"] = True
            raise A2ATaskOwnershipLostError(f"cancel task {task} lost its owner")

        return begin_cancel

    store.publish_event = publish_event
    store.mark_event_stream_incomplete = mark_event_stream_incomplete
    for count in (2, 3):
        store.begin_cancel = emits_then_fails(count)
        with pytest.raises(A2ATaskOwnershipLostError):
            await handler._cancel_owned("once")
    refusing["on"] = False
    await relay.close(immediate=True)
    entries = await redis_client.xrange("test:a2a:events:once")

    assert entries[-1][1] == {"closed": "1", "missing": "5"}


async def test_a_non_integer_cancel_limit_is_refused_with_its_name():
    from cogniverse_runtime.a2a_request_handler import (
        max_concurrent_cancels_from_env,
    )

    with pytest.raises(ValueError) as refused:
        max_concurrent_cancels_from_env({"A2A_MAX_CONCURRENT_CANCELS": "sixteen"})

    assert str(refused.value) == (
        "A2A_MAX_CONCURRENT_CANCELS must be an integer, got 'sixteen'"
    )
    assert max_concurrent_cancels_from_env({}) == 16
    assert max_concurrent_cancels_from_env({"A2A_MAX_CONCURRENT_CANCELS": "3"}) == 3


@pytest.fixture
def black_hole_redis():
    """An owned Redis a test can freeze: ``docker pause`` keeps its TCP
    endpoint accepting while nothing answers, the state of a Redis pod
    rescheduled from under its clients."""
    port = _free_port()
    container_name = f"redis-a2a-black-hole-{os.getpid()}-{uuid.uuid4().hex}"
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

    def docker(*args: str) -> None:
        done = subprocess.run(["docker", *args, container_name], capture_output=True)
        assert done.returncode == 0, done.stderr

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
        yield SimpleNamespace(
            url=f"redis://127.0.0.1:{port}/0",
            pause=lambda: docker("pause"),
            unpause=lambda: docker("unpause"),
        )
    finally:
        subprocess.run(["docker", "unpause", container_name], capture_output=True)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


async def _failure_within(call, bound_seconds: float):
    """``call``'s exception and how long it took; hanging past the bound fails."""
    started = time.monotonic()
    try:
        await asyncio.wait_for(call, timeout=bound_seconds)
    except asyncio.TimeoutError:
        pytest.fail(f"still waiting on Redis after {bound_seconds}s")
    except Exception as exc:
        return exc, time.monotonic() - started
    pytest.fail("the call succeeded against a Redis that answers nothing")


async def test_every_store_call_on_a_black_holed_redis_fails_within_its_timeout(
    black_hole_redis,
):
    store = await RedisTaskStore.from_url(black_hole_redis.url, key_prefix="test:a2a")
    try:
        lease = await store.acquire_execution(
            "bh-task", replica_id="replica-a", lease_seconds=60
        )
        await store.save(
            _task("bh-task", TaskState.working), _owned_context(store, lease)
        )
        black_hole_redis.pause()

        async def first_event():
            return await anext(store.subscribe_events("bh-task"))

        calls = {
            "get": store.get("bh-task"),
            "save": store.save(
                _task("bh-task", TaskState.working), _owned_context(store, lease)
            ),
            "acquire": store.acquire_execution(
                "bh-other", replica_id="replica-a", lease_seconds=60
            ),
            "renew": store.renew_execution(lease, lease_seconds=60),
            "publish": store.publish_event(
                "bh-task", _status_event("bh-task", TaskState.working)
            ),
            "listen": store.next_cancel(replica_id="replica-a"),
            "route": store.request_cancel(
                owner_replica_id="replica-b", task_id="bh-task", timeout_seconds=30
            ),
            "subscribe": first_event(),
        }
        outcomes = dict(
            zip(
                calls,
                await asyncio.gather(
                    *(_failure_within(call, 20) for call in calls.values())
                ),
            )
        )
    finally:
        black_hole_redis.unpause()
        await store.close()

    assert {name: str(exc) for name, (exc, _) in outcomes.items()} == {
        "get": "shared A2A task store unavailable: get task bh-task",
        "save": "shared A2A task store unavailable: save task bh-task",
        "acquire": "shared A2A task store unavailable: acquire task bh-other",
        "renew": "shared A2A task store unavailable: renew task bh-task",
        "publish": "shared A2A task store unavailable: publish event for task bh-task",
        "listen": "shared A2A task store unavailable: listen for cancels on replica "
        "replica-a",
        "route": "shared A2A task store unavailable: route cancel for task bh-task",
        "subscribe": "shared A2A task store unavailable: subscribe to task bh-task",
    }
    assert {type(exc) for exc, _ in outcomes.values()} == {A2ATaskStoreError}
    # One command timeout (5 s by default), plus reconnecting to the frozen
    # endpoint, which answers the TCP handshake but nothing after it.
    assert max(elapsed for _, elapsed in outcomes.values()) < 11


async def test_connecting_to_a_black_holed_redis_is_refused_within_its_timeout(
    black_hole_redis,
):
    black_hole_redis.pause()
    try:
        refused, elapsed = await _failure_within(
            RedisTaskStore.from_url(
                black_hole_redis.url, key_prefix="test:a2a", timeout_seconds=2
            ),
            20,
        )
    finally:
        black_hole_redis.unpause()

    assert type(refused) is A2ATaskStoreError
    assert str(refused) == (
        f"shared A2A task store unavailable: connect to {black_hole_redis.url}"
    )
    assert elapsed < 5


async def test_a_store_timeout_must_exceed_one_blocking_read():
    with pytest.raises(ValueError) as refused:
        await RedisTaskStore.from_url("redis://127.0.0.1:1/0", timeout_seconds=1)
    assert str(refused.value) == (
        "timeout_seconds must be > 1 (one blocking read), got 1"
    )


async def test_a_routed_cancel_waits_its_full_timeout_in_bounded_reads(redis_url):
    store = await RedisTaskStore.from_url(
        redis_url, key_prefix=f"test:{uuid.uuid4().hex}", timeout_seconds=2
    )
    try:
        refused, elapsed = await _failure_within(
            store.request_cancel(
                owner_replica_id="silent-owner", task_id="quiet", timeout_seconds=3.5
            ),
            20,
        )
    finally:
        await store.close()

    assert type(refused) is A2ACancelTimeoutError
    assert str(refused) == (
        "owner silent-owner did not acknowledge cancel for task quiet within 3.5s"
    )
    assert 3.4 < elapsed < 4.5


async def test_a2a_settings_default_to_production_and_refuse_non_numbers_by_name():
    from cogniverse_runtime.main import _a2a_settings_from_env

    assert _a2a_settings_from_env({}) == {
        "max_tasks": 10000,
        "lease_seconds": 30.0,
        "cancel_timeout_seconds": 10.0,
        "drain_timeout_seconds": 30.0,
        "max_concurrent_cancels": 16,
        "max_concurrent_resubscriptions": 64,
        "redis_timeout_seconds": 5.0,
        "redis_max_connections": 128,
    }
    assert _a2a_settings_from_env(
        {
            "A2A_MAX_CONCURRENT_RESUBSCRIPTIONS": "8",
            "A2A_REDIS_TIMEOUT_SECONDS": "2.5",
            "A2A_REDIS_MAX_CONNECTIONS": "20",
        }
    ) == {
        **_a2a_settings_from_env({}),
        "max_concurrent_resubscriptions": 8,
        "redis_timeout_seconds": 2.5,
        "redis_max_connections": 20,
    }
    refusals = {}
    for name, raw in (
        ("A2A_MAX_TASKS", "many"),
        ("A2A_REDIS_TIMEOUT_SECONDS", "soon"),
        ("A2A_REDIS_MAX_CONNECTIONS", "1.5"),
        ("A2A_MAX_CONCURRENT_RESUBSCRIPTIONS", "lots"),
    ):
        with pytest.raises(ValueError) as refused:
            _a2a_settings_from_env({name: raw})
        refusals[name] = str(refused.value)
    assert refusals == {
        "A2A_MAX_TASKS": "A2A_MAX_TASKS must be an integer, got 'many'",
        "A2A_REDIS_TIMEOUT_SECONDS": "A2A_REDIS_TIMEOUT_SECONDS must be a number, "
        "got 'soon'",
        "A2A_REDIS_MAX_CONNECTIONS": "A2A_REDIS_MAX_CONNECTIONS must be an integer, "
        "got '1.5'",
        "A2A_MAX_CONCURRENT_RESUBSCRIPTIONS": "A2A_MAX_CONCURRENT_RESUBSCRIPTIONS "
        "must be an integer, got 'lots'",
    }


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            {"max_concurrent_resubscriptions": 0},
            "A2A_MAX_CONCURRENT_RESUBSCRIPTIONS (max_concurrent_resubscriptions) "
            "must be >= 1, got 0",
        ),
        (
            {"max_concurrent_resubscriptions": 10, "redis_max_connections": 11},
            "A2A_REDIS_MAX_CONNECTIONS (redis_max_connections) must be at least "
            "A2A_MAX_CONCURRENT_RESUBSCRIPTIONS + 2 (12), got 11",
        ),
        (
            {"redis_timeout_seconds": 1.0},
            "A2A_REDIS_TIMEOUT_SECONDS (redis_timeout_seconds) must be > 1, got 1.0",
        ),
    ],
)
async def test_a_pool_that_resubscriptions_could_exhaust_is_refused_before_redis(
    overrides, message
):
    class Registry:
        def list_agents(self):
            raise AssertionError("registry accessed before the settings were checked")

    with pytest.raises(ValueError) as refused:
        await _build_shared_a2a_protocol(
            agent_registry=Registry(),
            dispatcher=SimpleNamespace(),
            redis_url=f"redis://127.0.0.1:{_free_port()}/0",
            replica_id="factory-replica",
            **_a2a_settings(**overrides),
        )
    assert str(refused.value) == message


async def _resubscribed(handler, task_id: str):
    """A resubscription waiting for its first event, or the refusal it got."""
    from a2a.types import TaskIdParams

    stream = handler.on_resubscribe_to_task(TaskIdParams(id=task_id))
    first = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.3)
    if first.done() and first.exception() is not None:
        return None, first.exception()
    return (stream, first), None


async def test_resubscriptions_past_the_replica_cap_are_refused_until_one_ends(
    redis_client,
):
    seed = _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a")
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    await seed.save(_task("resub", TaskState.working))
    await store.acquire_execution("resub", replica_id="replica-a", lease_seconds=30)
    handler = RedisRequestHandler(
        agent_executor=_CancellableExecutor(),
        task_store=store,
        replica_id="replica-a",
        max_concurrent_resubscriptions=2,
    )
    attempts = await asyncio.gather(
        *(_resubscribed(handler, "resub") for _ in range(5))
    )
    admitted = [waiting for waiting, _ in attempts if waiting is not None]
    refused = [error for _, error in attempts if error is not None]
    try:
        assert len(admitted) == 2
        assert [
            (type(error), type(error.error), error.error.message) for error in refused
        ] == [
            (
                ServerError,
                InvalidParamsError,
                "replica replica-a is already serving 2 resubscriptions; retry",
            )
        ] * 3

        ended_stream, ended_first = admitted.pop()
        ended_first.cancel()
        await asyncio.gather(ended_first, return_exceptions=True)
        await ended_stream.aclose()
        waiting, error = await _resubscribed(handler, "resub")
        assert error is None
        admitted.append(waiting)
        await store.publish_event(
            "resub", _status_event("resub", TaskState.working, "after-cap")
        )
        firsts = await asyncio.wait_for(
            asyncio.gather(*(first for _, first in admitted)), timeout=5
        )
        assert [event.metadata for event in firsts] == [{"mark": "after-cap"}] * 2
    finally:
        for stream, first in admitted:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()
    assert handler._resubscriptions == 0


async def test_blocking_reads_share_the_bounded_pool_without_exceeding_it(
    black_hole_redis,
):
    admin = aioredis.from_url(black_hole_redis.url, decode_responses=True)
    store = await RedisTaskStore.from_url(
        black_hole_redis.url,
        key_prefix="test:a2a",
        timeout_seconds=2,
        max_connections=2,
    )
    peak = 0
    reading = True

    async def watch_connections():
        nonlocal peak
        while reading:
            clients = await admin.client_list()
            peak = max(peak, len(clients) - 1)
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(watch_connections())
    started = time.monotonic()
    try:
        replies = await asyncio.gather(
            *(store.next_cancel(replica_id=f"idle-{index}") for index in range(4))
        )
        elapsed = time.monotonic() - started
    finally:
        reading = False
        await watcher
        await store.close()
        await admin.aclose()

    assert replies == [None] * 4
    assert peak == 2
    # Two rounds of two one-second reads, never four at once.
    assert 1.9 < elapsed < 3.5


_INTERRUPTED = (
    "Execution interrupted because its owning runtime stopped before completion."
)


def _end_if_hung(send: asyncio.Task) -> None:
    """Cancel a send the test already failed on for not ending, so a failing
    run reports instead of hanging in cleanup."""
    if not send.done():
        send.cancel()


async def _renewing_execution(
    store, executor, lease_seconds: float, drain_timeout_seconds: float = 30
):
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-renew",
        lease_seconds=lease_seconds,
        drain_timeout_seconds=drain_timeout_seconds,
    )
    send = asyncio.create_task(handler.on_message_send(_send_params("renew")))
    await asyncio.wait_for(executor.working.wait(), timeout=5)
    task_id = next(iter(handler._running_agents))
    # Nothing but the renewals writes to Redis from here on.
    async with asyncio.timeout(5):
        while True:
            stored = await store.get(task_id)
            if stored is not None and stored.status.state == TaskState.working:
                break
            await asyncio.sleep(0.01)
    return handler, send, task_id, handler._running_agents[task_id]


async def test_a_transient_renewal_failure_is_retried_not_fatal(
    redis_url, redis_client, caplog
):
    caplog.set_level("WARNING", logger="cogniverse_runtime.a2a_request_handler")
    store = await RedisTaskStore.from_url(
        redis_url, key_prefix="test:a2a", timeout_seconds=1.2
    )
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=6, drain_timeout_seconds=0.5
    )
    try:
        # The first renewal (at 2 s) meets a Redis that answers nothing for
        # longer than the command timeout; the lease still has 4 s left.
        await redis_client.execute_command("CLIENT", "PAUSE", "3400", "WRITE")
        await asyncio.sleep(5)
        running_after_outage = not producer.done()
        lease = await store.get_execution_lease(task_id)
        live = await store.has_live_owner(task_id)
    finally:
        # Shutdown cuts the still-running execution at its drain deadline,
        # which ends the blocking send.
        await handler.close()
        try:
            sent = await asyncio.wait_for(asyncio.shield(send), timeout=5)
        finally:
            _end_if_hung(send)
            await store.close()

    assert (sent.id, sent.status.state) == (task_id, TaskState.failed)
    assert sent.status.message.parts[0].root.text == _INTERRUPTED
    assert running_after_outage is True
    assert (lease.replica_id, live) == ("replica-renew", True)
    renewal_logs = [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if "renewal" in record.getMessage()
    ]
    assert len(renewal_logs) == 1
    level, message = renewal_logs[0]
    assert level == "WARNING"
    assert message.startswith(
        f"A2A execution lease renewal for task {task_id} failed with "
    )
    assert message.endswith(
        f"s of the lease left; retrying: shared A2A task store unavailable: "
        f"renew task {task_id}"
    )


async def test_renewal_that_cannot_reach_redis_cancels_only_once_the_lease_expired(
    redis_url, redis_client, caplog
):
    store = await RedisTaskStore.from_url(
        redis_url, key_prefix="test:a2a", timeout_seconds=1.2
    )
    executor = _CancellableExecutor()
    started = asyncio.get_running_loop().time()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=3
    )
    try:
        await redis_client.execute_command("CLIENT", "PAUSE", "8000", "WRITE")
        await asyncio.wait_for(asyncio.shield(asyncio.wait({producer})), timeout=15)
        cancelled_after = asyncio.get_running_loop().time() - started
        # The send ends on its own: the failed event its stop enqueues is the
        # final one, and saving it meets the same silent Redis.
        with pytest.raises(A2ATaskStoreError) as ended:
            await asyncio.wait_for(asyncio.shield(send), timeout=5)
        ended_after = asyncio.get_running_loop().time() - started
    finally:
        await redis_client.execute_command("CLIENT", "UNPAUSE")
        _end_if_hung(send)
        await asyncio.gather(send, return_exceptions=True)
        await handler.close()
        await store.close()

    assert producer.cancelled() is True
    # The first renewal fails at about 2.2 s; the lease runs to 3 s.
    assert 3.0 <= cancelled_after < 6
    assert str(ended.value) == (
        f"shared A2A task store unavailable: save task {task_id}"
    )
    # One command timeout (1.2 s) after the stop, while Redis is still paused.
    assert ended_after < cancelled_after + 3
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and "renewal" in record.getMessage()
    ] == [f"A2A execution lease renewal failed for task {task_id}"]


async def test_renewal_that_lost_ownership_cancels_at_once(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    started = asyncio.get_running_loop().time()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=3
    )
    peer = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    try:
        # The owner's own cancel path takes the next generation.
        await peer.begin_cancel(task_id, replica_id="replica-renew", lease_seconds=30)
        await asyncio.wait_for(asyncio.shield(asyncio.wait({producer})), timeout=10)
        cancelled_after = asyncio.get_running_loop().time() - started
        # The send ends on its own, refused by the fence the peer's
        # generation set: this node saves nothing after losing the task.
        with pytest.raises(A2ATaskOwnershipLostError) as ended:
            await asyncio.wait_for(asyncio.shield(send), timeout=2)
        ended_after = asyncio.get_running_loop().time() - started
        stored = await peer.get(task_id)
    finally:
        _end_if_hung(send)
        await asyncio.gather(send, return_exceptions=True)
        await handler.close()

    assert producer.cancelled() is True
    # At the first renewal (1 s), not at the lease's expiry (3 s).
    assert cancelled_after < 2
    assert ended_after < cancelled_after + 1
    assert str(ended.value) == (
        f"save task {task_id} with stale ownership generation 1"
    )
    assert stored.status.state == TaskState.working


class _StubbornExecutor(_CancellableExecutor):
    """An execution that keeps running for a while after being cancelled."""

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
        while not self.unwedge.is_set():
            try:
                await self.unwedge.wait()
            except asyncio.CancelledError:
                continue


async def test_close_finishes_within_twice_its_drain_budget(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    stubborn = _StubbornExecutor()
    handler = RedisRequestHandler(
        agent_executor=stubborn,
        task_store=store,
        replica_id="replica-drain",
        drain_timeout_seconds=1.0,
        cancel_timeout_seconds=30,
    )
    try:
        send, task_id = await _running_blocking_send(handler, store, stubborn)
        # A cancel whose executor never answers is running too.
        stubborn.wedged_cancels.add(task_id)
        stubborn.stubborn = True
        cancel = handler._coalesced_cancel(task_id)
        await asyncio.wait_for(stubborn.cancelling.wait(), timeout=5)

        started = time.monotonic()
        try:
            await asyncio.wait_for(handler.close(), timeout=10)
        finally:
            elapsed = time.monotonic() - started
    finally:
        stubborn.unwedge.set()
    # Released, the abandoned cancel publishes its canceled event, which is
    # what ends the blocking send.
    await asyncio.wait_for(
        asyncio.gather(send, cancel, return_exceptions=True), timeout=30
    )
    # One second to drain, at most one more for what it cancelled to stop.
    assert 1.9 < elapsed < 2.4


async def test_a_blocking_send_ends_when_close_cuts_its_producer_at_the_drain_deadline(
    redis_client,
):
    """The execution outlives the drain budget; close() cancels it while
    this node still owns the task, records it interrupted, releases the
    lease, and the blocking send answers with that failed task."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=30, drain_timeout_seconds=0.5
    )
    started = time.monotonic()
    await asyncio.wait_for(handler.close(), timeout=5)
    try:
        sent = await asyncio.wait_for(asyncio.shield(send), timeout=2)
    finally:
        _end_if_hung(send)
    ended_after = time.monotonic() - started
    stored = await store.get(task_id)

    assert producer.cancelled() is True
    assert (sent.id, sent.status.state) == (task_id, TaskState.failed)
    assert sent.status.message.parts[0].root.text == _INTERRUPTED
    assert ended_after < 2
    assert stored.status.state == TaskState.failed
    assert stored.status.message.parts[0].root.text == _INTERRUPTED
    assert await store.get_execution_lease(task_id) is None


async def test_a_drain_stop_with_redis_silent_still_ends_the_blocking_send(
    redis_url, redis_client
):
    """The stopped stream's relay close cannot reach Redis; the local queue
    closes anyway, so the send answers instead of waiting for process exit.
    Saving the interrupted state meets the same silent Redis, so the send
    answers with that error."""
    store = await RedisTaskStore.from_url(
        redis_url, key_prefix="test:a2a", timeout_seconds=1.2
    )
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=30, drain_timeout_seconds=0.5
    )
    try:
        await redis_client.execute_command("CLIENT", "PAUSE", "6000", "WRITE")
        started = time.monotonic()
        await asyncio.wait_for(handler.close(), timeout=10)
        try:
            with pytest.raises(A2ATaskStoreError) as ended:
                await asyncio.wait_for(asyncio.shield(send), timeout=4)
        finally:
            _end_if_hung(send)
        ended_after = time.monotonic() - started
    finally:
        await redis_client.execute_command("CLIENT", "UNPAUSE")
        await asyncio.gather(send, return_exceptions=True)
        await store.close()

    assert producer.cancelled() is True
    assert str(ended.value) == (
        f"shared A2A task store unavailable: save task {task_id}"
    )
    assert ended_after < 5


async def test_a_stop_on_a_finished_producer_records_nothing(redis_client):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    handler = _cancel_handler(store, "replica-a")
    finished = asyncio.create_task(asyncio.sleep(0))
    await finished

    handler._stop_producer(finished, "lease lost")

    assert handler._stop_reasons == {}
    assert finished.cancelled() is False


async def test_a_lease_loss_inside_the_owners_cancel_leaves_the_canceled_result(
    redis_client,
):
    """The owner's own cancel takes the next generation, so a renewal inside
    the cancel's window sees the lease lost; the sender still gets the
    canceled task, not a failed event ahead of it."""
    from a2a.types import TaskIdParams

    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    executor.wedge_every_cancel = True
    handler = RedisRequestHandler(
        agent_executor=executor,
        task_store=store,
        replica_id="replica-a",
        lease_seconds=0.6,
        cancel_timeout_seconds=10,
    )
    send, task_id = await _running_blocking_send(handler, store, executor)
    producer = handler._running_agents[task_id]
    cancel = asyncio.create_task(handler.on_cancel_task(TaskIdParams(id=task_id)))
    try:
        await asyncio.wait_for(executor.cancelling.wait(), timeout=5)
        # Renewals run every 0.2 s; the first to meet the cancel's generation
        # stops the producer while the cancel still holds its events.
        async with asyncio.timeout(5):
            while not producer.cancelling():
                await asyncio.sleep(0.05)
        executor.unwedge.set()
        canceled = await asyncio.wait_for(cancel, timeout=5)
        try:
            sent = await asyncio.wait_for(asyncio.shield(send), timeout=5)
        finally:
            _end_if_hung(send)
    finally:
        executor.unwedge.set()
        await asyncio.gather(send, cancel, return_exceptions=True)
        await handler.close()

    assert producer.cancelled() is True
    assert canceled.status.state == TaskState.canceled
    assert (sent.id, sent.status.state) == (task_id, TaskState.canceled)
    assert (await store.get(task_id)).status.state == TaskState.canceled


class _SilentTurnExecutor(AgentExecutor):
    """A turn that runs until cancelled and emits nothing, so the stored task
    keeps the state its previous turn left."""

    def __init__(self) -> None:
        self.running = asyncio.Event()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.running.set()
        await asyncio.Event().wait()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise AssertionError("no cancellation is expected")


class _SteppedExecutor(AgentExecutor):
    """Emits ``b-0``, then ``b-1`` once ``step`` is set, then completes with
    ``b-done`` once ``finish`` is set."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.step = asyncio.Event()
        self.finish = asyncio.Event()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            _status_event(context.task_id, TaskState.working, "b-0")
        )
        self.started.set()
        await self.step.wait()
        await event_queue.enqueue_event(
            _status_event(context.task_id, TaskState.working, "b-1")
        )
        await self.finish.wait()
        await event_queue.enqueue_event(
            _status_event(context.task_id, TaskState.completed, "b-done")
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise AssertionError("no cancellation is expected")


def _turn_params(task_id: str, text: str):
    """A continuation of ``task_id``, in the context its seeded task has."""
    params = _send_params(text, task_id=task_id)
    params.message.context_id = f"context-{task_id}"
    return params


async def _collect(stream, into: list) -> None:
    async for event in stream:
        into.append(event)


def _marks(events) -> list:
    return [
        (event.status.state, (event.metadata or {}).get("mark")) for event in events
    ]


async def _run_out_lease(redis_client, store: RedisTaskStore, task_id: str):
    """Record the task's lease as expired, as Redis sees a stalled owner's.

    Only the expiry changes: the owner still holds its generation and its
    execution keeps running, the state a node is in until it notices.
    """
    lease = await store.get_execution_lease(task_id)
    await redis_client.hset(
        "test:a2a:leases",
        task_id,
        json.dumps(
            {
                "replica_id": lease.replica_id,
                "request_id": lease.request_id,
                "generation": lease.generation,
                "expires_at_ms": 0,
            }
        ),
    )
    return lease


async def _two_nodes(redis_client, redis_url, task_id: str, *, drain_seconds: float):
    """Node A running a silent turn on ``task_id`` with its lease run out, and
    node B, on its own Redis client, ready to take the task over."""
    await _seed_store(redis_client, max_tasks=10, key_prefix="test:a2a").save(
        _task(task_id, TaskState.input_required)
    )
    client_b = aioredis.from_url(redis_url, decode_responses=True)
    silent = _SilentTurnExecutor()
    node_a = RedisRequestHandler(
        agent_executor=silent,
        task_store=RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a"),
        replica_id="replica-a",
        lease_seconds=30,
        drain_timeout_seconds=drain_seconds,
    )
    stepped = _SteppedExecutor()
    node_b = RedisRequestHandler(
        agent_executor=stepped,
        task_store=RedisTaskStore(client_b, max_tasks=10, key_prefix="test:a2a"),
        replica_id="replica-b",
        lease_seconds=30,
    )
    send_a = asyncio.create_task(
        node_a.on_message_send(_turn_params(task_id, "turn-a"))
    )
    await asyncio.wait_for(silent.running.wait(), timeout=5)
    await _run_out_lease(redis_client, node_a.task_store, task_id)
    return SimpleNamespace(
        node_a=node_a,
        node_b=node_b,
        stepped=stepped,
        send_a=send_a,
        client_b=client_b,
        cleanup=[send_a],
    )


async def _end_nodes(nodes) -> None:
    nodes.stepped.step.set()
    nodes.stepped.finish.set()
    for task in nodes.cleanup:
        _end_if_hung(task)
    await asyncio.gather(*nodes.cleanup, return_exceptions=True)
    await nodes.node_a.close()
    await nodes.node_b.close()
    await nodes.client_b.aclose()


async def _resubscribe_on_b(nodes, task_id: str, relayed: list) -> asyncio.Task:
    """B's resubscriber, reading the relay from B's first stored event on."""
    from a2a.types import TaskIdParams

    async with asyncio.timeout(5):
        while (await nodes.node_b.task_store.get(task_id)).status.state != (
            TaskState.working
        ):
            await asyncio.sleep(0.01)
    reader = asyncio.create_task(
        _collect(nodes.node_b.on_resubscribe_to_task(TaskIdParams(id=task_id)), relayed)
    )
    nodes.cleanup.append(reader)
    # The resubscription reads from its first XREAD on.
    await asyncio.sleep(0.3)
    return reader


async def _finish_on_b(nodes, reader: asyncio.Task, relayed: list) -> None:
    nodes.stepped.step.set()
    async with asyncio.timeout(5):
        while not relayed:
            assert reader.done() is False, f"B's resubscriber ended with {relayed}"
            await asyncio.sleep(0.01)
    nodes.stepped.finish.set()
    await asyncio.wait_for(reader, timeout=5)


async def test_a_node_that_lost_the_task_cannot_close_the_new_owners_relay(
    redis_client, redis_url
):
    """A's lease ran out and B took the task over; A's stopped turn then
    closes its queue. B's resubscriber keeps reading, and B ends the relay."""
    task_id = "task-taken"
    nodes = await _two_nodes(redis_client, redis_url, task_id, drain_seconds=0.2)
    relayed: list = []
    b_events: list = []
    try:
        stream_b = asyncio.create_task(
            _collect(
                nodes.node_b.on_message_send_stream(_turn_params(task_id, "turn-b")),
                b_events,
            )
        )
        nodes.cleanup.append(stream_b)
        await asyncio.wait_for(nodes.stepped.started.wait(), timeout=5)
        reader = await _resubscribe_on_b(nodes, task_id, relayed)

        # Shutdown stops A's turn at its drain deadline; its queue closes.
        await asyncio.wait_for(nodes.node_a.close(), timeout=5)
        await asyncio.sleep(0.3)
        assert reader.done() is False, f"B's resubscriber ended with {relayed}"

        await _finish_on_b(nodes, reader, relayed)
        await asyncio.wait_for(stream_b, timeout=5)
        entries = await redis_client.xrange(f"test:a2a:events:{task_id}")
        ttl = await redis_client.ttl(f"test:a2a:events:{task_id}")
        stored = await nodes.node_b.task_store.get(task_id)
    finally:
        await _end_nodes(nodes)

    assert _marks(relayed) == [
        (TaskState.working, "b-1"),
        (TaskState.completed, "b-done"),
    ]
    assert _marks(b_events) == [
        (TaskState.working, "b-0"),
        (TaskState.working, "b-1"),
        (TaskState.completed, "b-done"),
    ]
    assert _relay_states(entries) == [
        TaskState.working,
        TaskState.working,
        TaskState.completed,
        "closed",
    ]
    assert 0 < ttl <= 60
    assert stored.status.state == TaskState.completed


@pytest.mark.parametrize("takeover_after", [0.0, 0.03, 0.04, 0.045, 0.05, 0.07, 0.2])
async def test_a_close_racing_a_takeover_never_ends_the_new_owners_relay(
    redis_client, redis_url, takeover_after
):
    """A's drain stop (at 0.05 s) and B's takeover run at once, B starting
    at a range of offsets around it.

    Either A still owned the task when it stopped, and the task ends
    interrupted with B refused, or B took it over, and then B's resubscriber
    reads B's events up to B's own close whatever A's close did meanwhile.
    """
    task_id = f"task-race-{int(takeover_after * 1000)}"
    nodes = await _two_nodes(redis_client, redis_url, task_id, drain_seconds=0.05)
    relayed: list = []
    b_events: list = []
    try:
        close_a = asyncio.create_task(nodes.node_a.close())
        nodes.cleanup.append(close_a)
        await asyncio.sleep(takeover_after)
        stream_b = asyncio.create_task(
            _collect(
                nodes.node_b.on_message_send_stream(_turn_params(task_id, "turn-b")),
                b_events,
            )
        )
        nodes.cleanup.append(stream_b)
        started = asyncio.create_task(nodes.stepped.started.wait())
        nodes.cleanup.append(started)
        await asyncio.wait(
            {started, stream_b}, timeout=5, return_when="FIRST_COMPLETED"
        )
        taken_over = nodes.stepped.started.is_set()
        if taken_over:
            reader = await _resubscribe_on_b(nodes, task_id, relayed)
            await asyncio.wait_for(close_a, timeout=5)
            await _finish_on_b(nodes, reader, relayed)
            await asyncio.wait_for(stream_b, timeout=5)
        else:
            await asyncio.wait_for(close_a, timeout=5)
            refused = stream_b.exception()
        entries = await redis_client.xrange(f"test:a2a:events:{task_id}")
        stored = await nodes.node_b.task_store.get(task_id)
    finally:
        await _end_nodes(nodes)

    if taken_over:
        assert _marks(relayed) == [
            (TaskState.working, "b-1"),
            (TaskState.completed, "b-done"),
        ]
        # A close A made before B took the task over precedes B's events;
        # none lands among them.
        states = _relay_states(entries)
        assert states[states.index(TaskState.working) :] == [
            TaskState.working,
            TaskState.working,
            TaskState.completed,
            "closed",
        ]
        assert stored.status.state == TaskState.completed
    else:
        assert isinstance(refused, ServerError)
        assert refused.error.message == f"Task {task_id} is in terminal state: failed"
        assert stored.status.state == TaskState.failed
        assert stored.status.message.parts[0].root.text == (
            "Execution interrupted because its owning runtime stopped before "
            "completion."
        )


async def _get_task(handler: RedisRequestHandler, task_id: str) -> Task:
    """``tasks/get`` for ``task_id`` as ``handler`` answers it."""
    from a2a.types import TaskQueryParams

    return await handler.on_get_task(TaskQueryParams(id=task_id))


async def test_tasks_get_reports_a_turn_cut_at_the_drain_deadline_interrupted(
    redis_client,
):
    """The server cut the request before the drain, so no consumer is left
    to save the stopped turn's end; the stopping node records it anyway,
    and a peer's ``tasks/get`` right after shutdown reads it terminal."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=30, drain_timeout_seconds=0.5
    )
    peer = _cancel_handler(
        RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a"),
        "replica-peer",
    )
    # The request is cut, as uvicorn's graceful shutdown does first.
    send.cancel()
    try:
        await asyncio.wait_for(handler.close(), timeout=5)
        got = await _get_task(peer, task_id)
        lease = await store.get_execution_lease(task_id)
    finally:
        await asyncio.wait_for(asyncio.gather(send, return_exceptions=True), timeout=10)

    assert producer.cancelled() is True
    assert (got.id, got.status.state) == (task_id, TaskState.failed)
    assert got.status.message.parts[0].root.text == _INTERRUPTED
    assert lease is None


async def test_a_drain_after_the_task_was_lost_records_nothing(redis_client):
    """A peer took the task over before this node's drain deadline: the
    interrupted save is refused by the fence, so the peer's state stands."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=30, drain_timeout_seconds=0.5
    )
    peer = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    try:
        await _run_out_lease(redis_client, store, task_id)
        with pytest.raises(A2ATaskOwnershipLostError):
            await peer.acquire_execution(
                task_id, replica_id="replica-peer", lease_seconds=30
            )
        assert await peer.mark_owner_lost(task_id) is True
        peer_state = await redis_client.hget("test:a2a:tasks", task_id)
        generations = await redis_client.hget("test:a2a:generations", task_id)

        await asyncio.wait_for(handler.close(), timeout=5)
        with pytest.raises(A2ATaskOwnershipLostError) as ended:
            await asyncio.wait_for(asyncio.shield(send), timeout=5)
    finally:
        _end_if_hung(send)
        await asyncio.gather(send, return_exceptions=True)

    assert producer.cancelled() is True
    assert str(ended.value) == (
        f"save task {task_id} with stale ownership generation 1"
    )
    assert await redis_client.hget("test:a2a:tasks", task_id) == peer_state
    assert await redis_client.hget("test:a2a:generations", task_id) == generations
    assert await store.get_execution_lease(task_id) is None


async def test_tasks_get_reports_an_owner_that_died_without_draining_interrupted(
    redis_client,
):
    """The owner was killed mid-execution: no drain ran, its lease simply
    stops being renewed. Once it has expired a reader resolves the task."""
    dead = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    lease = await dead.acquire_execution(
        "task-killed", replica_id="replica-killed", lease_seconds=0.3
    )
    await dead.save(
        _task("task-killed", TaskState.working), _owned_context(dead, lease)
    )
    reader = _cancel_handler(
        RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a"),
        "replica-reader",
    )
    before_expiry = await _get_task(reader, "task-killed")
    await asyncio.sleep(0.4)

    got = await _get_task(reader, "task-killed")

    assert before_expiry.status.state == TaskState.working
    assert got.status.state == TaskState.failed
    assert got.status.message.parts[0].root.text == _INTERRUPTED
    assert await dead.get("task-killed") == got
    assert await dead.get_execution_lease("task-killed") is None


async def test_tasks_get_never_touches_a_task_whose_owner_keeps_renewing(
    redis_client,
):
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=0.6, drain_timeout_seconds=0.5
    )
    reader = _cancel_handler(
        RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a"),
        "replica-reader",
    )
    loop = asyncio.get_running_loop()
    try:
        sequence = await redis_client.get("test:a2a:generation-seq")
        owned = await store.get_execution_lease(task_id)
        states = []
        # Three lease lengths, each outlived only by renewing.
        polled_until = loop.time() + 1.8
        while loop.time() < polled_until:
            states.append((await _get_task(reader, task_id)).status.state)
            await asyncio.sleep(0.05)
        renewed = await store.get_execution_lease(task_id)
        sequence_after = await redis_client.get("test:a2a:generation-seq")
        running = not producer.done()
    finally:
        await handler.close()
        _end_if_hung(send)
        await asyncio.gather(send, return_exceptions=True)

    assert running is True
    assert len(states) >= 20
    assert set(states) == {TaskState.working}
    assert (renewed.request_id, renewed.generation) == (
        owned.request_id,
        owned.generation,
    )
    assert renewed.expires_at_ms > owned.expires_at_ms
    assert sequence_after == sequence


async def test_concurrent_readers_of_an_expired_owners_task_write_it_once(
    redis_client, redis_url
):
    dead = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    lease = await dead.acquire_execution(
        "task-read-twice", replica_id="replica-killed", lease_seconds=0.2
    )
    await dead.save(
        _task("task-read-twice", TaskState.working), _owned_context(dead, lease)
    )
    clients = [aioredis.from_url(redis_url, decode_responses=True) for _ in range(2)]
    readers = [
        _cancel_handler(
            RedisTaskStore(client, max_tasks=10, key_prefix="test:a2a"),
            f"replica-reader-{index}",
        )
        for index, client in enumerate(clients)
    ]
    await asyncio.sleep(0.3)
    sequence = int(await redis_client.get("test:a2a:generation-seq"))
    try:
        got = await asyncio.gather(
            *(_get_task(reader, "task-read-twice") for reader in readers)
        )
    finally:
        for client in clients:
            await client.aclose()
    sequence_after = int(await redis_client.get("test:a2a:generation-seq"))

    assert [task.status.state for task in got] == [TaskState.failed] * 2
    assert got[0].status.message.parts[0].root.text == _INTERRUPTED
    # One interruption: one new generation, one stored message both read.
    assert sequence_after == sequence + 1
    assert got[0] == got[1] == await dead.get("task-read-twice")


def _renewal_errors(caplog) -> list:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and "renewal" in record.getMessage()
    ]


async def test_a_renewal_after_the_drain_releases_the_lease_stays_quiet(
    redis_client, caplog
):
    """The request was cut, so the stopped producer's queue close waits its
    full bound; renewals every 0.3 s fall inside it, after the drain released
    the lease. None may report a lost lease or stop the producer again."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=0.9, drain_timeout_seconds=0.2
    )
    send.cancel()
    try:
        await asyncio.wait_for(handler.close(), timeout=15)
    finally:
        await asyncio.wait_for(asyncio.gather(send, return_exceptions=True), 10)
    stored = await store.get(task_id)

    assert producer.cancelled() is True
    assert producer.cancelling() == 1
    assert _renewal_errors(caplog) == []
    assert stored.status.state == TaskState.failed
    assert stored.status.message.parts[0].root.text == _INTERRUPTED
    assert await store.get_execution_lease(task_id) is None


async def test_a_renewal_landing_after_the_drain_release_leaves_the_send_failed(
    redis_client, caplog
):
    """A renewal tick lands between the drain's lease release and the end of
    the stopped stream; the blocking send still answers the stored failure."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=0.9, drain_timeout_seconds=0.2
    )
    real_release = store.release_execution
    released = []

    async def release_then_let_a_renewal_tick(lease):
        released.append(await real_release(lease))
        # Longer than one renewal interval (0.3 s).
        await asyncio.sleep(0.45)
        return released[-1]

    store.release_execution = release_then_let_a_renewal_tick
    try:
        await asyncio.wait_for(handler.close(), timeout=15)
        sent = await asyncio.wait_for(asyncio.shield(send), timeout=5)
    finally:
        _end_if_hung(send)
        await asyncio.gather(send, return_exceptions=True)
    stored = await store.get(task_id)

    assert released[0] is True
    assert producer.cancelled() is True
    assert producer.cancelling() == 1
    assert _renewal_errors(caplog) == []
    assert (sent.id, sent.status.state) == (task_id, TaskState.failed)
    assert sent.status.message.parts[0].root.text == _INTERRUPTED
    assert stored.status.state == TaskState.failed
    assert await store.get_execution_lease(task_id) is None


def _artifact(name: str) -> Artifact:
    return Artifact(
        artifact_id=f"artifact-{name}",
        name=name,
        parts=[Part(root=TextPart(text=f"{name} text"))],
    )


async def _stalled_owner(redis_client, task_id: str):
    """An owner whose lease expired while it still holds the task's current
    generation, so its saves still land."""
    owner = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    lease = await owner.acquire_execution(
        task_id, replica_id="replica-stalled", lease_seconds=0.2
    )
    context = _owned_context(owner, lease)
    await owner.save(_task(task_id, TaskState.working), context)
    await asyncio.sleep(0.3)
    return owner, context


def _saving_before_each_interrupt(client, owner, context, task_id: str, count: int):
    """Make the stalled owner save a new artifact right before each of the
    next ``count`` interruption scripts runs on ``client``."""
    from cogniverse_runtime.a2a_task_store import _INTERRUPT_SCRIPT

    real_eval = client.eval
    saved = []

    async def eval_after_an_owner_save(script, *args):
        if script == _INTERRUPT_SCRIPT and len(saved) < count:
            task = await owner.get(task_id)
            task.artifacts = [*task.artifacts, _artifact(f"late-{len(saved)}")]
            await owner.save(task, context)
            saved.append(task.artifacts[-1].artifact_id)
        return await real_eval(script, *args)

    client.eval = eval_after_an_owner_save
    return saved


async def test_an_owner_save_racing_the_interruption_is_kept(redis_client, redis_url):
    owner, context = await _stalled_owner(redis_client, "task-late-save")
    client = aioredis.from_url(redis_url, decode_responses=True)
    reader = RedisTaskStore(client, max_tasks=10, key_prefix="test:a2a")
    saved = _saving_before_each_interrupt(
        client, owner, context, "task-late-save", count=1
    )
    try:
        interrupted = await reader.mark_owner_lost("task-late-save")
    finally:
        await client.aclose()
    stored = await owner.get("task-late-save")

    assert saved == ["artifact-late-0"]
    assert interrupted is True
    assert stored.status.state == TaskState.failed
    assert stored.status.message.parts[0].root.text == _INTERRUPTED
    assert [artifact.artifact_id for artifact in stored.artifacts] == [
        "artifact-task-late-save",
        "artifact-late-0",
    ]


async def test_an_interruption_the_owner_keeps_changing_gives_up_unwritten(
    redis_client, redis_url
):
    owner, context = await _stalled_owner(redis_client, "task-busy")
    client = aioredis.from_url(redis_url, decode_responses=True)
    reader = RedisTaskStore(client, max_tasks=10, key_prefix="test:a2a")
    saved = _saving_before_each_interrupt(client, owner, context, "task-busy", 99)
    try:
        with pytest.raises(A2ATaskConflictError) as refused:
            await reader.mark_owner_lost("task-busy")
    finally:
        await client.aclose()
    stored = await owner.get("task-busy")

    assert str(refused.value) == (
        "task task-busy kept changing while its interruption was saved"
    )
    assert len(saved) == 5
    assert stored.status.state == TaskState.working
    assert stored.artifacts[-1].artifact_id == "artifact-late-4"


@pytest.mark.parametrize("late_state", [TaskState.working, TaskState.completed])
async def test_a_late_save_of_the_drained_generation_leaves_the_task_failed(
    redis_client, late_state
):
    """A consumer of the stopped execution saves under its generation after
    the drain recorded the task failed and released the lease; the failure
    stands."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=30, drain_timeout_seconds=0.5
    )
    lease = await store.get_execution_lease(task_id)
    try:
        await asyncio.wait_for(handler.close(), timeout=5)
        sent = await asyncio.wait_for(asyncio.shield(send), timeout=5)
    finally:
        _end_if_hung(send)
    drained = await redis_client.hget("test:a2a:tasks", task_id)
    late = _task(task_id, late_state)
    late.context_id = sent.context_id

    await store.save(late, _owned_context(store, lease))
    stored = await store.get(task_id)

    assert sent.status.state == TaskState.failed
    assert await store.get_execution_lease(task_id) is None
    assert await redis_client.hget("test:a2a:tasks", task_id) == drained
    assert stored.status.state == TaskState.failed
    assert stored.status.message.parts[0].root.text == _INTERRUPTED


async def test_a_shared_relay_is_never_created_without_its_owning_generation(
    redis_client,
):
    """A relay with no generation could never close the shared stream, so
    creating one is refused; a served relay is bound as it is created."""
    store = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
    executor = _CancellableExecutor()
    handler, send, task_id, producer = await _renewing_execution(
        store, executor, lease_seconds=30, drain_timeout_seconds=0.5
    )
    try:
        with pytest.raises(ValueError) as refused:
            await handler._queue_manager.create_or_tap("task-unowned")
        lease = await store.get_execution_lease(task_id)
        served = await handler._queue_manager.get(task_id)
    finally:
        await handler.close()
        _end_if_hung(send)
        await asyncio.gather(send, return_exceptions=True)

    assert str(refused.value) == (
        "A2A task task-unowned: a shared relay needs its owning generation"
    )
    assert await handler._queue_manager.get("task-unowned") is None
    assert served._generation == lease.generation


async def test_a_cancel_after_an_abandoned_one_ends_the_relay_it_publishes_on(
    redis_client,
):
    """The first cancel is abandoned while it holds the relay committed; the
    second takes a newer generation and must own the relay from then on, so
    its canceled event and the relay's close both land."""
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
    sent = await handler.on_message_send(_send_params("twice", blocking=False))
    try:
        with pytest.raises(ServerError):
            await asyncio.wait_for(
                handler.on_cancel_task(TaskIdParams(id=sent.id)), timeout=5
            )
        executor.wedge_every_cancel = False
        canceled = await asyncio.wait_for(
            handler.on_cancel_task(TaskIdParams(id=sent.id)), timeout=5
        )
        await asyncio.wait_for(handler.close(), timeout=10)
    finally:
        executor.unwedge.set()
    relay = await redis_client.xrange(f"test:a2a:events:{sent.id}")
    ttl = await redis_client.ttl(f"test:a2a:events:{sent.id}")

    assert canceled.status.state == TaskState.canceled
    assert (await store.get(sent.id)).status.state == TaskState.canceled
    assert _relay_states(relay) == [TaskState.working, TaskState.canceled, "closed"]
    assert 0 < ttl <= 60
