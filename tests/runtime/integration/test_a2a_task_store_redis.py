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
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

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
    assert (lease_a.generation, lease_b.generation, lease_c.generation) == (1, 1, 1)
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
    seed = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
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
    keys are per task id, so leaving them behind grows without bound.
    """
    store = RedisTaskStore(redis_client, max_tasks=2, key_prefix="test:a2a")
    await store.save(_task("deleted"))
    await store.acquire_execution("deleted", replica_id="replica-a", lease_seconds=2)
    await store.publish_event(
        "deleted",
        TaskStatusUpdateEvent(
            task_id="deleted",
            context_id="context-deleted",
            final=False,
            status=TaskStatus(state=TaskState.working),
        ),
    )
    await store.delete("deleted")

    await store.save(_task("evicted"))
    await store.acquire_execution("evicted", replica_id="replica-a", lease_seconds=2)
    await store.save(_task("kept"))
    await store.save(_task("incoming"))

    assert await store.get("evicted") is None
    assert await store.get("kept") is not None
    assert await redis_client.hkeys("test:a2a:generations") == []
    assert await redis_client.hkeys("test:a2a:leases") == []
    assert await redis_client.exists("test:a2a:events:deleted") == 0


async def test_expired_owner_is_marked_interrupted_without_reexecution(redis_client):
    seed = RedisTaskStore(redis_client, max_tasks=10, key_prefix="test:a2a")
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
    )
    try:
        assert protocol.skill_ids == ("search_agent",)
        assert protocol.handler.task_store is protocol.task_store
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
