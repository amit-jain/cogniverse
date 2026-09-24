"""
Integration tests for A2A multi-turn conversation support.

Full stack: real A2A app -> real CogniverseAgentExecutor -> real AgentDispatcher
-> real DSPy query rewrite -> real Vespa search. Exercises contextId-based
history accumulation through RedisRequestHandler over RedisTaskStore (the
production store) on an owned Redis.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import httpx
import pytest
import redis as sync_redis
import redis.asyncio as aioredis
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from cogniverse_runtime.a2a_request_handler import RedisRequestHandler
from cogniverse_runtime.a2a_task_store import RedisTaskStore
from tests.runtime.integration.conftest import skip_if_no_lm

logger = logging.getLogger(__name__)


class _ProcessExecutor(AgentExecutor):
    def __init__(self, redis_url: str, key_prefix: str) -> None:
        self._redis_url = redis_url
        self._process_key = f"{key_prefix}:test-process"
        self._release_key = f"{key_prefix}:test-release"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        if query == "long-process":
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
            )
            redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await redis.hset(
                self._process_key,
                mapping={"pid": str(process.pid), "returncode": "running"},
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    final=False,
                    status=TaskStatus(state=TaskState.working),
                )
            )
            try:
                await process.wait()
            except asyncio.CancelledError:
                process.terminate()
                returncode = await process.wait()
                await redis.hset(self._process_key, "returncode", str(returncode))
                raise
            finally:
                await redis.aclose()
            return

        if query.startswith("delay-"):
            # Hold the turn, and its lease, until the test releases it.
            redis = aioredis.from_url(self._redis_url, decode_responses=True)
            try:
                async with asyncio.timeout(10):
                    while not await redis.exists(self._release_key):
                        await asyncio.sleep(0.02)
            finally:
                await redis.aclose()
        history_ids = [
            message.message_id
            for message in (
                context.current_task.history if context.current_task else []
            )
        ]
        response = Message(
            message_id=f"agent-{query}",
            context_id=context_id,
            task_id=task_id,
            role=Role.agent,
            parts=[
                Part(
                    root=TextPart(
                        text=json.dumps(
                            {"query": query, "history_message_ids": history_ids}
                        )
                    )
                )
            ],
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                final=True,
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=response,
                ),
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id or "",
                context_id=context.context_id or "",
                final=True,
                status=TaskStatus(state=TaskState.canceled),
            )
        )


def _serve_shared_a2a(
    port: int, redis_url: str, key_prefix: str, replica_id: str
) -> None:
    async def _serve() -> None:
        store = await RedisTaskStore.from_url(
            redis_url,
            max_tasks=100,
            key_prefix=key_prefix,
            enforce_leases=True,
        )
        handler = RedisRequestHandler(
            agent_executor=_ProcessExecutor(redis_url, key_prefix),
            task_store=store,
            replica_id=replica_id,
            lease_seconds=0.6,
            cancel_timeout_seconds=3,
        )
        card = AgentCard(
            name="Cogniverse Runtime",
            description="shared task process test",
            url=f"http://127.0.0.1:{port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="search_agent",
                    name="search_agent",
                    description="deterministic process test",
                    tags=["search"],
                )
            ],
        )
        app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
        await handler.start()
        try:
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=port,
                    log_level="error",
                )
            )
            await server.serve()
        finally:
            await handler.close()
            await store.close()

    asyncio.run(_serve())


def _free_process_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_process_port(port: int) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError(f"A2A process on port {port} did not start")


@pytest.fixture(scope="module")
def shared_a2a_redis_url():
    override = os.environ.get("COGNIVERSE_TEST_REDIS_URL")
    if override:
        yield override
        return

    port = _free_process_port()
    container_name = f"redis-a2a-process-{os.getpid()}"
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
    redis_url = f"redis://127.0.0.1:{port}/0"
    redis = sync_redis.Redis.from_url(redis_url, decode_responses=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if redis.ping() is True:
                break
        except sync_redis.RedisError:
            time.sleep(0.25)
    else:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        pytest.fail("Redis did not become ready within 30 seconds")
    try:
        yield redis_url
    finally:
        redis.close()
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def _start_replica(
    port: int, redis_url: str, key_prefix: str, replica_id: str
) -> multiprocessing.process.BaseProcess:
    """Serve one runtime A2A replica in its own OS process."""
    process = multiprocessing.get_context("spawn").Process(
        target=_serve_shared_a2a,
        args=(port, redis_url, key_prefix, replica_id),
    )
    process.start()
    return process


@pytest.fixture
def a2a_process_cluster(shared_a2a_redis_url):
    key_prefix = f"test:a2a-process:{uuid.uuid4().hex}"
    ports = (_free_process_port(), _free_process_port())
    processes = [
        _start_replica(port, shared_a2a_redis_url, key_prefix, f"replica-{index}")
        for index, port in enumerate(ports)
    ]
    try:
        for port in ports:
            _wait_process_port(port)
        yield ports, processes, shared_a2a_redis_url, key_prefix
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def _process_rpc(port: int, method: str, params: dict, rpc_id: str) -> dict:
    with httpx.Client(timeout=10) as client:
        response = client.post(
            f"http://127.0.0.1:{port}/",
            json={"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params},
        )
    assert response.status_code == 200
    return response.json()


def _process_send(
    port: int,
    *,
    text: str,
    message_id: str,
    context_id: str,
    task_id: str | None = None,
    blocking: bool = True,
) -> dict:
    message = {
        "role": "user",
        "messageId": message_id,
        "contextId": context_id,
        "parts": [{"kind": "text", "text": text}],
    }
    if task_id:
        message["taskId"] = task_id
    return _process_rpc(
        port,
        "message/send",
        {
            "message": message,
            "configuration": {
                "acceptedOutputModes": ["text"],
                "blocking": blocking,
            },
            "metadata": {
                "agent_name": "search_agent",
                "tenant_id": "test:shared-a2a",
            },
        },
        message_id,
    )


def _wait_subprocess_pid(redis, process_key: str) -> int:
    """Wait for the owned executor to publish its subprocess PID."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pid = redis.hget(process_key, "pid")
        if pid:
            return int(pid)
        time.sleep(0.05)
    pytest.fail("Long-running subprocess PID was not published")


def _process_resubscribe(port: int, task_id: str) -> tuple[int, list[dict]]:
    with httpx.Client(timeout=10) as client:
        with client.stream(
            "POST",
            f"http://127.0.0.1:{port}/",
            json={
                "jsonrpc": "2.0",
                "id": "peer-resubscribe",
                "method": "tasks/resubscribe",
                "params": {"id": task_id},
            },
        ) as response:
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
            return response.status_code, events


@pytest.mark.integration
class TestA2ASharedProcessIdentity:
    def test_peer_returns_the_created_task_and_continues_its_turn(
        self, a2a_process_cluster
    ):
        ports, _, _, _ = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        created = _process_send(
            ports[0],
            text="first-turn",
            message_id="user-first",
            context_id=context_id,
        )["result"]
        task_id = created["id"]

        peer = _process_rpc(ports[1], "tasks/get", {"id": task_id}, "peer-get")
        assert peer["result"] == created

        continued = _process_send(
            ports[1],
            text="second-turn",
            message_id="user-second",
            context_id=context_id,
            task_id=task_id,
        )["result"]

        assert continued["id"] == task_id
        assert continued["contextId"] == context_id
        assert [message["messageId"] for message in continued["history"]] == [
            "user-first",
            "agent-first-turn",
            "user-second",
        ]

    def test_peer_and_restarted_owner_read_and_continue_the_same_task(
        self, a2a_process_cluster
    ):
        ports, processes, redis_url, key_prefix = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        created = _process_send(
            ports[0],
            text="first-turn",
            message_id="user-first",
            context_id=context_id,
        )["result"]
        task_id = created["id"]

        processes[0].terminate()
        processes[0].join(timeout=10)
        assert processes[0].exitcode == -signal.SIGTERM
        processes[0] = _start_replica(ports[0], redis_url, key_prefix, "replica-0")
        _wait_process_port(ports[0])

        peer = _process_rpc(ports[1], "tasks/get", {"id": task_id}, "peer-get")
        restarted = _process_rpc(
            ports[0], "tasks/get", {"id": task_id}, "restarted-owner-get"
        )
        assert peer["result"] == created
        assert restarted["result"] == created

        continued = _process_send(
            ports[1],
            text="second-turn",
            message_id="user-second",
            context_id=context_id,
            task_id=task_id,
        )["result"]

        assert continued["id"] == task_id
        assert continued["contextId"] == context_id
        assert [message["messageId"] for message in continued["history"]] == [
            "user-first",
            "agent-first-turn",
            "user-second",
        ]

    def test_simultaneous_continuations_have_one_acknowledged_winner(
        self, a2a_process_cluster
    ):
        ports, _, redis_url, key_prefix = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        first = _process_send(
            ports[0],
            text="seed-turn",
            message_id="user-seed",
            context_id=context_id,
        )
        task_id = first["result"]["id"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _process_send,
                    port,
                    text=f"delay-{index}",
                    message_id=f"user-race-{index}",
                    context_id=context_id,
                    task_id=task_id,
                )
                for index, port in enumerate(ports)
            ]
            # The winner's executor holds its lease until released, so the
            # other continuation can only be answered with the conflict.
            done, pending = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            assert (len(done), len(pending)) == (1, 1)
            conflicts = [future.result() for future in done]
            redis = sync_redis.Redis.from_url(redis_url, decode_responses=True)
            try:
                redis.set(f"{key_prefix}:test-release", "1")
            finally:
                redis.close()
            winners = [future.result(timeout=10) for future in pending]

        assert "error" in conflicts[0], conflicts[0]
        assert "result" in winners[0], winners[0]
        assert conflicts[0]["error"]["message"].startswith(
            f"Task {task_id} is active on replica-"
        )

        winner_message = winners[0]["result"]["history"][-1]["messageId"]
        stored = _process_rpc(ports[0], "tasks/get", {"id": task_id}, "get-after-race")[
            "result"
        ]
        assert [message["messageId"] for message in stored["history"]] == [
            "user-seed",
            "agent-seed-turn",
            winner_message,
        ]

    def test_peer_cancel_stops_owner_subprocess_and_fences_late_writes(
        self, a2a_process_cluster
    ):
        ports, _, redis_url, key_prefix = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        started = _process_send(
            ports[0],
            text="long-process",
            message_id="user-long",
            context_id=context_id,
            blocking=False,
        )["result"]
        task_id = started["id"]
        assert started["status"]["state"] == "working"

        redis = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        process_key = f"{key_prefix}:test-process"
        pid = _wait_subprocess_pid(redis, process_key)

        canceled = _process_rpc(
            ports[1], "tasks/cancel", {"id": task_id}, "peer-cancel"
        )["result"]
        assert canceled["status"]["state"] == "canceled"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            returncode = redis.hget(process_key, "returncode")
            if returncode == "-15":
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"Subprocess did not report SIGTERM: {returncode}")
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

        owner_task = _process_rpc(
            ports[0], "tasks/get", {"id": task_id}, "owner-get-canceled"
        )["result"]
        peer_task = _process_rpc(
            ports[1], "tasks/get", {"id": task_id}, "peer-get-canceled"
        )["result"]
        redis.close()
        assert owner_task == peer_task
        assert peer_task["status"]["state"] == "canceled"

    def test_peer_cancels_an_idle_task_that_no_replica_owns(self, a2a_process_cluster):
        """The commonest cancellable state has no owner at all.

        A completed turn leaves the task paused in input_required, which is
        not an active state, so its execution lease is already released. The
        stock handler cancels any non-terminal task and this one must too,
        from either replica.
        """
        ports, _, _, _ = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        created = _process_send(
            ports[0],
            text="idle-turn",
            message_id="user-idle",
            context_id=context_id,
        )["result"]
        task_id = created["id"]
        assert created["status"]["state"] == "input-required"

        canceled = _process_rpc(
            ports[1], "tasks/cancel", {"id": task_id}, "peer-cancel-idle"
        )["result"]
        owner_view = _process_rpc(
            ports[0], "tasks/get", {"id": task_id}, "owner-get-idle-cancel"
        )["result"]

        assert canceled["status"]["state"] == "canceled"
        assert owner_view == canceled

    def test_owner_loss_interrupts_the_task_without_re_executing_it(
        self, a2a_process_cluster
    ):
        ports, processes, redis_url, key_prefix = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        started = _process_send(
            ports[0],
            text="long-process",
            message_id="user-orphan",
            context_id=context_id,
            blocking=False,
        )["result"]
        task_id = started["id"]
        redis = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        process_key = f"{key_prefix}:test-process"
        pid = _wait_subprocess_pid(redis, process_key)

        with ThreadPoolExecutor(max_workers=1) as pool:
            stream = pool.submit(_process_resubscribe, ports[1], task_id)
            time.sleep(0.2)
            processes[0].kill()
            processes[0].join(timeout=10)
            # The relay is never closed by a killed owner, so the peer's
            # stream has to end on the expired lease instead of hanging.
            status_code, events = stream.result(timeout=10)
        assert (status_code, events) == (200, [])

        rejected = _process_send(
            ports[1],
            text="second-turn",
            message_id="user-after-orphan",
            context_id=context_id,
            task_id=task_id,
        )["error"]
        interrupted = _process_rpc(
            ports[1], "tasks/get", {"id": task_id}, "peer-get-interrupted"
        )["result"]
        surviving = redis.hgetall(process_key)
        redis.close()
        os.kill(pid, signal.SIGKILL)

        assert rejected["message"] == (
            f"Task {task_id} was interrupted after its owner stopped; "
            "inspect the failed task before retrying"
        )
        assert interrupted["status"]["state"] == "failed"
        # A re-execution would have published a second subprocess over this
        # one; the interrupted tool call is reported, never silently replayed.
        assert surviving == {"pid": str(pid), "returncode": "running"}

    def test_peer_resubscribe_receives_owner_terminal_event(self, a2a_process_cluster):
        ports, _, _, _ = a2a_process_cluster
        context_id = f"context-{uuid.uuid4().hex}"
        started = _process_send(
            ports[0],
            text="long-process",
            message_id="user-stream",
            context_id=context_id,
            blocking=False,
        )["result"]
        task_id = started["id"]

        with ThreadPoolExecutor(max_workers=1) as pool:
            stream = pool.submit(_process_resubscribe, ports[1], task_id)
            time.sleep(0.2)
            canceled = _process_rpc(
                ports[1], "tasks/cancel", {"id": task_id}, "cancel-stream"
            )["result"]
            status_code, events = stream.result(timeout=10)

        assert canceled["status"]["state"] == "canceled"
        assert status_code == 200
        assert [event["result"]["status"]["state"] for event in events] == ["canceled"]


def _send_message(
    client,
    text: str,
    context_id: str,
    agent_name: str = "search_agent",
    tenant_id: str = "test:unit",
    task_id: str | None = None,
    rpc_id: int = 1,
    message_metadata: dict | None = None,
) -> dict:
    """Send an A2A JSON-RPC message/send request and return the response body."""
    message = {
        "role": "user",
        "messageId": str(uuid.uuid4()),
        "contextId": context_id,
        "parts": [{"kind": "text", "text": text}],
    }
    if task_id:
        message["taskId"] = task_id
    if message_metadata:
        message["metadata"] = message_metadata

    payload = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "message/send",
        "params": {
            "message": message,
            "metadata": {
                "agent_name": agent_name,
                "tenant_id": tenant_id,
            },
        },
    }

    response = client.post("/", json=payload)
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"
    body = response.json()
    assert "result" in body, f"JSON-RPC error: {body.get('error')}"
    return body


def _extract_task_id(body: dict) -> str:
    """Extract the task id from an A2A JSON-RPC message/send response.

    The result is a Task, whose identifier is ``id``; ``taskId`` only appears
    nested inside ``history`` entries, never at the result's top level.
    """
    result = body["result"]
    task_id = result["id"]
    assert task_id, f"Empty task id in response: {result}"
    return task_id


def _extract_response_text(body: dict) -> str:
    """Extract the agent's text response from A2A JSON-RPC response."""
    result = body["result"]
    # Navigate to the text part — could be in status.message.parts or parts directly
    status = result.get("status", {})
    message = status.get("message", {})
    parts = message.get("parts", [])
    if parts:
        return parts[0].get("text", "")
    # Fallback: check result.parts directly
    parts = result.get("parts", [])
    if parts:
        return parts[0].get("text", "")
    return ""


@pytest.fixture
def dispatch_history_spy(dispatcher, monkeypatch):
    """Record the ``conversation_history`` each ``dispatch()`` call receives.

    Wraps the real ``AgentDispatcher.dispatch`` on the same module-scoped
    instance the ``a2a_client`` drives, so a test can assert what history the
    dispatcher actually saw on each turn — the contract the docstrings claim —
    rather than only that a response came back. Returns the list of captured
    histories (one entry per dispatch, in call order). monkeypatch restores
    the original method after the test so the shared instance is left intact.
    """
    captured: list[dict] = []
    original = dispatcher.dispatch

    async def _recording_dispatch(*args, **kwargs):
        context = kwargs.get("context")
        if context is None and len(args) >= 3:
            context = args[2]
        captured.append(
            {
                "query": kwargs.get("query") or (args[1] if len(args) >= 2 else None),
                "history": list((context or {}).get("conversation_history", [])),
            }
        )
        return await original(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "dispatch", _recording_dispatch)
    return captured


async def _stored_task(redis_url: str, key_prefix: str, task_id: str) -> dict:
    """The task as a peer replica reads it from the shared store."""
    store = await RedisTaskStore.from_url(redis_url, key_prefix=key_prefix)
    try:
        task = await store.get(task_id)
    finally:
        await store.close()
    assert task is not None, f"task {task_id} is not in the shared store"
    return task.model_dump(mode="json", by_alias=True, exclude_none=True)


@pytest.mark.integration
class TestA2AClientServesFromTheSharedStore:
    def test_a_served_turn_is_the_task_a_peer_reads_from_redis(
        self, a2a_client, workflow_state_redis_url, a2a_key_prefix
    ):
        """The multi-turn client runs the real executor and dispatcher on
        the production handler, so its task lands in the shared store."""
        body = _send_message(
            a2a_client,
            "search for cat videos",
            f"test-shared-store-{uuid.uuid4()}",
            agent_name="unregistered_agent",
            rpc_id=60,
        )
        served = body["result"]

        assert served["status"]["state"] == "failed"
        assert json.loads(served["status"]["message"]["parts"][0]["text"]) == {
            "type": "error",
            "agent": "unregistered_agent",
            "error_type": "ValueError",
            "message": (
                "Agent 'unregistered_agent' failed with ValueError. "
                "See runtime logs for detail."
            ),
        }
        stored = asyncio.run(
            _stored_task(workflow_state_redis_url, a2a_key_prefix, served["id"])
        )
        assert stored == served


@pytest.mark.integration
@skip_if_no_lm
class TestA2AMultiTurnHistoryAccumulation:
    """Test multi-turn conversation history via A2A contextId."""

    def test_multiturn_history_accumulates_three_turns(
        self,
        a2a_client,
        dspy_lm,
        vespa_instance,
        dispatch_history_spy,
        tomoro_search_url,
    ):
        """3 A2A calls with same contextId -> turn 3 carries history from turns 1+2."""
        context_id = f"test-accumulate-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(a2a_client, "search for cat videos", context_id, rpc_id=1)
        task_id = _extract_task_id(resp1)
        text1 = _extract_response_text(resp1)
        assert text1, "Turn 1 should produce a response"

        # Turn 2 — same context, same task
        resp2 = _send_message(
            a2a_client,
            "now search for dog videos",
            context_id,
            task_id=task_id,
            rpc_id=2,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response"

        # Turn 3 — same context, same task
        resp3 = _send_message(
            a2a_client,
            "compare the two sets of results",
            context_id,
            task_id=task_id,
            rpc_id=3,
        )
        text3 = _extract_response_text(resp3)
        assert text3, "Turn 3 should produce a response"

        # Contract: the dispatcher must SEE accumulating history, not just
        # return a response. One dispatch per turn, in order.
        assert len(dispatch_history_spy) == 3
        turn1_hist = dispatch_history_spy[0]["history"]
        turn2_hist = dispatch_history_spy[1]["history"]
        turn3_hist = dispatch_history_spy[2]["history"]

        # Turn 1 has no prior conversation.
        assert turn1_hist == []

        # Turn 2 carries turn 1: the user's "cat videos" query AND the agent's
        # turn-1 response (role=agent), proving both directions are persisted.
        turn2_contents = [t["content"] for t in turn2_hist]
        assert any("cat videos" in c for c in turn2_contents), turn2_contents
        assert any(t["role"] == "agent" for t in turn2_hist), turn2_hist
        assert not any("dog videos" in c for c in turn2_contents), turn2_contents

        # Turn 3 carries BOTH prior user turns (1 and 2), proving accumulation
        # across more than the immediately-preceding turn.
        turn3_contents = [t["content"] for t in turn3_hist]
        assert any("cat videos" in c for c in turn3_contents), turn3_contents
        assert any("dog videos" in c for c in turn3_contents), turn3_contents

    def test_client_supplied_message_history_reaches_dispatch(
        self, a2a_client, dspy_lm, vespa_instance, dispatch_history_spy
    ):
        """A stateful client (the code REPL) that tracks its own turns and
        sends them in the message metadata — with a fresh contextId and no
        taskId, exactly as the CLI does — has that history reach the dispatcher.
        Before the fix the server only read task.history and dropped it, so
        every REPL turn arrived contextless."""
        client_history = [
            {"role": "user", "content": "search for cat videos"},
            {"role": "assistant", "content": "found three cat clips"},
        ]
        _send_message(
            a2a_client,
            "how about the second one",
            context_id=f"repl-{uuid.uuid4()}",  # fresh per call, no task threading
            message_metadata={"conversation_history": client_history},
            rpc_id=1,
        )

        assert len(dispatch_history_spy) == 1
        seen_contents = [t["content"] for t in dispatch_history_spy[0]["history"]]
        assert "search for cat videos" in seen_contents
        assert "found three cat clips" in seen_contents

    def test_context_id_isolation(
        self,
        a2a_client,
        dspy_lm,
        vespa_instance,
        dispatch_history_spy,
        tomoro_search_url,
    ):
        """Messages to different contextIds don't cross-contaminate history."""
        ctx_a = f"test-iso-a-{uuid.uuid4()}"
        ctx_b = f"test-iso-b-{uuid.uuid4()}"

        # Context A: cats
        resp_a1 = _send_message(a2a_client, "search for cat videos", ctx_a, rpc_id=10)
        task_a = _extract_task_id(resp_a1)

        # Context B: dogs (different context)
        resp_b1 = _send_message(a2a_client, "search for dog videos", ctx_b, rpc_id=11)
        task_b = _extract_task_id(resp_b1)

        # They should have different task IDs
        assert task_a != task_b, "Different contexts should create different tasks"

        # Context A turn 2: should NOT see dog history
        resp_a2 = _send_message(
            a2a_client,
            "show me more of those",
            ctx_a,
            task_id=task_a,
            rpc_id=12,
        )
        text_a2 = _extract_response_text(resp_a2)
        assert text_a2, "Context A turn 2 should produce a response"

        # Contract: context A turn 2's dispatch must carry context A's history
        # ("cat videos") and must NOT contain anything from context B ("dog
        # videos"). Identify it by its query rather than call order.
        a2_dispatch = next(
            d for d in dispatch_history_spy if d["query"] == "show me more of those"
        )
        a2_contents = [t["content"] for t in a2_dispatch["history"]]
        assert any("cat videos" in c for c in a2_contents), a2_contents
        assert not any("dog videos" in c for c in a2_contents), a2_contents

    def test_task_stays_alive_input_required(
        self, a2a_client, dspy_lm, vespa_instance, tomoro_search_url
    ):
        """TaskState.input_required keeps task non-terminal for subsequent turns."""
        context_id = f"test-alive-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(a2a_client, "search for videos", context_id, rpc_id=20)
        task_id = _extract_task_id(resp1)

        # Verify task state is input_required (non-terminal)
        result1 = resp1["result"]
        status = result1.get("status", {})
        state = status.get("state", "")
        assert state == "input-required", (
            f"Task state should be input-required for multi-turn, got '{state}'"
        )

        # Turn 2 — should succeed (task is alive)
        resp2 = _send_message(
            a2a_client,
            "filter by duration",
            context_id,
            task_id=task_id,
            rpc_id=21,
        )
        assert "result" in resp2, "Turn 2 should succeed on alive task"

    def test_agent_response_in_history(
        self,
        a2a_client,
        dspy_lm,
        vespa_instance,
        dispatch_history_spy,
        tomoro_search_url,
    ):
        """Turn 1 agent response appears in turn 2's conversation context."""
        context_id = f"test-agent-hist-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(
            a2a_client, "search for cat videos", context_id, rpc_id=30
        )
        task_id = _extract_task_id(resp1)

        resp2 = _send_message(
            a2a_client,
            "show me more like those",
            context_id,
            task_id=task_id,
            rpc_id=31,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response with history from turn 1"

        # Contract: turn 2's dispatch history must include turn 1's USER query
        # AND the AGENT response (role=agent) — i.e. both halves of turn 1 are
        # extracted from Task.history and threaded into turn 2's context.
        t2_dispatch = next(
            d for d in dispatch_history_spy if d["query"] == "show me more like those"
        )
        roles = [t["role"] for t in t2_dispatch["history"]]
        contents = [t["content"] for t in t2_dispatch["history"]]
        assert "user" in roles, roles
        assert "agent" in roles, roles
        assert any("cat videos" in c for c in contents), contents

    def test_first_turn_no_rewrite(self, a2a_client, dspy_lm, vespa_instance):
        """Single turn with no history -> no query rewrite in response."""
        context_id = f"test-no-rewrite-{uuid.uuid4()}"

        resp = _send_message(
            a2a_client,
            "search for dog videos",
            context_id,
            rpc_id=40,
        )
        text = _extract_response_text(resp)
        assert text, "First turn should produce a response"

        result_data = json.loads(text)
        assert "rewritten_query" not in result_data, (
            "First turn with no history should not have rewritten_query"
        )

    def test_multiturn_query_rewrite_end_to_end(
        self,
        a2a_client,
        dspy_lm,
        vespa_instance,
        dispatch_history_spy,
        tomoro_search_url,
    ):
        """Turn 1: 'cat videos' -> Turn 2: 'show me longer ones' -> rewritten query."""
        context_id = f"test-rewrite-e2e-{uuid.uuid4()}"

        # Turn 1: explicit query
        resp1 = _send_message(
            a2a_client, "search for cat videos", context_id, rpc_id=50
        )
        task_id = _extract_task_id(resp1)

        # Turn 2: anaphoric reference — should trigger query rewrite
        resp2 = _send_message(
            a2a_client,
            "show me longer ones",
            context_id,
            task_id=task_id,
            rpc_id=51,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response"

        # Turn 2's dispatch must have received non-empty history — that's what
        # makes the dispatcher run the rewrite path.
        t2_dispatch = next(
            d for d in dispatch_history_spy if d["query"] == "show me longer ones"
        )
        assert t2_dispatch["history"], "Turn 2 should dispatch with prior history"

        # Contract (agent_dispatcher.dispatch): when history is present the
        # response carries BOTH original_query and rewritten_query — no
        # conditional, no swallowed JSON error. original is the raw turn-2
        # query; rewritten is a non-empty resolved string.
        result_data = json.loads(text2)
        assert result_data["original_query"] == "show me longer ones"
        assert "rewritten_query" in result_data, result_data
        assert (
            isinstance(result_data["rewritten_query"], str)
            and result_data["rewritten_query"].strip()
        ), result_data["rewritten_query"]
