"""AG-UI threads saved to and read back from the runtime's conversation store.

Real router -> real ``AgentDispatcher`` (both the token-stream and the
dispatch paths) -> deterministic agents. Each run's turn is saved through the
dispatcher's real ``ConversationLedger`` on Redis into a real Mem0 store on
Vespa (behind a fault proxy, embedding with the token embedder), and every
assertion reads the turns back through ``GET /ag-ui/threads/{thread_id}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from redis.asyncio import Redis

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.conversation import (
    RUN_CANCELLED_ROLE,
    RUN_CANCELLED_TEXT,
    ConversationStore,
)
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_LEASE_S,
    AgentDispatcher,
)
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import ag_ui, openai_compat
from cogniverse_runtime.session_state import ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from tests.utils.web_client import free_port
from tests.utils.web_ops import memory_on_vespa

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT_A = "agthreads:alpha"
TENANT_B = "agthreads:beta"
KEY_A = "ag-ui-threads-key-a"
KEY_B = "ag-ui-threads-key-b"


class ThreadDeps(AgentDeps):
    pass


class ThreadInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class SummaryOutput(AgentOutput):
    summary: str = ""


class StreamingAgent(AgentBase[ThreadInput, SummaryOutput, ThreadDeps]):
    """Streams its answer token by token."""

    async def _process_impl(self, input: ThreadInput) -> SummaryOutput:
        summary = f"streamed reply to: {input.query}"
        accumulated = ""
        for start in range(0, len(summary), 7):
            chunk = summary[start : start + 7]
            accumulated += chunk
            self.emit_progress(
                "token",
                chunk,
                data={"accumulated": accumulated, "output_field": "summary"},
            )
            await asyncio.sleep(0)
        return SummaryOutput(summary=summary)


class AnswerOutput(AgentOutput):
    answer: str = ""


class DispatchAgent(AgentBase[ThreadInput, AnswerOutput, ThreadDeps]):
    """Answers with the query and how many turns of history it was sent."""

    async def _process_impl(self, input: ThreadInput) -> AnswerOutput:
        return AnswerOutput(
            answer=f"dispatched reply to: {input.query} "
            f"(history {len(input.conversation_history)})"
        )


class FailingAgent(AgentBase[ThreadInput, AnswerOutput, ThreadDeps]):
    async def _process_impl(self, input: ThreadInput) -> AnswerOutput:
        raise RuntimeError("secret-backend-detail")


in_flight = {"count": 0, "peak": 0}
all_in_flight = asyncio.Event()
CONCURRENT_RUNS = 6
# The exception types Mem0's Vespa store raises for a refused write and an
# unreachable read.
LOST_TURN_ERROR = "RuntimeError"
UNREADABLE_STORE_ERROR = "HTTPError"


class GatedAgent(AgentBase[ThreadInput, AnswerOutput, ThreadDeps]):
    """Holds every run until all CONCURRENT_RUNS are in flight at once."""

    async def _process_impl(self, input: ThreadInput) -> AnswerOutput:
        in_flight["count"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["count"])
        if in_flight["count"] == CONCURRENT_RUNS:
            all_in_flight.set()
        await asyncio.wait_for(all_in_flight.wait(), timeout=30)
        in_flight["count"] -= 1
        return AnswerOutput(answer=f"gated reply to: {input.query}")


HELD_PHASE = "searching"
# How long a held run waits for a client that never hangs up.
HOLD_SECONDS = 20.0
# A hung-up run's turn reads back within this; measured well under a second.
CANCELLED_SAVE_BUDGET_SECONDS = 15.0
CANCELLED_RUNS = 4
held: Dict[str, Any] = {"barrier": None, "cancelled": []}


class HeldAgent(AgentBase[ThreadInput, AnswerOutput, ThreadDeps]):
    """Reports a phase, waits for every run held with it when a barrier is
    set, then holds its turn until the client hangs up."""

    async def _process_impl(self, input: ThreadInput) -> AnswerOutput:
        self.emit_progress(HELD_PHASE, f"holding {input.query}")
        try:
            await asyncio.sleep(HOLD_SECONDS)
        except asyncio.CancelledError:
            held["cancelled"].append(input.query)
            raise
        return AnswerOutput(answer=f"held reply to: {input.query}")


_AGENT_CLASSES = {
    "streaming_agent": f"{__name__}:StreamingAgent",
    "dispatch_agent": f"{__name__}:DispatchAgent",
    "failing_agent": f"{__name__}:FailingAgent",
    "gated_agent": f"{__name__}:GatedAgent",
    "held_agent": f"{__name__}:HeldAgent",
}
_TOKEN_STREAMING = {"streaming_agent": True}


@pytest.fixture(scope="module")
def memory(vespa_instance, config_manager):
    with memory_on_vespa(vespa_instance, config_manager, (TENANT_A, TENANT_B)) as (
        managers,
        proxy,
    ):
        yield managers, proxy


@pytest.fixture(scope="module")
def dispatcher(config_manager, memory):
    managers, _ = memory
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    for name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url="http://localhost:8000",
                capabilities=["ag_ui"],
                streams_answer_tokens=_TOKEN_STREAMING.get(name, False),
            )
        )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    dispatcher._conversation_store_factory = lambda tenant_id: ConversationStore(
        managers[tenant_id], tenant_id
    )
    yield dispatcher
    for name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(name, None)


@pytest.fixture()
async def client(dispatcher, conversation_ledger, continuation_store, memory):
    _, proxy = memory
    dispatcher.set_conversation_ledger(conversation_ledger)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A, KEY_B: TENANT_B})
    openai_compat.set_key_resolver(None)
    openai_compat.set_continuation_store(continuation_store)
    app = FastAPI()
    app.include_router(ag_ui.router, prefix="/ag-ui")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=120.0
    ) as http_client:
        yield http_client
    proxy.intercept = None
    await dispatcher.drain_conversation_saves()
    dispatcher.set_conversation_ledger(None)
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_continuation_store(None)


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _thread(request) -> str:
    return f"{request.node.name}-{uuid.uuid4().hex[:8]}"


def _events(raw: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]


async def _run(
    client: httpx.AsyncClient,
    agent: str,
    thread: str,
    messages: List[Dict[str, Any]],
    key: str = KEY_A,
) -> List[Dict[str, Any]]:
    response = await client.post(
        f"/ag-ui/{agent}",
        json={
            "threadId": thread,
            "runId": f"run-{len(messages)}",
            "state": {},
            "messages": messages,
            "tools": [],
            "context": [],
            "forwardedProps": {},
        },
        headers=_auth(key),
    )
    assert response.status_code == 200, response.text
    return _events(response.text)


async def _read(client: httpx.AsyncClient, thread: str, key: str = KEY_A):
    response = await client.get(f"/ag-ui/threads/{thread}", headers=_auth(key))
    return response.status_code, response.json()


def _user(content: str, index: int) -> Dict[str, Any]:
    return {"id": f"u{index}", "role": "user", "content": content}


def _assistant(content: str, index: int) -> Dict[str, Any]:
    return {"id": f"a{index}", "role": "assistant", "content": content}


class TestThreadRoundTrip:
    async def test_each_runs_turn_reads_back_in_order(self, client, request):
        thread = _thread(request)
        first = await _run(client, "streaming_agent", thread, [_user("cats", 1)])
        assert first[-1]["type"] == "RUN_FINISHED"
        second = await _run(
            client,
            "dispatch_agent",
            thread,
            [
                _user("cats", 1),
                _assistant("streamed reply to: cats", 1),
                _user("and dogs?", 2),
            ],
        )
        assert second[-1]["type"] == "RUN_FINISHED"

        assert await _read(client, thread) == (
            200,
            {
                "thread_id": thread,
                "state": "loaded",
                "reason": None,
                "turns": [
                    {"role": "user", "content": "cats"},
                    {"role": "assistant", "content": "streamed reply to: cats"},
                    {"role": "user", "content": "and dogs?"},
                    {
                        "role": "assistant",
                        "content": "dispatched reply to: and dogs? (history 2)",
                    },
                ],
            },
        )

    async def test_a_failed_run_saves_the_users_message_alone(self, client, request):
        thread = _thread(request)
        events = await _run(client, "failing_agent", thread, [_user("break", 1)])
        # The run opens its "starting" step before the agent fails.
        assert [event["type"] for event in events] == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "STEP_FINISHED",
            "RUN_ERROR",
        ]

        assert await _read(client, thread) == (
            200,
            {
                "thread_id": thread,
                "state": "loaded",
                "reason": None,
                "turns": [{"role": "user", "content": "break"}],
            },
        )

    async def test_a_thread_never_run_has_no_turns(self, client, request):
        thread = _thread(request)
        assert await _read(client, thread) == (
            200,
            {"thread_id": thread, "state": "loaded", "reason": None, "turns": []},
        )


class TestIsolation:
    async def test_another_tenant_reading_the_same_thread_id_sees_nothing(
        self, client, request
    ):
        thread = _thread(request)
        await _run(client, "dispatch_agent", thread, [_user("tenant a only", 1)])

        assert await _read(client, thread, key=KEY_B) == (
            200,
            {"thread_id": thread, "state": "loaded", "reason": None, "turns": []},
        )
        status, body = await _read(client, thread)
        assert (status, body["turns"]) == (
            200,
            [
                {"role": "user", "content": "tenant a only"},
                {
                    "role": "assistant",
                    "content": "dispatched reply to: tenant a only (history 0)",
                },
            ],
        )

    async def test_a_read_without_a_key_is_refused(self, client):
        response = await client.get("/ag-ui/threads/anything")
        assert (response.status_code, response.json()["error"]["code"]) == (
            401,
            openai_compat.UNAUTHORIZED["code"],
        )


class TestConcurrency:
    async def test_runs_in_flight_together_each_keep_their_own_thread(
        self, client, request
    ):
        in_flight.update(count=0, peak=0)
        all_in_flight.clear()
        threads = [f"{_thread(request)}-{index}" for index in range(CONCURRENT_RUNS)]
        runs = await asyncio.gather(
            *(
                _run(client, "gated_agent", thread, [_user(f"query {index}", 1)])
                for index, thread in enumerate(threads)
            )
        )
        assert in_flight["peak"] == CONCURRENT_RUNS
        assert [run[-1]["type"] for run in runs] == ["RUN_FINISHED"] * CONCURRENT_RUNS

        reads = await asyncio.gather(*(_read(client, thread) for thread in threads))
        assert [read for read in reads] == [
            (
                200,
                {
                    "thread_id": thread,
                    "state": "loaded",
                    "reason": None,
                    "turns": [
                        {"role": "user", "content": f"query {index}"},
                        {
                            "role": "assistant",
                            "content": f"gated reply to: query {index}",
                        },
                    ],
                },
            )
            for index, thread in enumerate(threads)
        ]


class TestFaults:
    async def test_an_unreachable_store_answers_503_not_an_empty_thread(
        self, client, memory, request
    ):
        _, proxy = memory
        thread = _thread(request)
        await _run(client, "dispatch_agent", thread, [_user("kept", 1)])
        await _read(client, thread)
        proxy.intercept = lambda method, path, body: (503, {"error": "down"})

        status, body = await _read(client, thread)

        assert status == 503
        error_type = body["error"]["error_type"]
        assert body == {
            "error": {
                "message": f"The conversation store is unavailable ({error_type}). "
                "See server logs for detail.",
                "type": "server_error",
                "code": "service_unavailable",
                "error_type": error_type,
            }
        }
        assert error_type == UNREADABLE_STORE_ERROR

    async def test_a_turn_the_store_refused_reads_back_as_incomplete(
        self, client, dispatcher, memory, request
    ):
        _, proxy = memory
        thread = _thread(request)
        proxy.intercept = lambda method, path, body: (
            (400, {"error": "refused"}) if "/document/v1/" in path else None
        )
        events = await _run(client, "dispatch_agent", thread, [_user("lost", 1)])
        # The reply reached the client; its save fails in the background.
        assert events[-1]["type"] == "RUN_FINISHED"
        assert await dispatcher.drain_conversation_saves() is True
        proxy.intercept = None

        status, body = await _read(client, thread)
        assert (status, body) == (
            200,
            {
                "thread_id": thread,
                "state": "incomplete",
                "reason": f"a turn was not saved ({LOST_TURN_ERROR})",
                "turns": [],
            },
        )

    async def test_an_unreachable_ledger_fails_the_answered_run_and_the_read(
        self, client, dispatcher, request
    ):
        thread = _thread(request)
        dead = Redis.from_url(
            f"redis://127.0.0.1:{free_port()}",
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                dead, save_lease_s=5, failure_capacity=10, key_prefix="test:dead"
            )
        )
        try:
            events = await _run(client, "dispatch_agent", thread, [_user("no", 1)])
            status, body = await _read(client, thread)
        finally:
            await dead.aclose()

        assert [event["type"] for event in events] == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "STEP_FINISHED",
            "RUN_ERROR",
        ]
        assert events[-1] == {
            "type": "RUN_ERROR",
            "message": "The reply was not saved to this conversation "
            "(SessionStateUnavailable). See server logs for detail.",
            "code": "conversation_not_saved",
        }
        assert (status, body) == (
            503,
            {
                "error": {
                    "message": "The conversation ledger is unavailable "
                    "(SessionStateUnavailable). See server logs for detail.",
                    "type": "server_error",
                    "code": "service_unavailable",
                    "error_type": "SessionStateUnavailable",
                }
            },
        )


@pytest.fixture()
def live_server(dispatcher, continuation_store, memory, workflow_state_redis_url):
    """The AG-UI routes on a real socket, so a client hanging up is a real
    disconnect, with the conversation ledger opened on the server's loop."""
    _, proxy = memory

    @asynccontextmanager
    async def ledger_on_server_loop(_app):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                redis,
                save_lease_s=CONVERSATION_SAVE_LEASE_S,
                failure_capacity=CONVERSATION_PERSIST_FAILURE_CAPACITY,
                key_prefix=f"test:conversation:{uuid.uuid4().hex}",
            )
        )
        try:
            yield
        finally:
            await dispatcher.drain_conversation_saves()
            dispatcher.set_conversation_ledger(None)
            await redis.aclose()

    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A, KEY_B: TENANT_B})
    openai_compat.set_key_resolver(None)
    openai_compat.set_continuation_store(continuation_store)
    held["cancelled"] = []
    app = FastAPI(lifespan=ledger_on_server_loop)
    app.include_router(ag_ui.router, prefix="/ag-ui")
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=free_port(), log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{server.config.port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        proxy.intercept = None
        openai_compat.set_dispatcher_provider(None)
        openai_compat.set_api_keys({})
        openai_compat.set_continuation_store(None)
    assert not thread.is_alive()


def _hang_up_after_the_held_phase(
    base: str, thread: str, query: str, before_hang_up=lambda: None
) -> None:
    """Run held_agent and hang up once it reports its phase (and
    ``before_hang_up`` returns)."""
    with httpx.stream(
        "POST",
        f"{base}/ag-ui/held_agent",
        json={
            "threadId": thread,
            "runId": "run-1",
            "state": {},
            "messages": [_user(query, 1)],
            "tools": [],
            "context": [],
            "forwardedProps": {},
        },
        headers=_auth(KEY_A),
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            if event.get("type") == "CUSTOM" and event["value"]["phase"] == HELD_PHASE:
                before_hang_up()
                return
    raise AssertionError(f"held_agent never reported {HELD_PHASE!r}")


def _cancelled_thread(thread: str, query: str) -> Dict[str, Any]:
    return {
        "thread_id": thread,
        "state": "loaded",
        "reason": None,
        "turns": [
            {"role": "user", "content": query},
            {"role": RUN_CANCELLED_ROLE, "content": RUN_CANCELLED_TEXT},
        ],
    }


def _read_until(base: str, thread: str, expected: Dict[str, Any]):
    """Read the thread until it is ``expected`` or the save budget is spent;
    the last read is returned either way."""
    deadline = time.monotonic() + CANCELLED_SAVE_BUDGET_SECONDS
    while True:
        response = httpx.get(
            f"{base}/ag-ui/threads/{thread}", headers=_auth(KEY_A), timeout=30.0
        )
        read = (response.status_code, response.json())
        if read == (200, expected) or time.monotonic() > deadline:
            return read
        time.sleep(0.1)


class TestCancelledRun:
    """A run the client hangs up on before its reply keeps its user message
    and a cancelled marker, so a reloaded page shows both."""

    def test_a_hung_up_run_saves_the_message_and_a_cancelled_marker(
        self, live_server, request
    ):
        thread = _thread(request)
        _hang_up_after_the_held_phase(live_server, thread, "stop me")

        expected = _cancelled_thread(thread, "stop me")
        assert _read_until(live_server, thread, expected) == (200, expected)
        assert held["cancelled"] == ["stop me"]
        assert openai_compat.in_flight_count() == 0

    def test_runs_hung_up_together_each_keep_their_own_thread(
        self, live_server, request
    ):
        threads = [f"{_thread(request)}-{index}" for index in range(CANCELLED_RUNS)]
        barrier = threading.Barrier(CANCELLED_RUNS)

        def hang_up(index: int) -> None:
            # Every run is in flight before any of them hangs up.
            _hang_up_after_the_held_phase(
                live_server,
                threads[index],
                f"cancel {index}",
                before_hang_up=lambda: barrier.wait(timeout=30),
            )

        workers = [
            threading.Thread(target=hang_up, args=(index,))
            for index in range(CANCELLED_RUNS)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)

        reads = [
            _read_until(
                live_server, thread, _cancelled_thread(thread, f"cancel {index}")
            )
            for index, thread in enumerate(threads)
        ]
        assert reads == [
            (200, _cancelled_thread(thread, f"cancel {index}"))
            for index, thread in enumerate(threads)
        ]
        assert sorted(held["cancelled"]) == [
            f"cancel {index}" for index in range(CANCELLED_RUNS)
        ]

    def test_a_cancelled_save_the_ledger_refuses_is_logged_not_hung(
        self, live_server, dispatcher, request, caplog
    ):
        thread = _thread(request)
        dead = Redis.from_url(
            f"redis://127.0.0.1:{free_port()}",
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                dead, save_lease_s=5, failure_capacity=10, key_prefix="test:dead"
            )
        )
        with caplog.at_level(logging.ERROR, logger=ag_ui.logger.name):
            _hang_up_after_the_held_phase(live_server, thread, "lost")
            deadline = time.monotonic() + CANCELLED_SAVE_BUDGET_SECONDS
            while time.monotonic() < deadline and not [
                record for record in caplog.records if thread in record.getMessage()
            ]:
                time.sleep(0.05)

        assert [
            (record.levelname, record.getMessage(), record.exc_info[0].__name__)
            for record in caplog.records
            if thread in record.getMessage()
        ] == [
            (
                "ERROR",
                f"ag-ui thread {thread}: the cancelled held_agent run's message "
                "was not saved",
                "SessionStateUnavailable",
            )
        ]
        assert openai_compat.in_flight_count() == 0
