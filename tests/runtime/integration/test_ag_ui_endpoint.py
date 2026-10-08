"""The AG-UI surface over the real dispatcher, Redis and a real socket.

Real router -> real ``AgentDispatcher`` (both the token-stream and the
dispatch paths) -> deterministic agents, with suspended turns kept in the real
Redis continuation store. Every event a run streams is decoded and pinned in
order, so the event grammar a CopilotKit client relies on (start/content/end
pairing, one terminal event, the step brackets) is the contract under test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from typing import Any, Dict, List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from redis.asyncio import Redis

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.agent_registry_store import RedisAgentRegistryStore
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import ag_ui, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

TENANT_A = "acme:acme"
TENANT_B = "beta:prod"
KEY_A = "ag-ui-key-tenant-a"
KEY_B = "ag-ui-key-tenant-b"
QUERY = "Which clips show the tower at night?"
SUMMARY_BODY = "Three clips show the tower lit at night, the clearest at 0:42."
PLAN_TOKENS = "Find night clips. Rank by clarity."
STATUS_PHASE = "retrieval"
STATUS_MESSAGE = "Searching 2 profiles"
RANK_PHASE = "ranking"
RANK_MESSAGE = "Ranking 4 hits"
START_PHASE = "starting"
RESULTS = [{"id": "video-7", "score": 0.91}, {"id": "video-2", "score": 0.64}]
TOOL_CALL_ID = "call_ag_ui_write_file"
THREAD_ID = "thread-1"
RUN_ID = "run-1"
DISCONNECT_CANCEL_BUDGET_SECONDS = 3.0
# How long a gated agent holds its turn for a client that never releases it.
GATE_SECONDS = 10.0
INTERLEAVED_RUNS = 8


def _token_chunks(text: str, size: int = 11) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class AgUiDeps(AgentDeps):
    pass


class AgUiInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    attachments: list = []
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class SearchStreamOutput(AgentOutput):
    summary: str = ""
    results: list = []


class SearchStreamAgent(AgentBase[AgUiInput, SearchStreamOutput, AgUiDeps]):
    """A status phase, a non-answer token field, then the answer tokens."""

    async def _process_impl(self, input: AgUiInput) -> SearchStreamOutput:
        self.emit_progress(STATUS_PHASE, STATUS_MESSAGE)
        accumulated = ""
        for chunk in _token_chunks(PLAN_TOKENS):
            accumulated += chunk
            self.emit_progress(
                "token",
                chunk,
                data={"accumulated": accumulated, "output_field": "sub_questions"},
            )
            await asyncio.sleep(0)
        self.emit_progress(RANK_PHASE, RANK_MESSAGE)
        summary = f"[{input.tenant_id}] {SUMMARY_BODY}"
        accumulated = ""
        for chunk in _token_chunks(summary):
            accumulated += chunk
            self.emit_progress(
                "token",
                chunk,
                data={"accumulated": accumulated, "output_field": "summary"},
            )
            await asyncio.sleep(0)
        return SearchStreamOutput(summary=summary, results=list(RESULTS))


class ContextOutput(AgentOutput):
    answer: str = ""


class ContextAgent(AgentBase[AgUiInput, ContextOutput, AgUiDeps]):
    """Answers with what the dispatch handed it, as sorted JSON."""

    async def _process_impl(self, input: AgUiInput) -> ContextOutput:
        return ContextOutput(
            answer=json.dumps(
                {
                    "query": input.query,
                    "tenant_id": input.tenant_id,
                    "history": input.conversation_history,
                    "attachments": input.attachments,
                    "tool_names": [
                        tool["function"]["name"] for tool in input.external_tools
                    ],
                },
                sort_keys=True,
            )
        )


class ToolOutput(AgentOutput):
    answer: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class ToolAgent(AgentBase[AgUiInput, ToolOutput, AgUiDeps]):
    """Suspends on the client's first tool, then answers with what resumed it."""

    async def _process_impl(self, input: AgUiInput) -> ToolOutput:
        if not input.tool_results:
            return ToolOutput(
                pending_tool_calls=[
                    {
                        "id": TOOL_CALL_ID,
                        "name": input.external_tools[0]["function"]["name"],
                        "arguments": {"text": input.query},
                    }
                ],
                continuation_state={"plan": f"plan-for-{input.tenant_id}"},
            )
        return ToolOutput(
            answer=json.dumps(
                {
                    "tenant_id": input.tenant_id,
                    "resumed_state": input.continuation_state,
                    "results": [result["content"] for result in input.tool_results],
                },
                sort_keys=True,
            )
        )


class FailingOutput(AgentOutput):
    summary: str = ""


class FailingAgent(AgentBase[AgUiInput, FailingOutput, AgUiDeps]):
    async def _process_impl(self, input: AgUiInput) -> FailingOutput:
        raise RuntimeError(f"secret-backend-detail-{input.query}")


slow_agent_events: List[str] = []


class SlowOutput(AgentOutput):
    summary: str = ""


class SlowAgent(AgentBase[AgUiInput, SlowOutput, AgUiDeps]):
    """Streams one token, then holds the turn open for 20 s."""

    async def _process_impl(self, input: AgUiInput) -> SlowOutput:
        slow_agent_events.append("started")
        self.emit_progress(
            "token", "first ", data={"accumulated": "first ", "output_field": "summary"}
        )
        try:
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            slow_agent_events.append("cancelled")
            raise
        slow_agent_events.append("completed")
        return SlowOutput(summary="first done")


gated_release = threading.Event()


class GatedAgent(AgentBase[AgUiInput, SlowOutput, AgUiDeps]):
    """Reports a phase from the loop and one from a worker thread, then holds
    the turn until the client releases it."""

    async def _process_impl(self, input: AgUiInput) -> SlowOutput:
        self.emit_progress(STATUS_PHASE, f"{input.tenant_id} searching")
        await asyncio.to_thread(
            self.emit_progress, RANK_PHASE, f"{input.tenant_id} ranking"
        )
        released = await asyncio.to_thread(gated_release.wait, GATE_SECONDS)
        return SlowOutput(summary=f"released={released}")


interleave: Dict[str, asyncio.Barrier] = {}


class InterleavedAgent(AgentBase[AgUiInput, SlowOutput, AgUiDeps]):
    """Reports a phase, waits until every concurrent run has reported its
    own, then reports a second."""

    async def _process_impl(self, input: AgUiInput) -> SlowOutput:
        self.emit_progress("first", f"{input.tenant_id} first")
        async with asyncio.timeout(GATE_SECONDS):
            await interleave["barrier"].wait()
        self.emit_progress("second", f"{input.tenant_id} second")
        return SlowOutput(summary=f"{input.tenant_id} done")


class FailingAfterStatusAgent(AgentBase[AgUiInput, FailingOutput, AgUiDeps]):
    async def _process_impl(self, input: AgUiInput) -> FailingOutput:
        self.emit_progress(STATUS_PHASE, STATUS_MESSAGE)
        raise RuntimeError(f"secret-backend-detail-{input.query}")


class SlowDispatchAgent(AgentBase[AgUiInput, SlowOutput, AgUiDeps]):
    """Reports a phase, then holds the turn open for 20 s."""

    async def _process_impl(self, input: AgUiInput) -> SlowOutput:
        slow_agent_events.append("started")
        self.emit_progress(STATUS_PHASE, STATUS_MESSAGE)
        try:
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            slow_agent_events.append("cancelled")
            raise
        slow_agent_events.append("completed")
        return SlowOutput(summary="done")


_AGENT_CLASSES = {
    "search_stream_agent": f"{__name__}:SearchStreamAgent",
    "search_dispatch_agent": f"{__name__}:SearchStreamAgent",
    "context_agent": f"{__name__}:ContextAgent",
    "tool_agent": f"{__name__}:ToolAgent",
    "failing_stream_agent": f"{__name__}:FailingAgent",
    "failing_dispatch_agent": f"{__name__}:FailingAgent",
    "slow_agent": f"{__name__}:SlowAgent",
    "gated_agent": f"{__name__}:GatedAgent",
    "interleaved_agent": f"{__name__}:InterleavedAgent",
    "failing_after_status_agent": f"{__name__}:FailingAfterStatusAgent",
    "slow_dispatch_agent": f"{__name__}:SlowDispatchAgent",
}

_TOKEN_STREAMING = {
    "search_stream_agent": True,
    "search_dispatch_agent": False,
    "context_agent": False,
    "tool_agent": False,
    "failing_stream_agent": True,
    "failing_dispatch_agent": False,
    "slow_agent": True,
    "gated_agent": False,
    "interleaved_agent": False,
    "failing_after_status_agent": False,
    "slow_dispatch_agent": False,
}


def _registry(config_manager: ConfigManager) -> AgentRegistry:
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    for agent_name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=agent_name,
                url="http://localhost:8000",
                capabilities=["ag_ui"],
                streams_answer_tokens=_TOKEN_STREAMING[agent_name],
            )
        )
    return registry


@pytest.fixture(scope="module")
def config_manager():
    store = InMemoryConfigStore()
    store.initialize()
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    yield ConfigManager(store=store)
    for agent_name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(agent_name, None)


@pytest.fixture(scope="module")
def dispatcher(config_manager):
    dispatcher = AgentDispatcher(
        agent_registry=_registry(config_manager),
        config_manager=config_manager,
        schema_loader=None,
    )
    # No conversation memory is configured: a run's turn takes its place in
    # the ledger and is not stored (tests/runtime/integration/
    # test_ag_ui_threads.py stores and reads them back).
    dispatcher._conversation_store_factory = lambda tenant_id: None
    return dispatcher


@pytest.fixture()
def ag_ui_app(dispatcher, continuation_store, conversation_ledger):
    dispatcher.set_conversation_ledger(conversation_ledger)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A, KEY_B: TENANT_B})
    openai_compat.set_key_resolver(None)
    openai_compat.set_continuation_store(continuation_store)
    slow_agent_events.clear()
    gated_release.clear()
    app = FastAPI()
    app.include_router(ag_ui.router, prefix="/ag-ui")
    yield app
    dispatcher.set_conversation_ledger(None)
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_continuation_store(None)


@pytest.fixture()
async def client(ag_ui_app):
    transport = httpx.ASGITransport(app=ag_ui_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=60.0
    ) as http_client:
        yield http_client


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _run(messages: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "threadId": THREAD_ID,
        "runId": RUN_ID,
        "state": {},
        "messages": messages,
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    body.update(overrides)
    return body


def _user(content: Any, message_id: str = "u1") -> Dict[str, Any]:
    return {"id": message_id, "role": "user", "content": content}


def _events(raw: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]


def _types(events: List[Dict[str, Any]]) -> List[str]:
    return [event["type"] for event in events]


def _text(events: List[Dict[str, Any]]) -> str:
    return "".join(
        event["delta"] for event in events if event["type"] == "TEXT_MESSAGE_CONTENT"
    )


@pytest.fixture()
def live_server(ag_ui_app):
    port = _free_port()
    config = uvicorn.Config(ag_ui_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=20)
    assert not thread.is_alive()


def _opening_step(agent_name: str) -> List[Dict[str, Any]]:
    """The step every run opens with, before the agent reports a phase."""
    return [
        {"type": "STEP_STARTED", "stepName": START_PHASE},
        {
            "type": "CUSTOM",
            "name": ag_ui.STATUS_EVENT,
            "value": {"phase": START_PHASE, "message": f"Running {agent_name}"},
        },
        {"type": "STEP_FINISHED", "stepName": START_PHASE},
    ]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write text to a file in the user's workspace.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


class TestTokenStreamRun:
    """A token-streaming agent's run, event by event."""

    async def test_the_run_streams_steps_reply_result_and_finish_in_order(self, client):
        response = await client.post(
            "/ag-ui/search_stream_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        summary = f"[{TENANT_A}] {SUMMARY_BODY}"
        assert _types(events) == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "STEP_FINISHED",
            "STEP_STARTED",
            "CUSTOM",
            "STEP_FINISHED",
            "STEP_STARTED",
            "CUSTOM",
            "TEXT_MESSAGE_START",
            *["TEXT_MESSAGE_CONTENT"] * len(_token_chunks(summary)),
            "TEXT_MESSAGE_END",
            "STATE_SNAPSHOT",
            "STEP_FINISHED",
            "RUN_FINISHED",
        ]
        assert events[0] == {
            "type": "RUN_STARTED",
            "threadId": THREAD_ID,
            "runId": RUN_ID,
        }
        assert events[1:9] == [
            {"type": "STEP_STARTED", "stepName": START_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {
                    "phase": START_PHASE,
                    "message": "Running search_stream_agent",
                },
            },
            {"type": "STEP_FINISHED", "stepName": START_PHASE},
            {"type": "STEP_STARTED", "stepName": STATUS_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": STATUS_PHASE, "message": STATUS_MESSAGE},
            },
            {"type": "STEP_FINISHED", "stepName": STATUS_PHASE},
            {"type": "STEP_STARTED", "stepName": RANK_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": RANK_PHASE, "message": RANK_MESSAGE},
            },
        ]
        message_id = events[9]["messageId"]
        assert events[9] == {
            "type": "TEXT_MESSAGE_START",
            "messageId": message_id,
            "role": "assistant",
        }
        assert {
            event["messageId"]
            for event in events
            if event["type"].startswith("TEXT_MESSAGE")
        } == {message_id}
        assert [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ] == _token_chunks(summary)
        assert _text(events) == summary
        assert PLAN_TOKENS[:11] not in response.text
        assert events[-3] == {
            "type": "STATE_SNAPSHOT",
            "snapshot": {
                "agent": "search_stream_agent",
                "result": SearchStreamOutput(
                    summary=summary, results=list(RESULTS)
                ).model_dump(),
            },
        }
        assert events[-2] == {"type": "STEP_FINISHED", "stepName": RANK_PHASE}
        assert events[-1] == {
            "type": "RUN_FINISHED",
            "threadId": THREAD_ID,
            "runId": RUN_ID,
            "outcome": {"type": "success"},
        }

    async def test_the_streamed_reply_equals_the_dispatched_reply(self, client):
        streamed, dispatched = await asyncio.gather(
            client.post(
                "/ag-ui/search_stream_agent",
                json=_run([_user(QUERY)]),
                headers=_auth(KEY_A),
            ),
            client.post(
                "/ag-ui/search_dispatch_agent",
                json=_run([_user(QUERY)]),
                headers=_auth(KEY_A),
            ),
        )

        assert _text(_events(streamed.text)) == _text(_events(dispatched.text))
        assert _text(_events(dispatched.text)) == f"[{TENANT_A}] {SUMMARY_BODY}"


class TestDispatchRun:
    """An agent without answer-token streaming runs on the dispatch path."""

    async def test_the_agents_phases_stream_as_steps_before_the_reply(self, client):
        """The agent's own phases arrive as steps with their status; its
        token events are not the reply, which is the dispatched answer."""
        response = await client.post(
            "/ag-ui/search_dispatch_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
        )

        events = _events(response.text)
        summary = f"[{TENANT_A}] {SUMMARY_BODY}"
        message_id = events[9]["messageId"]
        assert events == [
            {"type": "RUN_STARTED", "threadId": THREAD_ID, "runId": RUN_ID},
            {"type": "STEP_STARTED", "stepName": START_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {
                    "phase": START_PHASE,
                    "message": "Running search_dispatch_agent",
                },
            },
            {"type": "STEP_FINISHED", "stepName": START_PHASE},
            {"type": "STEP_STARTED", "stepName": STATUS_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": STATUS_PHASE, "message": STATUS_MESSAGE},
            },
            {"type": "STEP_FINISHED", "stepName": STATUS_PHASE},
            {"type": "STEP_STARTED", "stepName": RANK_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": RANK_PHASE, "message": RANK_MESSAGE},
            },
            {
                "type": "TEXT_MESSAGE_START",
                "messageId": message_id,
                "role": "assistant",
            },
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": summary,
            },
            {"type": "TEXT_MESSAGE_END", "messageId": message_id},
            {
                "type": "STATE_SNAPSHOT",
                "snapshot": {
                    "agent": "search_dispatch_agent",
                    "result": {
                        "status": "success",
                        "agent": "search_dispatch_agent",
                        "summary": summary,
                        "results": RESULTS,
                        "answer": summary,
                    },
                },
            },
            {"type": "STEP_FINISHED", "stepName": RANK_PHASE},
            {
                "type": "RUN_FINISHED",
                "threadId": THREAD_ID,
                "runId": RUN_ID,
                "outcome": {"type": "success"},
            },
        ]
        assert message_id.startswith("msg-")

    def test_each_phase_reaches_the_client_while_the_turn_still_runs(self, live_server):
        """The agent holds its turn until the client has seen both of its
        phases, the one reported from a worker thread included. A run that
        sent its events only once the turn ended would answer
        ``released=False``."""
        events: List[Dict[str, Any]] = []
        with httpx.stream(
            "POST",
            f"{live_server}/ag-ui/gated_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
            timeout=30.0,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[len("data: ") :])
                    events.append(event)
                    if (
                        event["type"] == "CUSTOM"
                        and event["value"]["phase"] == RANK_PHASE
                    ):
                        gated_release.set()

        message_id = events[9]["messageId"]
        assert events[:9] == [
            {"type": "RUN_STARTED", "threadId": THREAD_ID, "runId": RUN_ID},
            *_opening_step("gated_agent"),
            {"type": "STEP_STARTED", "stepName": STATUS_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": STATUS_PHASE, "message": f"{TENANT_A} searching"},
            },
            {"type": "STEP_FINISHED", "stepName": STATUS_PHASE},
            {"type": "STEP_STARTED", "stepName": RANK_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": RANK_PHASE, "message": f"{TENANT_A} ranking"},
            },
        ]
        assert events[9:12] == [
            {
                "type": "TEXT_MESSAGE_START",
                "messageId": message_id,
                "role": "assistant",
            },
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": "released=True",
            },
            {"type": "TEXT_MESSAGE_END", "messageId": message_id},
        ]
        assert _types(events[12:]) == [
            "STATE_SNAPSHOT",
            "STEP_FINISHED",
            "RUN_FINISHED",
        ]
        assert events[13] == {"type": "STEP_FINISHED", "stepName": RANK_PHASE}

    async def test_the_conversation_reaches_the_agent_in_dispatch_form(self, client):
        """History, a developer note, an image part and the tool names all
        arrive as the dispatcher's own inputs; the client's activity message
        does not."""
        messages = [
            {"id": "d1", "role": "developer", "content": "Answer briefly."},
            _user("Show me the tower.", "u1"),
            {"id": "a1", "role": "assistant", "content": "Here are four clips."},
            {
                "id": "act1",
                "role": "activity",
                "activityType": "search_results",
                "content": {"hits": 4},
            },
            _user(
                [
                    {"type": "text", "text": QUERY},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "value": "https://example.test/frame.png",
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "data",
                            "value": "aGVsbG8=",
                            "mimeType": "image/png",
                        },
                    },
                ],
                "u2",
            ),
        ]

        response = await client.post(
            "/ag-ui/context_agent",
            json=_run(messages, tools=[WRITE_FILE_TOOL]),
            headers=_auth(KEY_A),
        )

        assert json.loads(_text(_events(response.text))) == {
            "attachments": [
                "https://example.test/frame.png",
                "data:image/png;base64,aGVsbG8=",
            ],
            "history": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Show me the tower."},
                {"role": "assistant", "content": "Here are four clips."},
            ],
            "query": QUERY,
            "tenant_id": TENANT_A,
            "tool_names": ["write_file"],
        }


class TestFrontendToolRoundTrip:
    """A suspended turn resumes from the shared store on the next run."""

    async def test_the_agent_suspends_on_a_tool_and_resumes_with_its_result(
        self, client
    ):
        first = await client.post(
            "/ag-ui/tool_agent",
            json=_run([_user(QUERY)], tools=[WRITE_FILE_TOOL]),
            headers=_auth(KEY_A),
        )

        events = _events(first.text)
        assert _types(events) == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "STEP_FINISHED",
            "RUN_FINISHED",
        ]
        assert events[1:3] == _opening_step("tool_agent")[:2]
        assert events[3]["toolCallId"] == TOOL_CALL_ID
        assert events[3]["toolCallName"] == "write_file"
        assert json.loads(events[4]["delta"]) == {"text": QUERY}
        assert events[5] == {"type": "TOOL_CALL_END", "toolCallId": TOOL_CALL_ID}
        assert events[6] == {"type": "STEP_FINISHED", "stepName": START_PHASE}
        assert events[7]["outcome"] == {
            "type": "success",
            "pendingToolCallIds": [TOOL_CALL_ID],
        }

        resumed = await client.post(
            "/ag-ui/tool_agent",
            json=_run(
                [
                    _user(QUERY),
                    {
                        "id": events[3]["parentMessageId"],
                        "role": "assistant",
                        "toolCalls": [
                            {
                                "id": TOOL_CALL_ID,
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": events[4]["delta"],
                                },
                            }
                        ],
                    },
                    {
                        "id": "t1",
                        "role": "tool",
                        "toolCallId": TOOL_CALL_ID,
                        "content": "written to notes.md",
                    },
                ],
                runId="run-2",
                tools=[WRITE_FILE_TOOL],
            ),
            headers=_auth(KEY_A),
        )

        assert json.loads(_text(_events(resumed.text))) == {
            "results": ["written to notes.md"],
            "resumed_state": {"plan": f"plan-for-{TENANT_A}"},
            "tenant_id": TENANT_A,
        }
        assert _events(resumed.text)[-1]["runId"] == "run-2"

    async def test_another_tenant_replaying_the_transcript_gets_no_state(self, client):
        first = await client.post(
            "/ag-ui/tool_agent",
            json=_run([_user(QUERY)], tools=[WRITE_FILE_TOOL]),
            headers=_auth(KEY_A),
        )
        args = _events(first.text)[4]["delta"]
        replay = [
            _user(QUERY),
            {
                "id": "a1",
                "role": "assistant",
                "toolCalls": [
                    {
                        "id": TOOL_CALL_ID,
                        "type": "function",
                        "function": {"name": "write_file", "arguments": args},
                    }
                ],
            },
            {"id": "t1", "role": "tool", "toolCallId": TOOL_CALL_ID, "content": "x"},
        ]

        stolen = await client.post(
            "/ag-ui/tool_agent",
            json=_run(replay, tools=[WRITE_FILE_TOOL]),
            headers=_auth(KEY_B),
        )

        assert json.loads(_text(_events(stolen.text))) == {
            "results": ["x"],
            "resumed_state": {},
            "tenant_id": TENANT_B,
        }


class TestRequestBoundary:
    """What is refused before a run starts."""

    async def test_a_missing_key_is_401(self, client):
        response = await client.post(
            "/ag-ui/search_stream_agent", json=_run([_user(QUERY)])
        )

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {
            "error": {
                "message": "Invalid or missing API key.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }

    async def test_an_unregistered_agent_is_404(self, client):
        response = await client.post(
            "/ag-ui/no_such_agent", json=_run([_user(QUERY)]), headers=_auth(KEY_A)
        )

        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "message": "Agent 'no_such_agent' is not registered.",
                "type": "invalid_request_error",
                "code": "agent_not_found",
            }
        }

    async def test_a_body_that_is_not_a_run_input_is_400(self, client):
        response = await client.post(
            "/ag-ui/search_stream_agent",
            json={"threadId": THREAD_ID, "messages": []},
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "Invalid AG-UI run input: runId: Field required",
            "type": "invalid_request_error",
            "code": "invalid_request",
        }

    async def test_an_audio_part_is_refused_naming_its_position(self, client):
        audio = {
            "type": "audio",
            "source": {"type": "url", "value": "https://example.test/a.wav"},
        }

        response = await client.post(
            "/ag-ui/context_agent",
            json=_run([_user([{"type": "text", "text": QUERY}, audio])]),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": (
                "messages[0] part 1 has unsupported type 'audio'; only text and "
                "image parts are accepted"
            ),
            "type": "invalid_request_error",
            "code": "invalid_request",
        }


class TestRunFailures:
    """A failed turn ends on RUN_ERROR and never carries the exception text."""

    async def test_a_failing_token_stream_ends_on_run_error(self, client):
        response = await client.post(
            "/ag-ui/failing_stream_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
        )

        events = _events(response.text)
        assert _types(events) == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "STEP_FINISHED",
            "RUN_ERROR",
        ]
        assert events[1:4] == _opening_step("failing_stream_agent")
        assert events[4] == {
            "type": "RUN_ERROR",
            "message": (
                "FailingAgent streaming failed with RuntimeError. See server "
                "logs for detail."
            ),
            "code": "internal_error",
        }
        assert "secret-backend-detail" not in response.text

    async def test_a_failing_dispatch_ends_on_run_error(self, client):
        response = await client.post(
            "/ag-ui/failing_dispatch_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
        )

        events = _events(response.text)
        assert _types(events) == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "STEP_FINISHED",
            "RUN_ERROR",
        ]
        assert events[1:4] == _opening_step("failing_dispatch_agent")
        assert events[4] == {
            "type": "RUN_ERROR",
            "message": (
                "failing_dispatch_agent failed with RuntimeError. See server logs "
                "for detail."
            ),
            "code": "internal_error",
        }
        assert "secret-backend-detail" not in response.text

    async def test_a_dispatch_failing_after_a_phase_ends_on_run_error(self, client):
        response = await client.post(
            "/ag-ui/failing_after_status_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
        )

        assert _events(response.text) == [
            {"type": "RUN_STARTED", "threadId": THREAD_ID, "runId": RUN_ID},
            *_opening_step("failing_after_status_agent"),
            {"type": "STEP_STARTED", "stepName": STATUS_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": STATUS_PHASE, "message": STATUS_MESSAGE},
            },
            {"type": "STEP_FINISHED", "stepName": STATUS_PHASE},
            {
                "type": "RUN_ERROR",
                "message": (
                    "failing_after_status_agent failed with RuntimeError. See "
                    "server logs for detail."
                ),
                "code": "internal_error",
            },
        ]
        assert "secret-backend-detail" not in response.text
        assert openai_compat.in_flight_count() == 0

    async def test_a_dead_continuation_store_fails_the_suspension(self, client, caplog):
        dead = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}",
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        openai_compat.set_continuation_store(
            ContinuationStore(dead, key_prefix="test:ag-ui:dead")
        )
        try:
            with caplog.at_level(logging.ERROR):
                response = await client.post(
                    "/ag-ui/tool_agent",
                    json=_run([_user(QUERY)], tools=[WRITE_FILE_TOOL]),
                    headers=_auth(KEY_A),
                )
        finally:
            await dead.aclose()

        events = _events(response.text)
        assert _types(events) == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "STEP_FINISHED",
            "RUN_ERROR",
        ]
        assert events[1:4] == _opening_step("tool_agent")
        assert events[4]["code"] == "service_unavailable"
        assert events[4]["message"] == (
            "tool_agent failed with SessionStateUnavailable. See server logs for "
            "detail."
        )

    async def test_a_dead_conversation_ledger_fails_the_answered_run_after_its_text(
        self, client, dispatcher, conversation_ledger
    ):
        dead = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}",
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                dead, save_lease_s=5, failure_capacity=10, key_prefix="test:dead"
            )
        )
        try:
            response = await client.post(
                "/ag-ui/search_stream_agent",
                json=_run([_user(QUERY)]),
                headers=_auth(KEY_A),
            )
        finally:
            dispatcher.set_conversation_ledger(conversation_ledger)
            await dead.aclose()

        events = _events(response.text)
        assert _text(events) == f"[{TENANT_A}] {SUMMARY_BODY}"
        assert _types(events)[-3:] == ["TEXT_MESSAGE_END", "STEP_FINISHED", "RUN_ERROR"]
        assert events[-1] == {
            "type": "RUN_ERROR",
            "message": "The reply was not saved to this conversation "
            "(SessionStateUnavailable). See server logs for detail.",
            "code": "conversation_not_saved",
        }

    async def test_a_failed_run_keeps_its_own_error_when_the_ledger_is_dead(
        self, client, dispatcher, conversation_ledger
    ):
        dead = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}",
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                dead, save_lease_s=5, failure_capacity=10, key_prefix="test:dead"
            )
        )
        try:
            response = await client.post(
                "/ag-ui/failing_stream_agent",
                json=_run([_user(QUERY)]),
                headers=_auth(KEY_A),
            )
        finally:
            dispatcher.set_conversation_ledger(conversation_ledger)
            await dead.aclose()

        events = _events(response.text)
        assert events[1:] == [
            {
                "type": "RUN_ERROR",
                "message": (
                    "FailingAgent streaming failed with RuntimeError. See server "
                    "logs for detail."
                ),
                "code": "internal_error",
            }
        ]

    async def test_an_unwired_dispatcher_is_503(self, client):
        openai_compat.set_dispatcher_provider(None)

        response = await client.post(
            "/ag-ui/search_stream_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 503
        assert response.json()["error"] == {
            "message": "Runtime initialising; dispatcher not wired.",
            "type": "server_error",
            "code": "service_unavailable",
        }

    async def test_an_unreachable_registry_is_503_naming_it(
        self, client, config_manager
    ):
        dead = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}",
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        registry = _registry(config_manager)
        registry.set_store(RedisAgentRegistryStore(dead, key_prefix="test:ag-ui"))
        outage_dispatcher = AgentDispatcher(
            agent_registry=registry,
            config_manager=config_manager,
            schema_loader=None,
        )
        openai_compat.set_dispatcher_provider(lambda: outage_dispatcher)
        try:
            response = await client.post(
                "/ag-ui/search_stream_agent",
                json=_run([_user(QUERY)]),
                headers=_auth(KEY_A),
            )
        finally:
            await dead.aclose()

        assert response.status_code == 503
        assert response.json()["error"] == {
            "message": (
                "The agent registry is unavailable "
                "(AgentRegistryUnavailableError). See server logs for detail."
            ),
            "type": "server_error",
            "code": "service_unavailable",
            "error_type": "AgentRegistryUnavailableError",
        }


class TestConcurrentRuns:
    """Concurrent runs never carry each other's tenant or text."""

    async def test_two_tenants_stream_concurrently_on_their_own_tenant(self, client):
        runs = await asyncio.gather(
            *(
                client.post(
                    "/ag-ui/search_stream_agent",
                    json=_run([_user(QUERY)], runId=f"run-{index}"),
                    headers=_auth(key),
                )
                for index, key in enumerate([KEY_A, KEY_B] * 4)
            )
        )

        for index, response in enumerate(runs):
            tenant = TENANT_A if index % 2 == 0 else TENANT_B
            other = TENANT_B if index % 2 == 0 else TENANT_A
            events = _events(response.text)
            assert _text(events) == f"[{tenant}] {SUMMARY_BODY}"
            assert other not in response.text
            assert events[0]["runId"] == f"run-{index}"
            assert events[-1]["runId"] == f"run-{index}"
        message_ids = {_events(response.text)[9]["messageId"] for response in runs}
        assert len(message_ids) == len(runs)
        assert openai_compat.in_flight_count() == 0


class TestConcurrentDispatchRuns:
    """Concurrent dispatch-path runs each stream only their own phases."""

    async def test_interleaved_runs_never_carry_each_others_phases(self, client):
        """Every run reports its first phase and waits at one barrier until
        all of them have, so the runs are in flight together when each
        reports its second."""
        interleave["barrier"] = asyncio.Barrier(INTERLEAVED_RUNS)
        keys = [KEY_A, KEY_B] * (INTERLEAVED_RUNS // 2)
        runs = await asyncio.gather(
            *(
                client.post(
                    "/ag-ui/interleaved_agent",
                    json=_run([_user(QUERY)], runId=f"run-{index}"),
                    headers=_auth(key),
                )
                for index, key in enumerate(keys)
            )
        )

        for index, response in enumerate(runs):
            tenant = TENANT_A if index % 2 == 0 else TENANT_B
            events = _events(response.text)
            assert [
                event["value"] for event in events if event["type"] == "CUSTOM"
            ] == [
                {"phase": START_PHASE, "message": "Running interleaved_agent"},
                {"phase": "first", "message": f"{tenant} first"},
                {"phase": "second", "message": f"{tenant} second"},
            ]
            assert _text(events) == f"{tenant} done"
            assert (events[0]["runId"], events[-1]["runId"]) == (
                f"run-{index}",
                f"run-{index}",
            )
        assert openai_compat.in_flight_count() == 0


class TestDisconnect:
    """A client that hangs up cancels the turn behind its run."""

    def test_disconnect_after_a_phase_cancels_the_dispatched_turn(self, live_server):
        seen: List[Dict[str, Any]] = []
        hangup = 0.0

        with httpx.stream(
            "POST",
            f"{live_server}/ag-ui/slow_dispatch_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
            timeout=30.0,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    seen.append(json.loads(line[len("data: ") :]))
                    if len(seen) == 6:
                        assert openai_compat.in_flight_count() == 1
                        hangup = time.perf_counter()
                        break

        assert seen == [
            {"type": "RUN_STARTED", "threadId": THREAD_ID, "runId": RUN_ID},
            *_opening_step("slow_dispatch_agent"),
            {"type": "STEP_STARTED", "stepName": STATUS_PHASE},
            {
                "type": "CUSTOM",
                "name": ag_ui.STATUS_EVENT,
                "value": {"phase": STATUS_PHASE, "message": STATUS_MESSAGE},
            },
        ]
        deadline = hangup + DISCONNECT_CANCEL_BUDGET_SECONDS
        while time.perf_counter() < deadline and "cancelled" not in slow_agent_events:
            time.sleep(0.02)
        assert slow_agent_events == ["started", "cancelled"]
        deadline = time.perf_counter() + DISCONNECT_CANCEL_BUDGET_SECONDS
        while time.perf_counter() < deadline and openai_compat.in_flight_count() != 0:
            time.sleep(0.02)
        assert openai_compat.in_flight_count() == 0

    def test_disconnect_mid_run_cancels_the_turn(self, live_server):
        seen: List[Dict[str, Any]] = []
        hangup = 0.0

        with httpx.stream(
            "POST",
            f"{live_server}/ag-ui/slow_agent",
            json=_run([_user(QUERY)]),
            headers=_auth(KEY_A),
            timeout=30.0,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    seen.append(json.loads(line[len("data: ") :]))
                    if len(seen) == 5:
                        assert openai_compat.in_flight_count() == 1
                        hangup = time.perf_counter()
                        break

        assert _types(seen) == [
            "RUN_STARTED",
            "STEP_STARTED",
            "CUSTOM",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
        ]
        assert seen[1:3] == _opening_step("slow_agent")[:2]
        assert seen[4]["delta"] == "first "
        deadline = hangup + DISCONNECT_CANCEL_BUDGET_SECONDS
        while time.perf_counter() < deadline and "cancelled" not in slow_agent_events:
            time.sleep(0.02)
        assert slow_agent_events == ["started", "cancelled"]
        deadline = time.perf_counter() + DISCONNECT_CANCEL_BUDGET_SECONDS
        while time.perf_counter() < deadline and openai_compat.in_flight_count() != 0:
            time.sleep(0.02)
        assert openai_compat.in_flight_count() == 0
