"""Live-token streaming on the /v1 surface, over the real dispatcher.

Real router -> real ``AgentDispatcher.dispatch_stream`` -> real
``AgentBase`` streaming machinery -> deterministic agents that emit their
token events through ``emit_progress`` exactly as ``call_dspy`` does. No LM
and no boundary mock: every token below is produced by the same code path a
served agent uses, so the field filter, the reconciliation with the final
payload, cancellation and the per-invocation queue scoping are all exercised
for real.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any, Dict, List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.harness_turn import extract_answer_text
from cogniverse_runtime.routers import openai_compat
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

TENANT_A = "acme:acme"
TENANT_B = "beta:prod"
KEY_A = "stream-key-tenant-a"
KEY_B = "stream-key-tenant-b"
QUERY = "When did the tower open and how many people visit it?"

SUMMARY_BODY = (
    "The tower opened in 1889 and remains the most visited paid monument "
    "in the world, drawing about seven million people a year."
)
SUB_QUESTIONS = ["When did it open?", "How many visitors a year?"]
GAPS = ["No source for the visitor count after 2019."]
FINDINGS = ["Opened 1889", "Roughly 7M visitors a year"]
REPORT_ALTERNATE = "A longer rendering of the same finding, for the report field."

# Both budgets are measured, not guessed: uvicorn cancels the response task
# ~40ms after should_exit, and the terminal frames are written from the
# except block that cancellation lands in.
SHUTDOWN_FRAME_BUDGET_SECONDS = 1.0
DISCONNECT_CANCEL_BUDGET_SECONDS = 3.0
DIVERGENT_PREFIX = "streamed-prefix"
DIVERGENT_ANSWER = "a completely different answer"
TOOL_CALL_ID = "call_stream_write_file"
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _token_chunks(text: str, size: int = 11) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class HarnessStreamDeps(AgentDeps):
    pass


class HarnessStreamInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    external_tools: list = []
    tool_results: list = []


class ResearchStreamOutput(AgentOutput):
    summary: str = ""
    sub_questions: list = []
    gaps: list = []


class ResearchStreamAgent(
    AgentBase[HarnessStreamInput, ResearchStreamOutput, HarnessStreamDeps]
):
    """Three streamed fields, one of which is the answer.

    Mirrors the deep-research shape: the decomposition and the gap list are
    emitted on the same channel as the summary and must never reach a client
    as reply text.
    """

    async def _process_impl(self, input: HarnessStreamInput) -> ResearchStreamOutput:
        for field, values in (("sub_questions", SUB_QUESTIONS), ("gaps", GAPS)):
            accumulated = ""
            for chunk in _token_chunks(" ".join(values)):
                accumulated += chunk
                self.emit_progress(
                    "token",
                    chunk,
                    data={"accumulated": accumulated, "output_field": field},
                )
                await asyncio.sleep(0)
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
        return ResearchStreamOutput(
            summary=summary, sub_questions=SUB_QUESTIONS, gaps=GAPS
        )


class ReportStreamOutput(AgentOutput):
    executive_summary: str = ""
    detailed_findings: list = []


class ReportStreamAgent(
    AgentBase[HarnessStreamInput, ReportStreamOutput, HarnessStreamDeps]
):
    """Streams its answer field while its answer carries a supporting field."""

    async def _process_impl(self, input: HarnessStreamInput) -> ReportStreamOutput:
        accumulated = ""
        for chunk in _token_chunks(SUMMARY_BODY):
            accumulated += chunk
            self.emit_progress(
                "token",
                chunk,
                data={
                    "accumulated": accumulated,
                    "output_field": "executive_summary",
                },
            )
            await asyncio.sleep(0)
        return ReportStreamOutput(
            executive_summary=SUMMARY_BODY, detailed_findings=list(FINDINGS)
        )


class DivergentStreamOutput(AgentOutput):
    summary: str = ""


class DivergentStreamAgent(
    AgentBase[HarnessStreamInput, DivergentStreamOutput, HarnessStreamDeps]
):
    """Streams text its finished answer does not begin with."""

    async def _process_impl(self, input: HarnessStreamInput) -> DivergentStreamOutput:
        self.emit_progress(
            "token",
            DIVERGENT_PREFIX,
            data={"accumulated": DIVERGENT_PREFIX, "output_field": "summary"},
        )
        return DivergentStreamOutput(summary=DIVERGENT_ANSWER)


class TwoFieldStreamOutput(AgentOutput):
    summary: str = ""
    report: str = ""


class TwoFieldStreamAgent(
    AgentBase[HarnessStreamInput, TwoFieldStreamOutput, HarnessStreamDeps]
):
    """Streams two fields that both head an answer rule.

    Only the one the extractor would pick for the finished payload is the
    reply; the other is a second rendering the client must not receive.
    """

    async def _process_impl(self, input: HarnessStreamInput) -> TwoFieldStreamOutput:
        for field, text in (("summary", SUMMARY_BODY), ("report", REPORT_ALTERNATE)):
            accumulated = ""
            for chunk in _token_chunks(text):
                accumulated += chunk
                self.emit_progress(
                    "token",
                    chunk,
                    data={"accumulated": accumulated, "output_field": field},
                )
                await asyncio.sleep(0)
        return TwoFieldStreamOutput(summary=SUMMARY_BODY, report=REPORT_ALTERNATE)


class ToolStreamOutput(AgentOutput):
    summary: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class ToolStreamAgent(
    AgentBase[HarnessStreamInput, ToolStreamOutput, HarnessStreamDeps]
):
    """Suspends on an external tool call from the token path."""

    async def _process_impl(self, input: HarnessStreamInput) -> ToolStreamOutput:
        return ToolStreamOutput(
            pending_tool_calls=[
                {"id": TOOL_CALL_ID, "name": "write_file", "arguments": {"x": 1}}
            ],
            continuation_state={"stage": "awaited"},
        )


slow_stream_events: List[str] = []


class SlowStreamOutput(AgentOutput):
    summary: str = ""


class SlowStreamAgent(
    AgentBase[HarnessStreamInput, SlowStreamOutput, HarnessStreamDeps]
):
    """Emits two tokens, then holds the turn open for 20 s."""

    async def _process_impl(self, input: HarnessStreamInput) -> SlowStreamOutput:
        slow_stream_events.append("started")
        for chunk in ("first ", "second "):
            self.emit_progress(
                "token",
                chunk,
                data={"accumulated": chunk, "output_field": "summary"},
            )
        try:
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            slow_stream_events.append("cancelled")
            raise
        slow_stream_events.append("completed")
        return SlowStreamOutput(summary="first second done")


_AGENT_CLASSES = {
    "research_stream_agent": f"{__name__}:ResearchStreamAgent",
    "chunked_research_agent": f"{__name__}:ResearchStreamAgent",
    "report_stream_agent": f"{__name__}:ReportStreamAgent",
    "divergent_stream_agent": f"{__name__}:DivergentStreamAgent",
    "two_field_stream_agent": f"{__name__}:TwoFieldStreamAgent",
    "tool_stream_agent": f"{__name__}:ToolStreamAgent",
    "slow_stream_agent": f"{__name__}:SlowStreamAgent",
}

# Every endpoint declares answer-token streaming except chunked_research_agent,
# which runs the same agent class without the declaration.
_TOKEN_STREAMING = {name: name != "chunked_research_agent" for name in _AGENT_CLASSES}

MODEL_MAP = {
    "cogniverse/research": "research_stream_agent",
    "cogniverse/chunked": "chunked_research_agent",
    "cogniverse/report": "report_stream_agent",
    "cogniverse/divergent": "divergent_stream_agent",
    "cogniverse/two-fields": "two_field_stream_agent",
    "cogniverse/tools": "tool_stream_agent",
    "cogniverse/slow": "slow_stream_agent",
}


@pytest.fixture(scope="module")
def dispatcher():
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    for agent_name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=agent_name,
                url="http://localhost:8000",
                capabilities=["harness_stream"],
                streams_answer_tokens=_TOKEN_STREAMING[agent_name],
            )
        )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    yield AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    for agent_name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(agent_name, None)


@pytest.fixture()
def compat_app(dispatcher):
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A, KEY_B: TENANT_B})
    openai_compat.set_model_map(MODEL_MAP)
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()
    slow_stream_events.clear()
    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    yield app
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})
    openai_compat.clear_continuations()


@pytest.fixture()
async def client(compat_app):
    transport = httpx.ASGITransport(app=compat_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=60.0
    ) as http_client:
        yield http_client


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _body(model: str, stream: bool = True, **overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": QUERY}],
        "stream": stream,
    }
    body.update(overrides)
    return body


def _data_lines(raw: str) -> List[str]:
    return [
        line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")
    ]


def _frames(raw: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in _data_lines(raw) if line != "[DONE]"]


def _content_of(raw: str) -> str:
    return "".join(
        frame["choices"][0]["delta"].get("content", "")
        for frame in _frames(raw)
        if frame.get("choices")
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestAnswerFieldFilter:
    """R9 — only the field that carries the answer reaches the client."""

    async def test_research_stream_carries_the_summary_and_nothing_else(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/research"),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        expected = f"[{TENANT_A}] {SUMMARY_BODY}"
        assert _content_of(response.text) == expected
        for leaked in SUB_QUESTIONS + GAPS:
            assert leaked not in response.text
        frames = _frames(response.text)
        assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert frames[-1]["choices"][0]["finish_reason"] == "stop"
        assert _data_lines(response.text)[-1] == "[DONE]"

    async def test_the_stream_arrives_as_many_token_deltas(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/research"),
            headers=_auth(KEY_A),
        )

        contents = [
            frame["choices"][0]["delta"]["content"]
            for frame in _frames(response.text)
            if frame["choices"][0]["delta"].get("content")
        ]

        expected = f"[{TENANT_A}] {SUMMARY_BODY}"
        assert contents == _token_chunks(expected)

    @pytest.mark.parametrize("model", ["cogniverse/research", "cogniverse/report"])
    async def test_streamed_deltas_equal_the_non_streamed_content(self, client, model):
        plain = await client.post(
            "/v1/chat/completions",
            json=_body(model, stream=False),
            headers=_auth(KEY_A),
        )
        assert plain.status_code == 200
        non_streamed = plain.json()["choices"][0]["message"]["content"]

        streamed = await client.post(
            "/v1/chat/completions", json=_body(model), headers=_auth(KEY_A)
        )

        assert streamed.status_code == 200
        assert _content_of(streamed.text) == non_streamed

    async def test_report_answer_tops_up_its_supporting_field(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/report"),
            headers=_auth(KEY_A),
        )

        assert _content_of(response.text) == (
            f"{SUMMARY_BODY}\n\n- Opened 1889\n- Roughly 7M visitors a year"
        )
        contents = [
            frame["choices"][0]["delta"]["content"]
            for frame in _frames(response.text)
            if frame["choices"][0]["delta"].get("content")
        ]
        assert contents[:-1] == _token_chunks(SUMMARY_BODY)
        assert contents[-1] == "\n\n- Opened 1889\n- Roughly 7M visitors a year"

    async def test_only_the_extractor_s_field_streams_when_two_qualify(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/two-fields"),
            headers=_auth(KEY_A),
        )

        assert _content_of(response.text) == SUMMARY_BODY
        assert REPORT_ALTERNATE not in response.text

    async def test_undeclared_agent_keeps_the_chunked_final_answer(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/chunked"),
            headers=_auth(KEY_A),
        )

        expected = f"[{TENANT_A}] {SUMMARY_BODY}"
        assert _content_of(response.text) == expected
        contents = [
            frame["choices"][0]["delta"]["content"]
            for frame in _frames(response.text)
            if frame["choices"][0]["delta"].get("content")
        ]
        assert contents == openai_compat.split_answer_chunks(expected)
        assert len(contents) == 1

    async def test_a_stream_that_diverges_from_its_answer_ends_in_an_error(
        self, client
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/divergent"),
            headers=_auth(KEY_A),
        )

        frames = _frames(response.text)
        assert _content_of(response.text) == DIVERGENT_PREFIX
        assert frames[-1]["error"] == {
            "message": (
                f"Agent 'divergent_stream_agent' streamed {len(DIVERGENT_PREFIX)} "
                "characters of 'summary' that its final answer does not begin "
                "with; the streamed reply cannot be completed"
            ),
            "type": "server_error",
            "code": "internal_error",
        }
        assert _data_lines(response.text)[-1] == "[DONE]"


class TestTokenStreamSelection:
    """R32 — a resume turn leaves the token path, an offer of tools does not."""

    def test_selection_keys_on_replayed_results(self, dispatcher):
        assert openai_compat.use_token_stream(
            dispatcher, "research_stream_agent", False
        )
        assert not openai_compat.use_token_stream(
            dispatcher, "research_stream_agent", True
        )
        assert not openai_compat.use_token_stream(
            dispatcher, "chunked_research_agent", False
        )

    async def test_declared_tools_alone_keep_the_token_path(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/research", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
        )

        contents = [
            frame["choices"][0]["delta"]["content"]
            for frame in _frames(response.text)
            if frame["choices"][0]["delta"].get("content")
        ]
        assert contents == _token_chunks(f"[{TENANT_A}] {SUMMARY_BODY}")

    async def test_a_resume_turn_falls_back_to_the_chunked_path(self, client):
        messages = [
            {"role": "user", "content": QUERY},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": TOOL_CALL_ID, "function": {"name": "write_file"}}
                ],
            },
            {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": "wrote it"},
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/research", tools=TOOL_DEFS, messages=messages),
            headers=_auth(KEY_A),
        )

        expected = f"[{TENANT_A}] {SUMMARY_BODY}"
        contents = [
            frame["choices"][0]["delta"]["content"]
            for frame in _frames(response.text)
            if frame["choices"][0]["delta"].get("content")
        ]
        assert contents == openai_compat.split_answer_chunks(expected)

    async def test_the_token_path_suspends_on_pending_tool_calls(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
        )

        frames = _frames(response.text)
        assert frames[1]["choices"][0]["delta"]["tool_calls"] == [
            {
                "index": 0,
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"x": 1}),
                },
            }
        ]
        assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert _data_lines(response.text)[-1] == "[DONE]"
        assert openai_compat.continuation_count() == 1

    async def test_a_suspended_turn_reports_usage_only_when_asked(self, client):
        asked = await client.post(
            "/v1/chat/completions",
            json=_body(
                "cogniverse/tools",
                tools=TOOL_DEFS,
                stream_options={"include_usage": True},
            ),
            headers=_auth(KEY_A),
        )
        plain = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
        )

        frames = _frames(asked.text)
        assert frames[-2]["choices"][0]["finish_reason"] == "tool_calls"
        assert frames[-1]["choices"] == []
        assert set(frames[-1]["usage"]) == {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
        assert frames[-1]["usage"]["total_tokens"] == (
            frames[-1]["usage"]["prompt_tokens"]
            + frames[-1]["usage"]["completion_tokens"]
        )
        assert [frame for frame in frames[:-1] if "usage" in frame] == []
        assert [frame for frame in _frames(plain.text) if "usage" in frame] == []
        assert _frames(plain.text)[-1]["choices"][0]["finish_reason"] == "tool_calls"


class TestToolChoice:
    """A client that forbids tools is never answered with a tool call."""

    async def test_tool_choice_none_turns_a_tool_request_into_502(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(
                "cogniverse/tools", stream=False, tools=TOOL_DEFS, tool_choice="none"
            ),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 502
        assert response.json()["error"] == {
            "message": (
                "Agent 'tool_stream_agent' requested 1 tool call(s) on a turn "
                'sent with tool_choice "none"'
            ),
            "type": "server_error",
            "code": "tool_choice_violation",
        }

    async def test_tool_choice_none_withholds_the_definitions(self, client):
        forbidden, plain = await asyncio.gather(
            client.post(
                "/v1/chat/completions",
                json=_body(
                    "cogniverse/research",
                    stream=False,
                    tools=TOOL_DEFS,
                    tool_choice="none",
                ),
                headers=_auth(KEY_A),
            ),
            client.post(
                "/v1/chat/completions",
                json=_body("cogniverse/research", stream=False, tools=TOOL_DEFS),
                headers=_auth(KEY_A),
            ),
        )

        assert forbidden.status_code == 200
        assert plain.status_code == 200
        assert forbidden.json()["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.parametrize(
        "choice",
        ["required", {"type": "function", "function": {"name": "write_file"}}],
    )
    async def test_a_forcing_tool_choice_is_400(self, client, choice):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/research", stream=False, tool_choice=choice),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": (
                f"tool_choice {choice!r} is not supported; this surface accepts "
                '"auto" and "none"'
            ),
            "type": "invalid_request_error",
            "code": "invalid_request",
        }

    async def test_tool_choice_auto_is_the_default_behaviour(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(
                "cogniverse/tools", stream=False, tools=TOOL_DEFS, tool_choice="auto"
            ),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


class TestTwoTenantStreamIsolation:
    """Concurrent streams never carry each other's text."""

    async def test_concurrent_streams_stay_on_their_own_tenant(self, client):
        first, second = await asyncio.gather(
            client.post(
                "/v1/chat/completions",
                json=_body("cogniverse/research"),
                headers=_auth(KEY_A),
            ),
            client.post(
                "/v1/chat/completions",
                json=_body("cogniverse/research"),
                headers=_auth(KEY_B),
            ),
        )

        assert _content_of(first.text) == f"[{TENANT_A}] {SUMMARY_BODY}"
        assert _content_of(second.text) == f"[{TENANT_B}] {SUMMARY_BODY}"
        assert TENANT_B not in first.text
        assert TENANT_A not in second.text

    async def test_one_agent_instance_serves_two_concurrent_streams(self):
        """The per-invocation queue is what keeps one instance safe.

        The router builds an agent per streamed turn, so this drives the
        shared-instance case directly against the base class the router
        streams through.
        """
        agent = ResearchStreamAgent(deps=HarnessStreamDeps())

        async def collect(tenant: str) -> str:
            text = ""
            async for event in await agent.process(
                HarnessStreamInput(query=QUERY, tenant_id=tenant), stream=True
            ):
                if (
                    event.get("phase") == "token"
                    and (event.get("data") or {}).get("output_field") == "summary"
                ):
                    text += event["message"]
            return text

        first, second = await asyncio.gather(collect(TENANT_A), collect(TENANT_B))

        assert first == f"[{TENANT_A}] {SUMMARY_BODY}"
        assert second == f"[{TENANT_B}] {SUMMARY_BODY}"


class TestStreamTermination:
    """R27 and the streamed half of R7, on a real uvicorn socket."""

    @pytest.fixture()
    def live_server(self, compat_app):
        port = _free_port()
        config = uvicorn.Config(
            compat_app, host="127.0.0.1", port=port, log_level="warning"
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn did not start"
        yield server, f"http://127.0.0.1:{port}"
        server.should_exit = True
        thread.join(timeout=20)
        assert not thread.is_alive()

    def test_shutdown_mid_stream_ends_with_an_error_frame_and_done(self, live_server):
        server, base_url = live_server
        seen: List[str] = []
        marks: List[float] = []
        read_error = ""

        def read():
            nonlocal read_error
            try:
                with httpx.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json=_body("cogniverse/slow"),
                    headers=_auth(KEY_A),
                    timeout=30.0,
                ) as response:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            seen.append(line[len("data: ") :])
                            if len(seen) == 3:
                                marks.append(time.perf_counter())
                                threading.Thread(
                                    target=lambda: setattr(server, "should_exit", True),
                                    daemon=True,
                                ).start()
                            elif line == "data: [DONE]":
                                marks.append(time.perf_counter())
            except Exception as exc:
                read_error = f"{type(exc).__name__}"

        reader = threading.Thread(target=read)
        reader.start()
        reader.join(timeout=30)
        assert not reader.is_alive()

        assert [json.loads(line)["choices"][0]["delta"] for line in seen[:3]] == [
            {"role": "assistant"},
            {"content": "first "},
            {"content": "second "},
        ]
        assert json.loads(seen[-2])["error"] == {
            "message": "Stream cancelled before the turn completed.",
            "type": "server_error",
            "code": "stream_cancelled",
        }
        assert seen[-1] == "[DONE]"
        assert marks[1] - marks[0] < SHUTDOWN_FRAME_BUDGET_SECONDS, (
            f"terminal frame {marks[1] - marks[0]:.4f}s after shutdown began"
        )
        assert read_error == "RemoteProtocolError"
        assert slow_stream_events == ["started", "cancelled"]
        assert openai_compat.in_flight_count() == 0

    def test_disconnect_mid_stream_cancels_the_turn(self, live_server):
        _, base_url = live_server
        seen: List[str] = []
        hangup = 0.0

        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=_body("cogniverse/slow"),
            headers=_auth(KEY_A),
            timeout=30.0,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    seen.append(line[len("data: ") :])
                    if len(seen) == 3:
                        assert openai_compat.in_flight_count() == 1
                        hangup = time.perf_counter()
                        break

        deadline = hangup + DISCONNECT_CANCEL_BUDGET_SECONDS
        while time.perf_counter() < deadline and "cancelled" not in slow_stream_events:
            time.sleep(0.02)
        observed = time.perf_counter() - hangup

        assert slow_stream_events == ["started", "cancelled"], (
            f"events {slow_stream_events} after {observed:.3f}s"
        )
        assert observed < DISCONNECT_CANCEL_BUDGET_SECONDS
        deadline = time.perf_counter() + DISCONNECT_CANCEL_BUDGET_SECONDS
        while time.perf_counter() < deadline and openai_compat.in_flight_count() != 0:
            time.sleep(0.02)
        assert openai_compat.in_flight_count() == 0


class TestStreamedAnswerMatchesTheExtractor:
    """The streamed body is the same text the extractor produces."""

    async def test_report_content_equals_extract_answer_text(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body("cogniverse/report"),
            headers=_auth(KEY_A),
        )

        assert _content_of(response.text) == extract_answer_text(
            {
                "executive_summary": SUMMARY_BODY,
                "detailed_findings": list(FINDINGS),
            }
        )
