"""A generation the adapter cannot turn into outputs ends the turn as an error.

Real FastAPI route -> real AgentDispatcher -> real SummarizerAgent -> real
DSPy/LiteLLM -> a real OpenAI-compatible HTTP provider. Nothing on that path
is mocked: the provider is an out-of-process-shaped HTTP server the test
drives, so a response that stops before the signature's required outputs is
the genuine wire condition a truncated generation produces. Every transport
the runtime serves that turn on — the agent route, non-streaming and
streaming ``/v1``, and A2A — is driven over its own socket or ASGI app.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import io
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import dspy
import httpx
import pytest
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from PIL import Image

from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.dspy import LenientJSONAdapter
from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.harness_turn import derive_request_seed
from cogniverse_runtime.routers import agents, openai_compat
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

MODEL = "openai/generation-contract"
SUMMARY_MODEL = "cogniverse-summary"
TEXT_MODEL = "cogniverse-text-analysis"
API_KEY = "generation-contract-key"
TENANTS = ("prodfixfoundation:generationa", "prodfixfoundation:generationb")
COMPLETE = {
    "reasoning": "Both telescopes are named and the start time is given.",
    "summary": "The observatory runs two telescopes and starts observing at sunset.",
    "key_points": "two telescopes, observations start at sunset",
    "confidence_score": "0.9",
}
# What each case leaves out of the response the provider sends back, and the
# fields the adapter must then report as never generated.
MISSING_BY_CASE = {
    "missing": ("summary",),
    "unknown": ("summary",),
    "truncated": ("confidence_score", "key_points", "summary"),
    "null": ("summary",),
    "empty": ("summary",),
}
# A completion with no JSON object at all names nothing the adapter can map.
UNPARSABLE = "prose"
PROSE = "Sure! The observatory runs two telescopes and observations begin at sunset."


class Provider:
    """OpenAI-compatible completion server driven by a marker in the prompt."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cases: list[str] = []
        self.max_tokens: list[int | None] = []
        self.barrier: threading.Barrier | None = None
        self.failures: set[str] = set()
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = json.dumps(body["messages"])
                case = next(
                    name
                    for name in (
                        "missing",
                        "unknown",
                        "truncated",
                        "null",
                        "empty",
                        "prose",
                        "blank",
                        "budgeted",
                        "valid",
                    )
                    if f"case:{name}" in prompt
                )
                with provider.lock:
                    provider.cases.append(case)
                    provider.max_tokens.append(body.get("max_tokens"))
                    barrier = provider.barrier
                    failing = case in provider.failures
                if barrier is not None:
                    barrier.wait(timeout=30)
                if failing:
                    self.respond(
                        503,
                        {
                            "error": {
                                "message": "generation interrupted",
                                "type": "server_error",
                            }
                        },
                    )
                    return
                fields = dict(COMPLETE)
                if case == "missing":
                    del fields["summary"]
                    content = json.dumps(fields)
                elif case == "unknown":
                    fields["unrecognized_field"] = fields.pop("summary")
                    content = json.dumps(fields)
                elif case == "truncated":
                    # A max-token cut: the object stops inside the reasoning
                    # string that ChainOfThought emits first.
                    content = '{"reasoning": "' + fields["reasoning"][:24]
                elif case == "null":
                    fields["summary"] = None
                    content = json.dumps(fields)
                elif case == "empty":
                    fields["summary"] = ""
                    content = json.dumps(fields)
                elif case == "prose":
                    content = PROSE
                elif case == "blank":
                    content = "{}"
                elif case == "budgeted":
                    # A real completion stops at the request's max_tokens; one
                    # character stands for one token here.
                    content = json.dumps(fields)[: body["max_tokens"]]
                else:
                    content = json.dumps(fields)
                self.respond(
                    200,
                    {
                        "id": f"chatcmpl-{case}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": (
                                    "length"
                                    if case == "truncated"
                                    or (
                                        case == "budgeted"
                                        and content != json.dumps(fields)
                                    )
                                    else "stop"
                                ),
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    },
                )

            def respond(self, status, payload):
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


@pytest.fixture
def provider():
    server = Provider()
    server.thread.start()
    try:
        yield server
    finally:
        server.server.shutdown()
        server.server.server_close()
        server.thread.join(5)


@pytest.fixture
def telemetry_off():
    """This module carries no telemetry backend; the agents the dispatcher
    builds still read the process manager."""
    from cogniverse_foundation.telemetry import manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import (
        TelemetryConfig,
        TelemetryManager,
    )

    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if (
        installed is not None
        and telemetry_manager_module._telemetry_manager is installed
    ):
        telemetry_manager_module._telemetry_manager = None


@pytest.fixture
def build_dispatcher(provider, tmp_path, monkeypatch, telemetry_off):
    config = json.loads(Path("configs/config.json").read_text())
    config.pop("active_video_profile", None)
    config["llm_config"] = {
        "primary": {
            "model": MODEL,
            "api_base": provider.url,
            "api_key": "not-required",
            "max_tokens": 256,
            "temperature": 0,
            "num_retries": 0,
            "request_timeout": 30,
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_file))

    def build(*, streams_answer_tokens: bool = False) -> AgentDispatcher:
        manager = ConfigManager(store=InMemoryConfigStore())
        registry = AgentRegistry(tenant_id=TENANTS[0], config_manager=manager)
        registry.register_agent(
            AgentEndpoint(
                name="summarizer_agent",
                url="http://127.0.0.1:8000",
                capabilities=["summarization"],
                health_endpoint="/health",
                streams_answer_tokens=streams_answer_tokens,
            )
        )
        registry.register_agent(
            AgentEndpoint(
                name="text_analysis_agent",
                url="http://127.0.0.1:8000",
                capabilities=["text_analysis"],
                health_endpoint="/health",
            )
        )
        return AgentDispatcher(
            agent_registry=registry,
            config_manager=manager,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        )

    return build


@pytest.fixture
async def summary_route(build_dispatcher, monkeypatch):
    monkeypatch.setattr(agents, "_dispatcher", build_dispatcher())
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    with dspy.context(adapter=LenientJSONAdapter()):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            yield client


@pytest.fixture
def compat_route(build_dispatcher):
    """Factory for the ``/v1`` app, wired the way the runtime lifespan wires it."""

    @asynccontextmanager
    async def make(*, streams_answer_tokens: bool):
        dispatcher = build_dispatcher(streams_answer_tokens=streams_answer_tokens)
        openai_compat.set_dispatcher_provider(lambda: dispatcher)
        openai_compat.set_api_keys({API_KEY: TENANTS[0]})
        openai_compat.set_model_map(
            {SUMMARY_MODEL: "summarizer_agent", TEXT_MODEL: "text_analysis_agent"}
        )
        openai_compat.set_key_resolver(None)
        openai_compat.clear_continuations()
        app = FastAPI()
        app.include_router(openai_compat.router, prefix="/v1")
        with dspy.context(adapter=LenientJSONAdapter()):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as client:
                yield client

    yield make
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()


@pytest.fixture
async def a2a_route(build_dispatcher):
    card = AgentCard(
        name="Generation contract",
        description="Generation failure transport contract",
        url="http://127.0.0.1:9999/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="summarizer_agent",
                name="summarizer_agent",
                description="Summarize content",
                tags=["summarization"],
            ),
            AgentSkill(
                id="text_analysis_agent",
                name="text_analysis_agent",
                description="Analyze text",
                tags=["text_analysis"],
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=CogniverseAgentExecutor(dispatcher=build_dispatcher()),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    with dspy.context(adapter=LenientJSONAdapter()):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            yield client


SCHEDULE_HIT = {
    "id": "grounded-source",
    "title": "Observatory schedule",
    "text_content": "The observatory has two telescopes. Observations start at sunset.",
    "score": 0.9,
}


def task(case: str, tenant: str = TENANTS[0]) -> dict:
    return {
        "agent_name": "summarizer_agent",
        "query": f"Summarize the observatory schedule. case:{case}",
        "context": {
            "tenant_id": tenant,
            "request_id": f"request-{case}",
            "search_results": [SCHEDULE_HIT],
        },
    }


def failed_turn(case: str) -> dict:
    produced = (
        "parsable output" if case == UNPARSABLE else ", ".join(MISSING_BY_CASE[case])
    )
    return {
        "status": "error",
        "agent": "summarizer_agent",
        "error": (
            f"Agent 'summarizer_agent' generation for request 'request-{case}' "
            f"produced no {produced}"
        ),
    }


@pytest.mark.parametrize(
    "case", ["missing", "unknown", "truncated", "null", "empty", "prose"]
)
async def test_incomplete_generation_is_a_failed_turn(summary_route, provider, case):
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=task(case)
    )
    assert response.status_code == 200, response.text
    assert response.json() == failed_turn(case)
    assert provider.cases == [case]


async def test_complete_generation_reaches_the_caller_byte_exact(
    summary_route, provider
):
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("valid")
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["agent"] == "summarizer_agent"
    assert body["result"]["summary"] == COMPLETE["summary"]
    assert body["answer"].startswith(COMPLETE["summary"])
    assert provider.cases == ["valid"]


async def test_concurrent_tenants_get_their_own_outcome(summary_route, provider):
    """Two tenants in flight at once: the incomplete one fails, the complete
    one answers, and neither borrows the other's fields."""
    provider.barrier = threading.Barrier(2)
    try:
        good, bad = await asyncio.gather(
            *[
                summary_route.post(
                    "/agents/summarizer_agent/process", json=task(case, tenant)
                )
                for case, tenant in zip(("valid", "missing"), TENANTS)
            ]
        )
    finally:
        provider.barrier = None
    assert sorted(provider.cases) == ["missing", "valid"]
    assert good.status_code == 200, good.text
    assert good.json()["status"] == "success"
    assert good.json()["result"]["summary"] == COMPLETE["summary"]
    assert bad.status_code == 200, bad.text
    assert bad.json() == failed_turn("missing")


async def test_provider_failure_is_not_answered_with_fabricated_output(
    summary_route, provider
):
    provider.failures = {"valid"}
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("valid")
    )
    assert provider.cases == ["valid"]
    assert response.status_code == 500, response.text
    assert response.json() == {
        "detail": (
            "Agent 'summarizer_agent' failed with ServiceUnavailableError "
            "(request_id=request-valid). See runtime logs for detail."
        )
    }
    provider.failures = set()
    recovered = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("valid", TENANTS[1])
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["result"]["summary"] == COMPLETE["summary"]


def budgeted_task(tenant: str, budget: object) -> dict:
    request = task("budgeted", tenant)
    request["context"]["max_output_tokens"] = budget
    return request


async def test_a_request_budget_caps_the_completion_and_ends_the_turn(
    summary_route, provider
):
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=budgeted_task(TENANTS[0], 40)
    )

    assert provider.max_tokens == [40]
    assert response.status_code == 200, response.text
    assert response.json() == {
        "status": "error",
        "agent": "summarizer_agent",
        "error": (
            "Agent 'summarizer_agent' generation for request 'request-budgeted' "
            "produced no confidence_score, key_points, summary"
        ),
    }


async def test_a_budget_never_raises_the_configured_completion(summary_route, provider):
    unbudgeted = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("budgeted", TENANTS[0])
    )
    above = await summary_route.post(
        "/agents/summarizer_agent/process", json=budgeted_task(TENANTS[1], 4096)
    )

    # The shipped-config fixture reserves 256 completion tokens.
    assert provider.max_tokens == [256, 256]
    assert [unbudgeted.status_code, above.status_code] == [200, 200]
    assert [
        unbudgeted.json()["result"]["summary"],
        above.json()["result"]["summary"],
    ] == [COMPLETE["summary"], COMPLETE["summary"]]


async def test_a_budgeted_request_is_never_answered_from_an_unbudgeted_cache(
    summary_route, provider
):
    """The response cache keys on the ``max_tokens`` sent, so a full answer
    cached for the tenant cannot stand in for a capped request."""
    full = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("budgeted", TENANTS[0])
    )
    capped = await summary_route.post(
        "/agents/summarizer_agent/process", json=budgeted_task(TENANTS[0], 40)
    )

    assert provider.max_tokens == [256, 40]
    assert full.json()["result"]["summary"] == COMPLETE["summary"], full.text
    assert capped.json()["status"] == "error", capped.text


async def test_concurrent_requests_keep_their_own_budget(summary_route, provider):
    """A budget bound for one request never reaches another in flight."""
    provider.barrier = threading.Barrier(2)
    try:
        capped, free = await asyncio.gather(
            summary_route.post(
                "/agents/summarizer_agent/process",
                json=budgeted_task(TENANTS[0], 40),
            ),
            summary_route.post(
                "/agents/summarizer_agent/process",
                json=task("budgeted", TENANTS[1]),
            ),
        )
    finally:
        provider.barrier = None

    assert sorted(provider.max_tokens) == [40, 256]
    assert capped.json()["status"] == "error", capped.text
    assert free.json()["result"]["summary"] == COMPLETE["summary"], free.text


@pytest.mark.parametrize("budget", [0, -3, "40", True, 40.0, None])
async def test_a_malformed_budget_is_refused_before_any_generation(
    summary_route, provider, budget
):
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=budgeted_task(TENANTS[0], budget)
    )

    assert provider.cases == []
    assert (response.status_code, response.json()) == (
        400,
        {
            "detail": (
                f"context.max_output_tokens must be a positive integer, got {budget!r}"
            )
        },
    )


def transport_query(case: str) -> str:
    return f"Two telescopes observe from sunset. case:{case}"


def no_answer_detail(agent: str, case: str) -> str:
    seed = derive_request_seed(transport_query(case), [])
    return (
        f"agent '{agent}' reported status=error: Agent '{agent}' generation for "
        f"request '{seed}' produced no parsable output"
    )


STREAM_FAILURE = (
    "SummarizerAgent streaming failed with LMOutputIncomplete. "
    "See server logs for detail."
)


def sse_frames(body: str) -> list[dict]:
    frames = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        payload = block[len("data:") :].strip()
        if payload != "[DONE]":
            frames.append(json.loads(payload))
    return frames


def delta_text(frames: list[dict]) -> str:
    return "".join(
        choice.get("delta", {}).get("content") or ""
        for frame in frames
        for choice in frame.get("choices", [])
    )


def chat_request(model: str, case: str, *, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": transport_query(case)}],
        "stream": stream,
    }


def observatory_photo() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 30, 90)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def attached_chat_request(model: str, case: str) -> dict:
    """A streamed turn carrying an image the summarizer answers from, so a
    tenant with nothing to search still generates."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": transport_query(case)},
                    {"type": "image_url", "image_url": {"url": observatory_photo()}},
                ],
            }
        ],
        "stream": True,
    }


async def test_v1_nonstreaming_refuses_the_turn(compat_route, provider):
    async with compat_route(streams_answer_tokens=False) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=chat_request(TEXT_MODEL, "prose", stream=False),
        )
    assert provider.cases == ["prose"]
    assert response.status_code == 502, response.text
    assert response.json() == {
        "error": {
            "message": no_answer_detail("text_analysis_agent", "prose"),
            "type": "server_error",
            "code": "upstream_no_answer",
        }
    }


async def test_v1_buffered_stream_ends_in_an_error_frame(
    compat_route, provider, caplog
):
    with caplog.at_level(logging.ERROR, logger=openai_compat.__name__):
        async with compat_route(streams_answer_tokens=False) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=chat_request(TEXT_MODEL, "prose", stream=True),
            )
    assert provider.cases == ["prose"]
    assert response.status_code == 200, response.text
    assert response.text.endswith("data: [DONE]\n\n")
    frames = sse_frames(response.text)
    assert frames[-1] == {
        "error": {
            "message": (
                "text_analysis_agent failed with NoAnswerError. "
                "See server logs for detail."
            ),
            "agent": "text_analysis_agent",
            "error_type": "NoAnswerError",
            "type": "server_error",
            "code": "internal_error",
        }
    }
    assert delta_text(frames) == ""
    assert [
        (record.getMessage(), str(record.exc_info[1]))
        for record in caplog.records
        if record.name == openai_compat.__name__
    ] == [
        (
            "chat.completions turn failed mid-stream",
            no_answer_detail("text_analysis_agent", "prose"),
        )
    ]


async def test_v1_token_stream_ends_in_an_error_frame(compat_route, provider):
    async with compat_route(streams_answer_tokens=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=attached_chat_request(SUMMARY_MODEL, "blank"),
        )
    assert provider.cases == ["blank"]
    assert response.status_code == 200, response.text
    assert response.text.endswith("data: [DONE]\n\n")
    frames = sse_frames(response.text)
    assert frames[-1] == {
        "error": {
            "message": STREAM_FAILURE,
            "type": "server_error",
            "code": "internal_error",
            "agent": "SummarizerAgent",
            "error_type": "LMOutputIncomplete",
        }
    }
    assert delta_text(frames) == ""


def a2a_request(agent: str, case: str, *, stream: bool) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/stream" if stream else "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "contextId": f"generation-{uuid.uuid4()}",
                "parts": [{"kind": "text", "text": transport_query(case)}],
            },
            "metadata": {
                "agent_name": agent,
                "tenant_id": TENANTS[0],
                "stream": stream,
            },
        },
    }


def a2a_events(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


async def test_a2a_turn_is_a_failed_task(a2a_route, provider):
    response = await a2a_route.post(
        "/", json=a2a_request("text_analysis_agent", "prose", stream=False)
    )
    assert provider.cases == ["prose"]
    assert response.status_code == 200, response.text
    status = response.json()["result"]["status"]
    assert status["state"] == "failed"
    assert json.loads(status["message"]["parts"][0]["text"]) == {
        "type": "error",
        "agent": "text_analysis_agent",
        "error_type": "NoAnswerError",
        "message": (
            "Agent 'text_analysis_agent' failed with NoAnswerError. "
            "See runtime logs for detail."
        ),
    }


@pytest.fixture
def schedule_grounding(monkeypatch):
    """Thread the schedule hit into the context the dispatcher's grounding
    resolution reads, the way a routed turn carries its search results. A2A
    request metadata has no field that reaches that context."""
    resolve = AgentDispatcher._resolve_answer_search_results

    async def threaded(self, query, tenant_id, context, top_k):
        return await resolve(
            self,
            query,
            tenant_id,
            {**(context or {}), "search_results": [SCHEDULE_HIT]},
            top_k,
        )

    monkeypatch.setattr(AgentDispatcher, "_resolve_answer_search_results", threaded)


async def test_a2a_stream_ends_in_a_failed_terminal_event(
    a2a_route, provider, schedule_grounding
):
    response = await a2a_route.post(
        "/", json=a2a_request("summarizer_agent", "blank", stream=True)
    )
    assert provider.cases == ["blank"]
    assert response.status_code == 200, response.text
    terminal = a2a_events(response.text)[-1]["result"]
    assert terminal["final"] is True
    assert terminal["status"]["state"] == "failed"
    payload = json.loads(terminal["status"]["message"]["parts"][0]["text"])
    assert payload["type"] == "error"
    assert payload["agent"] == "summarizer_agent"
    assert payload["error_type"] == "LMOutputIncomplete"
    assert payload["message"] == STREAM_FAILURE


NOTHING_TO_SEARCH = (
    f"Tenant {TENANTS[0]} has no servable search profile, so there is nothing "
    "to search for this request."
)
NO_SERVABLE_PROFILE = {
    "state": "no_servable_profile",
    "modalities": [],
    "profiles": [],
    "degraded_profiles": [],
    "degraded_query_rewrite": None,
    "undeployed_profiles": [],
    "result_count": 0,
}


async def test_streams_for_a_tenant_with_nothing_to_search_end_in_the_canned_reply(
    compat_route, a2a_route, provider
):
    async with compat_route(streams_answer_tokens=True) as client:
        v1 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=chat_request(SUMMARY_MODEL, "valid", stream=True),
        )
    a2a = await a2a_route.post(
        "/", json=a2a_request("summarizer_agent", "valid", stream=True)
    )
    assert provider.cases == []

    assert v1.status_code == 200, v1.text
    assert v1.text.endswith("data: [DONE]\n\n")
    frames = sse_frames(v1.text)
    assert [frame["choices"] for frame in frames] == [
        [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        [{"index": 0, "delta": {"content": NOTHING_TO_SEARCH}, "finish_reason": None}],
        [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    ]

    assert a2a.status_code == 200, a2a.text
    events = a2a_events(a2a.text)
    terminal = events[-1]["result"]
    assert terminal["final"] is True
    assert terminal["status"]["state"] == "input-required"
    assert json.loads(terminal["status"]["message"]["parts"][0]["text"]) == {
        "type": "final",
        "data": {
            "status": "success",
            "agent": "summarizer_agent",
            "message": NOTHING_TO_SEARCH,
            "grounding": NO_SERVABLE_PROFILE,
            "result": {
                "summary": NOTHING_TO_SEARCH,
                "key_points": [],
                "visual_insights": [],
                "confidence_score": 0.0,
                "thinking_phase": {
                    "key_themes": [],
                    "content_categories": [],
                    "relevance_scores": {},
                    "visual_elements": [],
                    "reasoning": NOTHING_TO_SEARCH,
                    "entity_insights": None,
                    "relationship_patterns": None,
                    "contextual_connections": None,
                },
                "metadata": {"grounding": NO_SERVABLE_PROFILE},
                "relationship_summary": None,
                "entity_analysis": None,
                "enhancement_applied": False,
            },
        },
    }
