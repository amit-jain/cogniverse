"""Report answer-model failures keep their failure state across transports.

The report agent and dispatcher are production implementations.  A local
OpenAI-compatible HTTP server supplies fixed grounding-independent model
responses or real socket failures; only retrieval and service configuration
are isolated so each assertion reaches the report-generation call.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_runtime.a2a_task_store import RedisTaskStore
from cogniverse_runtime.agent_dispatcher import AgentDispatcher, AnswerGrounding
from cogniverse_runtime.main import _build_shared_a2a_protocol
from cogniverse_runtime.routers import openai_compat
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

FAILING_TENANT = "reportfail:reportfail"
HEALTHY_TENANT = "reportok:reportok"
FAILING_KEY = "report-failure-key"
HEALTHY_KEY = "report-success-key"
MODEL = "cogniverse/report"
QUERY = "summarize the fixed launch evidence"
HIT = {
    "id": "launch-7",
    "title": "Launch review",
    "description": "The launch review records evidence code ORBIT-42.",
    "score": 0.91,
}
GROUNDED_SUMMARY = "The launch review records evidence code ORBIT-42."


class _ReportProvider:
    """OpenAI-compatible server with controlled HTTP and socket outcomes."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mode = str(body["model"])
                with provider.lock:
                    provider.calls.append(mode)
                provider.entered.set()

                if mode == "report-503":
                    self._send(
                        503,
                        {"error": {"message": "controlled report LM unavailable"}},
                    )
                    return
                if mode == "report-reset":
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if mode in {"report-timeout", "report-block"}:
                    provider.release.wait(timeout=10)
                    if mode == "report-timeout":
                        self._send(200, self._completion(mode, GROUNDED_SUMMARY))
                        return
                if mode == "report-parse":
                    self._send(200, self._completion(mode, "not structured output"))
                    return
                self._send(
                    200,
                    self._completion(
                        mode,
                        json.dumps(
                            {
                                "reasoning": "The fixed evidence directly answers the query.",
                                "executive_summary": GROUNDED_SUMMARY,
                                "key_findings": "ORBIT-42 is the recorded evidence code",
                                "recommendations": "Retain the launch review",
                                "confidence_score": "0.91",
                            }
                        ),
                    ),
                )

            @staticmethod
            def _completion(model: str, content: str) -> dict[str, Any]:
                return {
                    "id": f"chatcmpl-{model}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
        assert self.thread.is_alive() is False


class _RecordingConversationStore:
    def __init__(self) -> None:
        self.turns: list[dict[str, str]] = []

    def get_history(self, context_id, max_turns=20):
        return []

    def store_turn(self, context_id, role, content):
        self.turns.append({"role": role, "content": content})

    def get_missing_assistant_markers(self, context_id):
        return []

    def store_missing_assistant_marker(self, context_id, reason):
        self.turns.append({"role": "assistant_missing", "content": str(reason)})


def _agent(manager: ConfigManager, endpoint: str, mode: str) -> DetailedReportAgent:
    agent = DetailedReportAgent(
        DetailedReportDeps(
            thinking_enabled=False,
            visual_analysis_enabled=False,
            technical_analysis_enabled=True,
        ),
        config_manager=manager,
    )
    agent._llm_config = LLMEndpointConfig(
        model=f"openai/{mode}",
        api_base=endpoint,
        api_key="local-report-test",
        request_timeout=0.2 if mode == "report-timeout" else 2.0,
        num_retries=0,
        max_tokens=256,
        temperature=0.0,
    )
    return agent


@pytest.fixture
def provider():
    with _ReportProvider() as served:
        yield served


@pytest.fixture
def report_runtime(provider, monkeypatch):
    from cogniverse_foundation.telemetry import manager as telemetry_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    installed = TelemetryManager(TelemetryConfig(enabled=False))
    monkeypatch.setattr(telemetry_module, "_telemetry_manager", installed)

    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=FAILING_TENANT, config_manager=manager)
    registry.register_agent(
        AgentEndpoint(
            name="detailed_report_agent",
            url="http://unused",
            capabilities=["detailed_report"],
            streams_answer_tokens=True,
        )
    )
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=manager, schema_loader=None
    )
    conversations = {
        FAILING_TENANT: _RecordingConversationStore(),
        HEALTHY_TENANT: _RecordingConversationStore(),
    }
    modes = {
        FAILING_TENANT: "report-503",
        HEALTHY_TENANT: "report-ok",
    }

    async def grounded(*_args, **_kwargs):
        return AnswerGrounding(hits=[HIT], state="retrieved")

    def build_agent(_cls, _deps_cls, _name, tenant_id):
        return _agent(manager, provider.url, modes[tenant_id])

    dispatcher._resolve_answer_search_results = grounded
    dispatcher._build_answer_agent = build_agent
    dispatcher._init_agent_memory = lambda *_args, **_kwargs: None
    dispatcher.consult_egress_policy = lambda *_args, **_kwargs: None
    dispatcher._verify_egress = lambda *_args, **_kwargs: None
    dispatcher._conversation_store_factory = conversations.__getitem__
    return dispatcher, conversations, modes


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("report-503", "ServiceUnavailableError"),
        ("report-reset", "InternalServerError"),
        ("report-timeout", "Timeout"),
    ],
)
@pytest.mark.asyncio
async def test_direct_report_process_propagates_answer_model_failures(
    provider, mode, error_type
):
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    agent = _agent(manager, provider.url, mode)

    with pytest.raises(Exception) as raised:
        await agent.process(
            {
                "tenant_id": FAILING_TENANT,
                "query": QUERY,
                "search_results": [HIT],
                "include_visual_analysis": False,
            }
        )

    assert type(raised.value).__name__ == error_type


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("report-503", "ServiceUnavailableError"),
        ("report-reset", "InternalServerError"),
        ("report-timeout", "Timeout"),
    ],
)
@pytest.mark.asyncio
async def test_actual_dispatcher_propagates_model_failure_without_saving_success(
    report_runtime, mode, error_type
):
    dispatcher, conversations, modes = report_runtime
    modes[FAILING_TENANT] = mode

    with pytest.raises(Exception) as raised:
        await dispatcher.dispatch(
            "detailed_report_agent",
            QUERY,
            {
                "tenant_id": FAILING_TENANT,
                "context_id": "failed-report-context",
                "request_id": "failed-report-request",
                "include_visual_analysis": False,
            },
        )
    await dispatcher.drain_conversation_saves()

    assert type(raised.value).__name__ == error_type
    assert conversations[FAILING_TENANT].turns == []


@contextmanager
def _serving(app, lifespan: str = "off"):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan=lifespan))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started is True
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        listener.close()
        assert thread.is_alive() is False


@pytest.fixture
def compat_url(report_runtime):
    dispatcher, _, _ = report_runtime
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys(
        {FAILING_KEY: FAILING_TENANT, HEALTHY_KEY: HEALTHY_TENANT}
    )
    openai_compat.set_model_map({MODEL: "detailed_report_agent"})
    openai_compat.set_key_resolver(None)
    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    with _serving(app) as url:
        yield url
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})


_REDIS_CONTAINER = "redis-report-failure-transports"


@pytest.fixture(scope="module")
def redis_url():
    """Owned Redis for the handler that actually serves ``/a2a``."""
    override = os.environ.get("COGNIVERSE_TEST_REDIS_URL")
    if override:
        yield override
        return

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    subprocess.run(["docker", "rm", "-f", _REDIS_CONTAINER], capture_output=True)
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            _REDIS_CONTAINER,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        pytest.fail(f"Failed to start Redis: {started.stderr}")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", _REDIS_CONTAINER, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "rm", "-f", _REDIS_CONTAINER], capture_output=True)
        pytest.fail("Redis did not become ready within 30s")
    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", _REDIS_CONTAINER], capture_output=True)


@pytest.fixture
def a2a_url(report_runtime, redis_url):
    """``/a2a`` served by the handler production serves it with.

    The stock ``DefaultRequestHandler`` + ``InMemoryTaskStore`` this used to
    build is not what serves ``/a2a`` any more: ``RedisRequestHandler`` wraps
    the same execution in lease acquire/renew/release and mirrors every event
    through a Redis relay. Asserting the failure envelope against the stock
    handler proved it for a handler production does not use.
    """
    dispatcher, _, _ = report_runtime

    class _Registry:
        def list_agents(self):
            return ["detailed_report_agent"]

        def get_agent(self, name):
            if name != "detailed_report_agent":
                return None
            return SimpleNamespace(capabilities=["report"])

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Built inside the serving loop: the store's Redis client and the
        # handler's cancel listener are both bound to the loop that runs them.
        built = await _build_shared_a2a_protocol(
            agent_registry=_Registry(),
            dispatcher=dispatcher,
            redis_url=redis_url,
            replica_id="report-failure-replica",
            max_tasks=64,
            lease_seconds=30,
            cancel_timeout_seconds=10,
            drain_timeout_seconds=10,
        )
        app.mount("/a2a", built.app)
        try:
            yield
        finally:
            await built.close()

    app = FastAPI(lifespan=_lifespan)
    with _serving(app, lifespan="on") as url:
        yield url


def _chat_body(*, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": QUERY}],
        "stream": stream,
    }


def _sse_frames(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]


def _a2a_request(*, stream: bool) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": f"report-failure-{stream}",
        "method": "message/stream" if stream else "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "message_id": f"report-message-{stream}",
                "parts": [{"kind": "text", "text": QUERY}],
            },
            "metadata": {
                "agent_name": "detailed_report_agent",
                "query": QUERY,
                "tenant_id": FAILING_TENANT,
                "stream": stream,
            },
        },
    }


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("report-503", "ServiceUnavailableError"),
        ("report-reset", "InternalServerError"),
        ("report-timeout", "Timeout"),
    ],
)
@pytest.mark.asyncio
async def test_openai_nonstream_uses_the_existing_internal_error_contract(
    compat_url, report_runtime, mode, error_type
):
    _, _, modes = report_runtime
    modes[FAILING_TENANT] = mode
    async with httpx.AsyncClient(base_url=compat_url, timeout=60.0) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body(stream=False),
            headers={"Authorization": f"Bearer {FAILING_KEY}"},
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": (
            f"detailed_report_agent failed with {error_type}. See server logs for detail."
        ),
        "type": "server_error",
        "code": "internal_error",
        "agent": "detailed_report_agent",
        "error_type": error_type,
    }


@pytest.mark.asyncio
async def test_openai_stream_ends_with_one_existing_error_frame(compat_url):
    async with httpx.AsyncClient(base_url=compat_url, timeout=60.0) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body(stream=True),
            headers={"Authorization": f"Bearer {FAILING_KEY}"},
        )

    assert response.status_code == 200
    frames = _sse_frames(response.text)
    assert frames == [
        {
            "id": frames[0]["id"],
            "object": "chat.completion.chunk",
            "created": frames[0]["created"],
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "error": {
                "message": (
                    "DetailedReportAgent streaming failed with "
                    "ServiceUnavailableError. See server logs for detail."
                ),
                "type": "server_error",
                "code": "internal_error",
                "agent": "DetailedReportAgent",
                "error_type": "ServiceUnavailableError",
            }
        },
    ]
    assert response.text.rstrip().endswith("data: [DONE]")


def test_openai_socket_disconnect_cancels_blocked_report_without_late_success(
    compat_url, report_runtime, provider
):
    _, conversations, modes = report_runtime
    modes[FAILING_TENANT] = "report-block"

    with pytest.raises(httpx.ReadTimeout):
        httpx.post(
            f"{compat_url}/v1/chat/completions",
            json=_chat_body(stream=False),
            headers={"Authorization": f"Bearer {FAILING_KEY}"},
            timeout=0.5,
        )
    assert provider.entered.wait(timeout=5) is True

    deadline = time.monotonic() + 3
    while openai_compat.in_flight_count() != 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert openai_compat.in_flight_count() == 0

    provider.release.set()
    time.sleep(0.5)
    assert openai_compat.in_flight_count() == 0
    assert conversations[FAILING_TENANT].turns == []


@pytest.mark.asyncio
async def test_a2a_failure_envelope_is_readable_by_a_peer_through_redis(
    a2a_url, report_runtime, redis_url
):
    """The failed task a peer replica would read is the one served.

    ``/a2a`` is served by ``RedisRequestHandler`` over the shared store, so
    the envelope has to survive in Redis, not only in the responding
    process's memory — which is what the previous ``InMemoryTaskStore``
    fixture could never show.
    """
    from a2a.types import TaskState

    _, _, modes = report_runtime
    modes[FAILING_TENANT] = "report-503"
    async with httpx.AsyncClient(base_url=a2a_url, timeout=60.0) as client:
        response = await client.post("/a2a/", json=_a2a_request(stream=False))

    assert response.status_code == 200
    terminal = response.json()["result"]
    served_text = terminal["status"]["message"]["parts"][0]["text"]

    peer = await RedisTaskStore.from_url(redis_url, max_tasks=64)
    try:
        stored = await peer.get(terminal["id"])
    finally:
        await peer.close()

    assert stored is not None
    assert stored.status.state == TaskState.failed
    assert stored.status.message.parts[0].root.text == served_text
    assert json.loads(served_text) == {
        "type": "error",
        "agent": "detailed_report_agent",
        "error_type": "ServiceUnavailableError",
        "message": (
            "Agent 'detailed_report_agent' failed with ServiceUnavailableError. "
            "See runtime logs for detail."
        ),
    }


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("report-503", "ServiceUnavailableError"),
        ("report-reset", "InternalServerError"),
        ("report-timeout", "Timeout"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_a2a_terminal_task_is_failed_for_a_report_model_outage(
    a2a_url, report_runtime, stream, mode, error_type
):
    _, _, modes = report_runtime
    modes[FAILING_TENANT] = mode
    async with httpx.AsyncClient(base_url=a2a_url, timeout=60.0) as client:
        response = await client.post("/a2a/", json=_a2a_request(stream=stream))

    assert response.status_code == 200
    envelope = (
        [frame for frame in _sse_frames(response.text) if frame["result"].get("final")][
            0
        ]
        if stream
        else response.json()
    )
    terminal = envelope["result"]
    task_id = terminal["taskId"] if stream else terminal["id"]
    context_id = terminal["contextId"]
    status_message_id = terminal["status"]["message"]["messageId"]
    assert str(UUID(task_id)) == task_id
    assert str(UUID(context_id)) == context_id
    assert str(UUID(status_message_id)) == status_message_id
    if stream:
        status_note = " (LM HTTP 408)" if mode == "report-timeout" else ""
        error_message = (
            f"DetailedReportAgent streaming failed with {error_type}{status_note}. "
            "See server logs for detail."
        )
    else:
        error_message = (
            f"Agent 'detailed_report_agent' failed with {error_type}. "
            "See runtime logs for detail."
        )
    error_payload = {
        "type": "error",
        "agent": "detailed_report_agent",
        "error_type": error_type,
    }
    if stream and mode == "report-timeout":
        error_payload["status"] = 408
    error_payload["message"] = error_message
    error_text = json.dumps(error_payload)
    status = {
        "message": {
            "kind": "message",
            "messageId": status_message_id,
            "parts": [{"kind": "text", "text": error_text}],
            "role": "agent",
        },
        "state": "failed",
    }
    if stream:
        expected = {
            "id": f"report-failure-{stream}",
            "jsonrpc": "2.0",
            "result": {
                "contextId": context_id,
                "final": True,
                "kind": "status-update",
                "status": status,
                "taskId": task_id,
            },
        }
    else:
        expected = {
            "id": f"report-failure-{stream}",
            "jsonrpc": "2.0",
            "result": {
                "contextId": context_id,
                "history": [
                    {
                        "contextId": context_id,
                        "kind": "message",
                        "messageId": f"report-message-{stream}",
                        "parts": [{"kind": "text", "text": QUERY}],
                        "role": "user",
                        "taskId": task_id,
                    }
                ],
                "id": task_id,
                "kind": "task",
                "status": status,
            },
        }
    assert envelope == expected
    assert "controlled report LM unavailable" not in response.text


@pytest.mark.asyncio
async def test_concurrent_tenants_keep_failure_and_success_metadata_independent(
    report_runtime,
):
    dispatcher, conversations, _ = report_runtime

    failed, succeeded = await asyncio.gather(
        dispatcher.dispatch(
            "detailed_report_agent",
            QUERY,
            {"tenant_id": FAILING_TENANT, "request_id": "failing-concurrent"},
        ),
        dispatcher.dispatch(
            "detailed_report_agent",
            QUERY,
            {"tenant_id": HEALTHY_TENANT, "request_id": "healthy-concurrent"},
        ),
        return_exceptions=True,
    )

    assert type(failed).__name__ == "ServiceUnavailableError"
    assert succeeded["status"] == "success"
    assert succeeded["result"]["executive_summary"] == GROUNDED_SUMMARY
    assert succeeded["result"]["metadata"]["report_degraded"] is False
    assert succeeded["result"]["metadata"]["report_degraded_reason"] == ""
    assert conversations[FAILING_TENANT].turns == []
    assert conversations[HEALTHY_TENANT].turns == []


@pytest.mark.asyncio
async def test_parse_failure_keeps_the_existing_dispatcher_error_envelope(provider):
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=FAILING_TENANT, config_manager=manager)
    registry.register_agent(
        AgentEndpoint(
            name="detailed_report_agent",
            url="http://unused",
            capabilities=["detailed_report"],
        )
    )
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=manager, schema_loader=None
    )

    async def grounded(*_args, **_kwargs):
        return AnswerGrounding(hits=[HIT], state="retrieved")

    dispatcher._resolve_answer_search_results = grounded
    dispatcher._build_answer_agent = lambda *_args: _agent(
        manager, provider.url, "report-parse"
    )
    dispatcher._init_agent_memory = lambda *_args, **_kwargs: None

    result = await dispatcher.dispatch(
        "detailed_report_agent",
        QUERY,
        {
            "tenant_id": FAILING_TENANT,
            "request_id": "parse-report-request",
            "include_visual_analysis": False,
        },
    )

    assert result == {
        "status": "error",
        "agent": "detailed_report_agent",
        "error": (
            "Agent 'detailed_report_agent' generation for request "
            "'parse-report-request' produced no parsable output"
        ),
    }


@pytest.mark.asyncio
async def test_gateway_does_not_stamp_success_after_the_report_child_fails(
    report_runtime,
):
    dispatcher, _, _ = report_runtime

    class Gateway:
        async def _process_impl(self, _input):
            return SimpleNamespace(
                complexity="simple",
                routed_to="detailed_report_agent",
                detected_modalities=[],
                modality="document",
                generation_type="detailed_report",
                confidence=1.0,
                fast_path_confidence_threshold=0.8,
                gliner_threshold=0.7,
            )

    dispatcher._get_or_build_gateway_agent = lambda _tenant: asyncio.sleep(
        0, result=Gateway()
    )

    with pytest.raises(Exception) as raised:
        await dispatcher._execute_gateway_task(
            QUERY, {"tenant_id": FAILING_TENANT}, FAILING_TENANT
        )

    assert type(raised.value).__name__ == "ServiceUnavailableError"


@pytest.mark.asyncio
async def test_report_failure_is_an_unsuccessful_orchestrator_child(
    report_runtime, monkeypatch
):
    from contextlib import nullcontext

    from cogniverse_agents.orchestrator_agent import (
        AccumulatedEvidence,
        AgentStep,
        OrchestrationPlan,
        OrchestratorAgent,
        OrchestratorDeps,
        OrchestratorInput,
    )
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_runtime.routers import agents as agents_router

    dispatcher, conversations, modes = report_runtime
    modes[FAILING_TENANT] = "report-503"
    registry = AgentRegistry(
        tenant_id=FAILING_TENANT, config_manager=dispatcher._config_manager
    )
    registry.register_agent(
        AgentEndpoint(
            name="detailed_report_agent",
            url="http://report-runtime",
            process_endpoint="/agents/detailed_report_agent/process",
            capabilities=["detailed_report"],
        )
    )
    remembered: list[tuple[str, str]] = []
    scored_outcomes: list[dict[str, Any]] = []

    class PreparedOrchestrator(OrchestratorAgent):
        async def _create_plan(self, query, conversation_context, gateway_context):
            return OrchestrationPlan(
                query=query,
                reasoning="report required",
                steps=[
                    AgentStep(
                        agent_name="detailed_report_agent",
                        input_data={"query": query},
                        reasoning="write report",
                    )
                ],
            )

        async def _iterative_retrieval_loop(self, query, plan, **kwargs):
            executed = await self._execute_plan(
                plan,
                tenant_id=kwargs["tenant_id"],
                workflow_id=kwargs["workflow_id"],
                execution_order_sink=kwargs["execution_order_sink"],
                agent_observations_sink=kwargs["agent_observations_sink"],
            )
            kwargs["agent_results_sink"].update(executed)
            return AccumulatedEvidence(iterations_executed=1, exit_reason="max_iter")

        def _ensure_memory_for_tenant(self, tenant_id):
            return None

        def get_relevant_context(self, query):
            return ""

        def remember_success(self, query, summary):
            remembered.append((query, summary))

        async def _semantic_router_lm_context(self, tenant_id):
            return nullcontext()

        async def _emit_orchestration_span(self, **outcome):
            scored_outcomes.append(outcome)

    monkeypatch.setattr(agents_router, "_ensure_dispatcher", lambda: dispatcher)
    child_app = FastAPI()
    child_app.include_router(agents_router.router, prefix="/agents")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=child_app),
        base_url="http://report-runtime",
    ) as child_client:
        orchestrator = PreparedOrchestrator(
            deps=OrchestratorDeps(),
            registry=registry,
            config_manager=dispatcher._config_manager,
            http_client=child_client,
        )
        orchestrator.telemetry_manager = TelemetryManager(
            TelemetryConfig(enabled=False)
        )
        orchestrator.workflow_intelligence = None
        result = await orchestrator.process(
            OrchestratorInput(query=QUERY, tenant_id=FAILING_TENANT)
        )

    child_error = (
        "HTTPStatusError: Server error '500 Internal Server Error' for url "
        "'http://report-runtime/agents/detailed_report_agent/process'\n"
        "For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500"
    )
    assert result.final_output["status"] == "failed"
    assert result.final_output["aggregated_content"] == ""
    assert result.agent_results == {
        "detailed_report_agent": {"status": "error", "message": child_error}
    }
    assert remembered == []
    assert scored_outcomes == [
        {
            "tenant_id": FAILING_TENANT,
            "workflow_id": result.workflow_id,
            "query": QUERY,
            "agent_sequence": ["detailed_report_agent"],
            "execution_time": scored_outcomes[0]["execution_time"],
            "success": False,
            "tasks_completed": 0,
            "pattern": "sequential",
            "execution_order": ["detailed_report_agent"],
            "error_summary": f"detailed_report_agent: {child_error}",
            "agent_observations": [
                {
                    "agent_name": "detailed_report_agent",
                    "execution_time": scored_outcomes[0]["agent_observations"][0][
                        "execution_time"
                    ],
                    "success": False,
                }
            ],
        }
    ]
    assert conversations[FAILING_TENANT].turns == []


@pytest.mark.asyncio
async def test_cancelling_a_blocked_report_call_cannot_return_a_late_success(provider):
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    agent = _agent(manager, provider.url, "report-block")
    task = asyncio.create_task(
        agent.process(
            {
                "tenant_id": FAILING_TENANT,
                "query": QUERY,
                "search_results": [HIT],
                "include_visual_analysis": False,
            }
        )
    )
    entered = await asyncio.to_thread(provider.entered.wait, 5)
    assert entered is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    provider.release.set()
    await asyncio.sleep(0.2)

    assert task.cancelled() is True
