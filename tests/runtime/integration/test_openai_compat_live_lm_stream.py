"""The /v1 token stream driven end to end by a real LM.

The production app, the real dispatcher and the served summarizer stream a
real LM's answer through ``dispatch_stream``, the agent's token events and the
SSE framing. The summarizer resolves its LM from the configured endpoint, and
that endpoint is a recording HTTP proxy in front of the provisioned model, so
each assertion names the request that reached the LM. Fault cases point the
same endpoint at a dead port or at an HTTP server that answers 413.

Comparability: the streamed turn runs first under a per-run marker, so the
DSPy request cache cannot answer it and the model's tokens stream. The same
request served non-streamed is then answered from that cache with the same
completion, and the proxy proves it: one upstream request across both turns.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import logging
import os
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
import uvicorn

from cogniverse_agents.summarizer_agent import SummarizerAgent, SummarizerDeps
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime import main as runtime_main
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import openai_compat
from tests.utils.hermetic_llm import MODEL
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_CONFIG = json.loads((REPO_ROOT / "configs" / "config.json").read_text())
AGENT = "summarizer_agent"
AGENT_CLASS = SummarizerAgent.__name__
MODEL_NAME = next(
    name
    for name, agent in SHIPPED_CONFIG["harness"]["models"].items()
    if agent == AGENT
)
SUMMARY_CAP = SummarizerDeps.model_fields["max_summary_length"].default
TENANT_A = "sselive:alpha"
TENANT_B = "sselive:beta"
KEY_A = "sse-live-key-alpha"
KEY_B = "sse-live-key-beta"
RUN = uuid.uuid4().hex[:12]

# Budgets are measured (see the commit body): refusal plus one retry backoff
# ends well inside it, a hang-up lands inside one disconnect poll.
DEAD_PORT_BUDGET_SECONDS = 10.0
DISCONNECT_BUDGET_SECONDS = 5.0
# Measured with the same 5 ms ticker as the admin routes
# (tests/runtime/integration/test_admin_harness_keys.py:210-235), less the time
# a garbage collection (which stops every thread) spent inside each gap. Eight
# live streams keep the LM threads busy, and at the largest gaps a stack
# sampler finds the loop idle in select, waiting for the GIL: 31-110 ms over
# ten runs. The structural pins below separate a blocked loop from a busy
# process; this bound catches a blocking call on the loop.
LOOP_GAP_BUDGET_SECONDS = 0.15
# DSPy drives a streamed LM call as this coroutine on the loop that called
# streamify (anyio's from-thread bridge).
LM_STREAM_TASK_QUALNAME = "AsyncIOBackend.run_async_from_thread.<locals>.task_wrapper"
LM_413_ERROR_TYPE = "APIError"
# litellm reports a refused connection as InternalServerError.
LM_UNREACHABLE_ERROR_TYPE = "InternalServerError"
# Nothing is retrieved for a tenant with no content, so the summarizer sends
# the question as both its content and its query.
MARKER_COPIES_PER_REQUEST = 2

# Forces a long answer (past the summary cap), a paragraph break and a quoted
# phrase: whitespace, JSON escapes and truncation are all on the streamed path.
HARD_PROMPT = (
    "[{marker}] In three paragraphs of at least sixty words each, separated by "
    'blank lines, explain why ice floats on water. Quote the phrase "less '
    'dense" exactly once.'
)

CHUNK_KEYS = {"id", "object", "created", "model", "choices"}
CHOICE_KEYS = {"index", "delta", "finish_reason"}
USAGE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RecordingProxy:
    """A real HTTP hop in front of the LM that records every request body."""

    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.loads(raw)
                with proxy._lock:
                    proxy.requests.append(body)
                headers = {"Content-Type": "application/json"}
                if self.headers.get("Authorization"):
                    headers["Authorization"] = self.headers["Authorization"]
                path = (
                    self.path[len("/v1") :]
                    if self.path.startswith("/v1")
                    else self.path
                )
                with httpx.stream(
                    "POST",
                    f"{proxy.upstream}{path}",
                    content=raw,
                    headers=headers,
                    timeout=180.0,
                ) as upstream:
                    self.send_response(upstream.status_code)
                    self.send_header(
                        "Content-Type",
                        upstream.headers.get("content-type", "application/json"),
                    )
                    self.end_headers()
                    try:
                        for piece in upstream.iter_raw():
                            self.wfile.write(piece)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.api_base = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def recorded(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.requests)

    def clear(self) -> None:
        with self._lock:
            self.requests.clear()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class StatusServer:
    """A real HTTP LM endpoint that rejects every request with one status."""

    def __init__(self, status: int, body: bytes) -> None:
        self.hits = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                server.hits += 1
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.api_base = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _config_with_primary(api_base: str, directory: Path) -> Path:
    """The activated session config with the primary LM moved to ``api_base``."""
    source = json.loads(Path(os.environ["COGNIVERSE_CONFIG"]).read_text())
    config = copy.deepcopy(source)
    config["llm_config"]["primary"]["api_base"] = api_base
    target = directory / f"config-{uuid.uuid4().hex[:8]}.json"
    target.write_text(json.dumps(config))
    return target


@pytest.fixture(scope="module")
def upstream_base(ensure_host_ollama) -> str:
    return os.environ["TEST_LLM_API_BASE"]


@pytest.fixture(scope="module")
def proxy(upstream_base):
    hop = RecordingProxy(upstream_base)
    yield hop
    hop.close()


@pytest.fixture(scope="module")
def dispatcher():
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    shipped = SHIPPED_CONFIG["agents"][AGENT]
    registry.register_agent(
        AgentEndpoint(
            name=AGENT,
            url=shipped["url"],
            capabilities=shipped["capabilities"],
            streams_answer_tokens=shipped["streams_answer_tokens"],
        )
    )
    return AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )


def _point_primary_at(api_base: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "COGNIVERSE_CONFIG", str(_config_with_primary(api_base, tmp_path))
    )


@pytest.fixture()
def app(dispatcher, tmp_path):
    """The production app with the /v1 wiring its lifespan performs."""
    config = get_config(tenant_id=TENANT_A, config_manager=dispatcher._config_manager)
    runtime_main.configure_ambient_dspy(create_dspy_lm(config.get_llm_config().primary))
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A, KEY_B: TENANT_B})
    openai_compat.set_model_map(dict(SHIPPED_CONFIG["harness"]["models"]))
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()
    yield runtime_main.app
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})
    openai_compat.clear_continuations()


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=300.0
    ) as http_client:
        yield http_client


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _body(prompt: str, stream: bool, **extra: Any) -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        **extra,
    }


def _data_lines(raw: str) -> List[str]:
    return [
        line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")
    ]


def _frames(raw: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in _data_lines(raw) if line != "[DONE]"]


def _deltas(raw: str) -> List[str]:
    return [
        frame["choices"][0]["delta"]["content"]
        for frame in _frames(raw)
        if frame.get("choices") and "content" in frame["choices"][0]["delta"]
    ]


def _user_messages(request: Dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content"))
        for message in request["messages"]
        if message.get("role") == "user"
    )


def _assert_framing(raw: str, *, usage_requested: bool) -> str:
    """Pin the SSE framing of one completed streamed turn; return its text."""
    lines = _data_lines(raw)
    assert lines.count("[DONE]") == 1
    assert lines[-1] == "[DONE]"
    frames = _frames(raw)
    assert [frame for frame in frames if "error" in frame] == []
    assert {frame["id"] for frame in frames} == {frames[0]["id"]}
    assert frames[0]["id"].startswith("chatcmpl-")
    assert {frame["object"] for frame in frames} == {"chat.completion.chunk"}
    assert {frame["model"] for frame in frames} == {MODEL_NAME}
    assert {frame["created"] for frame in frames} == {frames[0]["created"]}

    usage_frames = [frame for frame in frames if "usage" in frame]
    choice_frames = [frame for frame in frames if frame["choices"]]
    if usage_requested:
        assert usage_frames == [frames[-1]]
        assert set(frames[-1]) == CHUNK_KEYS | {"usage"}
        assert frames[-1]["choices"] == []
        usage = frames[-1]["usage"]
        assert set(usage) == USAGE_KEYS
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        assert choice_frames == frames[:-1]
    else:
        assert usage_frames == []
        assert choice_frames == frames

    for frame in choice_frames:
        assert set(frame) == CHUNK_KEYS
        assert len(frame["choices"]) == 1
        assert set(frame["choices"][0]) == CHOICE_KEYS
        assert frame["choices"][0]["index"] == 0
    assert choice_frames[0]["choices"][0] == {
        "index": 0,
        "delta": {"role": "assistant"},
        "finish_reason": None,
    }
    assert choice_frames[-1]["choices"][0] == {
        "index": 0,
        "delta": {},
        "finish_reason": "stop",
    }
    middle = choice_frames[1:-1]
    for frame in middle:
        assert set(frame["choices"][0]["delta"]) == {"content"}
        assert frame["choices"][0]["finish_reason"] is None
    return "".join(frame["choices"][0]["delta"]["content"] for frame in middle)


class TestLiveTokenStream:
    """A real LM's answer, streamed through the real route."""

    async def test_stream_equals_the_non_streamed_turn_for_the_same_request(
        self, client, proxy, tmp_path, monkeypatch
    ):
        _point_primary_at(proxy.api_base, tmp_path, monkeypatch)
        proxy.clear()
        prompt = HARD_PROMPT.format(marker=f"{RUN}-equality")

        streamed = await client.post(
            "/v1/chat/completions",
            json=_body(prompt, stream=True, stream_options={"include_usage": True}),
            headers=_auth(KEY_A),
        )
        assert streamed.status_code == 200
        assert streamed.headers["content-type"].startswith("text/event-stream")
        text = _assert_framing(streamed.text, usage_requested=True)

        upstream = proxy.recorded()
        assert [request["model"] for request in upstream] == [MODEL]
        assert [request.get("stream") for request in upstream] == [True]
        assert (
            _user_messages(upstream[0]).count(f"{RUN}-equality")
            == MARKER_COPIES_PER_REQUEST
        )

        plain = await client.post(
            "/v1/chat/completions",
            json=_body(prompt, stream=False),
            headers=_auth(KEY_A),
        )
        assert plain.status_code == 200
        assert len(proxy.recorded()) == 1
        assert text == plain.json()["choices"][0]["message"]["content"]
        assert text[-1] == "…"
        assert len(text) <= SUMMARY_CAP

        replay = await client.post(
            "/v1/chat/completions",
            json=_body(prompt, stream=True),
            headers=_auth(KEY_A),
        )
        assert replay.status_code == 200
        assert _assert_framing(replay.text, usage_requested=False) == text
        assert _deltas(replay.text) == [text]
        assert len(proxy.recorded()) == 1
        live = _deltas(streamed.text)
        assert live != [text]
        assert "".join(live) == text
        assert [delta for delta in live if not delta] == []


class TestLmFaults:
    """The LM boundary down or rejecting: a named error frame, then [DONE]."""

    async def test_dead_lm_port_ends_in_the_named_error_frame(
        self, client, tmp_path, monkeypatch
    ):
        _point_primary_at(f"http://127.0.0.1:{_free_port()}/v1", tmp_path, monkeypatch)
        prompt = HARD_PROMPT.format(marker=f"{RUN}-dead-port")

        started = time.monotonic()
        response = await client.post(
            "/v1/chat/completions",
            json=_body(prompt, stream=True),
            headers=_auth(KEY_A),
        )
        elapsed = time.monotonic() - started

        assert response.status_code == 200
        lines = _data_lines(response.text)
        assert json.loads(lines[0])["choices"][0]["delta"] == {"role": "assistant"}
        assert [json.loads(line) for line in lines[1:-1]] == [
            {
                "error": {
                    "message": (
                        f"{AGENT_CLASS} streaming failed with {LM_UNREACHABLE_ERROR_TYPE}. "
                        "See server logs for detail."
                    ),
                    "type": "server_error",
                    "code": "internal_error",
                }
            }
        ]
        assert lines[-1] == "[DONE]"
        assert elapsed < DEAD_PORT_BUDGET_SECONDS

    async def test_lm_413_ends_in_an_error_frame_naming_the_status(
        self, client, tmp_path, monkeypatch
    ):
        rejecting = StatusServer(413, b"payload too large")
        try:
            _point_primary_at(rejecting.api_base, tmp_path, monkeypatch)
            prompt = HARD_PROMPT.format(marker=f"{RUN}-413")
            response = await client.post(
                "/v1/chat/completions",
                json=_body(prompt, stream=True),
                headers=_auth(KEY_A),
            )
        finally:
            rejecting.close()

        assert response.status_code == 200
        lines = _data_lines(response.text)
        assert json.loads(lines[0])["choices"][0]["delta"] == {"role": "assistant"}
        assert [json.loads(line) for line in lines[1:-1]] == [
            {
                "error": {
                    "message": (
                        f"{AGENT_CLASS} streaming failed with {LM_413_ERROR_TYPE} "
                        "(LM HTTP 413). See server logs for detail."
                    ),
                    "type": "server_error",
                    "code": "internal_error",
                }
            }
        ]
        assert lines[-1] == "[DONE]"
        assert rejecting.hits == 1


def _agent_turn_tasks() -> List[asyncio.Task]:
    """The agent turns running on this loop, found by the coroutine's name."""
    return [
        task
        for task in asyncio.all_tasks()
        if getattr(task.get_coro(), "__qualname__", "").endswith(
            "_stream_with_progress.<locals>._run_impl"
        )
    ]


class TestClientDisconnect:
    """A hang-up mid-stream stops the turn and leaves nothing behind."""

    async def test_hang_up_cancels_the_turn_without_an_error_or_a_continuation(
        self, app, proxy, tmp_path, monkeypatch, caplog
    ):
        _point_primary_at(proxy.api_base, tmp_path, monkeypatch)
        proxy.clear()
        caplog.set_level(logging.INFO)
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=port, lifespan="off", log_level="warning"
            )
        )
        serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        prompt = HARD_PROMPT.format(marker=f"{RUN}-hang-up")
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=300.0
            ) as http_client:
                async with http_client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=_body(prompt, stream=True),
                    headers=_auth(KEY_A),
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and '"content"' in line:
                            break
                    running = _agent_turn_tasks()
                    assert len(running) == 1
                    hung_up_at = time.monotonic()
            deadline = hung_up_at + DISCONNECT_BUDGET_SECONDS
            while (
                _agent_turn_tasks() or openai_compat.in_flight_count()
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert _agent_turn_tasks() == []
            assert [task.cancelled() for task in running] == [True]
            assert openai_compat.in_flight_count() == 0
            assert openai_compat.continuation_count() == 0
        finally:
            server.should_exit = True
            await serving

        messages = [record.getMessage() for record in caplog.records]
        assert [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.ERROR
        ] == []
        assert messages.count(f"chat.completions stream for {AGENT} cancelled") == 1
        assert [
            _user_messages(request).count(f"{RUN}-hang-up")
            for request in proxy.recorded()
        ] == [MARKER_COPIES_PER_REQUEST]


class TestConcurrentStreams:
    """Eight live streams for two tenants through one app."""

    async def test_no_token_cross_talk_and_the_loop_stays_free(
        self, client, proxy, tmp_path, monkeypatch
    ):
        _point_primary_at(proxy.api_base, tmp_path, monkeypatch)
        proxy.clear()
        construction_threads: List[threading.Thread] = []
        construct = SummarizerAgent.__init__

        def recording_init(agent, *args, **kwargs):
            construction_threads.append(threading.current_thread())
            construct(agent, *args, **kwargs)

        monkeypatch.setattr(SummarizerAgent, "__init__", recording_init)
        turns = [
            (key, HARD_PROMPT.format(marker=f"{RUN}-{tenant}-{index}"))
            for index in range(4)
            for key, tenant in ((KEY_A, TENANT_A), (KEY_B, TENANT_B))
        ]
        warmup = await client.post(
            "/v1/chat/completions",
            json=_body(HARD_PROMPT.format(marker=f"{RUN}-warmup"), stream=True),
            headers=_auth(KEY_A),
        )
        assert warmup.status_code == 200
        proxy.clear()
        construction_threads.clear()
        gaps: List[tuple] = []
        collections: List[tuple] = []
        lm_stream_tasks_on_loop: set = set()
        stop = asyncio.Event()

        def record_collection(phase: str, info: Dict[str, Any]) -> None:
            if phase == "start":
                collections.append((time.perf_counter(), None))
            elif collections:
                collections[-1] = (collections[-1][0], time.perf_counter())

        async def ticker() -> None:
            previous = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append((previous, now))
                previous = now
                lm_stream_tasks_on_loop.update(
                    id(task)
                    for task in asyncio.all_tasks()
                    if getattr(task.get_coro(), "__qualname__", "")
                    == LM_STREAM_TASK_QUALNAME
                )

        gc.callbacks.append(record_collection)
        ticking = asyncio.create_task(ticker())
        try:
            streamed = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json=_body(prompt, stream=True),
                        headers=_auth(key),
                    )
                    for key, prompt in turns
                )
            )
        finally:
            stop.set()
            await ticking
            gc.callbacks.remove(record_collection)

        assert [response.status_code for response in streamed] == [200] * 8
        texts = [_assert_framing(r.text, usage_requested=False) for r in streamed]
        markers = [prompt.split("]")[0][1:] for _, prompt in turns]
        upstream = proxy.recorded()
        assert sorted(
            marker
            for request in upstream
            for marker in markers
            if _user_messages(request).count(marker) == MARKER_COPIES_PER_REQUEST
        ) == sorted(markers)
        assert len(upstream) == 8

        plains = []
        for key, prompt in turns:
            plain = await client.post(
                "/v1/chat/completions",
                json=_body(prompt, stream=False),
                headers=_auth(key),
            )
            plains.append(plain.json()["choices"][0]["message"]["content"])
        assert len(proxy.recorded()) == 8
        assert texts == plains
        assert lm_stream_tasks_on_loop == set()
        assert len(construction_threads) == 16
        assert [
            thread
            for thread in construction_threads
            if thread is threading.main_thread()
        ] == []
        assert len({_frames(r.text)[0]["id"] for r in streamed}) == 8
        # A collection stops every thread; the part of a gap it spent is the
        # collector's, not a call holding the loop.
        pauses = [(begin, end) for begin, end in collections if end is not None]
        loop_gaps = [
            (end - begin)
            - sum(
                max(0.0, min(end, p_end) - max(begin, p_begin))
                for p_begin, p_end in pauses
            )
            for begin, end in gaps
        ]
        print(
            f"LOOP_MAX_GAP_MS={max(loop_gaps) * 1000:.3f} over {len(loop_gaps)} "
            f"ticks; with collections {max(e - b for b, e in gaps) * 1000:.3f}; "
            f"longest collection "
            f"{max((e - b for b, e in pauses), default=0.0) * 1000:.3f}"
        )
        assert max(loop_gaps) < LOOP_GAP_BUDGET_SECONDS
