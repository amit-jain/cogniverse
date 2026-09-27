"""Agent memory writes run in the background against real Mem0.

Real Vespa and real DenseOn back the Mem0 store; the LLM Mem0 extracts facts
with is a local OpenAI-compatible endpoint this module runs, so a test can
hold a write at the model, fail it, or count how many writes reach the model
at once. The agent path is DocumentAgent's real ``search_documents`` (with its
document backend stubbed, since memory is the boundary under test) and the
mixin's ``write_memory_in_background``.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_agents.background_memory_writes import (
    MEMORY_WRITE_CONCURRENCY,
    MEMORY_WRITE_DRAIN_TIMEOUT_S,
    MEMORY_WRITE_MAX_PENDING,
    drain_background_memory_writes,
)
from cogniverse_agents.document_agent import DocumentAgent, DocumentResult
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.tenant_helpers import MEMORY_TENANT_ID

pytestmark = [pytest.mark.integration]

# The search itself is one real memory read (DenseOn embed + Vespa query);
# a write held at the model is held far longer than this.
RESPONSE_BOUND_S = 20.0
MODEL_HOLD_LIMIT_S = 120.0
_TOKEN = re.compile(r"bgmem-[0-9a-f]{12}")


def _token() -> str:
    return f"bgmem-{uuid.uuid4().hex[:12]}"


class _ControlledLLM:
    """OpenAI-compatible chat endpoint answering Mem0's two extraction calls.

    The fact-extraction call returns the request's token as the one fact; the
    update call adds it. ``hold`` parks every call until ``release``;
    ``fail`` answers every call with a 500.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.released = threading.Event()
        self.released.set()
        self.fail = False
        self.in_flight = 0
        self.max_in_flight = 0
        endpoint = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = json.dumps(body["messages"])
                with endpoint.lock:
                    endpoint.in_flight += 1
                    endpoint.max_in_flight = max(
                        endpoint.max_in_flight, endpoint.in_flight
                    )
                try:
                    endpoint.released.wait(MODEL_HOLD_LIMIT_S)
                    if endpoint.fail:
                        self.respond(
                            500,
                            {
                                "error": {
                                    "message": "bgmem model overloaded",
                                    "type": "server_error",
                                }
                            },
                        )
                        return
                    token = _TOKEN.findall(prompt)[-1]
                    if "smart memory manager" in prompt:
                        content = {
                            "memory": [
                                {"id": "0", "text": f"{token} answered", "event": "ADD"}
                            ]
                        }
                    else:
                        content = {"facts": [f"{token} answered"]}
                    self.respond(
                        200,
                        {
                            "id": "chatcmpl-bgmem",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(content),
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        },
                    )
                finally:
                    with endpoint.lock:
                        endpoint.in_flight -= 1

            def respond(self, status, payload):
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def hold(self) -> None:
        self.released.clear()

    def release(self) -> None:
        self.released.set()

    def wait_in_flight(self, count: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.in_flight >= count:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"{count} memory write(s) never reached the model ({self.in_flight})"
        )


def _hit() -> DocumentResult:
    return DocumentResult(
        document_id="doc-7",
        document_url="s3://docs/doc-7.pdf",
        title="Quarterly filing",
        relevance_score=0.91,
        strategy_used="text",
    )


def _document_agent(*, shared_memory_vespa, shared_denseon, llm, agent_name):
    """A DocumentAgent whose memory is real Mem0 over real Vespa + DenseOn,
    extracting with ``llm``; only its document backend is stubbed."""
    agent = object.__new__(DocumentAgent)
    agent.memory_manager = None
    agent._memory_agent_name = None
    agent._memory_tenant_id = None
    agent._memory_initialized = False
    agent._memory_federation_enabled = False

    async def search_text(query, limit):
        return [_hit()]

    agent._search_text = search_text
    agent._deployed_strategy = lambda strategy: strategy

    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
        )
    )
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    assert (
        agent.initialize_memory(
            agent_name=agent_name,
            tenant_id=MEMORY_TENANT_ID,
            backend_host="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            backend_config_port=shared_memory_vespa["config_port"],
            llm_model="bgmem-extractor",
            llm_base_url=llm.url,
            llm_api_key="bgmem-stub",
            embedding_model="lightonai/DenseOn",
            embedder_base_url=shared_denseon,
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            auto_create_schema=False,
        )
        is True
    )
    return agent


@pytest.fixture
async def memory_env(shared_memory_vespa, shared_denseon):
    assert await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S) is True
    llm = _ControlledLLM()
    llm.thread.start()
    Mem0MemoryManager._instances.clear()
    agent_name = f"bg_memory_{uuid.uuid4().hex[:8]}"
    agent = _document_agent(
        shared_memory_vespa=shared_memory_vespa,
        shared_denseon=shared_denseon,
        llm=llm,
        agent_name=agent_name,
    )
    try:
        yield SimpleNamespace(llm=llm, agent=agent, agent_name=agent_name)
    finally:
        llm.release()
        await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S)
        agent.clear_memory()
        llm.server.shutdown()
        llm.server.server_close()
        Mem0MemoryManager._instances.clear()


def _stored(env, tokens) -> list[str]:
    """Memory texts for this agent that carry one of ``tokens``."""
    rows = env.agent.memory_manager.get_all_memories(
        tenant_id=MEMORY_TENANT_ID, agent_name=env.agent_name, limit=None
    )
    return sorted(
        row["memory"]
        for row in rows
        if any(token in row.get("memory", "") for token in tokens)
    )


def _stored_eventually(env, tokens, expected: list[str], timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    stored = _stored(env, tokens)
    while stored != expected and time.monotonic() < deadline:
        time.sleep(0.5)
        stored = _stored(env, tokens)
    return stored


def _queue_success(env, token: str) -> bool:
    return env.agent.write_memory_in_background(
        env.agent.remember_success,
        query=f"filing {token}",
        result={"result_count": 1},
    )


@pytest.mark.asyncio
async def test_the_search_answers_before_its_slow_memory_write_and_the_write_lands(
    memory_env,
):
    token = _token()
    memory_env.llm.hold()

    started = time.monotonic()
    results = await asyncio.wait_for(
        memory_env.agent.search_documents(f"filing {token}", strategy="text", limit=1),
        timeout=RESPONSE_BOUND_S,
    )
    answered_after = time.monotonic() - started
    # The write reached the model and is parked there, after the answer.
    memory_env.llm.wait_in_flight(1)
    stored_when_answered = _stored(memory_env, [token])
    memory_env.llm.release()
    drained = await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S)

    print(f"ANSWERED_WITH_WRITE_HELD_S={answered_after:.3f}")
    assert [r.document_id for r in results] == ["doc-7"]
    assert answered_after < RESPONSE_BOUND_S
    assert stored_when_answered == []
    assert drained is True
    assert _stored_eventually(memory_env, [token], [f"{token} answered"]) == [
        f"{token} answered"
    ]


@pytest.mark.asyncio
async def test_a_failing_memory_write_is_logged_and_the_search_is_unaffected(
    memory_env, caplog
):
    token = _token()
    memory_env.llm.fail = True

    with caplog.at_level(logging.ERROR):
        results = await asyncio.wait_for(
            memory_env.agent.search_documents(
                f"filing {token}", strategy="text", limit=1
            ),
            timeout=RESPONSE_BOUND_S,
        )
        drained = await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S)

    assert [r.document_id for r in results] == ["doc-7"]
    assert drained is True
    failures = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR and "bgmem model overloaded" in r.getMessage()
    ]
    assert len(failures) == 1
    assert MEMORY_TENANT_ID in failures[0]
    assert memory_env.agent_name in failures[0]
    assert _stored(memory_env, [token]) == []


@pytest.mark.asyncio
async def test_background_writes_reach_the_model_at_the_bounded_concurrency(
    memory_env, caplog
):
    tokens = [_token() for _ in range(MEMORY_WRITE_MAX_PENDING)]
    overflow_token = _token()
    memory_env.llm.hold()

    queued = [_queue_success(memory_env, token) for token in tokens]
    memory_env.llm.wait_in_flight(MEMORY_WRITE_CONCURRENCY)
    with caplog.at_level(logging.WARNING):
        overflow = _queue_success(memory_env, overflow_token)
    # Every write is queued and the model is held: nothing more may start.
    time.sleep(1.0)
    in_flight_while_held = memory_env.llm.in_flight
    memory_env.llm.release()
    drained = await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S)

    assert queued == [True] * MEMORY_WRITE_MAX_PENDING
    assert overflow is False
    assert in_flight_while_held == MEMORY_WRITE_CONCURRENCY
    assert memory_env.llm.max_in_flight == MEMORY_WRITE_CONCURRENCY
    dropped = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]
    assert len(dropped) == 1
    assert MEMORY_TENANT_ID in dropped[0]
    assert memory_env.agent_name in dropped[0]
    assert drained is True
    expected = sorted(f"{token} answered" for token in tokens)
    assert _stored_eventually(memory_env, tokens, expected) == expected
    assert _stored(memory_env, [overflow_token]) == []


@pytest.mark.asyncio
async def test_shutdown_drain_logs_the_writes_still_pending_at_its_budget(
    memory_env, caplog
):
    tokens = [_token() for _ in range(MEMORY_WRITE_CONCURRENCY + 1)]
    memory_env.llm.hold()
    for token in tokens:
        assert _queue_success(memory_env, token) is True
    memory_env.llm.wait_in_flight(MEMORY_WRITE_CONCURRENCY)

    with caplog.at_level(logging.WARNING):
        started = time.monotonic()
        drained_at_budget = await drain_background_memory_writes(1.0)
        drain_elapsed = time.monotonic() - started
    memory_env.llm.release()
    drained_after_release = await drain_background_memory_writes(
        MEMORY_WRITE_DRAIN_TIMEOUT_S
    )

    assert drained_at_budget is False
    assert drain_elapsed < 5.0
    report = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]
    assert len(report) == 1
    label = f"{MEMORY_TENANT_ID}/{memory_env.agent_name}"
    assert report[0].count(label) == len(tokens)
    assert drained_after_release is True
    # The writes already at the model land; the one still queued at the
    # budget was cancelled and is named in the report instead.
    landed = sorted(f"{token} answered" for token in tokens[:MEMORY_WRITE_CONCURRENCY])
    assert _stored_eventually(memory_env, tokens, landed) == landed
