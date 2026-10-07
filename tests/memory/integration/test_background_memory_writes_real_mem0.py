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
    get_background_memory_writer,
)
from cogniverse_agents.document_agent import DocumentAgent, DocumentResult
from cogniverse_agents.search_agent import SearchAgent
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.tenant_helpers import MEM0_ROUNDTRIP_TENANT_ID, MEMORY_TENANT_ID

pytestmark = [pytest.mark.integration]

# The search itself is one real memory read (DenseOn embed + Vespa query);
# a write held at the model is held far longer than this.
RESPONSE_BOUND_S = 20.0
SHORT_MODEL_TIMEOUT_S = 3.0
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
        self.requests = 0
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
                    endpoint.requests += 1
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

    def wait_requests(self, count: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.requests >= count:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"{count} model call(s) never arrived ({self.requests} did)"
        )

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


def _config_manager(shared_memory_vespa, shared_denseon) -> ConfigManager:
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
    return config_manager


def _initialize(
    agent, *, tenant_id, shared_memory_vespa, shared_denseon, llm, agent_name
):
    """What the dispatcher's ``_init_agent_memory`` does for one request."""
    agent.set_tenant_for_context(tenant_id)
    return agent.initialize_memory(
        agent_name=agent_name,
        tenant_id=tenant_id,
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        llm_model="bgmem-extractor",
        llm_base_url=llm.url,
        llm_api_key="bgmem-stub",
        embedding_model="lightonai/DenseOn",
        embedder_base_url=shared_denseon,
        config_manager=_config_manager(shared_memory_vespa, shared_denseon),
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        auto_create_schema=False,
    )


def _memoryless(agent_cls):
    agent = object.__new__(agent_cls)
    agent.memory_manager = None
    agent._memory_agent_name = None
    agent._memory_tenant_id = None
    agent._memory_initialized = False
    agent._memory_federation_enabled = False
    return agent


def _document_agent(*, shared_memory_vespa, shared_denseon, llm, agent_name):
    """A DocumentAgent whose memory is real Mem0 over real Vespa + DenseOn,
    extracting with ``llm``; only its document backend is stubbed."""
    agent = _memoryless(DocumentAgent)

    async def search_text(query, limit):
        return [_hit()]

    agent._search_text = search_text
    agent._deployed_strategy = lambda strategy: strategy
    assert (
        _initialize(
            agent,
            tenant_id=MEMORY_TENANT_ID,
            shared_memory_vespa=shared_memory_vespa,
            shared_denseon=shared_denseon,
            llm=llm,
            agent_name=agent_name,
        )
        is True
    )
    return agent


@pytest.fixture
async def memory_env(shared_memory_vespa, shared_denseon):
    async for env in _memory_env(shared_memory_vespa, shared_denseon):
        yield env


@pytest.fixture
async def memory_env_with_short_model_timeout(
    shared_memory_vespa, shared_denseon, monkeypatch
):
    """``memory_env`` whose Mem0 model calls time out after
    ``SHORT_MODEL_TIMEOUT_S`` instead of the shipped bound."""
    from cogniverse_core.memory import manager as manager_module

    monkeypatch.setattr(
        manager_module, "MEM0_LLM_CALL_TIMEOUT_S", SHORT_MODEL_TIMEOUT_S, raising=False
    )
    async for env in _memory_env(shared_memory_vespa, shared_denseon):
        yield env


async def _memory_env(shared_memory_vespa, shared_denseon):
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


def _queue_success_labelled(env, token: str, tenant_label: str) -> bool:
    """Queue a real success write whose writer accounting names
    ``tenant_label``; the write itself lands in the agent's own store."""
    return get_background_memory_writer().submit(
        lambda: env.agent.remember_success(f"filing {token}", {"result_count": 1}),
        tenant_id=tenant_label,
        agent_name=env.agent_name,
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
    from cogniverse_agents.background_memory_writes import (
        MEMORY_WRITE_MAX_PENDING_PER_TENANT,
    )

    labels = [MEMORY_TENANT_ID, "bgmem:t1", "bgmem:t2", "bgmem:t3"]
    assert len(labels) * MEMORY_WRITE_MAX_PENDING_PER_TENANT == MEMORY_WRITE_MAX_PENDING
    tokens = [_token() for _ in range(MEMORY_WRITE_MAX_PENDING)]
    overflow_token = _token()
    memory_env.llm.hold()

    queued = [
        _queue_success_labelled(
            memory_env, token, labels[i // MEMORY_WRITE_MAX_PENDING_PER_TENANT]
        )
        for i, token in enumerate(tokens)
    ]
    memory_env.llm.wait_in_flight(MEMORY_WRITE_CONCURRENCY)
    with caplog.at_level(logging.WARNING):
        overflow = _queue_success_labelled(memory_env, overflow_token, "bgmem:t4")
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
    assert "bgmem:t4" in dropped[0]
    assert memory_env.agent_name in dropped[0]
    assert drained is True
    expected = sorted(f"{token} answered" for token in tokens)
    assert _stored_eventually(memory_env, tokens, expected) == expected
    assert _stored(memory_env, [overflow_token]) == []


@pytest.mark.asyncio
async def test_a_hung_model_times_out_and_frees_the_writers_for_queued_writes(
    memory_env_with_short_model_timeout, caplog
):
    """Both writers' calls hang at the model; each times out, is logged, and
    frees its writer, so the write queued behind them reaches the model while
    the hung calls are still open."""
    env = memory_env_with_short_model_timeout
    tokens = [_token() for _ in range(MEMORY_WRITE_CONCURRENCY + 1)]
    env.llm.hold()

    with caplog.at_level(logging.ERROR):
        for token in tokens:
            assert _queue_success(env, token) is True
        try:
            env.llm.wait_requests(
                MEMORY_WRITE_CONCURRENCY + 1, timeout=10 * SHORT_MODEL_TIMEOUT_S
            )
        finally:
            env.llm.release()
        drained = await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S)

    assert drained is True
    timed_out = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.memory_aware_mixin"
        and r.levelno >= logging.ERROR
        and "timed out" in r.getMessage().lower()
    ]
    assert len(timed_out) == MEMORY_WRITE_CONCURRENCY
    assert all(env.agent_name in message for message in timed_out)
    assert _stored_eventually(env, [tokens[-1]], [f"{tokens[-1]} answered"]) == [
        f"{tokens[-1]} answered"
    ]
    assert _stored(env, tokens[:-1]) == []


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


class _FakeHit:
    def __init__(self, doc_id):
        self.document = SimpleNamespace(id=doc_id, metadata={"title": doc_id})
        self.score = 0.9


@pytest.fixture
async def shared_search_agent(shared_memory_vespa, shared_denseon):
    """One SearchAgent instance serving two tenants, as a shared agent does:
    each request binds its tenant's memory on it. Memory is real Mem0 over
    real Vespa + DenseOn; only the video search backend is stubbed."""
    assert await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S) is True
    llm = _ControlledLLM()
    llm.thread.start()
    Mem0MemoryManager._instances.clear()
    agent_name = f"bg_shared_{uuid.uuid4().hex[:8]}"
    agent = _memoryless(SearchAgent)
    agent.active_profile = "p1"
    # No profile names an inference service, so the stub encoder is used as is.
    agent.search_config = {}
    agent.query_encoder = SimpleNamespace(encode=lambda q: [[0.0] * 4])
    agent._build_date_filter = lambda *a, **k: None
    agent._search_backend = lambda query_dict: [_FakeHit("d1")]

    def bind(tenant_id):
        return _initialize(
            agent,
            tenant_id=tenant_id,
            shared_memory_vespa=shared_memory_vespa,
            shared_denseon=shared_denseon,
            llm=llm,
            agent_name=agent_name,
        )

    try:
        yield SimpleNamespace(agent=agent, agent_name=agent_name, bind=bind)
    finally:
        await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S)
        for tenant_id in (MEMORY_TENANT_ID, MEM0_ROUNDTRIP_TENANT_ID):
            Mem0MemoryManager(tenant_id=tenant_id).clear_agent_memory(
                tenant_id=tenant_id, agent_name=agent_name
            )
        llm.server.shutdown()
        llm.server.server_close()
        Mem0MemoryManager._instances.clear()


def _store_rows(store_tenant, agent_name, token) -> list[tuple[str, str]]:
    """``(partition, text)`` rows carrying ``token`` in ``store_tenant``'s
    Mem0 store, read under either tenant's partition."""
    manager = Mem0MemoryManager(tenant_id=store_tenant)
    found = []
    for partition in (MEMORY_TENANT_ID, MEM0_ROUNDTRIP_TENANT_ID):
        for row in manager.get_all_memories(
            tenant_id=partition, agent_name=agent_name, limit=None
        ):
            if token in row.get("memory", ""):
                found.append((partition, row["memory"]))
    return sorted(found)


def _store_rows_eventually(store_tenant, agent_name, token, expected, timeout=30.0):
    deadline = time.monotonic() + timeout
    rows = _store_rows(store_tenant, agent_name, token)
    while rows != expected and time.monotonic() < deadline:
        time.sleep(0.5)
        rows = _store_rows(store_tenant, agent_name, token)
    return rows


async def _recall(env, tenant_id, token) -> str:
    """What a later request for ``tenant_id`` recalls about ``token``."""
    env.agent.set_tenant_for_context(tenant_id)
    return await asyncio.to_thread(env.agent.get_relevant_context, token, 10) or ""


@pytest.mark.asyncio
async def test_a_queued_write_lands_in_its_tenant_store_after_another_tenant_rebinds(
    shared_search_agent,
):
    """Tenant A's search queues its success memory behind busy writers; tenant
    B's request binds B's memory on the same agent before A's write runs. The
    write still lands in A's store, findable by A, and B's store has none of
    it."""
    env = shared_search_agent
    token = _token()
    release_blockers = threading.Event()
    for _ in range(MEMORY_WRITE_CONCURRENCY):
        assert get_background_memory_writer().submit(
            lambda: release_blockers.wait(30), tenant_id="other", agent_name="blocker"
        )

    # Each request binds its tenant on the loop, as SearchAgent._process_impl
    # does, and its memory in a worker thread, as the dispatcher does.
    async def request_a():
        env.agent.set_tenant_for_context(MEMORY_TENANT_ID)
        assert await asyncio.to_thread(env.bind, MEMORY_TENANT_ID) is True
        return await asyncio.to_thread(
            env.agent._search_by_text,
            query=f"filing {token}",
            tenant_id=MEMORY_TENANT_ID,
            modality="video",
            top_k=1,
        )

    async def request_b():
        env.agent.set_tenant_for_context(MEM0_ROUNDTRIP_TENANT_ID)
        assert await asyncio.to_thread(env.bind, MEM0_ROUNDTRIP_TENANT_ID) is True

    try:
        results = await asyncio.create_task(request_a())
        await asyncio.create_task(request_b())
    finally:
        release_blockers.set()
    assert await drain_background_memory_writes(MEMORY_WRITE_DRAIN_TIMEOUT_S) is True

    expected = [(MEMORY_TENANT_ID, f"{token} answered")]
    assert [r["id"] for r in results] == ["d1"]
    assert (
        _store_rows_eventually(MEMORY_TENANT_ID, env.agent_name, token, expected)
        == expected
    )
    assert _store_rows(MEM0_ROUNDTRIP_TENANT_ID, env.agent_name, token) == []
    assert token in await _recall(env, MEMORY_TENANT_ID, token)


@pytest.mark.asyncio
async def test_concurrent_synchronous_writes_on_a_shared_agent_keep_their_tenants(
    shared_search_agent,
):
    """Two requests bind their tenants' memory on one agent, A first, then
    both write synchronously from worker threads at once. Each memory lands
    in its own tenant's store."""
    env = shared_search_agent
    token_a, token_b = _token(), _token()
    a_bound, b_bound = asyncio.Event(), asyncio.Event()

    async def request(tenant_id, token, bound, wait_before_bind, wait_before_write):
        await wait_before_bind()
        env.agent.set_tenant_for_context(tenant_id)
        assert await asyncio.to_thread(env.bind, tenant_id) is True
        bound.set()
        await wait_before_write()
        return await asyncio.to_thread(
            env.agent.remember_success, f"filing {token}", {"result_count": 1}
        )

    async def nothing():
        return None

    written = await asyncio.gather(
        request(MEMORY_TENANT_ID, token_a, a_bound, nothing, b_bound.wait),
        request(MEM0_ROUNDTRIP_TENANT_ID, token_b, b_bound, a_bound.wait, nothing),
    )

    assert written == [True, True]
    expected_a = [(MEMORY_TENANT_ID, f"{token_a} answered")]
    expected_b = [(MEM0_ROUNDTRIP_TENANT_ID, f"{token_b} answered")]
    assert (
        _store_rows_eventually(MEMORY_TENANT_ID, env.agent_name, token_a, expected_a)
        == expected_a
    )
    assert (
        _store_rows_eventually(
            MEM0_ROUNDTRIP_TENANT_ID, env.agent_name, token_b, expected_b
        )
        == expected_b
    )
    assert _store_rows(MEM0_ROUNDTRIP_TENANT_ID, env.agent_name, token_a) == []
    assert _store_rows(MEMORY_TENANT_ID, env.agent_name, token_b) == []
    assert token_a in await _recall(env, MEMORY_TENANT_ID, token_a)
    assert token_b in await _recall(env, MEM0_ROUNDTRIP_TENANT_ID, token_b)
