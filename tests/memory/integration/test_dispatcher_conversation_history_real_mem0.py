"""Server-managed conversation history through the REAL agent dispatcher
and REAL Mem0.

The messaging gateway sends only a ``context_id`` and no history; the
runtime's ``AgentDispatcher.dispatch`` loads that context's recent turns
from Mem0 before the agent runs and appends the two new turns after. This
drives the real dispatch path against a real Vespa-backed Mem0 store —
only the agent execution is stubbed, because Mem0 is the boundary under
test, not the search agent. Every assertion is on real stored/reloaded
content.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.conversation import ConversationStore
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_LOAD_TIMEOUT_S,
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_TIMEOUT_S,
    AgentDispatcher,
    ConversationPersistFailed,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model
from tests.utils.tenant_helpers import MEM0_ROUNDTRIP_TENANT_ID

pytestmark = [pytest.mark.integration]

TENANT = "acme:acme"


def _build_manager_with_cm(*, shared_memory_vespa, shared_denseon):
    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    mm = Mem0MemoryManager(tenant_id=SYSTEM_TENANT_ID)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    return mm, cm


def _build_manager(*, shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
    mm, _cm = _build_manager_with_cm(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    return mm


def _no_wiki(dispatcher: AgentDispatcher) -> None:
    """Keep the post-dispatch wiki auto-file out of scope.

    Stubs the hook itself rather than the dispatcher's background scheduling:
    conversation persistence is scheduled the same way, and disabling that
    would disable the behaviour under test.
    """

    async def _skip(*args, **kwargs):
        return None

    dispatcher._maybe_auto_file_wiki = _skip


def _dispatcher_with_real_store(mm) -> AgentDispatcher:
    """Real dispatcher whose conversation store is a real ConversationStore
    on the real Mem0 manager — no in-memory double."""
    config_manager = MagicMock()
    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )
    d._conversation_store_factory = lambda tenant_id: ConversationStore(mm, tenant_id)
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    _no_wiki(d)
    return d


def _dispatcher_with_real_construction(cm) -> AgentDispatcher:
    """Real dispatcher with NO store factory injected — it must build its own
    ConversationStore + Mem0MemoryManager from the real ConfigManager, the same
    way the served runtime does. Exercises the production construction path that
    the seam tests bypass."""
    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=cm,
        schema_loader=MagicMock(),
    )
    # factory deliberately left None: _build_conversation_store must construct
    # the real manager itself.
    assert d._conversation_store_factory is None
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    _no_wiki(d)
    return d


def _reply_with(dispatcher, replies_by_query, seen):
    async def _fake(query, tenant_id, top_k, conversation_history=None, **kwargs):
        seen.append(list(conversation_history or []))
        return {"message": replies_by_query.get(query, "ok"), "entities": []}

    dispatcher._execute_search_task = _fake


async def _dispatch(dispatcher, query, context_id, tenant=TENANT, **extra):
    context = {"tenant_id": tenant, "context_id": context_id}
    context.update(extra)
    return await dispatcher.dispatch(
        agent_name="search_agent", query=query, context=context
    )


@pytest.mark.asyncio
async def test_history_round_trips_through_real_mem0(
    shared_memory_vespa, shared_denseon
):
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(
        d,
        {"what is colpali": "a late-interaction model", "how many dims": "128"},
        seen,
    )

    r1 = await _dispatch(d, "what is colpali", ctx)
    assert r1["message"] == "a late-interaction model"
    assert seen[0] == []  # first turn, nothing prior

    r2 = await _dispatch(d, "how many dims", ctx)
    assert r2["message"] == "128"
    # The second dispatch saw the exact turns the first one persisted —
    # the user's query and the agent's real reply text, in order.
    assert seen[1] == [
        {"role": "user", "content": "what is colpali"},
        {"role": "assistant", "content": "a late-interaction model"},
    ]

    # And once the off-path saves land, the real Mem0 store holds all four
    # turns with exact content: every query the gateway sent and every reply
    # the agent returned.
    assert await d.drain_conversation_saves() is True
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}
    persisted = ConversationStore(mm, TENANT).get_history(ctx)
    assert persisted == [
        {"role": "user", "content": "what is colpali"},
        {"role": "assistant", "content": "a late-interaction model"},
        {"role": "user", "content": "how many dims"},
        {"role": "assistant", "content": "128"},
    ]


@pytest.mark.asyncio
async def test_contexts_do_not_bleed_in_real_mem0(shared_memory_vespa, shared_denseon):
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    ctx_a = f"chat{uuid.uuid4().hex[:10]}"
    ctx_b = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(d, {}, seen)

    await _dispatch(d, "message in A", ctx_a)
    await _dispatch(d, "message in B", ctx_b)
    # B's dispatch must not see A's turn.
    assert seen[1] == []

    assert await d.drain_conversation_saves() is True
    store = ConversationStore(mm, TENANT)
    assert store.get_history(ctx_a) == [
        {"role": "user", "content": "message in A"},
        {"role": "assistant", "content": "ok"},
    ]
    assert store.get_history(ctx_b) == [
        {"role": "user", "content": "message in B"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_explicit_history_bypasses_management_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """A caller supplying its own conversation_history is respected and
    nothing is persisted to Mem0 for that context."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(d, {}, seen)

    supplied = [{"role": "user", "content": "caller-managed"}]
    await _dispatch(d, "q", ctx, conversation_history=supplied)

    assert seen[0] == supplied
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}
    assert ConversationStore(mm, TENANT).get_history(ctx) == []


@pytest.mark.asyncio
async def test_same_context_saves_land_in_scheduling_order_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """Four concurrent messages on ONE context persist all eight turns to real
    Mem0 in the order their replies were produced.

    Persistence is off the reply path, so ordering is the chain's job, not the
    reply's. Each dispatch is held at its agent boundary and released in a
    fixed order, so the expected stored order is exact rather than whichever
    write finished first.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    gates = {i: asyncio.Event() for i in range(4)}
    at_agent: list = []

    async def _fake(query, tenant_id, top_k, conversation_history=None, **kwargs):
        index = int(query.rsplit(" ", 1)[1])
        at_agent.append(index)
        await gates[index].wait()
        return {"message": f"reply {index}", "entities": []}

    d._execute_search_task = _fake

    dispatches = [asyncio.create_task(_dispatch(d, f"turn {i}", ctx)) for i in range(4)]
    # Every dispatch has loaded its history and is parked at the agent, so no
    # save has been scheduled yet and the release order below fixes the chain.
    while len(at_agent) < 4:
        await asyncio.sleep(0.05)
    assert sorted(at_agent) == [0, 1, 2, 3]

    for index in (3, 2, 1, 0):
        gates[index].set()
        assert (await dispatches[index])["message"] == f"reply {index}"

    assert await d.drain_conversation_saves() is True
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}
    assert ConversationStore(mm, TENANT).get_history(ctx) == [
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "reply 3"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 0"},
        {"role": "assistant", "content": "reply 0"},
    ]


@pytest.mark.asyncio
async def test_gateway_simple_persists_downstream_answer_to_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """A gateway 'simple' route persists the DOWNSTREAM agent's answer as the
    assistant turn in real Mem0 — not the routing breadcrumb. Reloaded from real
    Mem0, the stored assistant turn is the exact answer the response path
    rendered. Pre-fix the stored turn was ``Routed '<q>' to search_agent
    (simple)``, which then fed the next turn's anaphora rewrite."""
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from cogniverse_agents.gateway_agent import GatewayOutput
    from cogniverse_runtime.agent_dispatcher import _GatewayAgentEntry

    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )

    config_manager = MagicMock()
    config_manager.get_system_config.return_value = SystemConfig(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )
    d._conversation_store_factory = lambda tenant_id: ConversationStore(mm, tenant_id)
    _no_wiki(d)

    gw = MagicMock()
    gw.capabilities = {"gateway"}
    se = MagicMock()
    se.capabilities = {"search"}
    d._registry.get_agent.side_effect = lambda name: {
        "gateway_agent": gw,
        "search_agent": se,
    }.get(name)

    # Force a 'simple' classification without building the real GatewayAgent.
    # The real GatewayOutput, so the dispatcher reads every field the gateway
    # contract declares instead of whichever subset a stand-in happened to set.
    gwout = GatewayOutput(
        query="show kubernetes storage",
        complexity="simple",
        modality="video",
        generation_type="raw_results",
        routed_to="search_agent",
        confidence=0.9,
        fast_path_confidence_threshold=0.4,
        gliner_threshold=0.3,
        reasoning="keyword-routed video retrieval",
    )
    d._gateway_agents.set(
        TENANT,
        _GatewayAgentEntry(
            agent=SimpleNamespace(_process_impl=AsyncMock(return_value=gwout)),
            loaded_at=time.monotonic(),
        ),
    )

    # Stub ONLY the downstream agent's answer at its execution boundary.
    answer = "Found 2 results for 'kubernetes storage'"

    async def _fake_search(
        query, tenant_id, top_k, conversation_history=None, **kwargs
    ):
        return {
            "status": "success",
            "agent": "search_agent",
            "message": answer,
            "results_count": 2,
            "results": [{"document_id": "v1"}, {"document_id": "v2"}],
            "profile": "p",
            "search_mode": "hybrid",
        }

    d._execute_search_task = _fake_search

    ctx = f"chat{uuid.uuid4().hex[:10]}"
    result = await d.dispatch(
        agent_name="gateway_agent",
        query="show kubernetes storage",
        context={"tenant_id": TENANT, "context_id": ctx},
    )
    assert result["message"] == answer

    # Reloaded from REAL Mem0 once the off-path save lands: the stored
    # assistant turn is the answer, not the routing breadcrumb.
    assert await d.drain_conversation_saves() is True
    persisted = ConversationStore(mm, TENANT).get_history(ctx)
    assert persisted == [
        {"role": "user", "content": "show kubernetes storage"},
        {"role": "assistant", "content": answer},
    ]


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_dispatch_degrades_when_memory_unavailable():
    """History is enrichment: when the store is unavailable the agent still
    answers with no history — the reply is never lost to a memory outage.
    The store build raises here (as get_all_memories does on a real Mem0
    outage), so this exercises the dispatcher's real degrade path with no
    infrastructure needed."""

    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
    )

    def _raise(_tenant):
        raise ConnectionError("mem0 unreachable")

    d._conversation_store_factory = _raise
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    _no_wiki(d)
    seen: list = []
    _reply_with(d, {"q": "answered anyway"}, seen)

    ctx = f"chat{uuid.uuid4().hex[:10]}"
    result = await _dispatch(d, "q", ctx)

    assert result["message"] == "answered anyway"
    assert seen[0] == []  # degraded to no history, still ran

    # The save could not run either, and that loss is readable rather than
    # silent: the outage reaches a consumer as the store's own error.
    assert await d.drain_conversation_saves() is True
    assert d.conversation_persist_status() == {"pending": 0, "failed": [(TENANT, ctx)]}
    failure = d.conversation_persist_failure(TENANT, ctx)
    assert type(failure) is ConversationPersistFailed
    assert type(failure.__cause__) is ConnectionError
    assert str(failure.__cause__) == "mem0 unreachable"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_recorded_persistence_failures_evict_oldest_first():
    """The failure record is bounded: contexts are unbounded (one per chat), so
    a sustained outage must not grow the dispatcher's record for the pod's
    lifetime. The newest failure displaces the oldest, and every retained entry
    still names its own context."""
    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
    )

    def _raise(_tenant):
        raise ConnectionError("mem0 unreachable")

    d._conversation_store_factory = _raise
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    _no_wiki(d)
    _reply_with(d, {"q": "answered anyway"}, [])

    contexts = [
        f"chat{index:04d}" for index in range(CONVERSATION_PERSIST_FAILURE_CAPACITY + 3)
    ]
    for context_id in contexts:
        await _dispatch(d, "q", context_id)
    assert await d.drain_conversation_saves() is True

    status = d.conversation_persist_status()
    assert status["pending"] == 0
    assert status["failed"] == [(TENANT, ctx) for ctx in contexts[3:]]
    assert d.conversation_persist_failure(TENANT, contexts[2]) is None
    oldest_kept = d.conversation_persist_failure(TENANT, contexts[3])
    assert type(oldest_kept) is ConversationPersistFailed
    assert oldest_kept.context_id == contexts[3]


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_dispatch_bounded_when_history_load_hangs(monkeypatch):
    """A hung Mem0 must not stall the reply. The history load is time-bounded,
    so once the budget elapses the agent answers with no history rather than
    waiting on the backend. No infrastructure needed — a hanging store stub
    drives the real bound."""
    from cogniverse_runtime import agent_dispatcher as _ad

    monkeypatch.setattr(_ad, "CONVERSATION_LOAD_TIMEOUT_S", 0.2)

    class _HangingLoadStore:
        def get_history(self, context_id, max_turns=10):
            time.sleep(2.0)  # far past the 0.2s budget
            return [{"role": "user", "content": "should never be seen"}]

        def store_turn(self, *args, **kwargs):
            pass

    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
    )
    d._conversation_store_factory = lambda _tenant: _HangingLoadStore()
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    _no_wiki(d)
    seen: list = []
    _reply_with(d, {"q": "answered without waiting"}, seen)

    start = time.monotonic()
    result = await _dispatch(d, "q", f"chat{uuid.uuid4().hex[:10]}")
    elapsed = time.monotonic() - start

    assert result["message"] == "answered without waiting"
    assert seen[0] == []  # degraded to no history when the load timed out
    assert elapsed < 1.5  # bounded well under the 2s hang
    assert await d.drain_conversation_saves() is True
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_reply_does_not_wait_for_a_hung_save(monkeypatch, caplog):
    """A hung save never touches the reply: the answer returns with the save
    still pending, and the save then fails at its own budget with the failure
    readable on the dispatcher and the exception TYPE in the log. No
    infrastructure needed — a hanging store stub drives the real bound."""
    from cogniverse_runtime import agent_dispatcher as _ad

    monkeypatch.setattr(_ad, "CONVERSATION_SAVE_TIMEOUT_S", 0.2)

    class _HangingSaveStore:
        def get_history(self, context_id, max_turns=10):
            return []

        def store_turn(self, *args, **kwargs):
            time.sleep(2.0)  # far past the 0.2s budget

    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
    )
    d._conversation_store_factory = lambda _tenant: _HangingSaveStore()
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    _no_wiki(d)
    seen: list = []
    _reply_with(d, {"q": "answered anyway"}, seen)

    ctx = f"chat{uuid.uuid4().hex[:10]}"
    start = time.monotonic()
    result = await _dispatch(d, "q", ctx)
    elapsed = time.monotonic() - start

    assert result["message"] == "answered anyway"
    # The reply is back before the save has even started running.
    assert d.conversation_persist_status() == {"pending": 1, "failed": []}
    assert elapsed < 0.2

    with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.agent_dispatcher"):
        assert await d.drain_conversation_saves() is True

    assert d.conversation_persist_status() == {"pending": 0, "failed": [(TENANT, ctx)]}
    failure = d.conversation_persist_failure(TENANT, ctx)
    assert type(failure) is ConversationPersistFailed
    assert type(failure.__cause__) is TimeoutError
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.agent_dispatcher"
    ] == [
        f"Conversation turns for context {ctx} were NOT persisted: "
        f"TimeoutError: TimeoutError()"
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_history_round_trips_through_real_construction(
    shared_memory_vespa, shared_denseon, monkeypatch
):
    """With NO store factory injected, the dispatcher builds its own real
    ConversationStore + Mem0MemoryManager from the ConfigManager (the served
    runtime's path, via lazy_init_memory) and history still round-trips: the
    second dispatch sees the exact turns the first one persisted through the
    dispatcher-built store.

    lazy_init_memory reads llm_config.primary from the config store and the
    Vespa config port + LLM endpoint from the environment — provide them
    exactly as the deployment does (the llm_config seam is the same one
    memory_init's own unit tests stub), then let the real construction run
    against real Vespa + DenseOn.
    """
    from cogniverse_runtime import memory_init

    mm, cm = _build_manager_with_cm(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    monkeypatch.setattr(
        memory_init,
        "get_config",
        lambda tenant_id, config_manager: {
            "llm_config": {
                "primary": {"model": get_llm_model(), "api_base": get_llm_base_url()}
            }
        },
    )
    monkeypatch.setenv("VESPA_CONFIG_PORT", str(shared_memory_vespa["config_port"]))
    monkeypatch.setenv("LLM_ENDPOINT", get_llm_base_url())
    d = _dispatcher_with_real_construction(cm)
    # The non-seam manager resolves its own per-tenant schema
    # (agent_memories_{canonical.replace(':','_')}), so dispatch under the
    # tenant the shared memory fixture provisioned that schema for.
    provisioned_tenant = MEM0_ROUNDTRIP_TENANT_ID
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(
        d,
        {"what is colpali": "a late-interaction model", "how many dims": "128"},
        seen,
    )

    r1 = await _dispatch(d, "what is colpali", ctx, tenant=provisioned_tenant)
    assert r1["message"] == "a late-interaction model"
    assert seen[0] == []

    r2 = await _dispatch(d, "how many dims", ctx, tenant=provisioned_tenant)
    assert r2["message"] == "128"
    # seen[1] is what the second dispatch's own _build_conversation_store loaded
    # from real Mem0 — proving the non-seam construction round-trips.
    assert seen[1] == [
        {"role": "user", "content": "what is colpali"},
        {"role": "assistant", "content": "a late-interaction model"},
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reply_returns_before_persistence_and_next_turn_reads_it(
    shared_memory_vespa, shared_denseon
):
    """The reply does not wait for the Mem0 write, and the next turn on the
    same context still reads the previous turn.

    Both latencies are real timings taken in this test: the reply comes back
    before either turn has been written, and the turn that follows it sees the
    exact pair the first turn persisted because the load waits on that
    context's save chain.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    write_durations: list = []

    class _RecordingStore:
        def __init__(self, inner):
            self._inner = inner

        def get_history(self, context_id, max_turns=10):
            return self._inner.get_history(context_id, max_turns)

        def store_turn(self, context_id, role, content):
            started = time.monotonic()
            self._inner.store_turn(context_id, role, content)
            write_durations.append(time.monotonic() - started)

    d = _dispatcher_with_real_store(mm)
    d._conversation_store_factory = lambda tenant_id: _RecordingStore(
        ConversationStore(mm, tenant_id)
    )
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(
        d,
        {"what is colpali": "a late-interaction model", "how many dims": "128"},
        seen,
    )

    started = time.monotonic()
    r1 = await _dispatch(d, "what is colpali", ctx)
    reply_elapsed = time.monotonic() - started

    assert r1["message"] == "a late-interaction model"
    # Nothing has run on the loop since the save was scheduled, so the reply
    # provably returned before persistence: the save is queued, the store empty.
    assert d.conversation_persist_status() == {"pending": 1, "failed": []}
    assert ConversationStore(mm, TENANT).get_history(ctx) == []

    r2 = await _dispatch(d, "how many dims", ctx)
    assert r2["message"] == "128"
    assert seen[1] == [
        {"role": "user", "content": "what is colpali"},
        {"role": "assistant", "content": "a late-interaction model"},
    ]

    assert await d.drain_conversation_saves() is True
    assert ConversationStore(mm, TENANT).get_history(ctx) == [
        {"role": "user", "content": "what is colpali"},
        {"role": "assistant", "content": "a late-interaction model"},
        {"role": "user", "content": "how many dims"},
        {"role": "assistant", "content": "128"},
    ]

    save_elapsed = write_durations[0] + write_durations[1]
    print(
        f"REPLY_ELAPSED_S={reply_elapsed:.3f} "
        f"FIRST_TURN_SAVE_ELAPSED_S={save_elapsed:.3f} "
        f"WRITE_DURATIONS_S={[round(x, 3) for x in write_durations]}"
    )
    assert reply_elapsed < save_elapsed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_contexts_persist_independently_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """One context's stalled save never holds another context's.

    Context A's write is parked on a barrier the test controls; context B is
    dispatched behind it and its turns land in real Mem0 while A's are still
    absent, then A's land once the barrier lifts.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    store = ConversationStore(mm, TENANT)
    # Warm the embedder so the timing below measures a steady-state save.
    store.store_turn(f"warm{uuid.uuid4().hex[:8]}", "user", "warm the write path")

    ctx_a = f"chat{uuid.uuid4().hex[:10]}"
    ctx_b = f"chat{uuid.uuid4().hex[:10]}"
    barrier = threading.Event()

    class _BarrierOnA:
        def __init__(self, inner):
            self._inner = inner

        def get_history(self, context_id, max_turns=10):
            return self._inner.get_history(context_id, max_turns)

        def store_turn(self, context_id, role, content):
            if context_id == ctx_a:
                assert barrier.wait(timeout=60.0), "A's write was never released"
            self._inner.store_turn(context_id, role, content)

    d = _dispatcher_with_real_store(mm)
    d._conversation_store_factory = lambda tenant_id: _BarrierOnA(
        ConversationStore(mm, tenant_id)
    )
    seen: list = []
    _reply_with(d, {"in A": "answer A", "in B": "answer B"}, seen)

    await _dispatch(d, "in A", ctx_a)
    started = time.monotonic()
    await _dispatch(d, "in B", ctx_b)
    await d._conversation_save_chains[(TENANT, ctx_b)]
    b_elapsed = time.monotonic() - started

    assert store.get_history(ctx_b) == [
        {"role": "user", "content": "in B"},
        {"role": "assistant", "content": "answer B"},
    ]
    # A is still parked at the barrier while B is durable.
    assert store.get_history(ctx_a) == []
    assert d.conversation_persist_status() == {"pending": 1, "failed": []}
    print(f"CONTEXT_B_SAVE_ELAPSED_S={b_elapsed:.3f}")
    # A steady-state save measures ~0.1s against real Mem0 on this host; B
    # cannot have waited out A's 60s barrier.
    assert b_elapsed < 5.0

    barrier.set()
    assert await d.drain_conversation_saves() is True
    assert store.get_history(ctx_a) == [
        {"role": "user", "content": "in A"},
        {"role": "assistant", "content": "answer A"},
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paused_mem0_loses_the_turn_observably_real_mem0(
    shared_memory_vespa, shared_denseon, caplog
):
    """With Mem0's backend paused, the reply is unaffected and the lost turn is
    reported: the save burns its own budget off the reply path, records a typed
    failure naming TimeoutError, and the next load degrades to no history
    within the load budget."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    seen: list = []
    _reply_with(d, {"first": "reply one", "second": "reply two"}, seen)
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    container = shared_memory_vespa["container_name"]

    healthy_started = time.monotonic()
    r1 = await _dispatch(d, "first", ctx)
    healthy_elapsed = time.monotonic() - healthy_started
    assert await d.drain_conversation_saves() is True
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}

    paused_started = time.monotonic()
    r2 = await _dispatch(d, "second", ctx)
    paused_elapsed = time.monotonic() - paused_started
    # The save task exists but has not run yet (nothing has awaited since it
    # was scheduled), so pausing here makes its write the one that fails.
    subprocess.run(["docker", "pause", container], check=True, capture_output=True)
    try:
        assert r2 == {
            "message": "reply two",
            "entities": [],
            "answer": "reply two",
        }
        assert set(r2) == set(r1)
        print(
            f"HEALTHY_REPLY_S={healthy_elapsed:.3f} PAUSED_REPLY_S={paused_elapsed:.3f}"
        )
        assert paused_elapsed < 2.0

        with caplog.at_level(
            logging.WARNING, logger="cogniverse_runtime.agent_dispatcher"
        ):
            drain_started = time.monotonic()
            assert await d.drain_conversation_saves() is True
            drain_elapsed = time.monotonic() - drain_started

            # The reply cost 2s at most while the save spent its whole budget.
            assert drain_elapsed >= CONVERSATION_SAVE_TIMEOUT_S
            assert d.conversation_persist_status() == {
                "pending": 0,
                "failed": [(TENANT, ctx)],
            }
            failure = d.conversation_persist_failure(TENANT, ctx)
            assert type(failure) is ConversationPersistFailed
            assert type(failure.__cause__) is TimeoutError
            assert failure.context_id == ctx
            assert failure.tenant_id == TENANT

            load_started = time.monotonic()
            degraded = await d._load_conversation_history(TENANT, ctx)
            load_elapsed = time.monotonic() - load_started
            assert degraded == []
            assert (
                CONVERSATION_LOAD_TIMEOUT_S
                <= load_elapsed
                < (CONVERSATION_LOAD_TIMEOUT_S + 2.0)
            )

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.agent_dispatcher"
        ]
        assert messages == [
            f"Conversation turns for context {ctx} were NOT persisted: "
            f"TimeoutError: TimeoutError()",
            f"Conversation history unavailable for context {ctx}: "
            f"TimeoutError: TimeoutError()",
        ]
    finally:
        subprocess.run(
            ["docker", "unpause", container], check=True, capture_output=True
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_shutdown_drains_a_pending_turn_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """A turn answered moments before shutdown still lands: the runtime's own
    shutdown seam drains the dispatcher's pending saves."""
    from cogniverse_runtime.routers import agents as agents_router

    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    seen: list = []
    _reply_with(d, {"before shutdown": "answered before shutdown"}, seen)
    ctx = f"chat{uuid.uuid4().hex[:10]}"

    await _dispatch(d, "before shutdown", ctx)
    assert d.conversation_persist_status() == {"pending": 1, "failed": []}

    previous = agents_router._dispatcher
    agents_router._dispatcher = d
    try:
        assert await agents_router.drain_conversation_saves() is True
    finally:
        agents_router._dispatcher = previous

    assert ConversationStore(mm, TENANT).get_history(ctx) == [
        {"role": "user", "content": "before shutdown"},
        {"role": "assistant", "content": "answered before shutdown"},
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_budget_covers_measured_store_turn_cost(
    shared_memory_vespa, shared_denseon
):
    """The conversation budgets are sized from measured Mem0 cost.

    Six real turn writes and one real read are timed here; the shipped budgets
    must hold at least twice the worst measured turn pair and read. The budget
    the dispatcher ships is sized from the first save a fresh process makes
    (~7.2s, dominated by the first embedding), which this steady-state
    measurement floors from below.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    store = ConversationStore(mm, TENANT)
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    writes: list = []
    for index in range(6):
        role = "user" if index % 2 == 0 else "assistant"
        started = time.monotonic()
        store.store_turn(ctx, role, f"budget probe {index}")
        writes.append(time.monotonic() - started)
    read_started = time.monotonic()
    history = store.get_history(ctx)
    read_elapsed = time.monotonic() - read_started

    worst_turn_pair = max(writes[i] + writes[i + 1] for i in range(5))
    print(
        f"STORE_TURN_DURATIONS_S={[round(x, 3) for x in writes]} "
        f"WORST_TURN_PAIR_S={worst_turn_pair:.3f} "
        f"GET_HISTORY_S={read_elapsed:.3f}"
    )
    assert history == [
        {"role": "user", "content": "budget probe 0"},
        {"role": "assistant", "content": "budget probe 1"},
        {"role": "user", "content": "budget probe 2"},
        {"role": "assistant", "content": "budget probe 3"},
        {"role": "user", "content": "budget probe 4"},
        {"role": "assistant", "content": "budget probe 5"},
    ]
    assert worst_turn_pair * 2 <= CONVERSATION_SAVE_TIMEOUT_S
    assert read_elapsed * 2 <= CONVERSATION_LOAD_TIMEOUT_S
