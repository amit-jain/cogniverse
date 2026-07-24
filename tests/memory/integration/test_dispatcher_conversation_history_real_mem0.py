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
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

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
    d._spawn_background = lambda coro: coro.close()
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
    d._spawn_background = lambda coro: coro.close()
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

    # And the real Mem0 store holds all four turns with exact content:
    # every query the gateway sent and every reply the agent returned.
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
    assert ConversationStore(mm, TENANT).get_history(ctx) == []


@pytest.mark.asyncio
async def test_concurrent_dispatches_persist_all_turns_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """Two concurrent messages for one context both persist their user +
    assistant turns to real Mem0 — no lost writes."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d = _dispatcher_with_real_store(mm)
    ctx = f"chat{uuid.uuid4().hex[:10]}"

    async def _fake(query, tenant_id, top_k, conversation_history=None, **kwargs):
        await asyncio.sleep(0)
        return {"message": f"reply to {query}", "entities": []}

    d._execute_search_task = _fake

    await asyncio.gather(
        _dispatch(d, "first", ctx),
        _dispatch(d, "second", ctx),
    )

    history = ConversationStore(mm, TENANT).get_history(ctx)
    contents = sorted(t["content"] for t in history)
    assert contents == [
        "first",
        "reply to first",
        "reply to second",
        "second",
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
    d._spawn_background = lambda coro: coro.close()

    gw = MagicMock()
    gw.capabilities = {"gateway"}
    se = MagicMock()
    se.capabilities = {"search"}
    d._registry.get_agent.side_effect = lambda name: {
        "gateway_agent": gw,
        "search_agent": se,
    }.get(name)

    # Force a 'simple' classification without building the real GatewayAgent.
    gwout = SimpleNamespace(
        complexity="simple",
        modality="video",
        generation_type="raw_results",
        routed_to="search_agent",
        confidence=0.9,
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

    # Reloaded from REAL Mem0: the stored assistant turn is the answer, not the
    # routing breadcrumb.
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
    d._spawn_background = lambda coro: coro.close()
    seen: list = []
    _reply_with(d, {"q": "answered anyway"}, seen)

    result = await _dispatch(d, "q", f"chat{uuid.uuid4().hex[:10]}")

    assert result["message"] == "answered anyway"
    assert seen[0] == []  # degraded to no history, still ran


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_dispatch_bounded_when_history_load_hangs(monkeypatch):
    """A hung Mem0 must not stall the reply. The history load is time-bounded,
    so once the budget elapses the agent answers with no history rather than
    waiting on the backend. No infrastructure needed — a hanging store stub
    drives the real bound."""
    from cogniverse_runtime import agent_dispatcher as _ad

    monkeypatch.setattr(_ad, "CONVERSATION_IO_TIMEOUT_S", 0.2)

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
    d._spawn_background = lambda coro: coro.close()
    seen: list = []
    _reply_with(d, {"q": "answered without waiting"}, seen)

    start = time.monotonic()
    result = await _dispatch(d, "q", f"chat{uuid.uuid4().hex[:10]}")
    elapsed = time.monotonic() - start

    assert result["message"] == "answered without waiting"
    assert seen[0] == []  # degraded to no history when the load timed out
    assert elapsed < 1.5  # bounded well under the 2s hang


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_dispatch_bounded_when_history_save_hangs(monkeypatch):
    """A hung save must not stall the reply either. The save is awaited (so the
    next turn reads its own writes) but time-bounded, so a stuck store_turn
    drops the turns and returns instead of holding the reply open."""
    from cogniverse_runtime import agent_dispatcher as _ad

    monkeypatch.setattr(_ad, "CONVERSATION_IO_TIMEOUT_S", 0.2)

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
    d._spawn_background = lambda coro: coro.close()
    seen: list = []
    _reply_with(d, {"q": "answered anyway"}, seen)

    start = time.monotonic()
    result = await _dispatch(d, "q", f"chat{uuid.uuid4().hex[:10]}")
    elapsed = time.monotonic() - start

    assert result["message"] == "answered anyway"
    assert elapsed < 1.5  # save bounded, reply not stalled by the hung write


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
    # (agent_memories_{canonical.replace(':','_')}); the fixture provisions
    # agent_memories_test_tenant, so dispatch under the tenant that maps there.
    provisioned_tenant = "test:tenant"
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
