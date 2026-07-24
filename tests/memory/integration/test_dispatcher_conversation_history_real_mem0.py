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
import uuid
from pathlib import Path

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


def _build_manager(*, shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
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
    return mm


def _dispatcher_with_real_store(mm) -> AgentDispatcher:
    """Real dispatcher whose conversation store is a real ConversationStore
    on the real Mem0 manager — no in-memory double."""
    from unittest.mock import MagicMock

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


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_dispatch_degrades_when_memory_unavailable():
    """History is enrichment: when the store is unavailable the agent still
    answers with no history — the reply is never lost to a memory outage.
    The store build raises here (as get_all_memories does on a real Mem0
    outage), so this exercises the dispatcher's real degrade path with no
    infrastructure needed."""
    from unittest.mock import MagicMock

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
