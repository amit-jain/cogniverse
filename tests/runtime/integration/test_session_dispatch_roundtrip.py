"""End-to-end: AgentDispatcher.dispatch(context={"session_id": ...}) →
real memory-aware agent → MemoryAwareMixin.update_memory → real Mem0
→ Vespa row with metadata.session_id stamped."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader

pytestmark = pytest.mark.integration


class SessionWriterInput(AgentInput):
    query: str
    tenant_id: str | None = None


class SessionWriterOutput(AgentOutput):
    ok: bool = False


class SessionWriterDeps(AgentDeps):
    pass


from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin  # noqa: E402


class SessionWriterAgent(
    MemoryAwareMixin,
    A2AAgent[SessionWriterInput, SessionWriterOutput, SessionWriterDeps],
):
    def __init__(self, deps: SessionWriterDeps) -> None:
        super().__init__(
            deps=deps,
            config=A2AAgentConfig(
                agent_name="session_writer_agent",
                agent_description="test fixture; writes one session_scratch row per dispatch",
                capabilities=["session_writer"],
                port=9999,
            ),
        )

    async def _process_impl(self, input: SessionWriterInput) -> SessionWriterOutput:
        if not self.is_memory_enabled() or self.memory_manager is None:
            raise RuntimeError("memory not initialised by dispatcher")
        # Use mixin update_memory (auto-stamps session_id), not manager.add_memory.
        # Omit session_id from metadata so the assertion proves it came from
        # the dispatcher → mixin path, not the agent itself.
        ok = self.update_memory(
            content=input.query,
            metadata={"kind": "session_scratch"},
            infer=False,
        )
        return SessionWriterOutput(ok=bool(ok))


_SESSION_WRITER_CLASS_PATH = (
    "tests.runtime.integration.test_session_dispatch_roundtrip:SessionWriterAgent"
)


@pytest.fixture
def session_dispatcher(memory_manager, config_manager, schema_loader, monkeypatch):
    from cogniverse_core.memory.schema import build_default_registry

    monkeypatch.setitem(
        ConfigLoader.AGENT_CLASSES,
        "session_writer_agent",
        _SESSION_WRITER_CLASS_PATH,
    )
    # Singleton is already initialised by the fixture without a registry;
    # wire it directly so the schema gate fires.
    monkeypatch.setattr(memory_manager, "_knowledge_registry", build_default_registry())
    registry = AgentRegistry(
        tenant_id=memory_manager.tenant_id, config_manager=config_manager
    )
    registry.register_agent(
        AgentEndpoint(
            name="session_writer_agent",
            url="http://localhost:9999",
            capabilities=["session_writer"],
        )
    )
    dispatcher = AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
        sandbox_manager=MagicMock(get_policy=lambda _name: None),
    )
    yield dispatcher, memory_manager
    try:
        memory_manager.clear_agent_memory(
            memory_manager.tenant_id, "session_writer_agent"
        )
    except Exception:
        pass


def _read_metadata_by_content(
    mm: Mem0MemoryManager, content: str
) -> Dict[str, Any] | None:
    """Pull the persisted row from real Vespa by matching its memory text.

    update_memory returns bool not the id, so the test identifies its own
    write by the unique query string it sent.
    """
    import json

    rows = mm.get_all_memories(mm.tenant_id, "session_writer_agent")
    for row in rows:
        text = row.get("memory") or row.get("content") or ""
        if text != content:
            continue
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return meta
    return None


@pytest.mark.asyncio
async def test_dispatch_with_session_id_stamps_metadata_in_vespa(
    session_dispatcher,
):
    dispatcher, mm = session_dispatcher
    session_id = "s_dispatch_alpha"
    content = "dispatched scratch note alpha"

    result = await dispatcher.dispatch(
        agent_name="session_writer_agent",
        query=content,
        context={"tenant_id": mm.tenant_id, "session_id": session_id},
    )

    assert result["status"] == "success", result
    assert result.get("ok") is True, (
        f"update_memory returned False; schema gate or mixin rejected "
        f"the write. result={result!r}"
    )

    meta = _read_metadata_by_content(mm, content)
    assert meta is not None, (
        f"row with content {content!r} not found in Vespa for tenant "
        f"{mm.tenant_id!r}; dispatcher → Mem0 chain broke before write"
    )
    assert meta.get("session_id") == session_id, (
        f"persisted metadata.session_id={meta.get('session_id')!r}; "
        f"expected {session_id!r}. _scoped_session must set the mixin's "
        f"session id before _process_impl, AND update_memory must "
        f"auto-stamp it. meta={meta!r}"
    )
    assert meta.get("kind") == "session_scratch", meta


@pytest.mark.asyncio
async def test_dispatch_without_session_id_rejects_session_kind_write(
    session_dispatcher, caplog
):
    import logging

    dispatcher, mm = session_dispatcher
    content = "rejected scratch note xyz"

    with caplog.at_level(logging.ERROR):
        result = await dispatcher.dispatch(
            agent_name="session_writer_agent",
            query=content,
            context={"tenant_id": mm.tenant_id},
        )

    assert result.get("ok") is False, (
        f"dispatch with no session_id must return ok=False (schema gate "
        f"rejects session_scratch); result={result!r}"
    )
    meta = _read_metadata_by_content(mm, content)
    assert meta is None, (
        f"session_scratch row landed in Vespa with no session_id — "
        f"schema gate did not fire. meta={meta!r}"
    )

    schema_log = "\n".join(
        rec.message for rec in caplog.records if rec.levelname == "ERROR"
    )
    assert "session_id" in schema_log, (
        f"expected an ERROR log naming session_id from the schema gate; "
        f"got:\n{schema_log!r}"
    )


@pytest.mark.asyncio
async def test_two_dispatches_with_different_sessions_get_different_metadata(
    session_dispatcher,
):
    dispatcher, mm = session_dispatcher
    content_a, content_b = "alpha note unique 4f7", "beta note unique 4f7"

    res_a = await dispatcher.dispatch(
        agent_name="session_writer_agent",
        query=content_a,
        context={"tenant_id": mm.tenant_id, "session_id": "s_alpha"},
    )
    res_b = await dispatcher.dispatch(
        agent_name="session_writer_agent",
        query=content_b,
        context={"tenant_id": mm.tenant_id, "session_id": "s_beta"},
    )
    assert res_a["ok"] is True and res_b["ok"] is True, (res_a, res_b)

    meta_a = _read_metadata_by_content(mm, content_a)
    meta_b = _read_metadata_by_content(mm, content_b)
    assert meta_a is not None and meta_b is not None
    assert meta_a["session_id"] == "s_alpha", meta_a
    assert meta_b["session_id"] == "s_beta", meta_b
