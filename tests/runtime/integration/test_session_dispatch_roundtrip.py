"""Session-id propagation: real dispatcher → real agent → real Mem0 → real Vespa.

The other session tests cover individual segments of the wire:

  * ``test_admin_session_endpoints.py`` — admin DELETE / fan-out close
    against real Mem0 + Vespa.
  * ``test_session_drop_integration.py`` — manager-level drop_session
    against real Vespa.
  * ``test_dispatcher_session_propagation.py`` — ``_scoped_session``
    contract against a stub mixin agent.

What was missing: a single test that proves the FULL chain
``AgentDispatcher.dispatch(context={"session_id": ...})`` →
real memory-aware agent → ``MemoryAwareMixin.update_memory`` →
real ``Mem0.add_memory`` → row landing in real Vespa with
``metadata.session_id`` set.

This file closes that gap. Test agent registered in this module
follows the conventions ``_execute_generic_agent`` expects (Deps,
Input, Output, ``process(typed_input)``); on every dispatch it
writes one ``session_scratch`` memory. The assertion reads the
resulting row out of Vespa and confirms the dispatcher's
``_scoped_session`` did its job.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pydantic import Field

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test agent: minimal memory-aware A2A agent that, on dispatch, writes one
# session_scratch row. Mounted into ConfigLoader.AGENT_CLASSES + the
# AgentRegistry by the fixture below so the dispatcher's
# _execute_generic_agent path can resolve and instantiate it.
# ---------------------------------------------------------------------------


class SessionWriterInput(AgentInput):
    query: str = Field(..., description="Note text to persist as session scratch")
    tenant_id: str | None = Field(default=None)


class SessionWriterOutput(AgentOutput):
    """``ok=True`` when the mixin's update_memory accepted the write."""

    ok: bool = Field(default=False, description="True if update_memory succeeded")


class SessionWriterDeps(AgentDeps):
    pass


from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin  # noqa: E402


class SessionWriterAgent(
    MemoryAwareMixin,
    A2AAgent[SessionWriterInput, SessionWriterOutput, SessionWriterDeps],
):
    """On dispatch, write one session_scratch memory.

    The dispatcher wraps the call in ``_scoped_session(self, session_id)``
    which sets ``self._memory_session_id`` for the duration of the call.
    The mixin's ``update_memory`` then auto-stamps that id onto
    ``metadata["session_id"]`` so the schema-gated EPHEMERAL_SESSION
    write succeeds.
    """

    def __init__(self, deps: SessionWriterDeps) -> None:
        super().__init__(
            deps=deps,
            config=A2AAgentConfig(
                agent_name="session_writer_agent",
                agent_description=(
                    "Test fixture agent: writes one session_scratch row "
                    "per dispatch so the dispatcher's _scoped_session "
                    "round-trip can be asserted end-to-end."
                ),
                capabilities=["session_writer"],
                port=9999,
            ),
        )

    async def _process_impl(self, input: SessionWriterInput) -> SessionWriterOutput:
        if not self.is_memory_enabled() or self.memory_manager is None:
            raise RuntimeError(
                "SessionWriterAgent requires a memory-initialised mixin; "
                "the dispatcher's _init_agent_memory should have run before "
                "this point"
            )
        # Goes through MemoryAwareMixin.update_memory — that is the method
        # carrying the auto-stamp of metadata.session_id from the
        # dispatcher's _scoped_session. Calling self.memory_manager.add_memory
        # directly would BYPASS the mixin and the test would not exercise
        # the wire it claims to. The session_id is INTENTIONALLY absent
        # from the metadata dict here so the assertion is unambiguous: any
        # session_id on the resulting Vespa row had to come from the
        # dispatcher → mixin auto-stamp path.
        ok = self.update_memory(
            content=input.query,
            metadata={"kind": "session_scratch"},
            infer=False,
        )
        return SessionWriterOutput(ok=bool(ok))


_SESSION_WRITER_CLASS_PATH = (
    "tests.runtime.integration.test_session_dispatch_roundtrip:SessionWriterAgent"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_dispatcher(memory_manager, config_manager, schema_loader, monkeypatch):
    """Real AgentDispatcher with the test SessionWriterAgent registered.

    Uses the runtime ``memory_manager`` fixture so the agent's writes
    land in the real Vespa instance the rest of the runtime tests
    share. Wires ``knowledge_registry=build_default_registry()`` onto the
    singleton manager — the runtime fixture does not set one by default,
    so the schema gate (which reads from the registry) would otherwise
    short-circuit and the test would silently pass even when the
    dispatcher's session id never reaches metadata.
    """
    from cogniverse_core.memory.schema import build_default_registry

    monkeypatch.setitem(
        ConfigLoader.AGENT_CLASSES,
        "session_writer_agent",
        _SESSION_WRITER_CLASS_PATH,
    )
    # The mixin's initialize_memory sees the singleton already-initialised
    # (memory is not None) and skips re-init, so we have to wire the
    # registry directly onto the singleton here.
    monkeypatch.setattr(
        memory_manager, "_knowledge_registry", build_default_registry()
    )
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_with_session_id_stamps_metadata_in_vespa(
    session_dispatcher,
):
    """Full chain: dispatch → mixin → Mem0 → Vespa. The persisted row
    must carry metadata.session_id matching the dispatch context."""
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
        f"update_memory returned False; the schema gate or the mixin "
        f"rejected the write. Full result={result!r}"
    )

    meta = _read_metadata_by_content(mm, content)
    assert meta is not None, (
        f"row with content {content!r} not found in Vespa for tenant "
        f"{mm.tenant_id!r}; dispatcher → Mem0 chain broke before write"
    )
    assert meta.get("session_id") == session_id, (
        f"persisted metadata.session_id={meta.get('session_id')!r}; "
        f"expected {session_id!r}. The dispatcher's _scoped_session must "
        "have set the mixin's session id BEFORE the agent's _process_impl "
        "ran, AND the mixin's update_memory must have auto-stamped it "
        "onto metadata. Full row metadata: " + repr(meta)
    )
    assert meta.get("kind") == "session_scratch", meta


@pytest.mark.asyncio
async def test_dispatch_without_session_id_rejects_session_kind_write(
    session_dispatcher, caplog
):
    """Schema gate: dispatch with no session_id → agent's session_scratch
    write is rejected and nothing lands in Vespa.

    Production semantics: ``Mem0MemoryManager._enforce_schema_on_write``
    raises ``SchemaViolationError``; the mixin's ``update_memory``
    catches it, logs the rejection, and returns False; the agent
    returns ``ok=False``; the dispatcher returns status=success
    (the agent did not raise — it simply could not persist).

    The test asserts (a) the row is NOT in Vespa, (b) ``ok`` is False,
    and (c) the schema-gate log line names ``session_id`` so an
    operator can fix the missing context.
    """
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
        f"dispatch with no session_id should return ok=False because the "
        f"schema gate rejected the session_scratch write; got result={result!r}"
    )

    # Vespa must not have the row.
    meta = _read_metadata_by_content(mm, content)
    assert meta is None, (
        f"session_scratch row landed in Vespa even though no session_id "
        f"was provided — the schema gate did not fire. Found meta={meta!r}"
    )

    # The mixin logs the schema violation at ERROR. The message must
    # name session_id so the operator knows what is missing.
    schema_log = "\n".join(
        rec.message for rec in caplog.records if rec.levelname == "ERROR"
    )
    assert "session_id" in schema_log, (
        f"expected an ERROR log line naming session_id from the schema "
        f"gate; got logs:\n{schema_log!r}"
    )


@pytest.mark.asyncio
async def test_two_dispatches_with_different_sessions_get_different_metadata(
    session_dispatcher,
):
    """Long-lived agent instance must not bleed one session into the next.

    The dispatcher's _scoped_session clears the id on exit. This test
    proves it by dispatching twice with different session ids and
    asserting each persisted row carries its own.
    """
    dispatcher, mm = session_dispatcher
    content_a = "alpha note unique 4f7"
    content_b = "beta note unique 4f7"

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
