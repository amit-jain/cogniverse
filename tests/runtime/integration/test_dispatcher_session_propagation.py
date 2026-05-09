"""Dispatcher propagates session_id onto memory-aware agents per request.

Wire coverage for the EPHEMERAL_SESSION lifecycle's middle hop. The
audit's call-out: it's not enough to ship the schema gate + drop_session
+ admin endpoint — the runtime has to actually *put* session_id onto
agent writes for any of it to fire. This test exercises
``AgentDispatcher._scoped_session`` directly with a stub mixin-shaped
agent and asserts:

  * the dispatcher calls ``set_session_id(session_id)`` before
    delegating to the agent,
  * the agent's writes inside the scope auto-stamp metadata.session_id,
  * the dispatcher clears the session id after the call so a long-lived
    agent instance does not bleed one request's session into the next,
  * agents that don't inherit the mixin no-op cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = pytest.mark.integration


class _StubMemoryManager:
    """Minimal Mem0MemoryManager stand-in that records writes in order."""

    def __init__(self) -> None:
        self.memory = MagicMock()  # truthy
        self.calls: list[Dict[str, Any]] = []

    def add_memory(
        self,
        *,
        content: str,
        tenant_id: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> str:
        self.calls.append({"content": content, "metadata": metadata})
        return f"id_{len(self.calls)}"


class _StubAgent(MemoryAwareMixin):
    """Minimal mixin-shaped agent — no AgentBase, no DSPy, no deps."""

    def __init__(self, manager: _StubMemoryManager) -> None:
        super().__init__()
        self.memory_manager = manager
        self._memory_agent_name = "stub_agent"
        self._memory_tenant_id = "session_propagation_tenant"
        self._memory_initialized = True


class _NonMemoryAgent:
    """Agent that does not inherit MemoryAwareMixin (no set_session_id)."""

    def __init__(self) -> None:
        self.processed = []

    def process(self, payload: Any) -> Any:
        self.processed.append(payload)
        return payload


class TestScopedSessionStampsThenClears:
    def test_writes_inside_scope_carry_session_id(self):
        mgr = _StubMemoryManager()
        agent = _StubAgent(mgr)
        session_id = "s_test_alpha"

        with AgentDispatcher._scoped_session(agent, session_id):
            agent.update_memory(
                "transient note",
                metadata={"kind": "session_scratch"},
                infer=False,
            )

        assert len(mgr.calls) == 1
        assert mgr.calls[0]["metadata"] == {
            "kind": "session_scratch",
            "session_id": session_id,
        }
        # Cleared on exit so the next write doesn't inherit the id.
        assert agent._memory_session_id is None

    def test_writes_outside_scope_have_no_session_id(self):
        mgr = _StubMemoryManager()
        agent = _StubAgent(mgr)

        with AgentDispatcher._scoped_session(agent, "s_inside"):
            agent.update_memory("inside", metadata={"kind": "entity_fact"}, infer=False)
        agent.update_memory("outside", metadata={"kind": "entity_fact"}, infer=False)

        assert mgr.calls[0]["metadata"]["session_id"] == "s_inside"
        assert "session_id" not in mgr.calls[1]["metadata"]

    def test_caller_session_id_wins_over_dispatcher(self):
        mgr = _StubMemoryManager()
        agent = _StubAgent(mgr)

        with AgentDispatcher._scoped_session(agent, "s_dispatcher"):
            agent.update_memory(
                "explicit",
                metadata={
                    "kind": "session_scratch",
                    "session_id": "s_caller_explicit",
                },
                infer=False,
            )

        assert mgr.calls[0]["metadata"]["session_id"] == "s_caller_explicit"

    def test_no_session_id_is_a_no_op(self):
        mgr = _StubMemoryManager()
        agent = _StubAgent(mgr)

        with AgentDispatcher._scoped_session(agent, None):
            agent.update_memory(
                "no-session", metadata={"kind": "entity_fact"}, infer=False
            )

        assert "session_id" not in (mgr.calls[0]["metadata"] or {})
        assert agent._memory_session_id is None

    def test_non_mixin_agent_silently_no_ops(self):
        agent = _NonMemoryAgent()
        # Must not raise — the dispatcher routes plenty of agents that
        # don't need memory at all (ImageSearchAgent, etc.).
        with AgentDispatcher._scoped_session(agent, "s_alpha"):
            agent.process("hello")
        assert agent.processed == ["hello"]

    def test_session_cleared_even_when_agent_raises(self):
        mgr = _StubMemoryManager()
        agent = _StubAgent(mgr)

        class _Boom(RuntimeError):
            pass

        with pytest.raises(_Boom):
            with AgentDispatcher._scoped_session(agent, "s_will_clear"):
                raise _Boom("agent crashed mid-request")

        assert agent._memory_session_id is None, (
            "session_id must be cleared even when the agent raises — "
            "otherwise long-lived instances bleed sessions across requests"
        )
