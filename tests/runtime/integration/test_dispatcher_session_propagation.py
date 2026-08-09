"""Dispatcher propagates session_id onto memory-aware agents per request.

The runtime has to put ``session_id`` onto agent writes for the
EPHEMERAL_SESSION lifecycle to work. This test exercises
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


class _FailingSessionSetterAgent:
    def __init__(self, *, fail_values: list[Optional[str]]) -> None:
        self._memory_agent_name = "failing_session_agent"
        self.current_session_id: Optional[str] = "stale_session"
        self.fail_values = list(fail_values)
        self.calls: list[Optional[str]] = []

    def set_session_id(self, session_id: Optional[str]) -> None:
        self.calls.append(session_id)
        if self.fail_values and session_id == self.fail_values[0]:
            self.fail_values.pop(0)
            raise ValueError(f"cannot set session to {session_id!r}")
        self.current_session_id = session_id


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
        assert agent.get_session_id() is None

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
        assert agent.get_session_id() is None

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

        assert agent.get_session_id() is None, (
            "session_id must be cleared even when the agent raises — "
            "otherwise long-lived instances bleed sessions across requests"
        )

    def test_present_setter_failure_stops_request_with_context(self):
        agent = _FailingSessionSetterAgent(fail_values=["s_broken"])
        body_executed = False

        with pytest.raises(
            RuntimeError,
            match=("failing_session_agent.*s_broken.*cannot set session to 's_broken'"),
        ):
            with AgentDispatcher._scoped_session(agent, "s_broken"):
                body_executed = True

        assert body_executed is False
        assert agent.calls == ["s_broken"]

    def test_no_session_request_clears_stale_scope_before_body(self):
        agent = _FailingSessionSetterAgent(fail_values=[])
        observed_session_ids: list[Optional[str]] = []

        with AgentDispatcher._scoped_session(agent, None):
            observed_session_ids.append(agent.current_session_id)

        assert observed_session_ids == [None]
        assert agent.calls == [None, None]

    def test_cleanup_failure_blocks_following_request_until_scope_clears(self):
        agent = _FailingSessionSetterAgent(fail_values=[None])

        with pytest.raises(
            RuntimeError,
            match=(
                "failing_session_agent.*s_alpha.*cleanup.*cannot set session to None"
            ),
        ):
            with AgentDispatcher._scoped_session(agent, "s_alpha"):
                assert agent.current_session_id == "s_alpha"

        assert agent.current_session_id == "s_alpha"

        observed_session_ids: list[Optional[str]] = []
        with AgentDispatcher._scoped_session(agent, None):
            observed_session_ids.append(agent.current_session_id)

        assert observed_session_ids == [None]
        assert agent.calls == ["s_alpha", None, None, None]

    def test_body_error_propagates_when_cleanup_also_fails(self):
        agent = _FailingSessionSetterAgent(fail_values=[None])

        with pytest.raises(KeyError, match="body_error"):
            with AgentDispatcher._scoped_session(agent, "s_gamma"):
                raise KeyError("body_error")

        assert agent.calls == ["s_gamma", None]
        assert agent.current_session_id == "s_gamma"
