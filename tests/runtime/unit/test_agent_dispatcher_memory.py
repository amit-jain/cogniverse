"""Unit tests for AgentDispatcher memory auto-initialization (audit fix #14).

Verifies that the dispatcher's ``_init_agent_memory`` helper:
1. Calls ``initialize_memory()`` for any agent inheriting MemoryAwareMixin
2. Silently no-ops for agents that don't inherit the mixin
3. Survives initialize_memory failures (logs but doesn't raise)
4. Sets the tenant for context lookup

Before this fix the dispatcher passed ``tenant_id`` through every execution
path but never called ``initialize_memory()``, so even agents that inherited
MemoryAwareMixin had ``is_memory_enabled() == False`` and silently skipped
strategy/memory injection.
"""

from unittest.mock import MagicMock

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_runtime.agent_dispatcher import AgentDispatcher


@pytest.fixture
def mock_dispatcher():
    """Build a dispatcher with stubbed dependencies — enough to call
    ``_init_agent_memory()`` without spinning up Vespa or Mem0."""
    sys_cfg = MagicMock()
    sys_cfg.backend_url = "http://localhost"
    sys_cfg.backend_port = 8080
    sys_cfg.llm_model = "qwen3:4b"
    sys_cfg.embedding_model = "nomic-embed-text"
    sys_cfg.base_url = "http://localhost:11434"

    config_manager = MagicMock()
    config_manager.get_system_config.return_value = sys_cfg

    schema_loader = MagicMock()
    registry = MagicMock()

    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


class _MemoryAgentStub(MemoryAwareMixin):
    """A trivial subclass of MemoryAwareMixin that records calls."""

    def __init__(self):
        super().__init__()
        self.initialize_calls = []
        self.set_tenant_calls = []

    def initialize_memory(self, **kwargs):
        self.initialize_calls.append(kwargs)
        self._memory_initialized = True
        self._memory_agent_name = kwargs.get("agent_name")
        self._memory_tenant_id = kwargs.get("tenant_id")
        self.memory_manager = MagicMock()
        return True

    def set_tenant_for_context(self, tenant_id):
        self.set_tenant_calls.append(tenant_id)
        self._memory_tenant_id = tenant_id


class _NonMemoryAgent:
    """An agent class that does NOT inherit MemoryAwareMixin."""

    pass


@pytest.mark.unit
@pytest.mark.ci_fast
class TestInitAgentMemory:
    def test_calls_initialize_memory_for_mixin_subclass(
        self, mock_dispatcher
    ):
        """The helper must call ``initialize_memory`` with the right
        agent_name, tenant_id, config_manager, and schema_loader."""
        agent = _MemoryAgentStub()

        mock_dispatcher._init_agent_memory(agent, "test_agent", "acme")

        assert len(agent.initialize_calls) == 1
        call = agent.initialize_calls[0]
        assert call["agent_name"] == "test_agent"
        assert call["tenant_id"] == "acme"
        assert call["backend_host"] == "http://localhost"
        assert call["backend_port"] == 8080
        assert call["config_manager"] is mock_dispatcher._config_manager
        assert call["schema_loader"] is mock_dispatcher._schema_loader

    def test_sets_tenant_for_context(self, mock_dispatcher):
        """The helper must always call ``set_tenant_for_context`` so
        instructions are loaded for the right tenant even if full memory
        init is delayed or fails."""
        agent = _MemoryAgentStub()

        mock_dispatcher._init_agent_memory(agent, "test_agent", "acme")

        assert agent.set_tenant_calls == ["acme"]

    def test_no_op_for_non_mixin_agent(self, mock_dispatcher):
        """Agents that don't inherit MemoryAwareMixin (e.g., ImageSearchAgent)
        must not raise when passed to the helper."""
        agent = _NonMemoryAgent()

        # Should not raise.
        mock_dispatcher._init_agent_memory(agent, "image_search_agent", "acme")

    def test_survives_initialize_memory_exception(self, mock_dispatcher):
        """If initialize_memory raises (e.g., Vespa unreachable), the helper
        must log and continue — memory is best-effort enrichment, not a
        hard dependency."""

        class _FailingAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()
                self.set_tenant_calls = []

            def initialize_memory(self, **kwargs):
                raise RuntimeError("vespa unreachable")

            def set_tenant_for_context(self, tenant_id):
                self.set_tenant_calls.append(tenant_id)

        agent = _FailingAgent()

        # Should NOT raise.
        mock_dispatcher._init_agent_memory(agent, "test_agent", "acme")

        # set_tenant_for_context should still have been called.
        assert agent.set_tenant_calls == ["acme"]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestExecuteCodingTaskWiresMemory:
    """End-to-end pin: verify _execute_coding_task calls _init_agent_memory.
    A regression where someone removes the call would silently turn off
    memory for the coding agent."""

    def test_execute_coding_task_calls_init_agent_memory(self):
        import inspect

        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        source = inspect.getsource(AgentDispatcher._execute_coding_task)
        assert "_init_agent_memory" in source, (
            "_execute_coding_task must call self._init_agent_memory() so the "
            "coding agent receives learned strategies and tenant memories. "
            "Audit fix #14 — see "
            "docs/superpowers/audits/2026-04-07-orphan-and-wiring-audit.md"
        )

    def test_execute_summarization_task_calls_init_agent_memory(self):
        import inspect

        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        source = inspect.getsource(
            AgentDispatcher._execute_summarization_task
        )
        assert "_init_agent_memory" in source, (
            "_execute_summarization_task must call _init_agent_memory()"
        )

    def test_execute_detailed_report_task_calls_init_agent_memory(self):
        import inspect

        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        source = inspect.getsource(
            AgentDispatcher._execute_detailed_report_task
        )
        assert "_init_agent_memory" in source, (
            "_execute_detailed_report_task must call _init_agent_memory()"
        )

    def test_routing_memory_default_is_true(self):
        """Audit fix #14 also flips the routing enable_memory default from
        False to True. Pin this so a careless commit doesn't revert it."""
        import inspect

        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        source = inspect.getsource(AgentDispatcher._execute_routing_task)
        # The default after the fix is True. Look for the new pattern.
        assert 'enable_memory", True' in source, (
            "routing_agent enable_memory default must be True (audit fix #14). "
            "Without this, the routing agent silently runs without memory."
        )
