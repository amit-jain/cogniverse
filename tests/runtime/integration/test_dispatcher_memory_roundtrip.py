"""Round-trip integration test for AgentDispatcher memory auto-init.

Verifies the full chain: AgentDispatcher._init_agent_memory → real
MemoryAwareMixin.initialize_memory → real Mem0 → real Vespa. The agent
should end up able to write a memory and read it back.

Audit fix #14 — before this fix, the dispatcher passed tenant_id through
every execution path but never called initialize_memory(), so agents had
memory capability but it was dormant in production. This test exercises
the production code path through real services.
"""

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.fixture
def real_dispatcher(config_manager, schema_loader):
    """Construct a real AgentDispatcher wired to the test Vespa via DI."""
    return AgentDispatcher(
        agent_registry=AgentRegistry(config_manager=config_manager),
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


class _RealMemoryAgent(MemoryAwareMixin):
    """Minimal agent that inherits MemoryAwareMixin — no other base classes.

    The dispatcher's _init_agent_memory only requires the agent to inherit
    from MemoryAwareMixin and accept the standard initialize_memory args.
    """

    def __init__(self):
        super().__init__()


@pytest.mark.integration
class TestDispatcherMemoryRoundTrip:
    def test_init_agent_memory_enables_real_memory(
        self, real_dispatcher, vespa_instance, memory_manager
    ):
        """The dispatcher's auto-init must result in is_memory_enabled() == True
        and the agent must be able to talk to the real Mem0 backend through
        the manager. This is the round-trip the previous unit test couldn't
        cover because it mocked everything."""
        agent = _RealMemoryAgent()

        # Before init: memory is off.
        assert agent.is_memory_enabled() is False

        # Production wiring: dispatcher initializes memory for this agent.
        real_dispatcher._init_agent_memory(
            agent, agent_name="roundtrip_test_agent", tenant_id="test:unit"
        )

        # After init: memory must be enabled with a real manager.
        assert agent.is_memory_enabled() is True, (
            "_init_agent_memory must set up a real Mem0 backend, not silently no-op"
        )
        assert agent.memory_manager is not None
        assert agent._memory_tenant_id == "test:unit"
        assert agent._memory_agent_name == "roundtrip_test_agent"

    def test_dispatcher_initialized_agent_can_write_and_read_memory(
        self, real_dispatcher, vespa_instance, memory_manager
    ):
        """End-to-end: after dispatcher init, the agent must successfully
        store a memory in real Vespa and retrieve it via Mem0 search.

        This is the test that would have caught the original audit bug:
        if the dispatcher silently skipped memory init (or initialized
        with the wrong port), this round-trip would fail because the
        write would go to the wrong place or the search would return
        nothing.
        """
        agent = _RealMemoryAgent()
        real_dispatcher._init_agent_memory(
            agent,
            agent_name="dispatcher_rt_agent",
            tenant_id="test:unit",
        )

        memory_id = agent.update_memory(
            "I prefer responses that include code examples"
        )
        assert memory_id is True, (
            "update_memory must return True after a successful Mem0 write"
        )

        wait_for_vespa_indexing(delay=5, description="memory indexing")

        context = agent.get_relevant_context(
            "code examples preference", top_k=5
        )
        assert context is not None, (
            "Newly stored memory should be findable via real Vespa search "
            "after indexing — if context is None the dispatcher's memory "
            "init didn't actually wire the agent to the test backend"
        )
        assert "code" in context.lower() or "example" in context.lower()

        agent.clear_memory()

    def test_non_mixin_agent_is_no_op(self, real_dispatcher):
        """Agents that don't inherit MemoryAwareMixin (e.g., ImageSearchAgent)
        must NOT raise when passed to the dispatcher's init helper. The helper
        is best-effort enrichment, not a hard dependency."""

        class _NonMemoryAgent:
            pass

        agent = _NonMemoryAgent()
        # Must not raise.
        real_dispatcher._init_agent_memory(
            agent, agent_name="non_memory", tenant_id="test:unit"
        )
