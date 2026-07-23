"""Unit tests for AgentDispatcher memory auto-initialization.

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

from unittest.mock import AsyncMock, MagicMock

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
    sys_cfg.embedding_model = "lightonai/DenseOn"
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
    def test_calls_initialize_memory_for_mixin_subclass(self, mock_dispatcher):
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

    def test_honors_deps_auto_create_memory_schema_false(self, mock_dispatcher):
        """An agent whose deps set auto_create_memory_schema=False (SearchAgent)
        must init memory WITHOUT auto-deploying the schema — the flag was
        dropped and the mixin always defaulted to True."""
        from types import SimpleNamespace

        agent = _MemoryAgentStub()
        agent.deps = SimpleNamespace(auto_create_memory_schema=False)

        mock_dispatcher._init_agent_memory(agent, "search_agent", "acme")

        assert agent.initialize_calls[0]["auto_create_schema"] is False

    def test_defaults_auto_create_schema_true_without_deps_field(self, mock_dispatcher):
        """Agents whose deps carry no such field keep the auto-deploy default."""
        agent = _MemoryAgentStub()  # no deps attribute

        mock_dispatcher._init_agent_memory(agent, "test_agent", "acme")

        assert agent.initialize_calls[0]["auto_create_schema"] is True

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


class _StopAfterInit(Exception):
    """Raised by the _init_agent_memory spy to abort before heavy agent work."""


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDispatchPathsWireMemory:
    """Each agent-execution path must call _init_agent_memory at runtime.
    Verified by EXECUTING the path with the call spied (it raises to abort
    before the agent actually runs), not by grepping the method source."""

    async def _capture_init_call(self, dispatcher, coro):
        seen = []

        def _spy(agent, name, tenant):
            seen.append((name, tenant))
            raise _StopAfterInit

        dispatcher._init_agent_memory = _spy
        with pytest.raises(_StopAfterInit):
            await coro
        return seen

    @pytest.mark.asyncio
    async def test_coding_task_initializes_agent_memory(
        self, mock_dispatcher, monkeypatch
    ):
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config", lambda **k: MagicMock()
        )
        # coding_lm is built via create_routed_lm, which resolves create_dspy_lm
        # through semantic_router's own binding — patch it there so no real
        # dspy.LM is constructed from the stubbed config.
        monkeypatch.setattr(
            "cogniverse_foundation.config.semantic_router.create_dspy_lm",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_agents.coding_agent.CodingAgent", lambda *a, **k: MagicMock()
        )
        seen = await self._capture_init_call(
            mock_dispatcher, mock_dispatcher._execute_coding_task("q", "acme:prod")
        )
        assert seen == [("coding_agent", "acme:prod")]

    @pytest.mark.asyncio
    async def test_summarization_task_initializes_agent_memory(
        self, mock_dispatcher, monkeypatch
    ):
        monkeypatch.setattr(
            "cogniverse_agents.summarizer_agent.SummarizerAgent",
            lambda *a, **k: MagicMock(),
        )
        seen = await self._capture_init_call(
            mock_dispatcher,
            mock_dispatcher._execute_summarization_task("q", "acme:prod", {}),
        )
        assert seen == [("summarizer_agent", "acme:prod")]

    @pytest.mark.asyncio
    async def test_detailed_report_task_initializes_agent_memory(
        self, mock_dispatcher, monkeypatch
    ):
        monkeypatch.setattr(
            "cogniverse_agents.detailed_report_agent.DetailedReportAgent",
            lambda *a, **k: MagicMock(),
        )
        seen = await self._capture_init_call(
            mock_dispatcher,
            mock_dispatcher._execute_detailed_report_task("q", "acme:prod"),
        )
        assert seen == [("detailed_report_agent", "acme:prod")]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestGatewayAgentCaching:
    """GatewayAgent must be cached on the dispatcher (GLiNER reload is
    expensive). Verified by constructing twice and counting, not by source."""

    @pytest.mark.asyncio
    async def test_gateway_agent_constructed_once_across_requests(
        self, mock_dispatcher, monkeypatch
    ):
        count = {"n": 0}

        class _FakeGateway:
            def __init__(self, *a, **k):
                count["n"] += 1
                self.telemetry_manager = None

            def _load_artifact(self):
                pass

            async def _process_impl(self, input_data):
                return MagicMock(complexity="simple", recommended_agent="search_agent")

        monkeypatch.setattr(
            "cogniverse_agents.gateway_agent.GatewayAgent", _FakeGateway
        )
        monkeypatch.setattr(
            "cogniverse_agents.gateway_agent.GatewayDeps", lambda *a, **k: MagicMock()
        )
        monkeypatch.setattr(mock_dispatcher, "_resolve_gliner_url", lambda: None)
        monkeypatch.setattr(mock_dispatcher, "_get_rail_chains", lambda t: None)
        # Gateway's "simple" path dispatches downstream — stub it so the test
        # stays focused on the cache, not on a real downstream agent.
        monkeypatch.setattr(
            mock_dispatcher,
            "_execute_downstream_agent",
            AsyncMock(return_value={"status": "success"}),
        )

        await mock_dispatcher._execute_gateway_task("first", {}, "acme:prod")
        await mock_dispatcher._execute_gateway_task("second", {}, "acme:prod")

        assert count["n"] == 1
        entry = mock_dispatcher._gateway_agents.get("acme:prod")
        assert entry is not None
        assert isinstance(entry.agent, _FakeGateway)
