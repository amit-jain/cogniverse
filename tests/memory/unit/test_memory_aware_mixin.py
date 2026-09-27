"""
Unit tests for MemoryAwareMixin
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _memory_config_manager():
    """The injected ConfigManager the mixin reads, over an in-memory store."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


class MockAgentWithMemory(MemoryAwareMixin):
    """Mock agent class using memory mixin for testing"""

    def __init__(self):
        super().__init__()
        self.bind_config_manager(_memory_config_manager())


# Default memory init kwargs for tests (all required params)
MEMORY_INIT_DEFAULTS = {
    "backend_host": "http://localhost",
    "backend_port": 8080,
    "llm_model": "test-llm",
    "embedding_model": "lightonai/DenseOn",
    "llm_base_url": "http://localhost:11434/v1",
    "embedder_base_url": "http://localhost:8000",
    "config_manager": MagicMock(),
    "schema_loader": MagicMock(),
}


class TestMemoryAwareMixin:
    """Test MemoryAwareMixin"""

    @pytest.fixture
    def agent(self):
        """Create test agent"""
        return MockAgentWithMemory()

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager"""
        manager = MagicMock()
        manager.config = MagicMock()
        manager.config.enabled = True
        manager.config.retrieval_top_k = 5
        return manager

    def test_initialization(self, agent):
        """Test mixin initialization"""
        assert agent.memory_manager is None
        assert agent._memory_agent_name is None
        assert agent._memory_tenant_id is None
        assert agent._memory_initialized is False

    def test_is_memory_enabled_false_by_default(self, agent):
        """Test memory is disabled by default"""
        assert agent.is_memory_enabled() is False

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_initialize_memory_success(self, mock_manager_class, agent):
        """Test successful memory initialization"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()  # Mem0 uses .memory attribute
        mock_manager_class.return_value = mock_manager

        # Initialize memory
        success = agent.initialize_memory(
            "test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS
        )

        assert success is True
        assert agent._memory_agent_name == "test_agent"
        assert agent._memory_tenant_id == "test_tenant"
        assert agent._memory_initialized is True

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_initialize_memory_with_vespa_config(self, mock_manager_class, agent):
        """Test memory initialization with Vespa configuration"""
        from cogniverse_core.memory.schema import KnowledgeRegistry

        mock_manager = MagicMock()
        mock_manager.memory = None  # Not initialized yet
        mock_manager_class.return_value = mock_manager

        mock_cm = MagicMock()
        mock_sl = MagicMock()
        success = agent.initialize_memory(
            "test_agent",
            "test_tenant",
            backend_host="backend.local",
            backend_port=9090,
            llm_model="test-llm",
            embedding_model="lightonai/DenseOn",
            llm_base_url="http://localhost:11434/v1",
            embedder_base_url="http://denseon.local:8000",
            config_manager=mock_cm,
            schema_loader=mock_sl,
        )

        assert success is True
        mock_manager.initialize.assert_called_once()
        kwargs = mock_manager.initialize.call_args.kwargs
        # MemoryAwareMixin wires knowledge_registry into initialize so
        # the schema layer (provenance + trust + contradiction) actually
        # runs in production. The instance type is what matters; the
        # registry contents are validated elsewhere.
        registry = kwargs.pop("knowledge_registry", None)
        assert isinstance(registry, KnowledgeRegistry), (
            "MemoryAwareMixin.initialize_memory must pass a KnowledgeRegistry "
            f"to Mem0MemoryManager.initialize; got {type(registry)!r}"
        )
        assert kwargs == {
            "backend_host": "backend.local",
            "backend_port": 9090,
            "llm_model": "test-llm",
            "embedding_model": "lightonai/DenseOn",
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key": None,
            "embedder_base_url": "http://denseon.local:8000",
            "config_manager": mock_cm,
            "schema_loader": mock_sl,
            "backend_config_port": None,
            "base_schema_name": "agent_memories",
            "auto_create_schema": True,
        }

    def test_get_relevant_context_without_initialization(self, agent):
        """Test getting context without initialization"""
        context = agent.get_relevant_context("test query")
        assert context is None

    def test_update_memory_without_initialization(self, agent):
        """Test updating memory without initialization"""
        success = agent.update_memory("test content")
        assert success is False

    def test_get_memory_state_without_initialization(self, agent):
        """Test getting memory state without initialization"""
        state = agent.get_memory_state()
        assert state is None

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_get_relevant_context(self, mock_manager_class, agent):
        """Test getting relevant context"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.search_memory.return_value = [
            {"memory": "Context 1"},
            {"memory": "Context 2"},
        ]
        mock_manager_class.return_value = mock_manager

        # Initialize and get context
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)
        context = agent.get_relevant_context("test query")

        assert context is not None
        assert "Context 1" in context
        assert "Context 2" in context
        mock_manager.search_memory.assert_called_once()

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_get_relevant_context_raises_on_search_failure(
        self, mock_manager_class, agent
    ):
        """Tenant-side memory outages must surface instead of returning None."""
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.search_memory.side_effect = RuntimeError("memory backend down")
        mock_manager_class.return_value = mock_manager

        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)

        with pytest.raises(RuntimeError, match="memory backend down"):
            agent.get_relevant_context("test query")

        mock_manager.search_memory.assert_called_once()

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_get_relevant_context_raises_on_federated_trunk_failure(
        self, mock_manager_class, agent
    ):
        """Org-trunk failures must surface when federation is enabled."""

        class _TenantManager:
            def __init__(self):
                self.memory = object()
                self._knowledge_registry = object()
                self.search_calls = 0

            def search_memory(self, **kwargs):
                self.search_calls += 1
                return [{"memory": "tenant context"}]

        class _TrunkManager:
            def __init__(self):
                self.memory = object()
                self.search_calls = 0

            def search_memory(self, **kwargs):
                self.search_calls += 1
                raise RuntimeError("org trunk down")

        tenant_manager = _TenantManager()
        trunk_manager = _TrunkManager()

        agent._memory_tenant_id = "test_tenant"
        agent._memory_agent_name = "test_agent"
        agent.memory_manager = tenant_manager

        with patch(
            "cogniverse_core.memory.manager.Mem0MemoryManager",
            return_value=trunk_manager,
        ):
            with pytest.raises(RuntimeError, match="org trunk down"):
                agent._federate_with_org_trunk(
                    "test query", [{"memory": "tenant context"}], 5
                )

        assert tenant_manager.search_calls == 0
        assert trunk_manager.search_calls == 1

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_update_memory(self, mock_manager_class, agent):
        """Test updating memory"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager

        # Initialize and update
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)
        success = agent.update_memory("test content")

        assert success is True
        mock_manager.add_memory.assert_called_once_with(
            content="test content",
            tenant_id="test_tenant",
            agent_name="test_agent",
            metadata=None,
            infer=True,
        )

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_session_id_auto_stamped_on_metadata(self, mock_manager_class, agent):
        """When set_session_id is active, update_memory adds session_id to metadata."""
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_456"
        mock_manager_class.return_value = mock_manager

        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)
        agent.set_session_id("s_session_a")
        agent.update_memory("scratch", metadata={"kind": "session_scratch"})

        call = mock_manager.add_memory.call_args
        assert call.kwargs["metadata"] == {
            "kind": "session_scratch",
            "session_id": "s_session_a",
        }

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_session_id_does_not_overwrite_caller_value(
        self, mock_manager_class, agent
    ):
        """Caller-supplied session_id wins over the dispatcher-set value."""
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_xy"
        mock_manager_class.return_value = mock_manager

        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)
        agent.set_session_id("s_dispatcher")
        agent.update_memory(
            "scratch",
            metadata={"kind": "session_scratch", "session_id": "s_caller_explicit"},
        )

        call = mock_manager.add_memory.call_args
        assert call.kwargs["metadata"]["session_id"] == "s_caller_explicit"

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_set_session_id_none_clears(self, mock_manager_class, agent):
        """Clearing the session id leaves metadata untouched on subsequent writes."""
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_z"
        mock_manager_class.return_value = mock_manager

        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)
        agent.set_session_id("s_open")
        agent.set_session_id(None)
        agent.update_memory("after-close", metadata={"kind": "entity_fact"})

        call = mock_manager.add_memory.call_args
        assert "session_id" not in (call.kwargs["metadata"] or {})

    def test_set_session_id_rejects_empty_string(self, agent):
        """Empty / whitespace session ids never reach a write."""
        with pytest.raises(ValueError, match="non-empty"):
            agent.set_session_id("")
        with pytest.raises(ValueError, match="non-empty"):
            agent.set_session_id("   ")

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_clear_memory(self, mock_manager_class, agent):
        """Test clearing memory"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.clear_agent_memory.return_value = True
        mock_manager_class.return_value = mock_manager

        # Initialize and clear
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)
        assert agent._memory_initialized is True

        success = agent.clear_memory()

        assert success is True
        mock_manager.clear_agent_memory.assert_called_once()

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_inject_context_into_prompt(self, mock_manager_class, agent):
        """Test injecting context into prompt"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.search_memory.return_value = [
            {"memory": "Context 1"},
        ]
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)

        # Inject context
        original_prompt = "Answer the query"
        enhanced_prompt = agent.inject_context_into_prompt(
            original_prompt, "test query"
        )

        assert original_prompt in enhanced_prompt
        assert "Context 1" in enhanced_prompt
        assert "Relevant Context from Memory" in enhanced_prompt

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_inject_context_no_results(self, mock_manager_class, agent):
        """Test injecting context when no results found"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.search_memory.return_value = []
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)

        # Inject context
        original_prompt = "Answer the query"
        enhanced_prompt = agent.inject_context_into_prompt(
            original_prompt, "test query"
        )

        # Should return original prompt unchanged
        assert enhanced_prompt == original_prompt

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_remember_success(self, mock_manager_class, agent):
        """Test remembering successful interaction"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)

        # Remember success
        success = agent.remember_success("test query", "test result", {"key": "value"})

        assert success is True
        # Verify content includes SUCCESS marker
        call_args = mock_manager.add_memory.call_args
        assert "SUCCESS" in call_args[1]["content"]
        assert "test query" in call_args[1]["content"]

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_remember_failure(self, mock_manager_class, agent):
        """Test remembering failed interaction"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)

        # Remember failure
        success = agent.remember_failure("test query", "test error")

        assert success is True
        # Verify content includes FAILURE marker
        call_args = mock_manager.add_memory.call_args
        assert "FAILURE" in call_args[1]["content"]
        assert "test error" in call_args[1]["content"]

    @pytest.mark.asyncio
    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    async def test_a_background_write_keeps_the_tenant_and_session_that_queued_it(
        self, mock_manager_class, agent
    ):
        """The write runs after the response, by when a shared agent instance
        may be serving another tenant; it still lands in the tenant and
        session of the request that queued it."""
        import threading

        from cogniverse_agents.background_memory_writes import (
            MEMORY_WRITE_CONCURRENCY,
            drain_background_memory_writes,
            get_background_memory_writer,
        )

        release = threading.Event()
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager
        agent.initialize_memory("test_agent", "tenant_a", **MEMORY_INIT_DEFAULTS)
        agent.set_session_id("session-a")

        # Occupy every writer thread so this write waits in the queue.
        for _ in range(MEMORY_WRITE_CONCURRENCY):
            get_background_memory_writer().submit(
                lambda: release.wait(5), tenant_id="other", agent_name="blocker"
            )
        queued = agent.write_memory_in_background(
            agent.remember_success, "test query", "test result"
        )
        agent._memory_tenant_id = "tenant_b"
        agent.set_session_id(None)
        release.set()
        assert await drain_background_memory_writes(5.0) is True

        assert queued is True
        call = mock_manager.add_memory.call_args
        assert call.kwargs["tenant_id"] == "tenant_a"
        assert call.kwargs["metadata"] == {"session_id": "session-a"}
        assert "test query" in call.kwargs["content"]

    @staticmethod
    def _per_tenant_managers(mock_manager_class):
        """Mem0MemoryManager is one instance per tenant; so is this double."""
        managers = {}

        def manager_for(tenant_id):
            if tenant_id not in managers:
                manager = MagicMock(name=f"mem0[{tenant_id}]")
                manager.memory = MagicMock()
                manager.add_memory.return_value = f"mem-{tenant_id}"
                managers[tenant_id] = manager
            return managers[tenant_id]

        mock_manager_class.side_effect = lambda tenant_id: manager_for(tenant_id)
        return managers

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_a_request_writes_through_its_own_tenants_manager_after_a_rebind(
        self, mock_manager_class, agent
    ):
        """A shared agent last bound to tenant B still writes tenant A's
        request through A's manager."""
        from cogniverse_agents.memory_aware_mixin import _MEMORY_TENANT_ID

        managers = self._per_tenant_managers(mock_manager_class)
        agent.initialize_memory("test_agent", "tenant_a", **MEMORY_INIT_DEFAULTS)
        agent.initialize_memory("test_agent", "tenant_b", **MEMORY_INIT_DEFAULTS)

        _MEMORY_TENANT_ID.set("tenant_a")
        written = agent.update_memory("a fact", infer=False)

        assert written is True
        assert managers["tenant_b"].add_memory.call_count == 0
        assert managers["tenant_a"].add_memory.call_args.kwargs["tenant_id"] == (
            "tenant_a"
        )

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_a_request_for_an_unbound_tenant_gets_no_other_tenants_memory(
        self, mock_manager_class, agent
    ):
        from cogniverse_agents.memory_aware_mixin import _MEMORY_TENANT_ID

        managers = self._per_tenant_managers(mock_manager_class)
        agent.initialize_memory("test_agent", "tenant_a", **MEMORY_INIT_DEFAULTS)

        _MEMORY_TENANT_ID.set("tenant_c")
        enabled = agent.is_memory_enabled()
        written = agent.update_memory("c fact", infer=False)

        assert enabled is False
        assert written is False
        assert managers["tenant_a"].add_memory.call_count == 0

    def test_a_directly_bound_manager_serves_a_request_for_its_tenant(self, agent):
        """Callers that bind memory by assignment (the knowledge router) keep
        working inside a request for that tenant."""
        from cogniverse_agents.memory_aware_mixin import _MEMORY_TENANT_ID

        manager = MagicMock()
        manager.add_memory.return_value = "mem-1"
        agent.memory_manager = manager
        agent._memory_tenant_id = "tenant_k"
        agent._memory_agent_name = "knowledge_agent"
        agent._memory_initialized = True

        _MEMORY_TENANT_ID.set("tenant_k")
        written = agent.update_memory("k fact", infer=False)

        assert written is True
        assert manager.add_memory.call_args.kwargs["tenant_id"] == "tenant_k"
        assert manager.add_memory.call_args.kwargs["agent_name"] == "knowledge_agent"

    @pytest.mark.asyncio
    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    async def test_a_queued_write_keeps_its_tenants_manager_after_a_rebind(
        self, mock_manager_class, agent
    ):
        import threading

        from cogniverse_agents.background_memory_writes import (
            MEMORY_WRITE_CONCURRENCY,
            drain_background_memory_writes,
            get_background_memory_writer,
        )

        managers = self._per_tenant_managers(mock_manager_class)
        release = threading.Event()
        agent.initialize_memory("test_agent", "tenant_a", **MEMORY_INIT_DEFAULTS)
        for _ in range(MEMORY_WRITE_CONCURRENCY):
            get_background_memory_writer().submit(
                lambda: release.wait(5), tenant_id="other", agent_name="blocker"
            )
        queued = agent.write_memory_in_background(
            agent.remember_success, "a query", "a result"
        )
        agent.initialize_memory("test_agent", "tenant_b", **MEMORY_INIT_DEFAULTS)
        release.set()
        assert await drain_background_memory_writes(5.0) is True

        assert queued is True
        assert managers["tenant_b"].add_memory.call_count == 0
        assert "a query" in managers["tenant_a"].add_memory.call_args.kwargs["content"]

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_get_memory_summary(self, mock_manager_class, agent):
        """Test getting memory summary"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.get_memory_stats.return_value = {
            "total_memories": 5,
            "enabled": True,
        }
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant", **MEMORY_INIT_DEFAULTS)

        # Get summary
        summary = agent.get_memory_summary()

        assert summary["enabled"] is True
        assert summary["agent_name"] == "test_agent"
        assert summary["tenant_id"] == "test_tenant"
        assert summary["initialized"] is True
        assert summary["total_memories"] == 5

    def test_get_memory_summary_uninitialized(self, agent):
        """Test getting memory summary when uninitialized"""
        summary = agent.get_memory_summary()

        assert summary["enabled"] is False
        assert summary["initialized"] is False


class TestTenantInstructionsFaultContract:
    """A ConfigStore outage is distinguishable from a tenant with no instructions."""

    def _agent_with_manager(self, manager):
        agent = MockAgentWithMemory()
        agent._memory_tenant_id = "acme:prod"
        agent._config_manager = manager
        return agent

    def test_status_vocabulary_is_the_wire_contract(self):
        from cogniverse_agents.memory_aware_mixin import (
            TENANT_INSTRUCTIONS_LOADED,
            TENANT_INSTRUCTIONS_NONE,
            TENANT_INSTRUCTIONS_SPAN_ATTRIBUTE,
            TENANT_INSTRUCTIONS_UNAVAILABLE,
        )

        assert TENANT_INSTRUCTIONS_LOADED == "loaded"
        assert TENANT_INSTRUCTIONS_NONE == "none"
        assert TENANT_INSTRUCTIONS_UNAVAILABLE == "unavailable"
        assert TENANT_INSTRUCTIONS_SPAN_ATTRIBUTE == "enrichment.tenant_instructions"

    def test_loaded_instructions_carry_loaded_status(self):
        manager = MagicMock()
        manager.get_tenant_instructions_config.return_value = {"text": "Be terse."}
        agent = self._agent_with_manager(manager)
        assert agent._get_tenant_instructions() == ("Be terse.", "loaded")

    def test_absent_instructions_carry_none_status(self):
        manager = MagicMock()
        manager.get_tenant_instructions_config.return_value = None
        agent = self._agent_with_manager(manager)
        assert agent._get_tenant_instructions() == (None, "none")

    def test_store_outage_carries_unavailable_status(self):
        manager = MagicMock()
        manager.get_tenant_instructions_config.side_effect = ConnectionError(
            "config store down"
        )
        agent = self._agent_with_manager(manager)
        assert agent._get_tenant_instructions() == (None, "unavailable")

    def test_inject_context_records_unavailable_status_on_agent(self):
        manager = MagicMock()
        manager.get_tenant_instructions_config.side_effect = ConnectionError(
            "config store down"
        )
        agent = self._agent_with_manager(manager)
        prompt = agent.inject_context_into_prompt("Answer the query", "q")
        assert prompt == "Answer the query"
        assert agent.last_tenant_instructions_status == "unavailable"

    def test_inject_context_records_loaded_status_on_agent(self):
        manager = MagicMock()
        manager.get_tenant_instructions_config.return_value = {"text": "Be terse."}
        agent = self._agent_with_manager(manager)
        prompt = agent.inject_context_into_prompt("Answer the query", "q")
        assert "## Tenant Instructions\nBe terse." in prompt
        assert agent.last_tenant_instructions_status == "loaded"

    def test_inject_context_span_attribute_marks_unavailable(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test-tenant-instructions")

        manager = MagicMock()
        manager.get_tenant_instructions_config.side_effect = ConnectionError(
            "config store down"
        )
        agent = self._agent_with_manager(manager)
        with tracer.start_as_current_span("dispatch"):
            agent.inject_context_into_prompt("Answer the query", "q")

        (span,) = exporter.get_finished_spans()
        assert span.attributes["enrichment.tenant_instructions"] == "unavailable"

    def test_concurrent_requests_mark_their_own_span(self):
        import threading

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from cogniverse_agents.memory_aware_mixin import _MEMORY_TENANT_ID

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test-tenant-instructions-concurrency")

        def _by_tenant(tenant_id):
            if tenant_id == "acme:down":
                raise ConnectionError("config store down")
            return {"text": "Be terse."}

        manager = MagicMock()
        manager.get_tenant_instructions_config.side_effect = _by_tenant
        agent = self._agent_with_manager(manager)

        barrier = threading.Barrier(2)
        failures = []

        def _run(tenant_id):
            try:
                _MEMORY_TENANT_ID.set(tenant_id)
                barrier.wait(timeout=10)
                with tracer.start_as_current_span(f"dispatch-{tenant_id}"):
                    agent.inject_context_into_prompt("Answer the query", "q")
            except Exception as exc:  # noqa: BLE001 - surfaced via assert below
                failures.append(exc)

        threads = [
            threading.Thread(target=_run, args=("acme:down",)),
            threading.Thread(target=_run, args=("acme:up",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert failures == []
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert (
            spans["dispatch-acme:down"].attributes["enrichment.tenant_instructions"]
            == "unavailable"
        )
        assert (
            spans["dispatch-acme:up"].attributes["enrichment.tenant_instructions"]
            == "loaded"
        )
