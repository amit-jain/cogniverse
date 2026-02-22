"""
Integration tests for agent session memory via MemoryAwareMixin.

Validates:
- Memory initialization with tenant_id and agent_name
- Context retrieval and memory updates
- Memory namespacing: (tenant_id, agent_name) isolation
- Multi-turn: agent remembers context across calls
- Memory summary and stats
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin


class MockAgent(MemoryAwareMixin):
    """Test agent combining MemoryAwareMixin with a simple base."""

    def __init__(self, agent_name: str = "test_agent"):
        super().__init__()
        self.agent_name = agent_name


class TestMemoryInitialization:
    """Test memory initialization lifecycle."""

    def test_starts_uninitialized(self):
        """Memory is disabled before initialize_memory()."""
        agent = MockAgent()
        assert not agent.is_memory_enabled()
        assert agent.memory_manager is None
        assert not agent._memory_initialized

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_initialize_memory_success(self, mock_mem0_cls):
        """Successful initialization sets all state."""
        mock_instance = MagicMock()
        mock_instance.memory = None  # Triggers initialize()
        mock_mem0_cls.return_value = mock_instance

        agent = MockAgent("search_agent")
        result = agent.initialize_memory(
            agent_name="search_agent",
            tenant_id="acme_corp",
            backend_host="localhost",
            backend_port=8080,
        )

        assert result is True
        assert agent.is_memory_enabled()
        assert agent._memory_agent_name == "search_agent"
        assert agent._memory_tenant_id == "acme_corp"
        mock_mem0_cls.assert_called_once_with(tenant_id="acme_corp")
        mock_instance.initialize.assert_called_once()

    def test_initialize_memory_requires_tenant_id(self):
        """Raises ValueError if tenant_id is empty."""
        agent = MockAgent()
        with pytest.raises(ValueError, match="tenant_id is required"):
            agent.initialize_memory(agent_name="test", tenant_id="")

    def test_initialize_memory_requires_nonempty_tenant_id(self):
        """Raises ValueError if tenant_id is None."""
        agent = MockAgent()
        with pytest.raises(ValueError, match="tenant_id is required"):
            agent.initialize_memory(agent_name="test", tenant_id=None)


class TestMemoryNamespacing:
    """Test (tenant_id, agent_name) memory isolation."""

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_different_tenants_get_separate_managers(self, mock_mem0_cls):
        """Each tenant_id gets its own Mem0MemoryManager."""
        mock_mem0_cls.return_value = MagicMock(memory=None)

        agent_a = MockAgent()
        agent_a.initialize_memory("search", tenant_id="tenant_a")

        agent_b = MockAgent()
        agent_b.initialize_memory("search", tenant_id="tenant_b")

        # Two separate Mem0MemoryManager instances created
        assert mock_mem0_cls.call_count == 2
        calls = mock_mem0_cls.call_args_list
        assert calls[0].kwargs["tenant_id"] == "tenant_a"
        assert calls[1].kwargs["tenant_id"] == "tenant_b"

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_different_agents_get_separate_names(self, mock_mem0_cls):
        """Different agent_names are tracked separately."""
        mock_mem0_cls.return_value = MagicMock(memory=None)

        search = MockAgent()
        search.initialize_memory("search_agent", tenant_id="acme")

        orchestrator = MockAgent()
        orchestrator.initialize_memory("orchestrator_agent", tenant_id="acme")

        assert search._memory_agent_name == "search_agent"
        assert orchestrator._memory_agent_name == "orchestrator_agent"


class TestContextRetrieval:
    """Test memory search and context retrieval."""

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_get_relevant_context_returns_formatted(self, mock_mem0_cls):
        """get_relevant_context() returns formatted memory results."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.search_memory.return_value = [
            {"memory": "User prefers short summaries"},
            {"memory": "User searches for ML content often"},
        ]
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("test", tenant_id="acme")

        context = agent.get_relevant_context("machine learning videos")

        assert context is not None
        assert "short summaries" in context
        assert "ML content" in context
        mock_manager.search_memory.assert_called_once_with(
            query="machine learning videos",
            tenant_id="acme",
            agent_name="test",
            top_k=5,
        )

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_get_context_when_no_memories(self, mock_mem0_cls):
        """Returns None when no memories match."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.search_memory.return_value = []
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("test", tenant_id="acme")

        context = agent.get_relevant_context("completely new topic")
        assert context is None

    def test_get_context_when_not_initialized(self):
        """Returns None when memory not initialized."""
        agent = MockAgent()
        context = agent.get_relevant_context("anything")
        assert context is None


class TestMemoryUpdates:
    """Test memory add/update operations."""

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_update_memory_stores_content(self, mock_mem0_cls):
        """update_memory() calls add_memory with correct namespace."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.add_memory.return_value = "mem-123"
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("search_agent", tenant_id="acme")

        result = agent.update_memory("User likes technical content")

        assert result is True
        mock_manager.add_memory.assert_called_once_with(
            content="User likes technical content",
            tenant_id="acme",
            agent_name="search_agent",
            metadata=None,
        )

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_remember_success_formats_content(self, mock_mem0_cls):
        """remember_success() stores formatted success memory."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.add_memory.return_value = "mem-456"
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("search_agent", tenant_id="acme")

        agent.remember_success("ML videos", {"results": 10})

        call_args = mock_manager.add_memory.call_args
        content = call_args.kwargs["content"]
        assert "SUCCESS" in content
        assert "ML videos" in content

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_remember_failure_formats_content(self, mock_mem0_cls):
        """remember_failure() stores formatted failure memory."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.add_memory.return_value = "mem-789"
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("search_agent", tenant_id="acme")

        agent.remember_failure("bad query", "Timeout error")

        call_args = mock_manager.add_memory.call_args
        content = call_args.kwargs["content"]
        assert "FAILURE" in content
        assert "Timeout error" in content


class TestMultiTurnMemory:
    """Test multi-turn memory flow (simulating conversation)."""

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_inject_context_into_prompt(self, mock_mem0_cls):
        """inject_context_into_prompt() enhances prompt with memories."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.search_memory.return_value = [
            {"memory": "User prefers ColPali results"},
        ]
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("search_agent", tenant_id="acme")

        base_prompt = "You are a search agent."
        enhanced = agent.inject_context_into_prompt(base_prompt, "find videos")

        assert "You are a search agent." in enhanced
        assert "ColPali results" in enhanced
        assert "find videos" in enhanced

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_memory_summary(self, mock_mem0_cls):
        """get_memory_summary() returns structured state."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.get_memory_stats.return_value = {"total_memories": 42}
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("search_agent", tenant_id="acme")

        summary = agent.get_memory_summary()

        assert summary["enabled"] is True
        assert summary["agent_name"] == "search_agent"
        assert summary["tenant_id"] == "acme"
        assert summary["total_memories"] == 42

    @patch("cogniverse_agents.memory_aware_mixin.Mem0MemoryManager")
    def test_clear_memory(self, mock_mem0_cls):
        """clear_memory() delegates to manager."""
        mock_manager = MagicMock(memory=MagicMock())
        mock_manager.clear_agent_memory.return_value = True
        mock_mem0_cls.return_value = mock_manager

        agent = MockAgent()
        agent.initialize_memory("search_agent", tenant_id="acme")

        result = agent.clear_memory()
        assert result is True
        mock_manager.clear_agent_memory.assert_called_once_with(
            tenant_id="acme", agent_name="search_agent"
        )
