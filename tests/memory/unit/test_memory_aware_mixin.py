"""
Unit tests for MemoryAwareMixin
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin


class TestAgent(MemoryAwareMixin):
    """Test agent class using memory mixin"""

    def __init__(self):
        super().__init__()


class TestMemoryAwareMixin:
    """Test MemoryAwareMixin"""

    @pytest.fixture
    def agent(self):
        """Create test agent"""
        return TestAgent()

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
        assert agent.agent_name is None
        assert agent.tenant_id is None
        assert agent._memory_initialized is False

    def test_is_memory_enabled_false_by_default(self, agent):
        """Test memory is disabled by default"""
        assert agent.is_memory_enabled() is False

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_initialize_memory_success(self, mock_manager_class, agent):
        """Test successful memory initialization"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()  # Mem0 uses .memory attribute
        mock_manager_class.return_value = mock_manager

        # Initialize memory
        success = agent.initialize_memory("test_agent", "test_tenant")

        assert success is True
        assert agent.agent_name == "test_agent"
        assert agent.tenant_id == "test_tenant"
        assert agent._memory_initialized is True

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_initialize_memory_with_vespa_config(self, mock_manager_class, agent):
        """Test memory initialization with Vespa configuration"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = None  # Not initialized yet
        mock_manager_class.return_value = mock_manager

        # Initialize memory with custom Vespa config
        success = agent.initialize_memory(
            "test_agent",
            "test_tenant",
            vespa_host="vespa.local",
            vespa_port=9090,
        )

        assert success is True
        mock_manager.initialize.assert_called_once_with(
            vespa_host="vespa.local",
            vespa_port=9090,
            base_schema_name="agent_memories",
            auto_create_schema=True,
        )

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

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
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
        agent.initialize_memory("test_agent", "test_tenant")
        context = agent.get_relevant_context("test query")

        assert context is not None
        assert "Context 1" in context
        assert "Context 2" in context
        mock_manager.search_memory.assert_called_once()

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_update_memory(self, mock_manager_class, agent):
        """Test updating memory"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager

        # Initialize and update
        agent.initialize_memory("test_agent", "test_tenant")
        success = agent.update_memory("test content")

        assert success is True
        mock_manager.add_memory.assert_called_once_with(
            content="test content",
            tenant_id="test_tenant",
            agent_name="test_agent",
            metadata=None,
        )

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_clear_memory(self, mock_manager_class, agent):
        """Test clearing memory"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.clear_agent_memory.return_value = True
        mock_manager_class.return_value = mock_manager

        # Initialize and clear
        agent.initialize_memory("test_agent", "test_tenant")
        assert agent._memory_initialized is True

        success = agent.clear_memory()

        assert success is True
        mock_manager.clear_agent_memory.assert_called_once()

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
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
        agent.initialize_memory("test_agent", "test_tenant")

        # Inject context
        original_prompt = "Answer the query"
        enhanced_prompt = agent.inject_context_into_prompt(original_prompt, "test query")

        assert original_prompt in enhanced_prompt
        assert "Context 1" in enhanced_prompt
        assert "Relevant Context from Memory" in enhanced_prompt

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_inject_context_no_results(self, mock_manager_class, agent):
        """Test injecting context when no results found"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.search_memory.return_value = []
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant")

        # Inject context
        original_prompt = "Answer the query"
        enhanced_prompt = agent.inject_context_into_prompt(original_prompt, "test query")

        # Should return original prompt unchanged
        assert enhanced_prompt == original_prompt

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_remember_success(self, mock_manager_class, agent):
        """Test remembering successful interaction"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant")

        # Remember success
        success = agent.remember_success("test query", "test result", {"key": "value"})

        assert success is True
        # Verify content includes SUCCESS marker
        call_args = mock_manager.add_memory.call_args
        assert "SUCCESS" in call_args[1]["content"]
        assert "test query" in call_args[1]["content"]

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
    def test_remember_failure(self, mock_manager_class, agent):
        """Test remembering failed interaction"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.memory = MagicMock()
        mock_manager.add_memory.return_value = "mem_123"
        mock_manager_class.return_value = mock_manager

        # Initialize
        agent.initialize_memory("test_agent", "test_tenant")

        # Remember failure
        success = agent.remember_failure("test query", "test error")

        assert success is True
        # Verify content includes FAILURE marker
        call_args = mock_manager.add_memory.call_args
        assert "FAILURE" in call_args[1]["content"]
        assert "test error" in call_args[1]["content"]

    @patch("cogniverse_core.agents.memory_aware_mixin.Mem0MemoryManager")
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
        agent.initialize_memory("test_agent", "test_tenant")

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
