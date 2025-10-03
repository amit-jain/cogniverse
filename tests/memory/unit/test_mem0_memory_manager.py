"""
Unit tests for Mem0MemoryManager
"""

import pytest
from unittest.mock import MagicMock, patch

from src.common.mem0_memory_manager import Mem0MemoryManager


class TestMem0MemoryManager:
    """Test Mem0MemoryManager"""

    @pytest.fixture
    def manager(self):
        """Create manager instance"""
        return Mem0MemoryManager()

    def test_singleton_pattern(self):
        """Test singleton pattern"""
        manager1 = Mem0MemoryManager()
        manager2 = Mem0MemoryManager()
        assert manager1 is manager2

    def test_initialization(self, manager):
        """Test initial state"""
        assert manager.memory is None
        assert manager.config is None
        assert manager.vector_store is None

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_initialize_success(self, mock_memory_class, mock_store_class, manager):
        """Test successful initialization"""
        # Setup mocks
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_memory = MagicMock()
        mock_memory_class.from_config.return_value = mock_memory

        # Initialize
        manager.initialize(
            vespa_host="localhost",
            vespa_port=8080,
            collection_name="test_memories",
        )

        assert manager.vector_store is not None
        assert manager.memory is not None
        assert manager.config is not None

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_add_memory(self, mock_memory_class, mock_store_class, manager):
        """Test adding memory"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"id": "mem_123"}
        manager.memory = mock_memory

        # Add memory
        memory_id = manager.add_memory(
            content="Test content",
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert memory_id == "mem_123"
        mock_memory.add.assert_called_once_with(
            "Test content",
            user_id="tenant1",
            agent_id="test_agent",
            metadata={},
        )

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_search_memory(self, mock_memory_class, mock_store_class, manager):
        """Test searching memory"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"memory": "Result 1"},
            {"memory": "Result 2"},
        ]
        manager.memory = mock_memory

        # Search
        results = manager.search_memory(
            query="test query",
            tenant_id="tenant1",
            agent_name="test_agent",
            top_k=5,
        )

        assert len(results) == 2
        mock_memory.search.assert_called_once_with(
            "test query",
            user_id="tenant1",
            agent_id="test_agent",
            limit=5,
        )

    def test_search_memory_not_initialized(self):
        """Test search when not initialized"""
        # Create fresh manager and ensure not initialized
        manager = Mem0MemoryManager()
        manager.memory = None  # Force not initialized state

        results = manager.search_memory(
            query="test",
            tenant_id="tenant1",
            agent_name="test_agent",
        )
        assert results == []

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_get_all_memories(self, mock_memory_class, mock_store_class, manager):
        """Test getting all memories"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [
            {"id": "mem_1", "text": "Memory 1"},
            {"id": "mem_2", "text": "Memory 2"},
        ]
        manager.memory = mock_memory

        # Get all
        memories = manager.get_all_memories(
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert len(memories) == 2
        mock_memory.get_all.assert_called_once_with(
            user_id="tenant1",
            agent_id="test_agent",
        )

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_delete_memory(self, mock_memory_class, mock_store_class, manager):
        """Test deleting memory"""
        # Setup
        mock_memory = MagicMock()
        manager.memory = mock_memory

        # Delete
        success = manager.delete_memory(
            memory_id="mem_123",
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert success is True
        mock_memory.delete.assert_called_once_with(
            "mem_123",
            user_id="tenant1",
            agent_id="test_agent",
        )

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_clear_agent_memory(self, mock_memory_class, mock_store_class, manager):
        """Test clearing all agent memory"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [
            {"id": "mem_1"},
            {"id": "mem_2"},
        ]
        manager.memory = mock_memory

        # Clear
        success = manager.clear_agent_memory(
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert success is True
        assert mock_memory.delete.call_count == 2

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_update_memory(self, mock_memory_class, mock_store_class, manager):
        """Test updating memory"""
        # Setup
        mock_memory = MagicMock()
        manager.memory = mock_memory

        # Update
        success = manager.update_memory(
            memory_id="mem_123",
            content="Updated content",
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert success is True
        mock_memory.update.assert_called_once()

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_health_check(self, mock_memory_class, mock_store_class, manager):
        """Test health check"""
        # Setup
        mock_store = MagicMock()
        mock_store.col_info.return_value = {"name": "test"}
        mock_memory = MagicMock()

        manager.memory = mock_memory
        manager.vector_store = mock_store

        # Check
        health = manager.health_check()
        assert health is True

    def test_health_check_not_initialized(self):
        """Test health check when not initialized"""
        manager = Mem0MemoryManager()
        manager.memory = None  # Force not initialized state
        manager.vector_store = None

        health = manager.health_check()
        assert health is False

    @patch("src.common.mem0_memory_manager.VespaVectorStore")
    @patch("src.common.mem0_memory_manager.Memory")
    def test_get_memory_stats(self, mock_memory_class, mock_store_class, manager):
        """Test getting memory stats"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        manager.memory = mock_memory

        # Get stats
        stats = manager.get_memory_stats(
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert stats["total_memories"] == 3
        assert stats["enabled"] is True
        assert stats["tenant_id"] == "tenant1"
        assert stats["agent_name"] == "test_agent"

    def test_get_memory_stats_not_initialized(self):
        """Test stats when not initialized"""
        manager = Mem0MemoryManager()
        manager.memory = None  # Force not initialized state

        stats = manager.get_memory_stats(
            tenant_id="tenant1",
            agent_name="test_agent",
        )
        assert stats["total_memories"] == 0
        assert stats["enabled"] is False
