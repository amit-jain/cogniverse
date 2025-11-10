"""
Unit tests for Mem0MemoryManager
"""

from unittest.mock import MagicMock, patch

import pytest
from cogniverse_core.memory.manager import Mem0MemoryManager


class TestMem0MemoryManager:
    """Test Mem0MemoryManager"""

    @pytest.fixture
    def manager(self):
        """Create manager instance"""
        # Clear singleton instance to ensure fresh state for each test
        Mem0MemoryManager._instances.pop("test_tenant", None)
        return Mem0MemoryManager(tenant_id="test_tenant")

    def test_per_tenant_singleton_pattern(self):
        """Test per-tenant singleton pattern"""
        manager1 = Mem0MemoryManager(tenant_id="tenant1")
        manager2 = Mem0MemoryManager(tenant_id="tenant1")
        manager3 = Mem0MemoryManager(tenant_id="tenant2")

        # Same tenant returns same instance
        assert manager1 is manager2
        # Different tenant returns different instance
        assert manager1 is not manager3
        assert manager2 is not manager3

    def test_initialization(self, manager):
        """Test initial state"""
        assert manager.memory is None
        assert manager.config is None

    @patch("cogniverse_core.registries.backend_registry.get_backend_registry")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_initialize_success(
        self,
        mock_memory_class,
        mock_get_backend_registry,
        manager,
    ):
        """Test successful initialization"""
        # Setup mocks
        mock_memory = MagicMock()
        mock_memory_class.from_config.return_value = mock_memory

        # Mock the Backend instance returned by registry
        mock_backend = MagicMock()
        mock_backend.get_tenant_schema_name.return_value = (
            "agent_memories_test_tenant"
        )
        mock_backend.deploy_schema.return_value = True

        # Mock the registry
        mock_registry = MagicMock()
        mock_registry.get_ingestion_backend.return_value = mock_backend
        mock_get_backend_registry.return_value = mock_registry

        # Create dependencies for dependency injection
        from pathlib import Path

        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
        )
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        config_manager = create_default_config_manager()
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

        # Initialize
        manager.initialize(
            backend_host="localhost",
            backend_port=8080,
            base_schema_name="agent_memories",
            config_manager=config_manager,
            schema_loader=schema_loader
        )

        assert manager.memory is not None
        assert manager.config is not None
        # Verify tenant-specific schema was used
        assert (
            manager.config["vector_store"]["config"]["collection_name"]
            == "agent_memories_test_tenant"
        )

    @patch("cogniverse_core.memory.manager.Memory")
    def test_add_memory(self, mock_memory_class, manager):
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

    @patch("cogniverse_core.memory.manager.Memory")
    def test_search_memory(self, mock_memory_class, manager):
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
        manager = Mem0MemoryManager(tenant_id="tenant1")
        manager.memory = None  # Force not initialized state

        results = manager.search_memory(
            query="test",
            tenant_id="tenant1",
            agent_name="test_agent",
        )
        assert results == []

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_all_memories(self, mock_memory_class, manager):
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

    @patch("cogniverse_core.memory.manager.Memory")
    def test_delete_memory(self, mock_memory_class, manager):
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
        # Implementation only passes memory_id (tenant_id and agent_name not used)
        mock_memory.delete.assert_called_once_with("mem_123")

    @patch("cogniverse_core.memory.manager.Memory")
    def test_clear_agent_memory(self, mock_memory_class, manager):
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

    @patch("cogniverse_core.memory.manager.Memory")
    def test_update_memory(self, mock_memory_class, manager):
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

    @patch("cogniverse_core.memory.manager.Memory")
    def test_health_check(self, mock_memory_class, manager):
        """Test health check"""
        # Setup
        mock_memory = MagicMock()
        manager.memory = mock_memory

        # Check
        health = manager.health_check()
        assert health is True

    def test_health_check_not_initialized(self):
        """Test health check when not initialized"""
        manager = Mem0MemoryManager(tenant_id="test_tenant")
        manager.memory = None  # Force not initialized state

        health = manager.health_check()
        assert health is False

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_memory_stats(self, mock_memory_class, manager):
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
        manager = Mem0MemoryManager(tenant_id="test_tenant")
        manager.memory = None  # Force not initialized state

        stats = manager.get_memory_stats(
            tenant_id="tenant1",
            agent_name="test_agent",
        )
        assert stats["total_memories"] == 0
        assert stats["enabled"] is False
