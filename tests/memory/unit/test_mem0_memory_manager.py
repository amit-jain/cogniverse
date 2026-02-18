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
        mock_backend.get_tenant_schema_name.return_value = "agent_memories_test_tenant"
        mock_backend.deploy_schema.return_value = True

        # Mock the registry
        mock_registry = MagicMock()
        mock_registry.get_ingestion_backend.return_value = mock_backend
        mock_get_backend_registry.return_value = mock_registry

        # Create dependencies for dependency injection
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
        )

        config_manager = create_default_config_manager()
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

        # Initialize
        manager.initialize(
            backend_host="localhost",
            backend_port=8080,
            base_schema_name="agent_memories",
            config_manager=config_manager,
            schema_loader=schema_loader,
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

    # --- cleanup_expired_memories ---

    @patch("cogniverse_core.memory.manager.time")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_cleanup_expired_memories(self, mock_memory_class, mock_time, manager):
        """Deletes expired memories (epoch timestamps), keeps recent ones."""
        now = 1700000000
        mock_time.time.return_value = now
        max_age = 3600  # 1 hour

        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "old1", "created_at": now - 7200},  # 2 hours ago → expired
                {"id": "old2", "created_at": now - 5000},  # ~83 min ago → expired
                {"id": "recent", "created_at": now - 100},  # 100s ago → kept
            ]
        }
        manager.memory = mock_memory

        deleted = manager.cleanup_expired_memories(max_age)

        assert deleted == 2
        assert mock_memory.delete.call_count == 2
        mock_memory.delete.assert_any_call("old1")
        mock_memory.delete.assert_any_call("old2")

    @patch("cogniverse_core.memory.manager.time")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_cleanup_expired_memories_iso_ts(
        self, mock_memory_class, mock_time, manager
    ):
        """Correctly parses ISO string timestamps and deletes expired."""
        now = 1700000000
        mock_time.time.return_value = now
        max_age = 3600

        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {
            "results": [
                {
                    "id": "iso_old",
                    "created_at": "2023-11-14T14:00:00+00:00",  # epoch ~1699970400
                },
                {
                    "id": "iso_recent",
                    "created_at": "2023-11-14T22:30:00+00:00",  # epoch ~1700001000
                },
            ]
        }
        manager.memory = mock_memory

        deleted = manager.cleanup_expired_memories(max_age)

        # iso_old is well before cutoff, iso_recent is within 1h
        assert deleted == 1
        mock_memory.delete.assert_called_once_with("iso_old")

    @patch("cogniverse_core.memory.manager.time")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_cleanup_expired_memories_none_expired(
        self, mock_memory_class, mock_time, manager
    ):
        """No expired memories → deletes 0."""
        now = 1700000000
        mock_time.time.return_value = now

        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "recent1", "created_at": now - 10},
                {"id": "recent2", "created_at": now - 60},
            ]
        }
        manager.memory = mock_memory

        deleted = manager.cleanup_expired_memories(3600)

        assert deleted == 0
        mock_memory.delete.assert_not_called()

    def test_cleanup_expired_memories_not_initialized(self, manager):
        """Raises RuntimeError when memory is None."""
        manager.memory = None

        with pytest.raises(RuntimeError, match="not initialized"):
            manager.cleanup_expired_memories(3600)

    def test_cleanup_expired_memories_invalid_max_age(self, manager):
        """Raises ValueError when max_age_seconds <= 0."""
        manager.memory = MagicMock()

        with pytest.raises(ValueError, match="must be positive"):
            manager.cleanup_expired_memories(0)

        with pytest.raises(ValueError, match="must be positive"):
            manager.cleanup_expired_memories(-100)

    @patch("cogniverse_core.memory.manager.time")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_cleanup_expired_memories_missing_fields(
        self, mock_memory_class, mock_time, manager
    ):
        """Memories with missing id or created_at are skipped."""
        now = 1700000000
        mock_time.time.return_value = now

        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "no_ts"},  # Missing created_at → skipped
                {"created_at": now - 7200},  # Missing id → skipped
                {"id": "valid_old", "created_at": now - 7200},  # Expired → deleted
                "not_a_dict",  # Not a dict → skipped
            ]
        }
        manager.memory = mock_memory

        deleted = manager.cleanup_expired_memories(3600)

        assert deleted == 1
        mock_memory.delete.assert_called_once_with("valid_old")
