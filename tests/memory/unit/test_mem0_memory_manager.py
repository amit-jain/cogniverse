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

    def test_instances_cache_is_lru_bounded(self):
        """A burst of unique tenants must stay within the cache capacity.

        The unbounded dict pattern held every tenant instance forever,
        driving OOM in long test runs. The LRU cache caps the working
        set; the least-recently-used tenant is evicted when new ones
        arrive.
        """
        from cogniverse_foundation.caching import TenantLRUCache

        assert isinstance(Mem0MemoryManager._instances, TenantLRUCache)
        capacity = Mem0MemoryManager._instances.capacity

        Mem0MemoryManager._instances.clear()
        for i in range(capacity * 3):
            Mem0MemoryManager(tenant_id=f"burst-tenant-{i}")

        assert len(Mem0MemoryManager._instances) == capacity
        # Most recent `capacity` tenants must be present
        for i in range(capacity * 3 - capacity, capacity * 3):
            assert f"burst-tenant-{i}" in Mem0MemoryManager._instances

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
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        config_manager = ConfigManager(store=store)
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

        # Initialize
        manager.initialize(
            backend_host="localhost",
            backend_port=8080,
            llm_model="test-llm",
            embedding_model="lightonai/DenseOn",
            llm_base_url="http://localhost:11434/v1",
            embedder_base_url="http://localhost:8000",
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
        # Default embedding dimension is DenseOn's 768.
        assert manager.config["vector_store"]["config"]["embedding_model_dims"] == 768

    @patch("cogniverse_core.registries.backend_registry.get_backend_registry")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_initialize_threads_embedding_dims(
        self,
        mock_memory_class,
        mock_get_backend_registry,
        manager,
    ):
        """The embedding dimension is config-driven, not hardcoded 768 — a
        non-DenseOn embedder's dimension flows through to the vector store."""
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        mock_memory_class.from_config.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_tenant_schema_name.return_value = "agent_memories_test_tenant"
        mock_backend.deploy_schema.return_value = True
        mock_registry = MagicMock()
        mock_registry.get_ingestion_backend.return_value = mock_backend
        mock_get_backend_registry.return_value = mock_registry

        store = InMemoryConfigStore()
        store.initialize()
        config_manager = ConfigManager(store=store)

        manager.initialize(
            backend_host="localhost",
            backend_port=8080,
            llm_model="test-llm",
            embedding_model="some/other-embedder",
            llm_base_url="http://localhost:11434/v1",
            embedder_base_url="http://localhost:8000",
            base_schema_name="agent_memories",
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            embedding_dims=1024,
        )

        assert manager.config["vector_store"]["config"]["embedding_model_dims"] == 1024

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
            infer=True,
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
            filters=None,
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
        # Real mem0 get_all returns {"results": [...]}, not a bare list.
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "mem_1", "text": "Memory 1"},
                {"id": "mem_2", "text": "Memory 2"},
            ]
        }
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
            filters=None,
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

    # cleanup_expired_memories was the legacy bulk-age path; deleted in
    # favor of schema-driven cleanup_with_schema (covered by
    # tests/memory/integration/test_schema_lifecycle_integration.py and
    # tests/memory/integration/test_soft_delete_lifecycle.py).


class TestInitializeIdempotent:
    """Repeat initialize() with identical wiring must not rebuild the Mem0
    stack — the dispatcher runs initialize_memory per dispatched request on
    a per-tenant singleton, so every request paid a full Memory.from_config
    (embedder + LLM + vector-store construction)."""

    @patch("cogniverse_core.registries.backend_registry.get_backend_registry")
    @patch("cogniverse_core.memory.manager.Memory")
    def test_same_wiring_builds_memory_once(
        self, mock_memory_class, mock_get_backend_registry
    ):
        from pathlib import Path

        from cogniverse_core.memory.manager import Mem0MemoryManager
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        def _make_config_manager():
            store = InMemoryConfigStore()
            store.initialize()
            return ConfigManager(store=store)

        mock_memory_class.from_config.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_tenant_schema_name.return_value = "agent_memories_t_idem"
        mock_backend.deploy_schema.return_value = True
        mock_registry = MagicMock()
        mock_registry.get_ingestion_backend.return_value = mock_backend
        mock_get_backend_registry.return_value = mock_registry

        Mem0MemoryManager._instances.clear()
        manager = Mem0MemoryManager(tenant_id="t_idem")
        kwargs = dict(
            backend_host="localhost",
            backend_port=8080,
            llm_model="test-llm",
            embedding_model="lightonai/DenseOn",
            llm_base_url="http://localhost:11434/v1",
            embedder_base_url="http://localhost:8000",
            base_schema_name="agent_memories",
            config_manager=_make_config_manager(),
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        )

        manager.initialize(**kwargs)
        manager.initialize(**kwargs)
        manager.initialize(**kwargs)
        assert mock_memory_class.from_config.call_count == 1

        # Different wiring must rebuild.
        changed = dict(kwargs, llm_model="another-llm")
        manager.initialize(**changed)
        assert mock_memory_class.from_config.call_count == 2

        # A late-supplied knowledge registry still lands on the no-op path.
        sentinel_registry = MagicMock()
        manager.initialize(**dict(changed, knowledge_registry=sentinel_registry))
        assert mock_memory_class.from_config.call_count == 2
        assert manager._knowledge_registry is sentinel_registry

        Mem0MemoryManager._instances.clear()


class TestDropSessionServerSideFilter:
    """drop_session filters by the promoted session_id field, and falls back
    to the full scan when the filter yields nothing — a tenant schema
    deployed before the field existed flattens the filter to an empty result,
    and session cleanup must never become a silent no-op."""

    def _manager_with_rows(self, rows, filter_supported=True):
        from cogniverse_core.memory.manager import Mem0MemoryManager

        Mem0MemoryManager._instances.clear()
        manager = Mem0MemoryManager(tenant_id="t_drop")
        manager.tenant_id = "t_drop"

        memory = MagicMock()
        calls = []

        def _get_all(*, user_id, agent_id=None, filters=None, **kwargs):
            calls.append(filters)
            if filters and filters.get("session_id"):
                if not filter_supported:
                    return {"results": []}  # old schema: filter flattens empty
                return {
                    "results": [
                        r
                        for r in rows
                        if r["metadata"].get("session_id") == filters["session_id"]
                    ]
                }
            return {"results": list(rows)}

        memory.get_all = _get_all
        memory.delete = MagicMock()
        manager.memory = memory
        return manager, memory, calls

    def _registry(self):
        from cogniverse_core.memory.schema import Retention

        registry = MagicMock()
        schema = MagicMock()
        schema.retention = Retention.EPHEMERAL_SESSION
        registry.get.return_value = schema
        return registry

    def test_new_schema_deletes_via_filter_without_full_scan(self):
        rows = [
            {"id": "a", "metadata": {"session_id": "s1", "kind": "session_scratch"}},
            {"id": "b", "metadata": {"session_id": "s2", "kind": "session_scratch"}},
        ]
        manager, memory, calls = self._manager_with_rows(rows, filter_supported=True)

        deleted = manager.drop_session("s1", self._registry())

        assert deleted == {"session_scratch": 1}
        memory.delete.assert_called_once_with("a")
        assert calls == [{"session_id": "s1"}], (
            "filtered path must not fall back to a full scan when rows match"
        )

    def test_old_schema_falls_back_to_scan_and_warns(self, caplog):
        import logging

        rows = [
            {"id": "a", "metadata": {"session_id": "s1", "kind": "session_scratch"}},
        ]
        manager, memory, calls = self._manager_with_rows(rows, filter_supported=False)

        with caplog.at_level(logging.WARNING):
            deleted = manager.drop_session("s1", self._registry())

        assert deleted == {"session_scratch": 1}
        memory.delete.assert_called_once_with("a")
        assert calls == [{"session_id": "s1"}, None]
        assert any("predates the session_id field" in r.message for r in caplog.records)

    def test_truly_empty_session_deletes_nothing(self):
        rows = [
            {"id": "b", "metadata": {"session_id": "s2", "kind": "session_scratch"}},
        ]
        manager, memory, calls = self._manager_with_rows(rows, filter_supported=True)

        deleted = manager.drop_session("s1", self._registry())

        assert deleted == {}
        memory.delete.assert_not_called()


class TestImportChainWithoutCv2:
    """The memory embedder import chain must not require opencv.

    Pods without video dependencies (dashboard) import
    ``cogniverse_core.memory.mem0_embedder``, which pulls
    ``cogniverse_core.common.models`` — a package whose modules must keep
    ``cv2`` imports local to the functions that use it.
    """

    def test_mem0_embedder_imports_with_cv2_blocked(self):
        import subprocess
        import sys

        code = (
            "import sys\n"
            "class _BlockCv2:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'cv2' or name.startswith('cv2.'):\n"
            "            raise ImportError('cv2 blocked for this test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _BlockCv2())\n"
            "import cogniverse_core.common.models\n"
            "import cogniverse_core.memory.mem0_embedder\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert "IMPORT_OK" in result.stdout


class TestContradictionScanReadFailure:
    def test_conflict_state_read_failure_skips_persistence_and_warns(self, caplog):
        """When the existing-conflicts read fails, the subject's current
        state is unknown — persisting blind writes a duplicate conflict_set
        every time the same conflict re-surfaces. The write must be skipped
        with a warning, not silently treated as 'no existing conflicts'."""
        import logging
        from unittest.mock import MagicMock, patch

        from cogniverse_core.memory.contradiction import CONFLICT_AGENT_NAME
        from cogniverse_core.memory.manager import Mem0MemoryManager

        Mem0MemoryManager._instances.clear()
        mm = Mem0MemoryManager(tenant_id="p5_tenant")
        mm._initialized = True
        mm.tenant_id = "p5_tenant"
        mm.config = None
        mm._knowledge_registry = object()

        memory = MagicMock()

        def get_all(**kwargs):
            if kwargs.get("agent_id") == CONFLICT_AGENT_NAME:
                raise RuntimeError("vespa blip")
            return {"results": []}

        memory.get_all = get_all
        mm.memory = memory

        conflict = MagicMock()
        detector = MagicMock()
        detector.detect.return_value = [conflict]

        with (
            patch(
                "cogniverse_core.memory.contradiction.ContradictionDetector",
                return_value=detector,
            ),
            caplog.at_level(logging.WARNING),
        ):
            mm._detect_and_persist_contradictions(
                memory_id="m1",
                tenant_id="p5_tenant",
                agent_name="search_agent",
                metadata={"subject_key": "user:alice:city", "kind": "knowledge"},
                content="alice lives in Paris",
            )

        memory.add.assert_not_called()
        assert any(
            "user:alice:city" in rec.message and "vespa blip" in rec.message
            for rec in caplog.records
        ), (
            f"expected a warning naming the subject and error: {[r.message for r in caplog.records]}"
        )
