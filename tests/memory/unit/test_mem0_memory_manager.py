"""
Unit tests for Mem0MemoryManager
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.memory.manager import Mem0MemoryManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def affirm_memory_profile(config_manager):
    """Register the memory profile as the runtime's startup does.

    Memory init reads this profile and refuses to write one, so a manager
    initialised against a store nothing affirmed has no profile to resolve.
    """
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.memory.manager import (
        MEMORY_BASE_SCHEMA,
        MEMORY_EMBEDDING_DIMS,
        build_memory_profile,
    )
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(
            MEMORY_BASE_SCHEMA,
            build_memory_profile(MEMORY_BASE_SCHEMA, MEMORY_EMBEDDING_DIMS),
        ),
        tenant_id=SYSTEM_TENANT_ID,
        service="backend",
    )


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
        affirm_memory_profile(config_manager)
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
        assert manager._storage_tenant_id == canonical_tenant_id("test_tenant")
        mock_registry.get_ingestion_backend.assert_called_once()
        assert mock_registry.get_ingestion_backend.call_args.kwargs[
            "tenant_id"
        ] == canonical_tenant_id("test_tenant")
        mock_backend.get_tenant_schema_name.assert_called_once_with(
            canonical_tenant_id("test_tenant"), "agent_memories"
        )
        # Verify tenant-specific schema was used
        assert (
            manager.config["vector_store"]["config"]["collection_name"]
            == "agent_memories_test_tenant"
        )
        assert manager.config["vector_store"]["config"][
            "tenant_id"
        ] == canonical_tenant_id("test_tenant")
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
        affirm_memory_profile(config_manager)

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
        # The write partitions by the canonicalized per-call tenant - the
        # same derivation search_memory and get_all_memories read with, so a
        # write is visible to the read that names the same tenant.
        mock_memory.add.assert_called_once_with(
            "Test content",
            user_id=canonical_tenant_id("tenant1"),
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
            user_id=canonical_tenant_id("tenant1"),
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

    @staticmethod
    def _bind_deployed_partition(manager, mock_memory) -> MagicMock:
        """Wire the state initialize() leaves behind: a Mem0 handle, the
        vector-store profile, and a backend whose partition schema exists."""
        manager.memory = mock_memory
        manager.config = {"vector_store": {"config": {"profile": "agent_memories"}}}
        backend = MagicMock()
        backend.schema_exists.return_value = True
        # IngestionBackend.delete_document is declared to return a bool; a
        # bare MagicMock return would let a broken delete read as success.
        backend.delete_document.return_value = True
        manager._resolve_backend = lambda: backend
        return backend

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
        backend = self._bind_deployed_partition(manager, mock_memory)

        # Get all
        memories = manager.get_all_memories(
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert memories == [
            {"id": "mem_1", "text": "Memory 1"},
            {"id": "mem_2", "text": "Memory 2"},
        ]
        backend.schema_exists.assert_called_once_with(
            "agent_memories", tenant_id=canonical_tenant_id("test_tenant")
        )
        mock_memory.get_all.assert_called_once_with(
            user_id=canonical_tenant_id("tenant1"),
            agent_id="test_agent",
            filters=None,
            limit=100,
        )

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_all_memories_limit_none_walks_every_page(
        self, mock_memory_class, manager
    ):
        """limit=None forwards to mem0 as the whole-partition walk signal."""
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {"results": []}
        self._bind_deployed_partition(manager, mock_memory)

        manager.get_all_memories(
            tenant_id="tenant1",
            agent_name="test_agent",
            limit=None,
        )

        mock_memory.get_all.assert_called_once_with(
            user_id=canonical_tenant_id("tenant1"),
            agent_id="test_agent",
            filters=None,
            limit=None,
        )

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_all_memories_returns_empty_when_schema_missing(
        self, mock_memory_class, manager
    ):
        """A tenant partition with no deployed schema is empty, not an error."""
        mock_memory = MagicMock()
        mock_memory.get_all.side_effect = AssertionError(
            "schema guard should short-circuit before get_all"
        )
        manager.memory = mock_memory
        manager.config = {"vector_store": {"config": {"profile": "agent_memories"}}}

        mock_backend = MagicMock()
        mock_backend.schema_exists.return_value = False
        manager._resolve_backend = lambda: mock_backend

        memories = manager.get_all_memories(
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert memories == []
        mock_backend.schema_exists.assert_called_once_with(
            "agent_memories",
            tenant_id=canonical_tenant_id("test_tenant"),
        )
        mock_memory.get_all.assert_not_called()

    @patch("cogniverse_core.memory.manager.Memory")
    def test_cleanup_refuses_a_tenant_whose_partition_schema_is_gone(
        self, mock_memory_class, manager
    ):
        """A vanished partition reads as "no memories", so it must not delete.

        ``get_all_memories`` answers ``[]`` for an undeployed partition, which
        is how the pin enumeration reads; sweeping against that answer treats
        every pinned memory as unpinned.
        """
        from cogniverse_core.memory.manager import MemoryPartitionMissingError
        from cogniverse_core.memory.schema import build_default_registry

        mock_memory = MagicMock()
        manager.memory = mock_memory
        manager.config = {"vector_store": {"config": {"profile": "agent_memories"}}}

        mock_backend = MagicMock()
        mock_backend.schema_exists.return_value = False
        manager._resolve_backend = lambda: mock_backend

        with pytest.raises(MemoryPartitionMissingError) as caught:
            manager.cleanup_with_schema(build_default_registry(), {"pinned_1"})

        assert str(caught.value) == (
            "memory partition schema for tenant test_tenant:test_tenant is not "
            "deployed; retention cannot tell a pinned memory from an absent one"
        )
        mock_memory.get_all.assert_not_called()
        mock_memory.delete.assert_not_called()

    @patch("cogniverse_core.memory.manager.Memory")
    def test_tenant_partition_schema_exists(self, mock_memory_class, manager):
        """The public schema-exists predicate forwards the backend verdict."""
        mock_memory = MagicMock()
        manager.memory = mock_memory
        manager.config = {"vector_store": {"config": {"profile": "agent_memories"}}}

        mock_backend = MagicMock()
        mock_backend.schema_exists.return_value = True
        manager._resolve_backend = lambda: mock_backend

        assert (
            manager.tenant_partition_schema_exists(canonical_tenant_id("tenant1"))
            is True
        )
        mock_backend.schema_exists.assert_called_once_with(
            "agent_memories",
            tenant_id=canonical_tenant_id("tenant1"),
        )

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_all_memories_propagates_schema_lookup_failure(
        self, mock_memory_class, manager
    ):
        """A schema lookup failure is not "no memories" and must raise."""
        mock_memory = MagicMock()
        mock_memory.get_all.side_effect = AssertionError(
            "schema lookup failure should short-circuit before get_all"
        )
        manager.memory = mock_memory
        manager.config = {"vector_store": {"config": {"profile": "agent_memories"}}}

        mock_backend = MagicMock()
        mock_backend.schema_exists.side_effect = ConnectionError(
            "schema registry unavailable"
        )
        manager._resolve_backend = lambda: mock_backend

        with pytest.raises(ConnectionError, match="schema registry unavailable"):
            manager.get_all_memories(tenant_id="tenant1", agent_name="test_agent")

        mock_backend.schema_exists.assert_called_once_with(
            "agent_memories",
            tenant_id=canonical_tenant_id("test_tenant"),
        )
        mock_memory.get_all.assert_not_called()

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_all_memories_raises_on_backend_outage(
        self, mock_memory_class, manager
    ):
        """A store outage must propagate, never flatten to [] — a caller
        reading [] would treat an outage as an empty partition."""
        import pytest

        mock_memory = MagicMock()
        mock_memory.get_all.side_effect = ConnectionError("vespa unreachable")
        self._bind_deployed_partition(manager, mock_memory)

        with pytest.raises(ConnectionError, match="vespa unreachable"):
            manager.get_all_memories(tenant_id="tenant1", agent_name="test_agent")

    @patch("cogniverse_core.memory.manager.Memory")
    def test_delete_memory(self, mock_memory_class, manager):
        """Test deleting memory"""
        # Setup
        mock_memory = MagicMock()
        backend = self._bind_deployed_partition(manager, mock_memory)

        # Delete
        success = manager.delete_memory(
            memory_id="mem_123",
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert success is True
        # Implementation only passes memory_id (tenant_id and agent_name not used)
        mock_memory.delete.assert_called_once_with("mem_123")
        # The indexed provenance row goes with the primary: a row left behind
        # is an orphan that every later citation read rejects.
        backend.delete_document.assert_called_once_with(
            f"prov-{canonical_tenant_id('test_tenant')}-mem_123",
            schema_name="provenance",
        )

    @patch("cogniverse_core.memory.manager.Memory")
    def test_clear_agent_memory(self, mock_memory_class, manager):
        """Test clearing all agent memory"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [
            {"id": "mem_1"},
            {"id": "mem_2"},
        ]
        self._bind_deployed_partition(manager, mock_memory)

        # Clear
        success = manager.clear_agent_memory(
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert success is True
        assert mock_memory.delete.call_count == 2

    @patch("cogniverse_core.memory.manager.Memory")
    def test_get_all_memories_cross_partition_uses_managers_own_schema(
        self, mock_memory_class, manager
    ):
        """The storage schema belongs to the manager's binding; a read scoped
        to another partition (the org-strategy read pattern, and seeded test
        tenants) must not consult a schema derived from the passed partition
        id - that schema never exists and the guard silently returned []."""
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [{"id": "m1", "memory": "s"}]
        backend = self._bind_deployed_partition(manager, mock_memory)

        rows = manager.get_all_memories(tenant_id="filter_test_123", agent_name="ns")

        assert [r["id"] for r in rows] == ["m1"]
        backend.schema_exists.assert_called_once_with(
            "agent_memories", tenant_id=canonical_tenant_id("test_tenant")
        )
        mock_memory.get_all.assert_called_once_with(
            user_id=canonical_tenant_id("filter_test_123"),
            agent_id="ns",
            filters=None,
            limit=100,
        )

    def test_delete_memory_returns_false_only_on_not_found(self, manager):
        """mem0 raises ValueError for a genuinely missing id; that maps to
        False. Anything else is a backend failure and must propagate."""
        mock_memory = MagicMock()
        mock_memory.delete.side_effect = ValueError("Memory with id mem_123 not found")
        manager.memory = mock_memory

        assert (
            manager.delete_memory(
                memory_id="mem_123", tenant_id="tenant1", agent_name="a"
            )
            is False
        )

    def test_delete_memory_raises_on_backend_outage(self, manager):
        """A backend outage must never read as not-found: the admin route
        turned the swallowed False into a definitive 404 for a memory that
        still exists."""
        mock_memory = MagicMock()
        mock_memory.delete.side_effect = RuntimeError(
            "Vespa delete failed: connection refused"
        )
        manager.memory = mock_memory

        with pytest.raises(RuntimeError, match="connection refused"):
            manager.delete_memory(
                memory_id="mem_123", tenant_id="tenant1", agent_name="a"
            )

    @patch("cogniverse_core.memory.manager.Memory")
    def test_clear_agent_memory_raises_on_backend_outage(
        self, mock_memory_class, manager
    ):
        """An outage mid-clear must propagate, not return the False the admin
        route discarded while reporting {"status": "cleared"}."""
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [{"id": "mem_1"}, {"id": "mem_2"}]
        mock_memory.delete.side_effect = RuntimeError(
            "Vespa delete failed: connection refused"
        )
        self._bind_deployed_partition(manager, mock_memory)

        with pytest.raises(RuntimeError, match="connection refused"):
            manager.clear_agent_memory(tenant_id="tenant1", agent_name="test_agent")

    @patch("cogniverse_core.memory.manager.Memory")
    def test_clear_agent_memory_treats_missing_as_cleared(
        self, mock_memory_class, manager
    ):
        """A concurrent deletion between the listing and the delete is the
        goal state, not a failure: clear still reports success and attempts
        every listed id."""
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = [{"id": "mem_1"}, {"id": "mem_2"}]
        mock_memory.delete.side_effect = [
            ValueError("Memory with id mem_1 not found"),
            None,
        ]
        self._bind_deployed_partition(manager, mock_memory)

        assert (
            manager.clear_agent_memory(tenant_id="tenant1", agent_name="test_agent")
            is True
        )
        assert mock_memory.delete.call_count == 2

    @patch("cogniverse_core.memory.manager.Memory")
    def test_update_memory(self, mock_memory_class, manager):
        """Test updating memory"""
        # Setup
        mock_memory = MagicMock()
        mock_memory.get.return_value = {
            "id": "mem_123",
            "memory": "Original content",
            "metadata": {},
        }
        manager.memory = mock_memory

        # Update
        success = manager.update_memory(
            memory_id="mem_123",
            content="Updated content",
            tenant_id="tenant1",
            agent_name="test_agent",
        )

        assert success is True
        mock_memory.update.assert_called_once_with(
            "mem_123",
            data="Updated content",
            metadata={},
        )

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
        self._bind_deployed_partition(manager, mock_memory)

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
            manager = ConfigManager(store=store)
            affirm_memory_profile(manager)
            return manager

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


class TestSearchMemoryFaultContract:
    """A backend outage during search must raise — flattening it to [] reads
    as "no relevant memories" and agents silently run without memory context.
    get_all_memories already follows this contract; the two read paths must
    agree."""

    @pytest.mark.unit
    def test_search_memory_raises_on_backend_failure(self):
        manager = Mem0MemoryManager(tenant_id="tenant1")
        mock_memory = MagicMock()
        mock_memory.search.side_effect = ConnectionError("backend down")
        manager.memory = mock_memory

        with pytest.raises(ConnectionError):
            manager.search_memory(query="q", tenant_id="tenant1", agent_name="agent")

    @pytest.mark.unit
    def test_search_memory_empty_result_still_empty(self):
        manager = Mem0MemoryManager(tenant_id="tenant1")
        mock_memory = MagicMock()
        mock_memory.search.return_value = []
        manager.memory = mock_memory

        assert (
            manager.search_memory(query="q", tenant_id="tenant1", agent_name="a") == []
        )


def test_memory_init_refuses_to_register_the_profile_itself():
    """Memory init reads its backend profile; it does not write one.

    It used to register the profile into the system tenant's backend config
    when the profile looked missing — and it always looked missing, because it
    read ``profiles`` at the top level of the config where only
    ``backend.profiles`` exists. Every new tenant's first request then
    read-merged-wrote-pruned that document on the serving path.
    """
    from cogniverse_core.memory.manager import (
        MEMORY_BASE_SCHEMA,
        MemoryProfileMissingError,
    )

    manager = Mem0MemoryManager(tenant_id="profile:missing")
    config_manager = MagicMock()
    unregistered = MagicMock()
    unregistered.get.side_effect = lambda key, default=None: (
        {"profiles": {}} if key == "backend" else default
    )

    with patch(
        "cogniverse_foundation.config.utils.get_config", return_value=unregistered
    ):
        with pytest.raises(MemoryProfileMissingError, match=MEMORY_BASE_SCHEMA):
            manager._build_and_store_memory(
                backend_host="http://localhost",
                backend_port=8080,
                llm_model="m",
                embedding_model="lightonai/DenseOn",
                llm_base_url="http://lm",
                embedder_base_url="http://denseon",
                config_manager=config_manager,
                schema_loader=MagicMock(),
                llm_api_key="k",
                backend_config_port=19071,
                base_schema_name=MEMORY_BASE_SCHEMA,
                auto_create_schema=False,
                embedding_dims=768,
                knowledge_registry=None,
                fingerprint=(),
                storage_tenant_id="profile:missing",
            )

    assert config_manager.add_backend_profile.call_args_list == []


class TestProvenanceWriteLeaseScope:
    """The provenance write lease costs a cluster-wide per-tenant mutex.

    It exists to keep a primary and its indexed provenance row consistent, so
    it belongs only on operations that have both. Sized for a memory write,
    not for a Vespa application-package activation, and recoverable: a record
    left behind by a holder that died must not outlive it.
    """

    @staticmethod
    def _manager(tenant_id: str):
        from tests.utils.memory_store import InMemoryConfigStore

        Mem0MemoryManager._instances.pop(tenant_id, None)
        manager = Mem0MemoryManager(tenant_id=tenant_id)
        manager._initialized = True
        manager.tenant_id = tenant_id
        manager.config = None
        manager._knowledge_registry = None
        manager.memory = MagicMock()
        manager._provenance_store = MagicMock()
        manager._provenance_lease_store = InMemoryConfigStore()
        return manager

    @staticmethod
    def _lease_record(manager):
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        return manager._provenance_lease_store.get_config(
            tenant_id="__system__",
            scope=ConfigScope.SCHEMA,
            service="provenance_write_lease",
            config_key=manager._storage_tenant_id,
        )

    @staticmethod
    def _provenance_metadata():
        from cogniverse_core.memory.provenance import (
            CitationRef,
            DerivationKind,
            attach_to_metadata,
            make_provenance,
        )

        return attach_to_metadata(
            {"kind": "entity_fact"},
            make_provenance(
                written_by="agent:lease-scope",
                derivation_kind=DerivationKind.SYNTHESIS,
                confidence=0.5,
                derived_from=[CitationRef.external("https://source.test/scope")],
            ),
        )

    def test_a_provenance_free_add_takes_no_store_lease(self):
        """Conversation turns and agent remembers carry no provenance, so the
        lease can only cost them a cluster round trip and a global mutex."""
        manager = self._manager("lease_scope_tenant")
        manager.memory.add.return_value = {"results": [{"id": "m1", "event": "ADD"}]}

        assert (
            manager.add_memory(
                content="a conversation turn",
                tenant_id="lease_scope_tenant",
                agent_name="conversation",
                metadata={"type": "conversation"},
                infer=False,
            )
            == "m1"
        )
        assert self._lease_record(manager) is None

    def test_a_provenance_free_update_takes_no_store_lease(self):
        manager = self._manager("lease_scope_update_tenant")
        manager.memory.get.return_value = {"id": "m1", "memory": "before"}

        assert (
            manager.update_memory(
                memory_id="m1",
                content="after",
                tenant_id="lease_scope_update_tenant",
                agent_name="agent",
                metadata={"kind": "note"},
            )
            is True
        )
        assert self._lease_record(manager) is None

    def test_a_provenance_bearing_add_holds_a_memory_sized_lease(self):
        """The hold covers a memory write with margin; the wait exceeds the
        hold, so a stalled holder can always be waited out within one wait."""
        from cogniverse_core.memory import manager as manager_module

        assert (
            manager_module.PROVENANCE_WAIT_SECONDS
            > manager_module.PROVENANCE_LEASE_SECONDS
        )
        assert manager_module.PROVENANCE_LEASE_SECONDS < 600.0

        manager = self._manager("lease_hold_tenant")
        manager.memory.add.return_value = {"results": [{"id": "m2", "event": "ADD"}]}
        manager.memory.get.return_value = {"id": "m2", "memory": "content"}
        held = {}

        def record_hold(*args, **kwargs):
            held["record"] = self._lease_record(manager).config_value
            return "prov-row"

        manager._provenance_store.attach.side_effect = record_hold

        assert (
            manager.add_memory(
                content="content",
                tenant_id="lease_hold_tenant",
                agent_name="agent",
                metadata=self._provenance_metadata(),
                infer=False,
            )
            == "m2"
        )
        assert held["record"]["lease_seconds"] == (
            manager_module.PROVENANCE_LEASE_SECONDS
        )
        assert self._lease_record(manager).config_value["holder"] is None

    def test_a_provenance_write_recovers_a_lease_abandoned_by_a_dead_holder(self):
        """The lease must not be a one-way door: a record left behind by a
        holder that died has to be recoverable by the next writer."""
        import os
        import socket
        import uuid

        from cogniverse_sdk.interfaces.config_store import ConfigScope

        manager = self._manager("lease_recover_tenant")
        abandoned = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        manager._provenance_lease_store.compare_and_set_config(
            tenant_id="__system__",
            scope=ConfigScope.SCHEMA,
            service="provenance_write_lease",
            config_key=manager._storage_tenant_id,
            config_value={"holder": abandoned, "lease_seconds": 600.0},
            expected_version=0,
        )
        manager.memory.add.return_value = {"results": [{"id": "m3", "event": "ADD"}]}
        manager.memory.get.return_value = {"id": "m3", "memory": "content"}
        manager._provenance_store.attach.return_value = "prov-row"

        assert (
            manager.add_memory(
                content="content",
                tenant_id="lease_recover_tenant",
                agent_name="agent",
                metadata=self._provenance_metadata(),
                infer=False,
            )
            == "m3"
        )
        assert self._lease_record(manager).config_value["holder"] is None

    def test_a_stalled_foreign_holder_is_waited_out_within_one_wait(self, monkeypatch):
        """No liveness proof exists for another node's holder, so the wait
        itself has to outlast the hold. Scaled down, same inequality."""
        import uuid

        from cogniverse_core.memory import manager as manager_module
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        monkeypatch.setattr(manager_module, "PROVENANCE_LEASE_SECONDS", 0.4)
        monkeypatch.setattr(manager_module, "PROVENANCE_WAIT_SECONDS", 5.0)

        manager = self._manager("lease_foreign_tenant")
        foreign = f"another-host:4242:{uuid.uuid4().hex}"
        manager._provenance_lease_store.compare_and_set_config(
            tenant_id="__system__",
            scope=ConfigScope.SCHEMA,
            service="provenance_write_lease",
            config_key=manager._storage_tenant_id,
            config_value={"holder": foreign, "lease_seconds": 0.4},
            expected_version=0,
        )
        manager.memory.add.return_value = {"results": [{"id": "m4", "event": "ADD"}]}
        manager.memory.get.return_value = {"id": "m4", "memory": "content"}
        manager._provenance_store.attach.return_value = "prov-row"

        assert (
            manager.add_memory(
                content="content",
                tenant_id="lease_foreign_tenant",
                agent_name="agent",
                metadata=self._provenance_metadata(),
                infer=False,
            )
            == "m4"
        )

    def test_a_clear_sweep_acquires_per_row_instead_of_once_for_the_sweep(self):
        """One lease for a whole retention sweep excludes every other writer
        for the sweep's duration; per row it excludes them per row."""
        manager = self._manager("lease_sweep_tenant")
        manager.tenant_partition_schema_exists = lambda *args, **kwargs: True
        manager.memory.get_all.return_value = {
            "results": [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]
        }
        holders = []

        def record_holder(memory_id):
            holders.append(self._lease_record(manager).config_value["holder"])

        manager.memory.delete.side_effect = record_holder

        assert (
            manager.clear_agent_memory(
                tenant_id="lease_sweep_tenant", agent_name="agent"
            )
            is True
        )
        assert len(holders) == 3
        assert len(set(holders)) == 3
        assert None not in holders
        assert self._lease_record(manager).config_value["holder"] is None

    def test_update_surfaces_a_provenance_write_failure_it_cannot_undo(self):
        """The primary is already rewritten when attach fails; reporting
        False says nothing happened, and the index now disagrees with it."""
        from cogniverse_core.memory.provenance_store import ProvenanceWriteError

        manager = self._manager("lease_update_fail_tenant")
        manager.memory.get.return_value = {"id": "m5", "memory": "after"}
        manager._provenance_store.attach.side_effect = ProvenanceWriteError(
            memory_id="m5", row_id="prov-row"
        )

        with pytest.raises(ProvenanceWriteError) as caught:
            manager.update_memory(
                memory_id="m5",
                content="after",
                tenant_id="lease_update_fail_tenant",
                agent_name="agent",
                metadata=self._provenance_metadata(),
            )
        assert caught.value.memory_id == "m5"
        manager.memory.update.assert_called_once()

    def test_an_update_of_a_provenance_bearing_primary_takes_the_lease(self):
        """Dropping a stored primary's provenance still changes what its
        indexed row has to agree with, so repair must stay excluded."""
        import os

        from cogniverse_core.memory.manager import PROVENANCE_LEASE_SECONDS

        manager = self._manager("lease_update_declared_tenant")
        manager.memory.get.return_value = {
            "id": "m6",
            "memory": "before",
            "metadata": self._provenance_metadata(),
        }
        held = {}

        def record_hold(*args, **kwargs):
            held["record"] = self._lease_record(manager).config_value

        manager.memory.update.side_effect = record_hold

        assert (
            manager.update_memory(
                memory_id="m6",
                content="after",
                tenant_id="lease_update_declared_tenant",
                agent_name="agent",
                metadata={"kind": "note"},
            )
            is True
        )
        assert held["record"]["lease_seconds"] == PROVENANCE_LEASE_SECONDS
        assert str(os.getpid()) in held["record"]["holder"].split(":")
        manager._provenance_store.attach.assert_not_called()
        assert self._lease_record(manager).config_value["holder"] is None

    def test_an_unreadable_primary_is_updated_under_the_lease(self):
        """An outage on the before-read cannot prove the primary carries no
        provenance; the update keeps its False contract and stays leased."""
        manager = self._manager("lease_update_outage_tenant")
        manager.memory.get.side_effect = ConnectionError("vespa unreachable")

        assert (
            manager.update_memory(
                memory_id="m7",
                content="after",
                tenant_id="lease_update_outage_tenant",
                agent_name="agent",
                metadata={"kind": "note"},
            )
            is False
        )
        assert self._lease_record(manager).config_value["holder"] is None
        manager.memory.update.assert_not_called()


class TestAddMemoryErrorPrecedence:
    """Malformed input reports the first contract it breaks, as before the
    provenance lease scope was decided ahead of the write."""

    @staticmethod
    def _malformed_metadata():
        return {"kind": "entity_fact", "provenance": "not-a-provenance-block"}

    def test_an_uninitialized_manager_reports_that_first(self):
        Mem0MemoryManager._instances.pop("precedence_uninit_tenant", None)
        manager = Mem0MemoryManager(tenant_id="precedence_uninit_tenant")
        manager.memory = None

        with pytest.raises(RuntimeError, match="Mem0MemoryManager not initialized"):
            manager.add_memory(
                content="content",
                tenant_id="precedence_uninit_tenant",
                agent_name="agent",
                metadata=self._malformed_metadata(),
                infer=False,
            )

    def test_a_schema_violation_wins_over_a_malformed_provenance_block(self):
        from cogniverse_core.memory.provenance_store import ProvenanceWriteError
        from cogniverse_core.memory.schema import (
            KnowledgeSchema,
            SchemaViolationError,
        )
        from tests.utils.memory_store import InMemoryConfigStore

        Mem0MemoryManager._instances.pop("precedence_schema_tenant", None)
        manager = Mem0MemoryManager(tenant_id="precedence_schema_tenant")
        manager._initialized = True
        manager.config = None
        manager.memory = MagicMock()
        manager._provenance_store = MagicMock()
        manager._provenance_lease_store = InMemoryConfigStore()
        registry = MagicMock()
        registry.get.return_value = KnowledgeSchema(
            kind="entity_fact", provenance_required=True
        )
        manager._knowledge_registry = registry

        with pytest.raises(SchemaViolationError) as caught:
            manager.add_memory(
                content="content",
                tenant_id="precedence_schema_tenant",
                agent_name="agent",
                metadata=self._malformed_metadata(),
                infer=False,
            )
        assert not isinstance(caught.value, ProvenanceWriteError)
        manager.memory.add.assert_not_called()


class TestRepairOfAMalformedPrimary:
    def test_repair_reports_a_malformed_primary_as_inconsistent(self):
        """Repair reads stored state; a bad stored payload is torn provenance,
        not a failed write of the caller's own request."""
        from cogniverse_core.memory.provenance import ProvenanceConsistencyError
        from tests.utils.memory_store import InMemoryConfigStore

        Mem0MemoryManager._instances.pop("repair_malformed_tenant", None)
        manager = Mem0MemoryManager(tenant_id="repair_malformed_tenant")
        manager._initialized = True
        manager.config = None
        manager._knowledge_registry = None
        manager.memory = MagicMock()
        manager.memory.get.return_value = {
            "id": "m8",
            "memory": "stored",
            "metadata": {"kind": "entity_fact", "provenance": {"written_by": 7}},
        }
        manager._provenance_store = MagicMock()
        manager._provenance_lease_store = InMemoryConfigStore()

        with pytest.raises(
            ProvenanceConsistencyError, match="primary provenance payload is malformed"
        ) as caught:
            manager.repair_provenance("m8")
        assert caught.value.memory_id == "m8"
        manager._provenance_store.attach.assert_not_called()
