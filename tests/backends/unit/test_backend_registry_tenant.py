"""
Unit tests for tenant-aware BackendRegistry.

Tests that:
- Search backends are shared across tenants (single instance, tenant_id in query_dict)
- Ingestion backends are properly isolated per tenant
"""

import pytest

from cogniverse_core.registries.backend_registry import (
    get_backend_registry,
)
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_sdk.interfaces.backend import IngestionBackend, SearchBackend


class MockSearchBackend(SearchBackend):
    """Mock search backend for testing"""

    def __init__(self, backend_config, schema_loader, config_manager):
        self.initialized = False
        self.config = None
        self.backend_config = backend_config
        self.schema_loader = schema_loader
        self.config_manager = config_manager
        # Mock schema_registry attribute (will be injected by BackendRegistry)
        self.schema_registry = None

    def initialize(self, config: dict):
        self.initialized = True
        self.config = config

    def search(self, query: str, **kwargs):
        return []

    def get_document(self, doc_id: str):
        return None

    def batch_get_documents(self, doc_ids: list):
        return []

    def health_check(self) -> bool:
        return True

    def get_statistics(self) -> dict:
        return {}

    def get_embedding_requirements(self, schema_name: str) -> dict:
        """Mock implementation of get_embedding_requirements"""
        return {
            "needs_float": True,
            "needs_binary": False,
            "float_field": "embedding",
            "binary_field": "embedding_binary",
        }


class MockIngestionBackend(IngestionBackend):
    """Mock ingestion backend for testing"""

    def __init__(self, backend_config, schema_loader, config_manager):
        self.initialized = False
        self.config = None
        self.backend_config = backend_config
        self.schema_loader = schema_loader
        self.config_manager = config_manager

    def initialize(self, config: dict):
        self.initialized = True
        self.config = config

    def ingest_documents(self, documents: list):
        pass

    def ingest_stream(self, document_stream):
        pass

    def update_document(self, doc_id: str, updates: dict):
        pass

    def delete_document(self, doc_id: str):
        pass

    def validate_schema(self, schema: dict) -> bool:
        return True

    def get_schema_info(self) -> dict:
        return {}


class TestBackendRegistrySearchShared:
    """Test that search backends are shared across tenants"""

    @pytest.fixture
    def config_manager(self, backend_config_env):
        """Create a config_manager for testing"""
        return create_default_config_manager()

    @pytest.fixture
    def schema_loader(self):
        """Create a schema_loader for testing"""
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        return FilesystemSchemaLoader(Path("configs/schemas"))

    def setup_method(self):
        """Save original backends and clear instances before each test"""
        registry = get_backend_registry()
        # Save original backend registrations (e.g., Vespa auto-registered on import)
        self._saved_ingestion = registry._ingestion_backends.copy()
        self._saved_search = registry._search_backends.copy()
        self._saved_full = registry._full_backends.copy()
        # Clear instances but keep backend registrations
        registry.clear_instances()

    def teardown_method(self):
        """Restore original backends and clear instances after each test"""
        registry = get_backend_registry()
        registry.clear_instances()
        # Restore original backend registrations (don't delete real backends!)
        registry._ingestion_backends = self._saved_ingestion
        registry._search_backends = self._saved_search
        registry._full_backends = self._saved_full

    def test_search_backend_shared_instance(self, config_manager, schema_loader):
        """Test that get_search_backend returns the same shared instance"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        backend_1 = registry.get_search_backend(
            "mock",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        backend_2 = registry.get_search_backend(
            "mock",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Should be the same shared instance
        assert backend_1 is backend_2
        assert id(backend_1) == id(backend_2)

    def test_search_cache_key_has_no_tenant(self, config_manager, schema_loader):
        """Test that search cache key is 'search_{name}' without tenant suffix"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        registry.get_search_backend(
            "mock",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        cache_keys = list(registry._backend_instances.keys())
        assert "search_mock" in cache_keys
        # No tenant suffix in any search key
        assert not any("tenant" in key for key in cache_keys)

    def test_multiple_search_backend_types(self, config_manager, schema_loader):
        """Test that different backend types get separate shared instances"""
        registry = get_backend_registry()
        registry.register_search("search1", MockSearchBackend)
        registry.register_search("search2", MockSearchBackend)

        backend1 = registry.get_search_backend(
            "search1",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        backend2 = registry.get_search_backend(
            "search2",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Different types should be different instances
        assert backend1 is not backend2
        assert len(registry._backend_instances) == 2

    def test_clear_instances_removes_shared_search(self, config_manager, schema_loader):
        """Test that clear_instances removes the shared search backend"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        registry.get_search_backend(
            "mock",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        assert len(registry._backend_instances) == 1

        registry.clear_instances()

        assert len(registry._backend_instances) == 0

    def test_search_backend_no_tenant_id_in_init_config(
        self, config_manager, schema_loader
    ):
        """Test that search backend init config does not contain tenant_id"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        backend = registry.get_search_backend(
            "mock",
            config={"test_key": "test_value"},
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Config should NOT have tenant_id injected
        assert "tenant_id" not in backend.config

    def test_backend_not_found_error_message(self, config_manager, schema_loader):
        """Test that helpful error is raised for unknown backend"""
        registry = get_backend_registry()

        with pytest.raises(ValueError, match="not found"):
            registry.get_search_backend(
                "nonexistent",
                config_manager=config_manager,
                schema_loader=schema_loader,
            )


class TestBackendRegistryIngestionTenantIsolation:
    """Test tenant isolation for ingestion backends (unchanged)"""

    @pytest.fixture
    def config_manager(self, backend_config_env):
        """Create a config_manager for testing"""
        return create_default_config_manager()

    @pytest.fixture
    def schema_loader(self):
        """Create a schema_loader for testing"""
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        return FilesystemSchemaLoader(Path("configs/schemas"))

    def setup_method(self):
        """Save original backends and clear instances before each test"""
        registry = get_backend_registry()
        self._saved_ingestion = registry._ingestion_backends.copy()
        self._saved_search = registry._search_backends.copy()
        self._saved_full = registry._full_backends.copy()
        registry.clear_instances()

    def teardown_method(self):
        """Restore original backends and clear instances after each test"""
        registry = get_backend_registry()
        registry.clear_instances()
        registry._ingestion_backends = self._saved_ingestion
        registry._search_backends = self._saved_search
        registry._full_backends = self._saved_full

    def test_tenant_id_required_for_ingestion_backend(self):
        """Test that tenant_id is required for get_ingestion_backend"""
        registry = get_backend_registry()
        registry.register_ingestion("mock", MockIngestionBackend)

        with pytest.raises(ValueError, match="tenant_id is required"):
            registry.get_ingestion_backend("mock", tenant_id="")

    def test_ingestion_backend_tenant_isolation(self, config_manager, schema_loader):
        """Test tenant isolation for ingestion backends"""
        registry = get_backend_registry()
        registry.register_ingestion("mock", MockIngestionBackend)

        backend_a = registry.get_ingestion_backend(
            "mock",
            tenant_id="tenant_a",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        backend_b = registry.get_ingestion_backend(
            "mock",
            tenant_id="tenant_b",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Should be different instances
        assert backend_a is not backend_b

    def test_full_backend_tenant_isolation(self, config_manager, schema_loader):
        """Test tenant isolation for full backends (both search and ingestion)"""

        class MockFullBackend(SearchBackend, IngestionBackend):
            def __init__(self, backend_config, schema_loader, config_manager):
                self.initialized = False
                self.backend_config = backend_config
                self.schema_loader = schema_loader
                self.config_manager = config_manager
                self.schema_registry = None

            def initialize(self, config: dict):
                self.initialized = True

            def search(self, query: str, **kwargs):
                return []

            def get_document(self, doc_id: str):
                return None

            def batch_get_documents(self, doc_ids: list):
                return []

            def health_check(self) -> bool:
                return True

            def get_statistics(self) -> dict:
                return {}

            def get_embedding_requirements(self, schema_name: str) -> dict:
                return {
                    "needs_float": True,
                    "needs_binary": False,
                    "float_field": "embedding",
                    "binary_field": "embedding_binary",
                }

            def ingest_documents(self, documents: list):
                pass

            def ingest_stream(self, document_stream):
                pass

            def update_document(self, doc_id: str, updates: dict):
                pass

            def delete_document(self, doc_id: str):
                pass

            def validate_schema(self, schema: dict) -> bool:
                return True

            def get_schema_info(self) -> dict:
                return {}

        registry = get_backend_registry()
        registry.register_backend("mock_full", MockFullBackend)

        # Get as ingestion backend for tenant A
        ingest_a = registry.get_ingestion_backend(
            "mock_full",
            tenant_id="tenant_a",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Get as ingestion backend for tenant B (should be different)
        ingest_b = registry.get_ingestion_backend(
            "mock_full",
            tenant_id="tenant_b",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        assert ingest_a is not ingest_b


class TestBackendRegistrySingleton:
    """Test singleton behavior"""

    @pytest.fixture
    def config_manager(self, backend_config_env):
        """Create a config_manager for testing"""
        return create_default_config_manager()

    @pytest.fixture
    def schema_loader(self):
        """Create a schema_loader for testing"""
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        return FilesystemSchemaLoader(Path("configs/schemas"))

    def setup_method(self):
        """Save original backends and clear instances before each test"""
        registry = get_backend_registry()
        self._saved_ingestion = registry._ingestion_backends.copy()
        self._saved_search = registry._search_backends.copy()
        self._saved_full = registry._full_backends.copy()
        registry.clear_instances()

    def teardown_method(self):
        """Restore original backends and clear instances after each test"""
        registry = get_backend_registry()
        registry.clear_instances()
        registry._ingestion_backends = self._saved_ingestion
        registry._search_backends = self._saved_search
        registry._full_backends = self._saved_full

    def test_get_backend_registry_returns_singleton(self):
        """Test that get_backend_registry always returns same instance"""
        registry1 = get_backend_registry()
        registry2 = get_backend_registry()

        assert registry1 is registry2

    def test_singleton_preserves_shared_search_cache(
        self, config_manager, schema_loader
    ):
        """Test that singleton preserves the shared search backend cache"""
        registry1 = get_backend_registry()
        registry1.register_search("mock", MockSearchBackend)
        registry1.get_search_backend(
            "mock",
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        registry2 = get_backend_registry()

        # Should have the cached instance
        assert len(registry2._backend_instances) == 1
