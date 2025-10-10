"""
Unit tests for tenant-aware BackendRegistry.

Tests that backend instances are properly isolated per tenant.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from cogniverse_core.registries.backend_registry import (
    BackendRegistry,
    get_backend_registry,
)
from cogniverse_core.interfaces.backend import SearchBackend, IngestionBackend


class MockSearchBackend(SearchBackend):
    """Mock search backend for testing"""

    def __init__(self):
        self.initialized = False
        self.config = None

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


class MockIngestionBackend(IngestionBackend):
    """Mock ingestion backend for testing"""

    def __init__(self):
        self.initialized = False
        self.config = None

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


class TestBackendRegistryTenantIsolation:
    """Test tenant-scoped backend caching"""

    def setup_method(self):
        """Clear registry before each test"""
        registry = get_backend_registry()
        registry.clear_instances()
        registry._ingestion_backends.clear()
        registry._search_backends.clear()
        registry._full_backends.clear()

    def test_tenant_id_required_for_search_backend(self):
        """Test that tenant_id is required for get_search_backend"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        with pytest.raises(ValueError, match="tenant_id is required"):
            registry.get_search_backend("mock", tenant_id="")

    def test_tenant_id_required_for_ingestion_backend(self):
        """Test that tenant_id is required for get_ingestion_backend"""
        registry = get_backend_registry()
        registry.register_ingestion("mock", MockIngestionBackend)

        with pytest.raises(ValueError, match="tenant_id is required"):
            registry.get_ingestion_backend("mock", tenant_id="")

    def test_different_tenants_get_different_instances(self):
        """Test that different tenants get separate backend instances"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        # Get backend for tenant A
        backend_a = registry.get_search_backend("mock", tenant_id="tenant_a")

        # Get backend for tenant B
        backend_b = registry.get_search_backend("mock", tenant_id="tenant_b")

        # Should be different instances
        assert backend_a is not backend_b
        assert id(backend_a) != id(backend_b)

    def test_same_tenant_gets_cached_instance(self):
        """Test that the same tenant gets the cached instance"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        # Get backend for tenant A
        backend_1 = registry.get_search_backend("mock", tenant_id="tenant_a")

        # Get backend again for tenant A
        backend_2 = registry.get_search_backend("mock", tenant_id="tenant_a")

        # Should be the same instance
        assert backend_1 is backend_2
        assert id(backend_1) == id(backend_2)

    def test_cache_key_includes_tenant_id(self):
        """Test that cache keys include tenant_id"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        # Get backends for two tenants
        registry.get_search_backend("mock", tenant_id="tenant_a")
        registry.get_search_backend("mock", tenant_id="tenant_b")

        # Check cache keys
        cache_keys = list(registry._backend_instances.keys())
        assert any("tenant_a" in key for key in cache_keys)
        assert any("tenant_b" in key for key in cache_keys)

    def test_multiple_backends_per_tenant(self):
        """Test that a tenant can have multiple different backends"""
        registry = get_backend_registry()
        registry.register_search("search1", MockSearchBackend)
        registry.register_search("search2", MockSearchBackend)

        tenant_id = "tenant_a"

        # Get two different backends for same tenant
        backend1 = registry.get_search_backend("search1", tenant_id=tenant_id)
        backend2 = registry.get_search_backend("search2", tenant_id=tenant_id)

        # Should be different instances
        assert backend1 is not backend2

        # Both should be cached
        assert len(registry._backend_instances) == 2

    def test_tenant_isolation_with_configuration(self):
        """Test that tenant-specific configurations are isolated"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        config_a = {"tenant_setting": "value_a"}
        config_b = {"tenant_setting": "value_b"}

        # Get backend for tenant A with config A
        backend_a = registry.get_search_backend(
            "mock", tenant_id="tenant_a", config=config_a
        )

        # Get backend for tenant B with config B
        backend_b = registry.get_search_backend(
            "mock", tenant_id="tenant_b", config=config_b
        )

        # Each should have their own config
        assert backend_a.config == config_a
        assert backend_b.config == config_b

    def test_clear_instances_removes_all_tenants(self):
        """Test that clear_instances removes all tenant caches"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        # Create instances for multiple tenants
        registry.get_search_backend("mock", tenant_id="tenant_a")
        registry.get_search_backend("mock", tenant_id="tenant_b")
        registry.get_search_backend("mock", tenant_id="tenant_c")

        assert len(registry._backend_instances) == 3

        # Clear all instances
        registry.clear_instances()

        assert len(registry._backend_instances) == 0

    def test_ingestion_backend_tenant_isolation(self):
        """Test tenant isolation for ingestion backends"""
        registry = get_backend_registry()
        registry.register_ingestion("mock", MockIngestionBackend)

        # Get ingestion backends for two tenants
        backend_a = registry.get_ingestion_backend("mock", tenant_id="tenant_a")
        backend_b = registry.get_ingestion_backend("mock", tenant_id="tenant_b")

        # Should be different instances
        assert backend_a is not backend_b

    def test_full_backend_tenant_isolation(self):
        """Test tenant isolation for full backends (both search and ingestion)"""

        class MockFullBackend(SearchBackend, IngestionBackend):
            def __init__(self):
                self.initialized = False

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

        # Get as search backend for tenant A
        search_a = registry.get_search_backend("mock_full", tenant_id="tenant_a")

        # Get as ingestion backend for tenant A (should be same instance)
        ingest_a = registry.get_ingestion_backend("mock_full", tenant_id="tenant_a")

        # Should be the same instance for same tenant
        assert search_a is ingest_a

        # Get for tenant B (should be different)
        search_b = registry.get_search_backend("mock_full", tenant_id="tenant_b")

        assert search_a is not search_b

    def test_backend_not_found_error_message(self):
        """Test that helpful error is raised for unknown backend"""
        registry = get_backend_registry()

        with pytest.raises(ValueError, match="not found"):
            registry.get_search_backend("nonexistent", tenant_id="tenant_a")

    def test_cache_stats_per_tenant(self):
        """Test that we can track cache statistics per tenant"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        # Create instances for multiple tenants
        registry.get_search_backend("mock", tenant_id="tenant_a")
        registry.get_search_backend("mock", tenant_id="tenant_b")
        registry.get_search_backend("mock", tenant_id="tenant_c")

        # Check cache size
        cache_size = len(registry._backend_instances)
        assert cache_size == 3

        # Get cached instance
        registry.get_search_backend("mock", tenant_id="tenant_a")

        # Should still be 3 (reused cached instance)
        assert len(registry._backend_instances) == 3


class TestBackendRegistryEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Clear registry before each test"""
        registry = get_backend_registry()
        registry.clear_instances()

    def test_none_tenant_id_rejected(self):
        """Test that None tenant_id is rejected"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        with pytest.raises(ValueError, match="tenant_id is required"):
            registry.get_search_backend("mock", tenant_id=None)

    def test_empty_string_tenant_id_rejected(self):
        """Test that empty string tenant_id is rejected"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        with pytest.raises(ValueError, match="tenant_id is required"):
            registry.get_search_backend("mock", tenant_id="")

    def test_whitespace_tenant_id_accepted(self):
        """Test that whitespace-only tenant_id is actually accepted (truthy)"""
        registry = get_backend_registry()
        registry.register_search("mock", MockSearchBackend)

        # Whitespace is truthy, so it should work
        backend = registry.get_search_backend("mock", tenant_id="   ")
        assert backend is not None


class TestBackendRegistrySingleton:
    """Test singleton behavior with tenants"""

    def setup_method(self):
        """Clear registry before each test"""
        registry = get_backend_registry()
        registry.clear_instances()

    def test_get_backend_registry_returns_singleton(self):
        """Test that get_backend_registry always returns same instance"""
        registry1 = get_backend_registry()
        registry2 = get_backend_registry()

        assert registry1 is registry2

    def test_singleton_preserves_tenant_caches(self):
        """Test that singleton preserves tenant caches across calls"""
        registry1 = get_backend_registry()
        registry1.register_search("mock", MockSearchBackend)
        registry1.get_search_backend("mock", tenant_id="tenant_a")

        registry2 = get_backend_registry()

        # Should have the cached instance
        assert len(registry2._backend_instances) == 1
