#!/usr/bin/env python3
"""
Unit tests for the backend registry system.

Tests the pluggable backend architecture including:
- Backend registration
- Auto-discovery
- Interface compliance
- Registry operations
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_core.common.document import ContentType, Document
from cogniverse_core.registries.backend_registry import (
    BackendRegistry,
    get_backend_registry,
    register_backend,
    register_ingestion_backend,
    register_search_backend,
)
from cogniverse_sdk.interfaces.backend import Backend, IngestionBackend, SearchBackend


class MockIngestionBackend(IngestionBackend):
    """Mock ingestion backend for testing."""

    def __init__(self, config_manager=None, schema_loader=None, backend_config=None):
        self.config_manager = config_manager
        self.schema_loader = schema_loader
        self.backend_config = backend_config

    def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.initialized = True

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        return {"success_count": len(documents), "error_count": 0}

    def ingest_stream(self, documents: Iterator[Document]) -> Iterator[Dict[str, Any]]:
        batch = []
        for doc in documents:
            batch.append(doc)
            if len(batch) >= 2:
                yield self.ingest_documents(batch)
                batch = []
        if batch:
            yield self.ingest_documents(batch)

    def update_document(self, document_id: str, document: Document) -> bool:
        return True

    def delete_document(self, document_id: str) -> bool:
        return True

    def get_schema_info(self) -> Dict[str, Any]:
        return {"name": "mock_ingestion", "fields": []}

    def validate_schema(self, schema_name: str) -> bool:
        return schema_name == "mock_ingestion"


class MockSearchBackend(SearchBackend):
    """Mock search backend for testing."""

    def __init__(self, config_manager=None, schema_loader=None, backend_config=None):
        self.config_manager = config_manager
        self.schema_loader = schema_loader
        self.backend_config = backend_config

    def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.initialized = True

    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: Optional[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {"document_id": f"doc_{i}", "score": 1.0 - i * 0.1}
            for i in range(min(3, top_k))
        ]

    def get_document(self, document_id: str) -> Optional[Document]:
        if document_id.startswith("doc_"):
            return Document(
                id=document_id, content_type=ContentType.VIDEO, metadata={"test": True}
            )
        return None

    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        return [self.get_document(doc_id) for doc_id in document_ids]

    def get_statistics(self) -> Dict[str, Any]:
        return {"document_count": 100, "backend": "mock_search"}

    def health_check(self) -> bool:
        return True

    def get_embedding_requirements(self, schema_name: str) -> Dict[str, Any]:
        """Mock implementation of get_embedding_requirements"""
        return {
            "needs_float": True,
            "needs_binary": False,
            "float_field": "embedding",
            "binary_field": "embedding_binary",
        }


class MockFullBackend(Backend):
    """Mock full backend for testing."""

    def __init__(self, config_manager=None, schema_loader=None, backend_config=None):
        super().__init__("mock_full")
        self.config_manager = config_manager
        self.schema_loader = schema_loader
        self.backend_config = backend_config
        self.documents = {}

    def _initialize_backend(self, config: Dict[str, Any]) -> None:
        self.config = config

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        for doc in documents:
            self.documents[doc.id] = doc
        return {"success_count": len(documents), "error_count": 0}

    def ingest_stream(self, documents: Iterator[Document]) -> Iterator[Dict[str, Any]]:
        batch = []
        for doc in documents:
            batch.append(doc)
            if len(batch) >= 2:
                yield self.ingest_documents(batch)
                batch = []
        if batch:
            yield self.ingest_documents(batch)

    def update_document(self, document_id: str, document: Document) -> bool:
        self.documents[document_id] = document
        return True

    def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    def get_schema_info(self) -> Dict[str, Any]:
        return {"name": "mock_full", "document_count": len(self.documents)}

    def validate_schema(self, schema_name: str) -> bool:
        return schema_name == "mock_full"

    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: Optional[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for doc_id, doc in list(self.documents.items())[:top_k]:
            results.append({"document_id": doc_id, "score": 0.9})
        return results

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id)

    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        return [self.documents.get(doc_id) for doc_id in document_ids]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "document_count": len(self.documents),
            "backend": "mock_full",
            "status": "healthy",
        }

    def health_check(self) -> bool:
        return True

    def get_embedding_requirements(self, schema_name: str) -> Dict[str, Any]:
        """Mock implementation of get_embedding_requirements"""
        return {
            "needs_float": True,
            "needs_binary": False,
            "float_field": "embedding",
            "binary_field": "embedding_binary",
        }

    # Phase 2 Backend interface methods
    def deploy_schema(
        self, schema_name: str, tenant_id: Optional[str] = None, **kwargs
    ) -> bool:
        """Deploy or ensure schema exists for tenant."""
        return True

    def deploy_schemas(
        self, schema_names: List[str], tenant_id: str, force: bool = False
    ) -> Dict[str, bool]:
        """Deploy multiple schemas for a tenant."""
        return {schema_name: True for schema_name in schema_names}

    def delete_schema(
        self, schema_name: str, tenant_id: Optional[str] = None
    ) -> List[str]:
        """Delete tenant schema(s)."""
        return [f"{schema_name}_{tenant_id}" if tenant_id else schema_name]

    def schema_exists(self, schema_name: str, tenant_id: Optional[str] = None) -> bool:
        """Check if schema exists."""
        return True

    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
        """Get tenant-specific schema name."""
        return f"{base_schema_name}_{tenant_id}"

    def create_metadata_document(
        self, schema: str, doc_id: str, fields: Dict[str, Any]
    ) -> bool:
        """Create or update metadata document."""
        return True

    def get_metadata_document(
        self, schema: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata document by ID."""
        return {"id": doc_id, "schema": schema}

    def query_metadata_documents(
        self,
        schema: str,
        query: Optional[str] = None,
        yql: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Query metadata documents."""
        return []

    def delete_metadata_document(self, schema: str, doc_id: str) -> bool:
        """Delete metadata document."""
        return True


class TestBackendRegistry(unittest.TestCase):
    """Test cases for backend registry."""

    def setUp(self):
        """Set up test fixtures."""
        # Get fresh registry and clear any existing registrations
        self.registry = BackendRegistry()
        # Create config_manager for tests
        import tempfile
        from unittest.mock import MagicMock

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_vespa.config.config_store import VespaConfigStore

        self.temp_dir = tempfile.mkdtemp()
        # Create a mock config store for testing
        mock_store = MagicMock(spec=VespaConfigStore)
        mock_store.get_config.return_value = None
        mock_store.set_config.return_value = {"version": 1}
        self.config_manager = ConfigManager(store=mock_store)
        # Test fixture pattern: Create schema_loader in setUp for use across test methods
        self.schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        # Save current registry state to restore in tearDown
        self._saved_ingestion = BackendRegistry._ingestion_backends.copy()
        self._saved_search = BackendRegistry._search_backends.copy()
        self._saved_full = BackendRegistry._full_backends.copy()
        self._saved_instances = BackendRegistry._backend_instances.copy()
        # Clear class-level registrations
        BackendRegistry._ingestion_backends.clear()
        BackendRegistry._search_backends.clear()
        BackendRegistry._full_backends.clear()
        BackendRegistry._backend_instances.clear()

    def tearDown(self):
        """Restore registry state after test."""
        # Restore saved registry state
        BackendRegistry._ingestion_backends = self._saved_ingestion
        BackendRegistry._search_backends = self._saved_search
        BackendRegistry._full_backends = self._saved_full
        BackendRegistry._backend_instances = self._saved_instances
        # Cleanup temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_ingestion_backend(self):
        """Test registering an ingestion backend."""
        self.registry.register_ingestion("test_ingestion", MockIngestionBackend)

        self.assertIn("test_ingestion", self.registry.list_ingestion_backends())
        self.assertTrue(self.registry.is_registered("test_ingestion", "ingestion"))
        self.assertFalse(self.registry.is_registered("test_ingestion", "search"))

    def test_register_search_backend(self):
        """Test registering a search backend."""
        self.registry.register_search("test_search", MockSearchBackend)

        self.assertIn("test_search", self.registry.list_search_backends())
        self.assertTrue(self.registry.is_registered("test_search", "search"))
        self.assertFalse(self.registry.is_registered("test_search", "ingestion"))

    def test_register_full_backend(self):
        """Test registering a full backend."""
        self.registry.register_backend("test_full", MockFullBackend)

        # Should be registered in all three registries
        self.assertIn("test_full", self.registry.list_ingestion_backends())
        self.assertIn("test_full", self.registry.list_search_backends())
        self.assertIn("test_full", self.registry.list_full_backends())

        self.assertTrue(self.registry.is_registered("test_full", "full"))
        self.assertTrue(self.registry.is_registered("test_full", "ingestion"))
        self.assertTrue(self.registry.is_registered("test_full", "search"))

    def test_get_ingestion_backend(self):
        """Test getting an ingestion backend instance."""
        self.registry.register_ingestion("test_ingestion", MockIngestionBackend)

        config = {"test_config": "value"}
        backend = self.registry.get_ingestion_backend(
            "test_ingestion",
            "test_tenant",
            config,
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        self.assertIsInstance(backend, MockIngestionBackend)
        # Config should have tenant_id injected by registry
        self.assertEqual(backend.config, {**config, "tenant_id": "test_tenant"})
        self.assertTrue(backend.initialized)

    def test_get_search_backend(self):
        """Test getting a search backend instance."""
        self.registry.register_search("test_search", MockSearchBackend)

        config = {"search_config": "value"}
        backend = self.registry.get_search_backend(
            "test_search",
            "test_tenant",
            config,
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        self.assertIsInstance(backend, MockSearchBackend)
        # Config should have tenant_id injected by registry
        self.assertEqual(backend.config, {**config, "tenant_id": "test_tenant"})
        self.assertTrue(backend.initialized)

    def test_backend_instance_caching(self):
        """Test that backend instances are cached per tenant."""
        self.registry.register_backend("test_full", MockFullBackend)

        # Get instance twice for same tenant
        backend1 = self.registry.get_search_backend(
            "test_full",
            "test_tenant",
            {"config": 1},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )
        backend2 = self.registry.get_search_backend(
            "test_full",
            "test_tenant",
            {"config": 2},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        # Should be the same instance (cached)
        self.assertIs(backend1, backend2)

    def test_clear_instances(self):
        """Test clearing cached instances."""
        self.registry.register_backend("test_full", MockFullBackend)

        backend1 = self.registry.get_search_backend(
            "test_full",
            "test_tenant",
            {},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )
        self.registry.clear_instances()
        backend2 = self.registry.get_search_backend(
            "test_full",
            "test_tenant",
            {},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        # Should be different instances after clearing
        self.assertIsNot(backend1, backend2)

    def test_unregistered_backend_error(self):
        """Test that getting unregistered backend raises error."""
        with self.assertRaises(ValueError) as context:
            self.registry.get_search_backend(
                "nonexistent",
                "test_tenant",
                {},
                config_manager=self.config_manager,
                schema_loader=self.schema_loader,
            )

        self.assertIn("nonexistent", str(context.exception))
        self.assertIn("not found", str(context.exception))

    def test_backend_operations(self):
        """Test actual backend operations work correctly."""
        self.registry.register_backend("test_full", MockFullBackend)

        # Get backend instance - note: same instance for ingestion and search due to caching
        backend = self.registry.get_ingestion_backend(
            "test_full",
            "test_tenant",
            {},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        # Test ingestion
        doc = Document(
            id="test_doc_1",
            content_type=ContentType.VIDEO,
            metadata={"title": "Test Document"},
        )

        result = backend.ingest_documents([doc])
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["error_count"], 0)

        # Get the SAME backend instance for search (due to caching)
        # This is important - it's the same object that has the ingested documents
        search_backend = self.registry.get_search_backend(
            "test_full",
            "test_tenant",
            {},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        # Verify it's the same instance
        self.assertIs(backend, search_backend)

        # Now the document should be there
        retrieved_doc = search_backend.get_document("test_doc_1")

        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.id, "test_doc_1")

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Use module-level functions
        register_ingestion_backend("conv_ingestion", MockIngestionBackend)
        register_search_backend("conv_search", MockSearchBackend)
        register_backend("conv_full", MockFullBackend)

        # Check they're registered in the global registry
        global_registry = get_backend_registry()
        self.assertIn("conv_ingestion", global_registry.list_ingestion_backends())
        self.assertIn("conv_search", global_registry.list_search_backends())
        self.assertIn("conv_full", global_registry.list_full_backends())

    @patch("importlib.import_module")
    def test_auto_import_backend(self, mock_import):
        """Test auto-import functionality."""
        # Mock successful import
        mock_import.return_value = MagicMock()

        # Try to get unregistered backend - should trigger auto-import
        self.registry._try_import_backend("test_auto")

        # Should have tried to import from SDK package structure first
        mock_import.assert_any_call("cogniverse_test_auto.backend")

    def test_vespa_backend_registration(self):
        """Test that real Vespa backend registers correctly."""
        try:
            # Import Vespa backend and manually register it after setUp cleared everything
            from cogniverse_vespa.backend import VespaBackend

            # Get a fresh registry instance
            registry = get_backend_registry()

            # Register Vespa backend (since setUp cleared the auto-registration)
            registry.register_backend("vespa", VespaBackend)

            # Check it's registered
            self.assertTrue(registry.is_registered("vespa", "full"))
            self.assertIn("vespa", registry.list_full_backends())

            # Test getting Vespa backend instance
            config = {
                "vespa_url": "http://localhost",
                "vespa_port": 8080,
                "schema_name": "test_schema",
            }

            # Should not raise error (but may fail to connect)
            try:
                backend = registry.get_search_backend(
                    "vespa",
                    "test_tenant",
                    config,
                    config_manager=self.config_manager,
                    schema_loader=self.schema_loader,
                )
                self.assertIsNotNone(backend)
            except Exception as e:
                # OK if it fails to connect, just checking registration
                if "not found" in str(e):
                    self.fail(f"Vespa backend not registered: {e}")
        except ImportError:
            # Skip if Vespa dependencies not available
            self.skipTest("Vespa dependencies not available")


class TestBackendIntegration(unittest.TestCase):
    """Integration tests for backend system."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile
        from unittest.mock import MagicMock

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_vespa.config.config_store import VespaConfigStore

        self.temp_dir = tempfile.mkdtemp()
        # Create a mock config store for testing
        mock_store = MagicMock(spec=VespaConfigStore)
        mock_store.get_config.return_value = None
        mock_store.set_config.return_value = {"version": 1}
        self.config_manager = ConfigManager(store=mock_store)
        # Test fixture pattern: Create schema_loader in setUp for use across test methods
        self.schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """Test complete workflow with backend registry."""
        # Clear any previous registrations
        BackendRegistry._backend_instances.clear()

        # Register backend
        registry = get_backend_registry()
        register_backend("workflow_test", MockFullBackend)

        # Get backend instance
        backend = registry.get_ingestion_backend(
            "workflow_test",
            "test_tenant",
            {"test": True},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        # Ingest documents
        docs = [
            Document(
                id=f"doc_{i}", content_type=ContentType.VIDEO, metadata={"index": i}
            )
            for i in range(5)
        ]

        result = backend.ingest_documents(docs)
        self.assertEqual(result["success_count"], 5)

        # Get search backend - should be same instance due to caching
        search_backend = registry.get_search_backend(
            "workflow_test",
            "test_tenant",
            {},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        # Verify it's the same instance (has the ingested documents)
        self.assertIs(backend, search_backend)

        # Now search should work
        search_results = search_backend.search(None, "test", top_k=3)

        self.assertEqual(len(search_results), 3)

        # Get statistics
        stats = search_backend.get_statistics()
        self.assertEqual(stats["document_count"], 5)

        # Delete a document
        self.assertTrue(backend.delete_document("doc_0"))

        # Check it's gone
        doc = search_backend.get_document("doc_0")
        self.assertIsNone(doc)


if __name__ == "__main__":
    unittest.main()
