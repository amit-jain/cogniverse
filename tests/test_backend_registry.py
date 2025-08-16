#!/usr/bin/env python3
"""
Unit tests for the backend registry system.

Tests the pluggable backend architecture including:
- Backend registration
- Auto-discovery
- Interface compliance
- Registry operations
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from src.common.core.backend_registry import (
    BackendRegistry, 
    get_backend_registry,
    register_backend,
    register_ingestion_backend,
    register_search_backend
)
from src.common.core.interfaces import Backend, IngestionBackend, SearchBackend
from src.common.core.documents import Document, MediaType


class MockIngestionBackend(IngestionBackend):
    """Mock ingestion backend for testing."""
    
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
    
    def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.initialized = True
    
    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: Optional[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [{"document_id": f"doc_{i}", "score": 1.0 - i*0.1} for i in range(min(3, top_k))]
    
    def get_document(self, document_id: str) -> Optional[Document]:
        if document_id.startswith("doc_"):
            return Document(
                doc_id=document_id,
                media_type=MediaType.VIDEO_FRAME,
                metadata={"test": True}
            )
        return None
    
    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        return [self.get_document(doc_id) for doc_id in document_ids]
    
    def get_statistics(self) -> Dict[str, Any]:
        return {"document_count": 100, "backend": "mock_search"}
    
    def health_check(self) -> bool:
        return True


class MockFullBackend(Backend):
    """Mock full backend for testing."""
    
    def __init__(self):
        super().__init__("mock_full")
        self.documents = {}
    
    def _initialize_backend(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        for doc in documents:
            self.documents[doc.doc_id] = doc
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
        ranking_strategy: Optional[str] = None
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
            "status": "healthy"
        }
    
    def health_check(self) -> bool:
        return True


class TestBackendRegistry(unittest.TestCase):
    """Test cases for backend registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Get fresh registry and clear any existing registrations
        self.registry = BackendRegistry()
        # Clear class-level registrations
        BackendRegistry._ingestion_backends.clear()
        BackendRegistry._search_backends.clear()
        BackendRegistry._full_backends.clear()
        BackendRegistry._backend_instances.clear()
    
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
        backend = self.registry.get_ingestion_backend("test_ingestion", config)
        
        self.assertIsInstance(backend, MockIngestionBackend)
        self.assertEqual(backend.config, config)
        self.assertTrue(backend.initialized)
    
    def test_get_search_backend(self):
        """Test getting a search backend instance."""
        self.registry.register_search("test_search", MockSearchBackend)
        
        config = {"search_config": "value"}
        backend = self.registry.get_search_backend("test_search", config)
        
        self.assertIsInstance(backend, MockSearchBackend)
        self.assertEqual(backend.config, config)
        self.assertTrue(backend.initialized)
    
    def test_backend_instance_caching(self):
        """Test that backend instances are cached."""
        self.registry.register_backend("test_full", MockFullBackend)
        
        # Get instance twice
        backend1 = self.registry.get_search_backend("test_full", {"config": 1})
        backend2 = self.registry.get_search_backend("test_full", {"config": 2})
        
        # Should be the same instance (cached)
        self.assertIs(backend1, backend2)
    
    def test_clear_instances(self):
        """Test clearing cached instances."""
        self.registry.register_backend("test_full", MockFullBackend)
        
        backend1 = self.registry.get_search_backend("test_full", {})
        self.registry.clear_instances()
        backend2 = self.registry.get_search_backend("test_full", {})
        
        # Should be different instances after clearing
        self.assertIsNot(backend1, backend2)
    
    def test_unregistered_backend_error(self):
        """Test that getting unregistered backend raises error."""
        with self.assertRaises(ValueError) as context:
            self.registry.get_search_backend("nonexistent", {})
        
        self.assertIn("nonexistent", str(context.exception))
        self.assertIn("not found", str(context.exception))
    
    def test_backend_operations(self):
        """Test actual backend operations work correctly."""
        self.registry.register_backend("test_full", MockFullBackend)
        
        # Get backend instance - note: same instance for ingestion and search due to caching
        backend = self.registry.get_ingestion_backend("test_full", {})
        
        # Test ingestion
        doc = Document(
            doc_id="test_doc_1",
            media_type=MediaType.VIDEO_FRAME,
            metadata={"title": "Test Document"}
        )
        
        result = backend.ingest_documents([doc])
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["error_count"], 0)
        
        # Get the SAME backend instance for search (due to caching)
        # This is important - it's the same object that has the ingested documents
        search_backend = self.registry.get_search_backend("test_full", {})
        
        # Verify it's the same instance
        self.assertIs(backend, search_backend)
        
        # Now the document should be there
        retrieved_doc = search_backend.get_document("test_doc_1")
        
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.doc_id, "test_doc_1")
    
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
    
    @patch('importlib.import_module')
    def test_auto_import_backend(self, mock_import):
        """Test auto-import functionality."""
        # Mock successful import
        mock_import.return_value = MagicMock()
        
        # Try to get unregistered backend - should trigger auto-import
        self.registry._try_import_backend("test_auto")
        
        # Should have tried to import
        mock_import.assert_any_call("src.backends.test_auto")
    
    def test_vespa_backend_registration(self):
        """Test that real Vespa backend registers correctly."""
        try:
            # Import Vespa backend (triggers self-registration)
            import src.backends.vespa
            
            # Check it's registered
            registry = get_backend_registry()
            self.assertTrue(registry.is_registered("vespa", "full"))
            self.assertIn("vespa", registry.list_full_backends())
            
            # Test getting Vespa backend instance
            config = {
                "vespa_url": "http://localhost",
                "vespa_port": 8080,
                "schema_name": "test_schema"
            }
            
            # Should not raise error (but may fail to connect)
            try:
                backend = registry.get_search_backend("vespa", config)
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
    
    def test_full_workflow(self):
        """Test complete workflow with backend registry."""
        # Clear any previous registrations
        BackendRegistry._backend_instances.clear()
        
        # Register backend
        registry = get_backend_registry()
        register_backend("workflow_test", MockFullBackend)
        
        # Get backend instance
        backend = registry.get_ingestion_backend("workflow_test", {"test": True})
        
        # Ingest documents
        docs = [
            Document(doc_id=f"doc_{i}", media_type=MediaType.VIDEO_FRAME, metadata={"index": i})
            for i in range(5)
        ]
        
        result = backend.ingest_documents(docs)
        self.assertEqual(result["success_count"], 5)
        
        # Get search backend - should be same instance due to caching
        search_backend = registry.get_search_backend("workflow_test", {})
        
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