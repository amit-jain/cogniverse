"""
Unit tests for the Adapter Registry.

Tests the registry models, registry interface, and vespa store with mocks.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from cogniverse_finetuning.registry.models import AdapterMetadata


class TestAdapterMetadata:
    """Tests for AdapterMetadata dataclass."""

    def test_create_adapter_metadata(self):
        """Test creating adapter metadata with required fields."""
        metadata = AdapterMetadata(
            adapter_id="test-adapter-123",
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            agent_type="routing",
            training_method="sft",
            adapter_path="/path/to/adapter",
        )

        assert metadata.adapter_id == "test-adapter-123"
        assert metadata.tenant_id == "tenant1"
        assert metadata.name == "routing_sft"
        assert metadata.version == "1.0.0"
        assert metadata.status == "inactive"  # Default
        assert metadata.is_active is False  # Default

    def test_to_vespa_doc(self):
        """Test conversion to Vespa document format."""
        metadata = AdapterMetadata(
            adapter_id="test-adapter-123",
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            agent_type="routing",
            training_method="sft",
            adapter_path="/path/to/adapter",
            status="active",
            is_active=True,
            metrics={"train_loss": 0.5},
            training_config={"epochs": 3},
        )

        doc = metadata.to_vespa_doc()

        assert doc["adapter_id"] == "test-adapter-123"
        assert doc["tenant_id"] == "tenant1"
        assert doc["is_active"] == 1  # Converted to int
        assert doc["metrics"] == '{"train_loss": 0.5}'  # JSON string
        assert doc["training_config"] == '{"epochs": 3}'  # JSON string

    def test_from_vespa_doc(self):
        """Test creating metadata from Vespa document."""
        doc = {
            "fields": {
                "adapter_id": "test-adapter-123",
                "tenant_id": "tenant1",
                "name": "routing_sft",
                "version": "1.0.0",
                "base_model": "SmolLM-135M",
                "model_type": "llm",
                "agent_type": "routing",
                "training_method": "sft",
                "adapter_path": "/path/to/adapter",
                "status": "active",
                "is_active": 1,
                "metrics": '{"train_loss": 0.5}',
                "training_config": '{"epochs": 3}',
                "experiment_run_id": "run_123",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }
        }

        metadata = AdapterMetadata.from_vespa_doc(doc)

        assert metadata.adapter_id == "test-adapter-123"
        assert metadata.is_active is True  # Converted from int
        assert metadata.metrics == {"train_loss": 0.5}  # Parsed JSON
        assert metadata.training_config == {"epochs": 3}  # Parsed JSON

    def test_str_representation(self):
        """Test string representation."""
        metadata = AdapterMetadata(
            adapter_id="test-adapter-123",
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            agent_type="routing",
            training_method="sft",
            adapter_path="/path/to/adapter",
            is_active=True,
        )

        str_repr = str(metadata)
        assert "routing_sft" in str_repr
        assert "v1.0.0" in str_repr
        assert "[ACTIVE]" in str_repr


class TestAdapterRegistry:
    """Tests for AdapterRegistry with mocked store."""

    @pytest.fixture
    def mock_store(self):
        """Create a mocked VespaAdapterStore."""
        store = MagicMock()
        return store

    @pytest.fixture
    def registry(self, mock_store):
        """Create registry with mocked store."""
        from cogniverse_finetuning.registry import AdapterRegistry

        return AdapterRegistry(store=mock_store)

    def test_register_adapter(self, registry, mock_store):
        """Test registering a new adapter."""
        mock_store.save_adapter.return_value = "test-adapter-123"

        adapter_id = registry.register_adapter(
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/path/to/adapter",
            agent_type="routing",
            metrics={"train_loss": 0.5},
        )

        assert adapter_id is not None
        mock_store.save_adapter.assert_called_once()

        # Verify the metadata passed to save_adapter
        call_args = mock_store.save_adapter.call_args[0][0]
        assert call_args["tenant_id"] == "tenant1"
        assert call_args["name"] == "routing_sft"
        assert call_args["version"] == "1.0.0"

    def test_get_adapter(self, registry, mock_store):
        """Test getting adapter by ID."""
        mock_store.get_adapter.return_value = {
            "fields": {
                "adapter_id": "test-adapter-123",
                "tenant_id": "tenant1",
                "name": "routing_sft",
                "version": "1.0.0",
                "base_model": "SmolLM-135M",
                "model_type": "llm",
                "agent_type": "routing",
                "training_method": "sft",
                "adapter_path": "/path/to/adapter",
                "status": "inactive",
                "is_active": 0,
                "metrics": "{}",
                "training_config": "{}",
                "experiment_run_id": "",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        }

        adapter = registry.get_adapter("test-adapter-123")

        assert adapter is not None
        assert adapter.adapter_id == "test-adapter-123"
        assert adapter.name == "routing_sft"
        mock_store.get_adapter.assert_called_once_with("test-adapter-123")

    def test_get_adapter_not_found(self, registry, mock_store):
        """Test getting non-existent adapter."""
        mock_store.get_adapter.return_value = None

        adapter = registry.get_adapter("nonexistent")

        assert adapter is None

    def test_list_adapters(self, registry, mock_store):
        """Test listing adapters."""
        mock_store.list_adapters.return_value = [
            {
                "fields": {
                    "adapter_id": "adapter-1",
                    "tenant_id": "tenant1",
                    "name": "routing_sft",
                    "version": "1.0.0",
                    "base_model": "SmolLM-135M",
                    "model_type": "llm",
                    "agent_type": "routing",
                    "training_method": "sft",
                    "adapter_path": "/path/1",
                    "status": "active",
                    "is_active": 1,
                    "metrics": "{}",
                    "training_config": "{}",
                    "experiment_run_id": "",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            {
                "fields": {
                    "adapter_id": "adapter-2",
                    "tenant_id": "tenant1",
                    "name": "routing_dpo",
                    "version": "1.0.0",
                    "base_model": "SmolLM-135M",
                    "model_type": "llm",
                    "agent_type": "routing",
                    "training_method": "dpo",
                    "adapter_path": "/path/2",
                    "status": "inactive",
                    "is_active": 0,
                    "metrics": "{}",
                    "training_config": "{}",
                    "experiment_run_id": "",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        ]

        adapters = registry.list_adapters("tenant1")

        assert len(adapters) == 2
        assert adapters[0].name == "routing_sft"
        assert adapters[1].name == "routing_dpo"

    def test_get_active_adapter(self, registry, mock_store):
        """Test getting active adapter."""
        mock_store.get_active_adapter.return_value = {
            "fields": {
                "adapter_id": "active-adapter",
                "tenant_id": "tenant1",
                "name": "routing_sft",
                "version": "2.0.0",
                "base_model": "SmolLM-135M",
                "model_type": "llm",
                "agent_type": "routing",
                "training_method": "sft",
                "adapter_path": "/path/active",
                "status": "active",
                "is_active": 1,
                "metrics": "{}",
                "training_config": "{}",
                "experiment_run_id": "",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        }

        adapter = registry.get_active_adapter("tenant1", "routing")

        assert adapter is not None
        assert adapter.adapter_id == "active-adapter"
        assert adapter.is_active is True

    def test_activate_adapter(self, registry, mock_store):
        """Test activating an adapter."""
        mock_store.get_adapter.return_value = {
            "fields": {
                "adapter_id": "test-adapter",
                "tenant_id": "tenant1",
                "name": "routing_sft",
                "version": "1.0.0",
                "base_model": "SmolLM-135M",
                "model_type": "llm",
                "agent_type": "routing",
                "training_method": "sft",
                "adapter_path": "/path/to/adapter",
                "status": "inactive",
                "is_active": 0,
                "metrics": "{}",
                "training_config": "{}",
                "experiment_run_id": "",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        }

        registry.activate_adapter("test-adapter")

        mock_store.set_active.assert_called_once_with(
            "test-adapter", "tenant1", "routing"
        )

    def test_deprecate_adapter(self, registry, mock_store):
        """Test deprecating an adapter."""
        registry.deprecate_adapter("test-adapter")

        mock_store.deprecate_adapter.assert_called_once_with("test-adapter")

    def test_delete_adapter(self, registry, mock_store):
        """Test deleting an adapter."""
        mock_store.delete_adapter.return_value = True

        result = registry.delete_adapter("test-adapter")

        assert result is True
        mock_store.delete_adapter.assert_called_once_with("test-adapter")


class TestVespaAdapterStore:
    """Tests for VespaAdapterStore with mocked Vespa client."""

    @pytest.fixture
    def mock_vespa_app(self):
        """Create a mocked Vespa application."""
        return MagicMock()

    @pytest.fixture
    def adapter_store(self, mock_vespa_app):
        """Create adapter store with mocked Vespa."""
        from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

        return VespaAdapterStore(vespa_app=mock_vespa_app)

    def test_save_adapter(self, adapter_store, mock_vespa_app):
        """Test saving adapter to Vespa."""
        metadata = {
            "adapter_id": "test-adapter-123",
            "tenant_id": "tenant1",
            "name": "routing_sft",
            "version": "1.0.0",
            "status": "inactive",
            "is_active": 0,
        }

        adapter_id = adapter_store.save_adapter(metadata)

        assert adapter_id == "test-adapter-123"
        mock_vespa_app.feed_data_point.assert_called_once()

    def test_save_adapter_missing_id(self, adapter_store):
        """Test saving adapter without adapter_id raises error."""
        metadata = {"tenant_id": "tenant1", "name": "test"}

        with pytest.raises(ValueError, match="adapter_id is required"):
            adapter_store.save_adapter(metadata)

    def test_get_adapter(self, adapter_store, mock_vespa_app):
        """Test getting adapter from Vespa."""
        mock_response = MagicMock()
        mock_response.hits = [
            {
                "fields": {
                    "adapter_id": "test-adapter-123",
                    "tenant_id": "tenant1",
                    "name": "routing_sft",
                }
            }
        ]
        mock_vespa_app.query.return_value = mock_response

        result = adapter_store.get_adapter("test-adapter-123")

        assert result is not None
        assert result["fields"]["adapter_id"] == "test-adapter-123"

    def test_get_adapter_not_found(self, adapter_store, mock_vespa_app):
        """Test getting non-existent adapter returns None."""
        mock_response = MagicMock()
        mock_response.hits = []
        mock_vespa_app.query.return_value = mock_response

        result = adapter_store.get_adapter("nonexistent")

        assert result is None

    def test_list_adapters(self, adapter_store, mock_vespa_app):
        """Test listing adapters with filters."""
        mock_response = MagicMock()
        mock_response.hits = [
            {"fields": {"adapter_id": "adapter-1", "tenant_id": "tenant1"}},
            {"fields": {"adapter_id": "adapter-2", "tenant_id": "tenant1"}},
        ]
        mock_vespa_app.query.return_value = mock_response

        results = adapter_store.list_adapters(
            tenant_id="tenant1", agent_type="routing", status="active"
        )

        assert len(results) == 2
        # Verify query was constructed with filters
        call_args = mock_vespa_app.query.call_args
        assert "tenant_id" in call_args.kwargs["yql"]
        assert "agent_type" in call_args.kwargs["yql"]
        assert "status" in call_args.kwargs["yql"]

    def test_get_active_adapter(self, adapter_store, mock_vespa_app):
        """Test getting active adapter."""
        mock_response = MagicMock()
        mock_response.hits = [
            {
                "fields": {
                    "adapter_id": "active-adapter",
                    "is_active": 1,
                }
            }
        ]
        mock_vespa_app.query.return_value = mock_response

        result = adapter_store.get_active_adapter("tenant1", "routing")

        assert result is not None
        assert result["fields"]["adapter_id"] == "active-adapter"
        # Verify is_active filter in query
        call_args = mock_vespa_app.query.call_args
        assert "is_active = 1" in call_args.kwargs["yql"]

    def test_delete_adapter(self, adapter_store, mock_vespa_app):
        """Test deleting adapter."""
        result = adapter_store.delete_adapter("test-adapter-123")

        assert result is True
        mock_vespa_app.delete_data.assert_called_once()

    def test_health_check_healthy(self, adapter_store, mock_vespa_app):
        """Test health check when Vespa is healthy."""
        mock_response = MagicMock()
        mock_response.hits = []
        mock_vespa_app.query.return_value = mock_response

        assert adapter_store.health_check() is True

    def test_health_check_unhealthy(self, adapter_store, mock_vespa_app):
        """Test health check when Vespa is down."""
        mock_vespa_app.query.side_effect = Exception("Connection refused")

        assert adapter_store.health_check() is False


class TestAdapterMetadataUri:
    """Tests for AdapterMetadata URI methods."""

    def test_get_effective_uri_with_adapter_uri(self):
        """Test get_effective_uri returns adapter_uri when set."""
        metadata = AdapterMetadata(
            adapter_id="test-adapter-123",
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            agent_type="routing",
            training_method="sft",
            adapter_path="/local/path/adapter",
            adapter_uri="hf://myorg/routing-adapter/v1",
        )

        assert metadata.get_effective_uri() == "hf://myorg/routing-adapter/v1"

    def test_get_effective_uri_without_adapter_uri(self):
        """Test get_effective_uri returns file:// URI from path when no adapter_uri."""
        metadata = AdapterMetadata(
            adapter_id="test-adapter-123",
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            agent_type="routing",
            training_method="sft",
            adapter_path="/local/path/adapter",
        )

        assert metadata.get_effective_uri() == "file:///local/path/adapter"

    def test_get_effective_uri_empty_path(self):
        """Test get_effective_uri returns empty string when both are empty."""
        metadata = AdapterMetadata(
            adapter_id="test-adapter-123",
            tenant_id="tenant1",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            agent_type="routing",
            training_method="sft",
            adapter_path="",
        )

        assert metadata.get_effective_uri() == ""


class TestLocalStorage:
    """Tests for LocalStorage with real filesystem."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temp directories for testing."""
        source_dir = tmp_path / "source_adapter"
        source_dir.mkdir()
        (source_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        (source_dir / "adapter_model.safetensors").write_bytes(b"fake model data")

        dest_dir = tmp_path / "destination"
        dest_dir.mkdir()

        return {"source": source_dir, "dest": dest_dir, "base": tmp_path}

    def test_upload_local_storage(self, temp_dirs):
        """Test uploading adapter to local storage."""
        from cogniverse_finetuning.registry.storage import LocalStorage

        storage = LocalStorage()
        source = str(temp_dirs["source"])
        dest = str(temp_dirs["dest"] / "uploaded_adapter")

        result_uri = storage.upload(source, dest)

        assert result_uri.startswith("file://")
        assert "uploaded_adapter" in result_uri
        # Verify files were copied
        dest_path = temp_dirs["dest"] / "uploaded_adapter"
        assert (dest_path / "adapter_config.json").exists()
        assert (dest_path / "adapter_model.safetensors").exists()

    def test_download_local_storage(self, temp_dirs):
        """Test downloading adapter from local storage."""
        from cogniverse_finetuning.registry.storage import LocalStorage

        storage = LocalStorage()
        source = f"file://{temp_dirs['source']}"
        dest = str(temp_dirs["dest"] / "downloaded_adapter")

        result_path = storage.download(source, dest)

        assert result_path == dest
        # Verify files were copied
        dest_path = temp_dirs["dest"] / "downloaded_adapter"
        assert (dest_path / "adapter_config.json").exists()

    def test_exists_local_storage(self, temp_dirs):
        """Test checking if adapter exists in local storage."""
        from cogniverse_finetuning.registry.storage import LocalStorage

        storage = LocalStorage()

        assert storage.exists(str(temp_dirs["source"])) is True
        assert storage.exists(f"file://{temp_dirs['source']}") is True
        assert storage.exists("/nonexistent/path") is False

    def test_upload_same_location(self, temp_dirs):
        """Test uploading to same location (no-op)."""
        from cogniverse_finetuning.registry.storage import LocalStorage

        storage = LocalStorage()
        source = str(temp_dirs["source"])

        result_uri = storage.upload(source, source)

        assert result_uri.startswith("file://")


class TestGetStorageBackend:
    """Tests for get_storage_backend factory function."""

    def test_get_local_storage_file_uri(self):
        """Test getting LocalStorage for file:// URI."""
        from cogniverse_finetuning.registry.storage import (
            LocalStorage,
            get_storage_backend,
        )

        storage = get_storage_backend("file:///path/to/adapter")

        assert isinstance(storage, LocalStorage)

    def test_get_local_storage_plain_path(self):
        """Test getting LocalStorage for plain path."""
        from cogniverse_finetuning.registry.storage import (
            LocalStorage,
            get_storage_backend,
        )

        storage = get_storage_backend("/path/to/adapter")

        assert isinstance(storage, LocalStorage)

    def test_get_hf_storage(self):
        """Test getting HuggingFaceStorage for hf:// URI."""
        from cogniverse_finetuning.registry.storage import (
            HuggingFaceStorage,
            get_storage_backend,
        )

        storage = get_storage_backend("hf://myorg/my-repo")

        assert isinstance(storage, HuggingFaceStorage)

    def test_unsupported_scheme(self):
        """Test error for unsupported storage scheme."""
        from cogniverse_finetuning.registry.storage import get_storage_backend

        with pytest.raises(ValueError, match="Unsupported storage scheme"):
            get_storage_backend("unknown://bucket/path")


class TestInferenceHelpers:
    """Tests for inference helper functions."""

    def test_resolve_adapter_path_file_uri(self):
        """Test resolving file:// URI to local path."""
        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        path = resolve_adapter_path("file:///data/adapters/routing_sft")

        assert path == "/data/adapters/routing_sft"

    def test_resolve_adapter_path_plain_path(self):
        """Test resolving plain local path."""
        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        path = resolve_adapter_path("/data/adapters/routing_sft")

        assert path == "/data/adapters/routing_sft"

    def test_adapter_info_dataclass(self):
        """Test AdapterInfo dataclass."""
        from cogniverse_finetuning.registry.inference import AdapterInfo

        info = AdapterInfo(
            adapter_id="test-123",
            name="routing_sft",
            version="1.0.0",
            base_model="SmolLM-135M",
            adapter_uri="hf://myorg/routing-adapter",
            adapter_path="/local/path",
        )

        assert info.adapter_id == "test-123"
        assert info.adapter_uri == "hf://myorg/routing-adapter"
        assert info.adapter_path == "/local/path"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_adapter_dir(self, tmp_path):
        """Create a temp adapter directory."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "config.json").write_text('{"test": true}')
        return adapter_dir

    def test_upload_adapter_function(self, temp_adapter_dir, tmp_path):
        """Test upload_adapter convenience function."""
        from cogniverse_finetuning.registry.storage import upload_adapter

        dest = str(tmp_path / "uploaded")
        result = upload_adapter(str(temp_adapter_dir), dest)

        assert result.startswith("file://")
        assert (tmp_path / "uploaded" / "config.json").exists()

    def test_download_adapter_function(self, temp_adapter_dir, tmp_path):
        """Test download_adapter convenience function."""
        from cogniverse_finetuning.registry.storage import download_adapter

        dest = str(tmp_path / "downloaded")
        result = download_adapter(f"file://{temp_adapter_dir}", dest)

        assert result == dest
        assert (tmp_path / "downloaded" / "config.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
