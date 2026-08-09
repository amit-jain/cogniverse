"""
Unit tests for the Adapter Registry.

Tests the registry models, registry interface, and vespa store with mocks.
"""

from datetime import datetime
from types import SimpleNamespace
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

    def test_get_latest_version_uses_semver_not_lexical(self, registry, mock_store):
        """1.10.0 must rank above 1.9.0 — lexical sort got 1.9.0 wrong."""

        def _doc(version: str) -> dict:
            return {
                "fields": {
                    "adapter_id": f"a-{version}",
                    "tenant_id": "tenant1",
                    "name": "routing_sft",
                    "version": version,
                    "base_model": "SmolLM-135M",
                    "model_type": "llm",
                    "agent_type": "routing",
                    "training_method": "sft",
                    "adapter_path": f"/path/{version}",
                    "status": "inactive",
                    "is_active": 0,
                    "metrics": "{}",
                    "training_config": "{}",
                    "experiment_run_id": "",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            }

        mock_store.list_adapters.return_value = [
            _doc("1.2.0"),
            _doc("1.9.0"),
            _doc("1.10.0"),
        ]

        latest = registry.get_latest_version(
            tenant_id="tenant1", name="routing_sft", agent_type="routing"
        )

        assert latest is not None
        assert latest.version == "1.10.0"

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

    @staticmethod
    def _healthy_response(hits):
        """Query response modeling the real pyvespa shape: ``hits`` mirrors
        ``root.children`` and ``get_json`` exposes a clean body (no
        root.errors, no degraded coverage) so the degraded-read guard sees
        the healthy contract."""
        response = MagicMock()
        response.hits = hits
        response.get_json.return_value = {"root": {"children": hits}}
        return response

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
        mock_response = self._healthy_response(
            [
                {
                    "fields": {
                        "adapter_id": "test-adapter-123",
                        "tenant_id": "tenant1",
                        "name": "routing_sft",
                    }
                }
            ]
        )
        mock_vespa_app.query.return_value = mock_response

        result = adapter_store.get_adapter("test-adapter-123")

        assert result is not None
        assert result["fields"]["adapter_id"] == "test-adapter-123"

    def test_get_adapter_not_found(self, adapter_store, mock_vespa_app):
        """Test getting non-existent adapter returns None."""
        mock_response = self._healthy_response([])
        mock_vespa_app.query.return_value = mock_response

        result = adapter_store.get_adapter("nonexistent")

        assert result is None

    def test_list_adapters(self, adapter_store, mock_vespa_app):
        """Test listing adapters with filters."""
        mock_response = self._healthy_response(
            [
                {"fields": {"adapter_id": "adapter-1", "tenant_id": "tenant1"}},
                {"fields": {"adapter_id": "adapter-2", "tenant_id": "tenant1"}},
            ]
        )
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
        mock_response = self._healthy_response(
            [
                {
                    "fields": {
                        "adapter_id": "active-adapter",
                        "is_active": 1,
                    }
                }
            ]
        )
        mock_vespa_app.query.return_value = mock_response

        result = adapter_store.get_active_adapter("tenant1", "routing")

        assert result is not None
        assert result["fields"]["adapter_id"] == "active-adapter"
        # Verify is_active filter in query
        call_args = mock_vespa_app.query.call_args
        assert "is_active = 1" in call_args.kwargs["yql"]

    def test_set_active_batches_field_updates_one_read_write_per_adapter(self):
        """set_active flips is_active + status on the old and the new adapter.
        Each adapter must be read once and written once — not once per field.
        Field-by-field updates cost 4 reads + 4 writes for a single set_active;
        this pins the batched 2-reads + 2-writes (plus the get_active query)."""

        class _Resp:
            def __init__(self, hits):
                self.hits = hits

        class _FakeVespaApp:
            def __init__(self, docs):
                self.docs = docs
                self.query_count = 0
                self.feeds = []

            def query(self, yql):
                self.query_count += 1
                if "is_active = 1" in yql:
                    hits = [
                        {"fields": f}
                        for f in self.docs.values()
                        if f.get("is_active") == 1
                    ]
                elif "adapter_id contains" in yql:
                    hits = [{"fields": f} for aid, f in self.docs.items() if aid in yql]
                else:
                    hits = []
                return _Resp(hits[:1])

            def feed_data_point(self, schema, data_id, fields):
                self.feeds.append(dict(fields))
                self.docs[fields["adapter_id"]] = dict(fields)

        from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

        fake = _FakeVespaApp(
            {
                "adp_old": {
                    "adapter_id": "adp_old",
                    "tenant_id": "t1",
                    "agent_type": "routing",
                    "is_active": 1,
                    "status": "active",
                },
                "adp_new": {
                    "adapter_id": "adp_new",
                    "tenant_id": "t1",
                    "agent_type": "routing",
                    "is_active": 0,
                    "status": "inactive",
                },
            }
        )
        store = VespaAdapterStore(vespa_app=fake)

        store.set_active("adp_new", "t1", "routing")

        # get_active (1) + read adp_old (1) + read adp_new (1) = 3 queries.
        assert fake.query_count == 3, fake.query_count
        # One write per adapter, not one per field.
        assert len(fake.feeds) == 2, [f.get("adapter_id") for f in fake.feeds]

        by_id = {f["adapter_id"]: f for f in fake.feeds}
        # Both fields landed in the SAME write for each adapter.
        assert by_id["adp_old"]["is_active"] == 0
        assert by_id["adp_old"]["status"] == "inactive"
        assert by_id["adp_new"]["is_active"] == 1
        assert by_id["adp_new"]["status"] == "active"

    def test_delete_adapter(self, adapter_store, mock_vespa_app):
        """Delete returns True for HTTP 200 and False only for a real 404."""
        mock_vespa_app.delete_data.return_value = SimpleNamespace(status_code=200)

        result = adapter_store.delete_adapter("test-adapter-123")

        assert result is True
        mock_vespa_app.delete_data.assert_called_once()

        mock_vespa_app.delete_data.return_value = SimpleNamespace(status_code=404)
        assert adapter_store.delete_adapter("test-adapter-123") is False

        mock_vespa_app.delete_data.return_value = SimpleNamespace(status_code=503)
        with pytest.raises(RuntimeError, match="HTTP 503"):
            adapter_store.delete_adapter("test-adapter-123")

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

    def test_get_hf_storage_forwards_token(self):
        from cogniverse_finetuning.registry.storage import get_storage_backend

        storage = get_storage_backend("hf://myorg/my-repo", token="hf_secret_abc")

        assert storage.token == "hf_secret_abc"

    def test_get_s3_storage_reads_connection_settings_from_env(self, monkeypatch):
        from cogniverse_finetuning.registry.storage import (
            S3Storage,
            S3StorageConfig,
            get_storage_backend,
        )

        monkeypatch.setenv("MINIO_ENDPOINT", "http://minio.example:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "minio_access")
        monkeypatch.setenv("MINIO_SECRET_KEY", "minio_secret")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-2")

        storage = get_storage_backend("s3://adapter-bucket/adapters/model")

        assert isinstance(storage, S3Storage)
        assert storage.config == S3StorageConfig(
            endpoint_url="http://minio.example:9000",
            access_key="minio_access",
            secret_key="minio_secret",
            region="eu-west-2",
        )

    def test_s3_storage_uses_explicit_config(self, monkeypatch):
        import boto3

        from cogniverse_finetuning.registry.storage import (
            S3Storage,
            S3StorageConfig,
        )

        monkeypatch.setenv("MINIO_ENDPOINT", "http://env-endpoint")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "env-access")
        monkeypatch.setenv("MINIO_SECRET_KEY", "env-secret")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "env-region")

        captured = {}

        def _fake_client(service_name, **kwargs):
            captured["service_name"] = service_name
            captured["kwargs"] = kwargs

            class _DummyClient:
                pass

            return _DummyClient()

        monkeypatch.setattr(boto3, "client", _fake_client)

        storage = S3Storage(
            S3StorageConfig(
                endpoint_url="http://explicit-endpoint:9000",
                access_key="explicit-access",
                secret_key="explicit-secret",
                region="eu-central-1",
            )
        )
        storage._client()

        assert captured["service_name"] == "s3"
        assert captured["kwargs"]["endpoint_url"] == "http://explicit-endpoint:9000"
        assert captured["kwargs"]["aws_access_key_id"] == "explicit-access"
        assert captured["kwargs"]["aws_secret_access_key"] == "explicit-secret"
        assert captured["kwargs"]["region_name"] == "eu-central-1"

    def test_upload_adapter_forwards_token_to_backend(self, monkeypatch):
        from cogniverse_finetuning.registry import storage as storage_mod

        captured = {}

        class _SpyStorage:
            def upload(self, local_path, destination_uri):
                return destination_uri

        def _spy_backend(uri, **kwargs):
            captured["token"] = kwargs.get("token")
            return _SpyStorage()

        monkeypatch.setattr(storage_mod, "get_storage_backend", _spy_backend)

        result = storage_mod.upload_adapter(
            "/tmp/adapter", "hf://myorg/my-repo", token="hf_secret_abc"
        )

        assert result == "hf://myorg/my-repo"
        assert captured["token"] == "hf_secret_abc"


class TestInferenceHelpers:
    """Tests for inference helper functions."""

    def test_resolve_adapter_path_file_uri(self, tmp_path):
        """Test resolving file:// URI to local path."""
        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        # cache_dir is mandatory even for file:// URIs — uniform call
        # signature across all callers (no branching defaults).
        path = resolve_adapter_path(
            "file:///data/adapters/routing_sft", cache_dir=str(tmp_path)
        )

        assert path == "/data/adapters/routing_sft"

    def test_resolve_adapter_path_plain_path(self, tmp_path):
        """Test resolving plain local path."""
        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        path = resolve_adapter_path(
            "/data/adapters/routing_sft", cache_dir=str(tmp_path)
        )

        assert path == "/data/adapters/routing_sft"

    def test_resolve_adapter_path_rejects_empty_cache_dir(self):
        """Empty cache_dir must raise — no fallback to hardcoded paths
        or env reads."""
        import pytest

        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        with pytest.raises(ValueError, match="non-empty cache_dir"):
            resolve_adapter_path("file:///data/x", cache_dir="")

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

    def test_get_active_adapter_for_inference_raises_on_registry_outage(self):
        """A registry outage must propagate — returning None reads as "no
        fine-tuned adapter" and silently routes inference to the base model."""
        from unittest.mock import patch

        import pytest

        from cogniverse_finetuning.registry.inference import (
            get_active_adapter_for_inference,
        )

        with patch(
            "cogniverse_finetuning.registry.AdapterRegistry"
        ) as mock_registry_cls:
            mock_registry_cls.return_value.get_active_adapter.side_effect = (
                ConnectionError("registry down")
            )
            with pytest.raises(ConnectionError):
                get_active_adapter_for_inference("tenant1", "routing")

    def test_get_active_adapter_for_inference_none_when_genuinely_absent(self):
        from unittest.mock import patch

        from cogniverse_finetuning.registry.inference import (
            get_active_adapter_for_inference,
        )

        with patch(
            "cogniverse_finetuning.registry.AdapterRegistry"
        ) as mock_registry_cls:
            mock_registry_cls.return_value.get_active_adapter.return_value = None
            assert get_active_adapter_for_inference("tenant1", "routing") is None

    def test_list_available_adapters_raises_on_registry_outage(self):
        """An outage must not read as "no adapters available"."""
        from unittest.mock import patch

        import pytest

        from cogniverse_finetuning.registry.inference import (
            list_available_adapters,
        )

        with patch(
            "cogniverse_finetuning.registry.AdapterRegistry"
        ) as mock_registry_cls:
            mock_registry_cls.return_value.list_adapters.side_effect = ConnectionError(
                "registry down"
            )
            with pytest.raises(ConnectionError):
                list_available_adapters("tenant1")


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


class _FakeModalBatchUpload:
    def __init__(self, root):
        from pathlib import Path

        self.root = Path(root)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def put_directory(self, local_path, remote_path, recursive=True):
        from pathlib import Path

        source = Path(local_path)
        target_root = self.root / remote_path.lstrip("/")
        for file_path in sorted(source.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(source)
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file_path.read_bytes())

    def put_file(self, local_file, remote_path, mode=None):
        from pathlib import Path

        source = Path(local_file)
        target = self.root / remote_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


class _FakeModalVolume:
    def __init__(self, root, fail_stage=None):
        from pathlib import Path

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fail_stage = fail_stage

    def batch_upload(self, force=False):
        if self.fail_stage == "upload":
            raise RuntimeError("modal volume down")
        return _FakeModalBatchUpload(self.root)

    def listdir(self, path, recursive=True):
        from modal.volume import FileEntry, FileEntryType

        if self.fail_stage == "listdir":
            raise RuntimeError("modal volume down")

        base = self.root / path.lstrip("/")
        if not base.exists():
            raise FileNotFoundError(path)

        if base.is_file():
            return [
                FileEntry(
                    path=path.lstrip("/"),
                    type=FileEntryType.FILE,
                    mtime=0,
                    size=base.stat().st_size,
                )
            ]

        entries = []
        glob = base.rglob("*") if recursive else base.glob("*")
        for file_path in sorted(glob):
            if not file_path.is_file():
                continue
            entries.append(
                FileEntry(
                    path=file_path.relative_to(self.root).as_posix(),
                    type=FileEntryType.FILE,
                    mtime=0,
                    size=file_path.stat().st_size,
                )
            )
        return entries

    def read_file_into_fileobj(self, path, fileobj, progress_cb=None):
        if self.fail_stage == "read":
            raise RuntimeError("modal volume down")

        data = (self.root / path.lstrip("/")).read_bytes()
        fileobj.write(data)
        return len(data)


@pytest.mark.unit
class TestInferenceResolverStorageContracts:
    def test_resolve_adapter_path_hf_uri_uses_downloader(self, tmp_path, monkeypatch):
        from pathlib import Path

        import cogniverse_finetuning.registry as registry_pkg
        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        captured = {}

        def _fake_download(uri, local_path):
            captured["uri"] = uri
            captured["local_path"] = local_path
            Path(local_path).mkdir(parents=True, exist_ok=True)
            (Path(local_path) / "config.json").write_text("{}")
            return local_path

        monkeypatch.setattr(registry_pkg, "download_adapter", _fake_download)

        result = resolve_adapter_path("hf://myorg/my-adapter", cache_dir=str(tmp_path))

        assert result == str(tmp_path / "my-adapter")
        assert captured == {
            "uri": "hf://myorg/my-adapter",
            "local_path": str(tmp_path / "my-adapter"),
        }
        assert (tmp_path / "my-adapter" / "config.json").exists()

    def test_resolve_adapter_path_rejects_gs_uri(self, tmp_path):
        from cogniverse_finetuning.registry.inference import resolve_adapter_path

        with pytest.raises(ValueError, match="gs:// adapter URIs are not supported"):
            resolve_adapter_path("gs://bucket/adapters/model", cache_dir=str(tmp_path))


@pytest.mark.unit
class TestModalVolumeStorage:
    def test_factory_builds_modal_storage(self, tmp_path):
        from cogniverse_finetuning.registry.storage import (
            ModalVolumeStorage,
            get_storage_backend,
        )

        fake_volume = _FakeModalVolume(tmp_path / "volume")
        storage = get_storage_backend(
            "modal://adapter-volume/adapters/routing_sft",
            volume=fake_volume,
        )

        assert isinstance(storage, ModalVolumeStorage)
        assert storage.volume_name == "adapter-volume"
        assert storage.volume_path == "adapters/routing_sft"

    def test_round_trip_preserves_nested_files(self, tmp_path):
        from cogniverse_finetuning.registry.storage import ModalVolumeStorage

        fake_volume = _FakeModalVolume(tmp_path / "volume")
        storage = ModalVolumeStorage(
            volume_name="adapter-volume",
            volume_path="adapters/routing_sft",
            volume=fake_volume,
        )

        source = tmp_path / "source"
        (source / "nested").mkdir(parents=True)
        (source / "config.json").write_text('{"name":"routing_sft"}')
        (source / "nested" / "weights.bin").write_bytes(b"modal-weights")

        destination_uri = "modal://adapter-volume/adapters/routing_sft"
        uploaded_uri = storage.upload(str(source), destination_uri)

        assert uploaded_uri == destination_uri
        assert storage.exists(destination_uri) is True

        downloaded = tmp_path / "downloaded"
        result_path = storage.download(destination_uri, str(downloaded))

        assert result_path == str(downloaded)
        assert (downloaded / "config.json").read_text() == '{"name":"routing_sft"}'
        assert (downloaded / "nested" / "weights.bin").read_bytes() == b"modal-weights"

    def test_download_raises_when_volume_call_fails(self, tmp_path):
        from cogniverse_finetuning.registry.storage import ModalVolumeStorage

        fake_volume = _FakeModalVolume(tmp_path / "volume", fail_stage="listdir")
        storage = ModalVolumeStorage(
            volume_name="adapter-volume",
            volume_path="adapters/routing_sft",
            volume=fake_volume,
        )

        with pytest.raises(
            RuntimeError,
            match="failed to download adapter from modal://adapter-volume/adapters/routing_sft",
        ):
            storage.download(
                "modal://adapter-volume/adapters/routing_sft",
                str(tmp_path / "downloaded"),
            )


@pytest.mark.unit
class TestAdapterStoreEntryPointDiscovery:
    """The ``cogniverse.adapter.stores`` entry-point group is the production
    path for resolving an ``AdapterStore``; the rest of the suite injects
    ``store=`` directly, so the discovery path itself was never exercised."""

    def test_vespa_discovered_and_instantiated_via_entry_point(self):
        from unittest.mock import MagicMock

        from cogniverse_core.registries import AdapterStoreRegistry
        from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

        AdapterStoreRegistry.reset()
        try:
            # Real importlib.metadata resolution of the entry-point group.
            assert AdapterStoreRegistry.is_available("vespa"), (
                AdapterStoreRegistry.list_available()
            )
            store = AdapterStoreRegistry.get(
                name="vespa", config={"vespa_app": MagicMock()}
            )
            assert isinstance(store, VespaAdapterStore)
            # Config-scoped cache: same backend key returns the same instance.
            store2 = AdapterStoreRegistry.get(
                name="vespa", config={"vespa_app": MagicMock()}
            )
            assert store2 is store
        finally:
            AdapterStoreRegistry.reset()

    def test_unknown_store_name_raises(self):
        from cogniverse_core.registries import AdapterStoreRegistry

        AdapterStoreRegistry.reset()
        try:
            with pytest.raises(ValueError, match="not found"):
                AdapterStoreRegistry.get(name="nonexistent", config={})
        finally:
            AdapterStoreRegistry.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.unit
class TestUpdateFieldFiltering:
    def test_system_field_in_updates_never_reaches_the_feed(self):
        """The system-field filter must run AFTER the updates merge — filtering
        first let a caller-supplied Vespa system field reach feed_data_point
        (Vespa 400)."""

        class _Resp:
            def __init__(self, hits):
                self.hits = hits

        class _App:
            def __init__(self):
                self.fed = None

            def query(self, yql):
                return _Resp(
                    [{"fields": {"adapter_id": "a1", "sddocname": "adapters"}}]
                )

            def feed_data_point(self, schema, data_id, fields):
                self.fed = fields

        from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

        app = _App()
        store = VespaAdapterStore(vespa_app=app)
        store._update_adapter_fields("a1", {"documentid": "HACK", "status": "x"})

        assert "documentid" not in app.fed
        assert "sddocname" not in app.fed
        assert app.fed["status"] == "x"


@pytest.mark.unit
def test_resolve_adapter_path_rejects_underivable_name(tmp_path):
    from cogniverse_finetuning.registry import resolve_adapter_path

    with pytest.raises(ValueError, match="adapter name"):
        resolve_adapter_path("s3://", cache_dir=str(tmp_path))


class TestSetActiveCrashConsistency:
    def _fake_app(self, fail_on_adapter=None):
        class _Resp:
            def __init__(self, hits):
                self.hits = hits

        class _FakeVespaApp:
            def __init__(self, docs):
                self.docs = docs
                self.feeds = []

            def query(self, yql):
                if "is_active = 1" in yql:
                    hits = [
                        {"fields": f}
                        for f in self.docs.values()
                        if f.get("is_active") == 1
                    ]
                elif "adapter_id contains" in yql:
                    hits = [{"fields": f} for aid, f in self.docs.items() if aid in yql]
                else:
                    hits = []
                return _Resp(hits[:1])

            def feed_data_point(self, schema, data_id, fields):
                if fields["adapter_id"] == fail_on_adapter:
                    raise RuntimeError("vespa write failed mid-switch")
                self.feeds.append(dict(fields))
                self.docs[fields["adapter_id"]] = dict(fields)

        return _FakeVespaApp(
            {
                "adp_old": {
                    "adapter_id": "adp_old",
                    "tenant_id": "t1",
                    "agent_type": "routing",
                    "is_active": 1,
                    "status": "active",
                },
                "adp_new": {
                    "adapter_id": "adp_new",
                    "tenant_id": "t1",
                    "agent_type": "routing",
                    "is_active": 0,
                    "status": "inactive",
                },
            }
        )

    def test_activation_failure_restores_previous_active(self):
        """A failure between deactivate-old and activate-new must not leave
        the tenant with ZERO active adapters — inference silently reverts to
        the base model until a retry. The switch compensates by re-activating
        the previous adapter before surfacing the error."""
        from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

        fake = self._fake_app(fail_on_adapter="adp_new")
        store = VespaAdapterStore(vespa_app=fake)

        with pytest.raises(RuntimeError, match="mid-switch"):
            store.set_active("adp_new", "t1", "routing")

        assert fake.docs["adp_old"]["is_active"] == 1, (
            "previous active adapter was not restored — tenant has no "
            "active adapter after a failed switch"
        )
        assert fake.docs["adp_old"]["status"] == "active"
        assert fake.docs["adp_new"]["is_active"] == 0

    def test_happy_switch_unchanged(self):
        from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

        fake = self._fake_app()
        store = VespaAdapterStore(vespa_app=fake)

        store.set_active("adp_new", "t1", "routing")

        assert fake.docs["adp_old"]["is_active"] == 0
        assert fake.docs["adp_new"]["is_active"] == 1
