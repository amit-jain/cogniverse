"""
Unit tests for SchemaRegistry.

Tests validation, orchestration logic, and schema tracking without requiring Vespa.
"""

from unittest.mock import MagicMock

import pytest

from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_sdk.interfaces.config_store import ConfigScope


class _StoredEntry:
    def __init__(self, config_value):
        self.config_value = config_value


@pytest.fixture
def mock_config_manager():
    """ConfigManager whose store actually persists — set_config writes to a
    backing dict that list_all_configs/get_config read back, so the
    register/deploy → reload round-trip behaves like the real store."""
    backing: dict = {}

    def _set(*, tenant_id, scope, service, config_key, config_value):
        backing[(tenant_id, scope, service, config_key)] = config_value

    def _list(*, scope, service):
        return [
            _StoredEntry(v)
            for (_t, sc, sv, _k), v in backing.items()
            if sc == scope and sv == service
        ]

    def _get(*, tenant_id, scope, service, config_key):
        v = backing.get((tenant_id, scope, service, config_key))
        return _StoredEntry(v) if v is not None else None

    store = MagicMock()
    store.set_config.side_effect = _set
    store.list_all_configs.side_effect = _list
    store.get_config.side_effect = _get

    config_manager = MagicMock()
    config_manager.store = store

    def _seed_schema(config_value):
        _set(
            tenant_id=config_value["tenant_id"],
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=f"schema_{config_value['base_schema_name']}",
            config_value=config_value,
        )

    config_manager.seed_schema = _seed_schema
    return config_manager


@pytest.fixture
def mock_backend():
    """Mock backend for testing"""
    backend = MagicMock()
    backend.deploy_schemas.return_value = True
    return backend


@pytest.fixture
def mock_schema_loader():
    """Mock schema loader for testing"""
    loader = MagicMock()
    loader.load_schema.return_value = {
        "name": "base_schema",
        "fields": [{"name": "id", "type": "string"}],
    }
    return loader


@pytest.fixture
def schema_registry(mock_config_manager, mock_backend, mock_schema_loader):
    """Create SchemaRegistry with mocked dependencies"""
    return SchemaRegistry(
        config_manager=mock_config_manager,
        backend=mock_backend,
        schema_loader=mock_schema_loader,
    )


class TestSchemaRegistryValidation:
    """Test input validation"""

    def test_validate_tenant_id_valid(self, schema_registry):
        """Test valid tenant IDs pass validation"""
        # Should not raise
        schema_registry._validate_tenant_id("acme")
        schema_registry._validate_tenant_id("tenant_123")
        schema_registry._validate_tenant_id("org:tenant")
        schema_registry._validate_tenant_id("UPPERCASE")

    def test_tenant_id_validation_agrees_with_creation(self, schema_registry):
        """Creation and deploy must AGREE on which ids are valid — no gap that
        lets a created tenant fail at deploy time. Hyphens are rejected by both
        (they can't be Vespa schema-name chars, and sanitizing "-"→"_" would
        collide distinct tenants like acme-corp / acme_corp).
        """
        from cogniverse_runtime.admin.tenant_manager import validate_tenant_name

        for name in ["acme", "tenant_123", "tenant456"]:
            validate_tenant_name(name)  # accepted by creation
            schema_registry._validate_tenant_id(name)  # and by deploy
            schema_registry._validate_tenant_id(f"{name}:{name}")

        for bad in ["my-team", "acme-corp"]:
            with pytest.raises(ValueError, match="only alphanumeric"):
                validate_tenant_name(bad)
            with pytest.raises(ValueError, match="only alphanumeric"):
                schema_registry._validate_tenant_id(bad)

    def test_validate_tenant_id_empty_rejected(self, schema_registry):
        """Test empty tenant_id is rejected"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            schema_registry._validate_tenant_id("")

    def test_validate_tenant_id_invalid_chars_rejected(self, schema_registry):
        """Test tenant_id with truly-illegal characters is rejected"""
        for bad in ["tenant with space", "tenant.with.dot", "tenant/slash"]:
            with pytest.raises(ValueError, match="only alphanumeric"):
                schema_registry._validate_tenant_id(bad)

    def test_validate_tenant_id_wrong_type_rejected(self, schema_registry):
        """Test non-string tenant_id is rejected"""
        with pytest.raises(TypeError, match="tenant_id must be string"):
            schema_registry._validate_tenant_id(123)

    def test_validate_schema_name_valid(self, schema_registry):
        """Test valid schema names pass validation"""
        # Should not raise
        schema_registry._validate_schema_name("video_colpali_smol500_mv_frame")
        schema_registry._validate_schema_name("simple_schema")

    def test_validate_schema_name_empty_rejected(self, schema_registry):
        """Test empty schema_name is rejected"""
        with pytest.raises(ValueError, match="schema_name is required"):
            schema_registry._validate_schema_name("")

    def test_validate_schema_name_wrong_type_rejected(self, schema_registry):
        """Test non-string schema_name is rejected"""
        with pytest.raises(TypeError, match="schema_name must be string"):
            schema_registry._validate_schema_name(123)


class TestSchemaRegistryDeployment:
    """Test schema deployment orchestration"""

    def test_deploy_schema_success(
        self, schema_registry, mock_backend, mock_schema_loader, mock_config_manager
    ):
        """Test successful schema deployment"""
        result = schema_registry.deploy_schema("acme", "test_schema")

        # Should return tenant-specific name
        assert result == "test_schema_acme_acme"

        # Should load base schema
        mock_schema_loader.load_schema.assert_called_once_with("test_schema")

        # Should call backend.deploy_schemas with correct structure
        mock_backend.deploy_schemas.assert_called_once()
        call_args = mock_backend.deploy_schemas.call_args[0][0]
        assert len(call_args) == 1  # Only new schema (no existing)
        assert call_args[0]["name"] == "test_schema_acme_acme"
        assert call_args[0]["tenant_id"] == "acme:acme"
        assert call_args[0]["base_schema_name"] == "test_schema"

        # Should register in ConfigManager store
        mock_config_manager.store.set_config.assert_called_once()

    def test_deploy_schema_includes_existing_schemas(
        self, mock_config_manager, mock_backend, mock_schema_loader
    ):
        """Test that deployment includes all existing schemas"""
        mock_config_manager.seed_schema(
            {
                "tenant_id": "existing_tenant",
                "base_schema_name": "existing_schema",
                "full_schema_name": "existing_schema_existing_tenant",
                "schema_definition": '{"name": "existing_schema_existing_tenant"}',
                "deployment_time": "2024-01-01T00:00:00",
                "config": {},
            }
        )

        registry = SchemaRegistry(
            config_manager=mock_config_manager,
            backend=mock_backend,
            schema_loader=mock_schema_loader,
        )

        registry.deploy_schema("new_tenant", "new_schema")

        # Should deploy both existing + new
        call_args = mock_backend.deploy_schemas.call_args[0][0]
        assert len(call_args) == 2

        # First should be existing
        assert call_args[0]["name"] == "existing_schema_existing_tenant"
        assert call_args[0]["tenant_id"] == "existing_tenant"

        # Second should be new (tenant_id canonicalized to org:tenant)
        assert call_args[1]["name"] == "new_schema_new_tenant_new_tenant"
        assert call_args[1]["tenant_id"] == "new_tenant:new_tenant"

    def test_deploy_schema_idempotent(self, schema_registry):
        """Test deploying same schema twice is idempotent"""
        # First deployment
        result1 = schema_registry.deploy_schema("acme", "test_schema")

        # Second deployment (should skip since exists)
        result2 = schema_registry.deploy_schema("acme", "test_schema")

        assert result1 == result2
        assert result1 == "test_schema_acme_acme"

    def test_deploy_schema_redeploys_when_tracked_name_is_legacy(
        self, schema_registry, mock_backend
    ):
        """A tracking record from before tenant-id canonicalization carries
        the single-suffix name — deploy must NOT short-circuit on it, or it
        returns a schema name Vespa never deployed."""
        schema_registry.register_schema(
            tenant_id="acme",
            base_schema_name="test_schema",
            full_schema_name="test_schema_acme",  # legacy single suffix
            schema_definition="{}",
        )
        mock_backend.reset_mock()

        result = schema_registry.deploy_schema("acme", "test_schema")

        assert result == "test_schema_acme_acme"
        mock_backend.deploy_schemas.assert_called_once()

    def test_deploy_schema_force_flag_bypasses_exists_check(
        self, schema_registry, mock_backend
    ):
        """Test force=True redeploys even if schema exists"""
        # First deployment
        schema_registry.deploy_schema("acme", "test_schema")

        # Reset mock
        mock_backend.reset_mock()

        # Second deployment with force=True should redeploy
        result = schema_registry.deploy_schema("acme", "test_schema", force=True)

        assert result == "test_schema_acme_acme"
        # Should have called backend again
        mock_backend.deploy_schemas.assert_called_once()

    def test_deploy_schema_validates_inputs(self, schema_registry):
        """Test that deploy_schema validates inputs"""
        # Invalid tenant_id (space is illegal; hyphens are allowed now)
        with pytest.raises(ValueError, match="only alphanumeric"):
            schema_registry.deploy_schema("tenant invalid", "schema")

        # Empty schema_name
        with pytest.raises(ValueError, match="schema_name is required"):
            schema_registry.deploy_schema("tenant", "")

    def test_deploy_schema_backend_failure_raises(self, schema_registry, mock_backend):
        """Test that backend deployment failure raises exception"""
        mock_backend.deploy_schemas.return_value = False

        with pytest.raises(Exception, match="Backend failed to deploy"):
            schema_registry.deploy_schema("acme", "test_schema")

    def test_deploy_schema_schema_load_failure_raises(
        self, schema_registry, mock_schema_loader
    ):
        """Test that schema load failure raises exception"""
        mock_schema_loader.load_schema.side_effect = FileNotFoundError(
            "Schema not found"
        )

        with pytest.raises(Exception, match="Failed to load base schema"):
            schema_registry.deploy_schema("acme", "nonexistent_schema")

    def test_deploy_schema_requires_backend(
        self, mock_config_manager, mock_schema_loader
    ):
        """Test that backend is required at construction time (fail fast)"""
        # Backend must be provided at construction - cannot be None
        with pytest.raises(ValueError, match="backend is required"):
            SchemaRegistry(
                config_manager=mock_config_manager,
                backend=None,  # No backend - should fail at construction
                schema_loader=mock_schema_loader,
            )

    def test_deploy_schema_requires_schema_loader(
        self, mock_config_manager, mock_backend
    ):
        """Test that schema_loader is required at construction time (fail fast)"""
        # Schema loader must be provided at construction - cannot be None
        with pytest.raises(ValueError, match="schema_loader is required"):
            SchemaRegistry(
                config_manager=mock_config_manager,
                backend=mock_backend,
                schema_loader=None,  # No loader - should fail at construction
            )


class TestSchemaRegistryTracking:
    """Test schema tracking operations"""

    def test_get_tenant_schemas_empty(self, schema_registry):
        """Test getting schemas for tenant with no schemas"""
        schemas = schema_registry.get_tenant_schemas("new_tenant")
        assert schemas == []

    def test_get_tenant_schemas_returns_only_tenant_schemas(self, schema_registry):
        """Test that get_tenant_schemas returns only schemas for specified tenant"""
        # Deploy for multiple tenants
        schema_registry.deploy_schema("tenant_a", "schema1")
        schema_registry.deploy_schema("tenant_b", "schema2")
        schema_registry.deploy_schema("tenant_a", "schema3")

        # Get tenant_a schemas
        schemas_a = schema_registry.get_tenant_schemas("tenant_a")
        assert len(schemas_a) == 2
        base_names = {s.base_schema_name for s in schemas_a}
        assert base_names == {"schema1", "schema3"}

        # Get tenant_b schemas
        schemas_b = schema_registry.get_tenant_schemas("tenant_b")
        assert len(schemas_b) == 1
        assert schemas_b[0].base_schema_name == "schema2"

    def test_schema_exists_returns_true_when_exists(self, schema_registry):
        """Test schema_exists returns True for deployed schema"""
        schema_registry.deploy_schema("acme", "test_schema")

        assert schema_registry.schema_exists("acme", "test_schema") is True

    def test_schema_exists_returns_false_when_not_exists(self, schema_registry):
        """Test schema_exists returns False for non-existent schema"""
        assert schema_registry.schema_exists("acme", "nonexistent") is False

    def test_schema_exists_tenant_isolation(self, schema_registry):
        """Test schema_exists respects tenant isolation"""
        schema_registry.deploy_schema("tenant_a", "schema1")

        # Should exist for tenant_a
        assert schema_registry.schema_exists("tenant_a", "schema1") is True

        # Should not exist for tenant_b
        assert schema_registry.schema_exists("tenant_b", "schema1") is False

    def test_register_schema_adds_to_tracking(self, schema_registry):
        """Test register_schema adds schema to in-memory tracking"""
        schema_registry.register_schema(
            tenant_id="acme",
            base_schema_name="test_schema",
            full_schema_name="test_schema_acme_acme",
            schema_definition='{"name": "test_schema_acme_acme"}',
            config={"profile": "test"},
        )

        # Should be tracked
        assert schema_registry.schema_exists("acme", "test_schema") is True

        # Should be in tenant schemas
        schemas = schema_registry.get_tenant_schemas("acme")
        assert len(schemas) == 1
        assert schemas[0].full_schema_name == "test_schema_acme_acme"
        assert schemas[0].config == {"profile": "test"}

    def test_unregister_schema_removes_from_tracking(
        self, schema_registry, mock_config_manager
    ):
        """Test unregister_schema removes schema from tracking"""
        # Register first
        schema_registry.register_schema(
            tenant_id="acme",
            base_schema_name="test_schema",
            full_schema_name="test_schema_acme_acme",
            schema_definition='{"name": "test_schema_acme_acme"}',
        )

        assert schema_registry.schema_exists("acme", "test_schema") is True

        # Unregister
        schema_registry.unregister_schema("acme", "test_schema")

        # Should be removed
        assert schema_registry.schema_exists("acme", "test_schema") is False

        # Should have marked schema as deleted in store
        # Called twice: once for register, once for unregister
        assert mock_config_manager.store.set_config.call_count == 2


class TestSchemaRegistryInitialization:
    """Test SchemaRegistry initialization"""

    def test_requires_config_manager(self, mock_backend, mock_schema_loader):
        """Test that ConfigManager is required"""
        with pytest.raises(ValueError, match="config_manager is required"):
            SchemaRegistry(
                config_manager=None,
                backend=mock_backend,
                schema_loader=mock_schema_loader,
            )

    def test_requires_backend(self, mock_config_manager, mock_schema_loader):
        """Test that backend is required"""
        with pytest.raises(ValueError, match="backend is required"):
            SchemaRegistry(
                config_manager=mock_config_manager,
                backend=None,
                schema_loader=mock_schema_loader,
            )

    def test_requires_schema_loader(self, mock_config_manager, mock_backend):
        """Test that schema_loader is required"""
        with pytest.raises(ValueError, match="schema_loader is required"):
            SchemaRegistry(
                config_manager=mock_config_manager,
                backend=mock_backend,
                schema_loader=None,
            )

    def test_loads_schemas_from_storage_on_init(
        self, mock_config_manager, mock_backend, mock_schema_loader
    ):
        """Test that existing schemas are loaded from ConfigManager on init"""
        # Tenant_id stored in canonical org:tenant form — matches what
        # register_schema's canonical-id normalization produces today.
        mock_config_manager.seed_schema(
            {
                "tenant_id": "acme:acme",
                "base_schema_name": "schema1",
                "full_schema_name": "schema1_acme_acme",
                "schema_definition": '{"name": "schema1_acme_acme"}',
                "deployment_time": "2024-01-01T00:00:00",
                "config": {},
            }
        )
        mock_config_manager.seed_schema(
            {
                "tenant_id": "startup:startup",
                "base_schema_name": "schema2",
                "full_schema_name": "schema2_startup_startup",
                "schema_definition": '{"name": "schema2_startup_startup"}',
                "deployment_time": "2024-01-01T00:00:00",
                "config": {},
            }
        )

        registry = SchemaRegistry(
            config_manager=mock_config_manager,
            backend=mock_backend,
            schema_loader=mock_schema_loader,
        )

        # Should have loaded both schemas — schema_exists canonicalizes
        # the lookup tenant so the bare-form arg resolves the same.
        assert registry.schema_exists("acme", "schema1") is True
        assert registry.schema_exists("startup", "schema2") is True


class TestReloadReflectsDeletions:
    """A storage refresh must reflect peer deletions, not just additions.

    _load_schemas_from_storage rebuilt the dict additively, so a schema
    deleted from storage by a peer instance lingered in memory after reload
    and was even re-collected by _get_all_schemas for redeploy.
    """

    @staticmethod
    def _entry(tenant_id, base):
        class _E:
            config_value = {
                "tenant_id": tenant_id,
                "base_schema_name": base,
                "full_schema_name": f"{base}_{tenant_id}",
                "schema_definition": {"name": base},
                "deployment_time": "2026-01-01T00:00:00",
            }

        return _E()

    def test_reload_drops_schema_deleted_from_storage(
        self, mock_backend, mock_schema_loader
    ):
        store = MagicMock()
        store.list_all_configs.return_value = [
            self._entry("acme", "video"),
            self._entry("acme", "audio"),
        ]
        config_manager = MagicMock()
        config_manager.store = store

        registry = SchemaRegistry(
            config_manager=config_manager,
            backend=mock_backend,
            schema_loader=mock_schema_loader,
        )
        assert {s.base_schema_name for s in registry._get_all_schemas()} == {
            "video",
            "audio",
        }

        # Peer deletes "audio" from storage.
        store.list_all_configs.return_value = [self._entry("acme", "video")]

        assert {s.base_schema_name for s in registry._get_all_schemas()} == {"video"}
