"""
Tests for backend configuration management with tenant overrides.

Tests:
- Auto-discovery of config.json
- Backend config merging (system + tenant)
- Tenant profile overrides
- Partial profile updates
"""

import json

import pytest

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
)
from cogniverse_foundation.config.utils import ConfigUtils


class TestBackendConfigDataclasses:
    """Test BackendProfileConfig and BackendConfig dataclasses"""

    def test_backend_profile_config_creation(self):
        """Test creating a BackendProfileConfig"""
        profile = BackendProfileConfig(
            profile_name="test_profile",
            type="video",
            description="Test profile",
            schema_name="test_schema",
            embedding_model="test/model",
            pipeline_config={"extract_keyframes": True},
            strategies={"embedding": {"class": "TestStrategy"}},
            embedding_type="frame_based",
            schema_config={"embedding_dim": 128},
        )

        assert profile.profile_name == "test_profile"
        assert profile.type == "video"
        assert profile.schema_name == "test_schema"
        assert profile.embedding_model == "test/model"

    def test_backend_profile_config_to_dict(self):
        """Test BackendProfileConfig serialization"""
        profile = BackendProfileConfig(
            profile_name="test_profile",
            type="video",
            description="Test profile",
            schema_name="test_schema",
            embedding_model="test/model",
        )

        data = profile.to_dict()
        assert data["type"] == "video"
        assert data["schema_name"] == "test_schema"
        assert data["embedding_model"] == "test/model"
        assert "profile_name" not in data  # Not included in serialization

    def test_backend_profile_config_from_dict(self):
        """Test BackendProfileConfig deserialization"""
        data = {
            "type": "video",
            "description": "Test profile",
            "schema_name": "test_schema",
            "embedding_model": "test/model",
            "pipeline_config": {"extract_keyframes": True},
            "strategies": {},
            "embedding_type": "frame_based",
            "schema_config": {},
        }

        profile = BackendProfileConfig.from_dict("test_profile", data)
        assert profile.profile_name == "test_profile"
        assert profile.type == "video"
        assert profile.schema_name == "test_schema"

    def test_backend_config_creation(self):
        """Test creating a BackendConfig"""
        profile1 = BackendProfileConfig(
            profile_name="profile1", schema_name="schema1", embedding_model="model1"
        )
        profile2 = BackendProfileConfig(
            profile_name="profile2", schema_name="schema2", embedding_model="model2"
        )

        config = BackendConfig(
            tenant_id="test_tenant",
            backend_type="vespa",
            url="http://localhost",
            port=8080,
            profiles={"profile1": profile1, "profile2": profile2},
        )

        assert config.tenant_id == "test_tenant"
        assert config.backend_type == "vespa"
        assert len(config.profiles) == 2

    def test_backend_config_get_profile(self):
        """Test getting a profile by name"""
        profile = BackendProfileConfig(
            profile_name="test_profile", schema_name="test_schema"
        )
        config = BackendConfig(profiles={"test_profile": profile})

        retrieved = config.get_profile("test_profile")
        assert retrieved is not None
        assert retrieved.profile_name == "test_profile"

        missing = config.get_profile("nonexistent")
        assert missing is None

    def test_backend_config_add_profile(self):
        """Test adding a profile"""
        config = BackendConfig()
        assert len(config.profiles) == 0

        profile = BackendProfileConfig(
            profile_name="new_profile", schema_name="new_schema"
        )
        config.add_profile(profile)

        assert len(config.profiles) == 1
        assert "new_profile" in config.profiles

    def test_backend_config_merge_profile(self):
        """Test merging overrides into a profile"""
        base_profile = BackendProfileConfig(
            profile_name="base",
            schema_name="base_schema",
            embedding_model="base/model",
            pipeline_config={"extract_keyframes": True, "transcribe_audio": False},
        )

        config = BackendConfig(profiles={"base": base_profile})

        # Merge partial overrides
        merged = config.merge_profile(
            "base",
            {
                "embedding_model": "custom/model",
                "pipeline_config": {"transcribe_audio": True},
            },
        )

        assert merged.embedding_model == "custom/model"
        assert merged.schema_name == "base_schema"  # Unchanged
        assert merged.pipeline_config["extract_keyframes"] is True  # Preserved
        assert merged.pipeline_config["transcribe_audio"] is True  # Overridden

    def test_backend_config_merge_profile_deep(self):
        """Test deep merge of nested dicts"""
        base_profile = BackendProfileConfig(
            profile_name="base",
            strategies={
                "embedding": {"class": "BaseStrategy", "params": {"dim": 128}},
                "segmentation": {"class": "SegStrategy", "params": {"fps": 1.0}},
            },
        )

        config = BackendConfig(profiles={"base": base_profile})

        # Deep merge strategies
        merged = config.merge_profile(
            "base",
            {
                "strategies": {
                    "embedding": {"params": {"dim": 256}},  # Override dim, keep class
                    "transcription": {"class": "TransStrategy"},  # Add new strategy
                }
            },
        )

        assert merged.strategies["embedding"]["class"] == "BaseStrategy"  # Preserved
        assert merged.strategies["embedding"]["params"]["dim"] == 256  # Overridden
        assert merged.strategies["segmentation"]["class"] == "SegStrategy"  # Preserved
        assert "transcription" in merged.strategies  # Added

    def test_backend_config_merge_profile_not_found(self):
        """Test merging with nonexistent profile raises error"""
        config = BackendConfig()

        with pytest.raises(ValueError, match="Base profile 'nonexistent' not found"):
            config.merge_profile("nonexistent", {"embedding_model": "new/model"})


class TestConfigManagerBackendMethods:
    """Test ConfigManager backend configuration methods"""

    @pytest.fixture
    def config_manager(self, config_manager_memory):
        """Create ConfigManager with in-memory store for unit tests"""
        return config_manager_memory

    def test_get_backend_config_default_empty(self, config_manager):
        """Test getting backend config returns empty config if not set"""
        config = config_manager.get_backend_config(tenant_id="test_tenant")

        assert config.tenant_id == "test_tenant"
        assert len(config.profiles) == 0

    def test_set_and_get_backend_config(self, config_manager):
        """Test setting and getting backend config"""
        profile = BackendProfileConfig(
            profile_name="test_profile", schema_name="test_schema"
        )
        backend_config = BackendConfig(
            tenant_id="test_tenant", profiles={"test_profile": profile}
        )

        config_manager.set_backend_config(backend_config)
        retrieved = config_manager.get_backend_config(tenant_id="test_tenant")

        assert retrieved.tenant_id == "test_tenant"
        assert "test_profile" in retrieved.profiles
        assert retrieved.profiles["test_profile"].schema_name == "test_schema"

    def test_get_backend_profile(self, config_manager):
        """Test getting a specific backend profile"""
        profile = BackendProfileConfig(
            profile_name="test_profile", schema_name="test_schema"
        )
        backend_config = BackendConfig(
            tenant_id="test_tenant", profiles={"test_profile": profile}
        )
        config_manager.set_backend_config(backend_config)

        retrieved = config_manager.get_backend_profile(
            "test_profile", tenant_id="test_tenant"
        )

        assert retrieved is not None
        assert retrieved.profile_name == "test_profile"
        assert retrieved.schema_name == "test_schema"

    def test_add_backend_profile(self, config_manager):
        """Test adding a backend profile"""
        profile = BackendProfileConfig(
            profile_name="new_profile", schema_name="new_schema"
        )

        config_manager.add_backend_profile(profile, tenant_id="test_tenant")

        retrieved = config_manager.get_backend_profile(
            "new_profile", tenant_id="test_tenant"
        )
        assert retrieved is not None
        assert retrieved.profile_name == "new_profile"

    def test_update_backend_profile(self, config_manager):
        """Test updating a backend profile with overrides"""
        # Create base profile for "default" tenant
        base_profile = BackendProfileConfig(
            profile_name="base_profile",
            schema_name="base_schema",
            embedding_model="base/model",
            pipeline_config={"extract_keyframes": True},
        )
        config_manager.add_backend_profile(base_profile, tenant_id="default")

        # Tenant "acme" wants to tweak the embedding model
        updated = config_manager.update_backend_profile(
            profile_name="base_profile",
            overrides={"embedding_model": "acme/custom-model"},
            base_tenant_id="default",
            target_tenant_id="acme",
        )

        assert updated.embedding_model == "acme/custom-model"
        assert updated.schema_name == "base_schema"  # Unchanged

        # Verify it was saved to acme tenant
        acme_profile = config_manager.get_backend_profile(
            "base_profile", tenant_id="acme"
        )
        assert acme_profile.embedding_model == "acme/custom-model"

        # Verify default tenant unchanged
        default_profile = config_manager.get_backend_profile(
            "base_profile", tenant_id="default"
        )
        assert default_profile.embedding_model == "base/model"

    def test_tenant_isolation(self, config_manager):
        """Test that different tenants have isolated backend configs"""
        profile_a = BackendProfileConfig(
            profile_name="shared_profile", schema_name="schema_a"
        )
        profile_b = BackendProfileConfig(
            profile_name="shared_profile", schema_name="schema_b"
        )

        config_manager.add_backend_profile(profile_a, tenant_id="tenant_a")
        config_manager.add_backend_profile(profile_b, tenant_id="tenant_b")

        retrieved_a = config_manager.get_backend_profile(
            "shared_profile", tenant_id="tenant_a"
        )
        retrieved_b = config_manager.get_backend_profile(
            "shared_profile", tenant_id="tenant_b"
        )

        assert retrieved_a.schema_name == "schema_a"
        assert retrieved_b.schema_name == "schema_b"


class TestConfigUtilsBackendConfig:
    """Test ConfigUtils backend config auto-discovery and merging"""

    @pytest.fixture(autouse=True)
    def setup_env(self, backend_config_env):
        """Ensure backend environment is configured for all tests."""
        pass

    @pytest.fixture
    def memory_config_manager(self, config_manager_memory):
        """Provide in-memory config manager for unit tests."""
        return config_manager_memory

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config.json file"""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config_data = {
            "backend": {
                "type": "vespa",
                "url": "http://localhost",
                "port": 8080,
                "profiles": {
                    "system_profile": {
                        "type": "video",
                        "description": "System profile",
                        "schema_name": "system_schema",
                        "embedding_model": "system/model",
                        "pipeline_config": {"extract_keyframes": True},
                        "strategies": {},
                        "embedding_type": "frame_based",
                        "schema_config": {},
                    }
                },
            }
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        return config_file

    def test_auto_discovery_from_configs_dir(
        self, temp_config_file, tmp_path, monkeypatch, memory_config_manager
    ):
        """Test auto-discovery finds config.json in configs/ directory"""
        # Change to temp directory so configs/config.json can be found
        monkeypatch.chdir(tmp_path)

        config_utils = ConfigUtils(
            tenant_id="default", config_manager=memory_config_manager
        )
        backend = config_utils.get("backend")

        assert backend is not None
        assert backend["type"] == "vespa"
        assert "system_profile" in backend["profiles"]

    def test_system_and_tenant_merge(
        self, temp_config_file, tmp_path, monkeypatch, memory_config_manager
    ):
        """Test merging system config with tenant overrides"""
        monkeypatch.chdir(tmp_path)

        # Create tenant-specific profile
        tenant_profile = BackendProfileConfig(
            profile_name="tenant_profile",
            schema_name="tenant_schema",
            embedding_model="tenant/model",
        )
        memory_config_manager.add_backend_profile(tenant_profile, tenant_id="acme")

        # Get merged config for acme tenant
        config_utils = ConfigUtils(
            tenant_id="acme", config_manager=memory_config_manager
        )
        backend = config_utils.get("backend")

        # Should have both system and tenant profiles
        assert "system_profile" in backend["profiles"]  # From config.json
        assert "tenant_profile" in backend["profiles"]  # From ConfigManager

    def test_tenant_override_system_profile(
        self, temp_config_file, tmp_path, monkeypatch, memory_config_manager
    ):
        """Test tenant overriding a system profile"""
        monkeypatch.chdir(tmp_path)

        # Tenant wants to override system_profile
        overridden_profile = BackendProfileConfig(
            profile_name="system_profile",  # Same name as system profile
            schema_name="tenant_custom_schema",
            embedding_model="tenant/custom-model",
        )
        memory_config_manager.add_backend_profile(overridden_profile, tenant_id="acme")

        config_utils = ConfigUtils(
            tenant_id="acme", config_manager=memory_config_manager
        )
        backend = config_utils.get("backend")

        # Tenant profile should win
        assert (
            backend["profiles"]["system_profile"]["schema_name"]
            == "tenant_custom_schema"
        )
        assert (
            backend["profiles"]["system_profile"]["embedding_model"]
            == "tenant/custom-model"
        )


class TestBackendConfigEdgeCases:
    """Test edge cases and error handling"""

    def test_deep_merge_preserves_original(self):
        """Test that deep merge modifies base dict in place"""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        overrides = {"a": {"b": 99}, "e": 4}

        BackendConfig._deep_merge(base, overrides)

        # base should be modified in place
        assert base["a"]["b"] == 99
        assert base["a"]["c"] == 2  # Preserved
        assert base["e"] == 4  # Added

    def test_empty_backend_config_to_dict(self):
        """Test serializing empty backend config"""
        config = BackendConfig()
        data = config.to_dict()

        assert data["tenant_id"] == "default"
        assert data["type"] == "vespa"
        assert data["profiles"] == {}

    def test_backend_profile_optional_fields(self):
        """Test that optional fields can be omitted"""
        profile = BackendProfileConfig(profile_name="minimal", schema_name="test")

        data = profile.to_dict()
        assert data["type"] == "video"  # Default
        assert data["description"] == ""
        assert "process_type" not in data  # Optional, not included if None
