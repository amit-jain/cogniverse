"""
Unit Tests for ProfileValidator

Tests profile validation logic including:
- Profile name format validation
- Schema template existence
- Strategy class validation
- Embedding type validation
- Uniqueness checking
- Update field restrictions
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.validation.profile_validator import ProfileValidator
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendProfileConfig


@pytest.fixture
def temp_schema_dir(tmp_path: Path) -> Path:
    """Create temporary directory with mock schema templates."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()

    # Create valid schema template
    valid_schema = {
        "name": "video_test",
        "document": {
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "embedding", "type": "tensor<float>(x[128])"},
            ]
        },
    }
    with open(schema_dir / "video_test_schema.json", "w") as f:
        json.dump(valid_schema, f)

    # Create invalid schema (missing document)
    invalid_schema = {"name": "invalid_schema"}
    with open(schema_dir / "invalid_schema_schema.json", "w") as f:
        json.dump(invalid_schema, f)

    # Create malformed JSON
    with open(schema_dir / "malformed_schema.json", "w") as f:
        f.write("{invalid json")

    return schema_dir


@pytest.fixture
def mock_config_manager() -> MagicMock:
    """Create mock ConfigManager for testing."""
    manager = MagicMock(spec=ConfigManager)
    manager.get_backend_profile.return_value = None  # No existing profiles by default
    return manager


@pytest.fixture
def validator(
    mock_config_manager: MagicMock, temp_schema_dir: Path
) -> ProfileValidator:
    """Create ProfileValidator instance with mocked dependencies."""
    return ProfileValidator(
        config_manager=mock_config_manager,
        schema_templates_dir=temp_schema_dir,
    )


@pytest.fixture
def valid_profile() -> BackendProfileConfig:
    """Create valid profile configuration for testing."""
    return BackendProfileConfig(
        profile_name="test_profile",
        type="video",
        description="Test profile",
        schema_name="video_test",
        embedding_model="vidore/colsmol-500m",
        pipeline_config={"keyframe_fps": 30.0},
        strategies={
            "segmentation": {
                "class": "FrameSegmentationStrategy",
                "params": {"fps": 30.0},
            },
            "embedding": {
                "class": "MultiVectorEmbeddingStrategy",
                "params": {},
            },
        },
        embedding_type="frame_based",
        schema_config={"embedding_dim": 128},
        model_specific=None,
    )


class TestProfileNameValidation:
    """Test profile name validation."""

    def test_valid_profile_names(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Valid profile names should pass validation."""
        valid_names = [
            "simple",
            "with_underscore",
            "with-hyphen",
            "mixed_style-123",
            "a" * 100,  # Max length
        ]

        for name in valid_names:
            valid_profile.profile_name = name
            errors = validator._validate_profile_name(name)
            assert (
                not errors
            ), f"Profile name '{name}' should be valid, got errors: {errors}"

    def test_invalid_profile_names(self, validator: ProfileValidator):
        """Invalid profile names should fail validation."""
        invalid_cases = [
            ("", "Profile name cannot be empty"),
            ("with spaces", "only alphanumeric, underscore, and hyphen allowed"),
            ("with@special", "only alphanumeric, underscore, and hyphen allowed"),
            ("with.dots", "only alphanumeric, underscore, and hyphen allowed"),
            ("a" * 101, "Profile name too long"),
        ]

        for name, expected_error_substring in invalid_cases:
            errors = validator._validate_profile_name(name)
            assert errors, f"Profile name '{name}' should be invalid"
            assert any(
                expected_error_substring in err for err in errors
            ), f"Expected error containing '{expected_error_substring}', got: {errors}"

    def test_non_string_profile_name(self, validator: ProfileValidator):
        """Non-string profile names should fail validation."""
        errors = validator._validate_profile_name(123)  # type: ignore
        assert errors
        assert "must be string" in errors[0]


class TestProfileTypeValidation:
    """Test profile type validation."""

    def test_valid_profile_types(self, validator: ProfileValidator):
        """Valid profile types should pass validation."""
        valid_types = ["video", "image", "audio", "text"]

        for profile_type in valid_types:
            errors = validator._validate_profile_type(profile_type)
            assert not errors, f"Profile type '{profile_type}' should be valid"

    def test_invalid_profile_types(self, validator: ProfileValidator):
        """Invalid profile types should fail validation."""
        invalid_types = ["", "document", "pdf", "unknown"]

        for profile_type in invalid_types:
            errors = validator._validate_profile_type(profile_type)
            assert errors, f"Profile type '{profile_type}' should be invalid"


class TestSchemaTemplateValidation:
    """Test schema template validation."""

    def test_valid_schema_template(self, validator: ProfileValidator):
        """Schema template that exists and is valid should pass."""
        errors = validator._validate_schema_template("video_test")
        assert not errors

    def test_missing_schema_template(self, validator: ProfileValidator):
        """Missing schema template should fail validation."""
        errors = validator._validate_schema_template("nonexistent")
        assert errors
        assert any("Schema template not found" in err for err in errors)

    def test_invalid_schema_template(self, validator: ProfileValidator):
        """Invalid schema template (missing document) should fail."""
        errors = validator._validate_schema_template("invalid_schema")
        assert errors
        assert any("missing 'document' field" in err for err in errors)

    def test_malformed_schema_json(self, validator: ProfileValidator):
        """Malformed JSON in schema template should fail."""
        errors = validator._validate_schema_template("malformed")
        assert errors
        assert any("Invalid JSON" in err for err in errors)

    def test_empty_schema_name(self, validator: ProfileValidator):
        """Empty schema name should fail validation."""
        errors = validator._validate_schema_template("")
        assert errors
        assert "Schema name is required" in errors[0]


class TestEmbeddingModelValidation:
    """Test embedding model validation."""

    def test_valid_embedding_models(self, validator: ProfileValidator):
        """Valid embedding model identifiers should pass."""
        valid_models = [
            "vidore/colsmol-500m",
            "openai/clip-vit-base",
            "custom-model-v2",
        ]

        for model in valid_models:
            errors = validator._validate_embedding_model(model)
            assert not errors, f"Embedding model '{model}' should be valid"

    def test_empty_embedding_model(self, validator: ProfileValidator):
        """Empty embedding model should fail."""
        errors = validator._validate_embedding_model("")
        assert errors
        assert "Embedding model is required" in errors[0]

    def test_non_string_embedding_model(self, validator: ProfileValidator):
        """Non-string embedding model should fail."""
        errors = validator._validate_embedding_model(123)  # type: ignore
        assert errors
        assert "must be string" in errors[0]


class TestEmbeddingTypeValidation:
    """Test embedding type validation."""

    def test_valid_embedding_types(self, validator: ProfileValidator):
        """Valid embedding types should pass."""
        valid_types = [
            "frame_based",
            "video_chunks",
            "direct_video_segment",
            "single_vector",
        ]

        for embedding_type in valid_types:
            errors = validator._validate_embedding_type(embedding_type)
            assert not errors, f"Embedding type '{embedding_type}' should be valid"

    def test_invalid_embedding_types(self, validator: ProfileValidator):
        """Invalid embedding types should fail."""
        invalid_types = ["", "unknown", "multi_vector", "chunks"]

        for embedding_type in invalid_types:
            errors = validator._validate_embedding_type(embedding_type)
            assert errors, f"Embedding type '{embedding_type}' should be invalid"


class TestStrategyValidation:
    """Test strategy configuration validation."""

    def test_valid_strategies(self, validator: ProfileValidator):
        """Valid strategy configurations should pass."""
        valid_strategies = {
            "segmentation": {
                "class": "FrameSegmentationStrategy",
                "params": {"fps": 30.0},
            },
            "embedding": {
                "class": "MultiVectorEmbeddingStrategy",
                "params": {},
            },
        }

        with patch.object(validator, "_strategy_class_exists", return_value=True):
            errors = validator._validate_strategies(valid_strategies)
            assert not errors

    def test_missing_strategy_class(self, validator: ProfileValidator):
        """Strategy without 'class' field should fail."""
        invalid_strategies = {
            "segmentation": {
                "params": {"fps": 30.0},
            }
        }

        errors = validator._validate_strategies(invalid_strategies)
        assert errors
        assert any("missing 'class' field" in err for err in errors)

    def test_nonexistent_strategy_class(self, validator: ProfileValidator):
        """Strategy with nonexistent class should fail."""
        invalid_strategies = {
            "segmentation": {
                "class": "NonexistentStrategy",
                "params": {},
            }
        }

        with patch.object(validator, "_strategy_class_exists", return_value=False):
            errors = validator._validate_strategies(invalid_strategies)
            assert errors
            assert any("not found" in err for err in errors)

    def test_non_dict_strategy_config(self, validator: ProfileValidator):
        """Strategy config that's not a dict should fail."""
        invalid_strategies = {"segmentation": "not a dict"}  # type: ignore

        errors = validator._validate_strategies(invalid_strategies)
        assert errors
        assert any("must be dict" in err for err in errors)

    def test_empty_strategies(self, validator: ProfileValidator):
        """Empty strategies should log warning but not fail."""
        errors = validator._validate_strategies({})
        assert not errors  # Warning only, not an error


class TestUniquenessValidation:
    """Test profile uniqueness validation."""

    def test_unique_profile_name(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """New profile with unique name should pass."""
        validator.config_manager.get_backend_profile.return_value = None
        errors = validator._validate_uniqueness(valid_profile, tenant_id="test_tenant")
        assert not errors

    def test_duplicate_profile_name(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """New profile with duplicate name should fail."""
        # Mock existing profile
        validator.config_manager.get_backend_profile.return_value = valid_profile
        errors = validator._validate_uniqueness(valid_profile, tenant_id="test_tenant")
        assert errors
        assert any("already exists" in err for err in errors)


class TestUpdateFieldValidation:
    """Test update field validation."""

    def test_valid_update_fields(self, validator: ProfileValidator):
        """Mutable fields should be allowed in updates."""
        valid_updates = {
            "pipeline_config": {"keyframe_fps": 60.0},
            "strategies": {"segmentation": {"class": "NewStrategy"}},
            "description": "Updated description",
            "model_specific": {"param": "value"},
        }

        errors = validator.validate_update_fields(valid_updates)
        assert not errors

    def test_immutable_field_updates(self, validator: ProfileValidator):
        """Immutable fields should not be allowed in updates."""
        immutable_fields = ["schema_name", "embedding_model", "schema_config", "type"]

        for field in immutable_fields:
            updates = {field: "new_value"}
            errors = validator.validate_update_fields(updates)
            assert errors, f"Field '{field}' should be immutable"
            assert any("cannot be updated" in err for err in errors)


class TestEmbeddingDimensionValidation:
    """Test embedding dimension validation."""

    def test_valid_embedding_dimension(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Valid embedding dimension should pass."""
        valid_profile.schema_config = {"embedding_dim": 128}
        errors = validator._validate_embedding_dimensions(valid_profile)
        assert not errors

    def test_missing_embedding_dimension(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Missing embedding dimension should be allowed (optional)."""
        valid_profile.schema_config = {}
        errors = validator._validate_embedding_dimensions(valid_profile)
        assert not errors

    def test_invalid_embedding_dimension_type(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Non-integer embedding dimension should fail."""
        valid_profile.schema_config = {"embedding_dim": "not_a_number"}
        errors = validator._validate_embedding_dimensions(valid_profile)
        assert errors
        assert any("Invalid embedding_dim" in err for err in errors)

    def test_out_of_range_embedding_dimension(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Out-of-range embedding dimension should fail."""
        test_cases = [0, -1, 100001]

        for dim in test_cases:
            valid_profile.schema_config = {"embedding_dim": dim}
            errors = validator._validate_embedding_dimensions(valid_profile)
            assert errors, f"Dimension {dim} should be out of range"
            assert any("out of reasonable range" in err for err in errors)


class TestFullProfileValidation:
    """Test end-to-end profile validation."""

    def test_valid_profile_passes_all_checks(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Fully valid profile should pass all validations."""
        with patch.object(validator, "_strategy_class_exists", return_value=True):
            errors = validator.validate_profile(
                profile=valid_profile,
                tenant_id="test_tenant",
                is_update=False,
            )
            assert not errors, f"Valid profile should pass, got errors: {errors}"

    def test_multiple_validation_errors_collected(self, validator: ProfileValidator):
        """Multiple validation errors should all be collected."""
        invalid_profile = BackendProfileConfig(
            profile_name="",  # Invalid: empty
            type="invalid_type",  # Invalid: not in allowed types
            description="Test",
            schema_name="nonexistent",  # Invalid: schema doesn't exist
            embedding_model="",  # Invalid: empty
            pipeline_config={},
            strategies={},
            embedding_type="invalid_type",  # Invalid: not in allowed types
            schema_config={},
            model_specific=None,
        )

        errors = validator.validate_profile(
            profile=invalid_profile,
            tenant_id="test_tenant",
            is_update=False,
        )

        # Should have multiple errors
        assert (
            len(errors) >= 4
        )  # At least: name, type, schema, embedding_model, embedding_type
        error_text = " ".join(errors)
        assert "Profile name" in error_text
        assert "Profile type" in error_text or "type" in error_text
        assert "Schema template" in error_text or "schema" in error_text
        assert "Embedding" in error_text

    def test_update_skips_uniqueness_check(
        self, validator: ProfileValidator, valid_profile: BackendProfileConfig
    ):
        """Update validation should skip uniqueness check."""
        # Mock existing profile with same name
        validator.config_manager.get_backend_profile.return_value = valid_profile

        with patch.object(validator, "_strategy_class_exists", return_value=True):
            errors = validator.validate_profile(
                profile=valid_profile,
                tenant_id="test_tenant",
                is_update=True,  # Update mode
            )
            # Should pass even though profile "already exists"
            assert not errors


class TestStrategyClassExists:
    """Test strategy class importability checking."""

    def test_simple_class_name_found(self, validator: ProfileValidator):
        """Simple class name in common location should be found."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.FrameSegmentationStrategy = MagicMock
            mock_import.return_value = mock_module

            result = validator._strategy_class_exists("FrameSegmentationStrategy")
            assert result

    def test_simple_class_name_not_found(self, validator: ProfileValidator):
        """Simple class name not in any location should return False."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            result = validator._strategy_class_exists("NonexistentStrategy")
            assert not result

    def test_fully_qualified_class_name(self, validator: ProfileValidator):
        """Fully qualified class path should be importable."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.CustomStrategy = MagicMock
            mock_import.return_value = mock_module

            result = validator._strategy_class_exists("custom.module.CustomStrategy")
            assert result

    def test_invalid_module_path(self, validator: ProfileValidator):
        """Invalid module path should return False."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'invalid'")

            result = validator._strategy_class_exists("invalid.module.Class")
            assert not result
