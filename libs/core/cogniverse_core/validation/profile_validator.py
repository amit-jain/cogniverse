"""
Profile Validator

Validates backend profile configurations before creation/update.
Ensures schema templates exist, strategy classes are importable,
and profile settings are consistent.
"""

import importlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from cogniverse_core.config.manager import ConfigManager
    from cogniverse_core.config.unified_config import BackendProfileConfig

logger = logging.getLogger(__name__)


class ProfileValidator:
    """Validates backend profile configurations."""

    # Valid embedding types
    VALID_EMBEDDING_TYPES = [
        "frame_based",
        "video_chunks",
        "direct_video_segment",
        "single_vector",
    ]

    # Valid profile types
    VALID_PROFILE_TYPES = ["video", "image", "audio", "text"]

    def __init__(
        self,
        config_manager: "ConfigManager",
        schema_templates_dir: Optional[Path] = None,
    ):
        """
        Initialize ProfileValidator.

        Args:
            config_manager: ConfigManager instance for checking existing profiles
            schema_templates_dir: Directory containing schema template JSON files
                                 (defaults to configs/schemas/)
        """
        self.config_manager = config_manager
        self.schema_templates_dir = schema_templates_dir or Path("configs/schemas")

    def validate_profile(
        self, profile: "BackendProfileConfig", tenant_id: str, is_update: bool = False
    ) -> List[str]:
        """
        Validate a backend profile configuration.

        Args:
            profile: BackendProfileConfig to validate
            tenant_id: Tenant identifier
            is_update: Whether this is an update (skip uniqueness check)

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check uniqueness (only for new profiles)
        if not is_update:
            errors.extend(self._validate_uniqueness(profile, tenant_id))

        # Validate profile name format
        errors.extend(self._validate_profile_name(profile.profile_name))

        # Validate profile type
        errors.extend(self._validate_profile_type(profile.type))

        # Validate schema template exists
        errors.extend(self._validate_schema_template(profile.schema_name))

        # Validate embedding model
        errors.extend(self._validate_embedding_model(profile.embedding_model))

        # Validate embedding type
        errors.extend(self._validate_embedding_type(profile.embedding_type))

        # Validate strategy classes
        errors.extend(self._validate_strategies(profile.strategies))

        # Validate embedding dimensions (if specified)
        errors.extend(self._validate_embedding_dimensions(profile))

        return errors

    def _validate_uniqueness(
        self, profile: "BackendProfileConfig", tenant_id: str
    ) -> List[str]:
        """Check if profile name is unique for tenant."""
        errors = []

        try:
            existing = self.config_manager.get_backend_profile(
                tenant_id=tenant_id,
                profile_name=profile.profile_name,
            )
            if existing:
                errors.append(
                    f"Profile '{profile.profile_name}' already exists for tenant '{tenant_id}'"
                )
        except Exception:
            # Profile doesn't exist (which is good for new profiles)
            pass

        return errors

    def _validate_profile_name(self, profile_name: str) -> List[str]:
        """Validate profile name format."""
        errors = []

        if not profile_name:
            errors.append("Profile name cannot be empty")
            return errors

        if not isinstance(profile_name, str):
            errors.append(f"Profile name must be string, got {type(profile_name)}")
            return errors

        # Allow alphanumeric, underscore, hyphen
        if not profile_name.replace("_", "").replace("-", "").isalnum():
            errors.append(
                f"Invalid profile name '{profile_name}': "
                "only alphanumeric, underscore, and hyphen allowed"
            )

        if len(profile_name) > 100:
            errors.append(f"Profile name too long ({len(profile_name)} chars, max 100)")

        return errors

    def _validate_profile_type(self, profile_type: str) -> List[str]:
        """Validate profile type."""
        errors = []

        if not profile_type:
            errors.append("Profile type is required")
            return errors

        if profile_type not in self.VALID_PROFILE_TYPES:
            errors.append(
                f"Invalid profile type '{profile_type}'. "
                f"Must be one of: {self.VALID_PROFILE_TYPES}"
            )

        return errors

    def _validate_schema_template(self, schema_name: str) -> List[str]:
        """Validate that schema template file exists."""
        errors = []

        if not schema_name:
            errors.append("Schema name is required")
            return errors

        # Check if schema template file exists
        schema_file = self.schema_templates_dir / f"{schema_name}_schema.json"

        if not schema_file.exists():
            errors.append(
                f"Schema template not found: {schema_file}. "
                "Create schema template in configs/schemas/ before creating profile."
            )
            return errors

        # Try to load and validate schema JSON
        try:
            with open(schema_file, "r") as f:
                schema_json = json.load(f)

            # Basic schema validation
            if "name" not in schema_json:
                errors.append(f"Schema template missing 'name' field: {schema_file}")

            if "document" not in schema_json:
                errors.append(
                    f"Schema template missing 'document' field: {schema_file}"
                )
            elif "fields" not in schema_json.get("document", {}):
                errors.append(
                    f"Schema template document missing 'fields': {schema_file}"
                )

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in schema template {schema_file}: {e}")
        except Exception as e:
            errors.append(f"Error loading schema template {schema_file}: {e}")

        return errors

    def _validate_embedding_model(self, embedding_model: str) -> List[str]:
        """Validate embedding model identifier."""
        errors = []

        if not embedding_model:
            errors.append("Embedding model is required")
            return errors

        if not isinstance(embedding_model, str):
            errors.append(
                f"Embedding model must be string, got {type(embedding_model)}"
            )
            return errors

        # Basic format check (allows "org/model" format like "vidore/colsmol-500m")
        if "/" not in embedding_model and "-" not in embedding_model:
            logger.warning(
                f"Embedding model '{embedding_model}' has unusual format. "
                "Expected format: 'org/model' or 'model-name'"
            )

        return errors

    def _validate_embedding_type(self, embedding_type: str) -> List[str]:
        """Validate embedding type."""
        errors = []

        if not embedding_type:
            errors.append("Embedding type is required")
            return errors

        if embedding_type not in self.VALID_EMBEDDING_TYPES:
            errors.append(
                f"Invalid embedding type '{embedding_type}'. "
                f"Must be one of: {self.VALID_EMBEDDING_TYPES}"
            )

        return errors

    def _validate_strategies(self, strategies: dict) -> List[str]:
        """Validate strategy configurations."""
        errors = []

        if not strategies:
            logger.warning("No strategies defined for profile")
            return errors

        for strategy_name, strategy_config in strategies.items():
            if not isinstance(strategy_config, dict):
                errors.append(
                    f"Strategy '{strategy_name}' config must be dict, "
                    f"got {type(strategy_config)}"
                )
                continue

            # Check for 'class' field
            strategy_class = strategy_config.get("class")
            if not strategy_class:
                errors.append(
                    f"Strategy '{strategy_name}' missing 'class' field. "
                    "Each strategy must specify a class to use."
                )
                continue

            # Validate strategy class exists
            if not self._strategy_class_exists(strategy_class):
                errors.append(
                    f"Strategy class '{strategy_class}' not found. "
                    "Ensure the class is importable from the configured module path."
                )

        return errors

    def _strategy_class_exists(self, class_path: str) -> bool:
        """
        Check if strategy class can be imported.

        Args:
            class_path: Full class path like "FrameSegmentationStrategy"
                       or "module.ClassName"

        Returns:
            True if class is importable, False otherwise
        """
        try:
            # Handle simple class names (assume they're in cogniverse packages)
            if "." not in class_path:
                # Try common locations
                possible_modules = [
                    "cogniverse_runtime.ingestion.processors.segmentation.strategies",
                    "cogniverse_runtime.ingestion.processors.embedding_generator.strategies",
                ]

                for module_path in possible_modules:
                    try:
                        module = importlib.import_module(module_path)
                        if hasattr(module, class_path):
                            return True
                    except ImportError:
                        continue

                logger.warning(
                    f"Strategy class '{class_path}' not found in common locations. "
                    "Consider using fully qualified path."
                )
                return False

            # Handle fully qualified paths
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return hasattr(module, class_name)

        except ImportError as e:
            logger.warning(f"Cannot import strategy class '{class_path}': {e}")
            return False
        except Exception as e:
            logger.warning(f"Error checking strategy class '{class_path}': {e}")
            return False

    def _validate_embedding_dimensions(
        self, profile: "BackendProfileConfig"
    ) -> List[str]:
        """Validate embedding dimensions match schema."""
        errors = []

        # Check if schema_config has embedding_dim
        embedding_dim = profile.schema_config.get("embedding_dim")
        if embedding_dim is None:
            # Not specified, skip validation
            return errors

        try:
            embedding_dim = int(embedding_dim)
        except (ValueError, TypeError):
            errors.append(
                f"Invalid embedding_dim in schema_config: {embedding_dim}. Must be integer."
            )
            return errors

        # Validate dimension is reasonable
        if embedding_dim < 1 or embedding_dim > 100000:
            errors.append(
                f"Embedding dimension {embedding_dim} out of reasonable range (1-100000)"
            )

        # TODO: Validate against model's actual output dimension
        # This would require loading the model or querying a model registry
        # For now, just check it's a reasonable value

        return errors

    def validate_update_fields(self, update_fields: dict) -> List[str]:
        """
        Validate fields for profile update.

        Schema-related fields cannot be updated (create new profile instead).

        Args:
            update_fields: Dictionary of fields to update

        Returns:
            List of validation errors
        """
        errors = []

        # Fields that cannot be updated
        immutable_fields = {"schema_name", "embedding_model", "schema_config", "type"}

        for field in immutable_fields:
            if field in update_fields:
                errors.append(
                    f"Field '{field}' cannot be updated. "
                    "Create a new profile instead for schema changes."
                )

        return errors
