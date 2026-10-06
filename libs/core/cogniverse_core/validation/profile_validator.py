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
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

logger = logging.getLogger(__name__)

SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "config.json"
NO_SHIPPED_PROFILE_TYPES_ERROR = (
    "Profile type validation failed: no backend profile types are "
    "configured in configs/config.json backend.profiles"
)


class ProfileValidator:
    """Validates backend profile configurations."""

    # Valid embedding types — describes storage pattern only.
    # Model loading is driven by model_loader, not embedding_type.
    VALID_EMBEDDING_TYPES = [
        "multi_vector",
        "single_vector",
    ]

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
        (
            self._valid_profile_types,
            self._profile_type_source_error,
            self._model_loader_types,
        ) = self._load_valid_profile_types()
        self._valid_profile_type_set = frozenset(self._valid_profile_types)

    def _load_valid_profile_types(
        self,
    ) -> tuple[list[str], Optional[str], frozenset[str]]:
        """Derive valid profile types from the shipped backend config, with
        the types every shipped profile of which names a ``model_loader``:
        content of those types is embedded at ingestion, so a profile of one
        without a loader cannot be ingested into."""
        try:
            json_config = json.loads(SHIPPED_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [], NO_SHIPPED_PROFILE_TYPES_ERROR, frozenset()

        backend_config = json_config.get("backend")
        if not isinstance(backend_config, dict):
            return [], NO_SHIPPED_PROFILE_TYPES_ERROR, frozenset()

        profiles = backend_config.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            return [], NO_SHIPPED_PROFILE_TYPES_ERROR, frozenset()

        valid_profile_types: list[str] = []
        for profile_name, profile_config in profiles.items():
            if not isinstance(profile_config, dict):
                return (
                    [],
                    "Profile type validation failed: backend profile "
                    f"'{profile_name}' must be an object in "
                    "configs/config.json backend.profiles",
                    frozenset(),
                )

            profile_type = profile_config.get("type")
            if not isinstance(profile_type, str) or not profile_type.strip():
                return (
                    [],
                    "Profile type validation failed: backend profile "
                    f"'{profile_name}' must declare a non-empty string type in "
                    "configs/config.json backend.profiles",
                    frozenset(),
                )
            if profile_type != profile_type.strip():
                return (
                    [],
                    "Profile type validation failed: backend profile "
                    f"'{profile_name}' type contains surrounding whitespace in "
                    "configs/config.json backend.profiles",
                    frozenset(),
                )

            if profile_type not in valid_profile_types:
                valid_profile_types.append(profile_type)

        loaderless_types = {
            profile_config["type"]
            for profile_config in profiles.values()
            if not profile_config.get("model_loader")
        }
        model_loader_types = frozenset(valid_profile_types) - loaderless_types
        return valid_profile_types, None, model_loader_types

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

        if not is_update:
            errors.extend(self._validate_uniqueness(profile, tenant_id))

        errors.extend(self._validate_profile_name(profile.profile_name))
        errors.extend(self._validate_profile_type(profile.type))
        errors.extend(self._validate_schema_template(profile.schema_name))
        errors.extend(self._validate_embedding_model(profile.embedding_model))
        errors.extend(self._validate_embedding_type(profile.embedding_type))
        errors.extend(self._validate_model_loader(profile))
        errors.extend(self._validate_process_type(profile.process_type))
        errors.extend(self._validate_extra_config(profile.extra_config))
        errors.extend(self._validate_strategies(profile.strategies))
        errors.extend(self._validate_embedding_dimensions(profile))

        return errors

    def _validate_uniqueness(
        self, profile: "BackendProfileConfig", tenant_id: str
    ) -> List[str]:
        """Check if profile name is unique for tenant.

        Reads the tenant's backend config as the store holds it now, not the
        manager's held copy: a profile another process deleted moments ago
        must not refuse its re-creation here.
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        errors = []

        stored = self.config_manager.store.get_config(
            tenant_id=canonical_tenant_id(tenant_id),
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="backend_config",
        )
        profiles = {} if stored is None else stored.config_value.get("profiles", {})
        if profile.profile_name in profiles:
            errors.append(
                f"Profile '{profile.profile_name}' already exists for tenant '{tenant_id}'"
            )

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

        if self._profile_type_source_error:
            errors.append(self._profile_type_source_error)
            return errors

        if not profile_type:
            errors.append("Profile type is required")
            return errors

        if profile_type not in self._valid_profile_type_set:
            errors.append(
                f"Invalid profile type '{profile_type}'. "
                f"Must be one of: {self._valid_profile_types}"
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

        # Basic format check (allows "org/model" format like "TomoroAI/tomoro-colqwen3-embed-4b")
        if "/" not in embedding_model and "-" not in embedding_model:
            logger.warning(
                f"Embedding model '{embedding_model}' has unusual format. "
                "Expected format: 'org/model' or 'model-name'"
            )

        return errors

    def _validate_model_loader(self, profile: "BackendProfileConfig") -> List[str]:
        """The loader must be one ingestion embeds with, and is required for
        the types whose content is embedded."""
        from cogniverse_core.common.models.model_loaders import (
            EMBEDDING_MODEL_LOADERS,
        )

        if not profile.model_loader:
            if profile.type in self._model_loader_types:
                return [
                    f"Profile type '{profile.type}' requires a model_loader, "
                    f"one of: {sorted(EMBEDDING_MODEL_LOADERS)}"
                ]
            return []
        if profile.model_loader not in EMBEDDING_MODEL_LOADERS:
            return [
                f"Invalid model_loader '{profile.model_loader}'. "
                f"Must be one of: {sorted(EMBEDDING_MODEL_LOADERS)}"
            ]
        return []

    def _validate_process_type(self, process_type: Optional[str]) -> List[str]:
        """An unset process type lets ingestion infer it from the profile."""
        from cogniverse_foundation.config.unified_config import PROCESS_TYPES

        if process_type is None or process_type in PROCESS_TYPES:
            return []
        return [
            f"Invalid process_type '{process_type}'. "
            f"Must be one of: {list(PROCESS_TYPES)}"
        ]

    def _validate_extra_config(self, extra_config: dict) -> List[str]:
        """Extra keys sit beside the named fields in the stored profile, so
        one named like a field would overwrite it."""
        from cogniverse_foundation.config.unified_config import (
            BackendProfileConfig,
        )

        clashing = sorted(set(extra_config) & BackendProfileConfig._KNOWN_KEYS)
        if clashing:
            return [
                f"extra_config keys {clashing} are profile fields; "
                "set them as fields instead"
            ]
        return []

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
                    "cogniverse_runtime.ingestion.strategies",
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

    def _validate_embedding_dimensions(
        self, profile: "BackendProfileConfig"
    ) -> List[str]:
        """Validate embedding dimensions match schema."""
        errors = []

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
        immutable_fields = {
            "schema_name",
            "embedding_model",
            "schema_config",
            "type",
            "model_loader",
        }

        for field in immutable_fields:
            if field in update_fields:
                errors.append(
                    f"Field '{field}' cannot be updated. "
                    "Create a new profile instead for schema changes."
                )

        # Validate the VALUES of mutable fields too — otherwise an update can
        # write a malformed strategies block that create-time validation would
        # have rejected.
        if "strategies" in update_fields:
            errors.extend(self._validate_strategies(update_fields["strategies"]))

        return errors
