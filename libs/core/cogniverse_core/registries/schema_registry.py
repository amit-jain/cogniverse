"""
Schema Registry - Tracks deployed schemas across tenants

Facade over ConfigManager that provides schema-specific operations.
Ensures all schemas are tracked and can be redeployed together.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cogniverse_core.registries.exceptions import (
    BackendDeploymentError,
    RegistryStorageError,
    SchemaConvergenceError,
    SchemaLoadError,
    SchemaRegistryInitializationError,
)
from cogniverse_core.registries.schema_deployment_intents import SchemaDeploymentIntents

logger = logging.getLogger(__name__)

_SCHEMA_INTENT_GRACE_S = 90


@dataclass
class SchemaInfo:
    """Information about a deployed schema"""

    tenant_id: str
    base_schema_name: str
    full_schema_name: str
    schema_definition: str
    config: Dict[str, Any]
    deployment_time: str


class SchemaRegistry:
    """
    Registry for tracking deployed schemas.

    Facade over ConfigManager that provides schema-specific operations.
    Ensures all schemas are tracked and can be redeployed together.

    This prevents the critical schema wipeout bug where deploying a new schema
    would inadvertently delete all existing schemas from Vespa.

    Usage:
        from cogniverse_foundation.config.utils import create_default_config_manager
        config_manager = create_default_config_manager()
        registry = SchemaRegistry(config_manager, backend, schema_loader)

        # Register a newly deployed schema
        registry.register_schema(
            tenant_id="test_tenant",
            base_schema_name="video_colpali_smol500_mv_frame",
            full_schema_name="video_colpali_smol500_mv_frame_test_tenant",
            schema_definition=str(schema),
            config={"profile": "video_colpali_smol500_mv_frame"}
        )

        # Get all schemas for a tenant before redeployment
        schemas = registry.get_tenant_schemas("test_tenant")

        # Check if schema already deployed
        exists = registry.schema_exists("test_tenant", "video_colpali_smol500_mv_frame")
    """

    # Serializes deploy→register across every registry instance in the
    # process. Vespa activation (deploy_schemas) makes a schema live seconds
    # before it is recorded in the ConfigStore (register_schema); a concurrent
    # deploy that observed the live-but-unregistered schema in that gap would
    # treat it as an orphan. Holding this lock across both steps closes the gap.
    _deploy_lock = threading.RLock()
    # Serializes startup reads so concurrent backend initialization does not
    # stampede the shared storage read.
    _storage_read_lock = threading.RLock()

    def __init__(self, config_manager, backend, schema_loader):
        """
        Initialize SchemaRegistry with required dependencies.

        All parameters are REQUIRED - no optional dependencies.
        Fail fast at construction if dependencies are missing.

        Args:
            config_manager: ConfigManager instance (REQUIRED)
            backend: Backend instance for schema deployment (REQUIRED)
            schema_loader: SchemaLoader instance for loading schema definitions (REQUIRED)

        Raises:
            ValueError: If any required parameter is None
            SchemaRegistryInitializationError: If schema loading fails
        """
        if config_manager is None:
            raise ValueError("config_manager is required")
        if backend is None:
            raise ValueError("backend is required")
        if schema_loader is None:
            raise ValueError("schema_loader is required")

        self._config_manager = config_manager
        self._backend = backend
        self._schema_loader = schema_loader
        self._deployment_intents = SchemaDeploymentIntents(config_manager.store)

        # In-memory registry of all deployed schemas
        # Key: (tenant_id, base_schema_name), Value: SchemaInfo
        self._schemas: Dict[tuple, SchemaInfo] = {}

        # Load all previously deployed schemas from persistent storage
        self._load_schemas_from_storage()

    def _load_schemas_from_storage(self):
        """
        Load all schemas from ConfigManager into in-memory registry on startup.

        Behavior:
        - Empty storage is valid and loads as an empty registry.
        - A genuine HTTP 404 from the storage read is normalized by the store.
        - Storage failures are wrapped once with SchemaRegistry context.

        Raises:
            SchemaRegistryInitializationError: If storage cannot be read
        """
        with SchemaRegistry._storage_read_lock:
            try:
                # Load all schemas across all tenants using generic ConfigManager methods
                from cogniverse_sdk.interfaces.config_store import ConfigScope

                all_schema_data = self._config_manager.store.list_all_configs(
                    scope=ConfigScope.SCHEMA,
                    service="schema_registry",
                )
            except Exception as exc:
                message = (
                    "Cannot initialize SchemaRegistry: failed to read schema storage: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(message)
                raise SchemaRegistryInitializationError(message) from exc

            # Rebuild into a fresh dict so a peer's deletions are reflected on
            # reload; swap in only after a successful load so a failure falls
            # back to the existing cache rather than wiping it.
            loaded: Dict[tuple, SchemaInfo] = {}
            for entry in all_schema_data:
                schema_data = entry.config_value
                # Skip deleted schemas
                if schema_data.get("deleted", False):
                    continue

                tenant_id = schema_data["tenant_id"]
                base_schema_name = schema_data["base_schema_name"]
                key = (tenant_id, base_schema_name)

                loaded[key] = SchemaInfo(
                    tenant_id=tenant_id,
                    base_schema_name=base_schema_name,
                    full_schema_name=schema_data["full_schema_name"],
                    schema_definition=schema_data["schema_definition"],
                    config=schema_data.get("config", {}),
                    deployment_time=schema_data["deployment_time"],
                )

            self._schemas = loaded
            logger.info(f"Loaded {len(loaded)} schemas from storage")
            return

    def register_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        full_schema_name: str,
        schema_definition: str,
        config: Optional[Dict[str, Any]] = None,
        deployment_time: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> None:
        """
        Register a newly deployed schema.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name (e.g., 'video_colpali_smol500_mv_frame')
            full_schema_name: Tenant-scoped schema name (e.g., 'video_colpali_smol500_mv_frame_test_tenant')
            schema_definition: Full .sd file content as string
            config: Optional schema configuration
            deployment_time: Original intent timestamp when completing a deployment
            expected_version: Conditional registration version; zero requires absence

        Example:
            registry.register_schema(
                tenant_id="test_tenant",
                base_schema_name="video_colpali_smol500_mv_frame",
                full_schema_name="video_colpali_smol500_mv_frame_test_tenant",
                schema_definition=str(schema),
                config={"profile": "video_colpali_smol500_mv_frame"}
            )
        """
        from datetime import datetime, timezone

        from cogniverse_core.common.tenant_utils import canonical_tenant_id
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        # Canonicalize so register/lookup/deploy paths converge on the
        # same storage key (matches deploy_schema's canonicalization).
        tenant_id = canonical_tenant_id(tenant_id)

        logger.info(
            f"Registering schema '{base_schema_name}' for tenant '{tenant_id}' "
            f"(full name: '{full_schema_name}')"
        )

        # Store schema metadata using generic ConfigManager storage
        config_key = f"schema_{base_schema_name}"
        if deployment_time is None:
            deployment_time = datetime.now(timezone.utc).isoformat()

        value = {
            "tenant_id": tenant_id,
            "base_schema_name": base_schema_name,
            "full_schema_name": full_schema_name,
            "schema_definition": schema_definition,
            "config": config or {},
            "deployment_time": deployment_time,
        }

        coordinates = {
            "tenant_id": tenant_id,
            "scope": ConfigScope.SCHEMA,
            "service": "schema_registry",
            "config_key": config_key,
        }
        if expected_version is None:
            self._config_manager.store.set_config(**coordinates, config_value=value)
        else:
            saved = self._config_manager.store.compare_and_set_config(
                **coordinates, config_value=value, expected_version=expected_version
            )
            if saved is None:
                current = self._config_manager.store.get_config(**coordinates)
                if current is None or current.config_value != value:
                    raise RegistryStorageError(
                        f"Registration of {full_schema_name!r} conflicted with a newer registry revision"
                    )

        # Add to in-memory registry
        key = (tenant_id, base_schema_name)
        self._schemas[key] = SchemaInfo(
            tenant_id=tenant_id,
            base_schema_name=base_schema_name,
            full_schema_name=full_schema_name,
            schema_definition=schema_definition,
            config=config or {},
            deployment_time=deployment_time,
        )

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """
        Validate tenant ID format.

        Args:
            tenant_id: Tenant identifier to validate

        Raises:
            ValueError: If tenant_id is invalid
            TypeError: If tenant_id is not a string
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not isinstance(tenant_id, str):
            raise TypeError(f"tenant_id must be string, got {type(tenant_id)}")

        # No hyphens: the tenant id becomes part of the Vespa schema name
        # ([a-zA-Z0-9_] only) and sanitizing "-"→"_" would collide distinct
        # tenants (acme-corp vs acme_corp → same schema). Matches the
        # tenant-creation contract (validate_tenant_name / validate_org_id).
        # The ':' separates the org:tenant canonical form.
        import re

        if not re.match(r"^[a-zA-Z0-9_:]+$", tenant_id):
            raise ValueError(
                f"Invalid tenant_id '{tenant_id}': only alphanumeric, underscore, "
                "and colon allowed"
            )

    def _validate_schema_name(self, schema_name: str) -> None:
        """
        Validate schema name format.

        Args:
            schema_name: Schema name to validate

        Raises:
            ValueError: If schema_name is invalid
            TypeError: If schema_name is not a string
        """
        if not schema_name:
            raise ValueError("schema_name is required")
        if not isinstance(schema_name, str):
            raise TypeError(f"schema_name must be string, got {type(schema_name)}")

    def _rollback_deployment(
        self, previous_schemas: List[Dict[str, Any]], failed_schema_name: str
    ) -> None:
        """
        Rollback failed schema deployment by re-deploying previous schema set.

        This method is called when backend deployment succeeds but ConfigStore
        registration fails. It restores the backend to its previous state by
        re-deploying the old schema list (without the failed schema).

        The rollback uses backend.deploy_schemas() with the old list, which
        implicitly removes the newly deployed schema. This maintains consistency
        between backend and ConfigStore.

        Args:
            previous_schemas: List of schema definitions that existed before deployment
            failed_schema_name: Name of schema that failed registration (for logging)

        Note:
            Rollback is best-effort. If rollback fails, manual intervention is required.
            The system logs detailed error information to aid recovery.
        """
        logger.warning(
            f"Rolling back deployment of '{failed_schema_name}'. "
            f"Re-deploying {len(previous_schemas)} previous schemas."
        )

        try:
            # Re-deploy previous schema set (removes failed schema implicitly)
            success = self._backend.deploy_schemas(previous_schemas)
            if success:
                logger.info(
                    f"Successfully rolled back '{failed_schema_name}'. "
                    f"Backend state restored to {len(previous_schemas)} schemas."
                )
            else:
                logger.error(
                    f"Rollback of '{failed_schema_name}' reported failure. "
                    f"Backend state may be inconsistent. Manual intervention required."
                )
        except Exception as e:
            logger.error(
                f"Rollback of '{failed_schema_name}' failed with exception: {e}. "
                f"Backend state is inconsistent. Manual intervention required. "
                f"Expected schemas: {[s['name'] for s in previous_schemas]}"
            )

    def deploy_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        config: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> str:
        """Deploy and register one tenant schema through the batch path."""
        return self.deploy_schemas(tenant_id, [base_schema_name], config, force)[0]

    def deploy_schemas(
        self,
        tenant_id: str,
        base_schema_names: List[str],
        config: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> List[str]:
        """Journal every new schema, activate once, then register the batch.

        Returns full names in request order, including already registered names.
        All intents stay pending until every registration succeeds. The backend
        owns package reconstruction and the single convergence wait.
        """
        import json
        from collections import Counter
        from datetime import datetime, timezone

        from cogniverse_core.common.tenant_utils import canonical_tenant_id
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        self._validate_tenant_id(tenant_id)
        if not base_schema_names:
            raise ValueError("base_schema_names is required")
        for base in base_schema_names:
            self._validate_schema_name(base)
        duplicates = sorted(
            base for base, count in Counter(base_schema_names).items() if count != 1
        )
        if duplicates:
            raise ValueError(f"Duplicate base schema names: {duplicates}")
        tenant_id = canonical_tenant_id(tenant_id)
        names = [f"{base}_{tenant_id.replace(':', '_')}" for base in base_schema_names]

        def registered(base, name):
            return (
                not force
                and self.schema_exists(tenant_id, base)
                and self._schemas[(tenant_id, base)].full_schema_name == name
            )

        def load_definition(base, name):
            try:
                definition = self._schema_loader.load_schema(base)
            except Exception as exc:
                raise SchemaLoadError(
                    f"Failed to load base schema '{base}': {exc}"
                ) from exc
            definition["name"] = name
            return json.dumps(definition)

        definitions = {
            base: load_definition(base, name)
            for base, name in zip(base_schema_names, names)
            if not registered(base, name)
        }
        with SchemaRegistry._deploy_lock:
            existing_schemas = self._get_all_schemas()
            requested = [
                (base, name)
                for base, name in zip(base_schema_names, names)
                if not registered(base, name)
            ]
            if not requested:
                return names
            previous_schemas = [
                {
                    "name": info.full_schema_name,
                    "definition": info.schema_definition,
                    "tenant_id": info.tenant_id,
                    "base_schema_name": info.base_schema_name,
                }
                for info in existing_schemas
            ]
            registrations = []
            intents = {}
            for base, name in requested:
                if base not in definitions:
                    definitions[base] = load_definition(base, name)
                for existing in existing_schemas:
                    if existing.full_schema_name == name and (
                        existing.tenant_id,
                        existing.base_schema_name,
                    ) != (tenant_id, base):
                        raise RegistryStorageError(
                            f"Schema name {name!r} belongs to another tenant"
                        )
                registration = {
                    "tenant_id": tenant_id,
                    "base_schema_name": base,
                    "full_schema_name": name,
                    "schema_definition": definitions[base],
                    "config": config or {},
                    "deployment_time": datetime.now(timezone.utc).isoformat(),
                }
                stored = self._config_manager.store.get_config(
                    tenant_id=tenant_id,
                    scope=ConfigScope.SCHEMA,
                    service="schema_registry",
                    config_key=f"schema_{base}",
                )
                if stored is None or stored.config_value.get("deleted", False):
                    intent = self._deployment_intents.prepare(
                        registration,
                        grace_s=_SCHEMA_INTENT_GRACE_S,
                        registry_version=0 if stored is None else stored.version,
                    )
                    intents[name] = intent
                    registration = intent["registration"]
                registrations.append(registration)

            replacing = {row["full_schema_name"] for row in registrations}
            all_schemas = [
                schema for schema in previous_schemas if schema["name"] not in replacing
            ]
            all_schemas.extend(
                {
                    "name": row["full_schema_name"],
                    "definition": row["schema_definition"],
                    "tenant_id": tenant_id,
                    "base_schema_name": row["base_schema_name"],
                }
                for row in registrations
            )
            subject = (
                f"schema '{names[0]}'"
                if len(names) == 1
                else f"tenant '{tenant_id}' schemas {base_schema_names}"
            )
            try:
                if not self._backend.deploy_schemas(all_schemas):
                    raise BackendDeploymentError(f"Backend failed to deploy {subject}")
            except Exception as exc:
                activated = isinstance(exc, SchemaConvergenceError)
                deployment_error = BackendDeploymentError(
                    f"Backend deployment failed for {subject}: {exc}. "
                    + (
                        "The schema is live; its registration completes by recovery."
                        if activated and intents
                        else "The durable definition is retained for late activation."
                    )
                )
                if not activated:
                    for intent in intents.values():
                        try:
                            self._deployment_intents.retire(intent)
                        except Exception as retirement_exc:
                            detail = f"Intent retirement failed: {retirement_exc}. The durable record is retained for recovery."
                            deployment_error.add_note(detail)
                            logger.error(detail)
                raise deployment_error from exc

            for registration in registrations:
                name = registration["full_schema_name"]
                intent = intents.get(name)
                try:
                    if intent:
                        self.register_schema(
                            **registration, expected_version=intent["registry_version"]
                        )
                    else:
                        self.register_schema(**registration)
                except Exception as exc:
                    if intents:
                        detail = "Durable registration recovery is pending; the schema is preserved."
                    else:
                        self._rollback_deployment(previous_schemas, name)
                        detail = "Existing-schema deployment rollback was requested."
                    raise RegistryStorageError(
                        f"Failed to register schema '{name}' in ConfigStore: {exc}. {detail}"
                    ) from exc
            for intent in intents.values():
                self._deployment_intents.complete(intent)
        return names

    def reconcile_deployment_intents(self, live_names: set[str]) -> List[SchemaInfo]:
        """Complete due intents for schemas confirmed live by the config server.

        Recovery writes the owner's exact registry payload and never deletes or
        deploys schemas. Fresh intents wait 90 seconds for normal registration.
        """
        from dataclasses import asdict

        self._load_schemas_from_storage()
        registered = {
            info.full_schema_name: asdict(info) for info in self._schemas.values()
        }
        recovered = self._deployment_intents.reconcile(
            live_names,
            registered,
            lambda row, version: self.register_schema(**row, expected_version=version),
        )
        return [SchemaInfo(**row) for row in recovered]

    def reserved_schemas(self, live_names: set[str]) -> Dict[str, Dict[str, Any]]:
        """Full schema names owned by an activation in flight, with their
        exact registration payloads (see ``SchemaDeploymentIntents.reserved``).

        Any process rebuilding the application package must keep these as
        survivors: they are registrations in progress in another process,
        not orphans, and their registry record does not exist yet.
        """
        return self._deployment_intents.reserved(live_names)

    def get_tenant_schemas(self, tenant_id: str) -> List[SchemaInfo]:
        """
        Get all schemas deployed for a specific tenant.

        This is the critical method that prevents schema wipeout.
        Before deploying a new schema, call this to get ALL existing schemas
        and include them in the deployment.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of SchemaInfo objects for all deployed schemas

        Example:
            # Get all existing schemas before deploying new one
            existing_schemas = registry.get_tenant_schemas("test_tenant")

            # Create ApplicationPackage with ALL schemas
            app_package = ApplicationPackage("cogniverse")

            # Add the new schema
            app_package.add_schema(new_schema)

            # Add ALL existing schemas from registry
            for schema_info in existing_schemas:
                existing_schema = reconstruct_schema(schema_info.schema_definition)
                app_package.add_schema(existing_schema)

            # Now deployment won't wipe existing schemas
            deploy_package(app_package)
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        tenant_id = canonical_tenant_id(tenant_id)
        try:
            self._load_schemas_from_storage()
        except Exception as exc:
            logger.warning(
                f"get_tenant_schemas: refresh from storage failed, "
                f"falling back to in-memory cache: {exc}"
            )
        return [
            schema_info
            for (tid, _), schema_info in self._schemas.items()
            if tid == tenant_id
        ]

    def _get_all_schemas(self) -> List[SchemaInfo]:
        """
        Get all deployed schemas across all tenants (PRIVATE - internal use only).

        Used internally by deploy_schema() to collect existing schemas before deployment.
        This method provides cross-tenant schema access for backends that require
        all schemas to be redeployed together.

        Reloads from persistent storage on every call so peer SchemaRegistry
        instances (e.g., the graph backend's registry vs. the ingestion
        backend's registry) see schemas registered by the other since their
        last refresh. Without this, the docs-then-graph flow races: ingestion
        registers ``document_text_<tenant>`` in DB, graph deploy reads its
        own stale in-memory dict, the deploy package omits document_text,
        and Vespa rejects the deploy as "schema removal".

        Returns:
            List of all SchemaInfo objects across all tenants

        Note:
            This is a private method. External code should NOT call this directly.
            Use deploy_schema() for orchestrated deployment instead.
        """
        try:
            self._load_schemas_from_storage()
        except Exception as exc:
            logger.warning(
                f"_get_all_schemas: refresh from storage failed, "
                f"falling back to in-memory cache: {exc}"
            )
        return list(self._schemas.values())

    def schema_exists(self, tenant_id: str, base_schema_name: str) -> bool:
        """
        Check if schema already deployed for tenant.

        A cache hit answers from memory; a miss re-reads persistent storage
        first, so schemas registered by peer processes are visible.
        Use this before deploying to avoid unnecessary redeployments.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name

        Returns:
            True if schema is deployed and not deleted, False otherwise

        Example:
            if not registry.schema_exists("test_tenant", "video_colpali_smol500_mv_frame"):
                # Deploy schema
                deploy_schema(...)
                # Register after successful deployment
                registry.register_schema(...)
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        tenant_id = canonical_tenant_id(tenant_id)
        key = (tenant_id, base_schema_name)
        if key in self._schemas:
            return True
        # A miss is not authoritative: a peer process (another runtime
        # replica, a host-side manager) may have deployed and registered the
        # schema since this registry last read storage. Re-read, then answer.
        # A storage outage during the re-read raises (strict mode) — an
        # outage must never read as "not deployed".
        self._load_schemas_from_storage()
        return key in self._schemas

    def unregister_schema(self, tenant_id: str, base_schema_name: str) -> None:
        """
        Remove schema from registry (when deleted from backend).

        Marks schema as deleted in persistent storage and removes from in-memory registry.
        This preserves audit trail in persistent storage.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name

        Example:
            # After deleting schema from Vespa
            delete_schema_from_vespa(...)
            # Unregister from registry
            registry.unregister_schema("test_tenant", "video_colpali_smol500_mv_frame")
        """
        from datetime import datetime, timezone

        from cogniverse_core.common.tenant_utils import canonical_tenant_id
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        # Canonicalize so register/deploy/exists/unregister all converge
        # on the same storage key.
        tenant_id = canonical_tenant_id(tenant_id)

        logger.info(
            f"Unregistering schema '{base_schema_name}' for tenant '{tenant_id}'"
        )

        # Mark schema as deleted in persistent storage (preserves audit trail)
        config_key = f"schema_{base_schema_name}"

        # Get existing entry to mark as deleted
        entry = self._config_manager.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=config_key,
        )

        schema_info = (
            dict(entry.config_value)
            if entry
            else {
                "tenant_id": tenant_id,
                "base_schema_name": base_schema_name,
                "full_schema_name": f"{base_schema_name}_{tenant_id.replace(':', '_')}",
            }
        )
        schema_info["deleted"] = True
        schema_info["deleted_at"] = datetime.now(timezone.utc).isoformat()
        self._config_manager.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=config_key,
            config_value=schema_info,
        )

        # Remove from in-memory registry
        key = (tenant_id, base_schema_name)
        if key in self._schemas:
            del self._schemas[key]

    def unregister_tenant_schemas(self, tenant_id: str) -> None:
        """
        Remove all schemas for a tenant.

        Call this when deleting a tenant to clean up their schemas.

        Args:
            tenant_id: Tenant identifier

        Example:
            # When deleting a tenant
            delete_tenant_from_vespa(...)
            registry.unregister_tenant_schemas("test_tenant")
        """
        logger.info(f"Unregistering all schemas for tenant '{tenant_id}'")
        schemas = self.get_tenant_schemas(tenant_id)
        for schema in schemas:
            self.unregister_schema(tenant_id, schema.base_schema_name)
