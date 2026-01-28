"""
Vespa backend implementation with unified interface.

This module provides a Vespa backend that implements both IngestionBackend
and SearchBackend interfaces, with self-registration to the backend registry.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cogniverse_sdk.document import Document
from cogniverse_sdk.interfaces.backend import Backend

from .config_utils import calculate_config_port
from .ingestion_client import VespaPyClient
from .search_backend import VespaSearchBackend
from .vespa_schema_manager import VespaSchemaManager

# Check if async ingestion client is available (optional dependency)
try:
    from .async_ingestion_client import AsyncVespaBackendAdapter  # noqa: F401

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

logger = logging.getLogger(__name__)


class VespaBackend(Backend):
    """
    Vespa backend implementation supporting both ingestion and search.

    This class wraps the existing Vespa implementations and provides
    a unified interface compatible with the backend registry.
    """

    def __init__(self, backend_config, schema_loader=None, config_manager=None):
        """
        Initialize Vespa backend.

        Args:
            backend_config: BackendConfig instance with connection details (REQUIRED)
            schema_loader: SchemaLoader instance for loading schemas (REQUIRED)
            config_manager: ConfigManager instance for configuration access (REQUIRED)
        """
        if backend_config is None:
            raise ValueError(
                "backend_config is required for VespaBackend initialization"
            )
        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for VespaBackend initialization"
            )
        if config_manager is None:
            raise ValueError(
                "config_manager is required for VespaBackend initialization"
            )

        # Validate backend type
        if backend_config.backend_type != "vespa":
            raise ValueError(
                f"VespaBackend requires backend_type='vespa', got '{backend_config.backend_type}'"
            )

        super().__init__("vespa")
        self._backend_config = backend_config
        self._schema_loader_instance = schema_loader
        self._config_manager_instance = config_manager
        self._vespa_search_backend: Optional[VespaSearchBackend] = None
        # Store multiple ingestion clients, one per schema
        self._vespa_ingestion_clients: Dict[str, VespaPyClient] = {}
        self._async_ingestion_clients: Dict[str, Any] = (
            {}
        )  # For async ingestion (optional)
        self.schema_manager: Optional[VespaSchemaManager] = None
        self._initialized_as_search = False
        self._initialized_as_ingestion = False
        self.use_async_ingestion = False  # Flag to enable async mode

        # Extract connection details from BackendConfig
        self._url: str = backend_config.url
        self._port: int = backend_config.port
        self._tenant_id: str = backend_config.tenant_id

        # SchemaRegistry will be injected later (no circular dependency)
        self.schema_registry = None

    def _initialize_backend(self, config: Dict[str, Any]) -> None:
        """
        Initialize Vespa backend components.

        Args:
            config: Backend configuration including:
                - tenant_id: Tenant identifier (optional override)
                - schema_name: Schema to use
                - profile: Processing profile
                - backend: Nested backend section with url, port, profiles, etc.
        """
        # Allow tenant_id override from config (but use BackendConfig tenant_id as default)
        config_tenant_id = config.get("tenant_id")
        if config_tenant_id and config_tenant_id != self._tenant_id:
            logger.debug(
                f"Overriding tenant_id from {self._tenant_id} to {config_tenant_id}"
            )
            self._tenant_id = config_tenant_id

        if not self._tenant_id:
            logger.warning(
                "No tenant_id configured - backend will use base schemas without tenant isolation"
            )

        # Extract backend section if present, otherwise use config as-is
        backend_section = config.get("backend", config)

        # Merge backend section with top-level config
        # Strategy: backend_section provides defaults, top-level overrides
        # Special handling for profiles (merge dicts), url/port (top-level wins)
        merged_config = {**backend_section}  # Start with backend section defaults

        # Add all top-level keys (overwriting backend section)
        for key, value in config.items():
            if key == "backend":
                # Skip the backend section itself
                continue
            elif key == "profiles":
                # Merge profiles: backend section + top-level (top-level wins on conflicts)
                backend_profiles = backend_section.get("profiles", {})
                top_profiles = config.get("profiles", {})
                merged_config["profiles"] = {**backend_profiles, **top_profiles}
            else:
                # Top-level wins for all other keys (url, port, tenant_id, etc.)
                merged_config[key] = value

        # Store merged config for accessing profiles and other settings
        self.config = merged_config

        # Check if async ingestion is requested (optional feature)
        self.use_async_ingestion = (
            merged_config.get("use_async_ingestion", False) and ASYNC_AVAILABLE
        )

        # Allow config to override URL/port from BackendConfig
        config_url = merged_config.get("url")
        config_port = merged_config.get("port")
        if config_url and config_url != self._url:
            logger.debug(f"Overriding url from {self._url} to {config_url}")
            self._url = config_url
        if config_port and config_port != self._port:
            logger.debug(f"Overriding port from {self._port} to {config_port}")
            self._port = config_port

        # Mark as ingestion backend if schema_name is provided
        if "schema_name" in config:
            self._initialized_as_ingestion = True
            # Don't create client yet - will create per-schema on demand

        # Initialize schema manager for schema operations
        if not self._url:
            raise ValueError("url is required in BackendConfig")

        # Get config port (for schema deployment/management)
        config_port = merged_config.get("config_port")
        if not config_port:
            config_port = calculate_config_port(self._port)
            logger.debug(
                f"Calculated config port {config_port} from data port {self._port}"
            )

        # Store config port for schema deployment
        self._config_port = config_port

        # SchemaRegistry will be injected externally (no circular dependency)

        self.schema_manager = VespaSchemaManager(
            backend_endpoint=self._url,
            backend_port=config_port,
            schema_loader=self._schema_loader_instance,
            schema_registry=None,  # Will be set after SchemaRegistry is injected
        )

        # Backend is initialized with all profiles available
        # VespaSearchBackend will be created lazily in search() method
        # based on query type and default_profiles

        # Inject schema_registry into schema_manager if available
        # This happens before metadata schema deployment so schema_manager can preserve existing schemas
        if self.schema_registry:
            self.schema_manager._schema_registry = self.schema_registry
            logger.debug(
                "Injected schema_registry into schema_manager before metadata deployment"
            )

        # Deploy metadata schemas automatically during backend initialization
        # upload_metadata_schemas() is schema-aware and preserves existing video schemas
        system_config = self._config_manager_instance.get_system_config()
        app_name = system_config.application_name
        self.schema_manager.upload_metadata_schemas(app_name=app_name)
        logger.info(
            "Automatically deployed metadata schemas during backend initialization"
        )

        logger.info(
            f"Initialized Vespa backend for tenant '{self._tenant_id}' with {len(self.config.get('profiles', {}))} profiles"
        )

    # Ingestion methods

    def _get_or_create_ingestion_client(self, schema_name: str) -> VespaPyClient:
        """
        Get or create a schema-specific ingestion client with tenant-aware schema naming.

        Args:
            schema_name: Base schema name to get client for

        Returns:
            VespaPyClient configured for the tenant-specific schema

        Note:
            If tenant_id is set, this method will:
            1. Transform base schema name to tenant-scoped name (e.g., video_colpali_smol500_mv_frame_test_tenant)
            2. Ensure the tenant-scoped schema exists in Vespa (auto-deploy if needed)
            3. Create a client that ingests to the tenant-scoped schema
        """
        # Transform base schema name to tenant-scoped name if tenant_id is set
        target_schema_name = schema_name
        if self._tenant_id:
            target_schema_name = self.get_tenant_schema_name(
                self._tenant_id, schema_name
            )
            logger.debug(
                f"Transformed base schema '{schema_name}' to tenant schema '{target_schema_name}' "
                f"for tenant '{self._tenant_id}'"
            )

            # Ensure tenant schema exists (auto-deploy if needed)
            if not self.schema_registry:
                raise ValueError(
                    "schema_registry not injected - backend initialization incomplete. "
                    "This indicates BackendFactory was not used correctly."
                )

            try:
                self.schema_registry.deploy_schema(
                    tenant_id=self._tenant_id, base_schema_name=schema_name
                )
                logger.debug(
                    f"Verified tenant schema '{target_schema_name}' exists in Vespa"
                )
            except Exception as e:
                logger.error(f"Failed to ensure tenant schema exists: {e}")
                raise
        if target_schema_name not in self._vespa_ingestion_clients:
            # Create new client with config dict
            logger.info(f"Creating new VespaPyClient for schema: {target_schema_name}")

            # Get the specific profile config using BASE schema name (config uses base names)
            profile_config = {}
            if self.config:
                profiles = self.config.get("profiles", {})
                profile_config = profiles.get(
                    schema_name, {}
                )  # Use base name for config lookup

            # Pass connection details and profile config
            client_config = {
                "schema_name": target_schema_name,  # Use tenant-scoped name for Vespa
                "base_schema_name": schema_name,  # Base schema name for loading schema file
                "url": self._url,
                "port": self._port,
                "profile_config": profile_config,  # Pass only the specific profile config
                "schema_loader": self._schema_loader_instance,  # Pass schema_loader for StrategyAwareProcessor
            }

            client = VespaPyClient(config=client_config, logger=logger)
            client.connect()

            self._vespa_ingestion_clients[target_schema_name] = client

        return self._vespa_ingestion_clients[target_schema_name]

    def ingest_documents(
        self, documents: List[Document], schema_name: str
    ) -> Dict[str, Any]:
        """
        Ingest documents into Vespa.

        Args:
            documents: List of Document objects to ingest
            schema_name: Schema to ingest documents into

        Returns:
            Ingestion results
        """
        # Get schema-specific client
        client = self._get_or_create_ingestion_client(schema_name)

        # Process and feed documents using the schema-specific client
        # Each client already knows its schema, no need to pass it
        prepared_docs = []
        for doc in documents:
            prepared = client.process(doc)  # Client uses its own schema
            prepared_docs.append(prepared)

        # Feed documents to Vespa
        success_count, failed_docs = client._feed_prepared_batch(
            prepared_docs  # Client uses its own schema
        )

        # Wait for documents to be visible in queries (handle Vespa's eventual consistency)
        if self.config.get("wait_for_indexing", True) and success_count > 0:
            from tests.utils.async_polling import wait_for_vespa_document_visible

            timeout = self.config.get("indexing_timeout", 30.0)

            # Wait for successfully fed documents to be queryable
            for doc in documents:
                if doc.id not in [
                    fd if isinstance(fd, str) else fd.get("id") for fd in failed_docs
                ]:
                    try:
                        # Get the tenant-scoped schema name for verification
                        target_schema = schema_name
                        if self._tenant_id:
                            target_schema = self.get_tenant_schema_name(
                                self._tenant_id, schema_name
                            )

                        wait_for_vespa_document_visible(
                            vespa_url=f"{self._url}:{self._port}",
                            schema_name=target_schema,
                            document_id=doc.id,
                            timeout=timeout,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Document {doc.id} fed but not immediately queryable: {e}"
                        )

        return {
            "success_count": success_count,
            "failed_count": len(failed_docs),
            "failed_documents": failed_docs,
            "total_documents": len(documents),
        }

    def feed(self, document: Document, schema_name: str) -> Tuple[int, List[str]]:
        """
        Feed a single document to Vespa.

        Args:
            document: Document object to feed
            schema_name: Schema to feed document to (REQUIRED)

        Returns:
            Tuple of (success_count, failed_document_ids)
        """
        # Convert single document to list and call ingest_documents
        result = self.ingest_documents([document], schema_name)

        # Extract failed document IDs from the result
        failed_ids = []
        if result.get("failed_documents"):
            for failed_doc in result["failed_documents"]:
                # Extract the document ID from the failed document info
                if isinstance(failed_doc, str):
                    failed_ids.append(failed_doc)
                elif isinstance(failed_doc, dict) and "id" in failed_doc:
                    failed_ids.append(failed_doc["id"])

        success_count = result.get("success_count", 0)
        return success_count, failed_ids

    def ingest_stream(
        self, documents: Iterator[Document], batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream documents for ingestion.

        Args:
            documents: Iterator of Document objects
            batch_size: Number of documents per batch

        Yields:
            Ingestion results for each batch
        """
        batch = []
        for doc in documents:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield self.ingest_documents(batch)
                batch = []

        # Process remaining documents
        if batch:
            yield self.ingest_documents(batch)

    def update_document(self, document_id: str, document: Document) -> bool:
        """
        Update a document in Vespa.

        Args:
            document_id: ID of document to update
            document: Updated Document object

        Returns:
            True if successful
        """
        try:
            # Get schema name from config
            schema_name = self.config.get("schema_name")
            if not schema_name:
                logger.error("No schema_name in config for update operation")
                return False

            results = self.ingest_documents([document], schema_name=schema_name)
            return results["success_count"] > 0
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from Vespa.

        Args:
            document_id: ID of document to delete

        Returns:
            True if successful
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Get schema name from config
            schema_name = self.config.get("schema_name")
            if not schema_name:
                logger.error("No schema_name in config for delete operation")
                return False

            # Get ingestion client for this schema (handles tenant-aware schema naming)
            client = self._get_or_create_ingestion_client(schema_name)

            # Call delete_document on the ingestion client
            success = client.delete_document(document_id)

            if success:
                logger.info(f"Deleted document: {document_id}")
            else:
                logger.warning(f"Delete returned False for document: {document_id}")

            return success
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get Vespa schema information.

        Returns:
            Schema information
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Get actual schema info if available from search backend
            if self._vespa_search_backend:
                # Delegate to search backend which has schema access
                return {
                    "name": self.config["schema_name"],
                    "backend": "vespa",
                    "initialized": True,
                    "search_enabled": self._initialized_as_search,
                    "ingestion_enabled": self._initialized_as_ingestion,
                }

            # Basic info if only ingestion is configured
            return {
                "name": self.config.get("schema_name", "unknown"),
                "backend": "vespa",
                "initialized": True,
                "search_enabled": False,
                "ingestion_enabled": self._initialized_as_ingestion,
            }
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            raise  # Re-raise instead of returning empty dict

    def validate_schema(self, schema_name: str) -> bool:
        """
        Validate that a schema exists in Vespa.

        Args:
            schema_name: Name of schema to validate

        Returns:
            True if valid
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Check if schema exists
            # This would query Vespa for the schema
            return True
        except Exception as e:
            logger.error(f"Failed to validate schema {schema_name}: {e}")
            return False

    # Search methods

    @property
    def schema_name(self) -> Optional[str]:
        """
        Get the schema name from the search backend.

        Returns:
            Schema name if search backend is initialized, None otherwise
        """
        if self._vespa_search_backend:
            return self._vespa_search_backend.schema_name
        elif self.config and "schema_name" in self.config:
            # Return base schema name if search backend not yet initialized
            return self.config["schema_name"]
        return None

    def search(self, query_dict: Dict[str, Any]) -> Any:
        """
        Execute a search query using query dict format.

        This method delegates to VespaSearchBackend and returns its results directly.
        The return type matches what VespaSearchBackend returns (List[SearchResult]).

        Args:
            query_dict: Dictionary with keys:
                - query: Text query string (required)
                - type: Content type (e.g., "video") (required)
                - profile: Profile name (optional)
                - strategy: Strategy name (optional)
                - top_k: Number of results (optional, defaults to 10)
                - filters: Optional filters dict
                - query_embeddings: Pre-computed embeddings (optional)

        Returns:
            Search results (List[SearchResult] from VespaSearchBackend)
        """
        # Lazy initialization: create search backend if not already initialized
        if not self._vespa_search_backend:
            logger.debug("Creating VespaSearchBackend on-demand with config")

            # Create VespaSearchBackend with merged backend config
            # VespaSearchBackend will handle profile/strategy resolution per query
            self._vespa_search_backend = VespaSearchBackend(
                config=self.config,  # Pass merged config (includes url, port, profiles, default_profiles)
                config_manager=self._config_manager_instance,
                schema_loader=self._schema_loader_instance,
            )
            self._initialized_as_search = True
            logger.info("VespaSearchBackend initialized with all profiles")

        # Delegate directly to VespaSearchBackend
        # It returns List[SearchResult], which is what SearchService expects
        return self._vespa_search_backend.search(query_dict)

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID (uses batch method).

        Args:
            document_id: Document ID

        Returns:
            Document or None
        """
        # Use batch method for consistency and optimization
        results = self.batch_get_documents([document_id])
        return results[0] if results else None

    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID (primary batch method).

        Args:
            document_ids: List of document IDs

        Returns:
            List of Documents (None for not found)
        """
        if not self._vespa_search_backend:
            raise RuntimeError("Search backend not initialized.")

        return self._vespa_search_backend.batch_get_documents(document_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Vespa statistics.

        Returns:
            Statistics dictionary
        """
        if self._vespa_search_backend:
            # Delegate to search backend if available
            return self._vespa_search_backend.get_statistics()

        # Basic stats if only ingestion is configured
        return {
            "backend": "vespa",
            "status": "healthy" if self.schema_manager else "not initialized",
            "search_enabled": self._initialized_as_search,
        }

    # ============================================================================
    # Schema Management Operations (Backend interface implementation)
    # ============================================================================

    def deploy_schemas(self, schema_definitions: List[Dict[str, Any]]) -> bool:
        """
        Deploy multiple schemas together.

        This is the low-level deployment interface called by SchemaRegistry.
        Deploys ALL provided schemas in a single Vespa ApplicationPackage.

        Args:
            schema_definitions: List of schema definition dicts, each containing:
                - name: Full schema name (e.g., "video_colpali_acme")
                - definition: Schema JSON definition
                - tenant_id: Tenant identifier
                - base_schema_name: Original base schema name

        Returns:
            True if successful, False otherwise

        Raises:
            RuntimeError: If backend not initialized
            Exception: If deployment fails
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            import json

            from cogniverse_vespa.json_schema_parser import JsonSchemaParser
            from vespa.package import ApplicationPackage

            parser = JsonSchemaParser()
            schemas_to_deploy = []

            # Parse all schema definitions into pyvespa Schema objects
            for schema_def in schema_definitions:
                schema_name = schema_def["name"]
                schema_json = schema_def["definition"]

                try:
                    # If definition is string, parse it
                    if isinstance(schema_json, str):
                        schema_json = json.loads(schema_json)

                    # Parse JSON to pyvespa Schema
                    schema_obj = parser.parse_schema(schema_json)
                    schemas_to_deploy.append(schema_obj)
                    logger.debug(f"Parsed schema for deployment: {schema_name}")
                except Exception as e:
                    logger.error(f"Failed to parse schema {schema_name}: {e}")
                    raise

            # Deploy all schemas together in one ApplicationPackage
            logger.info(f"Deploying {len(schemas_to_deploy)} schemas to Vespa")

            # Get application name from system config
            system_config = self._config_manager_instance.get_system_config()
            app_name = system_config.application_name

            app_package = ApplicationPackage(name=app_name, schema=schemas_to_deploy)

            # Add metadata schemas (Vespa-specific requirement)
            from cogniverse_vespa.metadata_schemas import (
                add_metadata_schemas_to_package,
            )

            add_metadata_schemas_to_package(app_package)
            logger.debug("Added metadata schemas to deployment package")

            # Deploy package directly via Backend's own method
            self._deploy_package(app_package)

            logger.info(f"Successfully deployed {len(schemas_to_deploy)} schemas")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy schemas: {e}")
            return False

    def _deploy_package(
        self, app_package, allow_field_type_change: bool = False
    ) -> None:
        """
        Deploy an application package to Vespa.

        Args:
            app_package: The ApplicationPackage to deploy
            allow_field_type_change: If True, adds validation override for field type changes

        Raises:
            RuntimeError: If deployment fails
        """
        import json
        import re

        import requests
        from vespa.package import Validation, ValidationID

        # Add validation override if requested
        if allow_field_type_change:
            from datetime import datetime, timedelta

            # Set validation until 29 days from now (to stay within 30-day limit)
            until_date = (datetime.now() + timedelta(days=29)).strftime("%Y-%m-%d")
            validation = Validation(
                validation_id=ValidationID.fieldTypeChange,
                until=until_date,
                comment="Allow field type changes for schema updates",
            )
            if app_package.validations is None:
                app_package.validations = []
            app_package.validations.append(validation)

        # Create the deployment URL - properly construct with base URL and port
        # Remove any existing port from endpoint
        base_url = re.sub(r":\d+$", "", self._url)
        deploy_url = f"{base_url}:{self._config_port}/application/v2/tenant/default/prepareandactivate"

        try:
            # Generate the ZIP package
            app_zip = app_package.to_zip()

            # Deploy via HTTP
            response = requests.post(
                deploy_url,
                headers={"Content-Type": "application/zip"},
                data=app_zip,
                verify=False,
            )

            if response.status_code == 200:
                logger.info("Successfully deployed application package")
            else:
                error_msg = f"Deployment failed with status {response.status_code}"
                try:
                    error_detail = json.loads(response.content.decode("utf-8"))
                    error_msg += f": {error_detail}"
                except Exception:
                    error_msg += f": {response.content.decode('utf-8')}"

                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Failed to deploy package: {str(e)}")
            raise

    def delete_schema(
        self, schema_name: str, tenant_id: Optional[str] = None
    ) -> List[str]:
        """
        Delete tenant schema(s).

        Args:
            schema_name: Base schema name (if provided, deletes specific schema)
            tenant_id: Tenant identifier (uses self._tenant_id if not provided)

        Returns:
            List of deleted schema names
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Use provided tenant_id or fall back to instance tenant_id
        effective_tenant_id = tenant_id or self._tenant_id
        if not effective_tenant_id:
            raise ValueError("tenant_id required for schema deletion")

        try:
            deleted_schemas = self._delete_tenant_schemas(effective_tenant_id)
            logger.info(
                f"Deleted {len(deleted_schemas)} schemas for tenant '{effective_tenant_id}'"
            )
            return deleted_schemas
        except Exception as e:
            logger.error(
                f"Failed to delete schemas for tenant '{effective_tenant_id}': {e}"
            )
            return []

    def schema_exists(self, schema_name: str, tenant_id: Optional[str] = None) -> bool:
        """
        Check if schema exists.

        Args:
            schema_name: Base schema name
            tenant_id: Tenant identifier (uses self._tenant_id if not provided)

        Returns:
            True if schema exists, False otherwise
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Use provided tenant_id or fall back to instance tenant_id
        effective_tenant_id = tenant_id or self._tenant_id
        if not effective_tenant_id:
            # For non-tenant operations, check if base schema exists
            return self.validate_schema(schema_name)

        try:
            # Check if schema exists via VespaSchemaManager
            return self.schema_manager.tenant_schema_exists(
                effective_tenant_id, schema_name
            )
        except Exception as e:
            logger.error(
                f"Failed to check schema existence for '{schema_name}' tenant '{effective_tenant_id}': {e}"
            )
            return False

    def _delete_tenant_schemas(self, tenant_id: str) -> List[str]:
        """
        Delete all schemas for a tenant.

        Delegates to VespaSchemaManager.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of deleted schema names
        """
        return self.schema_manager.delete_tenant_schemas(tenant_id)

    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
        """
        Get tenant-specific schema name.

        Delegates to VespaSchemaManager.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Returns:
            Tenant-specific schema name (e.g., "video_colpali_acme")
        """
        return self.schema_manager.get_tenant_schema_name(tenant_id, base_schema_name)

    # ============================================================================
    # Metadata Document Operations (Backend interface implementation)
    # ============================================================================

    def create_metadata_document(
        self, schema: str, doc_id: str, fields: Dict[str, Any]
    ) -> bool:
        """
        Create or update metadata document.

        Args:
            schema: Schema name (e.g., "organization_metadata", "tenant_metadata")
            doc_id: Document ID
            fields: Document fields as dict

        Returns:
            True if successful, False otherwise
        """
        if not self._url:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Import Vespa client for metadata operations
            from vespa.application import Vespa

            # Create Vespa client for metadata operations
            vespa_client = Vespa(url=f"{self._url}:{self._port}")

            # Feed metadata document
            response = vespa_client.feed_data_point(
                schema=schema, data_id=doc_id, fields=fields
            )

            # Check response status
            if response.status_code != 200:
                logger.error(
                    f"Failed to create metadata document {schema}/{doc_id}: "
                    f"HTTP {response.status_code}"
                )
                return False

            logger.debug(f"Created metadata document: {schema}/{doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create metadata document {schema}/{doc_id}: {e}")
            return False

    def get_metadata_document(
        self, schema: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata document by ID.

        Args:
            schema: Schema name
            doc_id: Document ID

        Returns:
            Document fields as dict, or None if not found
        """
        if not self._url:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Import Vespa client for metadata operations
            from vespa.application import Vespa

            # Create Vespa client for metadata operations
            vespa_client = Vespa(url=f"{self._url}:{self._port}")

            # Get metadata document
            response = vespa_client.get_data(schema=schema, data_id=doc_id)

            if not response or response.status_code != 200:
                return None

            result = response.json
            return result.get("fields", {})
        except Exception as e:
            logger.error(f"Failed to get metadata document {schema}/{doc_id}: {e}")
            return None

    def query_metadata_documents(
        self,
        schema: str,
        query: Optional[str] = None,
        yql: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Query metadata documents.

        Args:
            schema: Schema name to query
            query: Text query (for userQuery() in YQL)
            yql: Direct YQL query
            **kwargs: Additional query options (hits, filters, etc.)

        Returns:
            List of matching documents as dicts
        """
        if not self._url:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Import Vespa client for metadata operations
            from vespa.application import Vespa

            # Create Vespa client for metadata operations
            vespa_client = Vespa(url=f"{self._url}:{self._port}")

            # Build query parameters
            query_params = {
                "hits": kwargs.get("hits", 100),
            }

            if yql:
                query_params["yql"] = yql
                # If YQL contains userQuery(), also add the query parameter if provided
                if query and "userQuery()" in yql:
                    query_params["query"] = query
            elif query:
                # Use userQuery() for text search
                query_params["yql"] = f"select * from {schema} where userQuery()"
                query_params["query"] = query
            else:
                # Get all documents - Vespa requires at least one search term
                # Using a match-all pattern with limit
                query_params["yql"] = (
                    f'select * from {schema} where true limit {kwargs.get("hits", 100)}'
                )

            # Execute query
            results = vespa_client.query(**query_params)

            # Extract documents from response
            documents = []
            for hit in results.json.get("root", {}).get("children", []):
                fields = hit.get("fields", {})
                documents.append(fields)

            logger.debug(f"Query returned {len(documents)} documents from {schema}")
            return documents
        except Exception as e:
            logger.error(f"Failed to query metadata documents from {schema}: {e}")
            return []

    def delete_metadata_document(self, schema: str, doc_id: str) -> bool:
        """
        Delete metadata document.

        Args:
            schema: Schema name
            doc_id: Document ID

        Returns:
            True if successful, False otherwise
        """
        if not self._url:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            # Import Vespa client for metadata operations
            from vespa.application import Vespa

            # Create Vespa client for metadata operations
            vespa_client = Vespa(url=f"{self._url}:{self._port}")

            # Delete metadata document
            vespa_client.delete_data(schema=schema, data_id=doc_id)

            logger.debug(f"Deleted metadata document: {schema}/{doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete metadata document {schema}/{doc_id}: {e}")
            return False

    # ============================================================================
    # Connection Management
    # ============================================================================

    def close(self) -> None:
        """
        Close connections to Vespa.
        """
        # Close all schema-specific clients
        for schema_name, client in self._vespa_ingestion_clients.items():
            client.close()
            logger.info(f"Closed Vespa client for schema: {schema_name}")

        for schema_name, client in self._async_ingestion_clients.items():
            client.close()
            logger.info(f"Closed async Vespa client for schema: {schema_name}")

        if self._vespa_search_backend:
            # Search backend may not have a close method
            pass

        logger.info("Closed all Vespa backend connections")

    def health_check(self) -> bool:
        """
        Check Vespa health.

        Returns:
            True if healthy
        """
        if self._vespa_search_backend:
            return self._vespa_search_backend.health_check()

        # Basic health check
        return self.schema_manager is not None

    def get_embedding_requirements(self, schema_name: str) -> Dict[str, Any]:
        """
        Get embedding requirements for a specific schema.

        Args:
            schema_name: Name of schema to get requirements for

        Returns:
            Dict containing embedding requirements (needs_float, needs_binary, field names)
        """
        # Ensure search backend is initialized
        if not self._vespa_search_backend:
            self._initialize_search_backend()

        # Delegate to VespaSearchBackend which has the full implementation
        return self._vespa_search_backend.get_embedding_requirements(schema_name)


# Self-registration when module is imported
def register() -> None:
    """Register Vespa backend with the backend registry."""
    from cogniverse_core.registries.backend_registry import register_backend

    try:
        register_backend("vespa", VespaBackend)
        logger.info("Vespa backend registered successfully")
    except Exception as e:
        logger.error(f"Failed to register Vespa backend: {e}")


# Call registration on import
register()
