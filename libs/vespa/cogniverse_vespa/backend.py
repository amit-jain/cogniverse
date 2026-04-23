"""
Vespa backend implementation with unified interface.

This module provides a Vespa backend that implements both IngestionBackend
and SearchBackend interfaces, with self-registration to the backend registry.
"""

import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cogniverse_core.registries.exceptions import BackendDeploymentError
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
        self._async_ingestion_clients: Dict[
            str, Any
        ] = {}  # For async ingestion (optional)
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

        if merged_config.get("use_async_ingestion", False) and not ASYNC_AVAILABLE:
            raise ImportError(
                "Async ingestion requested (use_async_ingestion=True) but "
                "async_ingestion_client module is not available. "
                "Ensure cogniverse_vespa is installed with async extras."
            )
        self.use_async_ingestion = merged_config.get("use_async_ingestion", False)

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
        # so schema_manager can preserve existing tenant schemas when needed
        if self.schema_registry:
            self.schema_manager._schema_registry = self.schema_registry
            logger.debug("Injected schema_registry into schema_manager")

        # NOTE: Metadata schemas are NOT deployed here.
        # deploy_schemas() already includes metadata via add_metadata_schemas_to_package().
        # Standalone metadata deployment is done once at system startup (Runtime lifespan
        # or test conftest), not on every backend instantiation — rapid re-deployments
        # prevent Vespa data nodes from converging tenant schemas.

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

        # Return cached client if it exists (skip schema deploy check)
        if target_schema_name in self._vespa_ingestion_clients:
            return self._vespa_ingestion_clients[target_schema_name]

        # Deploy tenant schema only when creating a new client
        if self._tenant_id:
            if not self.schema_registry:
                raise ValueError(
                    "schema_registry not injected - backend initialization incomplete."
                )
            try:
                self.schema_registry.deploy_schema(
                    tenant_id=self._tenant_id, base_schema_name=schema_name
                )
            except Exception as e:
                logger.error(f"Failed to deploy tenant schema: {e}")
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
            import time as _time

            import requests as _requests

            timeout = self.config.get("indexing_timeout", 30.0)
            base_url = f"{self._url}:{self._port}"

            for doc in documents:
                if doc.id not in [
                    fd if isinstance(fd, str) else fd.get("id") for fd in failed_docs
                ]:
                    target_schema = schema_name
                    if self._tenant_id:
                        target_schema = self.get_tenant_schema_name(
                            self._tenant_id, schema_name
                        )
                    deadline = _time.monotonic() + timeout
                    while _time.monotonic() < deadline:
                        try:
                            resp = _requests.get(
                                f"{base_url}/search/",
                                params={
                                    "yql": f'select * from {target_schema} where id matches "{doc.id}" limit 1',
                                },
                                timeout=5,
                            )
                            if resp.status_code == 200:
                                total = (
                                    resp.json()
                                    .get("root", {})
                                    .get("fields", {})
                                    .get("totalCount", 0)
                                )
                                if total > 0:
                                    break
                        except _requests.RequestException:
                            pass
                        _time.sleep(0.5)
                    else:
                        logger.warning(
                            f"Document {doc.id} fed but not visible after {timeout}s"
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

    def update_document(
        self,
        document_id: str,
        document: Document,
        schema_name: Optional[str] = None,
    ) -> bool:
        """
        Update a document in Vespa.

        Args:
            document_id: ID of document to update
            document: Updated Document object
            schema_name: Vespa schema to write to. If omitted, falls back to
                ``self.config["schema_name"]``.

        Returns:
            True if successful
        """
        try:
            if not schema_name:
                schema_name = self.config.get("schema_name")
            if not schema_name:
                logger.error("No schema_name in config for update operation")
                return False

            results = self.ingest_documents([document], schema_name=schema_name)
            return results["success_count"] > 0
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def delete_document(
        self, document_id: str, schema_name: Optional[str] = None
    ) -> bool:
        """
        Delete a document from Vespa.

        Args:
            document_id: ID of document to delete
            schema_name: Vespa schema to delete from. If omitted, falls back to
                ``self.config["schema_name"]``. Callers that share a backend
                across multiple schemas (e.g. the Mem0 vector store) should
                pass this explicitly.

        Returns:
            True if successful
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            if not schema_name:
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

            # Ensure profiles are loaded (may be missing if ingestion created
            # the cached backend instance without passing profiles)
            if not self.config.get("profiles") and self._config_manager_instance:
                from cogniverse_foundation.config.utils import get_config

                config_utils = get_config(
                    tenant_id=self._tenant_id,
                    config_manager=self._config_manager_instance,
                )
                backend_section = config_utils.get("backend", {})
                if backend_section.get("profiles"):
                    self.config["profiles"] = backend_section["profiles"]
                    self.config["default_profiles"] = backend_section.get(
                        "default_profiles", {}
                    )
                    logger.info(
                        f"Loaded {len(self.config['profiles'])} profiles from config "
                        f"for tenant {self._tenant_id}"
                    )

            # Create VespaSearchBackend with merged backend config
            # VespaSearchBackend will handle profile/strategy resolution per query
            self._vespa_search_backend = VespaSearchBackend(
                config=self.config,  # Pass merged config (includes url, port, profiles, default_profiles)
                config_manager=self._config_manager_instance,
                schema_loader=self._schema_loader_instance,
            )
            self._initialized_as_search = True
            logger.info("VespaSearchBackend initialized with all profiles")

        # Delegate directly to VespaSearchBackend.
        # Caller MUST set tenant_id in query_dict — VespaSearchBackend raises
        # ValueError if missing.
        return self._vespa_search_backend.search(query_dict)

    def get_document(
        self, document_id: str, schema_name: Optional[str] = None
    ) -> Optional[Document]:
        """
        Retrieve a document by ID via the ingestion client (pyvespa get_data).

        This does not require the search subsystem — any VespaBackend instance
        (ingestion or search) can retrieve documents by ID.

        Args:
            document_id: Document ID
            schema_name: Vespa schema to fetch from. If omitted, falls back to
                ``self.config["schema_name"]``. Callers that share a backend
                across multiple schemas (e.g. the Mem0 vector store) should
                pass this explicitly.

        Returns:
            Document or None
        """
        if not schema_name:
            schema_name = self.config.get("schema_name")
        if not schema_name:
            raise ValueError(
                "No schema_name in config — cannot determine which Vespa schema to read from."
            )

        client = self._get_or_create_ingestion_client(schema_name)
        fields = client.get_document_data(document_id)
        if fields is None:
            return None

        return Document(
            id=document_id,
            text_content=fields.get("text", ""),
            metadata={
                k: v for k, v in fields.items() if k not in ("text", "embedding", "id")
            },
        )

    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID.

        Uses the search backend's YQL batch query if available (single round-trip),
        otherwise retrieves each document individually via the ingestion client.

        Args:
            document_ids: List of document IDs

        Returns:
            List of Documents (None for not found)
        """
        if not document_ids:
            return []

        if self._vespa_search_backend:
            return self._vespa_search_backend.batch_get_documents(document_ids)

        # No search backend — retrieve individually via ingestion client
        return [self.get_document(doc_id) for doc_id in document_ids]

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

    def deploy_schemas(
        self,
        schema_definitions: List[Dict[str, Any]],
        allow_schema_removal: bool = False,
    ) -> bool:
        """
        Deploy multiple schemas together.

        This is the low-level deployment interface called by SchemaRegistry.
        Deploys ALL provided schemas in a single Vespa ApplicationPackage,
        merging any schemas already present in the registry or the live
        Vespa cluster to avoid silently dropping them.

        Args:
            schema_definitions: List of schema definition dicts, each containing:
                - name: Full schema name (e.g., "video_colpali_acme")
                - definition: Schema JSON definition
                - tenant_id: Tenant identifier
                - base_schema_name: Original base schema name
            allow_schema_removal: When True, pass the Vespa
                ``contentTypeRemoval`` validation override and skip the
                discovery check that normally refuses to silently drop
                schemas. Defaults to False — an operator who actually wants
                to remove a schema must opt in explicitly.

        Returns:
            True if successful, False otherwise

        Raises:
            RuntimeError: If backend not initialized
            BackendDeploymentError: If the live cluster has schemas the
                registry can't reconstruct and ``allow_schema_removal`` is
                False.
            Exception: If deployment fails
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            import json

            from vespa.package import ApplicationPackage

            from cogniverse_vespa.json_schema_parser import JsonSchemaParser

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

            # Merge existing schemas into the deployment so the redeploy looks
            # like an "add" rather than "remove + add". Two sources feed the
            # merge:
            #
            #   1. SchemaRegistry — definitive schema JSON keyed by
            #      (tenant_id, base_schema) pair. Used to pick up every
            #      schema this process has ever deployed through the registry.
            #   2. Vespa itself (via schema_manager.list_deployed_document_types)
            #      — authoritative list of what the cluster currently has,
            #      catching schemas deployed out-of-band (tests pushing their
            #      own ApplicationPackage, prior crashes, or another process).
            #
            # A schema discovered only in Vespa is preserved using the
            # best-effort reconstruction from registry data keyed by name;
            # if no definition is available, deploy FAILS instead of silently
            # dropping the schema.
            new_schema_names = {s.name for s in schemas_to_deploy}
            merged_schemas = list(schemas_to_deploy)
            merged_schema_names = set(new_schema_names)

            parser_for_existing = JsonSchemaParser()

            if self.schema_registry is not None:
                try:
                    existing_schemas = self.schema_registry._get_all_schemas() or []
                    for schema_info in existing_schemas:
                        full_name = schema_info.full_schema_name
                        if full_name in merged_schema_names:
                            continue
                        try:
                            existing_def = schema_info.schema_definition
                            if isinstance(existing_def, str):
                                if not existing_def.strip():
                                    continue
                                existing_def = json.loads(existing_def)
                            existing_obj = parser_for_existing.parse_schema(
                                existing_def
                            )
                            merged_schemas.append(existing_obj)
                            merged_schema_names.add(full_name)
                        except Exception as merge_exc:
                            logger.warning(
                                f"Skipping existing schema {full_name} "
                                f"during registry merge: {merge_exc}"
                            )
                    logger.info(
                        f"Merged {len(merged_schemas) - len(schemas_to_deploy)} "
                        f"schemas from registry into deployment package"
                    )
                except Exception as registry_exc:
                    logger.warning(
                        f"Could not fetch existing schemas from registry: {registry_exc}"
                    )

            # Second source: ask Vespa what it currently has deployed. Any
            # schema name here that the registry didn't cover must be
            # reconstructed or the deploy fails — silently dropping a
            # peer-tenant schema is never acceptable. Discovery uses the
            # HTTP query port (self._port), not the config port the
            # schema_manager was initialised with.
            try:
                vespa_deployed = self.schema_manager.list_deployed_document_types(
                    query_port=self._port
                )
            except Exception as probe_exc:
                logger.warning(
                    f"Vespa schema discovery failed, relying on registry alone: "
                    f"{probe_exc}"
                )
                vespa_deployed = []

            # Skip Vespa-managed metadata schemas — they're re-added below via
            # add_metadata_schemas_to_package and shouldn't round-trip through
            # JsonSchemaParser (their definitions aren't in the registry).
            metadata_names = {
                "tenant_metadata",
                "organization_metadata",
                "config_metadata",
                "adapter_registry",
            }

            unknown_in_vespa = [
                name
                for name in vespa_deployed
                if name not in merged_schema_names and name not in metadata_names
            ]
            if unknown_in_vespa:
                # Try to reconstruct from registry-keyed-by-full-name (a
                # cross-instance registry may have the definition even if
                # the (tenant, base) lookup missed it).
                registry_by_full_name: Dict[str, Any] = {}
                if self.schema_registry is not None:
                    for schema_info in self.schema_registry._get_all_schemas() or []:
                        registry_by_full_name[schema_info.full_schema_name] = (
                            schema_info
                        )

                unresolved = []
                for full_name in unknown_in_vespa:
                    schema_info = registry_by_full_name.get(full_name)
                    if schema_info is None:
                        unresolved.append(full_name)
                        continue
                    try:
                        existing_def = schema_info.schema_definition
                        if isinstance(existing_def, str):
                            existing_def = json.loads(existing_def)
                        merged_schemas.append(
                            parser_for_existing.parse_schema(existing_def)
                        )
                        merged_schema_names.add(full_name)
                    except Exception as reconstruct_exc:
                        logger.error(
                            f"Schema {full_name} exists in Vespa but can't be "
                            f"reconstructed: {reconstruct_exc}"
                        )
                        unresolved.append(full_name)

                if unresolved and not allow_schema_removal:
                    raise BackendDeploymentError(
                        "Refusing to deploy: Vespa has schemas "
                        f"{sorted(unresolved)} that are not in SchemaRegistry "
                        "and cannot be reconstructed. Redeploying without them "
                        "would silently drop them. Register these schemas via "
                        "SchemaRegistry.register_schema() before redeploying, "
                        "or pass allow_schema_removal=True to explicitly "
                        "remove them."
                    )
                if unresolved and allow_schema_removal:
                    logger.warning(
                        "allow_schema_removal=True — dropping Vespa schemas "
                        f"{sorted(unresolved)} that could not be reconstructed."
                    )

            # Get application name from system config
            system_config = self._config_manager_instance.get_system_config()
            app_name = system_config.application_name

            app_package = ApplicationPackage(name=app_name, schema=merged_schemas)

            # Add metadata schemas (Vespa-specific requirement)
            from cogniverse_vespa.metadata_schemas import (
                add_metadata_schemas_to_package,
            )

            add_metadata_schemas_to_package(app_package)
            logger.debug("Added metadata schemas to deployment package")

            # Only pass the Vespa validation override when the caller has
            # explicitly asked for it. The merge above + live Vespa discovery
            # should make the override unnecessary; if something still slips
            # through, failing loudly beats silently dropping a schema.
            self._deploy_package(app_package, allow_schema_removal=allow_schema_removal)

            # Wait for content nodes to converge with the new schema
            # Vespa config server accepts the package immediately but content/distributor
            # nodes need time to pick up new document types.
            schema_names = [s.name for s in schemas_to_deploy]
            self._wait_for_schema_convergence(schema_names)

            logger.info(f"Successfully deployed {len(schemas_to_deploy)} schemas")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy schemas: {e}")
            return False

    def _deploy_package(
        self,
        app_package,
        allow_field_type_change: bool = False,
        allow_schema_removal: bool = False,
    ) -> None:
        """
        Deploy an application package to Vespa.

        Args:
            app_package: The ApplicationPackage to deploy
            allow_field_type_change: If True, adds validation override for field type changes
            allow_schema_removal: If True, adds validation override for content type
                removal. Required when the package contains fewer schemas than the
                cluster currently has — without this, partial deploys (e.g., adding a
                single tenant schema) get rejected because Vespa interprets the missing
                schemas as a destructive removal.

        Raises:
            RuntimeError: If deployment fails
        """
        import json

        import requests
        from vespa.package import Validation, ValidationID

        # Add validation overrides if requested
        if allow_field_type_change or allow_schema_removal:
            from datetime import datetime, timedelta

            # Set validation until 7 days from now. Vespa treats the date as an
            # exclusive end (until="2026-05-07" → "2026-05-08T00:00:00Z"), so
            # using 29 days can land exactly on the 30-day boundary and fail.
            until_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            if app_package.validations is None:
                app_package.validations = []

            if allow_field_type_change:
                app_package.validations.append(
                    Validation(
                        validation_id=ValidationID.fieldTypeChange,
                        until=until_date,
                        comment="Allow field type changes for schema updates",
                    )
                )

            if allow_schema_removal:
                app_package.validations.append(
                    Validation(
                        validation_id=ValidationID.contentTypeRemoval,
                        until=until_date,
                        comment=(
                            "Allow schema removal during partial deployments. "
                            "Required when deploy_schemas() merges existing schemas "
                            "from SchemaRegistry but the registry is incomplete."
                        ),
                    )
                )

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

    def _wait_for_schema_convergence(
        self, schema_names: List[str], timeout: int = 60
    ) -> None:
        """
        Wait for Vespa content nodes to converge after schema deployment.

        After deploying an application package, the config server accepts
        it immediately but content/distributor nodes need extra time to
        recognise new document types. Queries via ``/search/`` can return
        200 before content distributors are ready, and ``GET /document/v1/``
        returns 404 for *any* URL (even an unknown schema), so neither is
        discriminative.

        The only probe that actually fails until the schema is addressable
        on the feed path is ``POST /search/`` with ``model.restrict=<name>``:
        when the schema is unknown Vespa errors out listing every valid
        source ref, and the new schema appears in that list exactly when
        the content distributor has caught up. Once every probed schema is
        visible we add a short buffer — search visibility converges a
        beat before the document API accepts feeds, so without this
        buffer the first feed still races.

        Args:
            schema_names: Names of schemas that were just deployed
            timeout: Maximum seconds to wait for convergence
        """
        import time

        import requests

        # If there are no schemas to wait for (e.g., the rollback path
        # re-deploying 0 previous schemas), there's nothing to probe.
        if not schema_names:
            logger.debug("Skipping convergence probe: no schemas in deployment package")
            return

        base_url = re.sub(r":\d+$", "", self._url)
        probe_url = f"{base_url}:{self._port}/search/"

        logger.info(
            f"Waiting for content node convergence (schemas: {schema_names})..."
        )
        remaining = set(schema_names)
        for i in range(timeout):
            for name in list(remaining):
                try:
                    response = requests.post(
                        probe_url,
                        json={
                            "yql": "select documentid from sources * where true limit 0",
                            "hits": 0,
                            "model.restrict": name,
                        },
                        timeout=5,
                    )
                    if response.status_code != 200:
                        continue
                    body = response.json()
                    errors = body.get("root", {}).get("errors", [])
                    if not errors:
                        remaining.discard(name)
                        continue
                    # Vespa lists every deployed source ref in its error
                    # message as "cogniverse_content, cogniverse_content.<doc>,
                    # ...". Parse exactly; substring match gives false positives
                    # on name prefixes (e.g., lateon_mv in code_lateon_mv).
                    deployed: set[str] = set()
                    for err in errors:
                        match = re.search(
                            r"Valid source refs are (.+)$", err.get("message", "")
                        )
                        if not match:
                            continue
                        raw = match.group(1).rstrip().rstrip(".")
                        for part in raw.split(","):
                            part = part.strip()
                            if "." in part:
                                deployed.add(part.split(".", 1)[1])
                    if name in deployed:
                        remaining.discard(name)
                except (requests.exceptions.ConnectionError, ValueError):
                    pass

            if not remaining:
                logger.info(
                    f"Content nodes converged after {i + 1}s (schemas={schema_names})"
                )
                # Feed-path (document/v1) converges a beat after search
                # visibility — a short buffer eliminates the first-feed
                # race without significantly slowing deploys.
                time.sleep(3)
                return
            time.sleep(1)

        logger.warning(
            f"Content node convergence not confirmed after {timeout}s "
            f"(still missing: {sorted(remaining)}) — proceeding anyway "
            "(feed retries may compensate)"
        )

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
                    f"select * from {schema} where true limit {kwargs.get('hits', 100)}"
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

    # -----------------------------------------------------------------
    # Runtime profile mutation (SearchBackend interface override)
    # -----------------------------------------------------------------
    # Keep self.config["profiles"] and the owned VespaSearchBackend's
    # in-memory dict in sync so runtime-added profiles are visible to
    # both the ingestion path (reads config directly) and the search
    # path (reads via VespaSearchBackend.profiles).

    def add_profile(self, profile_name: str, profile_config: Dict[str, Any]) -> None:
        """Register a profile at runtime; mirror into owned search backend."""
        if hasattr(self, "config") and isinstance(self.config, dict):
            profiles = self.config.setdefault("profiles", {})
            profiles[profile_name] = dict(profile_config)
        if self._vespa_search_backend is not None:
            self._vespa_search_backend.add_profile(profile_name, profile_config)

    def remove_profile(self, profile_name: str) -> None:
        """Unregister a profile at runtime."""
        if hasattr(self, "config") and isinstance(self.config, dict):
            profiles = self.config.get("profiles")
            if isinstance(profiles, dict):
                profiles.pop(profile_name, None)
        if self._vespa_search_backend is not None:
            self._vespa_search_backend.remove_profile(profile_name)

    @property
    def profiles(self) -> Dict[str, Any]:
        """Expose the live profiles dict the way VespaSearchBackend does.

        The unified backend keeps profiles in ``self.config["profiles"]``
        at initialize time; exposing them under ``self.profiles`` lets
        callers (tests, introspection) use a single attribute name
        across both VespaBackend and VespaSearchBackend.
        """
        if hasattr(self, "config") and isinstance(self.config, dict):
            profiles = self.config.get("profiles")
            if isinstance(profiles, dict):
                return profiles
        return {}

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
