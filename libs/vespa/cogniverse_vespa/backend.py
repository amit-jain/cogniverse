"""
Vespa backend implementation with unified interface.

This module provides a Vespa backend that implements both IngestionBackend
and SearchBackend interfaces, with self-registration to the backend registry.
"""

import logging
import re
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cogniverse_sdk.document import Document
from cogniverse_sdk.interfaces.backend import Backend

from ._vespa_factory import make_persistent_vespa_ops
from .config_utils import calculate_config_port
from .ingestion_client import VespaPyClient, document_namespace
from .search_backend import VespaSearchBackend
from .vespa_schema_manager import DEPLOY_REQUEST_TIMEOUT_S, VespaSchemaManager

# Async ingestion uses pyvespa's built-in feed_async_iterable (an HTTP/2 async
# feeder callable from sync code) — no separate adapter module is needed.
logger = logging.getLogger(__name__)


def _http_status_of(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status from a pyvespa client exception.

    A 412 test-and-set rejection surfaces as a ``VespaError`` wrapping an
    ``HTTPError``; depending on the underlying HTTP client the status rides on
    a ``.response`` object (requests/httpx) or only in the message pyvespa
    builds as ``"HTTP <code>: ..."`` (its httpr client), so walk the
    cause/context chain checking both.
    """
    node: Optional[BaseException] = exc
    for _ in range(5):
        if node is None:
            return None
        status = getattr(getattr(node, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
        status = getattr(node, "status_code", None)
        if isinstance(status, int):
            return status
        match = re.match(r"HTTP (\d{3})\b", str(node))
        if match:
            return int(match.group(1))
        node = node.__cause__ or node.__context__
    return None


class VespaBackend(Backend):
    """
    Vespa backend implementation supporting both ingestion and search.

    This class wraps the existing Vespa implementations and provides
    a unified interface compatible with the backend registry.
    """

    # Class-level fallbacks keep partially-constructed instances (test slices
    # via object.__new__) safe. __init__ replaces both with per-instance locks
    # so real backends do not contend across instances.
    _metadata_app_lock = threading.Lock()
    _close_lock = threading.Lock()
    _ingestion_clients_lock = threading.RLock()
    _search_backend_lock = threading.RLock()

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
        self._schema_loader_instance = schema_loader
        self._config_manager_instance = config_manager
        self._vespa_search_backend: Optional[VespaSearchBackend] = None
        # Store multiple ingestion clients, one per schema
        self._vespa_ingestion_clients: Dict[str, VespaPyClient] = {}
        self.schema_manager: Optional[VespaSchemaManager] = None
        self._initialized_as_search = False
        self._initialized_as_ingestion = False
        self.use_async_ingestion = False  # Flag to enable async mode

        # Extract connection details from BackendConfig
        self._url: str = backend_config.url
        self._port: int = backend_config.port
        self._tenant_id: str = backend_config.tenant_id
        # Cached pyvespa app for metadata ops, rebuilt only if url/port change
        # (e.g. after a deploy-time override) so probes don't churn a new
        # connection pool per call.
        self._metadata_app = None
        self._metadata_app_key = None
        self._metadata_app_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._ingestion_clients_lock = threading.RLock()
        self._search_backend_lock = threading.RLock()

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

        # search() constructs the inner VespaSearchBackend from self.config,
        # which reads url/port with a localhost default. Reflect the resolved
        # authoritative values so a backend built from an empty config (the
        # config-less get_search_backend startup call) can't dial localhost.
        self.config["url"] = self._url
        self.config["port"] = self._port

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

        with self._ingestion_clients_lock:
            cached = self._vespa_ingestion_clients.get(target_schema_name)
            if cached is not None:
                return cached

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

            logger.info(f"Creating new VespaPyClient for schema: {target_schema_name}")

            profile_config = {}
            if self.config:
                profiles = self.config.get("profiles", {})
                profile_config = profiles.get(schema_name, {})

            client_config = {
                "schema_name": target_schema_name,
                "base_schema_name": schema_name,
                "url": self._url,
                "port": self._port,
                "profile_config": profile_config,
                "schema_loader": self._schema_loader_instance,
                "use_async_ingestion": self.use_async_ingestion,
            }

            client = VespaPyClient(config=client_config, logger=logger)
            if not client.connect():
                raise ConnectionError(
                    f"Failed to connect Vespa ingestion client for "
                    f"{target_schema_name} at {self._url}:{self._port}"
                )

            self._vespa_ingestion_clients[target_schema_name] = client
            return client

    def ingest_documents(
        self,
        documents: List[Document],
        schema_name: str,
        operation_type: str = "feed",
    ) -> Dict[str, Any]:
        """
        Ingest documents into Vespa.

        Args:
            documents: List of Document objects to ingest
            schema_name: Schema to ingest documents into
            operation_type: ``"feed"`` (PUT, full replace) or ``"update"``
                (partial assign — only the fields present on each Document are
                written, leaving the rest such as embedding tensors intact).

        Returns:
            Ingestion results
        """
        # Get schema-specific client
        client = self._get_or_create_ingestion_client(schema_name)

        # Process and feed documents using the schema-specific client
        # Each client already knows its schema, no need to pass it
        prepared_docs = []
        for doc in documents:
            # Pass operation_type so partial updates don't auto-stamp (and thus
            # clobber) an absent timestamp field via the partial assign.
            prepared = client.process(doc, operation_type=operation_type)
            prepared_docs.append(prepared)

        # Feed documents to Vespa
        success_count, failed_docs = client._feed_prepared_batch(
            prepared_docs,
            operation_type=operation_type,  # Client uses its own schema
        )

        # Wait for documents to be visible in queries (handle Vespa's eventual consistency)
        if self.config.get("wait_for_indexing", True) and success_count > 0:
            import time as _time

            import requests as _requests

            timeout = self.config.get("indexing_timeout", 30.0)
            base_url = f"{self._url}:{self._port}"

            target_schema = schema_name
            if self._tenant_id:
                target_schema = self.get_tenant_schema_name(
                    self._tenant_id, schema_name
                )
            # Document v1 GET is the only reliable per-doc visibility probe.
            # The previous YQL form `where id matches "<doc.id>"` returned
            # HTTP 400 ("Field 'id' does not exist") on every attempt, and
            # the loop only caught RequestException — a 4xx is *not* an
            # exception in the `requests` library, so the probe silently
            # spun for the full timeout and then logged a misleading
            # "fed but not visible" warning even though the doc was already
            # queryable. `documentid` is also not a YQL field on most
            # schemas; Document v1 GET keys directly off the doc id and
            # returns 200 vs 404 unambiguously.
            # Probe all fed docs in sweeps over one keep-alive session: docs
            # drop out as they become visible, sweeps (not docs) share the
            # 0.5s backoff, and the timeout bounds the whole batch. The old
            # per-doc loop paid a fresh TCP connection per probe and up to
            # ``timeout`` seconds of sleeps per document.
            namespace = getattr(client, "namespace", "content")
            failed_ids = {
                fd if isinstance(fd, str) else fd.get("id") for fd in failed_docs
            }
            pending = {
                doc.id: (
                    f"{base_url}/document/v1/{namespace}/{target_schema}/docid/{doc.id}"
                )
                for doc in documents
                if doc.id not in failed_ids
            }
            deadline = _time.monotonic() + timeout
            with _requests.Session() as session:
                while pending:
                    for doc_id, doc_url in list(pending.items()):
                        try:
                            resp = session.get(doc_url, timeout=5)
                            if resp.status_code == 200:
                                del pending[doc_id]
                            elif resp.status_code not in (404, 503):
                                logger.warning(
                                    f"Visibility probe unexpected status "
                                    f"{resp.status_code} for {doc_id}: "
                                    f"{resp.text[:200]}"
                                )
                                del pending[doc_id]
                        except _requests.RequestException:
                            pass
                    if not pending or _time.monotonic() >= deadline:
                        break
                    _time.sleep(0.5)
            for doc_id in pending:
                logger.warning(
                    f"Document {doc_id} fed but not visible after {timeout}s"
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
        self,
        documents: Iterator[Document],
        schema_name: str,
        batch_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream documents for ingestion.

        Args:
            documents: Iterator of Document objects
            schema_name: Schema to ingest each batch into
            batch_size: Number of documents per batch

        Yields:
            Ingestion results for each batch
        """
        batch = []
        for doc in documents:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield self.ingest_documents(batch, schema_name)
                batch = []

        # Process remaining documents
        if batch:
            yield self.ingest_documents(batch, schema_name)

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
        if not schema_name:
            schema_name = self.config.get("schema_name")
        if not schema_name:
            raise ValueError("No schema_name in config for update operation")

        if document.id != document_id:
            raise ValueError(
                f"update_document(document_id={document_id!r}) does not match "
                f"document.id={document.id!r}; the partial update would land "
                f"on the wrong doc id."
            )

        # Partial update (assign only present fields) so a metadata-only
        # update does not wipe the stored embedding tensors via a full PUT.
        # A backend outage propagates (ingest_documents raises) rather than
        # returning False the caller reads as "update rejected"; only a genuine
        # zero-success partial update returns False. The broad except that used
        # to swallow both an outage AND the id-mismatch programming error above
        # into a silent False is gone.
        results = self.ingest_documents(
            [document], schema_name=schema_name, operation_type="update"
        )
        return results["success_count"] > 0

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

        if not schema_name:
            schema_name = self.config.get("schema_name")
        if not schema_name:
            raise ValueError("schema_name is required for delete operations")

        client = self._get_or_create_ingestion_client(schema_name)
        success = client.delete_document(document_id)

        if success:
            logger.info(f"Deleted document: {document_id}")
        else:
            logger.warning(f"Delete returned False for document: {document_id}")

        return success

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
            deployed = self.schema_manager.list_deployed_document_types(
                raise_on_failure=True
            )
            return schema_name in deployed
        except Exception as e:
            # An enumeration failure is a backend outage, not "schema
            # invalid" — flattening it to False collapses the two. The probe
            # must raise (raise_on_failure=True); the default swallows it to
            # [], which returns False and never reaches this re-raise.
            logger.error(f"Failed to validate schema {schema_name}: {e}")
            raise

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
            with self._search_backend_lock:
                if not self._vespa_search_backend:
                    logger.debug("Creating VespaSearchBackend on-demand with config")

                    # Ensure profiles are loaded (may be missing if ingestion
                    # created the cached backend instance without profiles).
                    if (
                        not self.config.get("profiles")
                        and self._config_manager_instance
                    ):
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
                                f"Loaded {len(self.config['profiles'])} profiles "
                                f"from config for tenant {self._tenant_id}"
                            )

                    search_backend = VespaSearchBackend(
                        config=self.config,
                        config_manager=self._config_manager_instance,
                        schema_loader=self._schema_loader_instance,
                    )
                    self._vespa_search_backend = search_backend
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

        document = Document(
            id=document_id,
            text_content=fields.get("text", ""),
            metadata={
                k: v for k, v in fields.items() if k not in ("text", "embedding", "id")
            },
        )
        embedding = fields.get("embedding")
        if embedding is not None:
            if isinstance(embedding, dict) and "values" in embedding:
                embedding = embedding["values"]
            document.add_embedding("embedding", embedding)
        return document

    def batch_get_documents(
        self, document_ids: List[str], schema_name: Optional[str] = None
    ) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID.

        Uses the search backend's YQL batch query if available (single round-trip),
        otherwise retrieves each document individually via the ingestion client.

        Args:
            document_ids: List of document IDs
            schema_name: Vespa schema to read from. Defaults to this backend's
                configured schema. The shared search backend rewrites its own
                schema attribute per request, so the schema is passed explicitly
                rather than read off that shared state.

        Returns:
            List of Documents (None for not found)
        """
        if not document_ids:
            return []

        if not schema_name:
            schema_name = self.config.get("schema_name")
        if not schema_name:
            raise ValueError("schema_name is required for batch document reads")

        if self._vespa_search_backend:
            return self._vespa_search_backend.batch_get_documents(
                document_ids,
                schema_name=schema_name,
                namespace=document_namespace(schema_name),
            )

        # No search backend — retrieve individually via ingestion client
        return [
            self.get_document(doc_id, schema_name=schema_name)
            for doc_id in document_ids
        ]

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
                ``contentTypeRemoval`` validation override. Schema discovery
                and survivor reconstruction remain mandatory. Defaults to
                False — an operator who actually wants to remove a schema must
                opt in explicitly.

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

            from cogniverse_core.registries.exceptions import BackendDeploymentError
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

            registry_schemas: List[Any] = []
            if self.schema_registry is not None:
                try:
                    registry_schemas = self.schema_registry._get_all_schemas() or []
                    for schema_info in registry_schemas:
                        full_name = schema_info.full_schema_name
                        if full_name in merged_schema_names:
                            continue
                        try:
                            existing_def = schema_info.schema_definition
                            if isinstance(existing_def, str):
                                if not existing_def.strip():
                                    raise ValueError("schema definition is empty")
                                existing_def = json.loads(existing_def)
                            existing_obj = parser_for_existing.parse_schema(
                                existing_def
                            )
                            merged_schemas.append(existing_obj)
                            merged_schema_names.add(full_name)
                        except Exception as merge_exc:
                            raise BackendDeploymentError(
                                f"Cannot reconstruct registry schema "
                                f"{full_name!r}; refusing to deploy a package "
                                f"that would omit it: {merge_exc}"
                            ) from merge_exc
                    logger.info(
                        f"Merged {len(merged_schemas) - len(schemas_to_deploy)} "
                        f"schemas from registry into deployment package"
                    )
                except Exception as registry_exc:
                    if isinstance(registry_exc, BackendDeploymentError):
                        raise
                    raise BackendDeploymentError(
                        "Cannot enumerate the schema registry before deploy: "
                        f"{registry_exc}"
                    ) from registry_exc

            # Second source: ask the config server what is currently
            # deployed. Any schema here that the registry didn't cover
            # must be reconstructed or the deploy fails — silently
            # dropping a peer-tenant schema is never acceptable. The
            # config-server listing is authoritative. A successful empty list
            # is a valid first deployment; a failed enumeration must abort.
            try:
                vespa_deployed = self.schema_manager.list_deployed_document_types(
                    raise_on_failure=True
                )
            except Exception as probe_exc:
                raise BackendDeploymentError(
                    "Cannot enumerate Vespa-deployed schemas before deploy: "
                    f"{probe_exc}"
                ) from probe_exc
            logger.info(
                f"Vespa-discovered schemas: {sorted(vespa_deployed)} "
                f"(registry merge added "
                f"{len(merged_schemas) - len(schemas_to_deploy)} schemas)"
            )

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
                for schema_info in registry_schemas:
                    registry_by_full_name[schema_info.full_schema_name] = schema_info

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

                if unresolved:
                    # Live-in-Vespa but unregistered and not reconstructable.
                    # Deploying a package without them tells Vespa to remove the
                    # document types and destroy every document they hold — and
                    # when the orphan is a peer tenant's schema mid-registration
                    # that is silent cross-tenant data loss. Refuse: a transient
                    # orphan clears on retry once the peer registers; a
                    # persistent one is a registry inconsistency to resolve.
                    raise BackendDeploymentError(
                        f"Refusing to deploy: {len(unresolved)} schema(s) live in "
                        f"Vespa have no registry entry and cannot be reconstructed "
                        f"({sorted(unresolved)}); proceeding would remove them and "
                        f"destroy their documents."
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

        except BackendDeploymentError:
            # A data-loss refusal (unregistered, unreconstructable schemas that
            # a redeploy would destroy) is NOT a transient failure — surface it
            # so the caller does not mistake it for a retryable False and force
            # the destructive deploy. Transient failures still return False.
            raise
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

            # Vespa serializes app-package activation: if another deploy
            # activated its session between our prepare and our activate,
            # the config server returns 409 ACTIVATION_CONFLICT and asks
            # us to redeploy. Under e2e sweep load (every K-System test
            # deploys 2 per-tenant schemas) this race fires often enough
            # to break ~3 tests per sweep. Retry the whole
            # prepareandactivate with exponential backoff — each retry
            # ships a fresh prepare against whatever's now active, so the
            # conflict resolves naturally.
            response = None
            last_error: Optional[str] = None
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                response = requests.post(
                    deploy_url,
                    headers={"Content-Type": "application/zip"},
                    data=app_zip,
                    verify=False,
                    timeout=DEPLOY_REQUEST_TIMEOUT_S,
                )
                if response.status_code == 200:
                    break
                # Parse error to detect ACTIVATION_CONFLICT (retriable).
                try:
                    error_detail = json.loads(response.content.decode("utf-8"))
                except Exception:
                    error_detail = {"message": response.content.decode("utf-8")[:300]}
                is_activation_conflict = (
                    response.status_code == 409
                    and error_detail.get("error-code") == "ACTIVATION_CONFLICT"
                )
                if not is_activation_conflict or attempt == max_attempts:
                    last_error = (
                        f"Deployment failed with status {response.status_code}: "
                        f"{error_detail}"
                    )
                    raise RuntimeError(last_error)
                # Backoff: 0.5s, 1s, 2s, 4s before final attempt.
                wait = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "Vespa deploy ACTIVATION_CONFLICT (attempt %d/%d) — "
                    "retrying in %.1fs: %s",
                    attempt,
                    max_attempts,
                    wait,
                    error_detail.get("message", ""),
                )
                import time as _t

                _t.sleep(wait)
            if response is not None and response.status_code == 200:
                logger.info("Successfully deployed application package")

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

        The discriminative probe is a per-schema YQL query
        (``select documentid from <name> where true limit 0``): while the
        content distributor hasn't loaded the doctype Vespa returns
        ``root.errors``; once loaded, the query returns 200 with no errors.
        Once every probed schema is visible we add a short buffer — search
        visibility converges a beat before the document API accepts feeds,
        so without this buffer the first feed still races.

        Args:
            schema_names: Names of schemas that were just deployed
            timeout: Maximum seconds to wait for convergence

        Raises:
            RuntimeError: If any schema is still not query-visible when the
                timeout expires. Reporting success for a schema Vespa never
                activated lets callers feed/search a nonexistent doctype.
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
        # Probe each schema directly via YQL `from <schema>`. When content
        # distributors haven't loaded the doctype yet, Vespa returns errors
        # like "Schema 'X' not found" or 4xx; once it's loaded, the query
        # returns 200 with an empty hit list and no errors. The previous
        # `model.restrict` form was unreliable on recent Vespa versions —
        # it silently returns 200 with empty results for unknown schemas
        # instead of erroring, falsely confirming convergence after a single
        # probe.
        remaining = set(schema_names)
        for i in range(timeout):
            for name in list(remaining):
                try:
                    response = requests.post(
                        probe_url,
                        json={
                            "yql": f"select documentid from {name} where true limit 0",
                            "hits": 0,
                        },
                        timeout=5,
                    )
                    if response.status_code != 200:
                        continue
                    body = response.json()
                    errors = body.get("root", {}).get("errors", [])
                    if not errors:
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

        raise RuntimeError(
            f"Schema convergence not confirmed after {timeout}s — deploy was "
            f"accepted by the config server but these schemas never became "
            f"query-visible: {sorted(remaining)}"
        )

    def delete_schema(
        self, schema_name: str, tenant_id: Optional[str] = None
    ) -> List[str]:
        """
        Delete one tenant-namespaced schema.

        Routes through the guarded ``VespaSchemaManager.delete_schema`` and
        raises on failure — including the refusal when the removal redeploy
        would also drop deployed schemas the registry does not know. The
        previous implementation ignored ``schema_name`` (it deleted every
        schema the tenant had) and swallowed failures into an empty list,
        which callers read as "nothing to delete".

        Args:
            schema_name: Base schema name to delete.
            tenant_id: Tenant identifier (uses self._tenant_id if not provided)

        Returns:
            Single-element list with the full deleted schema name.
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Use provided tenant_id or fall back to instance tenant_id
        effective_tenant_id = tenant_id or self._tenant_id
        if not effective_tenant_id:
            raise ValueError("tenant_id required for schema deletion")

        full_name = self.schema_manager.delete_schema(effective_tenant_id, schema_name)
        logger.info(f"Deleted schema '{full_name}' for tenant '{effective_tenant_id}'")
        return [full_name]

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
            # A lookup failure is a backend/registry outage, not "schema
            # missing" — flattening it to False lets the deploy route treat an
            # outage as "not deployed" and redeploy. Surface it instead.
            logger.error(
                f"Failed to check schema existence for '{schema_name}' tenant '{effective_tenant_id}': {e}"
            )
            raise

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

    def _metadata_vespa_app(self):
        """Cached pyvespa app for metadata ops; rebuilt only when url/port
        change so repeated metadata calls reuse one connection pool."""
        key = (self._url, self._port)
        # Guard the lazy (re)build: two concurrent first-touches would each
        # construct a PersistentVespaOps and leak the loser's session pool.
        with self._metadata_app_lock:
            if self._metadata_app is None or self._metadata_app_key != key:
                if self._metadata_app is not None:
                    self._metadata_app.close()
                # Persistent session: metadata CRUD runs per ingest/deploy —
                # per-op VespaSync handshakes multiplied every operation.
                self._metadata_app = make_persistent_vespa_ops(
                    url=self._url, port=self._port
                )
                self._metadata_app_key = key
            return self._metadata_app

    # The single sanctioned Vespa Document v1 surface for callers that own
    # their field shapes (wiki pages, knowledge-graph nodes/edges, content
    # back-refs). Hand-built ``/document/v1`` HTTP bypassed session reuse,
    # escaping, and the backend error contract. Namespace defaults to
    # pyvespa's behavior (namespace == schema) so existing callers are
    # unaffected. All writes RAISE on failure — the status and body ride in
    # the exception text so callers can match transient convergence shapes.

    @staticmethod
    def _coerce_field_values(fields: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce numpy values to native Python so the JSON step can't choke.

        np.int64 is NOT an int subclass — ``json.dumps`` raises TypeError on
        it (np.float64 slips through as a float subclass). Recurses into
        lists/dicts and converts ndarrays via ``.tolist()``: the shallow pass
        handled only top-level scalars, so nested numpy values still reached
        pyvespa un-serializable.
        """
        import numpy as np

        def coerce(v):
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: coerce(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [coerce(x) for x in v]
            return v

        return {k: coerce(v) for k, v in fields.items()}

    @staticmethod
    def _check_document_response(resp, op: str, document_id: str):
        # Defensive net: pyvespa's data-plane ops call raise_for_status
        # internally, so a 4xx/5xx normally surfaces as a VespaError (whose
        # text still carries Vespa's message, which the graph retry matches)
        # BEFORE reaching here. This catches the rare returns-non-2xx-without-
        # raising case and the documented raise-with-status contract.
        status = getattr(resp, "status_code", None)
        if status is not None and not (200 <= status < 300):
            body = getattr(resp, "json", None)
            raise RuntimeError(
                f"Vespa document {op} failed for '{document_id}' "
                f"(HTTP {status}): {body}"
            )
        return resp

    def put_document(
        self,
        document,
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
        base_schema_name: Optional[str] = None,
    ) -> None:
        """Full-put a generic ``cogniverse_sdk.document.Document``.

        Serializes through the schema's declared ``document_mapping`` block
        (loaded from the BASE schema JSON — pass ``base_schema_name`` when
        ``schema_name`` is tenant-suffixed). Raises ValueError when the
        schema declares no mapping: a generic Document is only feedable to
        schemas that say how their fields map, never by guessing.
        """
        from cogniverse_sdk.document import DocumentFieldMapping

        schema_name = schema_name or self.config.get("schema_name")
        base_name = base_schema_name or schema_name
        schema_json = self._schema_loader_instance.load_schema(base_name)
        mapping = DocumentFieldMapping.from_schema_json(
            schema_json, schema_name=base_name, required=True
        )
        self.put_document_fields(
            document.id,
            document.to_schema_fields(mapping),
            schema_name=schema_name,
            namespace=namespace,
        )

    def conditional_put_document(
        self,
        document,
        *,
        condition: str,
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
        base_schema_name: Optional[str] = None,
        create: bool = True,
    ) -> bool:
        """Test-and-set full-field write of a generic ``Document``.

        Serializes ``document`` through the schema's declared
        ``document_mapping`` (like :meth:`put_document`) and issues a Document
        v1 conditional partial update. With ``create=True`` a missing document
        is inserted (Vespa ignores the condition when the target does not
        exist) and an existing one is overwritten only while ``condition``
        still holds against it, so a caller's read-modify-write is safe against
        a racing writer. Returns True when the write applied, False when Vespa
        rejected the condition (HTTP 412) because another writer advanced the
        document since it was read. Raises on transport failure or any other
        non-2xx status so a lost write is never mistaken for a successful one.
        """
        from cogniverse_sdk.document import DocumentFieldMapping

        schema_name = schema_name or self.config.get("schema_name")
        base_name = base_schema_name or schema_name
        schema_json = self._schema_loader_instance.load_schema(base_name)
        mapping_cfg = (schema_json or {}).get("document_mapping")
        if not mapping_cfg:
            raise ValueError(
                f"Schema {base_name!r} declares no document_mapping — "
                f"add one to its schema JSON or feed schema-specific "
                f"fields via put_document_fields"
            )
        mapping = DocumentFieldMapping.from_dict(mapping_cfg)
        return self._conditional_update_fields(
            document.id,
            document.to_schema_fields(mapping),
            condition=condition,
            schema_name=schema_name,
            namespace=namespace,
            create=create,
        )

    def _conditional_update_fields(
        self,
        document_id: str,
        fields: Dict[str, Any],
        *,
        condition: str,
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
        create: bool = False,
    ) -> bool:
        """Partial-update guarded by a Vespa test-and-set ``condition``.

        Returns True when applied, False on an HTTP 412 condition mismatch.
        Raises on transport failure or any other non-2xx status — a rejected
        condition is the only non-raising failure, and it is a real Vespa
        response, not a masked outage.
        """
        schema_name = schema_name or self.config.get("schema_name")
        try:
            resp = self._metadata_vespa_app().update_data(
                schema=schema_name,
                data_id=document_id,
                fields=self._coerce_field_values(fields),
                namespace=namespace,
                create=create,
                condition=condition,
            )
        except Exception as exc:
            if _http_status_of(exc) == 412:
                return False
            raise
        if getattr(resp, "status_code", None) == 412:
            return False
        self._check_document_response(resp, "conditional update", document_id)
        return True

    def put_document_fields(
        self,
        document_id: str,
        fields: Dict[str, Any],
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Full-put a raw Vespa ``fields`` dict as one document."""
        schema_name = schema_name or self.config.get("schema_name")
        resp = self._metadata_vespa_app().feed_data_point(
            schema=schema_name,
            data_id=document_id,
            fields=self._coerce_field_values(fields),
            namespace=namespace,
        )
        self._check_document_response(resp, "put", document_id)

    def get_document_fields(
        self,
        document_id: str,
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch one document's raw ``fields`` dict, or None when absent."""
        schema_name = schema_name or self.config.get("schema_name")
        resp = self._metadata_vespa_app().get_data(
            schema=schema_name,
            data_id=document_id,
            namespace=namespace,
            raise_on_not_found=False,
        )
        status = getattr(resp, "status_code", None)
        if status == 404:
            return None
        self._check_document_response(resp, "get", document_id)
        body = getattr(resp, "json", {}) or {}
        return body.get("fields", {})

    def update_document_fields(
        self,
        document_id: str,
        fields: Dict[str, Any],
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
        create: bool = False,
    ) -> None:
        """Partial-update raw fields (pyvespa auto-wraps assign semantics)."""
        schema_name = schema_name or self.config.get("schema_name")
        resp = self._metadata_vespa_app().update_data(
            schema=schema_name,
            data_id=document_id,
            fields=self._coerce_field_values(fields),
            namespace=namespace,
            create=create,
        )
        self._check_document_response(resp, "update", document_id)

    def delete_document_fields(
        self,
        document_id: str,
        schema_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Delete one document under an explicit namespace."""
        schema_name = schema_name or self.config.get("schema_name")
        resp = self._metadata_vespa_app().delete_data(
            schema=schema_name,
            data_id=document_id,
            namespace=namespace,
        )
        self._check_document_response(resp, "delete", document_id)

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
            vespa_client = self._metadata_vespa_app()

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
            # An outage is not a rejected write — raise so callers never
            # read a lost write as a clean failure they may retry-skip.
            logger.error(f"Failed to create metadata document {schema}/{doc_id}: {e}")
            raise

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
            vespa_client = self._metadata_vespa_app()

            # Get metadata document
            response = vespa_client.get_data(schema=schema, data_id=doc_id)

            if not response or response.status_code != 200:
                # A genuine not-found: pyvespa returns a 404 as a non-raising
                # response, so this is "document absent", not a backend failure.
                return None

            result = response.json
            return result.get("fields", {})
        except Exception as e:
            # A backend failure (connection error, 5xx) is NOT "document not
            # found" — pyvespa surfaces a 404 as a non-raising response handled
            # above, so reaching here means the backend is unreachable/erroring.
            # Raise so callers can return 503 instead of a misleading 404 that
            # reads as "the document was deleted".
            logger.error(f"Failed to get metadata document {schema}/{doc_id}: {e}")
            raise

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
            **kwargs: Additional Vespa query options. ``tenant_id`` resolves the
                base schema to that tenant's canonical physical schema.

        Returns:
            List of matching documents as dicts
        """
        if not self._url:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        tenant_id = kwargs.pop("tenant_id", None)
        query_schema = schema
        if tenant_id:
            if schema == "*":
                raise ValueError(
                    "tenant-scoped metadata queries require one explicit base schema"
                )
            query_schema = self.get_tenant_schema_name(tenant_id, schema)
            if yql:
                source_pattern = re.compile(
                    rf"(\bfrom\s+(?:sources\s+)?){re.escape(schema)}(?=\s|$)",
                    re.IGNORECASE,
                )
                yql, replacement_count = source_pattern.subn(
                    lambda match: f"{match.group(1)}{query_schema}",
                    yql,
                )
                if replacement_count != 1:
                    raise ValueError(
                        f"Tenant-scoped YQL must name base schema {schema!r} "
                        "exactly once in its source clause"
                    )

        try:
            vespa_client = self._metadata_vespa_app()

            # Build query parameters
            hits = kwargs.pop("hits", 100)
            offset = kwargs.pop("offset", None)
            query_params = dict(kwargs)
            query_params["hits"] = hits
            # The default query profile caps hits at 400. Raise the native
            # Vespa window per request so large and paged metadata walks stay
            # legal, while still honoring any explicit native caps callers
            # already supplied.
            query_params.setdefault("maxHits", hits)
            query_params.setdefault("maxOffset", (offset or 0) + hits)
            # Forward paging offset as Vespa's native query parameter. A YQL
            # `offset` alone is bounded by `hits`, so the second page of a
            # walk lands outside the hits window and returns empty; the
            # explicit parameter pages past it.
            if offset:
                query_params["offset"] = offset

            if yql:
                query_params["yql"] = yql
                # If YQL contains userQuery(), also add the query parameter if provided
                if query and "userQuery()" in yql:
                    query_params["query"] = query
            elif query:
                # Use userQuery() for text search
                query_params["yql"] = f"select * from {query_schema} where userQuery()"
                query_params["query"] = query
            else:
                # Get all documents - Vespa requires at least one search term
                # Using a match-all pattern with limit
                query_params["yql"] = (
                    f"select * from {query_schema} where true limit {hits}"
                )

            # Execute query
            results = vespa_client.query(body=query_params)

            # Belt-and-braces: pyvespa >=1.1 raises VespaError on non-2xx via
            # raise_for_status, but keep the explicit check so a rejected query
            # can never silently return [] indistinguishable from a clean
            # empty result. Siblings (get_metadata_document,
            # delete_metadata_document) check status_code the same way.
            if results.status_code != 200:
                error_body = results.json if hasattr(results, "json") else None
                raise RuntimeError(
                    f"Vespa query returned HTTP {results.status_code} for schema "
                    f"{query_schema}: {error_body!r}"
                )

            # A soft timeout or degraded coverage arrives as HTTP 200 with
            # root.errors (and only partial children) — consuming the children
            # without this check returns a partial listing recorded as success.
            # Match the convergence probe and vespa_search_children: raise so a
            # degraded scan is never mistaken for a complete one.
            root = results.json.get("root", {})
            errors = root.get("errors")
            if errors:
                raise RuntimeError(
                    f"Vespa query returned errors for schema {query_schema} "
                    f"(HTTP 200 with a soft timeout or degraded coverage yields "
                    f"partial results): {errors!r}"
                )
            coverage = root.get("coverage") or {}
            if coverage.get("degraded"):
                raise RuntimeError(
                    "Vespa query coverage degraded for schema "
                    f"{query_schema}: {coverage!r}"
                )

            # Extract documents from response
            documents = []
            for hit in root.get("children", []):
                fields = hit.get("fields", {})
                documents.append(fields)

            logger.debug(
                f"Query returned {len(documents)} documents from {query_schema}"
            )
            return documents
        except Exception as e:
            # A rejected query or outage must not read as "no rows" — raise,
            # matching the config/adapter store contract. Callers that
            # deliberately degrade (provenance fetch, memory list) catch this.
            logger.error(f"Failed to query metadata documents from {schema}: {e!r}")
            raise

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
            vespa_client = self._metadata_vespa_app()

            response = vespa_client.delete_data(schema=schema, data_id=doc_id)

            if response.status_code != 200:
                logger.error(
                    f"Failed to delete metadata document {schema}/{doc_id}: "
                    f"HTTP {response.status_code}"
                )
                return False

            logger.debug(f"Deleted metadata document: {schema}/{doc_id}")
            return True
        except Exception as e:
            # An outage is not "already deleted" — raise so callers never
            # report a delete that did not happen.
            logger.error(f"Failed to delete metadata document {schema}/{doc_id}: {e}")
            raise

    def close(self) -> None:
        """
        Close connections to Vespa.
        """
        with self._close_lock:
            search_backend = getattr(self, "_vespa_search_backend", None)
            self._vespa_search_backend = None

            ingestion_clients = list(
                getattr(self, "_vespa_ingestion_clients", {}).items()
            )
            self._vespa_ingestion_clients = {}

            metadata_app = getattr(self, "_metadata_app", None)
            self._metadata_app = None
            self._metadata_app_key = None

            failures: list[tuple[str, Exception]] = []

            if search_backend is not None:
                try:
                    search_backend.close()
                except Exception as exc:
                    failures.append(("search backend", exc))

            for schema_name, client in ingestion_clients:
                try:
                    client.close()
                    logger.info(f"Closed Vespa client for schema: {schema_name}")
                except Exception as exc:
                    failures.append((f"ingestion client {schema_name}", exc))

            if metadata_app is not None:
                try:
                    metadata_app.close()
                except Exception as exc:
                    failures.append(("metadata client", exc))

            if failures:
                details = "; ".join(f"{name}: {exc}" for name, exc in failures)
                raise RuntimeError(
                    f"Failed to close Vespa backend resources: {details}"
                ) from failures[0][1]

            logger.info("Closed all Vespa backend connections")

    def health_check(self) -> bool:
        """
        Check Vespa health.

        Returns:
            True if healthy
        """
        if self._vespa_search_backend:
            # The search backend returns a rich status dict; this method's
            # contract (SearchBackend ABC) is bool — coerce instead of leaking
            # a dict that is always truthy even when degraded.
            health = self._vespa_search_backend.health_check()
            if isinstance(health, dict):
                return health.get("status") == "healthy"
            return bool(health)

        # Basic health check
        return self.schema_manager is not None

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
