"""Unified search service that coordinates query encoding and backend search.

Profile-agnostic: accepts profile and tenant_id at search() time, not at construction.
Caches encoders by model_name and backends by tenant_id for efficiency.
"""

import logging
from typing import Any, Dict, List, Optional

from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import get_backend_registry

from .base import SearchResult

logger = logging.getLogger(__name__)


class SearchService:
    """Unified search service for video retrieval.

    Profile-agnostic: ONE instance serves all profiles and tenants.
    Encoders are cached by model_name (via QueryEncoderFactory).
    Backends are lazily created per tenant_id.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_manager=None,
        schema_loader=None,
    ):
        """
        Initialize search service (profile-agnostic).

        Args:
            config: Configuration dictionary (full config.json content)
            config_manager: ConfigManager instance for dependency injection (REQUIRED)
            schema_loader: SchemaLoader instance for dependency injection (REQUIRED)
        """
        self.config = config

        if config_manager is None:
            raise ValueError(
                "config_manager is required for SearchService. "
                "Dependency injection is mandatory - pass create_default_config_manager() explicitly."
            )
        self.config_manager = config_manager

        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for SearchService. "
                "Dependency injection is mandatory - pass SchemaLoader instance explicitly."
            )
        self.schema_loader = schema_loader

        # Lazy backend cache: tenant_id → search backend
        self._backends: Dict[str, Any] = {}

        # Initialize telemetry
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        get_telemetry_manager()

        logger.info("SearchService initialized (profile-agnostic)")

    def _get_profile_config(self, profile: str) -> Dict[str, Any]:
        """Get profile configuration from backend config."""
        backend_config = self.config.get("backend", {})
        profiles = backend_config.get("profiles", {})
        profile_config = profiles.get(profile)

        if not profile_config:
            raise ValueError(
                f"Profile '{profile}' not found in backend.profiles. "
                f"Available profiles: {list(profiles.keys())}"
            )

        return profile_config

    def _get_encoder(self, profile: str, profile_config: Dict[str, Any]):
        """Get or create cached encoder for the given profile."""
        model_name = profile_config.get("embedding_model")
        if not model_name:
            raise ValueError(
                f"Profile '{profile}' missing 'embedding_model' configuration"
            )

        return QueryEncoderFactory.create_encoder(
            profile, model_name, config=self.config
        )

    def _get_backend(self, tenant_id: str, profile: str, profile_config: Dict[str, Any], query_encoder):
        """Get or create cached search backend for the given tenant."""
        if tenant_id in self._backends:
            return self._backends[tenant_id]

        backend_type = self.config.get("search_backend", "vespa")
        schema_name = profile_config.get("schema_name")

        if not schema_name:
            raise ValueError(
                f"Profile '{profile}' missing 'schema_name' configuration"
            )

        backend_registry = get_backend_registry()

        backend_section = self.config.get("backend", {})
        backend_config = {
            "backend_url": self.config.get("backend_url")
            or backend_section.get("url", "http://localhost"),
            "backend_port": self.config.get("backend_port")
            or backend_section.get("port", 8080),
            "schema_name": schema_name,
            "profile": profile,
            "query_encoder": query_encoder,
            "profiles": backend_section.get("profiles", {}),
            "default_profiles": backend_section.get("default_profiles", {}),
        }

        backend = backend_registry.get_search_backend(
            backend_type,
            tenant_id,
            backend_config,
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        self._backends[tenant_id] = backend
        logger.info(
            f"Created {backend_type} search backend for tenant: {tenant_id}"
        )
        return backend

    def search(
        self,
        query: str,
        profile: str,
        tenant_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for videos matching the query.

        Args:
            query: Text query
            profile: Backend profile to use for this search
            tenant_id: Tenant identifier
            top_k: Number of results to return
            filters: Optional filters (date range, etc.)
            ranking_strategy: Optional ranking strategy override

        Returns:
            List of SearchResult objects
        """
        from cogniverse_foundation.telemetry.context import (
            add_embedding_details_to_span,
            add_search_results_to_span,
            backend_search_span,
            encode_span,
            search_span,
        )

        # Resolve profile config and encoder
        profile_config = self._get_profile_config(profile)
        query_encoder = self._get_encoder(profile, profile_config)
        search_backend = self._get_backend(tenant_id, profile, profile_config, query_encoder)

        logger.info(f"Searching profile={profile} tenant={tenant_id}")

        with search_span(
            tenant_id=tenant_id,
            query=query,
            top_k=top_k,
            ranking_strategy=ranking_strategy or "default",
            profile=profile,
            backend=self.config.get("search_backend", "vespa"),
        ) as search_span_ctx:

            if ranking_strategy:
                logger.info(f"Using ranking strategy: {ranking_strategy}")

            # Generate embeddings with telemetry
            query_embeddings = None
            if query_encoder:
                encoder_type = (
                    type(query_encoder)
                    .__name__.lower()
                    .replace("queryencoder", "")
                )

                with encode_span(
                    tenant_id=tenant_id,
                    encoder_type=encoder_type,
                    query_length=len(query),
                    query=query,
                ) as encode_span_ctx:
                    query_embeddings = query_encoder.encode(query)
                    add_embedding_details_to_span(encode_span_ctx, query_embeddings)

            # Add embeddings info to search span
            if query_embeddings is not None:
                search_span_ctx.set_attribute("has_embeddings", True)
                search_span_ctx.set_attribute(
                    "embedding_shape", str(query_embeddings.shape)
                )
            else:
                search_span_ctx.set_attribute("has_embeddings", False)

            # Call backend
            schema_name = profile_config.get("schema_name")
            with backend_search_span(
                tenant_id=tenant_id,
                backend_type="vespa",
                schema_name=schema_name,
                ranking_strategy=ranking_strategy or "default",
                top_k=top_k,
                has_embeddings=query_embeddings is not None,
                query_text=query,
            ) as backend_span_ctx:
                if query_embeddings is not None:
                    backend_span_ctx.set_attribute(
                        "embedding_shape", str(query_embeddings.shape)
                    )

                query_dict = {
                    "query": query,
                    "type": "video",
                    "profile": profile,
                    "strategy": ranking_strategy or "default",
                    "top_k": top_k,
                    "filters": filters,
                }
                if query_embeddings is not None:
                    query_dict["query_embeddings"] = query_embeddings
                results = search_backend.search(query_dict)

                add_search_results_to_span(backend_span_ctx, results)

            add_search_results_to_span(search_span_ctx, results)

            return results

    def get_document(self, document_id: str, tenant_id: str, profile: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: Document ID
            tenant_id: Tenant identifier
            profile: Backend profile

        Returns:
            Document as dictionary or None if not found
        """
        profile_config = self._get_profile_config(profile)
        query_encoder = self._get_encoder(profile, profile_config)
        backend = self._get_backend(tenant_id, profile, profile_config, query_encoder)

        doc = backend.get_document(document_id)
        if doc:
            return {
                "document_id": doc.id,
                "source_id": doc.content_id,
                "content_type": doc.content_type.value,
                "metadata": doc.metadata,
                "temporal_info": (
                    {
                        "start_time": doc.metadata.get("start_time"),
                        "end_time": doc.metadata.get("end_time"),
                    }
                    if doc.metadata.get("start_time") is not None
                    else None
                ),
            }
        return None
