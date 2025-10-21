"""
Tenant-Aware Vespa Search Client

Wraps VespaVideoSearchClient to provide automatic tenant schema routing and lazy creation.
All search operations are automatically scoped to the tenant's schema.

Key Features:
- Automatic schema name resolution: base_schema + tenant_id → tenant_schema
- Lazy schema creation on first use
- Transparent delegation to underlying VespaVideoSearchClient
- Thread-safe tenant isolation

Example:
    # Create tenant-aware client
    client = TenantAwareVespaSearchClient(
        tenant_id="acme",
        base_schema_name="video_colpali_smol500_mv_frame",
        vespa_url="http://localhost",
        vespa_port=8080
    )

    # Search - automatically uses tenant schema
    results = client.search(
        query_text="robots playing soccer",
        strategy="hybrid_float_bm25",
        top_k=10
    )

    # All search methods automatically scoped to tenant
    results = client.hybrid_search(...)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from cogniverse_vespa.tenant_schema_manager import (
    get_tenant_schema_manager,
)
from cogniverse_vespa.vespa_search_client import (
    VespaVideoSearchClient,
)

logger = logging.getLogger(__name__)


class TenantAwareVespaSearchClient:
    """
    Tenant-aware wrapper for VespaVideoSearchClient.

    Handles tenant schema routing and ensures schema exists before searching.
    All search operations are automatically scoped to the tenant's schema.
    """

    def __init__(
        self,
        tenant_id: str,
        base_schema_name: str,
        vespa_url: str = "http://localhost",
        vespa_port: int = 8080,
        auto_create_schema: bool = True,
    ):
        """
        Initialize tenant-aware search client.

        Args:
            tenant_id: Tenant identifier (REQUIRED)
            base_schema_name: Base schema name (e.g., "video_colpali_smol500_mv_frame")
            vespa_url: Vespa endpoint URL
            vespa_port: Vespa port number
            auto_create_schema: Automatically create schema if it doesn't exist

        Raises:
            ValueError: If tenant_id or base_schema_name is invalid
            SchemaNotFoundException: If base schema template not found
            SchemaDeploymentException: If schema deployment fails

        Example:
            >>> client = TenantAwareVespaSearchClient(
            ...     tenant_id="acme",
            ...     base_schema_name="video_colpali_smol500_mv_frame"
            ... )
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not base_schema_name:
            raise ValueError("base_schema_name is required")

        self.tenant_id = tenant_id
        self.base_schema_name = base_schema_name
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port

        # Initialize schema manager
        self.schema_manager = get_tenant_schema_manager(
            vespa_url=vespa_url, vespa_port=vespa_port
        )

        # Resolve tenant-specific schema name
        self.tenant_schema_name = self.schema_manager.get_tenant_schema_name(
            tenant_id, base_schema_name
        )

        # Lazy schema creation
        if auto_create_schema:
            logger.info(
                f"Ensuring schema exists: {self.tenant_schema_name} for tenant {tenant_id}"
            )
            self.schema_manager.ensure_tenant_schema_exists(tenant_id, base_schema_name)

        # Initialize underlying search client
        self.client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)

        logger.info(
            f"✅ TenantAwareVespaSearchClient initialized: "
            f"tenant={tenant_id}, schema={self.tenant_schema_name}"
        )

    def search(
        self,
        query_text: str = "",
        embeddings: Optional[np.ndarray] = None,
        strategy: str = "hybrid_float_bm25",
        top_k: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeout: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search videos within tenant's schema.

        Args:
            query_text: Text query
            embeddings: Query embeddings (optional, generated if not provided)
            strategy: Ranking strategy
            top_k: Number of results to return
            start_date: Filter results after this date (YYYY-MM-DD)
            end_date: Filter results before this date (YYYY-MM-DD)
            timeout: Query timeout in seconds

        Returns:
            List of search results

        Example:
            >>> results = client.search(
            ...     query_text="robots playing soccer",
            ...     strategy="hybrid_float_bm25",
            ...     top_k=10
            ... )
        """
        logger.debug(
            f"Search request for tenant {self.tenant_id}: "
            f"query='{query_text[:50]}...', strategy={strategy}"
        )

        # Package parameters for underlying VespaVideoSearchClient.search() method
        query_params = {
            "query": query_text,
            "ranking": strategy,
            "top_k": top_k,
        }

        # Add optional date filters if provided
        if start_date:
            query_params["start_date"] = start_date
        if end_date:
            query_params["end_date"] = end_date

        # Delegate to underlying client with tenant schema
        logger.info(f"🔍 [SEARCH] Schema name for search: '{self.tenant_schema_name}'")
        logger.info(f"🔍 [SEARCH] Query params: query='{query_text[:50]}', strategy={strategy}")
        return self.client.search(
            query_params=query_params,
            embeddings=embeddings,
            schema=self.tenant_schema_name,  # ✅ Tenant schema automatically used
        )

    def hybrid_search(
        self,
        query_text: str,
        embeddings: Optional[np.ndarray] = None,
        visual_weight: float = 0.7,
        text_weight: float = 0.3,
        top_k: int = 10,
        timeout: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining visual and text signals within tenant's schema.

        Args:
            query_text: Text query
            embeddings: Query embeddings (optional)
            visual_weight: Weight for visual similarity
            text_weight: Weight for text similarity
            top_k: Number of results
            timeout: Query timeout

        Returns:
            List of search results

        Example:
            >>> results = client.hybrid_search(
            ...     query_text="robots",
            ...     visual_weight=0.7,
            ...     text_weight=0.3
            ... )
        """
        logger.debug(
            f"Hybrid search for tenant {self.tenant_id}: "
            f"visual_weight={visual_weight}, text_weight={text_weight}"
        )

        # Delegate to underlying client with tenant schema
        return self.client.hybrid_search(
            query_text=query_text,
            embeddings=embeddings,
            visual_weight=visual_weight,
            text_weight=text_weight,
            top_k=top_k,
            timeout=timeout,
            schema=self.tenant_schema_name,  # ✅ Tenant schema automatically used
        )

    def health_check(self) -> bool:
        """
        Check if Vespa endpoint is healthy.

        Returns:
            True if healthy, False otherwise

        Example:
            >>> if client.health_check():
            ...     print("Vespa is healthy")
        """
        return self.client.health_check()

    def get_available_strategies(self) -> Dict[str, Dict[str, str]]:
        """
        Get available ranking strategies.

        Returns:
            Dictionary of strategy name to strategy info

        Example:
            >>> strategies = client.get_available_strategies()
            >>> print(strategies.keys())
        """
        return self.client.get_available_strategies()

    def validate_strategy_inputs(
        self, strategy: str, query_text: str, embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Validate inputs for a given strategy.

        Args:
            strategy: Ranking strategy name
            query_text: Text query
            embeddings: Query embeddings

        Returns:
            List of validation errors (empty if valid)

        Example:
            >>> errors = client.validate_strategy_inputs("hybrid_float_bm25", "test", None)
            >>> if not errors:
            ...     print("Inputs valid")
        """
        return self.client.validate_strategy_inputs(strategy, query_text, embeddings)

    def get_tenant_info(self) -> Dict[str, str]:
        """
        Get information about the tenant and schema.

        Returns:
            Dictionary with tenant_id, base_schema_name, tenant_schema_name

        Example:
            >>> info = client.get_tenant_info()
            >>> print(f"Tenant: {info['tenant_id']}, Schema: {info['tenant_schema_name']}")
        """
        return {
            "tenant_id": self.tenant_id,
            "base_schema_name": self.base_schema_name,
            "tenant_schema_name": self.tenant_schema_name,
            "vespa_url": self.vespa_url,
            "vespa_port": str(self.vespa_port),
        }

    def get_video_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for search results.

        Args:
            results: Search results from search() or hybrid_search()

        Returns:
            Dictionary with summary statistics

        Example:
            >>> results = client.search("robots")
            >>> summary = client.get_video_summary(results)
            >>> print(f"Found {summary['total_videos']} unique videos")
        """
        return self.client.get_video_summary(results)

    def recommend_strategy(
        self, query_text: str, has_embeddings: bool = True, latency_sensitive: bool = False
    ) -> str:
        """
        Recommend best ranking strategy based on query characteristics.

        Args:
            query_text: Text query
            has_embeddings: Whether embeddings are available
            latency_sensitive: Whether low latency is critical

        Returns:
            Recommended strategy name

        Example:
            >>> strategy = client.recommend_strategy("robots", has_embeddings=True)
            >>> results = client.search("robots", strategy=strategy)
        """
        return self.client.recommend_strategy(query_text, has_embeddings, latency_sensitive)

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TenantAwareVespaSearchClient(tenant={self.tenant_id}, "
            f"schema={self.tenant_schema_name})"
        )
