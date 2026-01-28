"""
Vespa-backed storage for adapter registry.

Stores trained adapter metadata in Vespa for multi-tenant management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from vespa.application import Vespa

logger = logging.getLogger(__name__)


class VespaAdapterStore:
    """
    Vespa-backed storage for adapter metadata.

    Stores adapter metadata as Vespa documents in the adapter_registry schema.
    Provides CRUD operations and activation management for adapters.

    Schema: adapter_registry
    Document structure:
    {
        "fields": {
            "adapter_id": "uuid-string",
            "tenant_id": "tenant1",
            "name": "routing_sft_v1",
            "version": "1.0.0",
            "base_model": "HuggingFaceTB/SmolLM-135M",
            "model_type": "llm",
            "agent_type": "routing",
            "training_method": "sft",
            "adapter_path": "/path/to/adapter",
            "status": "active",
            "is_active": 1,
            "metrics": "{}",
            "training_config": "{}",
            "experiment_run_id": "run_xxx",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    }
    """

    def __init__(
        self,
        vespa_app: Optional[Vespa] = None,
        vespa_url: str = "http://localhost",
        vespa_port: int = 8080,
        schema_name: str = "adapter_registry",
    ):
        """
        Initialize Vespa adapter store.

        Args:
            vespa_app: Existing Vespa application instance (optional)
            vespa_url: Vespa server URL
            vespa_port: Vespa server port
            schema_name: Vespa schema name for adapter storage
        """
        if vespa_app is not None:
            self.vespa_app = vespa_app
        else:
            self.vespa_app = Vespa(url=f"{vespa_url}:{vespa_port}")

        self.schema_name = schema_name
        logger.info(
            f"VespaAdapterStore initialized with schema: {schema_name} "
            f"at {vespa_url}:{vespa_port}"
        )

    def initialize(self) -> None:
        """
        Initialize the adapter store.

        For Vespa, this assumes the schema already exists.
        Schema must be deployed separately via vespa-cli or application package.
        """
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            logger.info(f"Vespa schema '{self.schema_name}' is accessible")
        except Exception as e:
            logger.warning(
                f"Could not verify Vespa schema '{self.schema_name}': {e}. "
                "Ensure schema is deployed before using VespaAdapterStore."
            )

    def save_adapter(self, metadata: Dict[str, Any]) -> str:
        """
        Save adapter metadata to Vespa.

        Args:
            metadata: Adapter metadata dict (from AdapterMetadata.to_vespa_doc())

        Returns:
            adapter_id of the saved adapter

        Raises:
            ValueError: If adapter_id is missing
            Exception: If Vespa operation fails
        """
        adapter_id = metadata.get("adapter_id")
        if not adapter_id:
            raise ValueError("adapter_id is required in metadata")

        # Create document ID
        doc_id = f"{self.schema_name}::{adapter_id}"

        # Feed document to Vespa
        try:
            self.vespa_app.feed_data_point(
                schema=self.schema_name,
                data_id=doc_id,
                fields=metadata,
            )
            logger.info(f"Saved adapter {adapter_id} to Vespa")
            return adapter_id

        except Exception as e:
            logger.error(f"Failed to save adapter to Vespa: {e}")
            raise

    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get adapter metadata by ID.

        Args:
            adapter_id: Unique adapter identifier

        Returns:
            Adapter metadata dict or None if not found
        """
        yql = f'select * from {self.schema_name} where adapter_id contains "{adapter_id}" limit 1'

        try:
            response = self.vespa_app.query(yql=yql)

            if not response.hits or len(response.hits) == 0:
                return None

            return {"fields": response.hits[0]["fields"]}

        except Exception as e:
            logger.error(f"Failed to retrieve adapter from Vespa: {e}")
            return None

    def list_adapters(
        self,
        tenant_id: str,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List adapters matching criteria.

        Args:
            tenant_id: Tenant identifier
            agent_type: Filter by agent type (None = all)
            status: Filter by status (None = all)
            model_type: Filter by model type (None = all)
            limit: Maximum number of results

        Returns:
            List of adapter metadata dicts
        """
        conditions = [f'tenant_id contains "{tenant_id}"']

        if agent_type:
            conditions.append(f'agent_type contains "{agent_type}"')

        if status:
            conditions.append(f'status contains "{status}"')

        if model_type:
            conditions.append(f'model_type contains "{model_type}"')

        where_clause = " and ".join(conditions)
        yql = f"select * from {self.schema_name} where {where_clause} limit {limit}"

        try:
            response = self.vespa_app.query(yql=yql)

            results = []
            for hit in response.hits:
                results.append({"fields": hit["fields"]})

            return results

        except Exception as e:
            logger.error(f"Failed to list adapters from Vespa: {e}")
            return []

    def get_active_adapter(
        self, tenant_id: str, agent_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the active adapter for a tenant and agent type.

        Args:
            tenant_id: Tenant identifier
            agent_type: Agent type

        Returns:
            Active adapter metadata or None if no active adapter
        """
        yql = (
            f"select * from {self.schema_name} "
            f'where tenant_id contains "{tenant_id}" '
            f'and agent_type contains "{agent_type}" '
            f"and is_active = 1 "
            f"limit 1"
        )

        try:
            response = self.vespa_app.query(yql=yql)

            if not response.hits or len(response.hits) == 0:
                return None

            return {"fields": response.hits[0]["fields"]}

        except Exception as e:
            logger.error(f"Failed to get active adapter from Vespa: {e}")
            return None

    def set_active(self, adapter_id: str, tenant_id: str, agent_type: str) -> None:
        """
        Set an adapter as active for a tenant and agent type.

        Deactivates any previously active adapter for the same tenant+agent_type.

        Args:
            adapter_id: Adapter to activate
            tenant_id: Tenant identifier
            agent_type: Agent type

        Raises:
            ValueError: If adapter not found
            Exception: If Vespa operation fails
        """
        # First, deactivate any currently active adapter
        current_active = self.get_active_adapter(tenant_id, agent_type)
        if current_active:
            current_id = current_active["fields"]["adapter_id"]
            if current_id != adapter_id:
                self._update_adapter_field(current_id, "is_active", 0)
                self._update_adapter_field(current_id, "status", "inactive")
                logger.info(f"Deactivated previous adapter {current_id}")

        # Activate the new adapter
        self._update_adapter_field(adapter_id, "is_active", 1)
        self._update_adapter_field(adapter_id, "status", "active")
        logger.info(f"Activated adapter {adapter_id} for {tenant_id}/{agent_type}")

    def deactivate_adapter(self, adapter_id: str) -> None:
        """
        Deactivate an adapter.

        Args:
            adapter_id: Adapter to deactivate
        """
        self._update_adapter_field(adapter_id, "is_active", 0)
        self._update_adapter_field(adapter_id, "status", "inactive")
        logger.info(f"Deactivated adapter {adapter_id}")

    def deprecate_adapter(self, adapter_id: str) -> None:
        """
        Deprecate an adapter.

        Marks adapter as deprecated and deactivates it.

        Args:
            adapter_id: Adapter to deprecate
        """
        self._update_adapter_field(adapter_id, "is_active", 0)
        self._update_adapter_field(adapter_id, "status", "deprecated")
        logger.info(f"Deprecated adapter {adapter_id}")

    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter from the registry.

        Args:
            adapter_id: Adapter to delete

        Returns:
            True if deleted, False if not found
        """
        doc_id = f"{self.schema_name}::{adapter_id}"

        try:
            self.vespa_app.delete_data(schema=self.schema_name, data_id=doc_id)
            logger.info(f"Deleted adapter {adapter_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete adapter {adapter_id}: {e}")
            return False

    def _update_adapter_field(
        self, adapter_id: str, field_name: str, field_value: Any
    ) -> None:
        """
        Update a single field on an adapter.

        Args:
            adapter_id: Adapter identifier
            field_name: Field to update
            field_value: New value

        Raises:
            ValueError: If adapter not found
        """
        # Get current adapter
        adapter = self.get_adapter(adapter_id)
        if not adapter:
            raise ValueError(f"Adapter not found: {adapter_id}")

        # Filter out Vespa system fields that can't be written back
        system_fields = {"sddocname", "documentid", "relevance"}
        fields = {k: v for k, v in adapter["fields"].items() if k not in system_fields}

        # Update field
        fields[field_name] = field_value
        fields["updated_at"] = datetime.utcnow().isoformat()

        # Save back
        doc_id = f"{self.schema_name}::{adapter_id}"
        self.vespa_app.feed_data_point(
            schema=self.schema_name,
            data_id=doc_id,
            fields=fields,
        )

    def health_check(self) -> bool:
        """
        Check if storage backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            return True
        except Exception as e:
            logger.error(f"Vespa health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            yql = f"select adapter_id, tenant_id, status from {self.schema_name} where true limit 1000"
            response = self.vespa_app.query(yql=yql)

            total_adapters = len(response.hits)
            unique_tenants = len(
                set(hit["fields"]["tenant_id"] for hit in response.hits)
            )

            # Count by status
            status_counts: Dict[str, int] = {}
            for hit in response.hits:
                status = hit["fields"].get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_adapters": total_adapters,
                "total_tenants": unique_tenants,
                "adapters_by_status": status_counts,
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
            }

        except Exception as e:
            logger.error(f"Failed to get stats from Vespa: {e}")
            return {
                "total_adapters": 0,
                "total_tenants": 0,
                "adapters_by_status": {},
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
                "error": str(e),
            }
