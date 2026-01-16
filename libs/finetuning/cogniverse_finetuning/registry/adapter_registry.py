"""
Adapter Registry

Central registry for managing trained LoRA adapters.
Provides high-level API for adapter lifecycle management.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from cogniverse_finetuning.registry.models import AdapterMetadata

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """
    Central registry for managing trained adapters.

    Provides high-level API for:
    - Registering new adapters after training
    - Querying available adapters
    - Activating/deactivating adapters for deployment
    - Managing adapter lifecycle (deprecation, deletion)

    Example:
        >>> registry = AdapterRegistry()
        >>> adapter_id = registry.register_adapter(
        ...     tenant_id="tenant1",
        ...     name="routing_sft_v1",
        ...     version="1.0.0",
        ...     base_model="HuggingFaceTB/SmolLM-135M",
        ...     model_type="llm",
        ...     agent_type="routing",
        ...     training_method="sft",
        ...     adapter_path="/path/to/adapter",
        ...     metrics={"train_loss": 0.5}
        ... )
        >>> registry.activate_adapter(adapter_id)
        >>> active = registry.get_active_adapter("tenant1", "routing")
    """

    def __init__(self, store: Optional[Any] = None):
        """
        Initialize adapter registry.

        Args:
            store: VespaAdapterStore instance (optional).
                   If not provided, creates one from BootstrapConfig.
        """
        if store is not None:
            self.store = store
        else:
            # Create store from BootstrapConfig
            from cogniverse_foundation.config.bootstrap import BootstrapConfig
            from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

            bootstrap = BootstrapConfig.from_environment()
            self.store = VespaAdapterStore(
                vespa_url=bootstrap.backend_url,
                vespa_port=bootstrap.backend_port,
            )

        logger.info("AdapterRegistry initialized")

    def register_adapter(
        self,
        tenant_id: str,
        name: str,
        version: str,
        base_model: str,
        model_type: Literal["llm", "embedding"],
        training_method: Literal["sft", "dpo", "embedding"],
        adapter_path: str,
        agent_type: Optional[str] = None,
        adapter_uri: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        experiment_run_id: Optional[str] = None,
    ) -> str:
        """
        Register a new adapter in the registry.

        Args:
            tenant_id: Tenant this adapter belongs to
            name: Human-readable adapter name
            version: Semantic version string (e.g., "1.0.0")
            base_model: Base model the adapter was trained on
            model_type: Type of model - "llm" or "embedding"
            training_method: Training method used (sft, dpo, embedding)
            adapter_path: Local filesystem path to adapter weights
            agent_type: Target agent type (for LLM adapters)
            adapter_uri: Cloud storage URI (s3://, modal://, file://) for production
            metrics: Training metrics (optional)
            training_config: Training configuration (optional)
            experiment_run_id: MLflow/Phoenix experiment run ID (optional)

        Returns:
            adapter_id of the registered adapter

        Example:
            >>> adapter_id = registry.register_adapter(
            ...     tenant_id="tenant1",
            ...     name="routing_sft",
            ...     version="1.0.0",
            ...     base_model="SmolLM-135M",
            ...     model_type="llm",
            ...     training_method="sft",
            ...     adapter_path="outputs/adapters/sft_routing_20240101/",
            ...     adapter_uri="s3://bucket/adapters/routing_sft_v1/",
            ...     agent_type="routing",
            ...     metrics={"train_loss": 0.45}
            ... )
        """
        # Generate unique adapter ID
        adapter_id = str(uuid.uuid4())

        # Create metadata
        metadata = AdapterMetadata(
            adapter_id=adapter_id,
            tenant_id=tenant_id,
            name=name,
            version=version,
            base_model=base_model,
            model_type=model_type,
            agent_type=agent_type,
            training_method=training_method,
            adapter_path=adapter_path,
            adapter_uri=adapter_uri,
            status="inactive",
            is_active=False,
            metrics=metrics or {},
            training_config=training_config or {},
            experiment_run_id=experiment_run_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Save to store
        self.store.save_adapter(metadata.to_vespa_doc())

        logger.info(
            f"Registered adapter {adapter_id}: {name} v{version} "
            f"for tenant={tenant_id}, agent={agent_type}"
        )

        return adapter_id

    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """
        Get adapter by ID.

        Args:
            adapter_id: Unique adapter identifier

        Returns:
            AdapterMetadata or None if not found
        """
        doc = self.store.get_adapter(adapter_id)
        if doc is None:
            return None

        return AdapterMetadata.from_vespa_doc(doc)

    def list_adapters(
        self,
        tenant_id: str,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> List[AdapterMetadata]:
        """
        List adapters matching criteria.

        Args:
            tenant_id: Tenant identifier
            agent_type: Filter by agent type (None = all)
            status: Filter by status (None = all)
            model_type: Filter by model type (None = all)

        Returns:
            List of AdapterMetadata objects
        """
        docs = self.store.list_adapters(
            tenant_id=tenant_id,
            agent_type=agent_type,
            status=status,
            model_type=model_type,
        )

        return [AdapterMetadata.from_vespa_doc(doc) for doc in docs]

    def get_active_adapter(
        self, tenant_id: str, agent_type: str
    ) -> Optional[AdapterMetadata]:
        """
        Get the active adapter for a tenant and agent type.

        Args:
            tenant_id: Tenant identifier
            agent_type: Agent type

        Returns:
            Active AdapterMetadata or None if no active adapter
        """
        doc = self.store.get_active_adapter(tenant_id, agent_type)
        if doc is None:
            return None

        return AdapterMetadata.from_vespa_doc(doc)

    def activate_adapter(self, adapter_id: str) -> None:
        """
        Activate an adapter for deployment.

        Deactivates any previously active adapter for the same tenant+agent_type.

        Args:
            adapter_id: Adapter to activate

        Raises:
            ValueError: If adapter not found
        """
        # Get adapter to find tenant_id and agent_type
        adapter = self.get_adapter(adapter_id)
        if adapter is None:
            raise ValueError(f"Adapter not found: {adapter_id}")

        if not adapter.agent_type:
            raise ValueError(
                f"Cannot activate adapter {adapter_id}: agent_type is required"
            )

        self.store.set_active(adapter_id, adapter.tenant_id, adapter.agent_type)
        logger.info(
            f"Activated adapter {adapter_id} for "
            f"{adapter.tenant_id}/{adapter.agent_type}"
        )

    def deactivate_adapter(self, adapter_id: str) -> None:
        """
        Deactivate an adapter.

        Args:
            adapter_id: Adapter to deactivate
        """
        self.store.deactivate_adapter(adapter_id)
        logger.info(f"Deactivated adapter {adapter_id}")

    def deprecate_adapter(self, adapter_id: str) -> None:
        """
        Deprecate an adapter.

        Marks adapter as deprecated and deactivates it.

        Args:
            adapter_id: Adapter to deprecate
        """
        self.store.deprecate_adapter(adapter_id)
        logger.info(f"Deprecated adapter {adapter_id}")

    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter from the registry.

        Note: This only removes the registry entry, not the actual adapter files.

        Args:
            adapter_id: Adapter to delete

        Returns:
            True if deleted, False if not found
        """
        result = self.store.delete_adapter(adapter_id)
        if result:
            logger.info(f"Deleted adapter {adapter_id} from registry")
        return result

    def get_latest_version(
        self, tenant_id: str, name: str, agent_type: Optional[str] = None
    ) -> Optional[AdapterMetadata]:
        """
        Get the latest version of an adapter by name.

        Args:
            tenant_id: Tenant identifier
            name: Adapter name
            agent_type: Agent type (optional filter)

        Returns:
            Latest AdapterMetadata or None if not found
        """
        adapters = self.list_adapters(
            tenant_id=tenant_id,
            agent_type=agent_type,
        )

        # Filter by name and find latest version
        matching = [a for a in adapters if a.name == name]
        if not matching:
            return None

        # Sort by version (assumes semantic versioning)
        matching.sort(key=lambda a: a.version, reverse=True)
        return matching[0]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        return self.store.get_stats()

    def health_check(self) -> bool:
        """
        Check if registry is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self.store.health_check()


def generate_adapter_id() -> str:
    """Generate a unique adapter ID."""
    return str(uuid.uuid4())
