"""AdapterStore Abstract Interface

Defines the interface for adapter registry storage backends.
Supports multiple implementations: Vespa, SQL databases, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AdapterStore(ABC):
    """
    Abstract base class for adapter metadata storage.

    Implementations must provide storage for trained adapter metadata,
    supporting CRUD operations and activation management.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the adapter store."""
        pass

    @abstractmethod
    def save_adapter(self, metadata: Dict[str, Any]) -> str:
        """
        Save adapter metadata.

        Args:
            metadata: Adapter metadata dictionary

        Returns:
            Adapter ID
        """
        pass

    @abstractmethod
    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get adapter metadata by ID.

        Args:
            adapter_id: Adapter identifier

        Returns:
            Adapter metadata or None if not found
        """
        pass

    @abstractmethod
    def list_adapters(
        self,
        tenant_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List adapters with optional filters.

        Args:
            tenant_id: Filter by tenant
            agent_type: Filter by agent type
            model_type: Filter by model type (llm, embedding)
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of adapter metadata dictionaries
        """
        pass

    @abstractmethod
    def get_active_adapter(
        self,
        tenant_id: str,
        agent_type: str,
        model_type: str = "llm",
    ) -> Optional[Dict[str, Any]]:
        """
        Get the active adapter for a tenant/agent/model combination.

        Args:
            tenant_id: Tenant identifier
            agent_type: Agent type (routing, search, etc.)
            model_type: Model type (llm, embedding)

        Returns:
            Active adapter metadata or None
        """
        pass

    @abstractmethod
    def set_active(self, adapter_id: str, tenant_id: str, agent_type: str) -> None:
        """
        Set an adapter as active for a tenant/agent combination.

        Args:
            adapter_id: Adapter to activate
            tenant_id: Tenant identifier
            agent_type: Agent type
        """
        pass

    @abstractmethod
    def deactivate_adapter(self, adapter_id: str) -> None:
        """
        Deactivate an adapter.

        Args:
            adapter_id: Adapter to deactivate
        """
        pass

    @abstractmethod
    def deprecate_adapter(self, adapter_id: str) -> None:
        """
        Deprecate an adapter.

        Marks adapter as deprecated and deactivates it.

        Args:
            adapter_id: Adapter to deprecate
        """
        pass

    @abstractmethod
    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter.

        Args:
            adapter_id: Adapter to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with stats (total_adapters, active_adapters, etc.)
        """
        pass
