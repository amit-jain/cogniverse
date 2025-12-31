"""
Tenant-Aware Mixin for Agents

Provides standardized multi-tenant support for all agents.
Handles tenant ID validation, storage, and context management.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_foundation.config.utils import get_config

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager

logger = logging.getLogger(__name__)


class TenantAwareAgentMixin:
    """
    Mixin class that adds multi-tenant capabilities to agents.

    This mixin eliminates code duplication by providing a standardized
    tenant_id validation and storage pattern used across all agents.

    Design Philosophy:
    - REQUIRED tenant_id: No defaults, explicit tenant identification
    - Fail-fast validation: Raises ValueError immediately on invalid tenant_id
    - Context helpers: Provides utilities for tenant-scoped operations
    - Config integration: Optionally loads tenant-specific configuration

    Usage with type-safe A2AAgent (preferred):
        class MyAgentDeps(AgentDeps):
            pass  # Only needs tenant_id from base

        class MyAgent(A2AAgent[MyInput, MyOutput, MyAgentDeps]):
            def __init__(self, deps: MyAgentDeps, port: int = 8000):
                # A2AAgent automatically handles tenant_id via deps.tenant_id
                super().__init__(deps=deps, config=..., dspy_module=...)

                # Now self.tenant_id is available via deps
                print(f"Agent initialized for tenant: {deps.tenant_id}")

    Legacy usage (for backward compatibility only):
        class MyAgent(A2AAgent, TenantAwareAgentMixin):
            def __init__(self, tenant_id: str, **kwargs):
                TenantAwareAgentMixin.__init__(self, tenant_id)
                super().__init__(...)

    Key Benefits:
    - Eliminates ~10 lines of duplicated validation code per agent
    - Consistent error messages across all agents
    - Standardized tenant context API
    - Easy to extend with additional tenant utilities
    """

    def __init__(
        self,
        tenant_id: str,
        config: Optional[SystemConfig] = None,
        config_manager: Optional["ConfigManager"] = None,
        **kwargs
    ):
        """
        Initialize tenant-aware agent mixin.

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Optional system configuration
            config_manager: Optional ConfigManager instance (if provided, will be used instead of creating new one)
            **kwargs: Passed to other base classes in MRO chain

        Raises:
            ValueError: If tenant_id is empty, None, or invalid format

        Examples:
            # Valid tenant IDs
            TenantAwareAgentMixin.__init__(self, "customer_a")
            TenantAwareAgentMixin.__init__(self, "acme:production")

            # Invalid - will raise ValueError
            TenantAwareAgentMixin.__init__(self, "")      # Empty
            TenantAwareAgentMixin.__init__(self, None)    # None
            TenantAwareAgentMixin.__init__(self, "  ")    # Whitespace only
        """
        # Validate tenant_id (fail fast)
        if not tenant_id:
            raise ValueError(
                "tenant_id is required - no default tenant. "
                "Agents must be explicitly initialized with a valid tenant identifier."
            )

        # Strip whitespace and validate again
        tenant_id = tenant_id.strip()
        if not tenant_id:
            raise ValueError(
                "tenant_id cannot be empty or whitespace only. "
                "Provide a valid tenant identifier (e.g., 'customer_a', 'acme:production')."
            )

        # Store tenant_id
        self.tenant_id = tenant_id

        # Initialize or get config manager
        # Use provided config_manager if available (for dependency injection)
        # Otherwise create a new one
        if config_manager is not None:
            self.config_manager = config_manager
        else:
            from cogniverse_foundation.config.utils import create_default_config_manager
            self.config_manager = create_default_config_manager()

        # Store or load configuration
        self.config = config
        if config is None:
            try:
                self.config = get_config(tenant_id=tenant_id, config_manager=self.config_manager)
            except Exception as e:
                logger.warning(f"Failed to load system config for tenant {tenant_id}: {e}")
                self.config = None

        # Initialize tenant-aware flag
        self._tenant_initialized = True

        logger.debug(f"Tenant context initialized: {tenant_id}")

        # Call super for MRO chain (if needed)
        # This allows other mixins to receive **kwargs
        # However, skip if the next class in MRO requires positional arguments
        # (like DSPyA2AAgentBase) to avoid TypeError
        if hasattr(super(), '__init__'):
            try:
                super().__init__(**kwargs)
            except TypeError as e:
                # Silently skip if the next class requires positional arguments
                # This happens with DSPyA2AAgentBase which requires agent_name, etc.
                # The child class will initialize it explicitly
                if "missing" in str(e) and "required positional argument" in str(e):
                    pass
                else:
                    raise

    def get_tenant_context(self) -> Dict[str, Any]:
        """
        Get tenant context for operations.

        Returns a dictionary with tenant information useful for:
        - Logging and debugging
        - Telemetry span attributes
        - Database query filtering
        - Cache key prefixes

        Returns:
            Dictionary with tenant context information

        Example:
            context = agent.get_tenant_context()
            # {
            #     "tenant_id": "customer_a",
            #     "agent_type": "RoutingAgent"
            # }
        """
        context = {
            "tenant_id": self.tenant_id,
        }

        # Add environment if available from config
        if self.config:
            # Try different ways config might expose environment
            if hasattr(self.config, 'environment'):
                context["environment"] = self.config.environment
            elif hasattr(self.config, 'get') and callable(self.config.get):
                env = self.config.get('environment')
                if env:
                    context["environment"] = env

        # Add agent type if available
        if hasattr(self, '__class__'):
            context["agent_type"] = self.__class__.__name__

        # Add agent name if available (from DSPyA2AAgentBase)
        if hasattr(self, 'agent_name'):
            context["agent_name"] = self.agent_name

        return context

    def validate_tenant_access(self, resource_tenant_id: str) -> bool:
        """
        Validate that this agent can access a resource owned by a tenant.

        Used for:
        - Cross-tenant data access checks
        - Security validation
        - Resource authorization

        Args:
            resource_tenant_id: Tenant ID that owns the resource

        Returns:
            True if agent's tenant matches resource tenant, False otherwise

        Example:
            # Agent initialized with tenant_id="customer_a"
            agent.validate_tenant_access("customer_a")  # True
            agent.validate_tenant_access("customer_b")  # False
        """
        if not resource_tenant_id:
            logger.warning(
                f"Attempted to validate access to resource with no tenant_id "
                f"(agent tenant: {self.tenant_id})"
            )
            return False

        return self.tenant_id == resource_tenant_id

    def get_tenant_scoped_key(self, key: str) -> str:
        """
        Generate a tenant-scoped key for caching, storage, etc.

        Args:
            key: Base key to scope

        Returns:
            Tenant-prefixed key

        Example:
            # Agent with tenant_id="customer_a"
            agent.get_tenant_scoped_key("embeddings/video_123")
            # Returns: "customer_a:embeddings/video_123"
        """
        return f"{self.tenant_id}:{key}"

    def is_tenant_initialized(self) -> bool:
        """
        Check if tenant context is properly initialized.

        Returns:
            True if tenant_id is set and validated
        """
        return (
            hasattr(self, '_tenant_initialized')
            and self._tenant_initialized
            and bool(self.tenant_id)
        )

    def get_tenant_id(self) -> str:
        """
        Get the tenant ID for this agent.

        Returns:
            Tenant identifier

        Note:
            This method exists for API completeness, but direct access
            to self.tenant_id is preferred for simplicity.
        """
        return self.tenant_id

    def log_tenant_operation(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ):
        """
        Log an operation with tenant context.

        Args:
            operation: Operation name (e.g., "search", "ingest", "route")
            details: Optional operation details
            level: Log level (debug, info, warning, error)

        Example:
            agent.log_tenant_operation(
                "search_completed",
                {"query": "machine learning", "results": 10}
            )
            # Logs: [customer_a] [RoutingAgent] search_completed: {'query': 'machine learning', 'results': 10}
        """
        log_func = getattr(logger, level, logger.info)

        agent_info = f"[{self.tenant_id}]"
        if hasattr(self, '__class__'):
            agent_info += f" [{self.__class__.__name__}]"

        message = f"{agent_info} {operation}"
        if details:
            message += f": {details}"

        log_func(message)

    def __repr__(self) -> str:
        """String representation including tenant context"""
        class_name = self.__class__.__name__ if hasattr(self, '__class__') else 'TenantAwareAgent'
        return f"{class_name}(tenant_id='{self.tenant_id}')"
