"""
Agent Registry for dynamic agent discovery and management.

Two layers make up the agents a registry serves. Configured agents are
registered from configuration in every process at startup, so every process
holds the same ones. Registrations made over HTTP live in a shared
:class:`AgentRegistryStore`, which can also hide a configured agent; every
process applies the store's current contents on :meth:`AgentRegistry.refresh`.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Protocol

import httpx

from cogniverse_core.common.agent_models import (
    DEFAULT_AGENT_CALL_TIMEOUT_SECONDS,
    AgentEndpoint,
)

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager

logger = logging.getLogger(__name__)


class AgentRegistryUnavailableError(RuntimeError):
    """Raised when the shared registration store cannot complete an operation."""


@dataclass(frozen=True)
class RegistryVersion:
    """Position of a store's contents.

    ``epoch`` names one lifetime of the store's data (a wiped store starts a
    new one); ``counter`` increases with every change inside an epoch.
    """

    epoch: str
    counter: int


@dataclass(frozen=True)
class RegistrySnapshot:
    """The store's registrations and removals at one version."""

    version: RegistryVersion
    registered: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    removed: FrozenSet[str] = frozenset()


class AgentRegistryStore(Protocol):
    """Registrations shared by every process that serves one registry.

    Every method raises :class:`AgentRegistryUnavailableError` when the store
    cannot answer.
    """

    async def version(self) -> RegistryVersion: ...

    async def snapshot(self) -> RegistrySnapshot: ...

    async def register(self, name: str, data: Dict[str, Any]) -> None:
        """Store ``data`` as the registration of ``name``, replacing any
        registration or removal recorded for it."""
        ...

    async def unregister(self, name: str, *, configured: bool) -> bool:
        """Remove ``name``; True when it was served before the call.

        ``configured`` says whether ``name`` is a configured agent: one is
        hidden by a recorded removal, any other loses its registration.
        """
        ...


def endpoint_data(agent: AgentEndpoint) -> Dict[str, Any]:
    """The registration fields of ``agent``, as a store keeps them."""
    return {
        "name": agent.name,
        "url": agent.url,
        "capabilities": list(agent.capabilities),
        "streams_answer_tokens": agent.streams_answer_tokens,
        "health_endpoint": agent.health_endpoint,
        "process_endpoint": agent.process_endpoint,
        "timeout": agent.timeout,
    }


def endpoint_from_data(data: Dict[str, Any]) -> AgentEndpoint:
    """Build an endpoint from registration fields, defaults filled in."""
    return AgentEndpoint(
        name=data.get("name"),
        url=data.get("url"),
        capabilities=list(data.get("capabilities", [])),
        streams_answer_tokens=data.get("streams_answer_tokens", False),
        health_endpoint=data.get("health_endpoint", "/health"),
        process_endpoint=data.get("process_endpoint", "/tasks/send"),
        timeout=data.get("timeout", DEFAULT_AGENT_CALL_TIMEOUT_SECONDS),
    )


class AgentRegistry:
    """
    Registry for managing available agents with health monitoring and load balancing.
    Uses dependency injection for ConfigManager instead of singleton pattern.
    """

    def __init__(
        self,
        tenant_id: str,
        config_manager: "ConfigManager" = None,
        store: Optional[AgentRegistryStore] = None,
    ):
        """Initialize agent registry with dependency injection

        Args:
            tenant_id: Tenant identifier for config isolation (required)
            config_manager: ConfigManager instance (required for dependency injection)
            store: Shared registration store. Without one, the registry serves
                its configured agents only and refuses registrations.
        """
        from cogniverse_core.common.tenant_utils import require_tenant_id

        if config_manager is None:
            raise ValueError(
                "config_manager is required for AgentRegistry. "
                "Dependency injection is mandatory - pass ConfigManager() explicitly."
            )

        # config_manager is required (DI hygiene — see the None check above) but
        # this registry resolves nothing from it: agents self-register over HTTP.
        self.tenant_id = require_tenant_id(tenant_id, source="AgentRegistry")
        # The served view: configured agents overlaid with the store's
        # registrations and removals. Rebuilt whole and swapped on change.
        self.agents: Dict[str, AgentEndpoint] = {}
        self.capabilities: Dict[str, List[str]] = {}  # capability -> agent names
        self._configured: Dict[str, AgentEndpoint] = {}
        self._registered: Dict[str, AgentEndpoint] = {}
        self._removed: FrozenSet[str] = frozenset()
        self._store = store
        self._store_version: Optional[RegistryVersion] = None
        # Constructed lazily on first use — a registry used only for local
        # agent lookup never opens (or has to close) an httpx client.
        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_client_lock = threading.Lock()

        # Initialize with system config agents
        self._initialize_from_config()

        logger.info(f"AgentRegistry initialized for tenant: {tenant_id}")

    def set_store(self, store: AgentRegistryStore) -> None:
        """Attach the shared registration store; the next refresh applies it."""
        self._store = store
        self._store_version = None

    async def refresh(self) -> None:
        """Apply the store's current registrations and removals.

        A request path calls this before it reads the registry, so a change
        made by any process is served by the next request on every process.
        Raises :class:`AgentRegistryUnavailableError` when the store cannot
        answer; the served view is then left as it was.
        """
        if self._store is None:
            return
        if await self._store.version() == self._store_version:
            return
        self._apply(await self._store.snapshot())

    async def add_registration(self, agent: AgentEndpoint) -> None:
        """Register ``agent`` for every process sharing the store."""
        if not agent.name or not agent.url:
            raise ValueError("Agent must have name and URL")
        await self._require_store().register(agent.name, endpoint_data(agent))
        await self.refresh()
        logger.info(f"Registered agent: {agent.name} at {agent.url}")

    async def remove_registration(self, agent_name: str) -> bool:
        """Stop serving ``agent_name`` in every process sharing the store.

        Returns False when it was not served. A configured agent stays
        removed until it is registered again.
        """
        store = self._require_store()
        removed = await store.unregister(
            agent_name, configured=agent_name in self._configured
        )
        await self.refresh()
        if removed:
            logger.info(f"Unregistered agent: {agent_name}")
        return removed

    def _require_store(self) -> AgentRegistryStore:
        if self._store is None:
            raise AgentRegistryUnavailableError(
                "AgentRegistry has no shared store; registrations need one"
            )
        return self._store

    def _apply(self, snapshot: RegistrySnapshot) -> None:
        """Serve ``snapshot`` unless a newer one of its epoch is served."""
        current = self._store_version
        if (
            current is not None
            and current.epoch == snapshot.version.epoch
            and current.counter >= snapshot.version.counter
        ):
            return
        registered: Dict[str, AgentEndpoint] = {}
        for name, data in sorted(snapshot.registered.items()):
            kept = self._registered.get(name)
            # An unchanged registration keeps its endpoint and with it the
            # health this process observed.
            registered[name] = (
                kept
                if kept is not None and endpoint_data(kept) == data
                else endpoint_from_data(data)
            )
        self._registered = registered
        self._removed = snapshot.removed
        self._store_version = snapshot.version
        self._rebuild()

    def _rebuild(self) -> None:
        agents = dict(self._configured)
        for name in self._removed:
            agents.pop(name, None)
        agents.update(self._registered)
        capabilities: Dict[str, List[str]] = {}
        for name, agent in agents.items():
            for capability in agent.capabilities:
                names = capabilities.setdefault(capability, [])
                if name not in names:
                    names.append(name)
        self.agents = agents
        self.capabilities = capabilities

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazily-created async client for remote agent health/dispatch calls."""
        if self._http_client is None:
            with self._http_client_lock:
                if self._http_client is None:
                    self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    def _initialize_from_config(self):
        """Initialize registry from system configuration

        Note: Registry now relies on agent self-registration via HTTP.
        This method is kept for emergency fallback only - agents should
        register themselves using the Curated Registry pattern.
        """
        logger.info("AgentRegistry initialized - waiting for agent self-registration")

    def register_agent(self, agent: AgentEndpoint) -> bool:
        """
        Register a configured agent in this process's registry.

        Every process registers the same configured agents at startup; a
        registration every process must serve goes through
        :meth:`add_registration`.

        Args:
            agent: Agent endpoint to register

        Returns:
            True if successfully registered
        """
        try:
            # Validate agent configuration
            if not agent.name or not agent.url:
                raise ValueError("Agent must have name and URL")

            self._configured[agent.name] = agent
            self._rebuild()

            logger.info(f"Registered agent: {agent.name} at {agent.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent.name}: {e}")
            return False

    def get_agent(self, agent_name: str) -> Optional[AgentEndpoint]:
        """
        Get agent endpoint by name.

        Args:
            agent_name: Name of agent

        Returns:
            Agent endpoint if found, None otherwise
        """
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """
        List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def find_agents_by_capability(self, capability: str) -> List[AgentEndpoint]:
        """
        Find agents that support a specific capability.

        Args:
            capability: Capability to search for

        Returns:
            List of agent endpoints that support the capability
        """
        agent_names = self.capabilities.get(capability, [])
        return [self.agents[name] for name in agent_names if name in self.agents]

    def get_healthy_agents(self) -> List[AgentEndpoint]:
        """
        Get all healthy agents.

        Returns:
            List of healthy agent endpoints
        """
        return [agent for agent in self.agents.values() if agent.is_healthy()]

    def get_agents_for_workflow(self, workflow_type: str) -> List[AgentEndpoint]:
        """
        Get agents needed for a specific workflow type.

        Args:
            workflow_type: Type of workflow (raw_results, summary, detailed_report)

        Returns:
            List of agent endpoints needed for the workflow
        """
        required_agents = []

        # All workflows need search agents
        search_agents = self.find_agents_by_capability("video_search")
        search_agents.extend(self.find_agents_by_capability("text_search"))
        required_agents.extend(search_agents)

        # Additional agents based on workflow type
        if workflow_type == "summary":
            summary_agents = self.find_agents_by_capability("summarization")
            required_agents.extend(summary_agents)
        elif workflow_type == "detailed_report":
            report_agents = self.find_agents_by_capability("detailed_analysis")
            required_agents.extend(report_agents)

        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in required_agents:
            if agent.name not in seen:
                unique_agents.append(agent)
                seen.add(agent.name)

        return unique_agents

    async def health_check_agent(self, agent_name: str) -> bool:
        """
        Perform health check on a specific agent.

        Args:
            agent_name: Name of agent to check

        Returns:
            True if agent is healthy
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return False

        try:
            health_url = f"{agent.url}{agent.health_endpoint}"
            response = await self.http_client.get(health_url, timeout=5.0)

            is_healthy = response.status_code == 200
            agent.health_status = "healthy" if is_healthy else "unhealthy"
            agent.last_health_check = datetime.now(timezone.utc)

            if is_healthy:
                logger.debug(f"Agent {agent_name} is healthy")
            else:
                logger.warning(
                    f"Agent {agent_name} health check failed: status {response.status_code}"
                )

            return is_healthy

        except httpx.TimeoutException:
            agent.health_status = "unreachable"
            agent.last_health_check = datetime.now(timezone.utc)
            logger.warning(f"Agent {agent_name} health check timed out")
            return False
        except Exception as e:
            agent.health_status = "unreachable"
            agent.last_health_check = datetime.now(timezone.utc)
            logger.warning(f"Agent {agent_name} health check failed: {e}")
            return False

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all registered agents.

        Returns:
            Dictionary mapping agent names to health status
        """
        health_results = {}

        # Check agents that need health checks
        tasks = []
        for agent_name, agent in self.agents.items():
            if agent.needs_health_check():
                tasks.append(self.health_check_agent(agent_name))
            else:
                # Use cached health status
                health_results[agent_name] = agent.is_healthy()

        # Execute health checks concurrently
        if tasks:
            agent_names = [
                name
                for name, agent in self.agents.items()
                if agent.needs_health_check()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent_name, result in zip(agent_names, results):
                if isinstance(result, Exception):
                    health_results[agent_name] = False
                else:
                    health_results[agent_name] = result

        return health_results

    def get_load_balanced_agent(self, capability: str) -> Optional[AgentEndpoint]:
        """
        Get a load-balanced agent for a specific capability.
        Currently implements simple round-robin, can be enhanced with actual load metrics.

        Args:
            capability: Required capability

        Returns:
            Agent endpoint or None if no healthy agents available
        """
        candidates = self.find_agents_by_capability(capability)
        healthy_candidates = [agent for agent in candidates if agent.is_healthy()]

        if not healthy_candidates:
            # Fallback to any agent with the capability
            return candidates[0] if candidates else None

        # Simple round-robin (could be enhanced with actual load metrics)
        # For now, just return the first healthy agent
        return healthy_candidates[0]

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        healthy_count = len(self.get_healthy_agents())
        total_count = len(self.agents)

        capability_stats = {}
        for capability, agent_names in self.capabilities.items():
            healthy_agents = [
                name for name in agent_names if self.agents[name].is_healthy()
            ]
            capability_stats[capability] = {
                "total_agents": len(agent_names),
                "healthy_agents": len(healthy_agents),
                "agents": agent_names,
            }

        return {
            "total_agents": total_count,
            "healthy_agents": healthy_count,
            "unhealthy_agents": total_count - healthy_count,
            "capabilities": capability_stats,
            "agent_details": {
                name: {
                    "url": agent.url,
                    "health_status": agent.health_status,
                    "last_health_check": (
                        agent.last_health_check.isoformat()
                        if agent.last_health_check
                        else None
                    ),
                    "capabilities": agent.capabilities,
                }
                for name, agent in self.agents.items()
            },
        }

    async def discover_agent_by_url(self, agent_url: str) -> Optional[AgentEndpoint]:
        """
        Discover agent by fetching its agent card via well-known URI.

        Args:
            agent_url: Base URL of the agent

        Returns:
            AgentEndpoint if successful, None otherwise

        Raises:
            Exception if agent card cannot be retrieved
        """
        response = await self.http_client.get(
            f"{agent_url}/.well-known/agent-card.json"
        )
        response.raise_for_status()
        card_data = response.json()

        # Convert agent card to AgentEndpoint
        agent_endpoint = AgentEndpoint(
            name=card_data.get("name", "unknown"),
            url=card_data.get("url", agent_url),
            capabilities=card_data.get("capabilities", []),
            health_endpoint="/health",
            process_endpoint=card_data.get("process_endpoint", "/tasks/send"),
            timeout=DEFAULT_AGENT_CALL_TIMEOUT_SECONDS,
        )

        logger.info(f"Discovered agent: {agent_endpoint.name} at {agent_url}")
        return agent_endpoint

    async def auto_register_from_urls(self, agent_urls: List[str]) -> Dict[str, bool]:
        """
        Discover and auto-register agents from a list of URLs.

        Args:
            agent_urls: List of agent base URLs

        Returns:
            Dictionary mapping agent URLs to registration success status
        """
        results = {}

        for url in agent_urls:
            try:
                agent_endpoint = await self.discover_agent_by_url(url)
                success = self.register_agent(agent_endpoint)
                results[url] = success
            except Exception as e:
                logger.error(f"Failed to auto-register agent from {url}: {e}")
                results[url] = False

        logger.info(f"Auto-registered {sum(results.values())}/{len(agent_urls)} agents")
        return results

    def register_agent_from_data(self, registration_data: Dict[str, Any]) -> bool:
        """
        Register a configured agent from registration data payload.

        Args:
            registration_data: Agent registration data containing name, url, capabilities

        Returns:
            True if successfully registered
        """
        try:
            return self.register_agent(endpoint_from_data(registration_data))
        except Exception as e:
            logger.error(f"Failed to register agent from data: {e}")
            return False

    async def close(self):
        """Close the HTTP client if one was ever created."""
        with self._http_client_lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            await client.aclose()
