"""
Agent Registry for dynamic agent discovery and management.
Provides centralized registry for all available agents with health monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_foundation.config.utils import get_config

from cogniverse_agents.tools.a2a_utils import A2AClient

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Agent capability definition"""

    name: str
    description: str
    input_types: List[str]
    output_types: List[str]


class AgentRegistry:
    """
    Registry for managing available agents with health monitoring and load balancing.
    """

    def __init__(self, tenant_id: str = "default", config_manager: "ConfigManager" = None):
        """Initialize agent registry

        Args:
            tenant_id: Tenant identifier for multi-tenancy support
            config_manager: ConfigManager instance for configuration access
        """
        if config_manager is None:
            raise ValueError("config_manager is required for AgentRegistry")

        self.tenant_id = tenant_id
        self.config_manager = config_manager
        self.config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        self.agents: Dict[str, AgentEndpoint] = {}
        self.capabilities: Dict[str, List[str]] = {}  # capability -> agent names
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self.a2a_client = A2AClient(timeout=10.0)

        # Initialize with system config agents
        self._initialize_from_config()

        logger.info("AgentRegistry initialized")

    def _initialize_from_config(self):
        """Initialize registry from system configuration"""
        # Try structured agents config first
        agents_config = self.config.get("agents", {})

        if agents_config:
            # Use structured config with explicit agent definitions
            self._register_from_structured_config(agents_config)
        else:
            # Fall back to legacy individual URL-based config for backward compatibility
            self._register_from_legacy_config()

        logger.info(f"Initialized {len(self.agents)} agents from configuration")

    def _register_from_structured_config(self, agents_config: Dict[str, Dict[str, Any]]):
        """Register agents from structured config with default ports and capabilities"""
        # Default agent configurations matching Phase 5 requirements
        default_agents = {
            "orchestrator": {
                "url": "http://localhost:8000",
                "enabled": True,
                "capabilities": ["orchestration", "workflow_planning", "agent_coordination"]
            },
            "entity_extraction": {
                "url": "http://localhost:8010",
                "enabled": True,
                "capabilities": ["entity_extraction", "relationship_extraction", "semantic_analysis"]
            },
            "profile_selection": {
                "url": "http://localhost:8011",
                "enabled": True,
                "capabilities": ["profile_selection", "ensemble_composition", "llm_reasoning"]
            },
            "query_enhancement": {
                "url": "http://localhost:8012",
                "enabled": True,
                "capabilities": ["query_enhancement", "query_expansion", "synonym_generation"]
            },
            "search": {
                "url": "http://localhost:8002",
                "enabled": True,
                "capabilities": ["search", "ensemble_search", "multimodal_search"]
            },
            "summarizer": {
                "url": "http://localhost:8003",
                "enabled": True,
                "capabilities": ["summarization", "content_condensation"]
            },
            "detailed_report": {
                "url": "http://localhost:8004",
                "enabled": True,
                "capabilities": ["detailed_reporting", "comprehensive_analysis"]
            },
        }

        # Merge user config with defaults
        for agent_name, default_config in default_agents.items():
            # Use user config if provided, otherwise use defaults
            agent_config = agents_config.get(agent_name, default_config)

            # Check if agent is enabled (default to True if not specified)
            if not agent_config.get("enabled", True):
                logger.debug(f"Agent {agent_name} is disabled in config, skipping registration")
                continue

            # Register agent
            self.register_agent(
                AgentEndpoint(
                    name=agent_name,
                    url=agent_config.get("url", default_config["url"]),
                    capabilities=agent_config.get("capabilities", default_config["capabilities"]),
                )
            )

    def _register_from_legacy_config(self):
        """Register agents from legacy individual URL config for backward compatibility"""
        # Video search agent
        video_agent_url = self.config.get("video_agent_url")
        if video_agent_url:
            self.register_agent(
                AgentEndpoint(
                    name="video_search",
                    url=video_agent_url,
                    capabilities=[
                        "video_search",
                        "multimodal_search",
                        "temporal_search",
                    ],
                )
            )

        # Text search agent
        text_agent_url = self.config.get("text_agent_url")
        if text_agent_url:
            self.register_agent(
                AgentEndpoint(
                    name="text_search",
                    url=text_agent_url,
                    capabilities=["text_search", "document_search", "hybrid_search"],
                )
            )

        # Routing agent (self-registration)
        routing_agent_port = self.config.get("routing_agent_port", 8001)
        self.register_agent(
            AgentEndpoint(
                name="routing_agent",
                url=f"http://localhost:{routing_agent_port}",
                capabilities=[
                    "query_routing",
                    "workflow_coordination",
                    "agent_orchestration",
                ],
            )
        )

    def register_agent(self, agent: AgentEndpoint) -> bool:
        """
        Register an agent in the registry.

        Args:
            agent: Agent endpoint to register

        Returns:
            True if successfully registered
        """
        try:
            # Validate agent configuration
            if not agent.name or not agent.url:
                raise ValueError("Agent must have name and URL")

            # Register agent
            self.agents[agent.name] = agent

            # Update capability mapping
            for capability in agent.capabilities:
                if capability not in self.capabilities:
                    self.capabilities[capability] = []
                if agent.name not in self.capabilities[capability]:
                    self.capabilities[capability].append(agent.name)

            logger.info(f"Registered agent: {agent.name} at {agent.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent.name}: {e}")
            return False

    def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister an agent from the registry.

        Args:
            agent_name: Name of agent to unregister

        Returns:
            True if successfully unregistered
        """
        if agent_name not in self.agents:
            return False

        agent = self.agents[agent_name]

        # Remove from capability mapping
        for capability in agent.capabilities:
            if capability in self.capabilities:
                if agent_name in self.capabilities[capability]:
                    self.capabilities[capability].remove(agent_name)
                if not self.capabilities[capability]:
                    del self.capabilities[capability]

        # Remove agent
        del self.agents[agent_name]

        logger.info(f"Unregistered agent: {agent_name}")
        return True

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
            agent.last_health_check = datetime.now()

            if is_healthy:
                logger.debug(f"Agent {agent_name} is healthy")
            else:
                logger.warning(
                    f"Agent {agent_name} health check failed: status {response.status_code}"
                )

            return is_healthy

        except httpx.TimeoutException:
            agent.health_status = "unreachable"
            agent.last_health_check = datetime.now()
            logger.warning(f"Agent {agent_name} health check timed out")
            return False
        except Exception as e:
            agent.health_status = "unreachable"
            agent.last_health_check = datetime.now()
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
        card_data = await self.a2a_client.get_agent_card(agent_url)

        # Convert agent card to AgentEndpoint
        agent_endpoint = AgentEndpoint(
            name=card_data.get("name", "unknown"),
            url=card_data.get("url", agent_url),
            capabilities=card_data.get("capabilities", []),
            health_endpoint="/health",
            process_endpoint=card_data.get("process_endpoint", "/tasks/send"),
            timeout=30,
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
        Register agent from registration data payload.

        Args:
            registration_data: Agent registration data containing name, url, capabilities

        Returns:
            True if successfully registered
        """
        try:
            agent_endpoint = AgentEndpoint(
                name=registration_data.get("name"),
                url=registration_data.get("url"),
                capabilities=registration_data.get("capabilities", []),
                health_endpoint=registration_data.get("health_endpoint", "/health"),
                process_endpoint=registration_data.get(
                    "process_endpoint", "/tasks/send"
                ),
                timeout=registration_data.get("timeout", 30),
            )

            return self.register_agent(agent_endpoint)

        except Exception as e:
            logger.error(f"Failed to register agent from data: {e}")
            return False

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
