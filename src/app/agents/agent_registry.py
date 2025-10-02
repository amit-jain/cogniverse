"""
Agent Registry for dynamic agent discovery and management.
Provides centralized registry for all available agents with health monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from src.common.agent_models import AgentEndpoint
from src.common.config import get_config

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

    def __init__(self):
        """Initialize agent registry"""
        self.config = get_config()
        self.agents: Dict[str, AgentEndpoint] = {}
        self.capabilities: Dict[str, List[str]] = {}  # capability -> agent names
        self.http_client = httpx.AsyncClient(timeout=10.0)

        # Initialize with system config agents
        self._initialize_from_config()

        logger.info("AgentRegistry initialized")

    def _initialize_from_config(self):
        """Initialize registry from system configuration"""
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

        logger.info(f"Initialized {len(self.agents)} agents from configuration")

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

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
