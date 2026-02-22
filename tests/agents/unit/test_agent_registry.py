"""
Unit tests for AgentRegistry
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from cogniverse_agents.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager


@pytest.mark.unit
class TestAgentEndpoint:
    """Test cases for AgentEndpoint class"""

    @pytest.mark.ci_fast
    def test_agent_endpoint_creation(self):
        """Test AgentEndpoint creation with basic parameters"""
        endpoint = AgentEndpoint(
            name="test_agent",
            url="http://localhost:8000",
            capabilities=["search", "analyze"],
        )

        assert endpoint.name == "test_agent"
        assert endpoint.url == "http://localhost:8000"
        assert endpoint.capabilities == ["search", "analyze"]
        assert endpoint.health_endpoint == "/health"
        assert endpoint.process_endpoint == "/process"
        assert endpoint.timeout == 30
        assert endpoint.health_status == "unknown"

    @pytest.mark.ci_fast
    def test_is_healthy(self):
        """Test health status checking"""
        endpoint = AgentEndpoint(
            name="test_agent", url="http://localhost:8000", capabilities=["search"]
        )

        # Initially unknown, not healthy
        assert not endpoint.is_healthy()

        # Set to healthy
        endpoint.health_status = "healthy"
        assert endpoint.is_healthy()

        # Set to unhealthy
        endpoint.health_status = "unhealthy"
        assert not endpoint.is_healthy()

    def test_needs_health_check(self):
        """Test health check timing logic"""
        endpoint = AgentEndpoint(
            name="test_agent",
            url="http://localhost:8000",
            capabilities=["search"],
            health_check_interval=60,
        )

        # No previous check, needs check
        assert endpoint.needs_health_check()

        # Recent check, doesn't need check
        endpoint.last_health_check = datetime.now()
        assert not endpoint.needs_health_check()

        # Old check, needs check
        endpoint.last_health_check = datetime.now() - timedelta(seconds=120)
        assert endpoint.needs_health_check()


@pytest.mark.unit
class TestAgentRegistry:
    """Test cases for AgentRegistry class"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with structured agents section"""
        return {
            "agents": {
                "orchestrator": {"url": "http://localhost:8013", "enabled": True},
                "entity_extraction": {"url": "http://localhost:8010", "enabled": True},
                "profile_selection": {"url": "http://localhost:8011", "enabled": True},
                "query_enhancement": {"url": "http://localhost:8012", "enabled": True},
                "search": {"url": "http://localhost:8002", "enabled": True},
                "summarizer": {"url": "http://localhost:8003", "enabled": True},
                "detailed_report": {"url": "http://localhost:8004", "enabled": True},
            },
        }

    @pytest.fixture
    def sample_agent(self):
        """Sample agent endpoint"""
        return AgentEndpoint(
            name="test_agent",
            url="http://localhost:8000",
            capabilities=["search", "analyze"],
        )

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.ci_fast
    def test_registry_initialization(self, mock_get_config, mock_config):
        """Test AgentRegistry initialization with structured agents config"""
        mock_get_config.return_value = mock_config

        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Should have initialized agents from structured config
        assert "orchestrator" in registry.agents
        assert "search" in registry.agents
        assert "summarizer" in registry.agents
        assert "detailed_report" in registry.agents

        # Check search agent
        search_agent = registry.get_agent("search")
        assert search_agent.url == "http://localhost:8002"
        assert "search" in search_agent.capabilities

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.ci_fast
    def test_registry_initialization_minimal_config(self, mock_get_config):
        """Test initialization with minimal structured config"""
        mock_get_config.return_value = {
            "agents": {
                "search": {"url": "http://localhost:8002", "enabled": True},
            }
        }

        registry = AgentRegistry(config_manager=create_default_config_manager())

        assert "search" in registry.agents
        # Default agents should also be registered
        assert "orchestrator" in registry.agents

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.ci_fast
    def test_register_agent(self, mock_get_config, sample_agent):
        """Test agent registration"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        result = registry.register_agent(sample_agent)

        assert result is True
        assert "test_agent" in registry.agents
        assert registry.get_agent("test_agent") == sample_agent

        # Check capability mapping
        assert "search" in registry.capabilities
        assert "test_agent" in registry.capabilities["search"]

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_register_agent_invalid(self, mock_get_config):
        """Test registration with invalid agent"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Agent with no name
        invalid_agent = AgentEndpoint(
            name="", url="http://localhost:8000", capabilities=["search"]
        )

        result = registry.register_agent(invalid_agent)
        assert result is False

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_unregister_agent(self, mock_get_config, sample_agent):
        """Test agent unregistration"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register first
        registry.register_agent(sample_agent)
        assert "test_agent" in registry.agents

        # Unregister
        result = registry.unregister_agent("test_agent")

        assert result is True
        assert "test_agent" not in registry.agents

        # Check capability mapping cleaned up
        if "search" in registry.capabilities:
            assert "test_agent" not in registry.capabilities["search"]

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_unregister_nonexistent_agent(self, mock_get_config):
        """Test unregistering non-existent agent"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        result = registry.unregister_agent("nonexistent")
        assert result is False

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.ci_fast
    def test_list_agents(self, mock_get_config, sample_agent):
        """Test listing agents"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        registry.register_agent(sample_agent)
        agent_names = registry.list_agents()

        assert "test_agent" in agent_names
        assert isinstance(agent_names, list)

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_find_agents_by_capability(self, mock_get_config):
        """Test finding agents by capability"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register agents with different capabilities
        agent1 = AgentEndpoint("agent1", "http://localhost:8001", ["search", "video"])
        agent2 = AgentEndpoint("agent2", "http://localhost:8002", ["search", "text"])
        agent3 = AgentEndpoint("agent3", "http://localhost:8003", ["analyze"])

        registry.register_agent(agent1)
        registry.register_agent(agent2)
        registry.register_agent(agent3)

        # Find by capability (3 = config-registered "search" + agent1 + agent2)
        search_agents = registry.find_agents_by_capability("search")
        assert len(search_agents) == 3
        assert any(agent.name == "search" for agent in search_agents)
        assert any(agent.name == "agent1" for agent in search_agents)
        assert any(agent.name == "agent2" for agent in search_agents)

        video_agents = registry.find_agents_by_capability("video")
        assert len(video_agents) == 1
        assert video_agents[0].name == "agent1"

        nonexistent_agents = registry.find_agents_by_capability("nonexistent")
        assert len(nonexistent_agents) == 0

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_get_healthy_agents(self, mock_get_config):
        """Test getting healthy agents"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register agents with different health statuses
        agent1 = AgentEndpoint("agent1", "http://localhost:8001", ["search"])
        agent1.health_status = "healthy"

        agent2 = AgentEndpoint("agent2", "http://localhost:8002", ["search"])
        agent2.health_status = "unhealthy"

        agent3 = AgentEndpoint("agent3", "http://localhost:8003", ["search"])
        agent3.health_status = "healthy"

        registry.register_agent(agent1)
        registry.register_agent(agent2)
        registry.register_agent(agent3)

        healthy_agents = registry.get_healthy_agents()

        assert len(healthy_agents) == 2
        healthy_names = [agent.name for agent in healthy_agents]
        assert "agent1" in healthy_names
        assert "agent3" in healthy_names
        assert "agent2" not in healthy_names

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_get_agents_for_workflow(self, mock_get_config):
        """Test getting agents for specific workflows"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register different types of agents
        video_agent = AgentEndpoint(
            "video_agent", "http://localhost:8001", ["video_search"]
        )
        text_agent = AgentEndpoint(
            "text_agent", "http://localhost:8002", ["text_search"]
        )
        summary_agent = AgentEndpoint(
            "summary_agent", "http://localhost:8003", ["summarization"]
        )
        analysis_agent = AgentEndpoint(
            "analysis_agent", "http://localhost:8004", ["detailed_analysis"]
        )

        registry.register_agent(video_agent)
        registry.register_agent(text_agent)
        registry.register_agent(summary_agent)
        registry.register_agent(analysis_agent)

        # Test raw_results workflow (only search agents)
        raw_agents = registry.get_agents_for_workflow("raw_results")
        raw_names = [agent.name for agent in raw_agents]
        assert "video_agent" in raw_names
        assert "text_agent" in raw_names
        assert "summary_agent" not in raw_names

        # Test summary workflow (search + summary agents)
        summary_agents = registry.get_agents_for_workflow("summary")
        summary_names = [agent.name for agent in summary_agents]
        assert "video_agent" in summary_names
        assert "text_agent" in summary_names
        assert "summary_agent" in summary_names

        # Test detailed_report workflow (search + analysis agents)
        report_agents = registry.get_agents_for_workflow("detailed_report")
        report_names = [agent.name for agent in report_agents]
        assert "video_agent" in report_names
        assert "text_agent" in report_names
        assert "analysis_agent" in report_names

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.asyncio
    async def test_health_check_agent_success(self, mock_get_config):
        """Test successful agent health check"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register agent
        agent = AgentEndpoint("test_agent", "http://localhost:8000", ["search"])
        registry.register_agent(agent)

        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        registry.http_client.get = AsyncMock(return_value=mock_response)

        result = await registry.health_check_agent("test_agent")

        assert result is True
        assert agent.health_status == "healthy"
        assert agent.last_health_check is not None

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.asyncio
    async def test_health_check_agent_failure(self, mock_get_config):
        """Test failed agent health check"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register agent
        agent = AgentEndpoint("test_agent", "http://localhost:8000", ["search"])
        registry.register_agent(agent)

        # Mock failed HTTP response
        mock_response = Mock()
        mock_response.status_code = 500
        registry.http_client.get = AsyncMock(return_value=mock_response)

        result = await registry.health_check_agent("test_agent")

        assert result is False
        assert agent.health_status == "unhealthy"
        assert agent.last_health_check is not None

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.asyncio
    async def test_health_check_agent_timeout(self, mock_get_config):
        """Test agent health check timeout"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register agent
        agent = AgentEndpoint("test_agent", "http://localhost:8000", ["search"])
        registry.register_agent(agent)

        # Mock timeout
        from httpx import TimeoutException

        registry.http_client.get = AsyncMock(side_effect=TimeoutException("Timeout"))

        result = await registry.health_check_agent("test_agent")

        assert result is False
        assert agent.health_status == "unreachable"
        assert agent.last_health_check is not None

    @patch("cogniverse_agents.agent_registry.get_config")
    @pytest.mark.asyncio
    async def test_health_check_nonexistent_agent(self, mock_get_config):
        """Test health check for non-existent agent"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        result = await registry.health_check_agent("nonexistent")
        assert result is False

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_get_load_balanced_agent(self, mock_get_config):
        """Test load-balanced agent selection"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Register agents with same capability
        agent1 = AgentEndpoint("agent1", "http://localhost:8001", ["search"])
        agent1.health_status = "healthy"

        agent2 = AgentEndpoint("agent2", "http://localhost:8002", ["search"])
        agent2.health_status = "unhealthy"

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        # Should return healthy agent
        selected = registry.get_load_balanced_agent("search")
        assert selected.name == "agent1"

        # Test with no healthy agents (should return any available)
        agent1.health_status = "unhealthy"
        selected = registry.get_load_balanced_agent("search")
        assert selected is not None

        # Test with no agents for capability
        selected = registry.get_load_balanced_agent("nonexistent")
        assert selected is None

    @patch("cogniverse_agents.agent_registry.get_config")
    def test_get_registry_stats(self, mock_get_config):
        """Test registry statistics"""
        mock_get_config.return_value = {
            "agents": {"search": {"url": "http://localhost:8002", "enabled": True}}
        }
        registry = AgentRegistry(config_manager=create_default_config_manager())

        # Clear auto-registered agents for clean test
        registry.agents.clear()
        registry.capabilities.clear()

        # Register agents
        agent1 = AgentEndpoint("agent1", "http://localhost:8001", ["search"])
        agent1.health_status = "healthy"

        agent2 = AgentEndpoint("agent2", "http://localhost:8002", ["search", "analyze"])
        agent2.health_status = "unhealthy"

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        stats = registry.get_registry_stats()

        assert stats["total_agents"] == 2
        assert stats["healthy_agents"] == 1
        assert stats["unhealthy_agents"] == 1

        # Check capability stats
        assert "search" in stats["capabilities"]
        search_stats = stats["capabilities"]["search"]
        assert search_stats["total_agents"] == 2
        assert search_stats["healthy_agents"] == 1

        # Check agent details
        assert "agent1" in stats["agent_details"]
        assert stats["agent_details"]["agent1"]["health_status"] == "healthy"


@pytest.mark.unit
class TestAgentRegistryStructuredConfig:
    """Test cases for structured agent configuration (Phase 5)"""

    @pytest.fixture
    def structured_config(self):
        """Structured config with agents dict"""
        return {
            "agents": {
                "orchestrator": {
                    "url": "http://localhost:8000",
                    "enabled": True,
                    "capabilities": ["orchestration", "workflow_planning"],
                },
                "entity_extraction": {
                    "url": "http://localhost:8010",
                    "enabled": True,
                    "capabilities": ["entity_extraction"],
                },
                "profile_selection": {
                    "url": "http://localhost:8011",
                    "enabled": False,  # Test disabled agent
                    "capabilities": ["profile_selection"],
                },
            }
        }

    @pytest.mark.ci_fast
    @patch("cogniverse_agents.agent_registry.get_config")
    def test_structured_config_registration(self, mock_get_config, structured_config):
        """Test agents are registered from structured config merged with defaults"""
        # Create mock that returns the structured_config dict when .get() is called
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: structured_config.get(
            key, default
        )
        mock_get_config.return_value = mock_config

        config_manager = create_default_config_manager()
        registry = AgentRegistry(tenant_id="test", config_manager=config_manager)

        # AgentRegistry merges user config with 7 defaults, registering all enabled agents
        # profile_selection is explicitly disabled in fixture, so 6 agents registered
        assert len(registry.agents) == 6
        assert "orchestrator" in registry.agents
        assert "entity_extraction" in registry.agents
        assert (
            "profile_selection" not in registry.agents
        )  # explicitly disabled in fixture

    @pytest.mark.ci_fast
    @patch("cogniverse_agents.agent_registry.get_config")
    def test_all_default_agents_registered(self, mock_get_config):
        """Test all 7 Phase 2-3 agents registered with defaults"""
        # Full agents config with all 7 agents enabled
        full_agents_config = MagicMock()
        full_agents_config.get.return_value = {
            "orchestrator": {"url": "http://localhost:8000", "enabled": True},
            "entity_extraction": {"url": "http://localhost:8010", "enabled": True},
            "profile_selection": {"url": "http://localhost:8011", "enabled": True},
            "query_enhancement": {"url": "http://localhost:8012", "enabled": True},
            "search": {"url": "http://localhost:8002", "enabled": True},
            "summarizer": {"url": "http://localhost:8003", "enabled": True},
            "detailed_report": {"url": "http://localhost:8004", "enabled": True},
        }

        mock_get_config.return_value = full_agents_config

        config_manager = create_default_config_manager()
        registry = AgentRegistry(tenant_id="test", config_manager=config_manager)

        # All 7 agents should be registered
        assert len(registry.agents) == 7
        assert "orchestrator" in registry.agents
        assert "entity_extraction" in registry.agents
        assert "profile_selection" in registry.agents
        assert "query_enhancement" in registry.agents
        assert "search" in registry.agents
        assert "summarizer" in registry.agents
        assert "detailed_report" in registry.agents

    @pytest.mark.ci_fast
    @patch("cogniverse_agents.agent_registry.get_config")
    def test_capability_based_discovery(self, mock_get_config, structured_config):
        """Test finding agents by capability"""
        # Create mock that returns the structured_config dict when .get() is called
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: structured_config.get(
            key, default
        )
        mock_get_config.return_value = mock_config

        config_manager = create_default_config_manager()
        registry = AgentRegistry(tenant_id="test", config_manager=config_manager)

        # Find by orchestration capability - use correct method name
        agents = registry.find_agents_by_capability("orchestration")
        assert len(agents) == 1
        assert agents[0].name == "orchestrator"

        # Find by entity_extraction capability
        agents = registry.find_agents_by_capability("entity_extraction")
        assert len(agents) == 1
        assert agents[0].name == "entity_extraction"

        # Nonexistent capability
        agents = registry.find_agents_by_capability("nonexistent")
        assert len(agents) == 0

    @pytest.mark.ci_fast
    @patch("cogniverse_agents.agent_registry.get_config")
    def test_missing_agents_section_raises_error(self, mock_get_config):
        """Test that config without 'agents' section raises ValueError"""
        mock_get_config.return_value = {
            "video_agent_url": "http://localhost:8002",
        }

        config_manager = create_default_config_manager()
        with pytest.raises(ValueError, match="Missing 'agents' section"):
            AgentRegistry(config_manager=config_manager)

    @pytest.mark.ci_fast
    @patch("cogniverse_agents.agent_registry.get_config")
    def test_custom_url_override(self, mock_get_config):
        """Test custom URL overrides default URL"""
        # Custom URL for orchestrator
        custom_config_data = {
            "agents": {
                "orchestrator": {
                    "url": "http://custom-host:9000",
                    "enabled": True,
                    "capabilities": ["orchestration"],
                }
            }
        }

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: custom_config_data.get(
            key, default
        )
        mock_get_config.return_value = mock_config

        config_manager = create_default_config_manager()
        registry = AgentRegistry(tenant_id="test", config_manager=config_manager)

        # Should use custom URL
        assert registry.agents["orchestrator"].url == "http://custom-host:9000"

    @pytest.mark.ci_fast
    @patch("cogniverse_agents.agent_registry.get_config")
    def test_disabled_agents_not_registered(self, mock_get_config, structured_config):
        """Test disabled agents are not registered"""
        # Create mock that returns the structured_config dict when .get() is called
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: structured_config.get(
            key, default
        )
        mock_get_config.return_value = mock_config

        config_manager = create_default_config_manager()
        registry = AgentRegistry(tenant_id="test", config_manager=config_manager)

        # profile_selection is disabled in fixture
        assert "profile_selection" not in registry.agents

        # AgentRegistry merges with 7 defaults, 1 disabled = 6 registered
        assert len(registry.agents) == 6
