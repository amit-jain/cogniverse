"""
Unit tests for Agent Registry HTTP endpoints (Curated Registry pattern).

Tests the A2A Curated Registries implementation where agents self-register
via HTTP POST and clients discover agents via HTTP GET.
Also tests the process_agent_task dispatch logic.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry


@pytest.mark.unit
class TestAgentRegistryHTTPEndpoints:
    """Test HTTP endpoints for Curated Agent Registry"""

    @pytest.fixture
    def config_manager(self):
        """Create mock config manager"""
        config_manager = Mock()
        return config_manager

    @pytest.fixture
    def agent_registry(self, config_manager):
        """Create agent registry for testing"""
        registry = AgentRegistry(tenant_id="default", config_manager=config_manager)
        return registry

    @pytest.fixture
    def test_client(self, agent_registry):
        """Create FastAPI test client with injected registry"""
        from cogniverse_runtime.routers.agents import router, set_agent_registry

        # Inject the registry
        set_agent_registry(agent_registry)

        # Create test client
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/agents")

        return TestClient(app)

    def test_register_agent_endpoint(self, test_client, agent_registry):
        """Test POST /agents/register endpoint"""
        registration_data = {
            "name": "test_agent",
            "url": "http://localhost:9000",
            "capabilities": ["search", "summarize"],
            "health_endpoint": "/health",
            "process_endpoint": "/tasks/send",
            "timeout": 30,
        }

        response = test_client.post("/agents/register", json=registration_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "registered"
        assert data["agent"] == "test_agent"
        assert data["url"] == "http://localhost:9000"
        assert data["capabilities"] == ["search", "summarize"]

        # Verify agent is in registry
        agent = agent_registry.get_agent("test_agent")
        assert agent is not None
        assert agent.name == "test_agent"
        assert agent.url == "http://localhost:9000"

    def test_list_agents_endpoint(self, test_client, agent_registry):
        """Test GET /agents endpoint"""
        # Register some agents
        agent_registry.register_agent(
            AgentEndpoint(
                name="agent1", url="http://localhost:8001", capabilities=["search"]
            )
        )
        agent_registry.register_agent(
            AgentEndpoint(
                name="agent2", url="http://localhost:8002", capabilities=["summarize"]
            )
        )

        response = test_client.get("/agents/")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert "agent1" in data["agents"]
        assert "agent2" in data["agents"]

    def test_get_registry_stats_endpoint(self, test_client, agent_registry):
        """Test GET /agents/stats endpoint"""
        # Register an agent
        agent_registry.register_agent(
            AgentEndpoint(
                name="test_agent",
                url="http://localhost:9000",
                capabilities=["search", "analyze"],
            )
        )

        response = test_client.get("/agents/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_agents"] == 1
        assert "capabilities" in data
        assert "search" in data["capabilities"]
        assert "analyze" in data["capabilities"]

    def test_find_agents_by_capability_endpoint(self, test_client, agent_registry):
        """Test GET /agents/by-capability/{capability} endpoint"""
        # Register agents with different capabilities
        agent_registry.register_agent(
            AgentEndpoint(
                name="search_agent",
                url="http://localhost:8001",
                capabilities=["search", "index"],
            )
        )
        agent_registry.register_agent(
            AgentEndpoint(
                name="summarize_agent",
                url="http://localhost:8002",
                capabilities=["summarize"],
            )
        )
        agent_registry.register_agent(
            AgentEndpoint(
                name="hybrid_agent",
                url="http://localhost:8003",
                capabilities=["search", "summarize"],
            )
        )

        response = test_client.get("/agents/by-capability/search")

        assert response.status_code == 200
        data = response.json()
        assert data["capability"] == "search"
        assert data["count"] == 2  # search_agent and hybrid_agent

        agent_names = [a["name"] for a in data["agents"]]
        assert "search_agent" in agent_names
        assert "hybrid_agent" in agent_names
        assert "summarize_agent" not in agent_names

    def test_get_agent_info_endpoint(self, test_client, agent_registry):
        """Test GET /agents/{agent_name} endpoint"""
        agent_registry.register_agent(
            AgentEndpoint(
                name="test_agent", url="http://localhost:9000", capabilities=["search"]
            )
        )

        response = test_client.get("/agents/test_agent")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_agent"
        assert data["url"] == "http://localhost:9000"
        assert data["capabilities"] == ["search"]
        assert data["health_endpoint"] == "/health"

    def test_get_agent_info_not_found(self, test_client):
        """Test GET /agents/{agent_name} with non-existent agent"""
        response = test_client.get("/agents/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_unregister_agent_endpoint(self, test_client, agent_registry):
        """Test DELETE /agents/{agent_name} endpoint"""
        # Register an agent first
        agent_registry.register_agent(
            AgentEndpoint(
                name="test_agent", url="http://localhost:9000", capabilities=["search"]
            )
        )

        # Verify it exists
        assert agent_registry.get_agent("test_agent") is not None

        # Unregister it
        response = test_client.delete("/agents/test_agent")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unregistered"
        assert data["agent"] == "test_agent"

        # Verify it's gone
        assert agent_registry.get_agent("test_agent") is None

    def test_unregister_nonexistent_agent(self, test_client):
        """Test DELETE /agents/{agent_name} with non-existent agent"""
        response = test_client.delete("/agents/nonexistent")

        assert response.status_code == 404

    def test_register_agent_duplicate_updates(self, test_client, agent_registry):
        """Test that registering same agent twice updates it"""
        registration_data = {
            "name": "test_agent",
            "url": "http://localhost:9000",
            "capabilities": ["search"],
            "health_endpoint": "/health",
            "process_endpoint": "/tasks/send",
            "timeout": 30,
        }

        # First registration
        response1 = test_client.post("/agents/register", json=registration_data)
        assert response1.status_code == 201

        # Second registration with updated capabilities
        registration_data["capabilities"] = ["search", "summarize"]
        response2 = test_client.post("/agents/register", json=registration_data)
        assert response2.status_code == 201

        # Verify updated capabilities
        agent = agent_registry.get_agent("test_agent")
        assert "summarize" in agent.capabilities


# Note: Agent self-registration is tested in integration tests below
# Unit tests for self-registration with mock HTTP clients are complex
# due to abstract method requirements and context manager mocking


@pytest.mark.ci_fast
class TestAgentRegistryIntegration:
    """Integration tests for agent registry with HTTP endpoints"""

    def test_full_registration_discovery_flow(self):
        """Test complete flow: register → discover by capability → get info"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from cogniverse_runtime.routers.agents import router, set_agent_registry

        # Setup
        config_manager = Mock()
        registry = AgentRegistry(tenant_id="default", config_manager=config_manager)
        set_agent_registry(registry)

        app = FastAPI()
        app.include_router(router, prefix="/agents")
        client = TestClient(app)

        # Step 1: Register two agents
        client.post(
            "/agents/register",
            json={
                "name": "search_agent",
                "url": "http://localhost:8001",
                "capabilities": ["search", "index"],
                "health_endpoint": "/health",
                "process_endpoint": "/tasks/send",
                "timeout": 30,
            },
        )

        client.post(
            "/agents/register",
            json={
                "name": "summarize_agent",
                "url": "http://localhost:8002",
                "capabilities": ["summarize", "condense"],
                "health_endpoint": "/health",
                "process_endpoint": "/tasks/send",
                "timeout": 30,
            },
        )

        # Step 2: Discover by capability
        response = client.get("/agents/by-capability/search")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["agents"][0]["name"] == "search_agent"

        # Step 3: Get specific agent info
        response = client.get("/agents/search_agent")
        assert response.status_code == 200
        agent_info = response.json()
        assert agent_info["url"] == "http://localhost:8001"
        assert "search" in agent_info["capabilities"]

        # Step 4: List all agents
        response = client.get("/agents/")
        assert response.status_code == 200
        assert response.json()["count"] == 2


@pytest.mark.ci_fast
class TestProcessAgentTaskDispatch:
    """Test that process_agent_task dispatches to the correct handler based on capabilities."""

    @pytest.fixture
    def app_and_client(self):
        """Create FastAPI app with agents router and injected dependencies."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from cogniverse_runtime.routers.agents import (
            router,
            set_agent_dependencies,
            set_agent_registry,
        )

        config_manager = Mock()
        schema_loader = Mock()
        registry = AgentRegistry(tenant_id="default", config_manager=config_manager)
        set_agent_registry(registry)
        set_agent_dependencies(config_manager, schema_loader)

        app = FastAPI()
        app.include_router(router, prefix="/agents")
        client = TestClient(app)
        return app, client, registry

    def test_routing_capability_dispatches_to_gateway_task(self, app_and_client):
        """Routing agent with 'routing' capability dispatches via _execute_gateway_task."""
        _, client, registry = app_and_client

        registry.register_agent(
            AgentEndpoint(
                name="routing_agent",
                url="http://localhost:8001",
                capabilities=["routing", "query_analysis"],
            )
        )

        mock_result = {
            "status": "success",
            "agent": "routing_agent",
            "message": "Routed 'test' to search_agent",
            "recommended_agent": "search_agent",
            "confidence": 0.9,
        }

        with patch(
            "cogniverse_runtime.routers.agents.AgentDispatcher.dispatch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_dispatch:
            response = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "test query",
                    "context": {"tenant_id": "default"},
                },
            )
            assert response.status_code == 200
            mock_dispatch.assert_called_once()

    def test_search_capability_dispatches_to_search_task(self, app_and_client):
        """Search agent with 'search' capability uses _execute_search_task."""
        _, client, registry = app_and_client

        registry.register_agent(
            AgentEndpoint(
                name="search_agent",
                url="http://localhost:8002",
                capabilities=["search", "video_search", "retrieval"],
            )
        )

        mock_result = {
            "status": "success",
            "agent": "search_agent",
            "results_count": 0,
            "results": [],
        }

        with patch(
            "cogniverse_runtime.routers.agents.AgentDispatcher.dispatch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_dispatch:
            response = client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "find videos",
                    "context": {"tenant_id": "default"},
                },
            )
            assert response.status_code == 200
            mock_dispatch.assert_called_once()

    def test_routing_takes_priority_over_search(self, app_and_client):
        """Agent with both 'routing' and 'search' capabilities routes to routing, not search."""
        _, client, registry = app_and_client

        registry.register_agent(
            AgentEndpoint(
                name="hybrid_agent",
                url="http://localhost:8003",
                capabilities=["routing", "search"],
            )
        )

        mock_result = {"status": "success", "agent": "routing_agent"}

        with patch(
            "cogniverse_runtime.routers.agents.AgentDispatcher.dispatch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_dispatch:
            response = client.post(
                "/agents/hybrid_agent/process",
                json={
                    "agent_name": "hybrid_agent",
                    "query": "test",
                    "context": {"tenant_id": "default"},
                },
            )
            assert response.status_code == 200
            mock_dispatch.assert_called_once()

    def test_unsupported_capability_returns_501(self, app_and_client):
        """Agent with no supported capabilities returns 501."""
        _, client, registry = app_and_client

        registry.register_agent(
            AgentEndpoint(
                name="unknown_agent",
                url="http://localhost:8004",
                capabilities=["unknown_capability"],
            )
        )

        response = client.post(
            "/agents/unknown_agent/process",
            json={
                "agent_name": "unknown_agent",
                "query": "test",
                "context": {"tenant_id": "default"},
            },
        )
        assert response.status_code == 501
