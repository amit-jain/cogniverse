"""
Integration tests for A2A Gateway
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestA2AGatewayIntegration:
    """Integration tests for A2A Gateway FastAPI endpoints"""

    @pytest.mark.ci_fast
    def test_health_check(self, telemetry_manager_without_phoenix):
        """Test health check endpoint"""
        from cogniverse_agents.a2a_gateway import A2AGateway

        gateway = A2AGateway(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_orchestration=False,
        )
        client = TestClient(gateway.app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "enhanced_system" in data
        assert "orchestration_enabled" in data
        assert "timestamp" in data
        assert data["orchestration_enabled"] is False

    @pytest.mark.ci_fast
    def test_health_check_with_orchestration(self, telemetry_manager_without_phoenix):
        """Test health check with orchestration enabled"""
        from cogniverse_agents.a2a_gateway import A2AGateway

        gateway = A2AGateway(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_orchestration=True,
        )
        client = TestClient(gateway.app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["orchestration_enabled"] is True

    @pytest.mark.ci_fast
    def test_stats_endpoint(self, telemetry_manager_without_phoenix):
        """Test statistics endpoint"""
        from cogniverse_agents.a2a_gateway import A2AGateway

        gateway = A2AGateway(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_orchestration=False,
        )
        client = TestClient(gateway.app)
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "enhanced_routing_stats" in data or "routing_stats" in data
        assert "total_requests" in data

    @pytest.mark.asyncio
    async def test_route_endpoint(self, telemetry_manager_without_phoenix):
        """Test /route endpoint with basic query"""
        from cogniverse_agents.a2a_gateway import A2AGateway

        gateway = A2AGateway(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_orchestration=False,
        )
        client = TestClient(gateway.app)

        request_data = {
            "query": "Show me videos of robots",
            "context": None,
            "user_id": "test_user",
        }

        response = client.post("/route", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "agent" in data or "recommended_agent" in data
        assert "confidence" in data
        assert "enhanced_query" in data
        assert "processing_time_ms" in data

    @pytest.mark.asyncio
    async def test_orchestrate_endpoint_disabled(
        self, telemetry_manager_without_phoenix
    ):
        """Test /orchestrate endpoint when orchestration is disabled"""
        from cogniverse_agents.a2a_gateway import A2AGateway

        gateway = A2AGateway(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_orchestration=False,
        )
        client = TestClient(gateway.app)

        request_data = {
            "query": "Complex query requiring orchestration",
            "force_orchestration": False,
        }

        response = client.post("/orchestrate", json=request_data)
        # Should fail or return error when orchestration disabled
        assert response.status_code in [400, 501, 503]

    @pytest.mark.asyncio
    async def test_orchestrate_endpoint_enabled(
        self, telemetry_manager_without_phoenix
    ):
        """Test /orchestrate endpoint when orchestration is enabled"""
        from cogniverse_agents.a2a_gateway import A2AGateway

        gateway = A2AGateway(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_orchestration=True,
        )
        client = TestClient(gateway.app)

        request_data = {
            "query": "Complex query requiring orchestration",
            "force_orchestration": True,
        }

        response = client.post("/orchestrate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "workflow_id" in data
        assert "status" in data or "execution_status" in data
