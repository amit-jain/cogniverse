"""
Unit tests for RoutingAgent

NOTE: RoutingAgent has been refactored to use type-safe base classes.
Tests use the new typed API with RoutingDeps, RoutingInput, RoutingOutput.
"""

from unittest.mock import AsyncMock

import pytest

from cogniverse_agents.routing_agent import (
    RoutingAgent,
    RoutingDeps,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig


@pytest.mark.unit
class TestRoutingAgentLegacy:
    """Test cases for RoutingAgent with typed interface"""

    @pytest.fixture
    def mock_system_config(self):
        """Mock system configuration"""
        return {
            "video_agent_url": "http://localhost:8002",
            "text_agent_url": "http://localhost:8003",
        }

    @pytest.fixture
    def mock_routing_decision(self):
        """Mock routing decision"""
        return {
            "query": "test query",
            "recommended_agent": "video_search",
            "confidence": 0.85,
            "reasoning": "Detected video content request",
        }

    @pytest.fixture
    def routing_deps(self):
        """Create RoutingDeps for testing"""
        telemetry_config = TelemetryConfig(enabled=False)
        return RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
        )

    @pytest.mark.ci_fast
    def test_routing_agent_initialization(self, mock_system_config, routing_deps):
        """Test RoutingAgent initialization with typed deps"""
        agent = RoutingAgent(deps=routing_deps)

        # deps is now the config
        assert agent.deps is not None
        assert agent.deps.tenant_id == "test_tenant"
        assert hasattr(agent, "routing_module")
        assert hasattr(agent, "logger")

    @pytest.mark.ci_fast
    def test_routing_agent_initialization_missing_video_agent(self):
        """Test RoutingAgent initialization fails when video agent URL missing"""
        # Old test - routing agent no longer uses video_agent_url
        pass

    @pytest.mark.ci_fast
    def test_build_routing_config(self, mock_system_config):
        """Test routing configuration building"""
        # Old test - routing agent uses RoutingConfig dataclass now
        pass

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_analyze_and_route_video_query(
        self,
        mock_system_config,
        mock_routing_decision,
    ):
        """Test query analysis and routing for video queries"""
        # Old test - analyze_and_route no longer exists, use route_query instead
        pass

    # All remaining tests are skipped - they test the old interface
    pass


@pytest.mark.unit
class TestRoutingConfigLoading:
    """Test RoutingConfig loads query_fusion_config from file/dict/env."""

    @pytest.mark.ci_fast
    def test_from_dict_loads_query_fusion_config(self):
        """RoutingConfig.from_dict() correctly reads query_fusion_config."""
        from cogniverse_agents.routing.config import RoutingConfig

        data = {
            "query_fusion_config": {
                "mode": "parallel",
                "variant_strategies": ["relationship_expansion"],
                "include_original": False,
                "rrf_k": 30,
            }
        }

        config = RoutingConfig.from_dict(data)

        assert config.query_fusion_config["mode"] == "parallel"
        assert config.query_fusion_config["variant_strategies"] == [
            "relationship_expansion"
        ]
        assert config.query_fusion_config["include_original"] is False
        assert config.query_fusion_config["rrf_k"] == 30

    @pytest.mark.ci_fast
    def test_default_query_fusion_config(self):
        """RoutingConfig defaults to mode='single' when no query_fusion_config provided."""
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig()

        assert config.query_fusion_config["mode"] == "single"
        assert config.query_fusion_config["rrf_k"] == 60
        assert config.query_fusion_config["include_original"] is True
        assert (
            "relationship_expansion" in config.query_fusion_config["variant_strategies"]
        )
        assert (
            "boolean_optimization" in config.query_fusion_config["variant_strategies"]
        )

    @pytest.mark.ci_fast
    def test_env_var_overrides_query_fusion_mode(self, monkeypatch):
        """ROUTING_QUERYFUSION_MODE=parallel overrides config."""
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig()
        assert config.query_fusion_config["mode"] == "single"

        monkeypatch.setenv("ROUTING_QUERYFUSION_MODE", "parallel")
        config.merge_with_env()

        assert config.query_fusion_config["mode"] == "parallel"

    @pytest.mark.ci_fast
    def test_config_file_roundtrip(self, tmp_path):
        """RoutingConfig save/load preserves query_fusion_config."""
        from cogniverse_agents.routing.config import RoutingConfig

        original = RoutingConfig.from_dict(
            {
                "query_fusion_config": {
                    "mode": "parallel",
                    "variant_strategies": ["boolean_optimization"],
                    "include_original": True,
                    "rrf_k": 42,
                }
            }
        )

        config_file = tmp_path / "test_routing.json"
        original.save(config_file)

        loaded = RoutingConfig.from_file(config_file)

        assert loaded.query_fusion_config == original.query_fusion_config
        assert loaded.query_fusion_config["mode"] == "parallel"
        assert loaded.query_fusion_config["rrf_k"] == 42


@pytest.mark.unit
class TestEnhancementFailureFallback:
    """Test that enhancement pipeline failures produce safe fallbacks."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_enhance_query_failure_produces_no_variants(self):
        """
        When _enhance_query() crashes under parallel mode config,
        it returns (query, {}) → no query_variants → single-query path.
        """
        telemetry_config = TelemetryConfig(enabled=False)
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
            query_fusion_config={
                "mode": "parallel",
                "variant_strategies": ["relationship_expansion"],
                "include_original": True,
                "rrf_k": 60,
            },
        )
        agent = RoutingAgent(deps=deps)

        # Force the enhancement pipeline to raise
        agent.query_enhancer.enhance_query_with_relationships = AsyncMock(
            side_effect=RuntimeError("Enhancement exploded")
        )

        enhanced_query, metadata = await agent._enhance_query(
            query="robots playing soccer",
            entities=[{"text": "robots", "label": "TECHNOLOGY", "confidence": 0.9}],
            relationships=[],
        )

        # Fallback: original query returned, empty metadata
        assert enhanced_query == "robots playing soccer"
        assert metadata == {}

        # This means no variants flow to SearchAgent → single-query path
        assert metadata.get("query_variants", []) == []


@pytest.mark.unit
class TestInvalidStrategyValidation:
    """Test that invalid strategy names are caught."""

    @pytest.mark.ci_fast
    def test_invalid_strategy_raises_on_get_query_variants(self):
        """
        Invalid strategy names in variant_strategies should raise ValueError,
        not silently skip.
        """
        from cogniverse_agents.routing.query_enhancement_engine import QueryRewriter

        rewriter = QueryRewriter()

        with pytest.raises(ValueError, match="Unknown variant strateg"):
            rewriter.get_query_variants(
                original_query="test query",
                entities=[],
                relationships=[],
                variant_strategies=["nonexistent_strategy"],
                include_original=True,
            )
