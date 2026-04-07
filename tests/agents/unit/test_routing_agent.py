"""
Unit tests for RoutingAgent

NOTE: RoutingAgent has been refactored to use type-safe base classes.
Tests use the new typed API with RoutingDeps, RoutingInput, RoutingOutput.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from cogniverse_agents.routing_agent import (
    RoutingAgent,
    RoutingDeps,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig


def _make_mock_telemetry_provider():
    """Create a mock TelemetryProvider with in-memory stores."""
    provider = MagicMock()
    datasets: dict[str, pd.DataFrame] = {}

    async def create_dataset(name, data, metadata=None):
        datasets[name] = data
        return f"ds-{name}"

    async def get_dataset(name):
        if name not in datasets:
            raise KeyError(f"Dataset {name} not found")
        return datasets[name]

    provider.datasets = MagicMock()
    provider.datasets.create_dataset = AsyncMock(side_effect=create_dataset)
    provider.datasets.get_dataset = AsyncMock(side_effect=get_dataset)
    provider.experiments = MagicMock()
    provider.experiments.create_experiment = AsyncMock(return_value="exp-test")
    provider.experiments.log_run = AsyncMock(return_value="run-test")
    return provider


@pytest.mark.unit
class TestRoutingConfigLoading:
    """Test RoutingConfig loads query_fusion_config from file/dict/env."""

    @pytest.mark.ci_fast
    def test_from_dict_loads_query_fusion_config(self):
        """RoutingConfig.from_dict() correctly reads query_fusion_config."""
        from cogniverse_agents.routing.config import RoutingConfig

        data = {
            "query_fusion_config": {
                "include_original": False,
                "rrf_k": 30,
            }
        }

        config = RoutingConfig.from_dict(data)

        assert config.query_fusion_config["include_original"] is False
        assert config.query_fusion_config["rrf_k"] == 30

    @pytest.mark.ci_fast
    def test_default_query_fusion_config(self):
        """RoutingConfig defaults have include_original=True, rrf_k=60."""
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig()

        assert config.query_fusion_config["rrf_k"] == 60
        assert config.query_fusion_config["include_original"] is True
        # No mode or variant_strategies — composable module always generates variants
        assert "mode" not in config.query_fusion_config
        assert "variant_strategies" not in config.query_fusion_config

    @pytest.mark.ci_fast
    def test_from_dict_overrides_query_fusion_rrf_k(self):
        """from_dict overrides rrf_k."""
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig.from_dict(
            {"query_fusion_config": {"rrf_k": 30, "include_original": True}}
        )
        assert config.query_fusion_config["rrf_k"] == 30

    @pytest.mark.ci_fast
    def test_config_file_roundtrip(self, tmp_path):
        """RoutingConfig save/load preserves query_fusion_config."""
        from cogniverse_agents.routing.config import RoutingConfig

        original = RoutingConfig.from_dict(
            {
                "query_fusion_config": {
                    "include_original": True,
                    "rrf_k": 42,
                }
            }
        )

        config_file = tmp_path / "test_routing.json"
        original.save(config_file)

        loaded = RoutingConfig.from_file(config_file)

        assert loaded.query_fusion_config == original.query_fusion_config
        assert loaded.query_fusion_config["rrf_k"] == 42

    @pytest.mark.ci_fast
    def test_entity_confidence_threshold_defaults(self):
        """RoutingConfig has composable module threshold defaults."""
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig()

        assert config.entity_confidence_threshold == 0.6
        assert config.min_entities_for_fast_path == 1


@pytest.mark.unit
class TestEnhancementFailureFallback:
    """Test that enhancement pipeline failures produce safe fallbacks."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_analyze_and_enhance_query_failure_produces_no_variants(self):
        """
        When _analyze_and_enhance_query() crashes,
        it returns ([], [], query, {}) → no query_variants → single-query path.
        """
        mock_provider = _make_mock_telemetry_provider()
        telemetry_config = TelemetryConfig(enabled=False)
        deps = RoutingDeps(
            telemetry_config=telemetry_config,
            query_fusion_config={
                "include_original": True,
                "rrf_k": 60,
            },
        )

        with patch.object(
            RoutingAgent, "_get_telemetry_provider", return_value=mock_provider
        ):
            agent = RoutingAgent(deps=deps)

        assert agent.query_enhancer is not None, (
            "query_enhancer must be initialized to test failure fallback"
        )

        agent.query_enhancer.enhance_query_with_relationships = AsyncMock(
            side_effect=RuntimeError("Enhancement exploded")
        )

        (
            entities,
            relationships,
            enhanced_query,
            metadata,
        ) = await agent._analyze_and_enhance_query(
            query="robots playing soccer",
        )

        # Fallback: original query returned, empty metadata, no entities/relationships
        assert enhanced_query == "robots playing soccer"
        assert metadata == {}
        assert entities == []
        assert relationships == []

        # This means no variants flow to SearchAgent → single-query path
        assert metadata.get("query_variants", []) == []


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRoutingDecisionContextInjection:
    """Audit fix #8 — verify _make_routing_decision wires the FULL context
    stack (instructions + strategies + memory) into the DSPy routing call.

    Before this fix the method only loaded raw _get_tenant_instructions(),
    so learned strategies and tenant memories never reached the router. The
    fix replaces that with self.inject_context_into_prompt(prompt=, query=)
    which gracefully handles uninitialized memory."""

    def test_make_routing_decision_calls_inject_context_into_prompt(self):
        """Pin the wiring at the source level. A regression where someone
        replaces inject_context_into_prompt() with a partial fallback (e.g.,
        just _get_tenant_instructions()) would silently drop strategies."""
        import inspect

        from cogniverse_agents.routing_agent import RoutingAgent

        source = inspect.getsource(RoutingAgent._make_routing_decision)
        assert "inject_context_into_prompt" in source, (
            "_make_routing_decision must call self.inject_context_into_prompt() "
            "to inject the FULL context stack (instructions + strategies + memory). "
            "Audit fix #8 requires this — see "
            "docs/superpowers/audits/2026-04-07-orphan-and-wiring-audit.md"
        )

    def test_routing_agent_inherits_extended_memory_aware_mixin(self):
        """RoutingAgent must inherit from the EXTENDED MemoryAwareMixin
        (cogniverse_agents.memory_aware_mixin) — the one with get_strategies
        and the strategies-aware inject_context_into_prompt. Audit fix #16
        deleted the base mixin in cogniverse_core, so this test pins that
        the agent gets the right one."""
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
        from cogniverse_agents.routing_agent import RoutingAgent

        assert issubclass(RoutingAgent, MemoryAwareMixin)
        # The extended mixin has get_strategies; the deleted base did not.
        assert hasattr(MemoryAwareMixin, "get_strategies")
        assert callable(MemoryAwareMixin.get_strategies)
