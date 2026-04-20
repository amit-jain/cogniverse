"""
Unit tests for RoutingAgent

Tests the thin DSPy decision-maker after gutting inline preprocessing.
RoutingAgent now receives pre-enriched data (entities, relationships,
enhanced_query) from upstream A2A agents.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing_agent import (
    RoutingAgent,
    RoutingDeps,
    RoutingInput,
    RoutingOutput,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig


def _make_deps(**overrides) -> RoutingDeps:
    """Create minimal RoutingDeps for testing."""
    defaults = {
        "telemetry_config": TelemetryConfig(enabled=False),
    }
    defaults.update(overrides)
    return RoutingDeps(**defaults)


def _make_agent(deps=None, **dep_overrides) -> RoutingAgent:
    """Create a RoutingAgent with mocked DSPy LM."""
    if deps is None:
        deps = _make_deps(**dep_overrides)

    def _mock_configure_dspy(self_agent, deps_arg):
        self_agent._dspy_lm = MagicMock()

    with patch.object(RoutingAgent, "_configure_dspy", _mock_configure_dspy):
        return RoutingAgent(deps=deps)


@pytest.mark.unit
class TestRoutingConfigLoading:
    """Test RoutingConfig loads query_fusion_config from file/dict/env."""

    @pytest.mark.ci_fast
    def test_from_dict_loads_query_fusion_config(self):
        from cogniverse_agents.routing.config import RoutingConfig

        data = {"query_fusion_config": {"include_original": False, "rrf_k": 30}}
        config = RoutingConfig.from_dict(data)
        assert config.query_fusion_config["include_original"] is False
        assert config.query_fusion_config["rrf_k"] == 30

    @pytest.mark.ci_fast
    def test_default_query_fusion_config(self):
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig()
        assert config.query_fusion_config["rrf_k"] == 60
        assert config.query_fusion_config["include_original"] is True
        assert "mode" not in config.query_fusion_config
        assert "variant_strategies" not in config.query_fusion_config

    @pytest.mark.ci_fast
    def test_from_dict_overrides_query_fusion_rrf_k(self):
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig.from_dict(
            {"query_fusion_config": {"rrf_k": 30, "include_original": True}}
        )
        assert config.query_fusion_config["rrf_k"] == 30

    @pytest.mark.ci_fast
    def test_config_file_roundtrip(self, tmp_path):
        from cogniverse_agents.routing.config import RoutingConfig

        original = RoutingConfig.from_dict(
            {"query_fusion_config": {"include_original": True, "rrf_k": 42}}
        )
        config_file = tmp_path / "test_routing.json"
        original.save(config_file)
        loaded = RoutingConfig.from_file(config_file)
        assert loaded.query_fusion_config == original.query_fusion_config
        assert loaded.query_fusion_config["rrf_k"] == 42

    @pytest.mark.ci_fast
    def test_entity_confidence_threshold_defaults(self):
        from cogniverse_agents.routing.config import RoutingConfig

        config = RoutingConfig()
        assert config.entity_confidence_threshold == 0.6
        assert config.min_entities_for_fast_path == 1


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRoutingDecisionContextInjection:
    """Verify _make_routing_decision wires the FULL context stack
    (instructions + strategies + memory) into the DSPy routing call."""

    def test_make_routing_decision_calls_inject_context_into_prompt(self):
        source = inspect.getsource(RoutingAgent._make_routing_decision)
        assert "inject_context_into_prompt" in source, (
            "_make_routing_decision must call self.inject_context_into_prompt() "
            "to inject the FULL context stack (instructions + strategies + memory)."
        )

    def test_routing_agent_inherits_extended_memory_aware_mixin(self):
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        assert issubclass(RoutingAgent, MemoryAwareMixin)
        assert hasattr(MemoryAwareMixin, "get_strategies")
        assert callable(MemoryAwareMixin.get_strategies)


@pytest.mark.unit
class TestRoutingInputOutput:
    """Test RoutingInput and RoutingOutput type definitions."""

    @pytest.mark.ci_fast
    def test_routing_input_accepts_enriched_data(self):
        inp = RoutingInput(
            query="find robots",
            tenant_id="t1",
            enhanced_query="find robot videos in tournaments",
            entities=[{"text": "robots", "label": "OBJECT"}],
            relationships=[{"subject": "robots", "relation": "play", "object": "soccer"}],
        )
        assert inp.query == "find robots"
        assert inp.enhanced_query == "find robot videos in tournaments"
        assert len(inp.entities) == 1
        assert len(inp.relationships) == 1

    @pytest.mark.ci_fast
    def test_routing_input_optional_enrichment(self):
        inp = RoutingInput(query="hello", tenant_id="t1")
        assert inp.enhanced_query is None
        assert inp.entities is None
        assert inp.relationships is None

    @pytest.mark.ci_fast
    def test_routing_output_slim(self):
        out = RoutingOutput(
            query="test",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Best match for search queries",
        )
        assert out.recommended_agent == "search_agent"
        assert out.confidence == 0.85
        assert out.fallback_agents == []
        assert out.metadata == {}

    @pytest.mark.ci_fast
    def test_routing_output_deprecated_fields_default_empty(self):
        """Deprecated fields default to empty for backward compat."""
        out = RoutingOutput(
            query="test",
            recommended_agent="search_agent",
            confidence=0.8,
            reasoning="test",
        )
        assert out.enhanced_query == ""
        assert out.entities == []
        assert out.relationships == []
        assert out.query_variants == []
        # Convenience properties still work
        assert out.extracted_entities == []
        assert out.extracted_relationships == []
        assert out.routing_metadata == {}
        # timestamp no longer exists
        assert "timestamp" not in RoutingOutput.model_fields


@pytest.mark.unit
class TestRoutingAgentInit:
    """Test RoutingAgent initialization."""

    @pytest.mark.ci_fast
    def test_init_creates_routing_module(self):
        agent = _make_agent()
        assert agent.routing_module is not None

    @pytest.mark.ci_fast
    def test_init_no_telemetry_manager_when_disabled(self):
        agent = _make_agent()
        assert agent.telemetry_manager is None

    @pytest.mark.ci_fast
    def test_init_no_preprocessing_attributes(self):
        """Gutted agent has no enhancement pipeline, optimizer, or MLflow."""
        agent = _make_agent()
        assert not hasattr(agent, "query_enhancer")
        assert not hasattr(agent, "analysis_module")
        assert not hasattr(agent, "grpo_optimizer")
        assert not hasattr(agent, "mlflow_integration")
        assert not hasattr(agent, "cache_manager")
        assert not hasattr(agent, "parallel_executor")
        assert not hasattr(agent, "contextual_analyzer")
        assert not hasattr(agent, "metrics_tracker")
        assert not hasattr(agent, "multi_modal_reranker")
        assert not hasattr(agent, "lazy_executor")

    @pytest.mark.ci_fast
    def test_init_sets_agent_name(self):
        agent = _make_agent()
        assert agent.agent_name == "routing_agent"


@pytest.mark.unit
class TestRoutingDecision:
    """Test routing decision logic."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_route_query_requires_tenant_id(self):
        agent = _make_agent()
        with pytest.raises(ValueError, match="tenant_id is required"):
            await agent.route_query(query="test", tenant_id=None)

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_route_query_returns_routing_output(self):
        agent = _make_agent()
        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.9
        mock_prediction.reasoning_chain = ["Route to search"]

        with patch.object(
            agent.routing_module, "forward", return_value=mock_prediction
        ):
            result = await agent.route_query(query="find videos", tenant_id="t1")

        assert isinstance(result, RoutingOutput)
        assert result.query == "find videos"
        assert result.recommended_agent == "search_agent"
        assert result.confidence == 0.9
        assert result.metadata["tenant_id"] == "t1"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_route_query_passes_enriched_data(self):
        agent = _make_agent()
        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "summarizer_agent"}
        mock_prediction.overall_confidence = 0.8
        mock_prediction.reasoning_chain = ["Summarization needed"]

        forward_calls = []

        def capture_forward(**kwargs):
            forward_calls.append(kwargs)
            return mock_prediction

        with patch.object(agent.routing_module, "forward", side_effect=capture_forward):
            result = await agent.route_query(
                query="summarize AI papers",
                enhanced_query="summarize recent AI research papers from 2024",
                entities=[{"text": "AI", "label": "TECH"}],
                relationships=[{"subject": "AI", "relation": "in", "object": "papers"}],
                tenant_id="t1",
            )

        assert result.recommended_agent == "summarizer_agent"
        # Enhanced query should have been used
        assert len(forward_calls) == 1
        assert "summarize recent AI research papers from 2024" in forward_calls[0]["query"]

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_route_query_fallback_on_error(self):
        agent = _make_agent()
        with patch.object(
            agent.routing_module,
            "forward",
            side_effect=RuntimeError("DSPy exploded"),
        ):
            result = await agent.route_query(query="test", tenant_id="t1")

        assert result.recommended_agent == "search_agent"
        assert result.confidence <= 0.3
        assert "Fallback" in result.reasoning or "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_process_impl_delegates_to_route_query(self):
        agent = _make_agent()
        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.7
        mock_prediction.reasoning_chain = ["Search query"]

        with patch.object(
            agent.routing_module, "forward", return_value=mock_prediction
        ):
            inp = RoutingInput(
                query="find videos",
                tenant_id="t1",
                entities=[{"text": "videos", "label": "MEDIA"}],
            )
            result = await agent._process_impl(inp)

        assert isinstance(result, RoutingOutput)
        assert result.recommended_agent == "search_agent"


@pytest.mark.unit
class TestParseConfidence:
    """Test _parse_confidence static method."""

    @pytest.mark.ci_fast
    def test_float_passthrough(self):
        assert RoutingAgent._parse_confidence(0.85) == 0.85

    @pytest.mark.ci_fast
    def test_string_decimal(self):
        assert RoutingAgent._parse_confidence("0.75") == 0.75

    @pytest.mark.ci_fast
    def test_string_percentage(self):
        assert RoutingAgent._parse_confidence("85%") == 0.85

    @pytest.mark.ci_fast
    def test_clamps_above_one(self):
        assert RoutingAgent._parse_confidence(1.5) == 1.0

    @pytest.mark.ci_fast
    def test_clamps_below_zero(self):
        assert RoutingAgent._parse_confidence(-0.3) == 0.0

    @pytest.mark.ci_fast
    def test_invalid_returns_default(self):
        assert RoutingAgent._parse_confidence("not_a_number") == 0.5

    @pytest.mark.ci_fast
    def test_none_returns_default(self):
        assert RoutingAgent._parse_confidence(None) == 0.5


@pytest.mark.unit
class TestExtractReasoning:
    """Test _extract_reasoning static method."""

    @pytest.mark.ci_fast
    def test_reasoning_chain_list(self):
        result = MagicMock()
        result.reasoning_chain = ["Step 1", "Step 2"]
        result.reasoning = "unused"
        assert RoutingAgent._extract_reasoning(result) == "Step 1 Step 2"

    @pytest.mark.ci_fast
    def test_reasoning_string(self):
        result = MagicMock(spec=[])
        result.reasoning = "Simple reasoning"
        # No reasoning_chain attribute
        delattr(result, "reasoning") if hasattr(result, "reasoning") else None
        result = MagicMock()
        result.reasoning_chain = None
        result.reasoning = "Simple reasoning"
        assert RoutingAgent._extract_reasoning(result) == "Simple reasoning"

    @pytest.mark.ci_fast
    def test_no_reasoning(self):
        result = MagicMock()
        result.reasoning_chain = None
        result.reasoning = ""
        assert RoutingAgent._extract_reasoning(result) == ""


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoutingArtifactLoading:
    @pytest.mark.asyncio
    async def test_loads_dspy_artifact(self):
        """RoutingAgent should load optimized DSPy module state."""
        import json
        from unittest.mock import AsyncMock

        agent = _make_agent()
        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()
        fake_state = {"router.predict": {"signature": {"fields": []}, "demos": []}}

        with patch("cogniverse_agents.optimizer.artifact_manager.ArtifactManager") as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(return_value=json.dumps(fake_state))

            agent.telemetry_manager = mock_tm
            agent._artifact_tenant_id = "test:unit"
            agent.routing_module = MagicMock()
            agent._load_artifact()

        agent.routing_module.load_state.assert_called_once_with(fake_state)

    def test_defaults_without_artifact(self):
        """Agent uses default routing module when no artifact exists."""
        agent = _make_agent()
        assert hasattr(agent, "routing_module")
        assert agent.routing_module is not None

    def test_no_telemetry_skips_loading(self):
        """_load_artifact is a no-op when telemetry_manager is not set."""
        agent = _make_agent()
        agent.telemetry_manager = None
        agent._load_artifact()

    @pytest.mark.asyncio
    async def test_artifact_load_failure_uses_defaults(self):
        """_load_artifact falls back to defaults when artifact load fails."""
        from unittest.mock import AsyncMock

        agent = _make_agent()
        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()

        with patch("cogniverse_agents.optimizer.artifact_manager.ArtifactManager") as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(side_effect=RuntimeError("connection refused"))
            agent.telemetry_manager = mock_tm
            agent._artifact_tenant_id = "test:unit"
            agent._load_artifact()
