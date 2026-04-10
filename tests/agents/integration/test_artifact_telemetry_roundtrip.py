"""
Integration tests for DSPy artifact telemetry round-trip.

Verifies the full save-then-load pipeline through ArtifactManager,
DSPyIntegrationMixin, LLMRoutingStrategy, and AdvancedRoutingOptimizer
against a REAL Phoenix Docker instance.

Requires Docker to be running. Uses the ``phoenix_container`` and
``telemetry_manager_with_phoenix`` fixtures from tests/conftest.py.
"""

import asyncio

import pytest

from cogniverse_agents.dspy_integration_mixin import DSPyIntegrationMixin
from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


@pytest.fixture
def real_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id="artifact-test")


class TestArtifactManagerRoundTrip:
    """Verify ArtifactManager save-then-load produces identical data
    against a real Phoenix instance."""

    @pytest.mark.asyncio
    async def test_prompt_round_trip(self, real_provider):
        """Save prompts to Phoenix, load them back, verify exact equality."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        original = {
            "system_prompt": "You are a routing agent.",
            "analysis_template": "Analyze: {query}",
            "fallback": "Default response",
        }

        dataset_id = await mgr.save_prompts("router", original)
        assert dataset_id  # Real Phoenix returns an actual ID

        loaded = await mgr.load_prompts("router")
        assert loaded == original

    @pytest.mark.asyncio
    async def test_demonstration_round_trip(self, real_provider):
        """Save demos to Phoenix, load them back, verify equality."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        original = [
            {
                "input": '{"query": "find cats"}',
                "output": '{"agent": "video_search"}',
                "metadata": "{}",
            },
            {
                "input": '{"query": "summarize"}',
                "output": '{"agent": "summarizer"}',
                "metadata": "{}",
            },
        ]

        await mgr.save_demonstrations("router", original)
        loaded = await mgr.load_demonstrations("router")

        assert len(loaded) == 2
        assert loaded[0]["input"] == original[0]["input"]
        assert loaded[1]["output"] == original[1]["output"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, real_provider):
        """Loading from Phoenix when no dataset exists returns None."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        result = await mgr.load_prompts("nonexistent_agent_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_optimization_metrics_logged(self, real_provider):
        """Optimization metrics are written to Phoenix experiment store."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        run_id = await mgr.log_optimization_run(
            "router",
            {
                "accuracy": 0.85,
                "latency_ms": 120,
                "num_examples": 50,
            },
        )

        assert run_id  # Real Phoenix returns an actual run ID


class TestTenantIsolation:
    """Verify tenant isolation against real Phoenix."""

    @pytest.mark.asyncio
    async def test_tenant_a_invisible_to_tenant_b(self, telemetry_manager_with_phoenix):
        """Prompts saved for tenant A should not be loadable by tenant B."""
        provider_a = telemetry_manager_with_phoenix.get_provider(
            tenant_id="isolation-tenant-a"
        )
        provider_b = telemetry_manager_with_phoenix.get_provider(
            tenant_id="isolation-tenant-b"
        )

        mgr_a = ArtifactManager(provider_a, tenant_id="isolation-tenant-a")
        mgr_b = ArtifactManager(provider_b, tenant_id="isolation-tenant-b")

        await mgr_a.save_prompts("router", {"system": "Tenant A prompt"})

        # Tenant B should get None (different dataset name)
        result_b = await mgr_b.load_prompts("router")
        assert result_b is None

        # Tenant A should get its own prompts
        result_a = await mgr_a.load_prompts("router")
        assert result_a == {"system": "Tenant A prompt"}

    @pytest.mark.asyncio
    async def test_both_tenants_coexist(self, telemetry_manager_with_phoenix):
        """Both tenants save and load independently from real Phoenix."""
        provider_a = telemetry_manager_with_phoenix.get_provider(
            tenant_id="coexist-tenant-a"
        )
        provider_b = telemetry_manager_with_phoenix.get_provider(
            tenant_id="coexist-tenant-b"
        )

        mgr_a = ArtifactManager(provider_a, tenant_id="coexist-tenant-a")
        mgr_b = ArtifactManager(provider_b, tenant_id="coexist-tenant-b")

        await mgr_a.save_prompts("router", {"system": "A's prompt"})
        await mgr_b.save_prompts("router", {"system": "B's prompt"})

        assert (await mgr_a.load_prompts("router")) == {"system": "A's prompt"}
        assert (await mgr_b.load_prompts("router")) == {"system": "B's prompt"}


class TestMixinAutoLoad:
    """Verify DSPyIntegrationMixin auto-loads prompts from real Phoenix."""

    @pytest.mark.asyncio
    async def test_mixin_loads_saved_prompts_on_init(self, real_provider):
        """Mixin constructor loads existing prompts from Phoenix."""
        tenant_id = "mixin-autoload-test"
        agent_type = "query_analysis"

        # Pre-save prompts to real Phoenix
        mgr = ArtifactManager(real_provider, tenant_id=tenant_id)
        await mgr.save_prompts(
            agent_type,
            {
                "system": "Optimized analysis prompt",
                "template": "Analyze: {query}",
            },
        )

        # Mixin should auto-load on init
        mixin = DSPyIntegrationMixin(
            tenant_id=tenant_id,
            agent_type=agent_type,
            telemetry_provider=real_provider,
        )

        # Allow async callback to complete
        await asyncio.sleep(0.5)

        assert mixin.dspy_enabled is True
        assert mixin.dspy_optimized_prompts == {
            "system": "Optimized analysis prompt",
            "template": "Analyze: {query}",
        }

        assert mixin.get_optimized_prompt("system", "fallback") == (
            "Optimized analysis prompt"
        )
        assert mixin.get_optimized_prompt("missing_key", "fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_mixin_without_prompts_stays_disabled(self, real_provider):
        """Mixin with no saved prompts in Phoenix keeps dspy_enabled=False."""
        mixin = DSPyIntegrationMixin(
            tenant_id="empty-mixin-tenant-xyz",
            agent_type="query_analysis",
            telemetry_provider=real_provider,
        )

        await asyncio.sleep(0.5)

        assert mixin.dspy_enabled is False
        assert mixin.dspy_optimized_prompts == {}
        assert mixin.get_optimized_prompt("system", "default") == "default"


class TestOptimizerExperienceRoundTrip:
    """Verify AdvancedRoutingOptimizer persists and reloads experiences
    through real Phoenix."""

    @pytest.mark.asyncio
    async def test_record_then_persist_then_reload(self, real_provider):
        """Record experiences, persist to Phoenix, reload in new optimizer."""
        config_kwargs = dict(
            tenant_id="optimizer-exp-test",
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            telemetry_provider=real_provider,
        )

        # Create optimizer and record an experience
        optimizer = AdvancedRoutingOptimizer(**config_kwargs)
        reward = await optimizer.record_routing_experience(
            query="find cats in videos",
            entities=["cat"],
            relationships=[],
            enhanced_query="find feline content in video library",
            chosen_agent="video_search",
            routing_confidence=0.9,
            search_quality=0.85,
            agent_success=True,
        )

        assert isinstance(reward, float)
        assert len(optimizer.experiences) == 1

        # Persist to real Phoenix
        await optimizer._persist_data()

        # Create a NEW optimizer from the same provider — should load from Phoenix
        optimizer2 = AdvancedRoutingOptimizer(**config_kwargs)
        # Allow async callback to complete
        await asyncio.sleep(0.5)

        assert len(optimizer2.experiences) == 1
        assert optimizer2.experiences[0].query == "find cats in videos"
        assert optimizer2.experiences[0].chosen_agent == "video_search"


# ---------------------------------------------------------------------------
# Agent artifact loading round-trip — real Phoenix, real agents
# ---------------------------------------------------------------------------


class TestGatewayAgentArtifactRoundTrip:
    """Save threshold config to real Phoenix → GatewayAgent loads it → routing changes."""

    @pytest.mark.asyncio
    async def test_gateway_loads_real_artifact_and_applies_thresholds(self, real_provider):
        """Full round-trip: save thresholds → load via _load_artifact → verify deps changed."""
        import json

        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps
        from cogniverse_foundation.telemetry.manager import TelemetryManager

        tenant_id = "gateway-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a threshold config with specific values different from defaults
        optimized_config = {
            "fast_path_confidence_threshold": 0.65,
            "gliner_threshold": 0.42,
        }
        dataset_id = await mgr.save_blob(
            "config", "gateway_thresholds", json.dumps(optimized_config)
        )
        assert dataset_id, "Failed to save threshold config to Phoenix"

        # Verify the blob is loadable (sanity check)
        loaded_blob = await mgr.load_blob("config", "gateway_thresholds")
        assert loaded_blob is not None
        assert json.loads(loaded_blob) == optimized_config

        # Create a GatewayAgent with defaults
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps)
        assert agent.deps.fast_path_confidence_threshold == 0.4  # default
        assert agent.deps.gliner_threshold == 0.3  # default

        # Inject telemetry and artifact tenant (simulates what dispatcher does)
        tm = TelemetryManager.get_instance()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id

        # Load artifact — this should update deps
        agent._load_artifact()

        # Verify the agent now has the optimized thresholds, not defaults
        assert agent.deps.fast_path_confidence_threshold == 0.65, (
            f"Expected 0.65, got {agent.deps.fast_path_confidence_threshold} — "
            "artifact loading did not apply the threshold"
        )
        assert agent.deps.gliner_threshold == 0.42, (
            f"Expected 0.42, got {agent.deps.gliner_threshold} — "
            "artifact loading did not apply the gliner threshold"
        )

    @pytest.mark.asyncio
    async def test_gateway_with_no_artifact_keeps_defaults(self, real_provider):
        """Agent without an artifact in Phoenix should keep default thresholds."""
        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps
        from cogniverse_foundation.telemetry.manager import TelemetryManager

        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps)

        tm = TelemetryManager.get_instance()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = "nonexistent-tenant-xyz"

        agent._load_artifact()

        assert agent.deps.fast_path_confidence_threshold == 0.4
        assert agent.deps.gliner_threshold == 0.3

    @pytest.mark.asyncio
    async def test_gateway_threshold_affects_routing_decision(self, real_provider):
        """Changing fast_path_confidence_threshold changes which queries go to orchestrator.

        A query that produces a GLiNER confidence of ~0.5 should be:
        - "simple" with default threshold 0.4 (0.5 > 0.4)
        - "complex" with raised threshold 0.8 (0.5 < 0.8)
        """
        import json

        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )
        from cogniverse_foundation.telemetry.manager import TelemetryManager

        tenant_id = "gateway-routing-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a HIGH threshold that pushes borderline queries to orchestrator
        high_threshold_config = {
            "fast_path_confidence_threshold": 0.95,
            "gliner_threshold": 0.3,
        }
        await mgr.save_blob(
            "config", "gateway_thresholds", json.dumps(high_threshold_config)
        )

        # Create agent with default thresholds, run a borderline query
        deps_default = GatewayDeps()
        agent_default = GatewayAgent(deps=deps_default)
        # Mock GLiNER to return a medium-confidence entity
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.6}
        ]
        agent_default._gliner_model = mock_model

        result_default = await agent_default._process_impl(
            GatewayInput(query="find videos")
        )
        # With default 0.4 threshold, confidence 0.6 should be "simple"
        assert result_default.complexity == "simple", (
            f"With default 0.4 threshold, 0.6 confidence should be simple, "
            f"got {result_default.complexity}"
        )

        # Now create another agent and load the high-threshold artifact
        deps_optimized = GatewayDeps()
        agent_optimized = GatewayAgent(deps=deps_optimized)
        agent_optimized._gliner_model = mock_model

        tm = TelemetryManager.get_instance()
        agent_optimized.telemetry_manager = tm
        agent_optimized._artifact_tenant_id = tenant_id
        agent_optimized._load_artifact()

        assert agent_optimized.deps.fast_path_confidence_threshold == 0.95

        result_optimized = await agent_optimized._process_impl(
            GatewayInput(query="find videos")
        )
        # With 0.95 threshold, 0.6 confidence should be "complex"
        assert result_optimized.complexity == "complex", (
            f"With 0.95 threshold, 0.6 confidence should be complex, "
            f"got {result_optimized.complexity}"
        )
        assert result_optimized.routed_to == "orchestrator_agent"


class TestDSPyAgentArtifactRoundTrip:
    """Save DSPy module state to real Phoenix → agent loads it → module state changes."""

    @pytest.mark.asyncio
    async def test_entity_extraction_loads_real_dspy_state(self, real_provider):
        """Save a DSPy module state → EntityExtractionAgent loads it → state applied."""
        import json
        from unittest.mock import patch

        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionModule,
        )
        from cogniverse_foundation.telemetry.manager import TelemetryManager

        tenant_id = "entity-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Create a real DSPy module, get its default state, mutate it to
        # simulate optimization, and save the mutated state as an artifact
        with patch("dspy.ChainOfThought"):
            original_module = EntityExtractionModule()
        default_state = original_module.dump_state()

        # Inject fake demos into the state to simulate optimization output
        optimized_state = default_state.copy()
        for key in optimized_state:
            if "predict" in key.lower():
                optimized_state[key] = dict(optimized_state[key])
                optimized_state[key]["demos"] = [
                    {
                        "query": "find ML transformer papers",
                        "entities": "ML|CONCEPT|0.9\ntransformer|CONCEPT|0.85",
                        "entity_types": "CONCEPT",
                    },
                    {
                        "query": "latest NVIDIA GPU benchmarks",
                        "entities": "NVIDIA|ORG|0.95\nGPU|TECHNOLOGY|0.8",
                        "entity_types": "ORG,TECHNOLOGY",
                    },
                ]

        # Save to real Phoenix
        state_json = json.dumps(optimized_state, default=str)
        dataset_id = await mgr.save_blob("model", "entity_extraction", state_json)
        assert dataset_id

        # Verify blob round-trips correctly
        loaded_json = await mgr.load_blob("model", "entity_extraction")
        assert loaded_json is not None
        loaded_state = json.loads(loaded_json)

        # Find the predict key and verify demos survived the round-trip
        predict_key = next(k for k in loaded_state if "predict" in k.lower())
        assert len(loaded_state[predict_key]["demos"]) == 2
        assert loaded_state[predict_key]["demos"][0]["query"] == "find ML transformer papers"
        assert "NVIDIA" in loaded_state[predict_key]["demos"][1]["entities"]

        # Now test that the agent loads this artifact correctly
        with patch("dspy.ChainOfThought"):
            deps = EntityExtractionDeps()
            agent = EntityExtractionAgent(deps=deps)

        # Agent's module should have empty demos initially
        agent_state_before = agent.dspy_module.dump_state()
        predict_key_before = next(
            (k for k in agent_state_before if "predict" in k.lower()), None
        )
        if predict_key_before:
            demos_before = agent_state_before[predict_key_before].get("demos", [])
            # Default module has 0 demos
            assert len(demos_before) == 0, (
                f"Fresh module should have 0 demos, got {len(demos_before)}"
            )

        # Load artifact
        tm = TelemetryManager.get_instance()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Agent's module should now have the 2 demos from the artifact
        agent_state_after = agent.dspy_module.dump_state()
        predict_key_after = next(k for k in agent_state_after if "predict" in k.lower())
        demos_after = agent_state_after[predict_key_after].get("demos", [])
        assert len(demos_after) == 2, (
            f"Agent should have loaded 2 demos from artifact, got {len(demos_after)}"
        )
        assert demos_after[0]["query"] == "find ML transformer papers"
        assert "NVIDIA" in demos_after[1]["entities"]
