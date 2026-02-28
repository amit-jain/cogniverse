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
from cogniverse_agents.routing.strategies import LLMRoutingStrategy
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

        run_id = await mgr.log_optimization_run("router", {
            "accuracy": 0.85,
            "latency_ms": 120,
            "num_examples": 50,
        })

        assert run_id  # Real Phoenix returns an actual run ID


class TestTenantIsolation:
    """Verify tenant isolation against real Phoenix."""

    @pytest.mark.asyncio
    async def test_tenant_a_invisible_to_tenant_b(
        self, telemetry_manager_with_phoenix
    ):
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


class TestOptimizerToStrategyWiring:
    """Verify the full optimizer-saves -> strategy-loads pipeline against
    real Phoenix.

    This is the test for the filename mismatch bug: the old code had
    router_optimizer writing to ``router_prompt_artifact.json`` while
    LLMRoutingStrategy read from ``routing_prompts.json``. Both now
    use dataset name ``dspy-prompts-{tenant}-router``.
    """

    @pytest.mark.asyncio
    async def test_strategy_loads_optimizer_prompts(self, real_provider):
        """Prompts saved to Phoenix are loaded by LLMRoutingStrategy."""
        tenant_id = "strategy-wiring-test"

        # Simulate optimizer saving prompts to real Phoenix
        mgr = ArtifactManager(real_provider, tenant_id=tenant_id)
        await mgr.save_prompts("router", {
            "system_prompt": "Optimized routing instructions here.",
            "prompt_template": "Route: {system_prompt}\n{conversation_history}\n{query}",
        })

        # Strategy loads them via the same provider + tenant
        strategy = LLMRoutingStrategy(config={
            "enable_dspy_optimization": True,
            "tenant_id": tenant_id,
            "telemetry_provider": real_provider,
            "model": "test-model",
        })

        # In an async context, prompts load via ensure_future + callback.
        await asyncio.sleep(0.5)

        assert strategy.optimized_prompts["system_prompt"] == (
            "Optimized routing instructions here."
        )
        assert "prompt_template" in strategy.optimized_prompts

        # The system prompt should be the optimized one, not the default
        assert strategy.system_prompt == "Optimized routing instructions here."

        # Optimization status should reflect loaded state
        status = strategy.get_optimization_status()
        assert status["dspy_enabled"] is True
        assert status["optimized_prompts_loaded"] is True
        assert status["using_optimized_system_prompt"] is True

    @pytest.mark.asyncio
    async def test_strategy_without_prompts_uses_default(self, real_provider):
        """Strategy with no saved prompts in Phoenix falls back to default."""
        strategy = LLMRoutingStrategy(config={
            "enable_dspy_optimization": True,
            "tenant_id": "empty-strategy-tenant-xyz",
            "telemetry_provider": real_provider,
            "model": "test-model",
        })

        await asyncio.sleep(0.5)

        # No prompts saved -> default prompt used
        assert strategy.optimized_prompts == {}
        assert "routing agent" in strategy.system_prompt.lower()

        status = strategy.get_optimization_status()
        assert status["dspy_enabled"] is True
        assert status["optimized_prompts_loaded"] is False


class TestMixinAutoLoad:
    """Verify DSPyIntegrationMixin auto-loads prompts from real Phoenix."""

    @pytest.mark.asyncio
    async def test_mixin_loads_saved_prompts_on_init(self, real_provider):
        """Mixin constructor loads existing prompts from Phoenix."""
        tenant_id = "mixin-autoload-test"
        agent_type = "query_analysis"

        # Pre-save prompts to real Phoenix
        mgr = ArtifactManager(real_provider, tenant_id=tenant_id)
        await mgr.save_prompts(agent_type, {
            "system": "Optimized analysis prompt",
            "template": "Analyze: {query}",
        })

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
            llm_config=LLMEndpointConfig(model="ollama/test-model"),
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
