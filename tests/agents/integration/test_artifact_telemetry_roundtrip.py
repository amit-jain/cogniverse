"""
Integration tests for DSPy artifact telemetry round-trip.

Verifies the full save-then-load pipeline through ArtifactManager,
DSPyIntegrationMixin, LLMRoutingStrategy, and AdvancedRoutingOptimizer.

These tests use in-memory mock telemetry providers at the DatasetStore
boundary — the full internal pipeline (serialization, DataFrame conversion,
naming conventions, tenant isolation) is exercised without a real Phoenix.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.dspy_integration_mixin import DSPyIntegrationMixin
from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer,
)
from cogniverse_agents.routing.strategies import LLMRoutingStrategy
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


def _make_mock_telemetry_provider():
    """In-memory telemetry provider that actually stores DataFrames by name."""
    provider = MagicMock()
    datasets: dict = {}

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
    # Expose the store for assertions
    provider._datasets = datasets
    return provider


class TestArtifactManagerRoundTrip:
    """Verify ArtifactManager save-then-load produces identical data."""

    @pytest.mark.asyncio
    async def test_prompt_round_trip(self):
        """Save prompts, load them back, verify exact equality."""
        provider = _make_mock_telemetry_provider()
        mgr = ArtifactManager(provider, tenant_id="acme")

        original = {
            "system_prompt": "You are a routing agent.",
            "analysis_template": "Analyze: {query}",
            "fallback": "Default response",
        }

        dataset_id = await mgr.save_prompts("router", original)
        assert dataset_id == "ds-dspy-prompts-acme-router"

        loaded = await mgr.load_prompts("router")
        assert loaded == original

    @pytest.mark.asyncio
    async def test_demonstration_round_trip(self):
        """Save demos, load them back, verify exact equality."""
        provider = _make_mock_telemetry_provider()
        mgr = ArtifactManager(provider, tenant_id="acme")

        original = [
            {"input": '{"query": "find cats"}', "output": '{"agent": "video_search"}', "metadata": "{}"},
            {"input": '{"query": "summarize"}', "output": '{"agent": "summarizer"}', "metadata": "{}"},
        ]

        await mgr.save_demonstrations("router", original)
        loaded = await mgr.load_demonstrations("router")

        assert len(loaded) == 2
        assert loaded[0]["input"] == original[0]["input"]
        assert loaded[1]["output"] == original[1]["output"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self):
        """Loading from empty store returns None, not an error."""
        provider = _make_mock_telemetry_provider()
        mgr = ArtifactManager(provider, tenant_id="acme")

        result = await mgr.load_prompts("nonexistent_agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_optimization_metrics_logged(self):
        """Optimization run metrics are logged to experiment store."""
        provider = _make_mock_telemetry_provider()
        mgr = ArtifactManager(provider, tenant_id="acme")

        run_id = await mgr.log_optimization_run("router", {
            "accuracy": 0.85,
            "latency_ms": 120,
            "num_examples": 50,
        })

        assert run_id == "run-test"
        provider.experiments.create_experiment.assert_awaited_once()
        provider.experiments.log_run.assert_awaited_once()

        # Verify the metrics were passed through
        call_kwargs = provider.experiments.log_run.call_args
        assert call_kwargs.kwargs["outputs"]["accuracy"] == 0.85


class TestTenantIsolation:
    """Verify tenant A's artifacts are invisible to tenant B."""

    @pytest.mark.asyncio
    async def test_tenant_a_invisible_to_tenant_b(self):
        """Prompts saved for tenant A should not be loadable by tenant B."""
        provider = _make_mock_telemetry_provider()

        mgr_a = ArtifactManager(provider, tenant_id="tenant_a")
        mgr_b = ArtifactManager(provider, tenant_id="tenant_b")

        await mgr_a.save_prompts("router", {"system": "Tenant A prompt"})

        # Tenant B should get None (dataset name differs)
        result_b = await mgr_b.load_prompts("router")
        assert result_b is None

        # Tenant A should get its own prompts
        result_a = await mgr_a.load_prompts("router")
        assert result_a == {"system": "Tenant A prompt"}

    @pytest.mark.asyncio
    async def test_both_tenants_coexist(self):
        """Both tenants save and load independently from the same provider."""
        provider = _make_mock_telemetry_provider()

        mgr_a = ArtifactManager(provider, tenant_id="tenant_a")
        mgr_b = ArtifactManager(provider, tenant_id="tenant_b")

        await mgr_a.save_prompts("router", {"system": "A's prompt"})
        await mgr_b.save_prompts("router", {"system": "B's prompt"})

        assert (await mgr_a.load_prompts("router")) == {"system": "A's prompt"}
        assert (await mgr_b.load_prompts("router")) == {"system": "B's prompt"}


class TestOptimizerToStrategyWiring:
    """Verify the full optimizer-saves → strategy-loads pipeline.

    This is the test for the filename mismatch bug: the old code had
    router_optimizer writing to `router_prompt_artifact.json` while
    LLMRoutingStrategy read from `routing_prompts.json`. Both now
    use dataset name `dspy-prompts-{tenant}-router`.
    """

    @pytest.mark.asyncio
    async def test_strategy_loads_optimizer_prompts(self):
        """Prompts saved by ArtifactManager are loaded by LLMRoutingStrategy."""
        provider = _make_mock_telemetry_provider()
        tenant_id = "production"

        # Simulate optimizer saving prompts
        mgr = ArtifactManager(provider, tenant_id=tenant_id)
        await mgr.save_prompts("router", {
            "system_prompt": "Optimized routing instructions here.",
            "prompt_template": "Route: {system_prompt}\n{conversation_history}\n{query}",
        })

        # Strategy loads them via the same provider + tenant
        strategy = LLMRoutingStrategy(config={
            "enable_dspy_optimization": True,
            "tenant_id": tenant_id,
            "telemetry_provider": provider,
            "model": "test-model",
        })

        # In an async context, prompts load via ensure_future + callback.
        # Give the event loop a tick to fire the callback.
        await asyncio.sleep(0.1)

        assert strategy.optimized_prompts["system_prompt"] == "Optimized routing instructions here."
        assert "prompt_template" in strategy.optimized_prompts

        # The system prompt should be the optimized one, not the default
        assert strategy.system_prompt == "Optimized routing instructions here."

        # Optimization status should reflect loaded state
        status = strategy.get_optimization_status()
        assert status["dspy_enabled"] is True
        assert status["optimized_prompts_loaded"] is True
        assert status["using_optimized_system_prompt"] is True

    def test_strategy_without_prompts_uses_default(self):
        """Strategy with no saved prompts falls back to default prompt."""
        provider = _make_mock_telemetry_provider()

        strategy = LLMRoutingStrategy(config={
            "enable_dspy_optimization": True,
            "tenant_id": "empty_tenant",
            "telemetry_provider": provider,
            "model": "test-model",
        })

        # No prompts saved → default prompt used
        assert strategy.optimized_prompts == {}
        assert "routing agent" in strategy.system_prompt.lower()

        status = strategy.get_optimization_status()
        assert status["dspy_enabled"] is True
        assert status["optimized_prompts_loaded"] is False


class TestMixinAutoLoad:
    """Verify DSPyIntegrationMixin auto-loads prompts at construction."""

    @pytest.mark.asyncio
    async def test_mixin_loads_saved_prompts_on_init(self):
        """Mixin constructor loads existing prompts and sets dspy_enabled."""
        provider = _make_mock_telemetry_provider()
        tenant_id = "test_tenant"
        agent_type = "query_analysis"

        # Pre-save prompts
        mgr = ArtifactManager(provider, tenant_id=tenant_id)
        await mgr.save_prompts(agent_type, {
            "system": "Optimized analysis prompt",
            "template": "Analyze: {query}",
        })

        # Mixin should auto-load on init
        mixin = DSPyIntegrationMixin(
            tenant_id=tenant_id,
            agent_type=agent_type,
            telemetry_provider=provider,
        )

        # Allow async callback to complete
        await asyncio.sleep(0.1)

        assert mixin.dspy_enabled is True
        assert mixin.dspy_optimized_prompts == {
            "system": "Optimized analysis prompt",
            "template": "Analyze: {query}",
        }

        # get_optimized_prompt should return the saved value
        assert mixin.get_optimized_prompt("system", "fallback") == "Optimized analysis prompt"
        assert mixin.get_optimized_prompt("missing_key", "fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_mixin_without_prompts_stays_disabled(self):
        """Mixin with no saved prompts keeps dspy_enabled=False."""
        provider = _make_mock_telemetry_provider()

        mixin = DSPyIntegrationMixin(
            tenant_id="empty_tenant",
            agent_type="query_analysis",
            telemetry_provider=provider,
        )

        await asyncio.sleep(0.1)

        assert mixin.dspy_enabled is False
        assert mixin.dspy_optimized_prompts == {}
        assert mixin.get_optimized_prompt("system", "default") == "default"


class TestOptimizerExperienceRoundTrip:
    """Verify AdvancedRoutingOptimizer persists and reloads experiences."""

    @pytest.mark.asyncio
    async def test_record_then_persist_then_reload(self):
        """Record experiences, persist via telemetry, verify they survive."""
        provider = _make_mock_telemetry_provider()
        config_kwargs = dict(
            tenant_id="test",
            llm_config=LLMEndpointConfig(model="ollama/test-model"),
            telemetry_provider=provider,
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

        # Persist to telemetry
        await optimizer._persist_data()

        # Verify demos were actually written to the store
        assert "dspy-demos-test-routing_optimizer" in provider._datasets

        # Create a NEW optimizer from the same provider — should load persisted data
        optimizer2 = AdvancedRoutingOptimizer(**config_kwargs)
        # Allow async callback to complete
        await asyncio.sleep(0.1)

        assert len(optimizer2.experiences) == 1
        assert optimizer2.experiences[0].query == "find cats in videos"
        assert optimizer2.experiences[0].chosen_agent == "video_search"
