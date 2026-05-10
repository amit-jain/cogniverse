"""Unit tests for RLM inference with query-level configuration.

Tests RLMOptions, RLMInference, and RLMAwareMixin for A/B testing support.
"""

import pytest

from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


class TestRLMOptions:
    """Test RLMOptions configuration."""

    def test_disabled_by_default(self):
        """RLM should be disabled by default."""
        opts = RLMOptions()
        assert opts.enabled is False
        assert opts.auto_detect is False
        assert opts.should_use_rlm(100_000) is False

    def test_explicit_enable(self):
        """Explicit enable should always use RLM regardless of context size."""
        opts = RLMOptions(enabled=True)
        assert opts.should_use_rlm(100) is True
        assert opts.should_use_rlm(1_000_000) is True

    def test_auto_detect_below_threshold(self):
        """Auto-detect should not enable RLM below threshold."""
        opts = RLMOptions(auto_detect=True, context_threshold=50_000)
        assert opts.should_use_rlm(30_000) is False
        assert opts.should_use_rlm(49_999) is False

    def test_auto_detect_above_threshold(self):
        """Auto-detect should enable RLM above threshold."""
        opts = RLMOptions(auto_detect=True, context_threshold=50_000)
        assert opts.should_use_rlm(50_001) is True
        assert opts.should_use_rlm(60_000) is True
        assert opts.should_use_rlm(1_000_000) is True

    def test_auto_detect_at_threshold(self):
        """Auto-detect should not enable RLM at exactly the threshold."""
        opts = RLMOptions(auto_detect=True, context_threshold=50_000)
        assert opts.should_use_rlm(50_000) is False

    def test_default_values(self):
        """Verify default configuration values."""
        opts = RLMOptions()
        assert opts.enabled is False
        assert opts.auto_detect is False
        assert opts.context_threshold == 50_000
        assert opts.max_iterations == 3
        assert opts.backend == "openai"
        assert opts.model is None

    def test_custom_configuration(self):
        """Custom configuration should be preserved."""
        opts = RLMOptions(
            enabled=True,
            max_iterations=5,
            backend="anthropic",
            model="claude-3-opus",
            context_threshold=100_000,
        )
        assert opts.enabled is True
        assert opts.max_iterations == 5
        assert opts.backend == "anthropic"
        assert opts.model == "claude-3-opus"
        assert opts.context_threshold == 100_000

    def test_max_iterations_bounds(self):
        """max_iterations should be bounded between 1 and 10."""
        opts = RLMOptions(max_iterations=1)
        assert opts.max_iterations == 1

        opts = RLMOptions(max_iterations=10)
        assert opts.max_iterations == 10

        # Invalid depths should raise validation errors
        with pytest.raises(ValueError):
            RLMOptions(max_iterations=0)

        with pytest.raises(ValueError):
            RLMOptions(max_iterations=11)

    def test_enabled_overrides_auto_detect(self):
        """Explicit enabled=True should work even with auto_detect=False."""
        opts = RLMOptions(enabled=True, auto_detect=False)
        # Should use RLM even for small context
        assert opts.should_use_rlm(100) is True


class TestRLMResult:
    """Test RLMResult dataclass."""

    def test_to_telemetry_dict(self):
        """Telemetry dict should contain expected metrics."""
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="test answer",
            depth_reached=2,
            total_calls=5,
            tokens_used=1500,
            latency_ms=2500.5,
            metadata={"extra": "data"},
        )

        telemetry = result.to_telemetry_dict()

        assert telemetry["rlm_enabled"] is True
        assert telemetry["rlm_depth_reached"] == 2
        assert telemetry["rlm_total_calls"] == 5
        assert telemetry["rlm_tokens_used"] == 1500
        assert telemetry["rlm_latency_ms"] == 2500.5
        assert telemetry["rlm_was_fallback"] is False

    def test_was_fallback_defaults_false(self):
        """RLMResult.was_fallback defaults to False (clean completion)."""
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="clean",
            depth_reached=1,
            total_calls=1,
            tokens_used=0,
            latency_ms=10.0,
        )
        assert result.was_fallback is False
        assert result.to_telemetry_dict()["rlm_was_fallback"] is False

    def test_was_fallback_propagates_to_telemetry(self):
        """When set True, telemetry dict reports rlm_was_fallback=True."""
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="best-effort",
            depth_reached=10,
            total_calls=10,
            tokens_used=0,
            latency_ms=5000.0,
            was_fallback=True,
        )
        assert result.was_fallback is True
        assert result.to_telemetry_dict()["rlm_was_fallback"] is True

    def test_trajectory_defaults_empty(self):
        """RLMResult.trajectory defaults to an empty list and reports length=0."""
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="x",
            depth_reached=1,
            total_calls=1,
            tokens_used=0,
            latency_ms=1.0,
        )
        assert result.trajectory == []
        assert result.to_telemetry_dict()["rlm_trajectory_length"] == 0

    def test_trajectory_propagates_length_to_telemetry(self):
        """Telemetry exposes trajectory length, useful for A/B comparison."""
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="x",
            depth_reached=3,
            total_calls=3,
            tokens_used=0,
            latency_ms=1.0,
            trajectory=[
                {"iteration": 1, "reasoning": "r1", "code": "c1"},
                {"iteration": 2, "reasoning": "r2", "code": "c2"},
                {"iteration": 3, "reasoning": "r3", "code": "c3"},
            ],
        )
        assert len(result.trajectory) == 3
        assert result.to_telemetry_dict()["rlm_trajectory_length"] == 3


class TestSerializeTrajectory:
    """Unit tests for the trajectory serializer used by RLMInference.process."""

    def test_empty_input_returns_empty_list(self):
        from cogniverse_agents.inference.rlm_inference import _serialize_trajectory

        assert _serialize_trajectory([], 32) == []
        assert _serialize_trajectory(None, 32) == []

    def test_caps_entries_at_max(self):
        from types import SimpleNamespace

        from cogniverse_agents.inference.rlm_inference import _serialize_trajectory

        raw = [SimpleNamespace(reasoning=f"r{i}", code=f"c{i}") for i in range(50)]
        out = _serialize_trajectory(raw, max_entries=10)
        assert len(out) == 10
        assert out[0]["iteration"] == 1
        assert out[9]["iteration"] == 10
        assert out[0]["reasoning"] == "r0"
        assert out[0]["code"] == "c0"

    def test_truncates_long_field_values(self):
        from types import SimpleNamespace

        from cogniverse_agents.inference.rlm_inference import _serialize_trajectory

        long_text = "x" * 5000
        raw = [SimpleNamespace(reasoning=long_text, code=long_text)]
        out = _serialize_trajectory(raw, max_entries=32)
        assert out[0]["reasoning"].endswith("…")
        assert len(out[0]["reasoning"]) < 5000
        assert len(out[0]["code"]) < 5000

    def test_skips_missing_fields(self):
        from types import SimpleNamespace

        from cogniverse_agents.inference.rlm_inference import _serialize_trajectory

        raw = [SimpleNamespace(reasoning="r1")]  # no code, no observation
        out = _serialize_trajectory(raw, max_entries=32)
        assert "reasoning" in out[0]
        assert "code" not in out[0]
        assert "observation" not in out[0]


class TestSumTrackerTokens:
    """Unit tests for the DSPy-tracker token aggregator used by process()."""

    def test_none_tracker_returns_zero(self):
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        assert _sum_tracker_tokens(None) == 0

    def test_uses_explicit_total_tokens_when_present(self):
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        class FakeTracker:
            usage_data = {
                "openai/test-model": [
                    {
                        "total_tokens": 150,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    },
                    {
                        "total_tokens": 220,
                        "prompt_tokens": 150,
                        "completion_tokens": 70,
                    },
                ]
            }

        assert _sum_tracker_tokens(FakeTracker()) == 370

    def test_falls_back_to_prompt_plus_completion(self):
        """When backend doesn't report total_tokens, sum prompt+completion."""
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        class FakeTracker:
            usage_data = {
                "openai/gpt-4o": [
                    {"prompt_tokens": 100, "completion_tokens": 50},  # = 150
                    {"prompt_tokens": 30, "completion_tokens": 20},  # = 50
                ]
            }

        assert _sum_tracker_tokens(FakeTracker()) == 200

    def test_sums_across_multiple_lms(self):
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        class FakeTracker:
            usage_data = {
                "openai/test-model": [{"total_tokens": 100}],
                "openai/gpt-4o": [
                    {"prompt_tokens": 50, "completion_tokens": 50},
                ],
            }

        assert _sum_tracker_tokens(FakeTracker()) == 200

    def test_handles_missing_keys_gracefully(self):
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        class FakeTracker:
            usage_data = {
                "weird/backend": [
                    {},  # empty entry
                    {"prompt_tokens": None, "completion_tokens": 10},  # None handling
                ]
            }

        # Should not raise; should add only the well-formed parts.
        assert _sum_tracker_tokens(FakeTracker()) == 10

    def test_zero_explicit_falls_back_to_components(self):
        """An explicit total_tokens of 0 must not block component-sum fallback."""
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        class FakeTracker:
            usage_data = {
                "x/y": [{"total_tokens": 0, "prompt_tokens": 5, "completion_tokens": 7}]
            }

        assert _sum_tracker_tokens(FakeTracker()) == 12


class TestRLMOptionsTrajectoryFields:
    """RLMOptions exposes the include_trajectory + trajectory_max_entries knobs."""

    def test_defaults(self):
        from cogniverse_core.agents.rlm_options import RLMOptions

        opts = RLMOptions()
        assert opts.include_trajectory is False
        assert opts.trajectory_max_entries == 32

    def test_custom_values(self):
        from cogniverse_core.agents.rlm_options import RLMOptions

        opts = RLMOptions(include_trajectory=True, trajectory_max_entries=8)
        assert opts.include_trajectory is True
        assert opts.trajectory_max_entries == 8

    def test_trajectory_max_entries_bounds(self):
        from cogniverse_core.agents.rlm_options import RLMOptions

        with pytest.raises(Exception):  # pydantic ValidationError
            RLMOptions(trajectory_max_entries=0)
        with pytest.raises(Exception):
            RLMOptions(trajectory_max_entries=999)

    def test_result_fields(self):
        """RLMResult should store all fields correctly."""
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="complex answer",
            depth_reached=3,
            total_calls=10,
            tokens_used=5000,
            latency_ms=10000.0,
            metadata={"context_size": 100000},
        )

        assert result.answer == "complex answer"
        assert result.depth_reached == 3
        assert result.total_calls == 10
        assert result.tokens_used == 5000
        assert result.latency_ms == 10000.0
        assert result.metadata == {"context_size": 100000}


class TestRLMAwareMixin:
    """Test RLMAwareMixin with query config."""

    def test_should_use_rlm_none_options(self):
        """None options should return False."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()
        assert mixin.should_use_rlm_for_query(None, "any context") is False

    def test_should_use_rlm_enabled(self):
        """Enabled options should return True."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()
        opts = RLMOptions(enabled=True)
        assert mixin.should_use_rlm_for_query(opts, "short context") is True

    def test_should_use_rlm_auto_detect(self):
        """Auto-detect should respect threshold."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()
        opts = RLMOptions(auto_detect=True, context_threshold=100)

        # Below threshold
        assert mixin.should_use_rlm_for_query(opts, "x" * 50) is False

        # Above threshold
        assert mixin.should_use_rlm_for_query(opts, "x" * 150) is True

    def test_get_rlm_telemetry_with_result(self):
        """Telemetry should include RLM metrics when result is provided."""
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()
        result = RLMResult(
            answer="test",
            depth_reached=2,
            total_calls=3,
            tokens_used=500,
            latency_ms=1000,
        )

        telemetry = mixin.get_rlm_telemetry(result, context_size=10000)

        assert telemetry["rlm_enabled"] is True
        assert telemetry["rlm_depth_reached"] == 2
        assert telemetry["rlm_total_calls"] == 3
        assert telemetry["context_size_chars"] == 10000

    def test_get_rlm_telemetry_without_result(self):
        """Telemetry should indicate RLM disabled when no result."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()

        telemetry = mixin.get_rlm_telemetry(None, context_size=5000)

        assert telemetry["rlm_enabled"] is False
        assert telemetry["context_size_chars"] == 5000


class TestRLMInference:
    """Test RLMInference wrapper."""

    def test_init_stores_config(self):
        """Initialization should store configuration."""
        from cogniverse_agents.inference.rlm_inference import RLMInference

        rlm = RLMInference(
            llm_config=LLMEndpointConfig(model="anthropic/claude-3-opus"),
            max_iterations=5,
            max_llm_calls=20,
        )

        assert rlm.model == "anthropic/claude-3-opus"
        assert rlm.max_iterations == 5
        assert rlm.max_llm_calls == 20
        assert rlm._rlm is None  # Lazy initialization

    def test_configure_lm_litellm_provider(self):
        """Should configure DSPy LM for any litellm provider prefix."""
        from cogniverse_agents.inference.rlm_inference import RLMInference

        rlm = RLMInference(llm_config=LLMEndpointConfig(model="openai/test-model"))

        assert rlm.model == "openai/test-model"


class TestSearchInputWithRLM:
    """Test SearchInput integration with RLM options."""

    def test_search_input_rlm_none_by_default(self):
        """SearchInput should have rlm=None by default."""
        from cogniverse_agents.search_agent import SearchInput

        input_data = SearchInput(query="test query", tenant_id="test_tenant")
        assert input_data.rlm is None

    def test_search_input_with_rlm_options(self):
        """SearchInput should accept RLMOptions."""
        from cogniverse_agents.search_agent import SearchInput

        rlm_opts = RLMOptions(enabled=True, max_iterations=5)
        input_data = SearchInput(
            query="test query", tenant_id="test_tenant", rlm=rlm_opts
        )

        assert input_data.rlm is not None
        assert input_data.rlm.enabled is True
        assert input_data.rlm.max_iterations == 5

    def test_search_input_serialization(self):
        """SearchInput with RLM should serialize correctly."""
        from cogniverse_agents.search_agent import SearchInput

        rlm_opts = RLMOptions(enabled=True)
        input_data = SearchInput(query="test", tenant_id="test_tenant", rlm=rlm_opts)

        # Convert to dict
        data_dict = input_data.model_dump()
        assert data_dict["rlm"]["enabled"] is True

        # Reconstruct from dict
        reconstructed = SearchInput.model_validate(data_dict)
        assert reconstructed.rlm.enabled is True
