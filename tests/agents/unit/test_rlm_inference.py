"""Unit tests for RLM inference with query-level configuration.

Tests RLMOptions, RLMInference, and RLMAwareMixin for A/B testing support.
"""

import pytest
from cogniverse_core.agents.rlm_options import RLMOptions


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
        assert opts.max_depth == 3
        assert opts.backend == "openai"
        assert opts.model is None

    def test_custom_configuration(self):
        """Custom configuration should be preserved."""
        opts = RLMOptions(
            enabled=True,
            max_depth=5,
            backend="anthropic",
            model="claude-3-opus",
            context_threshold=100_000,
        )
        assert opts.enabled is True
        assert opts.max_depth == 5
        assert opts.backend == "anthropic"
        assert opts.model == "claude-3-opus"
        assert opts.context_threshold == 100_000

    def test_max_depth_bounds(self):
        """Max depth should be bounded between 1 and 10."""
        # Valid depths
        opts = RLMOptions(max_depth=1)
        assert opts.max_depth == 1

        opts = RLMOptions(max_depth=10)
        assert opts.max_depth == 10

        # Invalid depths should raise validation errors
        with pytest.raises(ValueError):
            RLMOptions(max_depth=0)

        with pytest.raises(ValueError):
            RLMOptions(max_depth=11)

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
            backend="anthropic",
            model="claude-3-opus",
            sandbox="docker",
            max_iterations=5,
            max_llm_calls=20,
        )

        assert rlm.backend == "anthropic"
        assert rlm.model == "claude-3-opus"
        assert rlm.sandbox == "docker"
        assert rlm.max_iterations == 5
        assert rlm.max_depth == 5  # Backward compat property
        assert rlm.max_llm_calls == 20
        assert rlm._rlm is None  # Lazy initialization

    def test_configure_lm_ollama(self):
        """Should configure DSPy LM for Ollama backend."""
        from cogniverse_agents.inference.rlm_inference import RLMInference

        rlm = RLMInference(backend="ollama", model="qwen2.5:1.5b")

        assert rlm.backend == "ollama"
        assert rlm.model == "qwen2.5:1.5b"


class TestSearchInputWithRLM:
    """Test SearchInput integration with RLM options."""

    def test_search_input_rlm_none_by_default(self):
        """SearchInput should have rlm=None by default."""
        from cogniverse_agents.search_agent import SearchInput

        input_data = SearchInput(query="test query")
        assert input_data.rlm is None

    def test_search_input_with_rlm_options(self):
        """SearchInput should accept RLMOptions."""
        from cogniverse_agents.search_agent import SearchInput

        rlm_opts = RLMOptions(enabled=True, max_depth=5)
        input_data = SearchInput(query="test query", rlm=rlm_opts)

        assert input_data.rlm is not None
        assert input_data.rlm.enabled is True
        assert input_data.rlm.max_depth == 5

    def test_search_input_serialization(self):
        """SearchInput with RLM should serialize correctly."""
        from cogniverse_agents.search_agent import SearchInput

        rlm_opts = RLMOptions(enabled=True)
        input_data = SearchInput(query="test", rlm=rlm_opts)

        # Convert to dict
        data_dict = input_data.model_dump()
        assert data_dict["rlm"]["enabled"] is True

        # Reconstruct from dict
        reconstructed = SearchInput.model_validate(data_dict)
        assert reconstructed.rlm.enabled is True
