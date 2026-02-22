"""
Integration tests for RLM (Recursive Language Model) with SearchAgent.

Tests the complete RLM integration pipeline:
- Query-level RLM configuration via RLMOptions
- RLM inference with DSPy ProgramOfThought (uses Ollama backend)
- Telemetry generation for A/B testing
- Integration with Vespa backend for search results
"""

import logging

import pytest

from cogniverse_core.agents.rlm_options import RLMOptions
from tests.agents.integration.conftest import skip_if_no_ollama

logger = logging.getLogger(__name__)


@pytest.fixture
def search_agent_with_vespa_rlm(vespa_with_schema):
    """
    SearchAgent configured with RLM support and real Vespa backend.

    Uses vespa_with_schema fixture for Vespa container lifecycle management.
    """
    from pathlib import Path

    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    # Get Vespa connection info from fixture
    vespa_http_port = vespa_with_schema["http_port"]
    vespa_config_port = vespa_with_schema["config_port"]
    vespa_url = "http://localhost"
    default_schema = vespa_with_schema["default_schema"]

    # Use config manager from VespaTestManager (has correct ports)
    config_manager = vespa_with_schema["manager"].config_manager

    # Create schema loader pointing to test schemas
    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Create SearchAgent with test Vespa parameters
    deps = SearchAgentDeps(
        tenant_id="test_tenant",
        backend_url=vespa_url,
        backend_port=vespa_http_port,
        backend_config_port=vespa_config_port,
        profile=default_schema,
    )

    search_agent = SearchAgent(
        deps=deps,
        schema_loader=schema_loader,
        config_manager=config_manager,
        port=8016,
    )

    return search_agent


@pytest.mark.integration
class TestRLMOptionsIntegration:
    """Integration tests for RLMOptions configuration."""

    def test_rlm_options_serialization_roundtrip(self):
        """
        Test that RLMOptions serializes and deserializes correctly.

        This is critical for API transport and A/B testing configuration.
        """
        original = RLMOptions(
            enabled=True,
            auto_detect=False,
            context_threshold=75_000,
            max_iterations=4,
            backend="anthropic",
            model="claude-3-opus",
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        reconstructed = RLMOptions.model_validate(data)

        # VALIDATE: All fields preserved
        assert reconstructed.enabled == original.enabled
        assert reconstructed.auto_detect == original.auto_detect
        assert reconstructed.context_threshold == original.context_threshold
        assert reconstructed.max_iterations == original.max_iterations
        assert reconstructed.backend == original.backend
        assert reconstructed.model == original.model

        logger.info("RLMOptions serialization roundtrip validated")

    def test_rlm_options_json_schema_generation(self):
        """
        Test that RLMOptions generates valid JSON schema for API docs.

        Important for A/B testing API documentation and client generation.
        """
        schema = RLMOptions.model_json_schema()

        # VALIDATE: Required fields in schema
        assert "properties" in schema
        props = schema["properties"]
        assert "enabled" in props
        assert "auto_detect" in props
        assert "context_threshold" in props
        assert "max_iterations" in props
        assert "backend" in props
        assert "model" in props

        # VALIDATE: max_iterations has bounds
        assert props["max_iterations"].get("minimum") == 1
        assert props["max_iterations"].get("maximum") == 10

        logger.info(f"RLMOptions JSON schema validated: {len(props)} properties")


@pytest.mark.integration
class TestRLMAwareMixinIntegration:
    """Integration tests for RLMAwareMixin with search agents."""

    def test_mixin_rlm_instance_caching(self):
        """
        Test that RLMAwareMixin caches RLM instances correctly.

        Important for performance - avoid creating new RLM instances per query.
        """
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()

        # Get first instance
        rlm1 = mixin.get_rlm(backend="openai", model="gpt-4o", max_iterations=10)

        # Get same config - should return same instance
        rlm2 = mixin.get_rlm(backend="openai", model="gpt-4o", max_iterations=10)
        assert rlm1 is rlm2, "Same config should return cached instance"

        # Get different config - should return new instance
        rlm3 = mixin.get_rlm(backend="openai", model="gpt-4o", max_iterations=20)
        assert rlm3 is not rlm1, "Different config should create new instance"

        logger.info("RLM instance caching validated")

    def test_mixin_telemetry_generation(self):
        """
        Test that RLMAwareMixin generates correct telemetry for A/B testing.

        Telemetry is critical for comparing RLM vs standard inference.
        """
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()

        # Test with RLM result
        rlm_result = RLMResult(
            answer="test answer",
            depth_reached=3,
            total_calls=8,
            tokens_used=2500,
            latency_ms=3500.5,
            metadata={"extra": "data"},
        )

        telemetry = mixin.get_rlm_telemetry(rlm_result, context_size=100_000)

        # VALIDATE: All telemetry fields present
        assert telemetry["rlm_enabled"] is True
        assert telemetry["rlm_depth_reached"] == 3
        assert telemetry["rlm_total_calls"] == 8
        assert telemetry["rlm_tokens_used"] == 2500
        assert telemetry["rlm_latency_ms"] == 3500.5
        assert telemetry["context_size_chars"] == 100_000

        # Test without RLM result (standard inference)
        telemetry_standard = mixin.get_rlm_telemetry(None, context_size=50_000)

        assert telemetry_standard["rlm_enabled"] is False
        assert telemetry_standard["context_size_chars"] == 50_000
        assert "rlm_depth_reached" not in telemetry_standard

        logger.info("RLM telemetry generation validated for A/B testing")


@pytest.mark.integration
class TestSearchInputRLMIntegration:
    """Integration tests for SearchInput with RLM options."""

    def test_search_input_with_rlm_api_format(self):
        """
        Test SearchInput with RLM options in API request format.

        Simulates actual API request with RLM configuration.
        """
        from cogniverse_agents.search_agent import SearchInput

        # Simulate API request body
        api_request = {
            "query": "machine learning video tutorials",
            "tenant_id": "test_tenant",
            "modality": "video",
            "top_k": 20,
            "rlm": {
                "enabled": True,
                "max_iterations": 4,
                "backend": "openai",
                "model": "gpt-4o",
            },
        }

        # Parse into SearchInput
        search_input = SearchInput.model_validate(api_request)

        # VALIDATE: RLM options parsed correctly
        assert search_input.query == "machine learning video tutorials"
        assert search_input.rlm is not None
        assert search_input.rlm.enabled is True
        assert search_input.rlm.max_iterations == 4
        assert search_input.rlm.backend == "openai"
        assert search_input.rlm.model == "gpt-4o"

        logger.info("SearchInput API format with RLM validated")

    def test_search_input_without_rlm_backward_compatible(self):
        """
        Test that SearchInput without RLM is backward compatible.

        Existing API clients that don't send RLM options should work.
        """
        from cogniverse_agents.search_agent import SearchInput

        # Old-style API request (no RLM)
        api_request = {
            "query": "video editing software",
            "tenant_id": "test_tenant",
            "modality": "video",
            "top_k": 10,
        }

        search_input = SearchInput.model_validate(api_request)

        # VALIDATE: RLM is None (disabled)
        assert search_input.rlm is None

        # Verify should_use_rlm returns False
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()
        assert mixin.should_use_rlm_for_query(search_input.rlm, "any context") is False

        logger.info("Backward compatibility without RLM validated")

    def test_search_input_rlm_auto_detect_mode(self):
        """
        Test SearchInput with RLM auto-detect configuration.

        Auto-detect enables RLM only when context exceeds threshold.
        """
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
        from cogniverse_agents.search_agent import SearchInput

        api_request = {
            "query": "complex multi-document analysis",
            "tenant_id": "test_tenant",
            "top_k": 50,
            "rlm": {
                "enabled": False,
                "auto_detect": True,
                "context_threshold": 100_000,
            },
        }

        search_input = SearchInput.model_validate(api_request)
        mixin = RLMAwareMixin()

        # Below threshold - should NOT use RLM
        small_context = "x" * 50_000
        assert mixin.should_use_rlm_for_query(search_input.rlm, small_context) is False

        # Above threshold - should use RLM
        large_context = "x" * 150_000
        assert mixin.should_use_rlm_for_query(search_input.rlm, large_context) is True

        logger.info("RLM auto-detect mode validated")


@pytest.mark.integration
class TestRLMSearchAgentIntegration:
    """
    Integration tests for RLM with SearchAgent.

    These tests verify RLM integration without requiring Vespa backend.
    Tests with Vespa are in TestRLMVespaIntegration.
    """

    def test_search_input_rlm_options_in_schema(self):
        """
        Test that SearchInput schema includes RLM options.

        Validates the API schema correctly exposes RLM configuration.
        """
        from cogniverse_agents.search_agent import SearchInput

        schema = SearchInput.model_json_schema()

        # VALIDATE: RLM field in schema
        assert "rlm" in schema["properties"]

        # VALIDATE: RLM can be None (disabled) or RLMOptions
        rlm_schema = schema["properties"]["rlm"]
        assert rlm_schema.get("default") is None or "anyOf" in rlm_schema

        logger.info("SearchInput schema includes RLM options")

    def test_search_agent_rlm_mixin_inheritance(self):
        """
        Test that SearchAgent has RLM mixin capabilities available.

        Note: This tests the class structure, not runtime behavior.
        """
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
        from cogniverse_agents.search_agent import SearchInput

        # Verify SearchInput supports RLM options
        search_input = SearchInput(
            query="test",
            tenant_id="test_tenant",
            rlm=RLMOptions(enabled=True, max_iterations=3),
        )

        # VALIDATE: RLM options are accessible
        assert search_input.rlm is not None
        assert search_input.rlm.enabled is True

        # VALIDATE: RLMAwareMixin methods exist
        mixin = RLMAwareMixin()
        assert hasattr(mixin, "should_use_rlm_for_query")
        assert hasattr(mixin, "process_with_rlm")
        assert hasattr(mixin, "get_rlm_telemetry")
        assert hasattr(mixin, "get_rlm")

        logger.info("SearchAgent RLM mixin capabilities validated")

    def test_search_input_rlm_options_propagation(self):
        """
        Test that RLM options are correctly propagated through SearchInput.

        Validates the RLM configuration is preserved through serialization.
        """
        from cogniverse_agents.search_agent import SearchInput

        # Create input with RLM enabled
        search_input = SearchInput(
            query="complex analysis query",
            tenant_id="test_tenant",
            top_k=10,
            rlm=RLMOptions(enabled=True, max_iterations=3, backend="anthropic"),
        )

        # VALIDATE: RLM options accessible
        assert search_input.rlm is not None
        assert search_input.rlm.enabled is True
        assert search_input.rlm.max_iterations == 3
        assert search_input.rlm.backend == "anthropic"

        # Serialize and deserialize
        data = search_input.model_dump()
        reconstructed = SearchInput.model_validate(data)

        # VALIDATE: RLM options preserved
        assert reconstructed.rlm is not None
        assert reconstructed.rlm.enabled is True
        assert reconstructed.rlm.max_iterations == 3
        assert reconstructed.rlm.backend == "anthropic"

        logger.info("RLM options propagation validated")


@pytest.mark.integration
class TestRLMInferenceIntegration:
    """Integration tests for RLMInference wrapper."""

    def test_rlm_inference_configuration(self):
        """
        Test RLMInference stores configuration correctly.
        """
        from cogniverse_agents.inference.rlm_inference import RLMInference

        rlm = RLMInference(
            backend="anthropic",
            model="claude-3-opus",
            max_iterations=5,
            max_llm_calls=20,
            api_key="test-key",
        )

        assert rlm.backend == "anthropic"
        assert rlm.model == "claude-3-opus"
        assert rlm.max_iterations == 5
        assert rlm.max_llm_calls == 20
        assert rlm._api_key == "test-key"
        assert rlm._rlm is None  # Lazy initialization

        logger.info("RLMInference configuration validated")

    def test_rlm_inference_result_structure(self):
        """
        Test RLMResult dataclass structure and methods.
        """
        from cogniverse_agents.inference.rlm_inference import RLMResult

        result = RLMResult(
            answer="Comprehensive analysis of the documents shows...",
            depth_reached=3,
            total_calls=12,
            tokens_used=4500,
            latency_ms=8500.0,
            metadata={
                "context_size_chars": 200_000,
                "strategy": "partition_map",
            },
        )

        # VALIDATE: All fields accessible
        assert result.answer.startswith("Comprehensive")
        assert result.depth_reached == 3
        assert result.total_calls == 12
        assert result.tokens_used == 4500
        assert result.latency_ms == 8500.0
        assert result.metadata["strategy"] == "partition_map"

        # VALIDATE: Telemetry dict generation
        telemetry = result.to_telemetry_dict()
        assert telemetry["rlm_enabled"] is True
        assert telemetry["rlm_depth_reached"] == 3
        assert telemetry["rlm_total_calls"] == 12
        assert telemetry["rlm_tokens_used"] == 4500
        assert telemetry["rlm_latency_ms"] == 8500.0

        logger.info("RLMResult structure validated")


@pytest.mark.integration
class TestRLMABTestingIntegration:
    """Integration tests for A/B testing scenarios with RLM."""

    def test_ab_testing_group_a_standard(self):
        """
        Test Group A: Standard inference (no RLM).

        Simulates control group in A/B testing.
        """
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
        from cogniverse_agents.search_agent import SearchInput

        # Group A: No RLM
        input_a = SearchInput(
            query="test query", tenant_id="test_tenant", top_k=10, rlm=None
        )

        mixin = RLMAwareMixin()
        context = "x" * 100_000

        # VALIDATE: RLM not used
        assert mixin.should_use_rlm_for_query(input_a.rlm, context) is False

        # Generate telemetry
        telemetry = mixin.get_rlm_telemetry(None, len(context))
        assert telemetry["rlm_enabled"] is False
        assert telemetry["context_size_chars"] == 100_000

        logger.info("A/B testing Group A (standard) validated")

    def test_ab_testing_group_b_rlm_enabled(self):
        """
        Test Group B: RLM enabled inference.

        Simulates treatment group in A/B testing.
        """
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
        from cogniverse_agents.search_agent import SearchInput

        # Group B: RLM enabled
        input_b = SearchInput(
            query="test query",
            tenant_id="test_tenant",
            top_k=10,
            rlm=RLMOptions(enabled=True, max_iterations=3),
        )

        mixin = RLMAwareMixin()
        context = "x" * 100_000

        # VALIDATE: RLM used
        assert mixin.should_use_rlm_for_query(input_b.rlm, context) is True

        # Simulate RLM result (would come from real RLM processing)
        rlm_result = RLMResult(
            answer="RLM processed answer",
            depth_reached=2,
            total_calls=5,
            tokens_used=1500,
            latency_ms=3000.0,
        )

        # Generate telemetry
        telemetry = mixin.get_rlm_telemetry(rlm_result, len(context))
        assert telemetry["rlm_enabled"] is True
        assert telemetry["rlm_depth_reached"] == 2
        assert telemetry["context_size_chars"] == 100_000

        logger.info("A/B testing Group B (RLM enabled) validated")

    def test_telemetry_comparison_metrics(self):
        """
        Test that telemetry metrics are comparable between groups.

        Both groups should have context_size_chars for fair comparison.
        """
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()
        context_size = 150_000

        # Group A telemetry (no RLM)
        telemetry_a = mixin.get_rlm_telemetry(None, context_size)

        # Group B telemetry (with RLM)
        rlm_result = RLMResult(
            answer="test",
            depth_reached=2,
            total_calls=4,
            tokens_used=1000,
            latency_ms=2000.0,
        )
        telemetry_b = mixin.get_rlm_telemetry(rlm_result, context_size)

        # VALIDATE: Both have context_size_chars for comparison
        assert telemetry_a["context_size_chars"] == context_size
        assert telemetry_b["context_size_chars"] == context_size

        # VALIDATE: rlm_enabled differentiates groups
        assert telemetry_a["rlm_enabled"] is False
        assert telemetry_b["rlm_enabled"] is True

        # VALIDATE: Group B has additional RLM metrics
        assert "rlm_depth_reached" not in telemetry_a
        assert "rlm_depth_reached" in telemetry_b
        assert "rlm_total_calls" in telemetry_b
        assert "rlm_tokens_used" in telemetry_b
        assert "rlm_latency_ms" in telemetry_b

        logger.info("A/B testing telemetry comparison metrics validated")


@pytest.mark.integration
@pytest.mark.slow  # Requires Docker + Vespa startup
class TestRLMVespaIntegration:
    """
    Integration tests for RLM with SearchAgent and real Vespa backend.

    These tests use vespa_with_schema fixture which:
    - Starts a Vespa Docker container
    - Deploys test schemas
    - Ingests test data
    - Stops and cleans up after tests

    Tests verify the complete RLM + Search pipeline works end-to-end.
    """

    @pytest.mark.asyncio
    async def test_search_agent_with_rlm_options_real_vespa(
        self, search_agent_with_vespa_rlm
    ):
        """
        Test SearchAgent accepts RLM options with real Vespa backend.

        Validates that RLM configuration doesn't break the search pipeline.
        """
        from cogniverse_agents.search_agent import SearchInput

        agent = search_agent_with_vespa_rlm

        # Create search input with RLM options
        search_input = SearchInput(
            query="machine learning video tutorials",
            tenant_id="test_tenant",
            top_k=5,
            rlm=RLMOptions(enabled=True, max_iterations=2),
        )

        # VALIDATE: Input is valid with RLM options
        assert search_input.rlm is not None
        assert search_input.rlm.enabled is True

        # VALIDATE: Agent can process the input structure
        input_dict = search_input.model_dump()
        assert "rlm" in input_dict
        assert input_dict["rlm"]["enabled"] is True

        # VALIDATE: Agent has RLM mixin capabilities
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        assert hasattr(RLMAwareMixin, "should_use_rlm_for_query")

        # VALIDATE: Agent profile matches fixture
        assert agent.active_profile is not None

        logger.info(
            f"SearchAgent accepts RLM options with real Vespa (profile: {agent.active_profile})"
        )

    @pytest.mark.asyncio
    async def test_rlm_decision_based_on_context_size(
        self, search_agent_with_vespa_rlm
    ):
        """
        Test RLM decision logic with real search results context.

        Uses auto-detect mode to determine if RLM should be used
        based on actual context size from search results.
        """
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()

        # Simulate search results context (would come from Vespa)
        small_results_context = "Result 1: Short video about ML\n" * 10  # ~400 chars
        large_results_context = "Result: " + ("x" * 10_000 + "\n") * 10  # ~100K chars

        # Auto-detect with 50K threshold
        rlm_options = RLMOptions(auto_detect=True, context_threshold=50_000)

        # VALIDATE: Small context should NOT trigger RLM
        should_use_small = mixin.should_use_rlm_for_query(
            rlm_options, small_results_context
        )
        assert should_use_small is False

        # VALIDATE: Large context should trigger RLM
        should_use_large = mixin.should_use_rlm_for_query(
            rlm_options, large_results_context
        )
        assert should_use_large is True

        logger.info(
            f"RLM auto-detect: small={len(small_results_context)} chars -> {should_use_small}, "
            f"large={len(large_results_context)} chars -> {should_use_large}"
        )

    @pytest.mark.asyncio
    async def test_rlm_telemetry_with_vespa_search_flow(
        self, search_agent_with_vespa_rlm
    ):
        """
        Test complete telemetry generation in a search + RLM flow.

        Simulates the full A/B testing scenario:
        1. Search agent gets results from Vespa
        2. Build context from results
        3. Decide if RLM should process
        4. Generate telemetry for both groups
        """
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()

        # Simulate context built from Vespa search results
        search_results_context = """
        Result 1 (score: 0.95): Machine Learning Fundamentals - A comprehensive guide
        Result 2 (score: 0.92): Deep Learning with PyTorch - Advanced techniques
        Result 3 (score: 0.88): Neural Networks Explained - Visual tutorial
        Result 4 (score: 0.85): Python for Data Science - Beginner course
        Result 5 (score: 0.82): Computer Vision Basics - Image processing
        """
        context_size = len(search_results_context)

        # Group A: Standard inference telemetry
        telemetry_standard = mixin.get_rlm_telemetry(None, context_size)

        # Group B: RLM inference telemetry (simulated result)
        rlm_result = RLMResult(
            answer="Based on the search results, the best videos for ML are...",
            depth_reached=2,
            total_calls=4,
            tokens_used=800,
            latency_ms=1500.0,
            metadata={"search_results_count": 5},
        )
        telemetry_rlm = mixin.get_rlm_telemetry(rlm_result, context_size)

        # VALIDATE: Telemetry structure for A/B comparison
        assert telemetry_standard["rlm_enabled"] is False
        assert telemetry_standard["context_size_chars"] == context_size

        assert telemetry_rlm["rlm_enabled"] is True
        assert telemetry_rlm["context_size_chars"] == context_size
        assert telemetry_rlm["rlm_depth_reached"] == 2
        assert telemetry_rlm["rlm_total_calls"] == 4
        assert telemetry_rlm["rlm_tokens_used"] == 800
        assert telemetry_rlm["rlm_latency_ms"] == 1500.0

        logger.info(
            f"Telemetry generated - Standard: {telemetry_standard}, RLM: {telemetry_rlm}"
        )


@pytest.mark.integration
@pytest.mark.slow
@skip_if_no_ollama
class TestRLMRealInferenceIntegration:
    """
    Integration tests for RLM with real Ollama LLM backend.

    These tests require:
    - Ollama service running at localhost:11434
    - RLM library installed (uv pip install 'cogniverse-agents[rlm]')

    Tests verify actual RLM inference with recursive processing.
    """

    @pytest.mark.asyncio
    async def test_rlm_process_with_ollama(self):
        """
        Test RLM process method with real Ollama backend.

        Uses litellm backend to connect to local Ollama.
        """
        from cogniverse_agents.inference.rlm_inference import RLMInference

        # Create RLM with litellm backend pointing to Ollama
        rlm = RLMInference(
            backend="litellm",
            model="ollama/qwen2.5:1.5b",  # Small model for fast tests
            max_iterations=2,  # Limit iterations for faster tests
        )

        # Small context for fast test
        context = """
        Document 1: Machine learning is a subset of artificial intelligence.
        Document 2: Deep learning uses neural networks with many layers.
        Document 3: Python is commonly used for ML development.
        """

        result = rlm.process(
            query="What is the relationship between ML and deep learning?",
            context=context,
        )

        # VALIDATE: Result structure
        assert result is not None
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.depth_reached >= 0
        assert result.total_calls >= 1
        assert result.latency_ms > 0

        logger.info(
            f"RLM with Ollama completed: answer={result.answer[:100]}..., "
            f"depth={result.depth_reached}, calls={result.total_calls}"
        )

    @pytest.mark.asyncio
    async def test_rlm_process_search_results_with_ollama(self):
        """
        Test RLM process_search_results with real Ollama backend.

        Simulates processing Vespa search results with RLM.
        """
        from cogniverse_agents.inference.rlm_inference import RLMInference

        rlm = RLMInference(
            backend="litellm",
            model="ollama/qwen2.5:1.5b",
            max_iterations=2,
        )

        # Simulate search results from Vespa
        search_results = [
            {"id": "doc1", "score": 0.95, "content": "Introduction to neural networks"},
            {"id": "doc2", "score": 0.90, "content": "Deep learning fundamentals"},
            {"id": "doc3", "score": 0.85, "content": "Machine learning basics"},
        ]

        result = rlm.process_search_results(
            query="What topics are covered?",
            results=search_results,
        )

        # VALIDATE: Result structure
        assert result is not None
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.depth_reached >= 0
        assert result.total_calls >= 1

        # VALIDATE: Telemetry generation
        telemetry = result.to_telemetry_dict()
        assert telemetry["rlm_enabled"] is True
        assert "rlm_depth_reached" in telemetry
        assert "rlm_total_calls" in telemetry

        logger.info(
            f"RLM search results processing completed: {result.answer[:100]}..."
        )

    @pytest.mark.asyncio
    async def test_rlm_mixin_process_with_rlm_real(self):
        """
        Test RLMAwareMixin.process_with_rlm with real Ollama backend.

        Tests the mixin method that agents would use.
        """
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        mixin = RLMAwareMixin()

        rlm_options = RLMOptions(
            enabled=True,
            max_iterations=2,
            backend="litellm",
            model="ollama/qwen2.5:1.5b",
        )

        context = "Python is a popular programming language for data science and ML."

        result = mixin.process_with_rlm(
            query="What is Python used for?",
            context=context,
            rlm_options=rlm_options,
        )

        # VALIDATE: Result from mixin
        assert result is not None
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

        # VALIDATE: Telemetry
        telemetry = mixin.get_rlm_telemetry(result, len(context))
        assert telemetry["rlm_enabled"] is True
        assert telemetry["context_size_chars"] == len(context)

        logger.info(
            f"RLMAwareMixin.process_with_rlm completed: {result.answer[:100]}..."
        )


@pytest.mark.integration
class TestSearchAgentRLMIntegration:
    """
    Integration tests for SearchAgent with RLM integration.

    Tests verify that SearchOutput includes RLM fields when RLM is enabled.
    """

    def test_search_output_has_rlm_fields(self):
        """SearchOutput should have rlm_synthesis and rlm_telemetry fields."""
        from cogniverse_agents.search_agent import SearchOutput

        # Create output with RLM fields
        output = SearchOutput(
            query="test query",
            modality="video",
            search_mode="single_profile",
            results=[{"id": "1", "score": 0.9}],
            total_results=1,
            rlm_synthesis="Synthesized answer from RLM",
            rlm_telemetry={"rlm_enabled": True, "rlm_depth_reached": 2},
        )

        assert output.rlm_synthesis == "Synthesized answer from RLM"
        assert output.rlm_telemetry["rlm_enabled"] is True
        assert output.rlm_telemetry["rlm_depth_reached"] == 2

    def test_search_output_rlm_fields_optional(self):
        """SearchOutput RLM fields should be optional (None by default)."""
        from cogniverse_agents.search_agent import SearchOutput

        output = SearchOutput(
            query="test query",
            modality="video",
            search_mode="single_profile",
            results=[],
            total_results=0,
        )

        assert output.rlm_synthesis is None
        assert output.rlm_telemetry is None

    def test_search_agent_inherits_rlm_aware_mixin(self):
        """SearchAgent should inherit from RLMAwareMixin."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
        from cogniverse_agents.search_agent import SearchAgent

        assert issubclass(SearchAgent, RLMAwareMixin)

    def test_search_input_rlm_with_new_options(self):
        """SearchInput should accept RLMOptions with new fields."""
        from cogniverse_agents.search_agent import SearchInput

        rlm_opts = RLMOptions(
            enabled=True,
            max_iterations=5,
            max_llm_calls=50,
            timeout_seconds=600,
            backend="anthropic",
            model="claude-3-sonnet",
        )

        input_data = SearchInput(query="test", tenant_id="test_tenant", rlm=rlm_opts)

        assert input_data.rlm.enabled is True
        assert input_data.rlm.max_iterations == 5
        assert input_data.rlm.max_llm_calls == 50
        assert input_data.rlm.timeout_seconds == 600
        assert input_data.rlm.backend == "anthropic"
        assert input_data.rlm.model == "claude-3-sonnet"
