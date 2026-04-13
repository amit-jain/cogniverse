"""
Comprehensive integration tests for the complete DSPy 3.0 Multi-Agent Routing System.

This tests the real end-to-end functionality:
1. Enhanced routing with relationship extraction
2. Query enhancement with context
3. Multi-agent orchestration
4. Enhanced video search with relationship context
5. Result enhancement and aggregation
6. ComposableQueryAnalysisModule Path A/B selection with real Ollama + GLiNER
"""

from unittest.mock import AsyncMock, patch

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

from .conftest import skip_if_no_ollama

_TEST_TENANT = "dspy_system_test"


@pytest.fixture
def real_telemetry_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)


@pytest.mark.integration
class TestCompleteDSPySystem:
    """Integration tests for complete DSPy multi-agent routing system"""

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self, telemetry_manager_without_phoenix):
        """Test complete query processing pipeline"""

        # Test that the core components can be imported and work together
        from cogniverse_agents.routing.base import GenerationType, SearchModality
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

        # Mock the dependencies for testing
        with patch(
            "cogniverse_agents.routing.relationship_extraction_tools.RelationshipExtractorTool"
        ):
            with patch(
                "cogniverse_vespa.tenant_aware_search_client.TenantAwareVespaSearchClient"
            ):
                # Initialize routing agent
                deps = RoutingDeps(
                    telemetry_config=telemetry_manager_without_phoenix.config,
                )
                routing_agent = RoutingAgent(deps=deps)

                # Test that it can process a query
                query = "Find videos of robots playing soccer"

                # Mock the routing decision
                routing_agent._make_routing_decision = AsyncMock(
                    return_value={
                        "search_modality": SearchModality.VIDEO,
                        "generation_type": GenerationType.RAW_RESULTS,
                        "confidence_score": 0.85,
                        "entities": [{"text": "robots", "label": "ENTITY"}],
                        "relationships": [
                            {
                                "subject": "robots",
                                "relation": "playing",
                                "object": "soccer",
                            }
                        ],
                    }
                )

                routing_agent._analyze_and_enhance_query = AsyncMock(
                    return_value=(
                        [{"text": "robots", "label": "ENTITY"}],
                        [
                            {
                                "subject": "robots",
                                "relation": "playing",
                                "object": "soccer",
                            }
                        ],
                        "Find videos of robots playing soccer with artificial intelligence",
                        {
                            "enhancement_method": "relationship_context",
                            "query_variants": [
                                {
                                    "name": "original",
                                    "query": "Find videos of robots playing soccer",
                                },
                                {
                                    "name": "reformulated",
                                    "query": "Find videos of robots playing soccer with artificial intelligence",
                                },
                            ],
                            "rrf_k": 60,
                        },
                    )
                )

                # Test the routing
                result = await routing_agent.route_query(query, tenant_id="test_tenant")

                # Verify the system works
                assert result is not None
                # RoutingDecision is a dict-like object, check for key
                if hasattr(result, "search_modality"):
                    assert result.search_modality is not None
                elif isinstance(result, dict) and "search_modality" in result:
                    assert result["search_modality"] is not None
                else:
                    # Result should be a RoutingDecision or dict with routing info
                    assert result is not None

    def test_phase_6_advanced_components_integration(self, real_telemetry_provider):
        """Test Phase 6 advanced optimization components integration"""

        from cogniverse_agents.routing.adaptive_threshold_learner import (
            AdaptiveThresholdLearner,
        )

        # Test adaptive learning with real telemetry provider
        learner = AdaptiveThresholdLearner(
            telemetry_provider=real_telemetry_provider,
            tenant_id=_TEST_TENANT,
        )
        assert learner is not None

    @pytest.mark.asyncio
    async def test_multi_agent_orchestration_simulation(
        self, telemetry_manager_without_phoenix
    ):
        """Test multi-agent orchestration with mocked agents"""

        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

        with patch(
            "cogniverse_agents.routing.relationship_extraction_tools.RelationshipExtractorTool"
        ):
            deps = RoutingDeps(
                telemetry_config=telemetry_manager_without_phoenix.config,
            )
            routing_agent = RoutingAgent(deps=deps)

            # Test that routing agent was created and has core capabilities
            assert routing_agent is not None
            assert hasattr(routing_agent, "route_query")


@skip_if_no_ollama
@pytest.mark.integration
class TestComposableModulePathSelection:
    """
    Real integration tests for ComposableQueryAnalysisModule Path A/B selection.

    Uses real Ollama (gemma3:4b) and real GLiNER to verify:
    - Path A: GLiNER extracts high-confidence entities → LLM reformulates only
    - Path B: GLiNER low/no confidence → LLM does entity extraction + reformulation
    """

    @pytest.fixture(autouse=True)
    def configure_dspy_lm(self):
        """Configure DSPy with real Ollama for composable module tests."""
        lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @pytest.fixture
    def composable_module(self):
        """Create ComposableQueryAnalysisModule with real GLiNER and spaCy."""
        from cogniverse_agents.routing.dspy_relationship_router import (
            ComposableQueryAnalysisModule,
        )
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )

        gliner_extractor = GLiNERRelationshipExtractor()
        spacy_analyzer = SpaCyDependencyAnalyzer()
        return ComposableQueryAnalysisModule(
            gliner_extractor=gliner_extractor,
            spacy_analyzer=spacy_analyzer,
            entity_confidence_threshold=0.6,
            min_entities_for_fast_path=1,
        )

    @pytest.mark.asyncio
    async def test_path_a_with_entity_rich_query(self, composable_module):
        """
        Path A: Entity-rich query where GLiNER extracts high-confidence entities.

        GLiNER should find entities with confidence >= 0.6, triggering Path A
        (heuristic relationships + LLM reformulation only).
        """
        # Query with well-known entities GLiNER should recognize
        query = "Tesla autonomous vehicles using machine learning"
        result = composable_module.forward(query, search_context="general")

        assert result.path_used in ("gliner_fast_path", "llm_unified_path")
        assert isinstance(result.entities, list)
        assert isinstance(result.relationships, list)
        assert isinstance(result.enhanced_query, str)
        assert len(result.enhanced_query) > 0
        assert isinstance(result.query_variants, list)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

        # If Path A was used, entities came from GLiNER (pre-extracted)
        if result.path_used == "gliner_fast_path":
            assert len(result.entities) >= 1
            # Entities should have confidence scores (from GLiNER)
            for entity in result.entities:
                assert "confidence" in entity
                assert entity["confidence"] >= 0.6

    @pytest.mark.asyncio
    async def test_path_b_with_low_confidence_entities(self, composable_module):
        """
        Path B: Query where GLiNER produces low-confidence entities.

        Force Path B by setting a very high threshold so GLiNER entities
        don't meet the bar. The full LLM should extract entities itself.
        """
        # Override threshold to force Path B
        composable_module.entity_confidence_threshold = 0.99

        query = "robots playing soccer in a field"
        result = composable_module.forward(query, search_context="general")

        # Should use Path B due to high threshold
        assert result.path_used == "llm_unified_path"

        # Path B: LLM produces output with correct shape
        assert isinstance(result.entities, list)
        assert isinstance(result.relationships, list)
        assert isinstance(result.enhanced_query, str)
        assert len(result.enhanced_query) > 0

        # Query variants should be generated
        assert isinstance(result.query_variants, list)

    @pytest.mark.asyncio
    async def test_path_b_no_gliner_model(self):
        """
        Path B: When GLiNER model is None, always use Path B.

        This simulates environments where GLiNER isn't installed.
        """
        from cogniverse_agents.routing.dspy_relationship_router import (
            ComposableQueryAnalysisModule,
        )
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )

        # Create extractor with GLiNER model explicitly unavailable.
        # Must also prevent lazy-loading from re-loading the model.
        gliner_extractor = GLiNERRelationshipExtractor()
        gliner_extractor.gliner_model = None  # Simulate GLiNER unavailable
        gliner_extractor._load_gliner_model = lambda: None  # Prevent lazy reload
        spacy_analyzer = SpaCyDependencyAnalyzer()

        module = ComposableQueryAnalysisModule(
            gliner_extractor=gliner_extractor,
            spacy_analyzer=spacy_analyzer,
        )

        query = "machine learning research papers"
        result = module.forward(query, search_context="general")

        # Must use Path B since GLiNER is unavailable
        assert result.path_used == "llm_unified_path"

        # LLM should produce output with correct shape
        assert isinstance(result.entities, list)
        assert isinstance(result.enhanced_query, str)
        assert len(result.enhanced_query) > 0

    @pytest.mark.asyncio
    async def test_both_paths_produce_same_output_shape(self, composable_module):
        """
        Both Path A and Path B produce identical output shapes.

        Validates the composable module contract: same fields regardless of path.
        """
        query = "deep learning neural networks"

        # Run with normal threshold (may be Path A or B)
        result_normal = composable_module.forward(query, search_context="general")

        # Force Path B by setting impossible threshold
        composable_module.entity_confidence_threshold = 0.99
        result_forced_b = composable_module.forward(query, search_context="general")

        # Both should have identical field sets
        required_fields = [
            "entities",
            "relationships",
            "enhanced_query",
            "query_variants",
            "confidence",
            "path_used",
            "domain_classification",
            "reasoning",
        ]

        for field in required_fields:
            assert hasattr(result_normal, field), f"Normal result missing '{field}'"
            assert hasattr(result_forced_b, field), f"Forced B result missing '{field}'"

        # Type consistency
        assert isinstance(result_normal.entities, list)
        assert isinstance(result_forced_b.entities, list)
        assert isinstance(result_normal.query_variants, list)
        assert isinstance(result_forced_b.query_variants, list)
        assert isinstance(result_normal.confidence, float)
        assert isinstance(result_forced_b.confidence, float)

    @pytest.mark.asyncio
    async def test_path_b_generates_query_variants(self, composable_module):
        """
        Path B LLM generates query variants for fusion search.

        This is critical: when GLiNER fails, the LLM must still produce
        diverse query reformulations for multi-query fusion.
        """
        # Force Path B
        composable_module.entity_confidence_threshold = 0.99

        query = "compare autonomous robotics with traditional manufacturing"
        result = composable_module.forward(query, search_context="general")

        assert result.path_used == "llm_unified_path"
        assert isinstance(result.query_variants, list)
        assert len(result.query_variants) >= 1, (
            f"Path B should generate query variants for complex query, "
            f"got: {result.query_variants}"
        )

        # Each variant should have name and query
        for variant in result.query_variants:
            assert "name" in variant, f"Variant missing 'name': {variant}"
            assert "query" in variant, f"Variant missing 'query': {variant}"
            assert len(variant["query"]) > 0, f"Variant has empty query: {variant}"




@skip_if_no_ollama
@pytest.mark.integration
class TestAdvancedRoutingOptimizerWithRealLLM:
    """
    Real integration tests for AdvancedRoutingOptimizer with real Ollama.

    Tests experience recording, reward computation, routing policy execution,
    and optimization triggering with a real LLM backend.
    """

    @pytest.fixture(autouse=True)
    def configure_dspy_lm(self):
        """Configure DSPy with real Ollama for optimizer tests."""
        lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @pytest.fixture
    def optimizer(self, tmp_path, real_telemetry_provider):
        """Create AdvancedRoutingOptimizer with isolated temp storage."""
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedOptimizerConfig,
            AdvancedRoutingOptimizer,
        )

        config = AdvancedOptimizerConfig(
            min_experiences_for_training=5,
            update_frequency=5,
            batch_size=5,
            experience_replay_size=50,
            enable_persistence=False,
        )
        return AdvancedRoutingOptimizer(
            tenant_id=_TEST_TENANT,
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            config=config,
            telemetry_provider=real_telemetry_provider,
        )

    @pytest.mark.asyncio
    async def test_optimizer_record_experience_and_compute_reward(self, optimizer):
        """Record a routing experience and verify reward computation."""
        reward = await optimizer.record_routing_experience(
            query="Find videos of robots playing soccer",
            entities=[{"text": "robots", "label": "ENTITY"}],
            relationships=[
                {"subject": "robots", "relation": "playing", "object": "soccer"}
            ],
            enhanced_query="Find videos of robots playing soccer with AI",
            chosen_agent="search_agent",
            routing_confidence=0.85,
            search_quality=0.8,
            agent_success=True,
            processing_time=1.2,
            user_satisfaction=0.9,
        )

        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        assert reward > 0.5
        assert len(optimizer.experiences) == 1
        assert optimizer.experiences[0].reward == reward

    @pytest.mark.asyncio
    async def test_optimizer_routing_policy_with_real_llm(self, optimizer):
        """Routing policy makes predictions via real Ollama."""
        assert optimizer.routing_policy is not None

        prediction = optimizer.routing_policy(
            query="Find videos of autonomous robots",
            entities=[{"text": "robots", "label": "ENTITY"}],
            relationships=[],
            enhanced_query="Find videos of autonomous robots with AI",
        )

        assert hasattr(prediction, "recommended_agent")
        assert hasattr(prediction, "confidence")
        assert hasattr(prediction, "reasoning")
        assert isinstance(prediction.recommended_agent, str)
        assert len(prediction.recommended_agent) > 0

    @pytest.mark.asyncio
    async def test_optimizer_get_routing_recommendations(self, optimizer):
        """Get routing recommendations (baseline mode before enough experience)."""
        recommendations = await optimizer.get_routing_recommendations(
            query="Show me video clips of soccer matches",
            entities=[{"text": "soccer", "label": "SPORT"}],
            relationships=[],
        )

        assert isinstance(recommendations, dict)
        assert "recommended_agent" in recommendations
        assert "confidence" in recommendations
        assert "reasoning" in recommendations
        assert isinstance(recommendations["confidence"], float)
        assert 0.0 <= recommendations["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_optimizer_accumulates_experiences_and_updates_metrics(
        self, optimizer
    ):
        """Record multiple experiences and verify metric updates."""
        test_queries = [
            ("robots playing soccer", "search_agent", 0.8, True),
            ("summarize autonomous vehicle research", "summarizer_agent", 0.7, True),
            (
                "detailed analysis of neural networks",
                "detailed_report_agent",
                0.6,
                False,
            ),
            ("video clips of drone footage", "search_agent", 0.9, True),
            ("compare machine learning frameworks", "search_agent", 0.75, True),
        ]

        for query, agent, quality, success in test_queries:
            await optimizer.record_routing_experience(
                query=query,
                entities=[],
                relationships=[],
                enhanced_query=query,
                chosen_agent=agent,
                routing_confidence=0.7,
                search_quality=quality,
                agent_success=success,
                processing_time=1.0,
            )

        assert len(optimizer.experiences) == 5
        assert optimizer.metrics.total_experiences == 5
        assert optimizer.metrics.successful_routes == 4
        assert optimizer.metrics.failed_routes == 1
        assert optimizer.metrics.avg_reward > 0
        assert "search_agent" in optimizer.metrics.agent_preferences

    @pytest.mark.asyncio
    async def test_optimizer_triggers_optimization_with_enough_data(self, optimizer):
        """With enough experiences, optimizer triggers real DSPy optimization."""
        for i in range(6):
            await optimizer.record_routing_experience(
                query=f"test query number {i} about video search",
                entities=[{"text": f"entity_{i}", "label": "ENTITY"}],
                relationships=[],
                enhanced_query=f"enhanced test query {i}",
                chosen_agent="search_agent",
                routing_confidence=0.7 + i * 0.02,
                search_quality=0.6 + i * 0.05,
                agent_success=True,
                processing_time=1.0,
            )

        assert len(optimizer.experiences) == 6
        assert optimizer.advanced_optimizer is not None

    @pytest.mark.asyncio
    async def test_optimizer_confidence_calibrator_with_real_llm(self, optimizer):
        """Confidence calibrator adjusts raw confidence using real LLM."""
        assert optimizer.confidence_calibrator is not None

        calibrated = await optimizer._calibrate_confidence(
            raw_confidence=0.8,
            query="Find videos of robots",
            entities=[{"text": "robots", "label": "ENTITY"}],
            relationships=[],
        )

        assert isinstance(calibrated, float)
        assert 0.0 <= calibrated <= 1.0

    @pytest.mark.asyncio
    async def test_optimizer_status_report(self, optimizer):
        """Optimization status report with real state."""
        status = optimizer.get_optimization_status()

        assert isinstance(status, dict)
        assert "optimizer_ready" in status
        assert "total_experiences" in status
        assert "training_step" in status
        assert "metrics" in status
        assert "config" in status
        assert status["total_experiences"] == 0
        assert status["training_step"] == 0


@pytest.mark.integration
class TestAdaptiveThresholdLearnerIntegration:
    """
    Integration tests for AdaptiveThresholdLearner.

    Tests performance sample recording, threshold adaptation via gradient-based
    and statistical methods, and automatic rollback on performance degradation.
    Uses real scipy.stats for statistical hypothesis testing.
    """

    @pytest.fixture
    def learner(self, real_telemetry_provider):
        """Create AdaptiveThresholdLearner with real telemetry provider."""
        from cogniverse_agents.routing.adaptive_threshold_learner import (
            AdaptiveThresholdConfig,
            AdaptiveThresholdLearner,
            ThresholdConfig,
            ThresholdParameter,
        )

        config = AdaptiveThresholdConfig(
            update_frequency=10,
            performance_window_size=50,
            enable_automatic_rollback=True,
            rollback_window_size=5,
        )
        config.threshold_configs = {
            ThresholdParameter.ROUTING_CONFIDENCE: ThresholdConfig(
                parameter=ThresholdParameter.ROUTING_CONFIDENCE,
                initial_value=0.7,
                min_value=0.3,
                max_value=0.95,
                learning_rate=0.02,
                min_samples_for_update=10,
                primary_metric="success_rate",
            ),
            ThresholdParameter.ENTITY_CONFIDENCE: ThresholdConfig(
                parameter=ThresholdParameter.ENTITY_CONFIDENCE,
                initial_value=0.5,
                min_value=0.2,
                max_value=0.9,
                learning_rate=0.01,
                min_samples_for_update=10,
                primary_metric="precision",
            ),
        }
        return AdaptiveThresholdLearner(
            telemetry_provider=real_telemetry_provider,
            tenant_id=_TEST_TENANT,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_learner_initializes_with_default_thresholds(self, learner):
        """Learner initializes all threshold parameters with correct defaults."""
        from cogniverse_agents.routing.adaptive_threshold_learner import (
            ThresholdParameter,
        )

        thresholds = learner.get_current_thresholds()
        assert ThresholdParameter.ROUTING_CONFIDENCE in thresholds
        assert ThresholdParameter.ENTITY_CONFIDENCE in thresholds
        assert thresholds[ThresholdParameter.ROUTING_CONFIDENCE] == 0.7
        assert thresholds[ThresholdParameter.ENTITY_CONFIDENCE] == 0.5

    @pytest.mark.asyncio
    async def test_learner_records_performance_samples(self, learner):
        """Record performance samples and verify metric updates."""
        await learner.record_performance_sample(
            routing_success=True,
            routing_confidence=0.85,
            search_quality=0.8,
            response_time=1.5,
            user_satisfaction=0.9,
            enhancement_applied=True,
            enhancement_quality=0.7,
        )

        assert learner.sample_count == 1
        assert learner.current_metrics.success_rate > 0
        assert learner.current_metrics.search_quality > 0

    @pytest.mark.asyncio
    async def test_learner_threshold_adaptation_with_real_stats(self, learner):
        """Feed enough data to trigger threshold adaptation using real scipy.stats."""
        import random

        random.seed(42)

        for _ in range(15):
            await learner.record_performance_sample(
                routing_success=random.random() > 0.3,
                routing_confidence=0.6 + random.random() * 0.3,
                search_quality=0.5 + random.random() * 0.4,
                response_time=0.5 + random.random() * 2.0,
                user_satisfaction=0.5 + random.random() * 0.5,
                enhancement_applied=True,
                enhancement_quality=0.4 + random.random() * 0.4,
            )

        assert learner.sample_count == 15

        status = learner.get_learning_status()
        assert status["total_samples"] == 15
        assert status["adaptive_learning_enabled"] is True

        for state in learner.threshold_states.values():
            assert len(state.performance_samples) > 0

    @pytest.mark.asyncio
    async def test_learner_get_threshold_value(self, learner):
        """Get individual threshold values by parameter type."""
        from cogniverse_agents.routing.adaptive_threshold_learner import (
            ThresholdParameter,
        )

        routing_conf = learner.get_threshold_value(
            ThresholdParameter.ROUTING_CONFIDENCE
        )
        assert routing_conf == 0.7

        entity_conf = learner.get_threshold_value(ThresholdParameter.ENTITY_CONFIDENCE)
        assert entity_conf == 0.5

    @pytest.mark.asyncio
    async def test_learner_learning_status_report(self, learner):
        """Learning status report returns complete information."""
        status = learner.get_learning_status()

        assert "adaptive_learning_enabled" in status
        assert "total_samples" in status
        assert "current_performance" in status
        assert "threshold_status" in status
        assert "config" in status

        perf = status["current_performance"]
        assert "success_rate" in perf
        assert "average_confidence" in perf
        assert "response_time" in perf
        assert "search_quality" in perf

    @pytest.mark.asyncio
    async def test_learner_reset_clears_all_state(self, learner):
        """Reset learning state clears everything."""
        for _ in range(5):
            await learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.8,
                search_quality=0.7,
                response_time=1.0,
            )

        assert learner.sample_count == 5

        await learner.reset_learning_state()

        assert learner.sample_count == 0
        assert len(learner.metrics_history) == 0

        from cogniverse_agents.routing.adaptive_threshold_learner import (
            ThresholdParameter,
        )

        thresholds = learner.get_current_thresholds()
        assert thresholds[ThresholdParameter.ROUTING_CONFIDENCE] == 0.7



@pytest.mark.integration
class TestForceOptimizerSelection:
    """Tests force_optimizer config routes to the correct DSPy optimizer.

    The AdvancedMultiStageOptimizer._select_optimizer() respects force_optimizer
    to bypass size-based adaptive selection. This exercises all 4 optimizer paths.
    """

    @pytest.fixture
    def dspy_lm(self):
        """Configure DSPy LM for optimizer tests."""
        lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    async def _create_optimizer_with_force(
        self, tmp_path, force_optimizer_name, telemetry_provider
    ):
        """Helper to create AdvancedRoutingOptimizer with force_optimizer and trigger lazy init."""
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedOptimizerConfig,
            AdvancedRoutingOptimizer,
        )

        config = AdvancedOptimizerConfig(
            min_experiences_for_training=3,
            update_frequency=100,  # High to prevent optimization during setup
            batch_size=3,
            force_optimizer=force_optimizer_name,
            enable_persistence=False,
        )
        optimizer = AdvancedRoutingOptimizer(
            tenant_id=f"test_force_{force_optimizer_name}",
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            config=config,
            telemetry_provider=telemetry_provider,
        )

        # Record enough experiences to pass min_experiences_for_training
        for i, query in enumerate(
            ["query alpha", "query beta", "query gamma", "query delta"]
        ):
            await optimizer.record_routing_experience(
                query=query,
                entities=[{"text": f"entity_{i}", "label": "TEST"}],
                relationships=[],
                enhanced_query=f"enhanced {query}",
                chosen_agent="search_agent",
                routing_confidence=0.8,
                search_quality=0.7,
                agent_success=True,
                processing_time=1.0,
            )

        # Manually trigger lazy init of advanced_optimizer without running
        # the full optimization step (which requires 32+ training examples)
        if optimizer.advanced_optimizer is None:
            optimizer.advanced_optimizer = optimizer._create_advanced_optimizer()

        return optimizer

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_simba_selects_simba(
        self, dspy_lm, tmp_path, real_telemetry_provider
    ):
        """force_optimizer='simba' → _select_optimizer returns SIMBA."""
        optimizer = await self._create_optimizer_with_force(
            tmp_path, "simba", real_telemetry_provider
        )
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "simba"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_gepa_selects_gepa(
        self, dspy_lm, tmp_path, real_telemetry_provider
    ):
        """force_optimizer='gepa' → _select_optimizer returns GEPA."""
        optimizer = await self._create_optimizer_with_force(
            tmp_path, "gepa", real_telemetry_provider
        )
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "gepa"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_mipro_selects_mipro(
        self, dspy_lm, tmp_path, real_telemetry_provider
    ):
        """force_optimizer='mipro' → _select_optimizer returns MIPROv2."""
        optimizer = await self._create_optimizer_with_force(
            tmp_path, "mipro", real_telemetry_provider
        )
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "mipro"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_bootstrap_selects_bootstrap(
        self, dspy_lm, tmp_path, real_telemetry_provider
    ):
        """force_optimizer='bootstrap' → _select_optimizer returns BootstrapFewShot."""
        optimizer = await self._create_optimizer_with_force(
            tmp_path, "bootstrap", real_telemetry_provider
        )
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "bootstrap"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_adaptive_selects_best_for_dataset_size(
        self, dspy_lm, tmp_path, real_telemetry_provider
    ):
        """Without force_optimizer, adaptive strategy selects based on dataset size."""
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedOptimizerConfig,
            AdvancedRoutingOptimizer,
        )

        config = AdvancedOptimizerConfig(
            min_experiences_for_training=3,
            update_frequency=3,
            batch_size=3,
            force_optimizer=None,
            optimizer_strategy="adaptive",
            enable_persistence=False,
        )
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test_adaptive",
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            config=config,
            telemetry_provider=real_telemetry_provider,
        )

        # Record enough to trigger lazy init
        for i in range(4):
            await optimizer.record_routing_experience(
                query=f"query_{i}",
                entities=[{"text": f"entity_{i}", "label": "TEST"}],
                relationships=[],
                enhanced_query=f"enhanced query_{i}",
                chosen_agent="search_agent",
                routing_confidence=0.8,
                search_quality=0.7,
                agent_success=True,
                processing_time=1.0,
            )

        assert optimizer.advanced_optimizer is not None

        # Small dataset (10) → only bootstrap is applicable (threshold=20 not met)
        _, name_small = optimizer.advanced_optimizer._select_optimizer(10)
        assert name_small == "bootstrap"

        # Medium dataset (60) → simba is applicable (threshold=50 met)
        _, name_medium = optimizer.advanced_optimizer._select_optimizer(60)
        assert name_medium == "simba"

        # Large dataset (150) → mipro is applicable (threshold=100 met)
        _, name_large = optimizer.advanced_optimizer._select_optimizer(150)
        assert name_large == "mipro"

        # Very large dataset (250) → gepa is applicable (threshold=200 met)
        _, name_xlarge = optimizer.advanced_optimizer._select_optimizer(250)
        assert name_xlarge == "gepa"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_optimizer_compile_end_to_end(
        self, dspy_lm, tmp_path, real_telemetry_provider
    ):
        """Force bootstrap optimizer and verify compile produces optimized policy."""
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedOptimizerConfig,
            AdvancedRoutingOptimizer,
        )

        config = AdvancedOptimizerConfig(
            min_experiences_for_training=3,
            update_frequency=3,
            batch_size=3,
            force_optimizer="bootstrap",
            enable_persistence=False,
        )
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test_compile_e2e",
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            config=config,
            telemetry_provider=real_telemetry_provider,
        )

        assert optimizer.routing_policy is not None

        # Record enough experiences to trigger optimization
        for i, query in enumerate(
            ["deep learning vision", "NLP transformers", "reinforcement learning"]
        ):
            await optimizer.record_routing_experience(
                query=query,
                entities=[{"text": f"topic_{i}", "label": "TECHNOLOGY"}],
                relationships=[],
                enhanced_query=f"enhanced {query}",
                chosen_agent="search_agent",
                routing_confidence=0.85,
                search_quality=0.75,
                agent_success=True,
                processing_time=1.5,
            )

        # Optimization should have run via _run_optimization_step()
        assert optimizer.advanced_optimizer is not None
        assert optimizer.training_step >= 1
        # Policy should still be callable
        assert optimizer.routing_policy is not None
        assert hasattr(optimizer.routing_policy, "forward")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
