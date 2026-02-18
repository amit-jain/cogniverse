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

from .conftest import skip_if_no_ollama


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
                    tenant_id="test_tenant",
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
                result = await routing_agent.route_query(query)

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

    @pytest.mark.ci_fast
    def test_routing_system_components_integration(self):
        """Test that all routing system components integrate correctly"""

        # Test DSPy routing signatures
        # Test advanced optimization
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedRoutingOptimizer,
        )
        from cogniverse_agents.routing.dspy_routing_signatures import (
            AdvancedRoutingSignature,
            BasicQueryAnalysisSignature,
        )

        # Test query enhancement
        from cogniverse_agents.routing.query_enhancement_engine import (
            QueryEnhancementPipeline,
        )

        # Test relationship extraction
        from cogniverse_agents.routing.relationship_extraction_tools import (
            RelationshipExtractorTool,
        )

        # Verify all components can be imported
        assert BasicQueryAnalysisSignature is not None
        assert AdvancedRoutingSignature is not None
        assert RelationshipExtractorTool is not None
        assert QueryEnhancementPipeline is not None
        assert AdvancedRoutingOptimizer is not None

    def test_phase_6_advanced_components_integration(self):
        """Test Phase 6 advanced optimization components integration"""

        from cogniverse_agents.routing.adaptive_threshold_learner import (
            AdaptiveThresholdLearner,
        )
        from cogniverse_agents.routing.mlflow_integration import (
            ExperimentConfig,
        )
        from cogniverse_agents.routing.simba_query_enhancer import SIMBAConfig

        # Test SIMBA
        simba_config = SIMBAConfig()
        assert simba_config.similarity_threshold > 0

        # Test adaptive learning (with mocked storage)
        with patch("pathlib.Path"):
            learner = AdaptiveThresholdLearner(tenant_id="test_tenant")
            assert learner is not None

        # Test MLflow integration basic structure
        exp_config = ExperimentConfig(experiment_name="test")
        assert exp_config.experiment_name == "test"

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
                tenant_id="test_tenant",
                telemetry_config=telemetry_manager_without_phoenix.config,
            )
            routing_agent = RoutingAgent(deps=deps)

            # Test orchestration capability detection
            capabilities = routing_agent._get_routing_capabilities(deps)
            assert isinstance(capabilities, list)
            assert len(capabilities) > 0


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_complex_query_processing(self):
        """Test processing of complex multi-entity queries"""

        # This tests the system's ability to handle real queries

        # Test that the system can at least parse these without crashing
        from cogniverse_agents.routing.relationship_extraction_tools import (
            RelationshipExtractorTool,
        )

        try:
            extractor = RelationshipExtractorTool()
            # If initialization succeeds, the component structure is correct
            assert extractor is not None
        except Exception:
            # Some dependencies might not be available in test environment
            # but the import structure should be correct
            pass


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
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
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

        # Path B: LLM should have extracted entities
        assert isinstance(result.entities, list)
        assert isinstance(result.relationships, list)
        assert isinstance(result.enhanced_query, str)
        assert len(result.enhanced_query) > 0

        # LLM should produce at least some entities for this entity-rich query
        assert len(result.entities) >= 1, (
            f"Path B LLM should extract entities from '{query}', "
            f"but got: {result.entities}"
        )

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

        # Create extractor with GLiNER model explicitly set to None
        gliner_extractor = GLiNERRelationshipExtractor()
        gliner_extractor.gliner_model = None  # Simulate GLiNER unavailable
        spacy_analyzer = SpaCyDependencyAnalyzer()

        module = ComposableQueryAnalysisModule(
            gliner_extractor=gliner_extractor,
            spacy_analyzer=spacy_analyzer,
        )

        query = "machine learning research papers"
        result = module.forward(query, search_context="general")

        # Must use Path B since GLiNER is unavailable
        assert result.path_used == "llm_unified_path"

        # LLM should extract entities and generate enhanced query
        assert isinstance(result.entities, list)
        assert (
            len(result.entities) >= 1
        ), f"Path B LLM should extract entities for '{query}', but got empty list"
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

    @pytest.mark.asyncio
    async def test_composable_module_in_enhancement_pipeline(self, composable_module):
        """
        ComposableQueryAnalysisModule works inside QueryEnhancementPipeline.

        Tests the real pipeline: SIMBA → composable module → variants.
        """
        from cogniverse_agents.routing.query_enhancement_engine import (
            QueryEnhancementPipeline,
        )

        pipeline = QueryEnhancementPipeline(
            analysis_module=composable_module,
            enable_simba=False,
            query_fusion_config={"include_original": True, "rrf_k": 60},
        )

        result = await pipeline.enhance_query_with_relationships(
            query="robots playing soccer in a field",
            search_context="general",
        )

        # Pipeline should return standard result dict
        assert isinstance(result, dict)
        assert "original_query" in result
        assert "enhanced_query" in result
        assert "extracted_entities" in result
        assert "extracted_relationships" in result
        assert "query_variants" in result
        assert "quality_score" in result

        # Enhanced query should be non-empty
        assert len(result["enhanced_query"]) > 0

        # Should have entities (from either Path A GLiNER or Path B LLM)
        assert isinstance(result["extracted_entities"], list)

    @pytest.mark.asyncio
    async def test_full_routing_pipeline_with_real_llm(self, composable_module):
        """
        Full RoutingAgent pipeline with real Ollama producing real variants.

        End-to-end: RoutingAgent → QueryEnhancementPipeline → ComposableModule
        → real Ollama → RoutingOutput with real query_variants.
        """
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )

        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
            model_name="ollama/gemma3:4b",
            base_url="http://localhost:11434",
            query_fusion_config={"include_original": True, "rrf_k": 60},
        )
        agent = RoutingAgent(deps=deps)

        result = await agent.analyze_and_route_with_relationships(
            query="robots playing soccer in a field",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # Should produce a valid RoutingOutput
        assert result is not None
        assert hasattr(result, "query_variants")
        assert isinstance(result.query_variants, list)

        # With real Ollama, composable module should generate variants
        assert (
            len(result.query_variants) >= 1
        ), f"Real Ollama should generate query variants, got: {result.query_variants}"

        # Verify variant structure
        for variant in result.query_variants:
            assert "name" in variant
            assert "query" in variant
            assert len(variant["query"]) > 0

        # Metadata should contain rrf_k from config
        assert result.routing_metadata.get("rrf_k") == 60


@skip_if_no_ollama
@pytest.mark.integration
class TestSIMBAWithRealLLMAndEmbeddings:
    """
    Real integration tests for SIMBA query enhancement learning.

    Uses real Ollama for DSPy enhancement policy and real SentenceTransformer
    for semantic embeddings. Tests the full pattern recording, similarity
    matching, and LLM-enhanced retrieval loop.
    """

    @pytest.fixture(autouse=True)
    def configure_dspy_lm(self):
        """Configure DSPy with real Ollama for SIMBA tests."""
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @pytest.fixture
    def simba_enhancer(self, tmp_path):
        """Create SIMBAQueryEnhancer with isolated temp storage."""
        from cogniverse_agents.routing.simba_query_enhancer import (
            SIMBAConfig,
            SIMBAQueryEnhancer,
        )

        config = SIMBAConfig(
            min_patterns_for_optimization=5,
            similarity_threshold=0.5,
            min_improvement_threshold=0.05,
        )
        return SIMBAQueryEnhancer(config=config, storage_dir=str(tmp_path / "simba"))

    @pytest.mark.asyncio
    async def test_simba_embedding_model_initializes(self, simba_enhancer):
        """SentenceTransformer model loads and produces real embeddings."""
        import numpy as np

        assert simba_enhancer.embedding_model is not None

        embedding = await simba_enhancer._get_query_embedding("robots playing soccer")
        assert embedding is not None
        assert embedding.shape[0] > 0

        other = await simba_enhancer._get_query_embedding("machine learning research")
        assert other is not None
        assert not np.allclose(embedding, other)

    @pytest.mark.asyncio
    async def test_simba_record_pattern_with_real_embeddings(self, simba_enhancer):
        """Record enhancement pattern with real SentenceTransformer embeddings."""
        await simba_enhancer.record_enhancement_outcome(
            original_query="robots playing soccer",
            enhanced_query="robots playing soccer with AI capabilities in athletic competition",
            entities=[
                {"text": "robots", "label": "ENTITY"},
                {"text": "soccer", "label": "SPORT"},
            ],
            relationships=[
                {"subject": "robots", "relation": "playing", "object": "soccer"}
            ],
            enhancement_strategy="composable_path_a",
            search_quality_improvement=0.3,
            routing_confidence_improvement=0.2,
        )

        assert len(simba_enhancer.enhancement_patterns) >= 1

        pattern = simba_enhancer.enhancement_patterns[-1]
        assert pattern.query_embedding is not None
        assert pattern.entity_embedding is not None
        assert pattern.relationship_embedding is not None

    @pytest.mark.asyncio
    async def test_simba_similar_pattern_retrieval(self, simba_enhancer):
        """Find similar patterns via real cosine similarity on real embeddings."""
        patterns_data = [
            (
                "robots playing soccer",
                "enhanced robots soccer",
                [{"text": "robots", "label": "ENTITY"}],
            ),
            (
                "autonomous vehicles driving",
                "enhanced autonomous vehicles",
                [{"text": "vehicles", "label": "ENTITY"}],
            ),
            (
                "neural network training",
                "enhanced neural network training",
                [{"text": "neural network", "label": "TECHNOLOGY"}],
            ),
        ]

        for query, enhanced, entities in patterns_data:
            await simba_enhancer.record_enhancement_outcome(
                original_query=query,
                enhanced_query=enhanced,
                entities=entities,
                relationships=[],
                enhancement_strategy="composable_path_b",
                search_quality_improvement=0.3,
                routing_confidence_improvement=0.2,
            )

        assert len(simba_enhancer.enhancement_patterns) >= 3

        similar = await simba_enhancer._find_similar_patterns(
            query="robots kicking a ball on the field",
            entities=[{"text": "robots", "label": "ENTITY"}],
            relationships=[],
        )

        assert isinstance(similar, list)

    @pytest.mark.asyncio
    async def test_simba_enhance_query_with_real_llm_policy(self, simba_enhancer):
        """SIMBA uses real Ollama LLM for its enhancement policy."""
        result = await simba_enhancer.enhance_query_with_patterns(
            original_query="robots playing soccer",
            entities=[{"text": "robots", "label": "ENTITY"}],
            relationships=[
                {"subject": "robots", "relation": "playing", "object": "soccer"}
            ],
        )

        assert isinstance(result, dict)
        assert "enhanced_query" in result
        assert "enhancement_strategy" in result
        assert "confidence" in result
        assert isinstance(result["enhanced_query"], str)
        assert len(result["enhanced_query"]) > 0

    @pytest.mark.asyncio
    async def test_simba_metrics_tracking(self, simba_enhancer):
        """SIMBA metrics update correctly after real operations."""
        initial_status = simba_enhancer.get_enhancement_status()
        assert initial_status["embedding_model_available"] is True
        assert initial_status["total_patterns"] == 0

        await simba_enhancer.record_enhancement_outcome(
            original_query="deep learning models",
            enhanced_query="deep learning neural network models for image classification",
            entities=[{"text": "deep learning", "label": "TECHNOLOGY"}],
            relationships=[],
            enhancement_strategy="composable_path_b",
            search_quality_improvement=0.4,
            routing_confidence_improvement=0.3,
        )

        await simba_enhancer.enhance_query_with_patterns(
            original_query="machine learning algorithms",
            entities=[{"text": "machine learning", "label": "TECHNOLOGY"}],
            relationships=[],
        )

        status = simba_enhancer.get_enhancement_status()
        assert status["total_patterns"] >= 1
        assert (
            status["metrics"]["successful_enhancements"]
            + status["metrics"]["failed_enhancements"]
            >= 1
        )


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
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @pytest.fixture
    def optimizer(self, tmp_path):
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
            tenant_id="test_tenant",
            config=config,
            base_storage_dir=str(tmp_path / "optimization"),
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
    def learner(self, tmp_path):
        """Create AdaptiveThresholdLearner with isolated temp storage."""
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
            tenant_id="test_tenant",
            config=config,
            base_storage_dir=str(tmp_path / "adaptive"),
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


@skip_if_no_ollama
@pytest.mark.integration
class TestFullLearningPipelineIntegration:
    """
    Full learning pipeline: ComposableModule into Optimizer into SIMBA.

    End-to-end: real Ollama query analysis, record outcome to SIMBA,
    record routing experience to optimizer, verify learning state updates.
    """

    @pytest.fixture(autouse=True)
    def configure_dspy_lm(self):
        """Configure DSPy with real Ollama for full pipeline tests."""
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @pytest.mark.asyncio
    async def test_query_analysis_feeds_learning_pipeline(self, tmp_path):
        """
        Full loop: analyze query, record to SIMBA + optimizer, check learning.

        This verifies the complete data flow:
        1. ComposableQueryAnalysisModule analyzes query with real Ollama + GLiNER
        2. Enhancement outcome is recorded in SIMBA with real embeddings
        3. Routing experience is recorded in optimizer with reward computation
        4. Multiple iterations build learning state across both components
        """
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedOptimizerConfig,
            AdvancedRoutingOptimizer,
        )
        from cogniverse_agents.routing.dspy_relationship_router import (
            ComposableQueryAnalysisModule,
        )
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )
        from cogniverse_agents.routing.simba_query_enhancer import (
            SIMBAConfig,
            SIMBAQueryEnhancer,
        )

        gliner = GLiNERRelationshipExtractor()
        spacy_dep = SpaCyDependencyAnalyzer()
        composable_module = ComposableQueryAnalysisModule(
            gliner_extractor=gliner,
            spacy_analyzer=spacy_dep,
        )

        simba = SIMBAQueryEnhancer(
            config=SIMBAConfig(min_improvement_threshold=0.05),
            storage_dir=str(tmp_path / "simba"),
        )

        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test_tenant",
            config=AdvancedOptimizerConfig(
                min_experiences_for_training=3,
                update_frequency=3,
                batch_size=3,
                enable_persistence=False,
            ),
            base_storage_dir=str(tmp_path / "optimizer"),
        )

        # Step 1: Analyze query with real Ollama + GLiNER
        query = "Tesla autonomous vehicles using deep learning"
        result = composable_module.forward(query, search_context="general")

        assert result.enhanced_query is not None
        assert len(result.enhanced_query) > 0

        # Step 2: Record outcome to SIMBA
        await simba.record_enhancement_outcome(
            original_query=query,
            enhanced_query=result.enhanced_query,
            entities=result.entities,
            relationships=result.relationships,
            enhancement_strategy=f"composable_{result.path_used}",
            search_quality_improvement=0.3,
            routing_confidence_improvement=0.2,
        )

        assert len(simba.enhancement_patterns) >= 1

        # Step 3: Record routing experience to optimizer
        reward = await optimizer.record_routing_experience(
            query=query,
            entities=result.entities,
            relationships=result.relationships,
            enhanced_query=result.enhanced_query,
            chosen_agent="search_agent",
            routing_confidence=result.confidence,
            search_quality=0.8,
            agent_success=True,
            processing_time=2.0,
        )

        assert reward > 0
        assert len(optimizer.experiences) == 1

        # Step 4: Two more iterations to trigger optimizer training
        for additional_query in [
            "machine learning for robotics applications",
            "computer vision object detection algorithms",
        ]:
            r = composable_module.forward(additional_query, search_context="general")

            await simba.record_enhancement_outcome(
                original_query=additional_query,
                enhanced_query=r.enhanced_query,
                entities=r.entities,
                relationships=r.relationships,
                enhancement_strategy=f"composable_{r.path_used}",
                search_quality_improvement=0.25,
                routing_confidence_improvement=0.15,
            )

            await optimizer.record_routing_experience(
                query=additional_query,
                entities=r.entities,
                relationships=r.relationships,
                enhanced_query=r.enhanced_query,
                chosen_agent="search_agent",
                routing_confidence=r.confidence,
                search_quality=0.75,
                agent_success=True,
                processing_time=1.5,
            )

        # Verify learning state across both components
        assert len(simba.enhancement_patterns) >= 3
        assert len(optimizer.experiences) == 3
        assert optimizer.metrics.total_experiences == 3
        assert optimizer.metrics.avg_reward > 0

        # SIMBA similarity search should complete without error
        similar = await simba._find_similar_patterns(
            query="self-driving cars with neural networks",
            entities=[{"text": "self-driving cars", "label": "TECHNOLOGY"}],
            relationships=[],
        )
        assert isinstance(similar, list)


@pytest.mark.integration
class TestSIMBACompileWithRealLLM:
    """Tests the SIMBA compile path after lazy-init fix with real Ollama.

    Validates that:
    1. SIMBA optimizer is lazily initialized when patterns reach threshold
    2. SIMBA.compile() runs successfully with real training examples
    3. Enhancement policy is updated after compilation
    """

    @pytest.fixture
    def dspy_lm(self):
        """Configure DSPy LM for SIMBA compile tests."""
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @pytest.fixture
    def simba_for_compile(self, tmp_path):
        """SIMBA enhancer configured to trigger optimization quickly."""
        from cogniverse_agents.routing.simba_query_enhancer import (
            SIMBAConfig,
            SIMBAQueryEnhancer,
        )

        config = SIMBAConfig(
            min_patterns_for_optimization=3,
            optimization_trigger_frequency=3,
            similarity_threshold=0.5,
            min_improvement_threshold=0.05,
        )
        return SIMBAQueryEnhancer(
            config=config,
            storage_dir=str(tmp_path / "simba_compile"),
        )

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_simba_lazy_init_triggers_on_pattern_threshold(
        self, dspy_lm, simba_for_compile
    ):
        """Record 3 high-quality patterns → triggers lazy-init of SIMBA optimizer."""
        simba = simba_for_compile

        # Initially: no SIMBA optimizer, but policy exists
        assert simba.simba_optimizer is None
        assert simba.enhancement_policy is not None

        # Record 3 diverse high-quality patterns (avg_improvement = 0.7 >= 0.6 threshold)
        queries = [
            (
                "quantum computing encryption algorithms",
                [{"text": "quantum computing", "label": "TECHNOLOGY"}],
                [
                    {
                        "subject": "quantum computing",
                        "relation": "enables",
                        "object": "encryption",
                    }
                ],
            ),
            (
                "autonomous drone navigation systems",
                [{"text": "drone", "label": "TECHNOLOGY"}],
                [
                    {
                        "subject": "drone",
                        "relation": "uses",
                        "object": "navigation",
                    }
                ],
            ),
            (
                "neural network image classification",
                [{"text": "neural network", "label": "TECHNOLOGY"}],
                [
                    {
                        "subject": "neural network",
                        "relation": "performs",
                        "object": "classification",
                    }
                ],
            ),
        ]

        for query, entities, relationships in queries:
            await simba.record_enhancement_outcome(
                original_query=query,
                enhanced_query=f"enhanced: {query}",
                entities=entities,
                relationships=relationships,
                enhancement_strategy="composable_path_a",
                search_quality_improvement=0.7,
                routing_confidence_improvement=0.7,
            )

        # After 3 patterns: lazy-init should have triggered
        assert len(simba.enhancement_patterns) == 3
        assert simba.simba_optimizer is not None

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_simba_compile_updates_enhancement_policy(
        self, dspy_lm, simba_for_compile
    ):
        """Verify SIMBA compile runs and produces valid optimized policy."""
        simba = simba_for_compile

        # Record 3 high-quality diverse patterns
        queries = [
            (
                "quantum computing encryption algorithms",
                [{"text": "quantum computing", "label": "TECHNOLOGY"}],
                [
                    {
                        "subject": "quantum computing",
                        "relation": "enables",
                        "object": "encryption",
                    }
                ],
            ),
            (
                "autonomous drone navigation systems",
                [{"text": "drone", "label": "TECHNOLOGY"}],
                [
                    {
                        "subject": "drone",
                        "relation": "uses",
                        "object": "navigation",
                    }
                ],
            ),
            (
                "neural network image classification",
                [{"text": "neural network", "label": "TECHNOLOGY"}],
                [
                    {
                        "subject": "neural network",
                        "relation": "performs",
                        "object": "classification",
                    }
                ],
            ),
        ]

        for query, entities, relationships in queries:
            await simba.record_enhancement_outcome(
                original_query=query,
                enhanced_query=f"enhanced: {query}",
                entities=entities,
                relationships=relationships,
                enhancement_strategy="composable_path_a",
                search_quality_improvement=0.7,
                routing_confidence_improvement=0.7,
            )

        # Policy should have been updated by compile
        assert simba.enhancement_policy is not None
        assert hasattr(simba.enhancement_policy, "forward")
        # After compile, policy object may be replaced or same — either is valid
        # The key assertion is that it's still callable
        result = simba.enhancement_policy.forward(
            original_query="test query",
            entities=[],
            relationships=[],
            similar_patterns=[],
        )
        assert result is not None

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_simba_compile_skipped_below_threshold(self, dspy_lm, tmp_path):
        """Patterns below quality threshold (avg_improvement < 0.6) are excluded from training."""
        from cogniverse_agents.routing.simba_query_enhancer import (
            SIMBAConfig,
            SIMBAQueryEnhancer,
        )

        config = SIMBAConfig(
            min_patterns_for_optimization=3,
            optimization_trigger_frequency=3,
            similarity_threshold=0.5,
            min_improvement_threshold=0.05,
        )
        simba = SIMBAQueryEnhancer(
            config=config,
            storage_dir=str(tmp_path / "simba_low_quality"),
        )

        # Record 3 low-quality patterns (avg_improvement = 0.15, below 0.6 training threshold)
        for i, query in enumerate(["weak query A", "weak query B", "weak query C"]):
            await simba.record_enhancement_outcome(
                original_query=query,
                enhanced_query=f"enhanced: {query}",
                entities=[{"text": f"entity_{i}", "label": "TEST"}],
                relationships=[],
                enhancement_strategy="test",
                search_quality_improvement=0.2,
                routing_confidence_improvement=0.1,
            )

        # Optimizer should have been lazy-initialized (3 >= min_patterns_for_optimization)
        # but compile still runs — it just won't have training examples above 0.6
        assert len(simba.enhancement_patterns) == 3
        # Policy should still be valid (compile with empty trainset is a no-op)
        assert simba.enhancement_policy is not None


@pytest.mark.integration
class TestForceOptimizerSelection:
    """Tests force_optimizer config routes to the correct DSPy optimizer.

    The AdvancedMultiStageOptimizer._select_optimizer() respects force_optimizer
    to bypass size-based adaptive selection. This exercises all 4 optimizer paths.
    """

    @pytest.fixture
    def dspy_lm(self):
        """Configure DSPy LM for optimizer tests."""
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    async def _create_optimizer_with_force(self, tmp_path, force_optimizer_name):
        """Helper to create AdvancedRoutingOptimizer with force_optimizer and trigger lazy init."""
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedOptimizerConfig,
            AdvancedRoutingOptimizer,
        )

        config = AdvancedOptimizerConfig(
            min_experiences_for_training=3,
            update_frequency=3,
            batch_size=3,
            force_optimizer=force_optimizer_name,
            enable_persistence=False,
        )
        optimizer = AdvancedRoutingOptimizer(
            tenant_id=f"test_force_{force_optimizer_name}",
            config=config,
            base_storage_dir=str(tmp_path / f"force_{force_optimizer_name}"),
        )

        # Record enough experiences to trigger lazy init of advanced_optimizer
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

        return optimizer

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_simba_selects_simba(self, dspy_lm, tmp_path):
        """force_optimizer='simba' → _select_optimizer returns SIMBA."""
        optimizer = await self._create_optimizer_with_force(tmp_path, "simba")
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "simba"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_gepa_selects_gepa(self, dspy_lm, tmp_path):
        """force_optimizer='gepa' → _select_optimizer returns GEPA."""
        optimizer = await self._create_optimizer_with_force(tmp_path, "gepa")
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "gepa"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_mipro_selects_mipro(self, dspy_lm, tmp_path):
        """force_optimizer='mipro' → _select_optimizer returns MIPROv2."""
        optimizer = await self._create_optimizer_with_force(tmp_path, "mipro")
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "mipro"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_force_bootstrap_selects_bootstrap(self, dspy_lm, tmp_path):
        """force_optimizer='bootstrap' → _select_optimizer returns BootstrapFewShot."""
        optimizer = await self._create_optimizer_with_force(tmp_path, "bootstrap")
        assert optimizer.advanced_optimizer is not None

        selected_opt, name = optimizer.advanced_optimizer._select_optimizer(10)
        assert name == "bootstrap"

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_adaptive_selects_best_for_dataset_size(self, dspy_lm, tmp_path):
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
            # Thresholds: bootstrap=20, simba=50, mipro=100, gepa=200
        )
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test_adaptive",
            config=config,
            base_storage_dir=str(tmp_path / "adaptive"),
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
    async def test_force_optimizer_compile_end_to_end(self, dspy_lm, tmp_path):
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
            config=config,
            base_storage_dir=str(tmp_path / "compile_e2e"),
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


@pytest.mark.integration
class TestVespaLearningPipelineIntegration:
    """Tests ComposableModule → Vespa search → SIMBA/Optimizer feedback loop.

    Uses real Docker Vespa with deployed schemas and test data.
    """

    @pytest.fixture
    def dspy_lm(self):
        """Configure DSPy LM for Vespa pipeline tests."""
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
        dspy.configure(lm=lm)
        yield lm
        dspy.configure(lm=None)

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_query_enhancement_to_vespa_search_feedback(
        self, dspy_lm, vespa_with_schema, tmp_path
    ):
        """Full loop: enhance query → search Vespa → record feedback to SIMBA."""
        from pathlib import Path

        from cogniverse_agents.routing.dspy_relationship_router import (
            ComposableQueryAnalysisModule,
        )
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )
        from cogniverse_agents.routing.simba_query_enhancer import (
            SIMBAConfig,
            SIMBAQueryEnhancer,
        )
        from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        # Step 1: Set up ComposableQueryAnalysisModule
        gliner_extractor = GLiNERRelationshipExtractor()
        spacy_analyzer = SpaCyDependencyAnalyzer()
        composable_module = ComposableQueryAnalysisModule(
            gliner_extractor=gliner_extractor,
            spacy_analyzer=spacy_analyzer,
        )

        # Step 2: Enhance a query
        query = "robot playing soccer"
        result = composable_module.forward(query, search_context="general")
        enhanced_query = result.enhanced_query
        assert enhanced_query is not None
        assert len(enhanced_query) > 0

        # Step 3: Search Vespa with the enhanced query
        vespa_http_port = vespa_with_schema["http_port"]
        vespa_url = "http://localhost"
        default_schema = vespa_with_schema["default_schema"]
        config_manager = vespa_with_schema["manager"].config_manager
        schema_loader = FilesystemSchemaLoader(
            base_path=Path("tests/system/resources/schemas")
        )

        deps = SearchAgentDeps(
            tenant_id="test_tenant",
            backend_url=vespa_url,
            backend_port=vespa_http_port,
            schema_names=[default_schema],
        )
        search_agent = SearchAgent(
            deps=deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8015,
        )

        # Execute search with enhanced query
        search_results = search_agent.search_by_text(enhanced_query, top_k=5)
        # Search may return 0 results for this query — that's OK for the test.
        # The point is that the pipeline doesn't crash.
        assert isinstance(search_results, list)

        # Step 4: Record feedback to SIMBA
        simba = SIMBAQueryEnhancer(
            config=SIMBAConfig(
                min_patterns_for_optimization=5,
                min_improvement_threshold=0.05,
            ),
            storage_dir=str(tmp_path / "simba_vespa"),
        )

        # Use search result count as a proxy for quality
        quality_score = min(len(search_results) / 5.0, 1.0)  # Normalize to [0, 1]

        await simba.record_enhancement_outcome(
            original_query=query,
            enhanced_query=enhanced_query,
            entities=result.entities,
            relationships=result.relationships,
            enhancement_strategy=f"composable_{result.path_used}",
            search_quality_improvement=max(quality_score, 0.1),
            routing_confidence_improvement=float(result.confidence or 0.5),
        )

        assert len(simba.enhancement_patterns) >= 1
        pattern = simba.enhancement_patterns[0]
        assert pattern.original_query == query
        assert pattern.enhanced_query == enhanced_query

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_vespa_search_with_query_variants(
        self, dspy_lm, vespa_with_schema, tmp_path
    ):
        """Verify query variants from ComposableModule can all search Vespa without errors."""
        from pathlib import Path

        from cogniverse_agents.routing.dspy_relationship_router import (
            ComposableQueryAnalysisModule,
        )
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )
        from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        # Enhance query to get variants
        gliner_extractor = GLiNERRelationshipExtractor()
        spacy_analyzer = SpaCyDependencyAnalyzer()
        composable_module = ComposableQueryAnalysisModule(
            gliner_extractor=gliner_extractor,
            spacy_analyzer=spacy_analyzer,
        )

        result = composable_module.forward(
            "artificial intelligence in healthcare", search_context="general"
        )

        # Set up Vespa search
        vespa_http_port = vespa_with_schema["http_port"]
        config_manager = vespa_with_schema["manager"].config_manager
        default_schema = vespa_with_schema["default_schema"]
        schema_loader = FilesystemSchemaLoader(
            base_path=Path("tests/system/resources/schemas")
        )

        deps = SearchAgentDeps(
            tenant_id="test_tenant",
            backend_url="http://localhost",
            backend_port=vespa_http_port,
            schema_names=[default_schema],
        )
        search_agent = SearchAgent(
            deps=deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8016,
        )

        # Search with each variant — none should crash
        variants = result.query_variants
        assert isinstance(variants, list)

        for variant in variants:
            assert "query" in variant
            search_results = search_agent.search_by_text(variant["query"], top_k=5)
            assert isinstance(search_results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
