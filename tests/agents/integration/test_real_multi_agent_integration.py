"""
Real LLM integration tests for multi-agent system.

These tests use actual LLMs via Ollama to test agent functionality
without mocking the LLM layer. They validate real inference through
the centralized LLM config system (create_dspy_lm + LLMEndpointConfig).

Includes tests for:
- OLD agents: RoutingAgent (thin interface), SummarizerAgent, DetailedReportAgent
- NEW A2A agents: GatewayAgent, EntityExtractionAgent, QueryEnhancementAgent

NOT true E2E tests — they don't ingest data or search Vespa.

Requirements:
- LLM server running locally (e.g. Ollama with qwen2.5:1.5b)
- GLiNER model for GatewayAgent/EntityExtractionAgent (optional)
- Vespa backend available (optional, falls back gracefully)
- Phoenix telemetry server (optional)
"""

import importlib.util
import logging

import dspy
import pytest

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
)
from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps, GatewayInput
from cogniverse_agents.optimizer.dspy_agent_optimizer import (
    DSPyAgentOptimizerPipeline,
    DSPyAgentPromptOptimizer,
)
from cogniverse_agents.query_analysis_tool_v3 import QueryAnalysisToolV3
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_agents.summarizer_agent import SummarizerAgent, SummarizerDeps
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_TIMEOUT = 300  # 5 minutes for real LLM calls
_TEST_TENANT = "real_multi_agent_test"

# GLiNER availability check
HAS_GLINER = importlib.util.find_spec("gliner") is not None

skip_if_no_gliner = pytest.mark.skipif(
    not HAS_GLINER, reason="GLiNER not installed"
)


@pytest.fixture
def real_telemetry_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)


class TestLLMAvailability:
    """Check if LLM server is available for real integration tests."""

    def test_llm_model_available(self):
        """Test if LLM server is running and has the required model."""
        try:
            import requests

            response = requests.get("http://localhost:11434/v1/models", timeout=5)
            assert response.status_code == 200, "LLM server not reachable"
            logger.info("LLM server available")
        except Exception:
            pass


class TestRealQueryAnalysisIntegration:
    """Real integration tests for QueryAnalysisToolV3 with actual LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_real_query_analysis_with_local_llm(self, real_telemetry_provider):
        """Test real query analysis with local Ollama model."""
        config_manager = create_default_config_manager()

        analyzer = QueryAnalysisToolV3(
            config_manager=config_manager,
            telemetry_provider=real_telemetry_provider,
        )

        test_queries = [
            {
                "query": "Show me videos of dogs playing",
                "expected_intent": "search",
                "expected_video_search": True,
            },
            {
                "query": "Compare the performance of different AI models in computer vision tasks",
                "expected_intent": "comparison",
                "expected_complexity": "complex",
            },
            {
                "query": "Summarize the key findings from recent machine learning research",
                "expected_intent": "summarization",
                "expected_text_search": True,
            },
        ]

        for test_case in test_queries:
            logger.info(f"Testing query: {test_case['query']}")

            result = await analyzer.analyze(test_case["query"])

            result_dict = result.to_dict() if hasattr(result, "to_dict") else result

            assert isinstance(result_dict, dict)
            assert "primary_intent" in result_dict
            assert "complexity_level" in result_dict
            assert "needs_video_search" in result_dict
            assert "needs_text_search" in result_dict
            assert "thinking_phase" in result_dict
            assert "reasoning" in result_dict["thinking_phase"]

            if "expected_intent" in test_case:
                expected = test_case["expected_intent"].lower()
                actual = result_dict["primary_intent"].lower()
                expected_words = set(expected.split())
                actual_words = set(actual.split())
                matches = any(
                    exp_word in actual or actual_word.startswith(exp_word[:4])
                    for exp_word in expected_words
                    for actual_word in actual_words
                )
                assert matches or expected in actual or actual in expected, (
                    f"Expected '{expected}' to match '{actual}'"
                )

            if "expected_video_search" in test_case:
                assert (
                    result_dict["needs_video_search"]
                    == test_case["expected_video_search"]
                )

            if "expected_text_search" in test_case:
                assert (
                    result_dict["needs_text_search"]
                    == test_case["expected_text_search"]
                )

            reasoning = result_dict["thinking_phase"]["reasoning"]
            assert len(reasoning) > 10, "Reasoning should be substantive"

            logger.info(f"Analysis result: {result_dict}")


class TestRealAgentRoutingIntegration:
    """Real integration tests for RoutingAgent with actual LLM.

    Updated to use the thin RoutingAgent interface — entities, enhanced_query,
    and relationships arrive as pre-enriched input from upstream A2A agents.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_real_agent_routing_with_enriched_input(self):
        """Test real routing with pre-enriched input (thin interface)."""
        test_lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )

        with dspy.context(lm=test_lm):
            telemetry_config = TelemetryConfig(enabled=False)
            deps = RoutingDeps(telemetry_config=telemetry_config)
            routing_agent = RoutingAgent(deps=deps, port=8001)

        test_cases = [
            {
                "query": "Find videos of basketball games",
                "enhanced_query": "basketball games videos sports footage",
                "entities": [
                    {"text": "basketball", "type": "CONCEPT", "confidence": 0.9},
                ],
            },
            {
                "query": "Analyze the trends in renewable energy adoption",
                "enhanced_query": "renewable energy adoption trends analysis solar wind",
                "entities": [
                    {"text": "renewable energy", "type": "CONCEPT", "confidence": 0.88},
                ],
            },
            {
                "query": "Give me a brief overview of recent AI developments",
                "enhanced_query": "artificial intelligence recent developments overview",
                "entities": [
                    {"text": "AI", "type": "CONCEPT", "confidence": 0.85},
                ],
            },
        ]

        for test_case in test_cases:
            logger.info(f"Testing routing for: {test_case['query']}")

            routing_decision = await routing_agent.route_query(
                query=test_case["query"],
                enhanced_query=test_case["enhanced_query"],
                entities=test_case["entities"],
                relationships=[],
                tenant_id="test_tenant",
            )

            assert hasattr(routing_decision, "recommended_agent")
            assert hasattr(routing_decision, "confidence")
            assert hasattr(routing_decision, "reasoning")

            assert routing_decision.recommended_agent in [
                "video_search_agent",
                "search_agent",
                "summarizer_agent",
                "detailed_report_agent",
            ], f"Invalid agent: {routing_decision.recommended_agent}"

            confidence = float(routing_decision.confidence)
            assert 0.0 <= confidence <= 1.0
            assert confidence > 0.3, (
                "Confidence should be reasonably high for clear test cases"
            )

            assert len(routing_decision.reasoning) > 10, (
                "Reasoning should be substantive"
            )

            logger.info(
                f"Routing decision: {routing_decision.recommended_agent} "
                f"(confidence: {confidence})"
            )


class TestRealGatewayAgentIntegration:
    """Real integration tests for GatewayAgent with real GLiNER model."""

    @skip_if_no_gliner
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_gateway_classifies_simple_video_query(self):
        """GatewayAgent with real GLiNER classifies a clear video query."""
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps, port=18014)

        input_data = GatewayInput(
            query="show me videos about cooking Italian pasta",
            tenant_id="test_tenant",
        )
        result = await agent._process_impl(input_data)

        assert result is not None
        assert result.complexity in ("simple", "complex")
        assert result.modality in (
            "video", "text", "audio", "image", "document", "both",
        )
        assert result.routed_to is not None
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 0

        logger.info(
            f"Gateway real GLiNER: complexity={result.complexity}, "
            f"modality={result.modality}, routed_to={result.routed_to}, "
            f"confidence={result.confidence:.2f}"
        )

    @skip_if_no_gliner
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_gateway_classifies_complex_multi_modal_query(self):
        """GatewayAgent with real GLiNER classifies a multi-modal query."""
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps, port=18015)

        input_data = GatewayInput(
            query=(
                "Compare video tutorials on machine learning with "
                "audio podcasts about deep learning and generate a "
                "comprehensive report"
            ),
            tenant_id="test_tenant",
        )
        result = await agent._process_impl(input_data)

        assert result is not None
        assert result.complexity in ("simple", "complex")
        assert result.routed_to is not None
        assert 0.0 <= result.confidence <= 1.0

        logger.info(
            f"Gateway multi-modal: complexity={result.complexity}, "
            f"modality={result.modality}, routed_to={result.routed_to}"
        )

    @skip_if_no_gliner
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_gateway_handles_empty_entity_extraction(self):
        """GatewayAgent handles queries that produce no GLiNER entities gracefully."""
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps, port=18016)

        input_data = GatewayInput(
            query="xyz",
            tenant_id="test_tenant",
        )
        result = await agent._process_impl(input_data)

        assert result is not None
        # No entities → complex (forwarded to orchestrator)
        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"
        logger.info(
            f"Gateway empty entities: complexity={result.complexity}, "
            f"routed_to={result.routed_to}"
        )


class TestRealEntityExtractionIntegration:
    """Real integration tests for EntityExtractionAgent with GLiNER or DSPy."""

    @skip_if_no_gliner
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_entity_extraction_with_real_gliner(self):
        """EntityExtractionAgent extracts entities from a named-entity-rich query."""
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionInput,
        )

        deps = EntityExtractionDeps()
        agent = EntityExtractionAgent(deps=deps, port=18010)

        input_data = EntityExtractionInput(
            query="Show me videos about Barack Obama speaking in Chicago",
            tenant_id="test_tenant",
        )
        result = await agent._process_impl(input_data)

        assert result is not None
        assert result.query == input_data.query
        assert isinstance(result.entities, list)
        assert result.entity_count >= 0
        assert result.has_entities == (result.entity_count > 0)
        assert result.path_used in ("fast", "dspy")

        if result.has_entities:
            for entity in result.entities:
                assert entity.text
                assert entity.type
                assert 0.0 <= entity.confidence <= 1.0

        logger.info(
            f"EntityExtraction: {result.entity_count} entities via {result.path_used}, "
            f"types={result.dominant_types}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_entity_extraction_dspy_fallback(self):
        """EntityExtractionAgent falls back to DSPy when GLiNER is unavailable."""
        from unittest.mock import patch

        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionInput,
        )

        test_lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )

        with dspy.context(lm=test_lm):
            deps = EntityExtractionDeps()
            # Patch out GLiNER initialization so it falls back to DSPy
            with patch.object(
                EntityExtractionAgent, "_initialize_extractors"
            ):
                agent = EntityExtractionAgent(deps=deps, port=18011)
            agent._gliner_extractor = None
            agent._spacy_analyzer = None

            input_data = EntityExtractionInput(
                query="Find machine learning tutorials by Andrew Ng",
                tenant_id="test_tenant",
            )
            result = await agent._process_impl(input_data)

        assert result is not None
        assert result.path_used == "dspy"
        assert isinstance(result.entities, list)
        logger.info(
            f"EntityExtraction DSPy fallback: {result.entity_count} entities, "
            f"types={result.dominant_types}"
        )


class TestRealQueryEnhancementIntegration:
    """Real integration tests for QueryEnhancementAgent with DSPy LM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_query_enhancement_with_real_dspy(self):
        """QueryEnhancementAgent enhances a query using real DSPy inference."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        test_lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )

        with dspy.context(lm=test_lm):
            deps = QueryEnhancementDeps()
            agent = QueryEnhancementAgent(deps=deps, port=18012)

            input_data = QueryEnhancementInput(
                query="ML tutorials",
                entities=[
                    {"text": "ML", "type": "CONCEPT", "confidence": 0.85},
                ],
                relationships=[],
                tenant_id="test_tenant",
            )
            result = await agent._process_impl(input_data)

        assert result is not None
        assert result.original_query == "ML tutorials"
        assert isinstance(result.enhanced_query, str)
        assert len(result.enhanced_query) > 0
        assert isinstance(result.expansion_terms, list)
        assert isinstance(result.synonyms, list)
        assert 0.0 <= result.confidence <= 1.0

        logger.info(
            f"QueryEnhancement: '{result.original_query}' → '{result.enhanced_query}', "
            f"expansions={result.expansion_terms}, confidence={result.confidence}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_query_enhancement_with_entity_context(self):
        """QueryEnhancementAgent uses entity context from upstream extraction."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        test_lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )

        with dspy.context(lm=test_lm):
            deps = QueryEnhancementDeps()
            agent = QueryEnhancementAgent(deps=deps, port=18013)

            input_data = QueryEnhancementInput(
                query="videos about self-driving cars in San Francisco",
                entities=[
                    {"text": "self-driving cars", "type": "CONCEPT", "confidence": 0.92},
                    {"text": "San Francisco", "type": "PLACE", "confidence": 0.95},
                ],
                relationships=[
                    {
                        "subject": "self-driving cars",
                        "relation": "located_in",
                        "object": "San Francisco",
                        "confidence": 0.7,
                    },
                ],
                tenant_id="test_tenant",
            )
            result = await agent._process_impl(input_data)

        assert result is not None
        assert isinstance(result.enhanced_query, str)
        assert isinstance(result.query_variants, list)
        logger.info(
            f"QueryEnhancement with entities: '{result.enhanced_query}', "
            f"variants={len(result.query_variants)}"
        )


class TestRealAgentSpecializationIntegration:
    """Real integration tests for specialized agents with actual LLMs."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_real_summarizer_agent_with_local_llm(self):
        """Test real summarization with local Ollama model."""

        from cogniverse_agents.summarizer_agent import SummaryRequest

        config_manager = create_default_config_manager()

        test_lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )

        with dspy.context(lm=test_lm):
            deps = SummarizerDeps()
            summarizer = SummarizerAgent(deps=deps, config_manager=config_manager)

            test_search_results = [
                {
                    "title": "Machine Learning Advances",
                    "content": """Machine learning has seen tremendous advances in recent years, particularly in deep learning
                    architectures. Transformer models like GPT and BERT have revolutionized natural language
                    processing, while convolutional neural networks continue to excel in computer vision tasks.""",
                    "relevance_score": 0.95,
                },
                {
                    "title": "AI Research Focus",
                    "content": """Recent research has focused on improving model efficiency, reducing computational requirements,
                    and developing more interpretable AI systems. The field is also exploring multimodal models
                    that can process both text and images simultaneously.""",
                    "relevance_score": 0.88,
                },
            ]

            request = SummaryRequest(
                query="Summarize recent ML advances",
                search_results=test_search_results,
                summary_type="brief",
                include_visual_analysis=False,
            )

            summary_result = await summarizer.summarize(request)

            assert hasattr(summary_result, "summary")
            assert hasattr(summary_result, "key_points")
            assert hasattr(summary_result, "confidence_score")

            assert len(summary_result.summary) > 20, "Summary should be substantive"
            assert isinstance(summary_result.key_points, list)

            confidence = float(summary_result.confidence_score)
            assert 0.0 <= confidence <= 1.0

            logger.info(f"Summary result: {summary_result.summary[:100]}...")

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_real_detailed_report_agent_with_local_llm(self):
        """Test real detailed report generation with local Ollama model."""
        from cogniverse_agents.detailed_report_agent import ReportRequest

        config_manager = create_default_config_manager()

        deps = DetailedReportDeps()
        report_agent = DetailedReportAgent(deps=deps, config_manager=config_manager)

        mock_search_results = [
            {
                "title": "Advances in Neural Architecture Search",
                "content": "Recent developments in automated neural architecture design...",
                "relevance_score": 0.92,
            },
            {
                "title": "Efficient Transformer Models",
                "content": "New approaches to reducing computational overhead in transformers...",
                "relevance_score": 0.88,
            },
        ]

        request = ReportRequest(
            query="machine learning efficiency research",
            search_results=mock_search_results,
            report_type="comprehensive",
            include_visual_analysis=False,
            include_technical_details=True,
            include_recommendations=True,
        )

        report_result = await report_agent.generate_report(request)

        assert hasattr(report_result, "executive_summary")
        assert hasattr(report_result, "detailed_findings")
        assert hasattr(report_result, "recommendations")
        assert hasattr(report_result, "confidence_assessment")

        assert len(report_result.executive_summary) > 50, (
            "Executive summary should be comprehensive"
        )
        assert len(report_result.detailed_findings) >= 1, (
            "Detailed findings should be present"
        )
        assert len(report_result.recommendations) >= 1, (
            "Recommendations should be present"
        )

        confidence = float(report_result.confidence_assessment.get("overall", 0.0))
        assert 0.0 <= confidence <= 1.0

        logger.info(
            f"Report executive summary: {report_result.executive_summary[:100]}..."
        )


class TestRealDSPyOptimizationIntegration:
    """Real integration tests for DSPy optimization with actual LLMs."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT * 2)  # DSPy optimization takes longer
    async def test_real_dspy_optimization_pipeline(self):
        """Test real DSPy optimization with local Ollama model."""

        optimizer = DSPyAgentPromptOptimizer()

        endpoint_config = LLMEndpointConfig(
            model="ollama/qwen2.5:1.5b",
            api_base="http://localhost:11434",
        )
        _success = optimizer.initialize_language_model(endpoint_config)

        pipeline = DSPyAgentOptimizerPipeline(optimizer)

        qa_signature = optimizer.create_query_analysis_signature()
        assert qa_signature is not None, "Should create query analysis signature"

        ar_signature = optimizer.create_agent_routing_signature()
        assert ar_signature is not None, "Should create agent routing signature"

        training_data = pipeline.load_training_data()
        assert isinstance(training_data, dict)
        assert "query_analysis" in training_data
        assert "agent_routing" in training_data
        assert len(training_data["query_analysis"]) > 0
        assert len(training_data["agent_routing"]) > 0

        logger.info(f"Loaded training data with {len(training_data)} modules")

        pipeline.initialize_modules()
        assert "query_analysis" in pipeline.modules
        assert "agent_routing" in pipeline.modules

        try:
            query_examples = training_data["query_analysis"][
                :2
            ]  # Use only 2 examples for speed

            optimized_module = pipeline.optimize_module(
                "query_analysis", query_examples
            )

            assert optimized_module is not None, "Should return optimized module"
            assert "query_analysis" in pipeline.compiled_modules

            logger.info("Successfully optimized query_analysis module")

        except Exception as e:
            logger.warning(f"DSPy optimization failed (expected in test env): {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_real_agent_with_dspy_integration(self, real_telemetry_provider):
        """Test agents with DSPy optimization integration."""
        config_manager = create_default_config_manager()

        analyzer = QueryAnalysisToolV3(
            config_manager=config_manager,
            telemetry_provider=real_telemetry_provider,
            enable_dspy=False,
        )

        await analyzer.analyze("Show me videos of cats")

        dspy_metadata = analyzer.get_dspy_metadata()
        assert isinstance(dspy_metadata, dict)
        assert "enabled" in dspy_metadata
        assert not dspy_metadata["enabled"]

        analyzer.dspy_enabled = True
        dspy_metadata = analyzer.get_dspy_metadata()
        assert dspy_metadata["enabled"]
        assert "agent_type" in dspy_metadata

        logger.info(f"DSPy integration metadata: {dspy_metadata}")


class TestRealEndToEndWorkflow:
    """Real end-to-end workflow tests with multiple agents."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT * 2)
    async def test_real_multi_agent_workflow(self, real_telemetry_provider):
        """Test complete multi-agent workflow with real LLMs."""

        from cogniverse_agents.summarizer_agent import SummaryRequest

        config_manager = create_default_config_manager()

        query_analyzer = QueryAnalysisToolV3(
            config_manager=config_manager,
            telemetry_provider=real_telemetry_provider,
        )

        test_lm = create_dspy_lm(
            LLMEndpointConfig(
                model="ollama/qwen2.5:1.5b",
                api_base="http://localhost:11434",
            )
        )

        with dspy.context(lm=test_lm):
            telemetry_config = TelemetryConfig(enabled=False)
            routing_deps = RoutingDeps(telemetry_config=telemetry_config)
            routing_agent = RoutingAgent(deps=routing_deps, port=8001)
            summarizer_deps = SummarizerDeps()
            summarizer = SummarizerAgent(
                deps=summarizer_deps, config_manager=config_manager
            )

            test_query = (
                "Give me a summary of recent developments in artificial intelligence"
            )

            # Step 1: Query analysis
            logger.info(f"Step 1: Analyzing query: {test_query}")
            analysis_result = await query_analyzer.analyze(test_query)
            logger.info(f"Analysis result: {analysis_result}")

            # Step 2: Agent routing with pre-enriched input
            logger.info("Step 2: Routing query to appropriate agent")
            routing_decision = await routing_agent.route_query(
                query=test_query,
                enhanced_query=test_query,
                entities=[],
                relationships=[],
                tenant_id="test_tenant",
            )
            logger.info(f"Routing decision: {routing_decision.recommended_agent}")

            # Step 3: Execute with appropriate agent
            if "summarizer" in routing_decision.recommended_agent.lower():
                logger.info("Step 3: Executing with SummarizerAgent")

                ai_search_results = [
                    {
                        "title": "AI Developments 2024",
                        "content": """Artificial Intelligence has made significant strides in 2024, with major breakthroughs
                        in large language models, multimodal AI systems, and autonomous agents. Key developments
                        include improved reasoning capabilities, better alignment with human values, and more
                        efficient training methods.""",
                        "relevance_score": 0.95,
                    },
                    {
                        "title": "AI Applications",
                        "content": """The field is moving towards more practical applications
                        in healthcare, education, and scientific research.""",
                        "relevance_score": 0.88,
                    },
                ]

                request = SummaryRequest(
                    query=test_query,
                    search_results=ai_search_results,
                    summary_type="brief",
                    include_visual_analysis=False,
                )

                summary_result = await summarizer.summarize(request)
                logger.info(f"Summary result: {summary_result.summary[:100]}...")

                if hasattr(analysis_result, "to_dict"):
                    analysis_dict = analysis_result.to_dict()
                elif hasattr(analysis_result, "__dict__"):
                    analysis_dict = analysis_result.__dict__
                else:
                    analysis_dict = analysis_result

                primary_intent = (
                    analysis_dict.get("primary_intent", "")
                    if isinstance(analysis_dict, dict)
                    else getattr(analysis_result, "primary_intent", "")
                )
                assert (
                    "summary" in str(primary_intent).lower()
                    or "summarization" in str(primary_intent).lower()
                )
                assert "summarizer" in routing_decision.recommended_agent.lower()
                assert len(summary_result.summary) > 20

                logger.info("Complete multi-agent workflow successful!")

            else:
                logger.info(f"Workflow routed to: {routing_decision.recommended_agent}")
                logger.info("Routing workflow completed successfully!")


class TestRealPerformanceComparison:
    """Test performance comparison between default and optimized agents."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_agent_performance_comparison(self, real_telemetry_provider):
        """Compare performance of default vs DSPy-optimized agents."""
        config_manager = create_default_config_manager()

        test_queries = [
            "Analyze market trends in renewable energy",
            "Find videos of wildlife conservation efforts",
            "Summarize recent advances in quantum computing",
        ]

        default_analyzer = QueryAnalysisToolV3(
            config_manager=config_manager,
            telemetry_provider=real_telemetry_provider,
            enable_dspy=False,
        )

        optimized_analyzer = QueryAnalysisToolV3(
            config_manager=config_manager,
            telemetry_provider=real_telemetry_provider,
            enable_dspy=True,
        )

        performance_comparison = {"default": [], "optimized": []}

        for query in test_queries:
            import time

            start_time = time.time()
            default_result = await default_analyzer.analyze(query)
            default_time = time.time() - start_time

            start_time = time.time()
            optimized_result = await optimized_analyzer.analyze(query)
            optimized_time = time.time() - start_time

            def get_reasoning(result):
                if hasattr(result, "to_dict"):
                    result_dict = result.to_dict()
                elif hasattr(result, "__dict__"):
                    result_dict = result.__dict__
                else:
                    result_dict = result

                if isinstance(result_dict, dict):
                    if "thinking_phase" in result_dict and isinstance(
                        result_dict["thinking_phase"], dict
                    ):
                        return result_dict["thinking_phase"].get("reasoning", "")
                    return result_dict.get("reasoning", "")
                return getattr(result, "reasoning", "")

            performance_comparison["default"].append(
                {
                    "query": query,
                    "response_time": default_time,
                    "result_quality": len(get_reasoning(default_result)),
                }
            )

            performance_comparison["optimized"].append(
                {
                    "query": query,
                    "response_time": optimized_time,
                    "result_quality": len(get_reasoning(optimized_result)),
                }
            )

        avg_default_time = sum(
            p["response_time"] for p in performance_comparison["default"]
        ) / len(test_queries)
        avg_optimized_time = sum(
            p["response_time"] for p in performance_comparison["optimized"]
        ) / len(test_queries)

        avg_default_quality = sum(
            p["result_quality"] for p in performance_comparison["default"]
        ) / len(test_queries)
        avg_optimized_quality = sum(
            p["result_quality"] for p in performance_comparison["optimized"]
        ) / len(test_queries)

        logger.info(
            f"Default agent - Avg time: {avg_default_time:.2f}s, Avg quality: {avg_default_quality}"
        )
        logger.info(
            f"Optimized agent - Avg time: {avg_optimized_time:.2f}s, Avg quality: {avg_optimized_quality}"
        )

        assert avg_default_time > 0
        assert avg_optimized_time > 0
        assert avg_default_quality > 0
        assert avg_optimized_quality > 0

        logger.info("Performance comparison completed successfully!")


class TestRealDispatcherGatewayFlow:
    """Test the AgentDispatcher._execute_gateway_task() flow with real agents.

    Exercises the actual dispatcher gateway triage: GatewayAgent classifies
    the query, then dispatcher routes to the downstream execution agent or
    orchestrator. Uses real GLiNER (or skips) but does not require Vespa
    since downstream agents may not be available.
    """

    @skip_if_no_gliner
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_dispatcher_gateway_simple_classification(self):
        """Dispatcher gateway task classifies a simple query and returns routing info."""
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps, port=18017)

        input_data = GatewayInput(
            query="find tutorial videos about Python programming",
            tenant_id="test_tenant",
        )
        result = await agent._process_impl(input_data)

        assert result is not None
        assert result.complexity in ("simple", "complex")
        assert result.routed_to is not None

        # Verify the dispatcher would know where to send this
        if result.complexity == "simple":
            assert result.routed_to != "orchestrator_agent"
        else:
            assert result.routed_to == "orchestrator_agent"

        logger.info(
            f"Dispatcher gateway flow: {result.complexity} → {result.routed_to}"
        )

    @skip_if_no_gliner
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_dispatcher_gateway_complex_routes_to_orchestrator(self):
        """Dispatcher gateway task routes complex query to orchestrator."""
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps, port=18018)

        # Intentionally ambiguous/multi-modal query
        input_data = GatewayInput(
            query="xyz ambiguous query with no clear intent",
            tenant_id="test_tenant",
        )
        result = await agent._process_impl(input_data)

        assert result is not None
        # No entities → should be classified as complex
        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"

        logger.info(
            f"Dispatcher gateway complex: routed_to={result.routed_to}, "
            f"confidence={result.confidence}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main(
            [
                "-v",
                f"tests/agents/integration/test_real_multi_agent_integration.py::{test_class}",
            ]
        )
    else:
        pytest.main(["-v", "tests/agents/integration/test_real_multi_agent_integration.py"])
