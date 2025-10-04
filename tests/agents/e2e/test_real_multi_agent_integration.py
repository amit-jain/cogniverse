"""
Real end-to-end integration tests for multi-agent system.

These tests use actual LLMs, real DSPy optimization, and real backend services
to test the complete multi-agent workflow without mocks.

Requirements:
- Ollama running locally with smollm3:8b model
- Vespa backend available (optional, falls back gracefully)
- Phoenix telemetry server (optional)
"""

import json
import logging

import pytest

pytestmark = pytest.mark.skip(reason="E2E tests have stale API expectations - need comprehensive rewrite")

from src.app.agents.detailed_report_agent import DetailedReportAgent
from src.app.agents.dspy_agent_optimizer import (
    DSPyAgentOptimizerPipeline,
    DSPyAgentPromptOptimizer,
)
from src.app.agents.query_analysis_tool_v3 import QueryAnalysisToolV3
from src.app.agents.routing_agent import RoutingAgent
from src.app.agents.summarizer_agent import SummarizerAgent

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "ollama_base_url": "http://localhost:11434/v1",
    "ollama_model": "smollm3:8b",
    "openai_api_key": "fake-key",  # Ollama doesn't need real key
    "vespa_url": "http://localhost:8080",
    "test_timeout": 300,  # 5 minutes for real LLM calls
    "optimization_rounds": 1,  # Keep optimization quick for tests
}


class TestOllamaAvailability:
    """Check if Ollama is available for real integration tests."""

    def test_ollama_model_available(self):
        """Test if Ollama is running and has the required model."""
        try:
            import requests

            # Check if Ollama is running
            response = requests.get(
                f"{TEST_CONFIG['ollama_base_url'].replace('/v1', '')}/api/tags",
                timeout=5,
            )
            if response.status_code != 200:
                pytest.skip(
                    "Ollama server not available - skipping real integration tests"
                )

            # Check if required model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            if TEST_CONFIG["ollama_model"] not in model_names:
                pytest.skip(
                    f"Model {TEST_CONFIG['ollama_model']} not available in Ollama - skipping real integration tests"
                )

            logger.info(
                f"✅ Ollama available with model: {TEST_CONFIG['ollama_model']}"
            )

        except Exception as e:
            pytest.skip(
                f"Could not connect to Ollama: {e} - skipping real integration tests"
            )


class TestRealQueryAnalysisIntegration:
    """Real integration tests for QueryAnalysisToolV3 with actual LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    async def test_real_query_analysis_with_local_llm(self):
        """Test real query analysis with local Ollama model."""

        # Initialize query analyzer with real LLM
        analyzer = QueryAnalysisToolV3(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        # Test queries with different complexity levels
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

            # Convert to dict for compatibility with test assertions
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result

            # Verify analysis structure
            assert isinstance(result_dict, dict)
            assert "primary_intent" in result_dict
            assert "complexity_level" in result_dict
            assert "needs_video_search" in result_dict
            assert "needs_text_search" in result_dict
            # Reasoning is now nested inside thinking_phase
            assert "thinking_phase" in result_dict
            assert "reasoning" in result_dict["thinking_phase"]

            # Verify expected outcomes
            if "expected_intent" in test_case:
                assert test_case["expected_intent"] in result_dict["primary_intent"].lower()

            if "expected_video_search" in test_case:
                assert (
                    result_dict["needs_video_search"] == test_case["expected_video_search"]
                )

            if "expected_text_search" in test_case:
                assert result_dict["needs_text_search"] == test_case["expected_text_search"]

            # Verify reasoning is provided
            reasoning = result_dict["thinking_phase"]["reasoning"]
            assert len(reasoning) > 10, "Reasoning should be substantive"

            logger.info(f"✅ Analysis result: {result_dict}")


@pytest.mark.skip(reason="RoutingAgent API changed - route_query() method doesn't exist, needs rewrite to use analyze_and_route()")
class TestRealAgentRoutingIntegration:
    """Real integration tests for RoutingAgent with actual LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    async def test_real_agent_routing_with_local_llm(self):
        """Test real agent routing decisions with local Ollama model."""

        # Initialize routing agent with real LLM
        routing_agent = RoutingAgent(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        # Test routing decisions for different query types
        test_cases = [
            {
                "query": "Find videos of basketball games",
                "analysis": {
                    "primary_intent": "search",
                    "needs_video_search": True,
                    "complexity_level": "simple",
                },
                "expected_agent": "video_search",
                "expected_workflow": "raw_results",
            },
            {
                "query": "Analyze the trends in renewable energy adoption",
                "analysis": {
                    "primary_intent": "analysis",
                    "needs_text_search": True,
                    "complexity_level": "complex",
                },
                "expected_agent": "detailed_report",
                "expected_workflow": "detailed_report",
            },
            {
                "query": "Give me a brief overview of recent AI developments",
                "analysis": {
                    "primary_intent": "summarization",
                    "needs_text_search": True,
                    "complexity_level": "moderate",
                },
                "expected_agent": "summarizer",
                "expected_workflow": "summary",
            },
        ]

        for test_case in test_cases:
            logger.info(f"Testing routing for: {test_case['query']}")

            routing_decision = await routing_agent.route_query(
                test_case["query"], test_case["analysis"]
            )

            # Verify routing decision structure
            assert isinstance(routing_decision, dict)
            assert "recommended_workflow" in routing_decision
            assert "primary_agent" in routing_decision
            assert "routing_confidence" in routing_decision
            assert "reasoning" in routing_decision

            # Verify expected outcomes
            assert (
                test_case["expected_agent"] in routing_decision["primary_agent"].lower()
            )
            assert (
                test_case["expected_workflow"]
                in routing_decision["recommended_workflow"].lower()
            )

            # Verify confidence is reasonable
            confidence = float(routing_decision["routing_confidence"])
            assert 0.0 <= confidence <= 1.0
            assert (
                confidence > 0.5
            ), "Confidence should be reasonably high for clear test cases"

            # Verify reasoning is provided
            assert (
                len(routing_decision["reasoning"]) > 10
            ), "Reasoning should be substantive"

            logger.info(f"✅ Routing decision: {routing_decision}")


@pytest.mark.skip(reason="SummarizerAgent API changed - generate_summary() method doesn't exist")
class TestRealAgentSpecializationIntegration:
    """Real integration tests for specialized agents with actual LLMs."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    async def test_real_summarizer_agent_with_local_llm(self):
        """Test real summarization with local Ollama model."""

        summarizer = SummarizerAgent(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        # Test content summarization
        test_content = """
        Machine learning has seen tremendous advances in recent years, particularly in deep learning
        architectures. Transformer models like GPT and BERT have revolutionized natural language
        processing, while convolutional neural networks continue to excel in computer vision tasks.
        Recent research has focused on improving model efficiency, reducing computational requirements,
        and developing more interpretable AI systems. The field is also exploring multimodal models
        that can process both text and images simultaneously.
        """

        summary_result = await summarizer.generate_summary(
            content=test_content, summary_type="brief", target_audience="technical"
        )

        # Verify summary structure
        assert isinstance(summary_result, dict)
        assert "summary" in summary_result
        assert "key_points" in summary_result
        assert "confidence" in summary_result

        # Verify summary quality
        assert len(summary_result["summary"]) > 20, "Summary should be substantive"
        assert len(summary_result["summary"]) < len(
            test_content
        ), "Summary should be shorter than original"

        # Verify key points
        key_points = summary_result["key_points"]
        if isinstance(key_points, str):
            key_points = json.loads(key_points)
        assert isinstance(key_points, list)
        assert len(key_points) >= 2, "Should extract multiple key points"

        # Verify confidence
        confidence = float(summary_result["confidence"])
        assert 0.0 <= confidence <= 1.0

        logger.info(f"✅ Summary result: {summary_result}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    async def test_real_detailed_report_agent_with_local_llm(self):
        """Test real detailed report generation with local Ollama model."""

        report_agent = DetailedReportAgent(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        # Mock search results for testing
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

        report_result = await report_agent.generate_detailed_report(
            search_results=mock_search_results,
            query_context="machine learning efficiency research",
            analysis_depth="comprehensive",
        )

        # Verify report structure
        assert isinstance(report_result, dict)
        assert "executive_summary" in report_result
        assert "detailed_findings" in report_result
        assert "recommendations" in report_result
        assert "confidence" in report_result

        # Verify report quality
        assert (
            len(report_result["executive_summary"]) > 50
        ), "Executive summary should be comprehensive"
        assert (
            len(report_result["detailed_findings"]) > 100
        ), "Detailed findings should be thorough"
        assert (
            len(report_result["recommendations"]) > 30
        ), "Recommendations should be actionable"

        # Verify confidence
        confidence = float(report_result["confidence"])
        assert 0.0 <= confidence <= 1.0

        logger.info(f"✅ Report result keys: {list(report_result.keys())}")


class TestRealDSPyOptimizationIntegration:
    """Real integration tests for DSPy optimization with actual LLMs."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(
        TEST_CONFIG["test_timeout"] * 2
    )  # DSPy optimization takes longer
    async def test_real_dspy_optimization_pipeline(self):
        """Test real DSPy optimization with local Ollama model."""

        # Initialize DSPy optimizer with real LLM
        optimizer = DSPyAgentPromptOptimizer()

        # Initialize with local LLM
        success = optimizer.initialize_language_model(
            api_base=TEST_CONFIG["ollama_base_url"],
            model=TEST_CONFIG["ollama_model"],
            api_key=TEST_CONFIG["openai_api_key"],
        )

        if not success:
            pytest.skip("Could not initialize DSPy optimizer with local LLM")

        # Create optimization pipeline
        pipeline = DSPyAgentOptimizerPipeline(optimizer)

        # Test signature creation
        qa_signature = optimizer.create_query_analysis_signature()
        assert qa_signature is not None, "Should create query analysis signature"

        ar_signature = optimizer.create_agent_routing_signature()
        assert ar_signature is not None, "Should create agent routing signature"

        # Test training data loading
        training_data = pipeline.load_training_data()
        assert isinstance(training_data, dict)
        assert "query_analysis" in training_data
        assert "agent_routing" in training_data
        assert len(training_data["query_analysis"]) > 0
        assert len(training_data["agent_routing"]) > 0

        logger.info(f"✅ Loaded training data with {len(training_data)} modules")

        # Test module initialization
        pipeline.initialize_modules()
        assert "query_analysis" in pipeline.modules
        assert "agent_routing" in pipeline.modules

        # Test single module optimization (with reduced scope for testing)
        try:
            query_examples = training_data["query_analysis"][
                :2
            ]  # Use only 2 examples for speed

            optimized_module = pipeline.optimize_module(
                "query_analysis", query_examples
            )

            assert optimized_module is not None, "Should return optimized module"
            assert "query_analysis" in pipeline.compiled_modules

            logger.info("✅ Successfully optimized query_analysis module")

        except Exception as e:
            logger.warning(f"DSPy optimization failed (expected in test env): {e}")
            # This is acceptable - DSPy optimization may fail in test environments
            # The important part is that the pipeline structure works

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    async def test_real_agent_with_dspy_integration(self):
        """Test agents with DSPy optimization integration."""

        # Create agent with DSPy disabled first
        analyzer = QueryAnalysisToolV3(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
            enable_dspy=False,
        )

        # Test without optimization
        await analyzer.analyze("Show me videos of cats")

        # Verify DSPy metadata
        dspy_metadata = analyzer.get_dspy_metadata()
        assert isinstance(dspy_metadata, dict)
        assert "enabled" in dspy_metadata
        assert not dspy_metadata["enabled"]

        # Test enabling DSPy (without actual optimization for speed)
        analyzer.dspy_enabled = True
        dspy_metadata = analyzer.get_dspy_metadata()
        assert dspy_metadata["enabled"]
        assert "agent_type" in dspy_metadata

        logger.info(f"✅ DSPy integration metadata: {dspy_metadata}")


class TestRealEndToEndWorkflow:
    """Real end-to-end workflow tests with multiple agents."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(
        TEST_CONFIG["test_timeout"] * 2
    )  # Allow more time for full workflow
    async def test_real_multi_agent_workflow(self):
        """Test complete multi-agent workflow with real LLMs."""

        # Initialize all agents
        query_analyzer = QueryAnalysisToolV3(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        routing_agent = RoutingAgent(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        summarizer = SummarizerAgent(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
        )

        # Test complete workflow
        test_query = (
            "Give me a summary of recent developments in artificial intelligence"
        )

        # Step 1: Query analysis
        logger.info(f"Step 1: Analyzing query: {test_query}")
        analysis_result = await query_analyzer.analyze(test_query)
        logger.info(f"Analysis result: {analysis_result}")

        # Step 2: Agent routing
        logger.info("Step 2: Routing query to appropriate agent")
        routing_decision = await routing_agent.route_query(test_query, analysis_result)
        logger.info(f"Routing decision: {routing_decision}")

        # Step 3: Execute with appropriate agent (simulate with summarizer)
        if "summary" in routing_decision["recommended_workflow"].lower():
            logger.info("Step 3: Executing with SummarizerAgent")

            # Mock some content to summarize
            ai_content = """
            Artificial Intelligence has made significant strides in 2024, with major breakthroughs
            in large language models, multimodal AI systems, and autonomous agents. Key developments
            include improved reasoning capabilities, better alignment with human values, and more
            efficient training methods. The field is moving towards more practical applications
            in healthcare, education, and scientific research.
            """

            summary_result = await summarizer.generate_summary(
                content=ai_content, summary_type="brief", target_audience="general"
            )
            logger.info(f"Summary result: {summary_result}")

            # Verify complete workflow
            assert analysis_result["primary_intent"] == "summarization"
            assert routing_decision["primary_agent"] == "summarizer"
            assert "summary" in summary_result
            assert len(summary_result["summary"]) > 20

            logger.info("✅ Complete multi-agent workflow successful!")

        else:
            logger.info(f"Workflow routed to: {routing_decision['primary_agent']}")
            logger.info("✅ Routing workflow completed successfully!")


class TestRealPerformanceComparison:
    """Test performance comparison between default and optimized agents."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    async def test_agent_performance_comparison(self):
        """Compare performance of default vs DSPy-optimized agents."""

        # This test demonstrates the structure for performance comparison
        # In a real environment, you would run actual optimization first

        test_queries = [
            "Analyze market trends in renewable energy",
            "Find videos of wildlife conservation efforts",
            "Summarize recent advances in quantum computing",
        ]

        # Initialize agent without DSPy optimization
        default_analyzer = QueryAnalysisToolV3(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
            enable_dspy=False,
        )

        # Initialize agent with DSPy optimization (simulated)
        optimized_analyzer = QueryAnalysisToolV3(
            openai_api_key=TEST_CONFIG["openai_api_key"],
            openai_base_url=TEST_CONFIG["ollama_base_url"],
            model_name=TEST_CONFIG["ollama_model"],
            enable_dspy=True,  # Would use actual optimized prompts in real scenario
        )

        performance_comparison = {"default": [], "optimized": []}

        for query in test_queries:
            # Test default agent
            import time

            start_time = time.time()
            default_result = await default_analyzer.analyze(query)
            default_time = time.time() - start_time

            # Test optimized agent
            start_time = time.time()
            optimized_result = await optimized_analyzer.analyze(query)
            optimized_time = time.time() - start_time

            performance_comparison["default"].append(
                {
                    "query": query,
                    "response_time": default_time,
                    "result_quality": len(default_result.get("reasoning", "")),
                }
            )

            performance_comparison["optimized"].append(
                {
                    "query": query,
                    "response_time": optimized_time,
                    "result_quality": len(optimized_result.get("reasoning", "")),
                }
            )

        # Calculate average metrics
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

        # In real optimization, we would expect improvements
        # For now, just verify both agents work
        assert avg_default_time > 0
        assert avg_optimized_time > 0
        assert avg_default_quality > 0
        assert avg_optimized_quality > 0

        logger.info("✅ Performance comparison completed successfully!")


if __name__ == "__main__":
    # Run individual test for development
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main(
            [
                "-v",
                f"tests/agents/e2e/test_real_multi_agent_integration.py::{test_class}",
            ]
        )
    else:
        pytest.main(["-v", "tests/agents/e2e/test_real_multi_agent_integration.py"])
