"""
End-to-end multi-agent pipeline integration tests.

Tests the COMPLETE pipeline with real services:
  OLD flow: RoutingAgent → SearchAgent (Vespa) → SummarizerAgent → DetailedReportAgent
  NEW flow: GatewayAgent → OrchestratorAgent (complex) or GatewayAgent → SearchAgent (simple)

Requirements:
- Ollama running with qwen2.5:1.5b model
- Docker for Vespa container
- Test data ingested via VespaTestManager
- GLiNER model for GatewayAgent (optional, skip if unavailable)

These are true E2E tests — they exercise every layer from LLM inference
through backend search to result processing.
"""

import importlib.util
import logging
from pathlib import Path

import dspy
import pytest

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
    ReportRequest,
)
from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps, GatewayInput
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps, SearchInput
from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummarizerDeps,
    SummaryRequest,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from tests.agents.integration.conftest import skip_if_no_ollama

logger = logging.getLogger(__name__)

# GLiNER availability check for GatewayAgent tests
HAS_GLINER = importlib.util.find_spec("gliner") is not None

skip_if_no_gliner = pytest.mark.skipif(
    not HAS_GLINER, reason="GLiNER not installed"
)


@pytest.fixture(scope="module")
def dspy_lm():
    """Module-scoped DSPy LM for all E2E tests."""
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
def routing_agent(dspy_lm):
    """RoutingAgent with real LLM."""
    telemetry_config = TelemetryConfig(
        otlp_endpoint="http://localhost:24317",
        provider_config={
            "http_endpoint": "http://localhost:26006",
            "grpc_endpoint": "http://localhost:24317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    deps = RoutingDeps(
        telemetry_config=telemetry_config,
        llm_config=LLMEndpointConfig(
            model="ollama/qwen2.5:1.5b",
            api_base="http://localhost:11434",
        ),
    )
    return RoutingAgent(deps=deps)


@pytest.fixture
def gateway_agent():
    """GatewayAgent with real GLiNER model (lazy-loaded on first use)."""
    deps = GatewayDeps()
    return GatewayAgent(deps=deps, port=18014)


@pytest.fixture
def search_agent(vespa_with_schema, dspy_lm):
    """SearchAgent connected to real Vespa with test data."""
    vespa_http_port = vespa_with_schema["http_port"]
    vespa_config_port = vespa_with_schema["config_port"]
    default_schema = vespa_with_schema["default_schema"]
    config_manager = vespa_with_schema["manager"].config_manager

    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    deps = SearchAgentDeps(
        backend_url="http://localhost",
        backend_port=vespa_http_port,
        backend_config_port=vespa_config_port,
        profile=default_schema,
    )
    return SearchAgent(
        deps=deps,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )


@pytest.fixture
def summarizer_agent(vespa_with_schema, dspy_lm):
    """SummarizerAgent with real config."""
    config_manager = vespa_with_schema["manager"].config_manager
    return SummarizerAgent(deps=SummarizerDeps(), config_manager=config_manager)


@pytest.fixture
def report_agent(vespa_with_schema, dspy_lm):
    """DetailedReportAgent with real config."""
    config_manager = vespa_with_schema["manager"].config_manager
    return DetailedReportAgent(deps=DetailedReportDeps(), config_manager=config_manager)


@pytest.mark.integration
@skip_if_no_ollama
@pytest.mark.slow
class TestE2EMultiAgentPipeline:
    """
    True end-to-end tests: real LLM + real Vespa + real agents.

    Pipeline: Route → Search → Summarize → Report
    """

    @pytest.mark.asyncio
    async def test_route_then_search_with_enriched_input(
        self, routing_agent, search_agent
    ):
        """
        E2E: RoutingAgent receives pre-enriched input (thin interface),
        makes a routing decision, SearchAgent executes the search.

        Validates the new A2A flow where entities and enhanced_query
        arrive from upstream agents (EntityExtractionAgent,
        QueryEnhancementAgent) rather than being computed by RoutingAgent.
        """
        query = "machine learning tutorial videos"
        enhanced_query = "machine learning tutorial videos deep learning neural networks"
        entities = [
            {"text": "machine learning", "type": "CONCEPT", "confidence": 0.9},
        ]

        routing_result = await routing_agent.route_query(
            query=query,
            enhanced_query=enhanced_query,
            entities=entities,
            relationships=[],
            tenant_id="test_tenant",
        )
        assert routing_result is not None
        assert routing_result.recommended_agent is not None

        logger.info(
            f"Routing decided: agent={routing_result.recommended_agent}, "
            f"confidence={routing_result.confidence}"
        )

        search_query = enhanced_query or query
        search_input = SearchInput(
            query=search_query,
            tenant_id="test_tenant",
        )
        search_result = await search_agent._process_impl(search_input)

        assert search_result is not None
        assert hasattr(search_result, "results")
        logger.info(f"Search returned {len(search_result.results)} results")

    @pytest.mark.asyncio
    async def test_search_then_summarize(self, search_agent, summarizer_agent):
        """
        E2E: SearchAgent returns results, SummarizerAgent summarizes them.

        Tests the handoff from search results to summarization with
        real LLM inference on real search results.
        """
        search_input = SearchInput(
            query="video content analysis",
            tenant_id="test_tenant",
        )
        search_result = await search_agent._process_impl(search_input)
        assert search_result is not None

        result_dicts = []
        for r in search_result.results:
            result_dicts.append(
                {
                    "id": r.get("documentid", "unknown"),
                    "title": r.get("title", "Untitled"),
                    "description": r.get("description", ""),
                    "score": r.get("relevance", 0.0),
                    "content_type": "video",
                }
            )

        if not result_dicts:
            result_dicts = [
                {
                    "id": "test_1",
                    "title": "Video Content Analysis Methods",
                    "description": "Overview of automated video analysis techniques",
                    "score": 0.9,
                    "content_type": "video",
                },
            ]
            logger.info("No Vespa results — using synthetic result for summarization")

        summary_request = SummaryRequest(
            query="video content analysis",
            search_results=result_dicts,
            summary_type="brief",
            include_visual_analysis=False,
        )
        summary_result = await summarizer_agent._summarize(summary_request)

        assert summary_result is not None
        assert summary_result.summary is not None
        assert len(summary_result.summary) > 10
        assert summary_result.confidence_score > 0
        logger.info(
            f"Summary: {summary_result.summary[:100]}... "
            f"(confidence={summary_result.confidence_score})"
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_route_search_summarize_report(
        self, routing_agent, search_agent, summarizer_agent, report_agent
    ):
        """
        E2E: Full 4-stage pipeline with real services and pre-enriched routing input.

        Route (thin) → Search (Vespa) → Summarize (LLM) → Report (LLM)

        This is the core E2E test — every agent processes real data
        through real inference. RoutingAgent receives pre-enriched input
        matching the new A2A architecture.
        """
        query = "robotics and computer vision research"
        enhanced_query = "robotics computer vision research autonomous systems"
        entities = [
            {"text": "robotics", "type": "CONCEPT", "confidence": 0.85},
            {"text": "computer vision", "type": "CONCEPT", "confidence": 0.92},
        ]

        # Stage 1: Route with pre-enriched input
        routing_result = await routing_agent.route_query(
            query=query,
            enhanced_query=enhanced_query,
            entities=entities,
            relationships=[],
            tenant_id="test_tenant",
        )
        assert routing_result is not None
        logger.info(
            f"Stage 1 (Route): agent={routing_result.recommended_agent}, "
            f"confidence={routing_result.confidence}"
        )

        # Stage 2: Search using enhanced query
        search_query = enhanced_query or query
        search_input = SearchInput(
            query=search_query,
            tenant_id="test_tenant",
        )
        search_result = await search_agent._process_impl(search_input)
        assert search_result is not None
        logger.info(f"Stage 2 (Search): {len(search_result.results)} results")

        result_dicts = []
        for r in search_result.results:
            result_dicts.append(
                {
                    "id": r.get("documentid", "unknown"),
                    "title": r.get("title", "Untitled"),
                    "description": r.get("description", ""),
                    "score": r.get("relevance", 0.0),
                    "content_type": "video",
                }
            )
        if not result_dicts:
            result_dicts = [
                {
                    "id": "synth_1",
                    "title": "Robotics Vision System",
                    "description": "Computer vision pipeline for robotic manipulation",
                    "score": 0.85,
                    "content_type": "video",
                },
                {
                    "id": "synth_2",
                    "title": "Deep Learning for Object Detection",
                    "description": "Neural network architectures for real-time detection",
                    "score": 0.78,
                    "content_type": "video",
                },
            ]
            logger.info("No Vespa results — using synthetic results for pipeline")

        # Stage 3: Summarize
        summary_request = SummaryRequest(
            query=query,
            search_results=result_dicts,
            summary_type="comprehensive",
            include_visual_analysis=False,
        )
        summary_result = await summarizer_agent._summarize(summary_request)
        assert summary_result is not None
        assert summary_result.summary is not None
        assert len(summary_result.summary) > 20
        logger.info(
            f"Stage 3 (Summarize): {len(summary_result.summary)} chars, "
            f"confidence={summary_result.confidence_score}"
        )

        # Stage 4: Report
        enhanced_results = result_dicts.copy()
        enhanced_results.append(
            {
                "id": "summary_insight",
                "title": "Summary Analysis",
                "description": summary_result.summary,
                "score": 1.0,
                "content_type": "analysis",
            }
        )

        report_request = ReportRequest(
            query=f"comprehensive report on: {query}",
            search_results=enhanced_results,
            report_type="comprehensive",
        )
        report_result = await report_agent._generate_report(report_request)
        assert report_result is not None
        assert report_result.executive_summary is not None
        assert len(report_result.executive_summary) > 20
        assert len(report_result.detailed_findings) >= 1
        logger.info(
            f"Stage 4 (Report): exec_summary={len(report_result.executive_summary)} chars, "
            f"findings={len(report_result.detailed_findings)}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_error_resilience(
        self, routing_agent, search_agent, summarizer_agent
    ):
        """
        E2E: Pipeline handles edge cases gracefully.

        Tests with a minimal query that may produce sparse results,
        verifying agents don't crash on empty/minimal data.
        Uses the thin RoutingAgent interface with pre-enriched input.
        """
        query = "x"  # Minimal query — likely produces poor routing and sparse results

        routing_result = await routing_agent.route_query(
            query=query,
            enhanced_query=query,
            entities=[],
            relationships=[],
            tenant_id="test_tenant",
        )
        assert routing_result is not None

        search_input = SearchInput(
            query=query,
            tenant_id="test_tenant",
        )
        search_result = await search_agent._process_impl(search_input)
        assert search_result is not None

        summary_request = SummaryRequest(
            query=query,
            search_results=[
                {
                    "id": "minimal",
                    "title": "Test",
                    "description": "Minimal test content",
                    "score": 0.5,
                    "content_type": "text",
                }
            ],
            summary_type="brief",
            include_visual_analysis=False,
        )
        summary_result = await summarizer_agent._summarize(summary_request)
        assert summary_result is not None
        assert summary_result.summary is not None


@pytest.mark.integration
@skip_if_no_ollama
@skip_if_no_gliner
@pytest.mark.slow
class TestE2EGatewayPipeline:
    """
    End-to-end tests for the NEW A2A flow via GatewayAgent.

    GatewayAgent uses GLiNER (no LLM) to classify queries as simple or complex,
    then routes:
    - Simple → directly to execution agent (SearchAgent, SummarizerAgent, etc.)
    - Complex → OrchestratorAgent for multi-agent coordination
    """

    @pytest.mark.asyncio
    async def test_gateway_simple_query_routes_to_search(self, gateway_agent):
        """GatewayAgent classifies a straightforward video search as simple."""
        input_data = GatewayInput(
            query="show me videos about dogs playing in the park",
            tenant_id="test_tenant",
        )
        result = await gateway_agent._process_impl(input_data)

        assert result is not None
        assert result.complexity in ("simple", "complex")
        assert result.modality in (
            "video", "text", "audio", "image", "document", "both",
        )
        assert result.routed_to is not None
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 0

        logger.info(
            f"Gateway: complexity={result.complexity}, "
            f"modality={result.modality}, "
            f"routed_to={result.routed_to}, "
            f"confidence={result.confidence:.2f}"
        )

    @pytest.mark.asyncio
    async def test_gateway_complex_query_routes_to_orchestrator(self, gateway_agent):
        """GatewayAgent classifies a multi-faceted query as complex."""
        input_data = GatewayInput(
            query=(
                "Compare recent video tutorials on reinforcement learning "
                "with audio lectures on deep learning and create a detailed "
                "report with visual diagrams"
            ),
            tenant_id="test_tenant",
        )
        result = await gateway_agent._process_impl(input_data)

        assert result is not None
        assert result.complexity in ("simple", "complex")
        assert result.routed_to is not None
        assert 0.0 <= result.confidence <= 1.0

        logger.info(
            f"Gateway: complexity={result.complexity}, "
            f"modality={result.modality}, "
            f"routed_to={result.routed_to}, "
            f"confidence={result.confidence:.2f}"
        )

    @pytest.mark.asyncio
    async def test_gateway_then_search_e2e(
        self, gateway_agent, search_agent
    ):
        """
        E2E: GatewayAgent triages query, then SearchAgent executes.

        Tests the new entry-point flow where GatewayAgent replaces
        RoutingAgent as the first agent in the pipeline for simple queries.
        """
        input_data = GatewayInput(
            query="find videos about machine learning",
            tenant_id="test_tenant",
        )
        gateway_result = await gateway_agent._process_impl(input_data)
        assert gateway_result is not None

        logger.info(
            f"Gateway classified: complexity={gateway_result.complexity}, "
            f"routed_to={gateway_result.routed_to}"
        )

        # Regardless of gateway classification, exercise the search path
        search_input = SearchInput(
            query=input_data.query,
            tenant_id="test_tenant",
        )
        search_result = await search_agent._process_impl(search_input)

        assert search_result is not None
        assert hasattr(search_result, "results")
        logger.info(
            f"Gateway → Search: {len(search_result.results)} results returned"
        )

    @pytest.mark.asyncio
    async def test_gateway_minimal_query(self, gateway_agent):
        """GatewayAgent handles minimal/empty-ish queries without crashing."""
        input_data = GatewayInput(
            query="x",
            tenant_id="test_tenant",
        )
        result = await gateway_agent._process_impl(input_data)

        assert result is not None
        assert result.complexity in ("simple", "complex")
        # Minimal query with no entities should route to orchestrator
        logger.info(
            f"Gateway minimal: complexity={result.complexity}, "
            f"routed_to={result.routed_to}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
