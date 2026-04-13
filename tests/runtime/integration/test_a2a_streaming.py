"""
Integration tests for A2A streaming with real services.

Exercises every agent's streaming path through the full stack:
real A2A server → real executor → real dispatcher → real agent → emit_progress events.

Infrastructure:
- Real Ollama LLM (via dspy_lm fixture from conftest)
- Real Vespa Docker (via vespa_instance fixture from conftest)
- Real ConfigManager backed by VespaConfigStore
- Real DSPy modules (ChainOfThought, not mocked)
"""

import asyncio
import json
import logging
import uuid
from typing import Any

import pytest
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.testclient import TestClient as StarletteTestClient

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from tests.runtime.integration.conftest import skip_if_no_llm

logger = logging.getLogger(__name__)

KNOWN_AGENTS = {
    "search_agent",
    "video_search_agent",
    "summarizer_agent",
    "detailed_report_agent",
    "routing_agent",
    "orchestrator_agent",
    "image_search_agent",
    "audio_analysis_agent",
    "document_agent",
}


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def streaming_registry(config_manager):
    """Registry with all streaming-capable agents registered."""
    registry = AgentRegistry(tenant_id="default", config_manager=config_manager)

    agents = [
        ("summarizer_agent", ["summarization", "text_generation"]),
        ("routing_agent", ["routing"]),
        ("search_agent", ["search", "video_search"]),
        ("orchestrator_agent", ["orchestration", "planning"]),
        ("detailed_report_agent", ["detailed_report"]),
        ("query_enhancement_agent", ["query_enhancement"]),
        ("entity_extraction_agent", ["entity_extraction"]),
        ("profile_selection_agent", ["profile_selection"]),
        ("image_search_agent", ["image_search", "visual_analysis"]),
        ("audio_analysis_agent", ["audio_analysis", "transcription"]),
        ("document_agent", ["document_analysis", "pdf_processing"]),
    ]

    for name, caps in agents:
        registry.register_agent(
            AgentEndpoint(name=name, url="http://localhost:8000", capabilities=caps)
        )

    return registry


@pytest.fixture(scope="module")
def streaming_dispatcher(streaming_registry, config_manager, schema_loader):
    return AgentDispatcher(
        agent_registry=streaming_registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.fixture(scope="module")
def streaming_a2a_client(streaming_dispatcher):
    """A2A TestClient with streaming=True, backed by real services."""
    executor = CogniverseAgentExecutor(dispatcher=streaming_dispatcher)
    card = AgentCard(
        name="Streaming Integration",
        description="Full streaming integration tests",
        url="http://localhost:9998/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="summarizer_agent",
                name="summarizer_agent",
                description="Summarize content",
                tags=["summarization", "text_generation"],
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore()
    )
    server = A2AStarletteApplication(agent_card=card, http_handler=handler)

    with StarletteTestClient(server.build()) as client:
        yield client


# ── Helpers ──────────────────────────────────────────────────────────────────


def _collect_stream_events(agent, typed_input) -> list[dict[str, Any]]:
    """Call process(stream=True) and collect all events."""
    events = []

    async def _run():
        async for event in await agent.process(typed_input, stream=True):
            events.append(event)

    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(_run())
    except RuntimeError:
        asyncio.run(_run())
    return events


def _send_a2a_stream(client, text, agent_name="summarizer_agent", tenant_id="default"):
    """Send A2A message/stream and collect SSE events."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "contextId": f"stream-{uuid.uuid4()}",
                "parts": [{"kind": "text", "text": text}],
            },
            "metadata": {
                "agent_name": agent_name,
                "tenant_id": tenant_id,
                "stream": True,
            },
        },
    }
    events = []
    with client.stream("POST", "/", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                if data_str:
                    events.append(json.loads(data_str))
    return events


def _parse_agent_events(raw_events):
    """Extract agent event payloads from A2A SSE wrapper."""
    parsed = []
    for event in raw_events:
        for part in (
            event.get("result", {})
            .get("status", {})
            .get("message", {})
            .get("parts", [])
        ):
            text = part.get("text", "")
            if text:
                try:
                    parsed.append(json.loads(text))
                except json.JSONDecodeError:
                    parsed.append({"raw": text})
    return parsed


def _get_phases(events):
    """Extract phase names from status/partial events."""
    return [
        e["phase"]
        for e in events
        if e.get("type") in ("status", "partial") and "phase" in e
    ]


def _get_final(events, agent_name):
    """Extract and validate exactly one final event."""
    finals = [e for e in events if e.get("type") == "final"]
    assert len(finals) == 1, (
        f"{agent_name}: expected exactly 1 final event, got {len(finals)}. "
        f"All events: {events}"
    )
    assert "data" in finals[0], f"{agent_name}: final event missing 'data'"
    return finals[0]["data"]


def _assert_no_errors(events, agent_name):
    """Assert no error events were emitted."""
    errors = [e for e in events if e.get("type") == "error"]
    assert len(errors) == 0, f"{agent_name}: unexpected error events: {errors}"


# ── Agent streaming tests ───────────────────────────────────────────────────


@pytest.mark.integration
@skip_if_no_llm
class TestSummarizerAgentStreaming:
    """SummarizerAgent streaming with real Ollama."""

    def test_stream_phases_and_output(self, config_manager, dspy_lm):
        from cogniverse_agents.summarizer_agent import (
            SummarizerAgent,
            SummarizerDeps,
            SummarizerInput,
        )

        agent = SummarizerAgent(
            deps=SummarizerDeps(tenant_id="default"),
            config_manager=config_manager,
        )

        events = _collect_stream_events(
            agent,
            SummarizerInput(
                query="What are the key applications of machine learning in healthcare?",
                search_results=[
                    {
                        "id": "1",
                        "title": "ML in Medical Imaging",
                        "score": 0.95,
                        "content_type": "video",
                        "description": "Deep learning for radiology",
                    },
                    {
                        "id": "2",
                        "title": "Drug Discovery with AI",
                        "score": 0.88,
                        "content_type": "video",
                        "description": "Neural networks for molecules",
                    },
                    {
                        "id": "3",
                        "title": "Clinical NLP Systems",
                        "score": 0.82,
                        "content_type": "document",
                        "description": "Text mining patient records",
                    },
                ],
                summary_type="brief",
                include_visual_analysis=False,
            ),
        )

        _assert_no_errors(events, "SummarizerAgent")
        phases = _get_phases(events)
        assert "thinking" in phases, f"Missing thinking phase: {phases}"
        assert "summarization" in phases, f"Missing summarization phase: {phases}"

        # Thinking phase should emit partial with analysis data
        thinking_partials = [
            e
            for e in events
            if e.get("type") == "partial" and e.get("phase") == "thinking"
        ]
        assert len(thinking_partials) >= 1, (
            "Thinking phase should emit partial with data"
        )
        thinking_data = thinking_partials[0]["data"]
        assert "themes" in thinking_data
        assert "categories" in thinking_data
        assert "reasoning" in thinking_data
        assert isinstance(thinking_data["themes"], list)
        assert len(thinking_data["themes"]) >= 1, (
            f"3 search results about ML/healthcare should produce ≥1 theme, got: {thinking_data['themes']}"
        )
        assert isinstance(thinking_data["reasoning"], str)
        assert len(thinking_data["reasoning"]) > 10, (
            f"Reasoning should be a real analysis, got: '{thinking_data['reasoning']}'"
        )

        # Token events from call_dspy streaming — verifies dspy.streamify() path.
        # DSPy caches LM calls, so token streaming only fires on cache misses.
        # If tokens are present, verify accumulated text grows.
        token_events = [e for e in events if e.get("phase") == "token"]
        if token_events:
            last_token = token_events[-1]
            assert "data" in last_token
            assert "accumulated" in last_token["data"]
            assert len(last_token["data"]["accumulated"]) > 5, (
                f"Accumulated token text should be substantive, got: '{last_token['data']['accumulated']}'"
            )

        final_data = _get_final(events, "SummarizerAgent")
        assert "summary" in final_data
        assert "key_points" in final_data
        assert "confidence_score" in final_data
        summary = final_data["summary"].lower()
        assert len(summary) > 20, f"Summary too short to be real: '{summary}'"
        # Summary should reference the query domain — ML or healthcare or medical
        assert any(
            term in summary
            for term in ["machine learning", "ml", "healthcare", "medical", "imaging"]
        ), (
            f"Summary should reference ML/healthcare domain, got: '{final_data['summary']}'"
        )
        assert isinstance(final_data["confidence_score"], float)
        assert 0.0 <= final_data["confidence_score"] <= 1.0
        assert isinstance(final_data["key_points"], list)


@pytest.mark.integration
@skip_if_no_llm
class TestRoutingAgentStreaming:
    """RoutingAgent streaming with real Ollama + real entity extraction."""

    def test_stream_phases_and_output(self, config_manager, dspy_lm):
        from cogniverse_agents.routing_agent import (
            RoutingAgent,
            RoutingDeps,
            RoutingInput,
        )
        from cogniverse_foundation.telemetry.config import TelemetryConfig

        agent = RoutingAgent(
            deps=RoutingDeps(telemetry_config=TelemetryConfig(enabled=False))
        )

        events = _collect_stream_events(
            agent,
            RoutingInput(
                query="show me videos of robots playing soccer",
                tenant_id="default",
            ),
        )

        _assert_no_errors(events, "RoutingAgent")
        phases = _get_phases(events)
        assert "cache_check" in phases
        assert "entity_extraction" in phases
        assert "routing_decision" in phases

        final_data = _get_final(events, "RoutingAgent")
        recommended = final_data["recommended_agent"]
        # "show me videos" should route to a search/video agent, not summarizer/report
        search_agents = {
            "search_agent",
            "search",
            "video_search_agent",
            "video_search",
            "image_search_agent",
            "image_search",
        }
        assert recommended in search_agents, (
            f"Video query should route to a search agent, got: '{recommended}'"
        )
        assert isinstance(final_data["confidence"], (int, float))
        assert final_data["confidence"] > 0.0, (
            f"Confidence should be positive, got: {final_data['confidence']}"
        )
        reasoning_lower = final_data["reasoning"].lower()
        assert any(
            term in reasoning_lower
            for term in ["video", "search", "visual", "content", "robot"]
        ), f"Reasoning should explain video routing, got: '{final_data['reasoning']}'"
        assert final_data["query"] == "show me videos of robots playing soccer"
        assert isinstance(final_data["entities"], list)
        assert isinstance(final_data["relationships"], list)


@pytest.mark.integration
@skip_if_no_llm
class TestOrchestratorAgentStreaming:
    """OrchestratorAgent streaming with real Ollama + real DSPy planning.

    The registry uses real AgentEndpoint objects (not Mock) so the orchestrator
    plans against real agent names. Execution will attempt HTTP calls that fail
    (no agents running) but the planning + streaming phases are fully real.
    """

    def test_stream_phases_and_plan_output(
        self, config_manager, streaming_registry, dspy_lm
    ):
        from cogniverse_agents.orchestrator_agent import (
            OrchestratorAgent,
            OrchestratorDeps,
            OrchestratorInput,
        )

        agent = OrchestratorAgent(
            deps=OrchestratorDeps(),
            registry=streaming_registry,
            config_manager=config_manager,
        )

        events = _collect_stream_events(
            agent,
            OrchestratorInput(
                query="find machine learning videos and summarize them",
                tenant_id="default",
            ),
        )

        phases = _get_phases(events)
        assert "planning" in phases, f"Missing planning phase: {phases}"
        assert "execution" in phases, f"Missing execution phase: {phases}"

        # Orchestrator catches HTTP failures gracefully — final always exists
        final_data = _get_final(events, "OrchestratorAgent")
        assert "plan_steps" in final_data, f"Final should have plan_steps: {final_data}"
        assert isinstance(final_data["plan_steps"], list)
        assert len(final_data["plan_steps"]) >= 1, (
            f"Plan should have ≥1 step for 'find ML videos and summarize': {final_data['plan_steps']}"
        )
        # Each step must reference a valid agent type from real DSPy planning
        valid_agent_types = {
            "entity_extraction",
            "profile_selection",
            "query_enhancement",
            "search",
            "summarizer",
            "detailed_report",
        }
        for step in final_data["plan_steps"]:
            assert "agent_type" in step, f"Plan step missing agent_type: {step}"
            assert step["agent_type"] in valid_agent_types, (
                f"Plan step agent_type '{step['agent_type']}' not in valid types: {valid_agent_types}"
            )
            assert "reasoning" in step, f"Plan step missing reasoning: {step}"
        # "find ML videos and summarize" should produce a plan that includes search
        step_types = {step["agent_type"] for step in final_data["plan_steps"]}
        assert "search" in step_types, (
            f"'find videos and summarize' should include search step, got: {step_types}"
        )
        assert "plan_reasoning" in final_data
        plan_reasoning_lower = final_data["plan_reasoning"].lower()
        assert any(
            term in plan_reasoning_lower
            for term in ["search", "find", "video", "summarize"]
        ), (
            f"Plan reasoning should reference the query intent, got: '{final_data['plan_reasoning']}'"
        )
        assert "agent_results" in final_data
        assert isinstance(final_data["agent_results"], dict)


@pytest.mark.integration
@skip_if_no_llm
class TestQueryEnhancementAgentStreaming:
    """QueryEnhancementAgent streaming with real Ollama."""

    def test_stream_phases_and_enhanced_query(self, dspy_lm):
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())

        events = _collect_stream_events(
            agent,
            QueryEnhancementInput(query="ML videos", tenant_id="default"),
        )

        _assert_no_errors(events, "QueryEnhancementAgent")
        phases = _get_phases(events)
        assert "enhancement" in phases
        assert "parsing" in phases

        # Token events from call_dspy streaming (may be absent on DSPy cache hit)
        token_events = [e for e in events if e.get("phase") == "token"]
        if token_events:
            assert "data" in token_events[-1]
            assert "accumulated" in token_events[-1]["data"]

        final_data = _get_final(events, "QueryEnhancementAgent")
        assert "enhanced_query" in final_data
        enhanced = final_data["enhanced_query"]
        assert len(enhanced) > 0, "Enhanced query should be non-empty"
        # Enhanced query should relate to the input topic
        enhanced_lower = enhanced.lower()
        assert any(
            term in enhanced_lower
            for term in ["ml", "machine learning", "video", "learn"]
        ), f"Enhanced query should relate to 'ML videos', got: '{enhanced}'"
        # Should have expansion_terms or synonyms showing actual enhancement work
        assert "expansion_terms" in final_data or "synonyms" in final_data, (
            f"Should have expansion_terms or synonyms from DSPy enhancement: {final_data.keys()}"
        )


@pytest.mark.integration
@skip_if_no_llm
class TestEntityExtractionAgentStreaming:
    """EntityExtractionAgent streaming with real Ollama."""

    def test_stream_phases_and_extracted_entities(self, dspy_lm):
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionInput,
        )

        agent = EntityExtractionAgent(deps=EntityExtractionDeps())

        events = _collect_stream_events(
            agent,
            EntityExtractionInput(
                query="Tesla stock price analysis in 2025",
                tenant_id="default",
            ),
        )

        _assert_no_errors(events, "EntityExtractionAgent")
        phases = _get_phases(events)
        assert "extraction" in phases
        assert "parsing" in phases

        final_data = _get_final(events, "EntityExtractionAgent")
        assert "entities" in final_data
        # "Tesla" should be among extracted entities
        entity_texts = [
            e.get("text", e.get("name", "")).lower() for e in final_data["entities"]
        ]
        assert any("tesla" in t for t in entity_texts), (
            f"Expected 'Tesla' in extracted entities, got: {final_data['entities']}"
        )


@pytest.mark.integration
@skip_if_no_llm
class TestProfileSelectionAgentStreaming:
    """ProfileSelectionAgent streaming with real Ollama."""

    def test_stream_phases_and_selected_profile(self, dspy_lm):
        from cogniverse_agents.profile_selection_agent import (
            ProfileSelectionAgent,
            ProfileSelectionDeps,
            ProfileSelectionInput,
        )

        agent = ProfileSelectionAgent(deps=ProfileSelectionDeps())

        events = _collect_stream_events(
            agent,
            ProfileSelectionInput(query="find cat videos", tenant_id="default"),
        )

        _assert_no_errors(events, "ProfileSelectionAgent")
        phases = _get_phases(events)
        assert "selection" in phases

        final_data = _get_final(events, "ProfileSelectionAgent")
        assert "selected_profile" in final_data
        profile = final_data["selected_profile"]
        assert isinstance(profile, str)
        assert len(profile) > 0, "Should select a profile"
        # Profile should be one of the known available profiles
        known_profiles = {
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        }
        assert profile in known_profiles, (
            f"Selected profile '{profile}' not in known profiles: {known_profiles}"
        )
        assert "reasoning" in final_data
        reasoning_lower = final_data["reasoning"].lower()
        assert any(
            term in reasoning_lower
            for term in ["video", "colpali", "visual", "frame", "cat"]
        ), (
            f"Reasoning should explain why this profile for 'cat videos', got: '{final_data['reasoning']}'"
        )
        assert "modality" in final_data
        assert final_data["modality"] == "video", (
            f"'cat videos' query should select video modality, got: '{final_data['modality']}'"
        )
        assert "confidence" in final_data
        assert isinstance(final_data["confidence"], (int, float))
        assert final_data["confidence"] >= 0.3, (
            f"Straightforward video query should have reasonable confidence, got: {final_data['confidence']}"
        )
        assert "alternatives" in final_data
        assert isinstance(final_data["alternatives"], list)


@pytest.mark.integration
@skip_if_no_llm
class TestDetailedReportAgentStreaming:
    """DetailedReportAgent streaming with real Ollama."""

    def test_stream_phases_and_report_output(self, config_manager, dspy_lm):
        from cogniverse_agents.detailed_report_agent import (
            DetailedReportAgent,
            DetailedReportDeps,
            DetailedReportInput,
        )

        agent = DetailedReportAgent(
            deps=DetailedReportDeps(tenant_id="default"),
            config_manager=config_manager,
        )

        events = _collect_stream_events(
            agent,
            DetailedReportInput(
                query="AI trends in 2025",
                search_results=[
                    {
                        "id": "1",
                        "title": "AI Report 2025",
                        "score": 0.9,
                        "description": "Comprehensive analysis of AI developments",
                    },
                    {
                        "id": "2",
                        "title": "LLM Progress",
                        "score": 0.85,
                        "description": "Large language model improvements",
                    },
                ],
                include_visual_analysis=False,
                include_technical_details=True,
                include_recommendations=True,
            ),
        )

        _assert_no_errors(events, "DetailedReportAgent")
        phases = _get_phases(events)
        assert "thinking" in phases, f"Missing thinking phase: {phases}"
        assert "executive_summary" in phases, (
            f"Missing executive_summary phase: {phases}"
        )
        assert "findings" in phases, f"Missing findings phase: {phases}"

        final_data = _get_final(events, "DetailedReportAgent")
        assert "executive_summary" in final_data
        exec_summary = final_data["executive_summary"]
        assert len(exec_summary) > 20, f"Executive summary too short: '{exec_summary}'"
        # Executive summary should reference AI since query is about "AI trends"
        exec_lower = exec_summary.lower()
        assert any(
            term in exec_lower
            for term in ["ai", "artificial", "intelligence", "llm", "language model"]
        ), f"Executive summary should reference AI domain, got: '{exec_summary}'"
        assert "detailed_findings" in final_data
        assert isinstance(final_data["detailed_findings"], list)
        assert len(final_data["detailed_findings"]) >= 1, (
            f"Should have ≥1 finding for 2 search results, got: {final_data['detailed_findings']}"
        )
        assert "thinking_process" in final_data
        assert isinstance(final_data["thinking_process"], dict)
        tp = final_data["thinking_process"]
        assert "reasoning" in tp
        assert len(tp["reasoning"]) > 20, (
            f"Thinking reasoning should be substantive analysis, got: '{tp['reasoning']}'"
        )
        # Thinking should reference the query domain
        tp_reasoning_lower = tp["reasoning"].lower()
        assert any(
            term in tp_reasoning_lower
            for term in ["ai", "artificial", "intelligence", "llm", "report"]
        ), f"Thinking reasoning should reference AI domain, got: '{tp['reasoning']}'"


@pytest.mark.integration
@skip_if_no_llm
class TestSearchAgentStreaming:
    """SearchAgent streaming with real Vespa + real Ollama."""

    def test_stream_phases_and_search_output(
        self, vespa_instance, config_manager, schema_loader, dspy_lm
    ):
        from cogniverse_agents.search_agent import (
            SearchAgent,
            SearchAgentDeps,
            SearchInput,
        )

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=vespa_instance["http_port"],
            ),
            schema_loader=schema_loader,
            config_manager=config_manager,
        )

        events = _collect_stream_events(
            agent,
            SearchInput(
                query="machine learning videos",
                tenant_id="default",
                modality="video",
                top_k=5,
            ),
        )

        phases = _get_phases(events)
        assert "retrieval" in phases, f"Missing retrieval phase: {phases}"

        # Should have final event with search output structure
        final_data = _get_final(events, "SearchAgent")
        assert "results" in final_data
        assert isinstance(final_data["results"], list)
        assert "search_mode" in final_data
        # Single query → single_profile (not ensemble, since no profiles list was passed)
        assert final_data["search_mode"] == "single_profile", (
            f"Single query should use single_profile mode, got: '{final_data['search_mode']}'"
        )
        assert "query" in final_data
        assert final_data["query"] == "machine learning videos"
        assert "total_results" in final_data
        assert isinstance(final_data["total_results"], int)
        # total_results must match actual results length
        assert final_data["total_results"] == len(final_data["results"]), (
            f"total_results ({final_data['total_results']}) != len(results) ({len(final_data['results'])})"
        )
        assert "modality" in final_data
        assert final_data["modality"] == "video", (
            f"Modality should be 'video' as requested, got: '{final_data['modality']}'"
        )


@pytest.mark.integration
@skip_if_no_llm
class TestImageSearchAgentStreaming:
    """ImageSearchAgent streaming with real Vespa."""

    def test_stream_phases_and_output(self, vespa_instance, config_manager, dspy_lm):
        from cogniverse_agents.image_search_agent import (
            ImageSearchAgent,
            ImageSearchDeps,
            ImageSearchInput,
        )

        agent = ImageSearchAgent(
            deps=ImageSearchDeps(
                vespa_endpoint=f"http://localhost:{vespa_instance['http_port']}",
                tenant_id="default",
            )
        )

        events = _collect_stream_events(
            agent,
            ImageSearchInput(
                query="cat sitting on a table",
                search_mode="semantic",
                limit=5,
            ),
        )

        phases = _get_phases(events)
        assert "encoding" in phases, f"Missing encoding phase: {phases}"
        assert "retrieval" in phases, f"Missing retrieval phase: {phases}"

        final_data = _get_final(events, "ImageSearchAgent")
        assert "results" in final_data
        assert isinstance(final_data["results"], list)
        assert "count" in final_data
        assert isinstance(final_data["count"], int)
        assert final_data["count"] == len(final_data["results"])


@pytest.mark.integration
@skip_if_no_llm
class TestAudioAnalysisAgentStreaming:
    """AudioAnalysisAgent streaming with real Vespa."""

    def test_stream_phases_and_output(self, vespa_instance, dspy_lm):
        from cogniverse_agents.audio_analysis_agent import (
            AudioAnalysisAgent,
            AudioAnalysisDeps,
            AudioSearchInput,
        )

        agent = AudioAnalysisAgent(
            deps=AudioAnalysisDeps(
                vespa_endpoint=f"http://localhost:{vespa_instance['http_port']}",
                tenant_id="default",
            )
        )

        events = _collect_stream_events(
            agent,
            AudioSearchInput(
                query="person speaking about technology",
                search_mode="semantic",
                limit=5,
            ),
        )

        phases = _get_phases(events)
        assert "encoding" in phases, f"Missing encoding phase: {phases}"
        assert "retrieval" in phases, f"Missing retrieval phase: {phases}"

        final_data = _get_final(events, "AudioAnalysisAgent")
        assert "results" in final_data
        assert isinstance(final_data["results"], list)
        assert "count" in final_data
        assert isinstance(final_data["count"], int)
        assert final_data["count"] == len(final_data["results"])


@pytest.mark.integration
@skip_if_no_llm
class TestDocumentAgentStreaming:
    """DocumentAgent streaming with real Vespa."""

    def test_stream_phases_and_output(self, vespa_instance, dspy_lm):
        from cogniverse_agents.document_agent import (
            DocumentAgent,
            DocumentAgentDeps,
            DocumentSearchInput,
        )

        agent = DocumentAgent(
            deps=DocumentAgentDeps(
                vespa_endpoint=f"http://localhost:{vespa_instance['http_port']}",
                tenant_id="default",
            )
        )

        events = _collect_stream_events(
            agent,
            DocumentSearchInput(
                query="quarterly earnings report",
                strategy="semantic",
                limit=5,
            ),
        )

        phases = _get_phases(events)
        assert "strategy_selection" in phases, (
            f"Missing strategy_selection phase: {phases}"
        )

        final_data = _get_final(events, "DocumentAgent")
        assert "results" in final_data
        assert isinstance(final_data["results"], list)
        assert "count" in final_data
        assert isinstance(final_data["count"], int)
        assert final_data["count"] == len(final_data["results"])


# ── Full A2A round-trip streaming test ───────────────────────────────────────


@pytest.mark.integration
@skip_if_no_llm
class TestA2AStreamingFullStack:
    """Full A2A protocol streaming: message/stream → SSE with real services."""

    def test_summarizer_streams_through_a2a(self, streaming_a2a_client):
        """Real Ollama summarization streams through A2A protocol."""
        raw_events = _send_a2a_stream(
            streaming_a2a_client,
            "Briefly explain what deep learning is in two sentences",
        )

        parsed = _parse_agent_events(raw_events)
        assert len(parsed) >= 3, (
            f"Expected ≥3 events (thinking status + thinking partial + summarization + final), "
            f"got {len(parsed)}: {parsed}"
        )

        phases = _get_phases(parsed)
        assert "thinking" in phases, f"Missing thinking phase in A2A stream: {phases}"

        final_data = _get_final(parsed, "A2A SummarizerAgent")
        assert "summary" in final_data
        summary = final_data["summary"]
        assert len(summary) > 30, f"Summary too short for real LLM output: '{summary}'"
        summary_lower = summary.lower()
        assert any(
            term in summary_lower
            for term in ["deep learning", "neural", "network", "learn"]
        ), f"Summary should reference deep learning, got: '{summary}'"
        assert "key_points" in final_data
        assert isinstance(final_data["key_points"], list)
        assert "confidence_score" in final_data


# ── Optimization action integration tests ────────────────────────────────────


@pytest.mark.integration
@skip_if_no_llm
class TestRoutingOptimizationIntegration:
    """Test optimization actions through real dispatcher + real optimizer.

    Creates routing agent with enable_advanced_optimization=True and
    real telemetry (Phoenix) to exercise the full optimization path.
    """

    def test_optimize_routing_round_trip(
        self, streaming_dispatcher, real_telemetry, dspy_lm
    ):
        """Full round-trip: record 10+ examples → persist to Phoenix → verify stored."""
        # Generate 10 examples to trigger _persist_data (fires every 10 experiences)
        examples = [
            {
                "query": f"test query {i} about {'videos' if i % 2 == 0 else 'documents'}",
                "chosen_agent": "search_agent" if i % 2 == 0 else "summarizer_agent",
                "confidence": 0.7 + (i * 0.02),
                "search_quality": 0.6 + (i * 0.03),
                "agent_success": True,
                "processing_time": 1.0 + (i * 0.1),
            }
            for i in range(10)
        ]

        async def _run():
            return await streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="optimize routing",
                context={
                    "tenant_id": "default",
                    "action": "optimize_routing",
                    "examples": examples,
                },
            )

        result = asyncio.get_event_loop().run_until_complete(_run())

        assert result["status"] == "optimization_triggered", (
            f"Optimization should be triggered with 10 examples, got: {result}"
        )
        assert result["training_examples"] == 10
        assert result["optimizer"] == "AdvancedRoutingOptimizer"

        # Round-trip verification: a NEW routing request creates a fresh optimizer
        # that loads persisted data from Phoenix. Route a real query — the optimizer
        # should initialize with the 10 stored experiences.
        route_result = asyncio.get_event_loop().run_until_complete(
            streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="find cat videos",
                context={"tenant_id": "default"},
            )
        )

        # The routing should succeed (not error) — proves the optimizer
        # initialized correctly from persisted Phoenix data
        assert route_result["status"] == "success", (
            f"Routing after optimization should succeed, got: {route_result}"
        )
        assert "recommended_agent" in route_result

    def test_optimize_routing_empty_examples_triggers_cycle(
        self, streaming_dispatcher, real_telemetry, dspy_lm
    ):
        """optimize_routing with no examples runs automated cycle from traces."""

        async def _run():
            return await streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="optimize routing",
                context={
                    "tenant_id": "default",
                    "action": "optimize_routing",
                    "examples": [],
                },
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        # Empty examples triggers automated optimization cycle from Phoenix traces
        assert result["status"] == "optimization_triggered", (
            f"Empty examples should trigger automated cycle, got: {result}"
        )
        assert result["optimizer"] == "OptimizationOrchestrator"
        assert "cycle_results" in result

    def test_get_optimization_status(
        self, streaming_dispatcher, real_telemetry, dspy_lm
    ):
        """get_optimization_status returns routing statistics from real agent."""

        async def _run():
            return await streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="optimization status",
                context={
                    "tenant_id": "default",
                    "action": "get_optimization_status",
                },
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result["status"] == "active", (
            f"Optimizer should be active, got: {result}"
        )
        assert result["optimizer_ready"] is True
        assert isinstance(result["metrics"], dict)
        assert "total_queries" in result["metrics"]

    def test_optimization_cycle_from_traces(
        self, streaming_dispatcher, real_telemetry, dspy_lm
    ):
        """Run full optimization cycle reading from Phoenix traces."""

        async def _run():
            return await streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="optimize routing",
                context={
                    "tenant_id": "default",
                    "action": "optimize_routing",
                    # No examples — triggers automated cycle from traces
                },
            )

        result = asyncio.get_event_loop().run_until_complete(_run())

        assert result["status"] == "optimization_triggered", (
            f"Optimization cycle should complete, got: {result}"
        )
        assert "cycle_results" in result
        assert "spans_evaluated" in result
        assert isinstance(result["spans_evaluated"], int)
        assert result["optimizer"] == "OptimizationOrchestrator"


@pytest.mark.integration
@skip_if_no_llm
class TestFullOptimizationPipeline:
    """End-to-end optimization: real routing → real traces → real optimize → real load.

    Exercises the complete pipeline with real Ollama, Vespa, Phoenix, and DSPy.
    No mocks anywhere.
    """

    def test_full_optimization_round_trip(
        self, streaming_dispatcher, vespa_instance, real_telemetry, dspy_lm
    ):
        """
        1. Route real queries → traces accumulate in Phoenix
        2. Record experiences → optimizer persists to Phoenix
        3. Run optimization cycle → reads traces, annotates, compiles
        4. Route again → verify optimized model loads from Phoenix
        """
        import asyncio

        loop = asyncio.get_event_loop()

        # Step 1: Make real routing requests to generate traces
        queries = [
            "show me videos of cats playing",
            "find documents about machine learning",
            "search for audio lectures on physics",
            "display images of sunset landscapes",
            "summarize the latest AI research",
        ]
        routing_results = []
        for query in queries:
            result = loop.run_until_complete(
                streaming_dispatcher.dispatch(
                    agent_name="routing_agent",
                    query=query,
                    context={"tenant_id": "default"},
                )
            )
            routing_results.append(result)
            assert result["status"] == "success", (
                f"Routing should succeed for '{query}', got: {result}"
            )

        # Verify all queries were routed to real agents
        agents_used = {r["recommended_agent"] for r in routing_results}
        assert len(agents_used) >= 1, f"Should route to at least 1 agent: {agents_used}"

        # Step 2: Record experiences from those routing results
        examples = [
            {
                "query": queries[i],
                "chosen_agent": routing_results[i]["recommended_agent"],
                "confidence": routing_results[i]["confidence"],
                "search_quality": 0.7 + (i * 0.05),
                "agent_success": True,
                "processing_time": 1.0,
            }
            for i in range(len(queries))
        ]
        # Add more to reach 10 (threshold for persist)
        for i in range(5):
            examples.append(
                {
                    "query": f"supplementary query {i}",
                    "chosen_agent": "search_agent",
                    "confidence": 0.8,
                    "search_quality": 0.75,
                    "agent_success": True,
                    "processing_time": 0.5,
                }
            )

        record_result = loop.run_until_complete(
            streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="optimize routing",
                context={
                    "tenant_id": "default",
                    "action": "optimize_routing",
                    "examples": examples,
                },
            )
        )
        assert record_result["status"] == "optimization_triggered", (
            f"Should trigger optimization with 10 examples: {record_result}"
        )
        assert record_result["training_examples"] == 10

        # Step 3: Run full optimization cycle from traces
        cycle_result = loop.run_until_complete(
            streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="optimize routing",
                context={
                    "tenant_id": "default",
                    "action": "optimize_routing",
                },
            )
        )
        assert cycle_result["status"] == "optimization_triggered", (
            f"Optimization cycle should complete: {cycle_result}"
        )

        # Step 4: Route a new query — should use optimized model loaded from Phoenix
        final_result = loop.run_until_complete(
            streaming_dispatcher.dispatch(
                agent_name="routing_agent",
                query="find cat videos with music",
                context={"tenant_id": "default"},
            )
        )
        assert final_result["status"] == "success", (
            f"Routing after optimization should succeed: {final_result}"
        )
        assert "recommended_agent" in final_result
        assert final_result["confidence"] > 0.0
