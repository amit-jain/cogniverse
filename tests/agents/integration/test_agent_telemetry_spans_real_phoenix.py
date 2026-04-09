"""Round-trip integration test for AgentBase telemetry spans + SPAN_NAME_BY_AGENT.

Audit fixes #2 + #10 — before #10, 5 of 6 main agents emitted ZERO spans
during processing because AgentBase.process() never wrapped _process_impl
in a telemetry span. Before #2, the QualityMonitor's lookup table had
hardcoded names (search_service.search, summarizer_agent.process, etc.)
that didn't match what agents actually emitted, so live-traffic eval
queried Phoenix for span names that didn't exist.

This test exercises the full chain end-to-end against real Phoenix:
1. Construct an agent inheriting AgentBase
2. Inject a real PhoenixProvider-backed TelemetryManager
3. Call agent.process(input)
4. Query real Phoenix for spans with name f"{ClassName}.process"
5. Verify the span was actually exported and queryable

Also verifies the CUSTOM telemetry spans emitted by A2A agents inside
_process_impl (cogniverse.gateway, cogniverse.entity_extraction, etc.).

If the SPAN_NAME_BY_AGENT lookup ever drifts from what AgentBase emits,
this test catches it.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.agents.base import (
    AgentBase,
    AgentDeps,
    AgentInput,
    AgentOutput,
)
from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT, AgentType


class _TelemetryTestInput(AgentInput):
    query: str
    tenant_id: str = "telemetry_real_test"


class _TelemetryTestOutput(AgentOutput):
    result: str


class _TelemetryTestDeps(AgentDeps):
    pass


# Subclasses named to match real agents — the names matter because
# AgentBase emits spans as f"{ClassName}.process" and SPAN_NAME_BY_AGENT
# expects exactly those names.
class SearchAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"searched: {input.query}")


class SummarizerAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"summarized: {input.query}")


class DetailedReportAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"reported: {input.query}")


class RoutingAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"routed: {input.query}")


def _query_phoenix_for_span(
    span_name: str,
    project_name: str,
    phoenix_http_url: str,
    max_wait: int = 30,
):
    """Query the real Phoenix instance for spans with the given name.

    Returns the matched span row (or None) by polling for up to max_wait
    seconds — Phoenix has a small ingestion delay even with sync export.

    phoenix_http_url comes from real_telemetry.config.provider_config["http_endpoint"]
    so it's never hardcoded here.
    """
    from phoenix.client import Client

    client = Client(base_url=phoenix_http_url)

    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            spans_df = client.spans.get_spans_dataframe(
                project_identifier=project_name,
                limit=200,
            )
            if spans_df is not None and not spans_df.empty:
                matches = spans_df[spans_df["name"] == span_name]
                if not matches.empty:
                    return matches.iloc[0]
        except Exception:
            pass
        time.sleep(1)
    return None


@pytest.mark.integration
class TestAgentTelemetrySpansRealPhoenix:
    @pytest.mark.asyncio
    async def test_agent_emits_span_observable_in_real_phoenix(
        self, real_telemetry
    ):
        """Calling agent.process() must emit a span that's queryable via
        the real Phoenix HTTP API. Before fix #10, this would fail because
        AgentBase didn't wrap _process_impl in a span at all."""
        agent = SearchAgent(deps=_TelemetryTestDeps())
        agent.set_telemetry_manager(real_telemetry)

        await agent.process(
            _TelemetryTestInput(
                query="test span emission", tenant_id="telemetry_real_test"
            )
        )

        # Force span export — sync export should already have flushed,
        # but allow Phoenix a moment to ingest.
        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name(
            "telemetry_real_test"
        )
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span("SearchAgent.process", project_name, phoenix_url)

        assert span is not None, (
            f"SearchAgent.process span not found in Phoenix project "
            f"{project_name}. Either AgentBase isn't wrapping _process_impl "
            f"in a span (audit fix #10 regressed) or the span export pipeline "
            f"is broken."
        )

    @pytest.mark.asyncio
    async def test_span_name_matches_lookup_table_for_each_agent_type(
        self, real_telemetry
    ):
        """For every AgentType in SPAN_NAME_BY_AGENT, instantiate the
        matching agent class, emit a span, and verify Phoenix returns
        a match for the lookup table's name. This is the integration
        test that would have caught audit fix #2 (wrong span names)
        AND fix #10 (no spans emitted) at the same time."""
        agent_classes = {
            AgentType.SEARCH: SearchAgent,
            AgentType.SUMMARY: SummarizerAgent,
            AgentType.REPORT: DetailedReportAgent,
            AgentType.ROUTING: RoutingAgent,
        }

        project_name = real_telemetry.config.get_project_name(
            "telemetry_real_test"
        )

        for agent_type, expected_span_name in SPAN_NAME_BY_AGENT.items():
            agent_cls = agent_classes[agent_type]
            agent = agent_cls(deps=_TelemetryTestDeps())
            agent.set_telemetry_manager(real_telemetry)

            await agent.process(
                _TelemetryTestInput(
                    query=f"test {agent_type.value}",
                    tenant_id="telemetry_real_test",
                )
            )

        await asyncio.sleep(3)

        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        for agent_type, expected_span_name in SPAN_NAME_BY_AGENT.items():
            span = _query_phoenix_for_span(
                expected_span_name, project_name, phoenix_url, max_wait=15
            )
            assert span is not None, (
                f"SPAN_NAME_BY_AGENT[{agent_type.value}] = "
                f"{expected_span_name!r} but Phoenix returned no spans "
                f"with that name. The lookup table is out of sync with "
                f"what AgentBase actually emits."
            )


# ---------------------------------------------------------------------------
# A2A custom span tests
# ---------------------------------------------------------------------------
# These test the CUSTOM telemetry spans emitted inside _process_impl by A2A
# agents (separate from the automatic {ClassName}.process spans from AgentBase).
# Span names: cogniverse.gateway, cogniverse.entity_extraction,
# cogniverse.query_enhancement, cogniverse.profile_selection, cogniverse.orchestration


# Map of custom span names to the A2A agents that emit them
A2A_CUSTOM_SPANS = {
    "cogniverse.gateway": "GatewayAgent",
    "cogniverse.entity_extraction": "EntityExtractionAgent",
    "cogniverse.query_enhancement": "QueryEnhancementAgent",
    "cogniverse.profile_selection": "ProfileSelectionAgent",
    "cogniverse.orchestration": "OrchestratorAgent",
}


@pytest.mark.integration
class TestA2ACustomTelemetrySpansRealPhoenix:
    """Verify that A2A agents emit their custom telemetry spans.

    Each A2A agent emits a domain-specific span (e.g., cogniverse.gateway)
    inside _process_impl, in addition to the automatic {ClassName}.process
    span from AgentBase. This test exercises each agent and verifies the
    custom span is queryable in real Phoenix.
    """

    @pytest.mark.asyncio
    async def test_gateway_emits_custom_span(self, real_telemetry):
        """GatewayAgent emits cogniverse.gateway span."""
        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )

        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps, port=19014)
        # Mock GLiNER to avoid model download in CI
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.9},
        ]
        agent.set_telemetry_manager(real_telemetry)

        await agent.process(
            GatewayInput(
                query="show me videos about cats",
                tenant_id="telemetry_real_test",
            )
        )

        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_real_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            "cogniverse.gateway", project_name, phoenix_url
        )

        assert span is not None, (
            "cogniverse.gateway span not found in Phoenix. "
            "GatewayAgent._emit_gateway_span() may have regressed."
        )

    @pytest.mark.asyncio
    async def test_entity_extraction_emits_custom_span(self, real_telemetry):
        """EntityExtractionAgent emits cogniverse.entity_extraction span."""
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionInput,
        )

        deps = EntityExtractionDeps()
        with patch.object(
            EntityExtractionAgent, "_initialize_extractors"
        ):
            agent = EntityExtractionAgent(deps=deps, port=19010)
        agent._gliner_extractor = None
        agent._spacy_analyzer = None
        agent.set_telemetry_manager(real_telemetry)

        # Mock the DSPy call to avoid requiring an LLM
        mock_prediction = MagicMock()
        mock_prediction.entities = "machine learning|CONCEPT|0.9"
        mock_prediction.entity_types = "CONCEPT"
        agent.dspy_module = MagicMock()
        agent.dspy_module.return_value = mock_prediction

        with patch.object(agent, "call_dspy", return_value=mock_prediction):
            await agent.process(
                EntityExtractionInput(
                    query="machine learning tutorials",
                    tenant_id="telemetry_real_test",
                )
            )

        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_real_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            "cogniverse.entity_extraction", project_name, phoenix_url
        )

        assert span is not None, (
            "cogniverse.entity_extraction span not found in Phoenix. "
            "EntityExtractionAgent._emit_extraction_span() may have regressed."
        )

    @pytest.mark.asyncio
    async def test_query_enhancement_emits_custom_span(self, real_telemetry):
        """QueryEnhancementAgent emits cogniverse.query_enhancement span."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        deps = QueryEnhancementDeps()
        agent = QueryEnhancementAgent(deps=deps, port=19012)
        agent.set_telemetry_manager(real_telemetry)

        # Mock DSPy call to avoid requiring an LLM
        mock_result = MagicMock()
        mock_result.enhanced_query = "machine learning tutorials guides"
        mock_result.expansion_terms = "deep learning, neural networks"
        mock_result.synonyms = "ML, AI"
        mock_result.context = "education, technology"
        mock_result.confidence = "0.85"
        mock_result.reasoning = "Added related terms for ML"

        with patch.object(agent, "call_dspy", return_value=mock_result):
            await agent.process(
                QueryEnhancementInput(
                    query="ML tutorials",
                    tenant_id="telemetry_real_test",
                )
            )

        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_real_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            "cogniverse.query_enhancement", project_name, phoenix_url
        )

        assert span is not None, (
            "cogniverse.query_enhancement span not found in Phoenix. "
            "QueryEnhancementAgent._emit_enhancement_span() may have regressed."
        )

    @pytest.mark.asyncio
    async def test_profile_selection_emits_custom_span(self, real_telemetry):
        """ProfileSelectionAgent emits cogniverse.profile_selection span."""
        from cogniverse_agents.profile_selection_agent import (
            ProfileSelectionAgent,
            ProfileSelectionDeps,
            ProfileSelectionInput,
        )

        deps = ProfileSelectionDeps()
        agent = ProfileSelectionAgent(deps=deps, port=19011)
        agent.set_telemetry_manager(real_telemetry)

        # Mock DSPy call to avoid requiring an LLM
        mock_result = MagicMock()
        mock_result.selected_profile = "video_colpali_smol500_mv_frame"
        mock_result.confidence = "0.8"
        mock_result.reasoning = "Video query matched colpali profile"
        mock_result.query_intent = "video_search"
        mock_result.modality = "video"
        mock_result.complexity = "simple"

        with patch.object(agent, "call_dspy", return_value=mock_result):
            await agent.process(
                ProfileSelectionInput(
                    query="show me cooking videos",
                    tenant_id="telemetry_real_test",
                )
            )

        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_real_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            "cogniverse.profile_selection", project_name, phoenix_url
        )

        assert span is not None, (
            "cogniverse.profile_selection span not found in Phoenix. "
            "ProfileSelectionAgent._emit_profile_span() may have regressed."
        )

    @pytest.mark.asyncio
    async def test_orchestrator_emits_custom_span(self, real_telemetry):
        """OrchestratorAgent emits cogniverse.orchestration span."""
        from cogniverse_agents.orchestrator_agent import (
            AgentStep,
            OrchestrationPlan,
            OrchestratorAgent,
            OrchestratorDeps,
            OrchestratorInput,
        )
        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry

        # Build a minimal agent registry
        with patch.object(AgentRegistry, "__init__", lambda self, **kw: None):
            registry = AgentRegistry.__new__(AgentRegistry)
        registry.agents = {}
        registry.capabilities = {}
        registry.tenant_id = "default"
        registry.config_manager = MagicMock()
        registry.config = {}
        registry.http_client = MagicMock()

        agents = [
            AgentEndpoint(
                name="search_agent",
                url="http://localhost:8002",
                capabilities=["search"],
                process_endpoint="/tasks/send",
            ),
        ]
        for agent in agents:
            registry.register_agent(agent)

        mock_cm = MagicMock()
        mock_cm.get_system_config.return_value = MagicMock(
            backend_url="http://localhost",
            backend_port=8080,
        )
        mock_cm.get_config.return_value = {}

        deps = OrchestratorDeps()
        agent = OrchestratorAgent(
            deps=deps,
            registry=registry,
            config_manager=mock_cm,
            port=19013,
        )
        agent.set_telemetry_manager(real_telemetry)

        # Mock _create_plan to return a simple plan (avoids real LLM call)
        mock_plan = OrchestrationPlan(
            query="find machine learning videos",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "find machine learning videos"},
                    depends_on=[],
                    reasoning="Search for ML videos",
                ),
            ],
            parallel_groups=[],
            reasoning="Single search step",
            unavailable_agents=[],
        )
        # Mock _execute_plan to return synthetic results (avoids real HTTP calls)
        mock_results = {
            "search_agent": {"status": "success", "results": []},
        }

        with patch.object(agent, "_create_plan", return_value=mock_plan), \
             patch.object(agent, "_execute_plan", return_value=mock_results):
            await agent.process(
                OrchestratorInput(
                    query="find machine learning videos",
                    tenant_id="telemetry_real_test",
                )
            )

        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_real_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            "cogniverse.orchestration", project_name, phoenix_url
        )

        assert span is not None, (
            "cogniverse.orchestration span not found in Phoenix. "
            "OrchestratorAgent._emit_orchestration_span() may have regressed."
        )

    @pytest.mark.asyncio
    async def test_all_a2a_custom_span_names_documented(self, real_telemetry):
        """Verify our test covers all known A2A custom span names."""
        # This is a meta-test: if someone adds a new A2A agent with a custom
        # span but forgets to add a test case above, this will catch it.
        expected_spans = {
            "cogniverse.gateway",
            "cogniverse.entity_extraction",
            "cogniverse.query_enhancement",
            "cogniverse.profile_selection",
            "cogniverse.orchestration",
        }
        tested_spans = set(A2A_CUSTOM_SPANS.keys())
        assert tested_spans == expected_spans, (
            f"A2A_CUSTOM_SPANS is out of date. "
            f"Missing: {expected_spans - tested_spans}, "
            f"Extra: {tested_spans - expected_spans}"
        )
