"""Round-trip integration test for AgentBase telemetry spans + SPAN_NAME_BY_AGENT.

All six main agents must emit a processing span: AgentBase.process()
wraps _process_impl in a telemetry span, and QualityMonitor's
SPAN_NAME_BY_AGENT lookup must match the names agents actually emit (a
mismatch makes live-traffic eval query Phoenix for span names that don't
exist).

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
import inspect
import re
import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from cogniverse_core.agents.base import (
    AgentBase,
    AgentDeps,
    AgentInput,
    AgentOutput,
)
from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT, AgentType
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_ENTITY_EXTRACTION,
    SPAN_NAME_GATEWAY,
    SPAN_NAME_ORCHESTRATION,
    SPAN_NAME_PROFILE_SELECTION,
    SPAN_NAME_QUERY_ENHANCEMENT,
    SPAN_NAME_ROUTING,
)
from cogniverse_foundation.telemetry.span_contract import (
    OP_ENTITY_EXTRACTION,
    OP_GATEWAY,
    OP_ORCHESTRATION,
    OP_PROFILE_SELECTION,
    OP_QUERY_ENHANCEMENT,
    OP_ROUTING,
    read_span_attributes,
    read_span_io,
)


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


class GatewayAgent(
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
    from phoenix.client.types.spans import SpanQuery

    client = Client(base_url=phoenix_http_url)
    query = SpanQuery().where(f"name == '{span_name}'")

    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            spans_df = client.spans.get_spans_dataframe(
                query=query,
                project_identifier=project_name,
                timeout=30,
            )
            if spans_df is not None and not spans_df.empty:
                matches = spans_df
                if not matches.empty:
                    return matches.iloc[0]
        except Exception:
            pass
        time.sleep(1)
    raise AssertionError(
        f"Phoenix span {span_name!r} was not found in project {project_name!r} "
        f"within {max_wait}s at {phoenix_http_url!r}"
    )


def _tenant_id(prefix: str) -> str:
    return f"{prefix}:{uuid4().hex[:12]}"


def _project_name(real_telemetry, tenant_id: str) -> str:
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    return real_telemetry.config.get_project_name(canonical_tenant_id(tenant_id))


@pytest.mark.integration
class TestAgentTelemetrySpansRealPhoenix:
    @pytest.mark.asyncio
    async def test_agent_emits_span_observable_in_real_phoenix(self, real_telemetry):
        """Calling agent.process() must emit a span that's queryable via
        the real Phoenix HTTP API. Before fix #10, this would fail because
        AgentBase didn't wrap _process_impl in a span at all."""
        tenant_id = _tenant_id("agentbase-process-span")
        agent = SearchAgent(deps=_TelemetryTestDeps())
        agent.set_telemetry_manager(real_telemetry)

        await agent.process(
            _TelemetryTestInput(query="test span emission", tenant_id=tenant_id)
        )

        # Force span export — sync export should already have flushed,
        # but allow Phoenix a moment to ingest.
        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span("SearchAgent.process", project_name, phoenix_url)
        assert span["name"] == "SearchAgent.process"

    @pytest.mark.asyncio
    async def test_span_name_matches_lookup_table_for_each_agent_type(
        self, real_telemetry
    ):
        """Every AgentType in SPAN_NAME_BY_AGENT must map to a span name
        its agent really emits. The four process-convention types are
        exercised end-to-end: emit via AgentBase.process() and verify the
        lookup table's name is queryable in real Phoenix. The four domain
        span types are pinned to their emission sites: the lookup table's
        name must be the literal each agent passes to
        telemetry_manager.span(), so a rename on either side fails here."""
        from cogniverse_agents.entity_extraction_agent import EntityExtractionAgent
        from cogniverse_agents.gateway_agent import GatewayAgent as A2AGatewayAgent
        from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementAgent

        agent_classes = {
            AgentType.SEARCH: SearchAgent,
            AgentType.SUMMARY: SummarizerAgent,
            AgentType.REPORT: DetailedReportAgent,
            AgentType.GATEWAY: GatewayAgent,
        }
        domain_span_emitters = {
            AgentType.ROUTING: A2AGatewayAgent._emit_routing_span,
            AgentType.QUERY_ENHANCEMENT: QueryEnhancementAgent._emit_enhancement_span,
            AgentType.ENTITY_EXTRACTION: EntityExtractionAgent._emit_extraction_span,
            AgentType.PROFILE_SELECTION: ProfileSelectionAgent._emit_profile_span,
        }
        assert set(agent_classes).isdisjoint(domain_span_emitters)
        assert set(agent_classes) | set(domain_span_emitters) == set(
            SPAN_NAME_BY_AGENT
        ), (
            "SPAN_NAME_BY_AGENT has an AgentType this test does not map to "
            "an agent class or an emitting method; add it to agent_classes "
            "or domain_span_emitters."
        )

        tenant_id = _tenant_id("span-name-lookup")
        project_name = _project_name(real_telemetry, tenant_id)

        for agent_type, agent_cls in agent_classes.items():
            agent = agent_cls(deps=_TelemetryTestDeps())
            agent.set_telemetry_manager(real_telemetry)

            await agent.process(
                _TelemetryTestInput(
                    query=f"test {agent_type.value}",
                    tenant_id=tenant_id,
                )
            )

        await asyncio.sleep(3)

        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        for agent_type in agent_classes:
            expected_span_name = SPAN_NAME_BY_AGENT[agent_type]
            span = _query_phoenix_for_span(
                expected_span_name, project_name, phoenix_url, max_wait=15
            )
            assert span["name"] == expected_span_name

        for agent_type, emitter in domain_span_emitters.items():
            expected_span_name = SPAN_NAME_BY_AGENT[agent_type]
            emitter_source = re.sub(r"\s+", "", inspect.getsource(emitter))
            assert (
                f'.span("{expected_span_name}"' in emitter_source
                or f'name="{expected_span_name}"' in emitter_source
            ), (
                f"SPAN_NAME_BY_AGENT[{agent_type.value}] = "
                f"{expected_span_name!r} but {emitter.__qualname__} does not "
                f"pass that literal to telemetry_manager.span(). The lookup "
                f"table is out of sync with the emitting site."
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
    SPAN_NAME_GATEWAY: "GatewayAgent",
    SPAN_NAME_ENTITY_EXTRACTION: "EntityExtractionAgent",
    SPAN_NAME_QUERY_ENHANCEMENT: "QueryEnhancementAgent",
    SPAN_NAME_PROFILE_SELECTION: "ProfileSelectionAgent",
    SPAN_NAME_ORCHESTRATION: "OrchestratorAgent",
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

        tenant_id = _tenant_id("gateway-custom-span")
        agent = GatewayAgent(deps=GatewayDeps(), port=19014)
        agent.set_telemetry_manager(real_telemetry)
        query = "show me videos about cats"
        with (
            patch.object(
                agent,
                "_extract_entities",
                return_value=(
                    [{"text": "videos", "label": "video_content", "score": 0.9}],
                    False,
                ),
            ),
            patch.object(agent, "_classify_modality", return_value=("video", 0.91)),
            patch.object(agent, "_detected_modalities", return_value=["video"]),
            patch.object(
                agent,
                "_classify_generation_type",
                return_value=("raw_results", 0.88),
            ),
            patch.object(agent, "_is_complex", return_value=False),
        ):
            result = await agent.process(
                GatewayInput(
                    query=query,
                    tenant_id=tenant_id,
                )
            )

        assert result.model_dump() == {
            "query": query,
            "complexity": "simple",
            "modality": "video",
            "detected_modalities": ["video"],
            "generation_type": "raw_results",
            "routed_to": "search_agent",
            "confidence": 0.88,
            "fast_path_confidence_threshold": agent.deps.fast_path_confidence_threshold,
            "gliner_threshold": agent.deps.gliner_threshold,
            "reasoning": "Single video raw_results query routed to search_agent "
            "(confidence=0.88)",
            "entity_extraction_failed": False,
        }

        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(SPAN_NAME_GATEWAY, project_name, phoenix_url)
        assert span["name"] == SPAN_NAME_GATEWAY

        span_io = read_span_io(span)
        assert span_io == {
            "input": query,
            "output": {
                "complexity": "simple",
                "modality": "video",
                "generation_type": "raw_results",
                "routed_to": "search_agent",
                "confidence": 0.88,
            },
            "operation": OP_GATEWAY,
            "modality": None,
        }

        attrs = read_span_attributes(span)
        assert attrs["input.value"] == query
        assert attrs["operation"] == OP_GATEWAY

    @pytest.mark.asyncio
    async def test_routing_span_records_entity_extraction_failed(self, real_telemetry):
        """A GLiNER outage routes with a low confidence indistinguishable from a
        genuine low-confidence classification. The cogniverse.routing span must
        record entity_extraction_failed=True so RoutingEvaluator/QualityMonitor
        can exclude these from threshold recalibration — the flag was computed
        on GatewayOutput but reached NO telemetry consumer before this."""
        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )

        tenant_id = _tenant_id("routing-failed")
        agent = GatewayAgent(deps=GatewayDeps(), port=19015)
        # GLiNER present but FAILING mid-predict -> entity_extraction_failed=True.
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.side_effect = RuntimeError("gliner down")
        agent.set_telemetry_manager(real_telemetry)

        await agent.process(
            GatewayInput(
                query="find the report on q3 revenue",
                tenant_id=tenant_id,
            )
        )
        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        row = _query_phoenix_for_span(SPAN_NAME_ROUTING, project_name, phoenix_url)
        assert row["name"] == SPAN_NAME_ROUTING

        span_io = read_span_io(row)
        assert span_io == {
            "input": "find the report on q3 revenue",
            "output": {
                "chosen_agent": "orchestrator_agent",
                "recommended_agent": "orchestrator_agent",
                "confidence": 0.0,
                "reasoning": "Orchestrator needed: no entities detected; low confidence "
                "(0.00 < 0.4); analysis keywords: report",
                "complexity": "complex",
                "modality": "video",
                "generation_type": "raw_results",
                "fast_path_confidence_threshold": 0.4,
                "gliner_threshold": 0.3,
                "entity_extraction_failed": True,
            },
            "operation": OP_ROUTING,
            "modality": None,
        }

        attrs = read_span_attributes(row)
        assert attrs["input.value"] == "find the report on q3 revenue"
        assert attrs["operation"] == OP_ROUTING

    @pytest.mark.asyncio
    async def test_entity_extraction_emits_custom_span(self, real_telemetry):
        """EntityExtractionAgent emits cogniverse.entity_extraction span."""
        from cogniverse_agents.entity_extraction_agent import (
            Entity,
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionInput,
        )

        tenant_id = _tenant_id("entity-extraction-custom-span")
        with patch.object(EntityExtractionAgent, "_initialize_extractors"):
            agent = EntityExtractionAgent(deps=EntityExtractionDeps(), port=19010)
        agent._gliner_extractor = None
        agent._spacy_analyzer = None
        agent.set_telemetry_manager(real_telemetry)

        query = "PyTorch and Meta AI in Menlo Park"
        extracted_entities = [
            Entity(
                text="PyTorch",
                type="TECHNOLOGY",
                confidence=0.91,
                context="",
            ),
            Entity(
                text="Meta AI",
                type="ORG",
                confidence=0.87,
                context="",
            ),
        ]
        with patch.object(agent, "_extract_dspy_path", return_value=extracted_entities):
            result = await agent.process(
                EntityExtractionInput(
                    query=query,
                    tenant_id=tenant_id,
                )
            )

        assert result.model_dump() == {
            "query": query,
            "entities": [
                {
                    "text": "PyTorch",
                    "type": "TECHNOLOGY",
                    "confidence": 0.91,
                    "context": "",
                },
                {
                    "text": "Meta AI",
                    "type": "ORG",
                    "confidence": 0.87,
                    "context": "",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["TECHNOLOGY", "ORG"],
            "path_used": "dspy",
        }

        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            SPAN_NAME_ENTITY_EXTRACTION, project_name, phoenix_url
        )
        assert span["name"] == SPAN_NAME_ENTITY_EXTRACTION

        span_io = read_span_io(span)
        assert span_io == {
            "input": query,
            "output": {
                "entities": [
                    {
                        "text": "PyTorch",
                        "type": "TECHNOLOGY",
                        "confidence": 0.91,
                        "context": "",
                    },
                    {
                        "text": "Meta AI",
                        "type": "ORG",
                        "confidence": 0.87,
                        "context": "",
                    },
                ],
                "relationships": [],
                "entity_count": 2,
                "relationship_count": 0,
                "path_used": "dspy",
            },
            "operation": OP_ENTITY_EXTRACTION,
            "modality": None,
        }

        attrs = read_span_attributes(span)
        assert attrs["input.value"] == query
        assert attrs["operation"] == OP_ENTITY_EXTRACTION

    @pytest.mark.asyncio
    async def test_query_enhancement_emits_custom_span(self, real_telemetry):
        """QueryEnhancementAgent emits cogniverse.query_enhancement span."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        tenant_id = _tenant_id("query-enhancement-custom-span")
        agent = QueryEnhancementAgent(deps=QueryEnhancementDeps(), port=19012)
        agent.set_telemetry_manager(real_telemetry)

        # Mock DSPy call to avoid requiring an LLM
        mock_result = MagicMock()
        mock_result.enhanced_query = "machine learning tutorials guides"
        mock_result.expansion_terms = "deep learning, neural networks"
        mock_result.synonyms = "ML, AI"
        mock_result.context = "education, technology"
        mock_result.confidence = "0.85"
        mock_result.reasoning = "Added related terms for ML"
        query = "ML tutorials"
        source_text = "ML tutorials source text about machine learning"

        with patch.object(agent, "call_dspy", return_value=mock_result):
            result = await agent.process(
                QueryEnhancementInput(
                    query=query,
                    source_text=source_text,
                    tenant_id=tenant_id,
                )
            )

        assert result.model_dump() == {
            "original_query": query,
            "enhanced_query": "machine learning tutorials guides",
            "expansion_terms": ["deep learning", "neural networks"],
            "synonyms": ["ML", "AI"],
            "context_additions": ["education", "technology"],
            "query_variants": [
                "machine learning tutorials guides",
                "ML tutorials deep learning neural networks",
            ],
            "confidence": 0.85,
            "reasoning": "Added related terms for ML",
        }

        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            SPAN_NAME_QUERY_ENHANCEMENT, project_name, phoenix_url
        )
        assert span["name"] == SPAN_NAME_QUERY_ENHANCEMENT

        # The complete call round-trips through Phoenix: every prompt input
        # and every produced field, in the slots the SIMBA optimizer reads.
        span_io = read_span_io(span)
        assert span_io == {
            "input": query,
            "output": {
                "enhanced_query": "machine learning tutorials guides",
                "expansion_terms": ["deep learning", "neural networks"],
                "synonyms": ["ML", "AI"],
                "context_additions": ["education", "technology"],
                "variant_count": 2,
                "confidence": 0.85,
            },
            "operation": OP_QUERY_ENHANCEMENT,
            "modality": None,
        }
        attrs = read_span_attributes(span)
        assert attrs["input.value"] == query
        assert attrs["operation"] == OP_QUERY_ENHANCEMENT
        assert attrs["input.source_text"] == source_text
        assert attrs["input.grounding_context"] == ""

    @pytest.mark.asyncio
    async def test_profile_selection_emits_custom_span(self, real_telemetry):
        """ProfileSelectionAgent emits cogniverse.profile_selection span."""
        from cogniverse_agents.profile_selection_agent import (
            ProfileSelectionAgent,
            ProfileSelectionDeps,
            ProfileSelectionInput,
        )

        tenant_id = _tenant_id("profile-selection-custom-span")
        available_profiles = ["video_colpali_smol500_mv_frame"]
        agent = ProfileSelectionAgent(
            deps=ProfileSelectionDeps(available_profiles=available_profiles),
            port=19011,
        )
        agent.set_telemetry_manager(real_telemetry)

        # Mock DSPy call to avoid requiring an LLM
        mock_result = MagicMock()
        mock_result.selected_profile = available_profiles[0]
        mock_result.confidence = "0.8"
        mock_result.reasoning = "Video query matched colpali profile"
        mock_result.query_intent = "video_search"
        mock_result.modality = "video"
        mock_result.complexity = "simple"
        query = "show me cooking videos"

        with patch.object(agent, "call_dspy", return_value=mock_result):
            result = await agent.process(
                ProfileSelectionInput(
                    query=query,
                    available_profiles=available_profiles,
                    tenant_id=tenant_id,
                )
            )

        assert result.model_dump() == {
            "query": query,
            "selected_profile": available_profiles[0],
            "confidence": 0.8,
            "reasoning": "Video query matched colpali profile",
            "query_intent": "video_search",
            "modality": "video",
            "complexity": "simple",
            "alternatives": [],
        }

        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            SPAN_NAME_PROFILE_SELECTION, project_name, phoenix_url
        )
        assert span["name"] == SPAN_NAME_PROFILE_SELECTION

        span_io = read_span_io(span)
        assert span_io == {
            "input": query,
            "output": {
                "selected_profile": available_profiles[0],
                "modality": "video",
                "complexity": "simple",
                "intent": "video_search",
                "confidence": 0.8,
            },
            "operation": OP_PROFILE_SELECTION,
            "modality": None,
        }

        attrs = read_span_attributes(span)
        assert attrs["input.value"] == query
        assert attrs["operation"] == OP_PROFILE_SELECTION
        assert attrs["available_profiles"] == ", ".join(available_profiles)

    @pytest.mark.asyncio
    async def test_orchestrator_emits_custom_span(self, real_telemetry):
        """OrchestratorAgent emits cogniverse.orchestration span."""
        import cogniverse_agents.orchestrator_agent as orchestrator_mod
        from cogniverse_agents.orchestrator_agent import (
            AccumulatedEvidence,
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
        registry.tenant_id = "test:unit"
        registry.config_manager = MagicMock()
        registry.config = {}
        registry._http_client = MagicMock()

        registry.register_agent(
            AgentEndpoint(
                name="search_agent",
                url="http://localhost:8002",
                capabilities=["search"],
                process_endpoint="/tasks/send",
            )
        )

        from cogniverse_foundation.config.unified_config import SystemConfig

        _stub_sys_cfg = SystemConfig(
            backend_url="http://localhost",
            backend_port=8080,
            iter_retrieval_max_iter=3,
            iter_retrieval_token_budget=10000,
            iter_retrieval_wall_clock_ms=10000,
        )
        mock_cm = MagicMock()
        mock_cm.get_system_config.return_value = _stub_sys_cfg
        mock_cm.get_config.return_value = {}

        agent = OrchestratorAgent(
            deps=OrchestratorDeps(),
            registry=registry,
            config_manager=mock_cm,
            port=19013,
        )
        agent.set_telemetry_manager(real_telemetry)
        agent.remember_success = MagicMock()

        tenant_id = _tenant_id("orchestration-custom-span")
        query = "find machine learning videos"
        mock_plan = OrchestrationPlan(
            query=query,
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": query},
                    depends_on=[],
                    reasoning="Search for ML videos",
                ),
            ],
            parallel_groups=[],
            reasoning="Single search step",
            unavailable_agents=[],
        )
        final_output = {"content": "found it", "confidence": 1.0}

        async def _fake_iterative_retrieval_loop(
            *,
            query,
            plan,
            tenant_id,
            workflow_id,
            session_id,
            agent_results_sink,
            inbound_queue,
            execution_order_sink,
            agent_observations_sink,
            seed_evidence,
        ):
            agent_results_sink["search_agent"] = {
                "status": "success",
                "results": [{"id": "doc-1"}],
            }
            execution_order_sink.append("search_agent")
            return AccumulatedEvidence(
                evidence=[],
                iterations_executed=1,
                exit_reason="complete",
                final_gate_output={"missing_aspects": []},
                partial_due_to_budget=False,
                partial_due_to_timeout=False,
                trace_id="",
                inbound_constraints_applied=[],
                loop_trajectory=[],
                duration_ms=0.0,
                per_iter_duration_ms=[],
            )

        with (
            patch.object(agent, "_create_plan", return_value=mock_plan),
            patch.object(agent, "_aggregate_results", return_value=final_output),
            patch.object(
                orchestrator_mod.time,
                "monotonic",
                side_effect=[100.0, 101.0, 103.0, 107.5],
            ),
            patch.object(
                agent,
                "_iterative_retrieval_loop",
                side_effect=_fake_iterative_retrieval_loop,
            ),
        ):
            result = await agent.process(
                OrchestratorInput(
                    query=query,
                    tenant_id=tenant_id,
                )
            )

        assert result.model_dump() == {
            "query": query,
            "workflow_id": result.workflow_id,
            "plan_steps": [
                {
                    "agent_name": "search_agent",
                    "reasoning": "Search for ML videos",
                    "depends_on": [],
                }
            ],
            "parallel_groups": [],
            "plan_reasoning": "Single search step",
            "agent_results": {
                "search_agent": {"status": "success", "results": [{"id": "doc-1"}]}
            },
            "final_output": final_output,
            "execution_summary": (
                "Executed 1/1 steps (1 successful). Plan: Single search step"
            ),
            "metadata": {
                "iterative_loop": {
                    "iterations_executed": 1,
                    "exit_reason": "complete",
                    "evidence_count": 0,
                    "final_gate": {"missing_aspects": []},
                    "partial_due_to_budget": False,
                    "partial_due_to_timeout": False,
                    "trace_id": "",
                    "top_hits": [],
                    "missing_aspects": [],
                    "final_answer_id": "",
                    "inbound_constraints_applied": [],
                    "loop_trajectory": [],
                    "duration_ms": 0.0,
                    "per_iter_duration_ms": [],
                    "accumulated_evidence": [],
                }
            },
        }

        await asyncio.sleep(2)

        project_name = _project_name(real_telemetry, tenant_id)
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span(
            SPAN_NAME_ORCHESTRATION, project_name, phoenix_url
        )
        assert span["name"] == SPAN_NAME_ORCHESTRATION

        span_io = read_span_io(span)
        assert span_io == {
            "input": query,
            "output": {
                "workflow_id": result.workflow_id,
                "agent_sequence": ["search_agent"],
                "execution_order": ["search_agent"],
                "pattern": "sequential",
                "execution_time": 7.5,
                "success": True,
                "tasks_completed": 1,
            },
            "operation": OP_ORCHESTRATION,
            "modality": None,
        }

        attrs = read_span_attributes(span)
        assert attrs["input.value"] == query
        assert attrs["operation"] == OP_ORCHESTRATION

    @pytest.mark.asyncio
    async def test_all_a2a_custom_span_names_documented(self, real_telemetry):
        """Verify our test covers all known A2A custom span names."""
        # This is a meta-test: if someone adds a new A2A agent with a custom
        # span but forgets to add a test case above, this will catch it.
        expected_spans = {
            SPAN_NAME_GATEWAY,
            SPAN_NAME_ENTITY_EXTRACTION,
            SPAN_NAME_QUERY_ENHANCEMENT,
            SPAN_NAME_PROFILE_SELECTION,
            SPAN_NAME_ORCHESTRATION,
        }
        tested_spans = set(A2A_CUSTOM_SPANS.keys())
        assert tested_spans == expected_spans, (
            f"A2A_CUSTOM_SPANS is out of date. "
            f"Missing: {expected_spans - tested_spans}, "
            f"Extra: {tested_spans - expected_spans}"
        )
