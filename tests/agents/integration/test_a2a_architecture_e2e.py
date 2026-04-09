"""
End-to-end integration tests for A2A architecture.

Tests the full query flow: GatewayAgent -> OrchestratorAgent -> preprocessing agents -> execution.
Mocks external boundaries (GLiNER, DSPy LM, HTTP, telemetry) but exercises all internal logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
    EntityExtractionOutput,
)
from cogniverse_agents.gateway_agent import (
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
    GatewayOutput,
)
from cogniverse_agents.orchestrator_agent import (
    AgentStep,
    OrchestrationPlan,
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
    OrchestratorOutput,
)
from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
)
from cogniverse_agents.query_enhancement_agent import (
    QueryEnhancementAgent,
    QueryEnhancementDeps,
    QueryEnhancementInput,
    QueryEnhancementOutput,
)
from cogniverse_agents.routing_agent import (
    RoutingAgent,
    RoutingDeps,
    RoutingInput,
    RoutingOutput,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_manager():
    """Minimal ConfigManager mock for agents that require it."""
    cm = MagicMock()
    cm.get_system_config.return_value = MagicMock(
        backend_url="http://localhost",
        backend_port=8080,
    )
    cm.get_config.return_value = {}
    return cm


@pytest.fixture
def mock_agent_registry(mock_config_manager):
    """AgentRegistry with common agents pre-registered."""
    with patch.object(AgentRegistry, "__init__", lambda self, **kw: None):
        registry = AgentRegistry.__new__(AgentRegistry)
    registry.agents = {}
    registry.capabilities = {}
    registry.tenant_id = "default"
    registry.config_manager = mock_config_manager
    registry.config = {}
    registry.http_client = MagicMock()

    agents = [
        AgentEndpoint(
            name="entity_extraction_agent",
            url="http://localhost:8010",
            capabilities=["entity_extraction"],
            process_endpoint="/tasks/send",
        ),
        AgentEndpoint(
            name="query_enhancement_agent",
            url="http://localhost:8012",
            capabilities=["query_enhancement"],
            process_endpoint="/tasks/send",
        ),
        AgentEndpoint(
            name="profile_selection_agent",
            url="http://localhost:8011",
            capabilities=["profile_selection"],
            process_endpoint="/tasks/send",
        ),
        AgentEndpoint(
            name="search_agent",
            url="http://localhost:8002",
            capabilities=["search", "video_search", "retrieval"],
            process_endpoint="/tasks/send",
        ),
        AgentEndpoint(
            name="routing_agent",
            url="http://localhost:8001",
            capabilities=["routing", "intelligent_routing"],
            process_endpoint="/tasks/send",
        ),
        AgentEndpoint(
            name="gateway_agent",
            url="http://localhost:8014",
            capabilities=["gateway", "classification"],
            process_endpoint="/tasks/send",
        ),
        AgentEndpoint(
            name="summarizer_agent",
            url="http://localhost:8003",
            capabilities=["summarization", "text_generation"],
            process_endpoint="/tasks/send",
        ),
    ]
    for agent in agents:
        registry.register_agent(agent)
    return registry


@pytest.fixture
def mock_telemetry_manager():
    """TelemetryManager mock that records span calls."""
    tm = MagicMock()
    tm.spans = []

    def span_side_effect(name, tenant_id="default", attributes=None):
        tm.spans.append({
            "name": name,
            "tenant_id": tenant_id,
            "attributes": attributes or {},
        })
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.set_attribute = MagicMock()
        return ctx

    tm.span = MagicMock(side_effect=span_side_effect)
    return tm


@pytest.fixture
def gateway_agent():
    """GatewayAgent with mocked GLiNER model."""
    deps = GatewayDeps()
    agent = GatewayAgent(deps=deps, port=18014)
    # Pre-set GLiNER model mock so _ensure_model_loaded is a no-op
    agent._gliner_model = MagicMock()
    return agent


@pytest.fixture
def entity_extraction_agent():
    """EntityExtractionAgent with mocked extractors."""
    deps = EntityExtractionDeps()
    with patch(
        "cogniverse_agents.entity_extraction_agent.EntityExtractionAgent._initialize_extractors"
    ):
        agent = EntityExtractionAgent(deps=deps, port=18010)
    agent._gliner_extractor = None
    agent._spacy_analyzer = None
    return agent


@pytest.fixture
def query_enhancement_agent():
    """QueryEnhancementAgent with mocked DSPy module."""
    deps = QueryEnhancementDeps()
    agent = QueryEnhancementAgent(deps=deps, port=18012)
    return agent


@pytest.fixture
def profile_selection_agent():
    """ProfileSelectionAgent with mocked DSPy module."""
    deps = ProfileSelectionDeps()
    agent = ProfileSelectionAgent(deps=deps, port=18011)
    return agent


@pytest.fixture
def routing_agent():
    """RoutingAgent with mocked DSPy LM."""
    deps = RoutingDeps(
        telemetry_config=MagicMock(enabled=False),
        llm_config=LLMEndpointConfig(
            model="ollama/test-model",
            api_base="http://localhost:11434",
        ),
    )
    with patch(
        "cogniverse_agents.routing_agent.create_dspy_lm",
        return_value=MagicMock(),
    ):
        agent = RoutingAgent(deps=deps, port=18001)
    return agent


@pytest.fixture
def orchestrator_agent(mock_agent_registry, mock_config_manager):
    """OrchestratorAgent with mocked DSPy module and registry."""
    deps = OrchestratorDeps()
    with patch(
        "cogniverse_agents.orchestrator_agent.OrchestratorAgent._ensure_memory_for_tenant"
    ):
        agent = OrchestratorAgent(
            deps=deps,
            registry=mock_agent_registry,
            config_manager=mock_config_manager,
            port=18013,
        )
    return agent


# ===========================================================================
# 1. Simple query flow: GatewayAgent -> direct to execution agent
# ===========================================================================


class TestSimpleQueryFlow:
    """Simple query: Gateway classifies and routes directly to execution agent."""

    @pytest.mark.asyncio
    async def test_video_search_simple_route(self, gateway_agent):
        """High-confidence video query routes to search_agent."""
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "machine learning", "label": "video_content", "score": 0.9},
        ]

        result = await gateway_agent._process_impl(
            GatewayInput(query="find videos about machine learning")
        )

        assert isinstance(result, GatewayOutput)
        assert result.complexity == "simple"
        assert result.modality == "video"
        assert result.generation_type == "raw_results"
        assert result.routed_to == "search_agent"
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_summary_request_routes_to_summarizer(self, gateway_agent):
        """Summary generation type routes to summarizer_agent."""
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.85},
            {"text": "summarize", "label": "summary_request", "score": 0.9},
        ]

        result = await gateway_agent._process_impl(
            GatewayInput(query="summarize all videos about AI")
        )

        assert result.complexity == "simple"
        assert result.generation_type == "summary"
        assert result.routed_to == "summarizer_agent"

    @pytest.mark.asyncio
    async def test_low_confidence_routes_to_orchestrator(self, gateway_agent):
        """Low-confidence classification triggers complex path."""
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "data", "label": "video_content", "score": 0.3},
        ]

        result = await gateway_agent._process_impl(
            GatewayInput(query="find some data about things")
        )

        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"
        assert result.confidence < 0.7

    @pytest.mark.asyncio
    async def test_no_entities_routes_to_orchestrator(self, gateway_agent):
        """No entities detected routes to orchestrator for deeper analysis."""
        gateway_agent._gliner_model.predict_entities.return_value = []

        result = await gateway_agent._process_impl(
            GatewayInput(query="hello")
        )

        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"
        assert "No entities" in result.reasoning


# ===========================================================================
# 2. Complex query flow: GatewayAgent -> OrchestratorAgent -> A2A agents
# ===========================================================================


class TestComplexQueryFlow:
    """Complex query: Gateway -> Orchestrator -> A2A agents."""

    @pytest.mark.asyncio
    async def test_multi_modality_triggers_complex(self, gateway_agent):
        """Multiple modalities detected triggers orchestrator routing."""
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.85},
            {"text": "documents", "label": "pdf_content", "score": 0.80},
        ]

        result = await gateway_agent._process_impl(
            GatewayInput(
                query="find videos and documents about AI, then summarize the findings"
            )
        )

        assert result.complexity == "complex"
        assert result.modality == "both"
        assert result.routed_to == "orchestrator_agent"
        assert "Multiple modalities" in result.reasoning

    @pytest.mark.asyncio
    async def test_orchestrator_creates_plan_and_executes(
        self, orchestrator_agent
    ):
        """OrchestratorAgent creates a plan from DSPy and executes it via HTTP."""
        # Mock DSPy call_dspy to return a plan
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.agent_sequence = "entity_extraction_agent,search_agent"
        mock_prediction.parallel_steps = "None"
        mock_prediction.reasoning = "Extract entities first, then search"

        orchestrator_agent.call_dspy = AsyncMock(return_value=mock_prediction)
        orchestrator_agent._ensure_memory_for_tenant = MagicMock()
        orchestrator_agent.get_relevant_context = MagicMock(return_value="")
        orchestrator_agent.remember_success = MagicMock()

        # Mock HTTP responses for agent calls
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "results": []}
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await orchestrator_agent._process_impl(
                OrchestratorInput(
                    query="find videos and documents about AI",
                    tenant_id="test-tenant",
                )
            )

        assert isinstance(result, OrchestratorOutput)
        assert result.query == "find videos and documents about AI"
        assert len(result.plan_steps) == 2
        assert result.plan_steps[0]["agent_name"] == "entity_extraction_agent"
        assert result.plan_steps[1]["agent_name"] == "search_agent"
        assert result.workflow_id.startswith("workflow_")
        assert "entity_extraction_agent" in result.agent_results
        assert "search_agent" in result.agent_results

    @pytest.mark.asyncio
    async def test_orchestrator_handles_unavailable_agents(
        self, orchestrator_agent
    ):
        """Orchestrator records errors for agents not in registry."""
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.agent_sequence = (
            "entity_extraction_agent,nonexistent_agent,search_agent"
        )
        mock_prediction.parallel_steps = "None"
        mock_prediction.reasoning = "Plan with unknown agent"

        orchestrator_agent.call_dspy = AsyncMock(return_value=mock_prediction)
        orchestrator_agent._ensure_memory_for_tenant = MagicMock()
        orchestrator_agent.get_relevant_context = MagicMock(return_value="")
        orchestrator_agent.remember_success = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await orchestrator_agent._process_impl(
                OrchestratorInput(query="test unavailable agent", tenant_id="t1")
            )

        assert "nonexistent_agent" in result.agent_results
        assert result.agent_results["nonexistent_agent"]["status"] == "error"
        assert "not available" in result.agent_results["nonexistent_agent"]["message"]

    @pytest.mark.asyncio
    async def test_orchestrator_parallel_group_parsing(self, orchestrator_agent):
        """Orchestrator parses parallel step groups from DSPy output."""
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.agent_sequence = (
            "entity_extraction_agent,query_enhancement_agent,search_agent"
        )
        mock_prediction.parallel_steps = "0,1|2"
        mock_prediction.reasoning = "Extract and enhance in parallel, then search"

        orchestrator_agent.call_dspy = AsyncMock(return_value=mock_prediction)
        orchestrator_agent._ensure_memory_for_tenant = MagicMock()
        orchestrator_agent.get_relevant_context = MagicMock(return_value="")
        orchestrator_agent.remember_success = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await orchestrator_agent._process_impl(
                OrchestratorInput(query="parallel test", tenant_id="t1")
            )

        assert result.parallel_groups == [[0, 1], [2]]
        assert len(result.plan_steps) == 3


# ===========================================================================
# 3. Entity extraction -> Query enhancement data flow
# ===========================================================================


class TestEntityEnhancementDataFlow:
    """Entity extraction output feeds into query enhancement input."""

    @pytest.mark.asyncio
    async def test_entity_output_uses_type_field(self, entity_extraction_agent):
        """EntityExtractionAgent produces entities with 'type' field, not 'label'."""
        mock_dspy_result = MagicMock()
        mock_dspy_result.entities = "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9"
        mock_dspy_result.entity_types = "PERSON,PLACE"

        entity_extraction_agent.call_dspy = AsyncMock(return_value=mock_dspy_result)

        result = await entity_extraction_agent._process_impl(
            EntityExtractionInput(query="Show me videos about Barack Obama in Chicago")
        )

        assert isinstance(result, EntityExtractionOutput)
        assert result.has_entities is True
        assert result.entity_count == 2
        for entity in result.entities:
            assert hasattr(entity, "type")
            assert entity.type in ("PERSON", "PLACE")
            # Verify 'type' not 'label' is the field name
            entity_dict = entity.model_dump()
            assert "type" in entity_dict
            assert "label" not in entity_dict

    @pytest.mark.asyncio
    async def test_entity_output_feeds_enhancement_input(
        self, entity_extraction_agent, query_enhancement_agent
    ):
        """Full data flow: entities from extraction feed into enhancement."""
        # Step 1: Extract entities
        mock_entity_result = MagicMock()
        mock_entity_result.entities = "machine learning|CONCEPT|0.9\nNeural Networks|CONCEPT|0.85"
        mock_entity_result.entity_types = "CONCEPT"

        entity_extraction_agent.call_dspy = AsyncMock(return_value=mock_entity_result)

        entity_output = await entity_extraction_agent._process_impl(
            EntityExtractionInput(query="find videos about machine learning and neural networks")
        )

        # Step 2: Feed entity output into query enhancement
        entities_for_enhancement = [e.model_dump() for e in entity_output.entities]
        relationships_for_enhancement = [r.model_dump() for r in entity_output.relationships]

        mock_enhance_result = MagicMock()
        mock_enhance_result.enhanced_query = (
            "find videos about machine learning and neural networks deep learning"
        )
        mock_enhance_result.expansion_terms = "deep learning, AI, artificial intelligence"
        mock_enhance_result.synonyms = "ML, NN"
        mock_enhance_result.context = "research, education"
        mock_enhance_result.confidence = "0.85"
        mock_enhance_result.reasoning = "Added ML-related expansion terms"

        query_enhancement_agent.call_dspy = AsyncMock(return_value=mock_enhance_result)

        enhancement_output = await query_enhancement_agent._process_impl(
            QueryEnhancementInput(
                query="find videos about machine learning and neural networks",
                entities=entities_for_enhancement,
                relationships=relationships_for_enhancement,
            )
        )

        assert isinstance(enhancement_output, QueryEnhancementOutput)
        assert enhancement_output.enhanced_query != enhancement_output.original_query
        assert len(enhancement_output.expansion_terms) > 0

    @pytest.mark.asyncio
    async def test_enhancement_builds_entity_context(self, query_enhancement_agent):
        """QueryEnhancementAgent uses entities with 'type' field in context."""
        entities = [
            {"text": "Barack Obama", "type": "PERSON", "confidence": 0.95, "context": ""},
            {"text": "Chicago", "type": "PLACE", "confidence": 0.9, "context": ""},
        ]

        context = query_enhancement_agent._build_entity_context(entities, [])
        assert "Barack Obama (PERSON)" in context
        assert "Chicago (PLACE)" in context

    @pytest.mark.asyncio
    async def test_enhancement_handles_label_fallback(self, query_enhancement_agent):
        """QueryEnhancementAgent handles both 'type' and 'label' in entity dicts."""
        # Old-style with 'label' instead of 'type'
        entities = [
            {"text": "robots", "label": "CONCEPT", "confidence": 0.8, "context": ""},
        ]
        context = query_enhancement_agent._build_entity_context(entities, [])
        assert "robots (CONCEPT)" in context

    @pytest.mark.asyncio
    async def test_enhancement_includes_relationships(self, query_enhancement_agent):
        """Relationships from entity extraction are included in context."""
        relationships = [
            {"subject": "Obama", "relation": "visited", "object": "Chicago"},
        ]
        context = query_enhancement_agent._build_entity_context([], relationships)
        assert "Obama -visited-> Chicago" in context


# ===========================================================================
# 4. Span emission verification
# ===========================================================================


class TestSpanEmission:
    """All agents emit correct telemetry spans."""

    @pytest.mark.asyncio
    async def test_gateway_emits_span(self, gateway_agent, mock_telemetry_manager):
        """GatewayAgent emits cogniverse.gateway span."""
        gateway_agent.telemetry_manager = mock_telemetry_manager
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.9},
        ]

        await gateway_agent._process_impl(
            GatewayInput(query="find videos about robots", tenant_id="t1")
        )

        span_names = [s["name"] for s in mock_telemetry_manager.spans]
        assert "cogniverse.gateway" in span_names

        gateway_span = next(
            s for s in mock_telemetry_manager.spans if s["name"] == "cogniverse.gateway"
        )
        assert gateway_span["tenant_id"] == "t1"
        assert "gateway.complexity" in gateway_span["attributes"]
        assert "gateway.modality" in gateway_span["attributes"]

    @pytest.mark.asyncio
    async def test_entity_extraction_emits_span(
        self, entity_extraction_agent, mock_telemetry_manager
    ):
        """EntityExtractionAgent emits cogniverse.entity_extraction span."""
        entity_extraction_agent.telemetry_manager = mock_telemetry_manager

        mock_result = MagicMock()
        mock_result.entities = "AI|CONCEPT|0.9"
        mock_result.entity_types = "CONCEPT"
        entity_extraction_agent.call_dspy = AsyncMock(return_value=mock_result)

        await entity_extraction_agent._process_impl(
            EntityExtractionInput(query="AI research", tenant_id="t1")
        )

        span_names = [s["name"] for s in mock_telemetry_manager.spans]
        assert "cogniverse.entity_extraction" in span_names

    @pytest.mark.asyncio
    async def test_query_enhancement_emits_span(
        self, query_enhancement_agent, mock_telemetry_manager
    ):
        """QueryEnhancementAgent emits cogniverse.query_enhancement span."""
        query_enhancement_agent.telemetry_manager = mock_telemetry_manager

        mock_result = MagicMock()
        mock_result.enhanced_query = "enhanced query"
        mock_result.expansion_terms = "term1, term2"
        mock_result.synonyms = ""
        mock_result.context = ""
        mock_result.confidence = "0.8"
        mock_result.reasoning = "test"
        query_enhancement_agent.call_dspy = AsyncMock(return_value=mock_result)

        await query_enhancement_agent._process_impl(
            QueryEnhancementInput(query="test query", tenant_id="t1")
        )

        span_names = [s["name"] for s in mock_telemetry_manager.spans]
        assert "cogniverse.query_enhancement" in span_names

    @pytest.mark.asyncio
    async def test_profile_selection_emits_span(
        self, profile_selection_agent, mock_telemetry_manager
    ):
        """ProfileSelectionAgent emits cogniverse.profile_selection span."""
        profile_selection_agent.telemetry_manager = mock_telemetry_manager

        mock_result = MagicMock()
        mock_result.selected_profile = "video_colpali_smol500_mv_frame"
        mock_result.confidence = "0.85"
        mock_result.reasoning = "Video content detected"
        mock_result.query_intent = "video_search"
        mock_result.modality = "video"
        mock_result.complexity = "simple"
        profile_selection_agent.call_dspy = AsyncMock(return_value=mock_result)

        await profile_selection_agent._process_impl(
            ProfileSelectionInput(query="find video tutorials", tenant_id="t1")
        )

        span_names = [s["name"] for s in mock_telemetry_manager.spans]
        assert "cogniverse.profile_selection" in span_names

    @pytest.mark.asyncio
    async def test_routing_agent_emits_span(self, routing_agent, mock_telemetry_manager):
        """RoutingAgent emits cogniverse.routing span."""
        routing_agent.telemetry_manager = mock_telemetry_manager

        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "search_agent"
        mock_prediction.confidence = 0.85
        mock_prediction.reasoning_chain = ["Video search query detected"]
        mock_prediction.routing_decision = {
            "primary_agent": "search_agent",
        }
        mock_prediction.overall_confidence = 0.85

        routing_agent.call_dspy = AsyncMock(return_value=mock_prediction)

        await routing_agent.route_query(
            query="find videos about robots",
            tenant_id="test-tenant",
        )

        span_names = [s["name"] for s in mock_telemetry_manager.spans]
        assert "cogniverse.routing" in span_names

    @pytest.mark.asyncio
    async def test_orchestrator_emits_span(
        self, orchestrator_agent, mock_telemetry_manager
    ):
        """OrchestratorAgent emits cogniverse.orchestration span."""
        orchestrator_agent.telemetry_manager = mock_telemetry_manager

        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.agent_sequence = "search_agent"
        mock_prediction.parallel_steps = "None"
        mock_prediction.reasoning = "Direct search"

        orchestrator_agent.call_dspy = AsyncMock(return_value=mock_prediction)
        orchestrator_agent._ensure_memory_for_tenant = MagicMock()
        orchestrator_agent.get_relevant_context = MagicMock(return_value="")
        orchestrator_agent.remember_success = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            await orchestrator_agent._process_impl(
                OrchestratorInput(query="test orchestration span", tenant_id="t1")
            )

        span_names = [s["name"] for s in mock_telemetry_manager.spans]
        assert "cogniverse.orchestration" in span_names


# ===========================================================================
# 5. Agent dispatcher gateway flow
# ===========================================================================


class TestDispatcherGatewayFlow:
    """AgentDispatcher routes through GatewayAgent."""

    @pytest.fixture
    def dispatcher(self, mock_agent_registry, mock_config_manager):
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        schema_loader = MagicMock()
        return AgentDispatcher(
            agent_registry=mock_agent_registry,
            config_manager=mock_config_manager,
            schema_loader=schema_loader,
        )

    @pytest.mark.asyncio
    async def test_gateway_capability_routes_through_gateway(
        self, dispatcher, mock_agent_registry
    ):
        """Dispatching to agent with 'gateway' capability calls _execute_gateway_task."""
        with patch.object(
            dispatcher, "_execute_gateway_task", new_callable=AsyncMock
        ) as mock_gateway:
            mock_gateway.return_value = {
                "status": "success",
                "agent": "gateway_agent",
                "gateway": {
                    "complexity": "simple",
                    "routed_to": "search_agent",
                },
                "downstream_result": {"results": []},
            }

            result = await dispatcher.dispatch(
                agent_name="gateway_agent",
                query="find videos about robots",
                context={"tenant_id": "test"},
            )

        mock_gateway.assert_awaited_once()
        assert result["status"] == "success"
        assert result["agent"] == "gateway_agent"

    @pytest.mark.asyncio
    async def test_routing_capability_also_routes_through_gateway(
        self, dispatcher
    ):
        """Agent with 'routing' capability also routes through gateway task."""
        with patch.object(
            dispatcher, "_execute_gateway_task", new_callable=AsyncMock
        ) as mock_gateway:
            mock_gateway.return_value = {"status": "success", "agent": "gateway_agent"}

            await dispatcher.dispatch(
                agent_name="routing_agent",
                query="test routing",
                context={"tenant_id": "test"},
            )

        mock_gateway.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_simple_gateway_dispatches_downstream(self, dispatcher):
        """Simple classification dispatches directly to the target execution agent."""
        # Create a real gateway agent with mocked GLiNER
        gateway_agent = GatewayAgent(deps=GatewayDeps())
        gateway_agent._gliner_model = MagicMock()
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.95},
        ]
        dispatcher._gateway_agent = gateway_agent

        with patch.object(
            dispatcher, "_execute_downstream_agent", new_callable=AsyncMock
        ) as mock_downstream:
            mock_downstream.return_value = {
                "status": "success",
                "results": [{"id": "1", "title": "test"}],
            }

            result = await dispatcher._execute_gateway_task(
                query="find videos about robots",
                context={},
                tenant_id="test",
            )

        mock_downstream.assert_awaited_once()
        call_kwargs = mock_downstream.call_args
        assert call_kwargs.kwargs["agent_name"] == "search_agent"
        assert result["gateway"]["complexity"] == "simple"

    @pytest.mark.asyncio
    async def test_complex_gateway_dispatches_to_orchestrator(self, dispatcher):
        """Complex classification dispatches to orchestrator."""
        gateway_agent = GatewayAgent(deps=GatewayDeps())
        gateway_agent._gliner_model = MagicMock()
        gateway_agent._gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.85},
            {"text": "documents", "label": "pdf_content", "score": 0.80},
        ]
        dispatcher._gateway_agent = gateway_agent

        with patch.object(
            dispatcher, "_execute_orchestration_task", new_callable=AsyncMock
        ) as mock_orch:
            mock_orch.return_value = {
                "status": "success",
                "agent": "orchestrator_agent",
            }

            await dispatcher._execute_gateway_task(
                query="find videos and documents about AI then summarize",
                context={},
                tenant_id="test",
            )

        mock_orch.assert_awaited_once()
        # Verify gateway_context was passed to orchestrator
        call_kwargs = mock_orch.call_args
        assert call_kwargs.kwargs.get("gateway_context") is not None
        gw_ctx = call_kwargs.kwargs["gateway_context"]
        assert gw_ctx["modality"] == "both"


# ===========================================================================
# 6. Routing agent accepts pre-enriched input
# ===========================================================================


class TestRoutingAgentThinInterface:
    """RoutingAgent accepts pre-enriched input, no inline preprocessing."""

    @pytest.mark.asyncio
    async def test_routing_accepts_entities_and_enhanced_query(self, routing_agent):
        """RoutingAgent receives entities + relationships + enhanced_query."""
        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "search_agent"
        mock_prediction.confidence = 0.9
        mock_prediction.reasoning_chain = ["Entities indicate video search"]
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.9

        routing_agent.call_dspy = AsyncMock(return_value=mock_prediction)

        entities = [
            {"text": "robots", "type": "CONCEPT", "confidence": 0.9},
            {"text": "soccer", "type": "CONCEPT", "confidence": 0.85},
        ]
        relationships = [
            {"subject": "robots", "relation": "playing", "object": "soccer"},
        ]

        result = await routing_agent.route_query(
            query="find videos of robots playing soccer",
            enhanced_query="find videos of robots playing soccer tournament matches",
            entities=entities,
            relationships=relationships,
            tenant_id="test-tenant",
        )

        assert isinstance(result, RoutingOutput)
        assert result.recommended_agent == "search_agent"
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_routing_uses_enhanced_query_in_dspy(self, routing_agent):
        """RoutingAgent passes enhanced_query (not original) to DSPy module."""
        calls = []

        async def capture_call_dspy(module, output_field, **kwargs):
            calls.append(kwargs)
            prediction = MagicMock()
            prediction.recommended_agent = "search_agent"
            prediction.confidence = 0.8
            prediction.reasoning_chain = ["test"]
            prediction.routing_decision = {"primary_agent": "search_agent"}
            prediction.overall_confidence = 0.8
            return prediction

        routing_agent.call_dspy = capture_call_dspy

        await routing_agent.route_query(
            query="original query",
            enhanced_query="enhanced expanded query with more context",
            tenant_id="test-tenant",
        )

        assert len(calls) == 1
        # The effective query passed to DSPy should be the enhanced one
        assert calls[0]["query"] == "enhanced expanded query with more context"

    @pytest.mark.asyncio
    async def test_routing_falls_back_to_original_when_no_enhancement(
        self, routing_agent
    ):
        """Without enhanced_query, routing uses the original query."""
        calls = []

        async def capture_call_dspy(module, output_field, **kwargs):
            calls.append(kwargs)
            prediction = MagicMock()
            prediction.recommended_agent = "search_agent"
            prediction.confidence = 0.7
            prediction.reasoning_chain = ["fallback"]
            prediction.routing_decision = {"primary_agent": "search_agent"}
            prediction.overall_confidence = 0.7
            return prediction

        routing_agent.call_dspy = capture_call_dspy

        await routing_agent.route_query(
            query="plain query without enhancement",
            tenant_id="test-tenant",
        )

        assert len(calls) == 1
        assert calls[0]["query"] == "plain query without enhancement"

    @pytest.mark.asyncio
    async def test_routing_via_process_impl(self, routing_agent):
        """_process_impl delegates to route_query with all pre-enriched fields."""
        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "search_agent"
        mock_prediction.confidence = 0.85
        mock_prediction.reasoning_chain = ["test"]
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.85

        routing_agent.call_dspy = AsyncMock(return_value=mock_prediction)

        result = await routing_agent._process_impl(
            RoutingInput(
                query="videos of robots",
                enhanced_query="videos of autonomous robots in competitions",
                entities=[{"text": "robots", "type": "CONCEPT", "confidence": 0.9}],
                relationships=[],
                tenant_id="test-tenant",
            )
        )

        assert isinstance(result, RoutingOutput)
        assert result.recommended_agent == "search_agent"

    @pytest.mark.asyncio
    async def test_routing_error_fallback(self, routing_agent):
        """On DSPy failure, routing falls back to search_agent with low confidence."""
        routing_agent.call_dspy = AsyncMock(
            side_effect=RuntimeError("DSPy module crashed")
        )

        result = await routing_agent.route_query(
            query="test query", tenant_id="test-tenant"
        )

        assert result.recommended_agent == "search_agent"
        assert result.confidence <= 0.3
        assert "error" in result.reasoning.lower() or "fallback" in result.reasoning.lower()


# ===========================================================================
# 7. OrchestratorAgent features: streaming, cancellation, fusion
# ===========================================================================


class TestOrchestratorFeatures:
    """OrchestratorAgent streaming, cancellation, fusion."""

    @pytest.mark.asyncio
    async def test_cancel_active_workflow(self, orchestrator_agent):
        """cancel_workflow returns True for active workflow and removes it."""
        plan = OrchestrationPlan(
            query="test",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "test"},
                    reasoning="test step",
                )
            ],
            reasoning="test plan",
        )
        orchestrator_agent.active_workflows["wf_123"] = plan

        assert orchestrator_agent.cancel_workflow("wf_123") is True
        assert "wf_123" not in orchestrator_agent.active_workflows
        assert "wf_123" in orchestrator_agent._cancelled_workflows

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_workflow(self, orchestrator_agent):
        """cancel_workflow returns False for unknown workflow ID."""
        assert orchestrator_agent.cancel_workflow("nonexistent") is False

    @pytest.mark.asyncio
    async def test_fusion_merges_results_from_multiple_agents(
        self, orchestrator_agent
    ):
        """_aggregate_results fuses results from multiple agents."""
        agent_results = {
            "search_agent": {
                "status": "success",
                "results": [{"id": "1", "title": "Result 1"}],
                "confidence": 0.9,
            },
            "video_search_agent": {
                "status": "success",
                "results": [{"id": "2", "title": "Video 1"}],
                "confidence": 0.85,
            },
        }

        fused = orchestrator_agent._aggregate_results(
            "find videos and text about AI", agent_results
        )

        assert fused["status"] == "success"
        assert "search_agent" in fused["results"]
        assert "video_search_agent" in fused["results"]
        assert "fusion_strategy" in fused
        assert "fusion_quality" in fused
        assert fused["fusion_quality"]["modality_count"] >= 1

    @pytest.mark.asyncio
    async def test_fusion_empty_results(self, orchestrator_agent):
        """Fusion handles empty results gracefully."""
        fused = orchestrator_agent._aggregate_results("test", {})
        assert fused["status"] == "success"
        assert fused["results"] == {}

    @pytest.mark.asyncio
    async def test_streaming_emits_progress_events(self, orchestrator_agent):
        """process(stream=True) yields progress events from orchestration."""
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.agent_sequence = "search_agent"
        mock_prediction.parallel_steps = "None"
        mock_prediction.reasoning = "Simple search"

        orchestrator_agent.call_dspy = AsyncMock(return_value=mock_prediction)
        orchestrator_agent._ensure_memory_for_tenant = MagicMock()
        orchestrator_agent.get_relevant_context = MagicMock(return_value="")
        orchestrator_agent.remember_success = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        input_data = OrchestratorInput(query="stream test", tenant_id="t1")

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            events = []
            gen = await orchestrator_agent.process(input_data, stream=True)
            async for event in gen:
                events.append(event)

        # Must have at least one status event and the final result
        status_events = [e for e in events if e.get("type") == "status"]
        final_events = [e for e in events if e.get("type") == "final"]

        assert len(status_events) > 0, "Expected progress status events during streaming"
        assert len(final_events) == 1, "Expected exactly one final event"

        # Verify progress phases from orchestrator pipeline
        phases = [e["phase"] for e in status_events]
        assert "memory_context" in phases or "planning" in phases or "execution" in phases

    @pytest.mark.asyncio
    async def test_conversation_context_formatting(self, orchestrator_agent):
        """Conversation history is formatted as text for DSPy planner."""
        history = [
            {"role": "user", "content": "What videos do you have?"},
            {"role": "agent", "content": "We have many videos about AI."},
            {"role": "user", "content": "Show me the ones about robotics"},
        ]

        context = orchestrator_agent._format_conversation_context(history)

        assert "user: What videos" in context
        assert "agent: We have many" in context
        assert "user: Show me" in context

    @pytest.mark.asyncio
    async def test_conversation_context_empty(self, orchestrator_agent):
        """Empty conversation history returns empty string."""
        assert orchestrator_agent._format_conversation_context(None) == ""
        assert orchestrator_agent._format_conversation_context([]) == ""

    @pytest.mark.asyncio
    async def test_detect_agent_modality(self, orchestrator_agent):
        """Modality detection from agent names works correctly."""
        assert orchestrator_agent._detect_agent_modality("video_search_agent") == "video"
        assert orchestrator_agent._detect_agent_modality("image_search_agent") == "image"
        assert orchestrator_agent._detect_agent_modality("audio_analysis_agent") == "audio"
        assert orchestrator_agent._detect_agent_modality("document_agent") == "document"
        assert orchestrator_agent._detect_agent_modality("search_agent") == "text"

    @pytest.mark.asyncio
    async def test_orchestrator_empty_query(self, orchestrator_agent):
        """Empty query returns early with error output."""
        orchestrator_agent._ensure_memory_for_tenant = MagicMock()

        result = await orchestrator_agent._process_impl(
            OrchestratorInput(query="", tenant_id="t1")
        )

        assert result.final_output["status"] == "error"
        assert "Empty query" in result.final_output["message"]
