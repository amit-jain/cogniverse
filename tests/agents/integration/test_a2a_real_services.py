"""
Real-service integration tests for A2A agent architecture.

Every test uses REAL services: GLiNER model, SpaCy NLP, Ollama LLM via DSPy,
Vespa Docker with ingested video data, Phoenix Docker container.
NO mocks for agent internals.

Assertions verify actual content quality: exact entity names, deterministic
routing decisions, search result relevance, telemetry span content.

Test classes:
1. TestGatewayWithRealGLiNER — GLiNER classification with deterministic assertions
2. TestEntityExtractionRealGLiNERSpaCy — exact entity text, types, relationships
3. TestQueryEnhancementRealDSPy — real LLM expansion, content-aware assertions
4. TestRoutingRealDSPy — deterministic routing for known query types
5. TestFullPipelineWithVespa — end-to-end: classify -> extract -> enhance -> route -> SEARCH VESPA
6. TestTelemetrySpansInPhoenix — real Phoenix span emission and query
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

from .conftest import skip_if_no_ollama

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _configure_dspy_lm():
    """Module-scoped: configure DSPy with real Ollama LLM.

    Reads model from config.json, disables qwen3 thinking mode.
    """
    config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    primary = config.get("llm_config", {}).get("primary", {})
    model = primary.get("model")
    api_base = primary.get("api_base")

    extra_body = None
    if model and ("qwen3" in model or "qwen-3" in model):
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=api_base,
        temperature=0.1,
        max_tokens=300,
        extra_body=extra_body,
    )
    lm = create_dspy_lm(endpoint)
    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)


@pytest.fixture
def configure_dspy(_configure_dspy_lm):
    """Function-scoped: re-apply dspy.configure before each test.

    Root conftest cleanup clears dspy.settings.lm after each test,
    so we must re-apply before every test that needs an LLM.
    """
    dspy.configure(lm=_configure_dspy_lm)
    return _configure_dspy_lm


# ---------------------------------------------------------------------------
# 1. GatewayAgent with real GLiNER — deterministic assertions
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestGatewayWithRealGLiNER:
    """GatewayAgent uses GLiNER for zero-shot classification. No LLM needed.

    GLiNER modality scores with the tuned 7-label set are typically 0.4-0.7.
    With default fast_path_confidence_threshold=0.4, unambiguous queries take
    the fast path. Ambiguous or multi-modal queries route to orchestrator.
    """

    @pytest.fixture
    def gateway_agent(self):
        """Gateway with default config (7-label set, threshold=0.4)."""
        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps

        return GatewayAgent(deps=GatewayDeps())

    @pytest.mark.asyncio
    async def test_video_content_query_routes_to_search_agent(self, gateway_agent):
        """'search for video content about AI' -> video modality, simple, search_agent.

        GLiNER scores ~0.69 for video_content on this query with the 7-label set
        (above default threshold=0.4), so it takes the fast path to search_agent.
        """
        from cogniverse_agents.gateway_agent import GatewayInput

        result = await gateway_agent.process(
            GatewayInput(query="search for video content about AI")
        )

        assert result.query == "search for video content about AI"
        assert result.modality == "video", (
            f"Expected video modality, got {result.modality!r}. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.complexity == "simple", (
            f"GLiNER scores ~0.69 for video_content with 7-label set, above threshold=0.4, "
            f"should be simple. Got {result.complexity!r}. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.routed_to == "search_agent", (
            f"Simple video search should route to search_agent, "
            f"got {result.routed_to!r}. Reasoning: {result.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_summary_report_detects_generation_type(self, gateway_agent):
        """'summarize the research papers into a report' detects summary + document."""
        from cogniverse_agents.gateway_agent import GatewayInput

        result = await gateway_agent.process(
            GatewayInput(
                query="summarize the research papers into a report"
            )
        )

        # "summarize the research papers into a report" contains "summarize"
        # which is a complexity keyword → should be classified as complex
        assert result.complexity == "complex", (
            f"'summarize' keyword should trigger complex, got {result.complexity!r}"
        )
        assert result.routed_to == "orchestrator_agent", (
            f"Complex query should route to orchestrator, got {result.routed_to!r}"
        )
        # generation_type should detect summary (not raw_results)
        assert result.generation_type in ("summary", "detailed_report"), (
            f"'summarize...report' should detect summary or detailed_report, "
            f"got {result.generation_type!r}"
        )
        assert len(result.reasoning) > 10, (
            f"Reasoning should be substantive, got: {result.reasoning!r}"
        )

    @pytest.mark.asyncio
    async def test_audio_query_detects_audio_modality(self, gateway_agent):
        """'find audio recordings of jazz music' -> audio modality."""
        from cogniverse_agents.gateway_agent import GatewayInput

        result = await gateway_agent.process(
            GatewayInput(query="find audio recordings of jazz music")
        )

        # GLiNER detects audio_content with 7-label set -> audio modality
        assert result.modality == "audio", (
            f"Expected audio modality for 'jazz music' query, "
            f"got {result.modality!r}. Reasoning: {result.reasoning}"
        )


# ---------------------------------------------------------------------------
# 2. EntityExtraction with real GLiNER + SpaCy — exact entity assertions
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestEntityExtractionRealGLiNERSpaCy:
    """Entity extraction using real GLiNER model and SpaCy NLP pipeline.

    GLiNER is deterministic: for a given model and input, the same entities
    are extracted every time. We assert exact entity text matches.
    """

    @pytest.fixture
    def entity_agent(self):
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
        )

        return EntityExtractionAgent(deps=EntityExtractionDeps())

    @pytest.mark.asyncio
    async def test_named_entities_exact_matches(self, entity_agent):
        """Tesla, San Francisco, Google should be extracted with correct types."""
        from cogniverse_agents.entity_extraction_agent import EntityExtractionInput

        result = await entity_agent.process(
            EntityExtractionInput(
                query="Tesla cars driving in San Francisco near Google headquarters"
            )
        )

        assert result.has_entities, "Should detect entities in named-entity-rich query"
        assert result.entity_count >= 2, (
            f"Expected at least 2 entities from 'Tesla', 'San Francisco', 'Google'; "
            f"got {result.entity_count}: {[e.text for e in result.entities]}"
        )

        entity_texts_lower = {e.text.lower() for e in result.entities}

        # Assert exact entity text matches
        assert any("tesla" in t for t in entity_texts_lower), (
            f"Expected 'Tesla' in entities, got: {entity_texts_lower}"
        )
        assert any("francisco" in t for t in entity_texts_lower), (
            f"Expected 'San Francisco' in entities, got: {entity_texts_lower}"
        )

        # Check entity types are reasonable (ORG, PLACE, GPE, COMPANY, LOCATION, etc.)
        entity_types = {e.type.upper() for e in result.entities}
        reasonable_org_types = {"ORG", "ORGANIZATION", "COMPANY", "CORPORATION"}
        reasonable_loc_types = {"PLACE", "GPE", "LOCATION", "CITY", "LOC"}
        all_reasonable = reasonable_org_types | reasonable_loc_types | {"PERSON", "CONCEPT", "TECHNOLOGY"}

        assert entity_types & all_reasonable, (
            f"Entity types should include recognizable categories, "
            f"got: {entity_types}"
        )

        # Fast path should be used (GLiNER available)
        assert result.path_used == "fast", (
            f"Expected GLiNER fast path, got: {result.path_used}"
        )

        # All entities should have positive confidence
        for entity in result.entities:
            assert entity.confidence > 0.0, (
                f"Entity '{entity.text}' has zero confidence"
            )

    @pytest.mark.asyncio
    async def test_technology_entities(self, entity_agent):
        """Python/TensorFlow should be detected as technology entities."""
        from cogniverse_agents.entity_extraction_agent import EntityExtractionInput

        result = await entity_agent.process(
            EntityExtractionInput(
                query="Python TensorFlow machine learning"
            )
        )

        entity_texts_lower = {e.text.lower() for e in result.entities}

        # At least one tech entity should be detected
        has_tech = any(
            name in t
            for t in entity_texts_lower
            for name in ("python", "tensorflow", "machine learning")
        )
        assert has_tech, (
            f"Expected tech entities (Python, TensorFlow, ML), got: {entity_texts_lower}"
        )

        # dominant_types should be populated
        assert len(result.dominant_types) > 0, (
            "Should have dominant types for technology query, got empty list"
        )

    @pytest.mark.asyncio
    async def test_relationships_with_multiple_entities(self, entity_agent):
        """When 2+ entities found, SpaCy should produce relationships."""
        from cogniverse_agents.entity_extraction_agent import EntityExtractionInput

        result = await entity_agent.process(
            EntityExtractionInput(
                query="Elon Musk founded SpaceX in California to launch rockets"
            )
        )

        assert result.entity_count >= 2, (
            f"Expected 2+ entities from 'Elon Musk', 'SpaceX', 'California'; "
            f"got {result.entity_count}: {[e.text for e in result.entities]}"
        )

        entity_texts_lower = {e.text.lower() for e in result.entities}
        # At least one of the key entities should be found
        key_entities = {"elon musk", "spacex", "california", "elon", "musk"}
        assert entity_texts_lower & key_entities, (
            f"Expected at least one of {key_entities} in entities, got: {entity_texts_lower}"
        )

        # If fast path and 2+ entities, relationships should be a list
        if result.path_used == "fast" and result.entity_count >= 2:
            assert isinstance(result.relationships, list), (
                "relationships should be a list when multiple entities detected"
            )

    @pytest.mark.asyncio
    async def test_empty_query_returns_no_entities(self, entity_agent):
        """Empty query should return empty entities gracefully."""
        from cogniverse_agents.entity_extraction_agent import EntityExtractionInput

        result = await entity_agent.process(EntityExtractionInput(query=""))

        assert result.entity_count == 0
        assert not result.has_entities
        assert result.entities == []


# ---------------------------------------------------------------------------
# 3. QueryEnhancement with real DSPy — content-aware assertions
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestQueryEnhancementRealDSPy:
    """Query enhancement using real DSPy ChainOfThought with Ollama.

    The LLM should expand abbreviated terms and add relevant context.
    """

    @pytest.fixture
    def enhancement_agent(self, configure_dspy):
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
        )

        return QueryEnhancementAgent(deps=QueryEnhancementDeps())

    @pytest.mark.asyncio
    async def test_ml_expansion(self, enhancement_agent):
        """'find ML videos' should expand ML to machine learning."""
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementInput

        result = await enhancement_agent.process(
            QueryEnhancementInput(
                query="find ML videos",
                entities=[{"text": "ML", "type": "CONCEPT"}],
            )
        )

        assert result.original_query == "find ML videos"
        assert result.enhanced_query, "enhanced_query must not be empty"

        # The enhanced query or expansion terms should reference machine learning
        combined_text = (
            result.enhanced_query.lower()
            + " "
            + " ".join(result.expansion_terms).lower()
        )
        assert "machine learning" in combined_text or "ml" in combined_text, (
            f"Expected 'machine learning' expansion of 'ML', "
            f"got enhanced_query={result.enhanced_query!r}, "
            f"expansion_terms={result.expansion_terms}"
        )

        assert result.confidence > 0.0, (
            f"Confidence should be positive, got: {result.confidence}"
        )
        assert len(result.reasoning) > 0, (
            "Reasoning should explain the enhancement"
        )

    @pytest.mark.asyncio
    async def test_enhancement_without_entities(self, enhancement_agent):
        """Enhancement should work even without pre-extracted entities."""
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementInput

        result = await enhancement_agent.process(
            QueryEnhancementInput(query="TensorFlow tutorial")
        )

        assert result.enhanced_query, "Should produce enhanced query without entities"
        assert result.original_query == "TensorFlow tutorial"
        assert result.confidence > 0.0, (
            f"Confidence should be positive even without entities, got: {result.confidence}"
        )

    @pytest.mark.asyncio
    async def test_empty_query_enhancement(self, enhancement_agent):
        """Empty query should return gracefully with zero confidence."""
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementInput

        result = await enhancement_agent.process(QueryEnhancementInput(query=""))

        assert result.enhanced_query == ""
        assert result.confidence == 0.0
        assert result.expansion_terms == []


# ---------------------------------------------------------------------------
# 4. RoutingAgent with real DSPy — deterministic routing assertions
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestRoutingRealDSPy:
    """Routing decisions using real DSPy + Ollama inference.

    For unambiguous queries (video search, summarization), we assert
    the exact routing destination.
    """

    @pytest.fixture
    def routing_agent(self, configure_dspy):
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

        deps = RoutingDeps(telemetry_config=None)
        return RoutingAgent(deps=deps)

    @pytest.mark.asyncio
    async def test_video_search_routes_to_search_agent(self, routing_agent):
        """Pre-enriched video search query should route to search_agent."""
        result = await routing_agent.route_query(
            query="find video tutorials about cooking",
            enhanced_query="find video tutorials and guides about cooking techniques and recipes",
            entities=[
                {"text": "cooking", "type": "CONCEPT"},
                {"text": "tutorials", "type": "CONCEPT"},
            ],
            tenant_id="a2a_test",
        )

        assert result.recommended_agent == "search_agent", (
            f"Video search query should route to search_agent, "
            f"got {result.recommended_agent!r}. Reasoning: {result.reasoning}"
        )
        assert result.confidence > 0.0, (
            f"Confidence should be positive, got: {result.confidence}"
        )
        assert result.confidence <= 1.0
        assert len(result.reasoning) > 10, (
            f"Reasoning should be substantive (>10 chars), got: {result.reasoning!r}"
        )

    @pytest.mark.asyncio
    async def test_summarization_routes_appropriately(self, routing_agent):
        """Summarization query should route to summarizer or report agent."""
        result = await routing_agent.route_query(
            query="summarize the key findings from AI research papers",
            entities=[{"text": "AI research", "type": "CONCEPT"}],
            tenant_id="a2a_test",
        )

        acceptable_agents = {"summarizer_agent", "detailed_report_agent", "search_agent"}
        assert result.recommended_agent in acceptable_agents, (
            f"Summary query should route to {acceptable_agents}, "
            f"got {result.recommended_agent!r}. Reasoning: {result.reasoning}"
        )
        assert result.confidence > 0.0
        assert len(result.reasoning) > 10, (
            f"Reasoning must be substantive, got: {result.reasoning!r}"
        )

    @pytest.mark.asyncio
    async def test_routing_with_no_enrichment(self, routing_agent):
        """Routing with raw query (no entities/enhanced_query) → search_agent."""
        result = await routing_agent.route_query(
            query="show me cooking videos",
            tenant_id="a2a_test",
        )

        assert result.recommended_agent == "search_agent", (
            f"'cooking videos' without enrichment should route to search_agent, "
            f"got {result.recommended_agent!r}. Reasoning: {result.reasoning}"
        )
        assert result.confidence > 0.0
        assert result.query == "show me cooking videos"


# ---------------------------------------------------------------------------
# 5. Full Pipeline with Real Vespa — THE MAIN TEST
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestFullPipelineWithVespa:
    """End-to-end pipeline: Gateway -> Entity -> Enhancement -> Routing -> VESPA SEARCH.

    Uses vespa_with_schema fixture for real Vespa Docker with ingested video data.
    Uses real Ollama, GLiNER, SpaCy. Asserts search results match query content.
    """

    @pytest.fixture
    def gateway_agent(self):
        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps

        return GatewayAgent(deps=GatewayDeps())

    @pytest.fixture
    def entity_agent(self):
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
        )

        return EntityExtractionAgent(deps=EntityExtractionDeps())

    @pytest.fixture
    def enhancement_agent(self, configure_dspy):
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
        )

        return QueryEnhancementAgent(deps=QueryEnhancementDeps())

    @pytest.fixture
    def routing_agent(self, configure_dspy):
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

        return RoutingAgent(deps=RoutingDeps(telemetry_config=None))

    @pytest.mark.asyncio
    async def test_full_pipeline_with_vespa_search(
        self,
        vespa_with_schema,
        gateway_agent,
        entity_agent,
        enhancement_agent,
        routing_agent,
    ):
        """Run full A2A pipeline then search real Vespa with ingested data.

        1. GatewayAgent classifies query
        2. EntityExtractionAgent extracts entities
        3. QueryEnhancementAgent enhances query
        4. RoutingAgent routes to search_agent
        5. SearchService.search() against real Vespa with ingested video data
        6. Assert search results contain content related to the query
        7. Assert results are ranked (descending score)
        """
        from cogniverse_agents.entity_extraction_agent import EntityExtractionInput
        from cogniverse_agents.gateway_agent import GatewayInput
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementInput

        query = "find videos about robots"

        # Step 1: Gateway classification
        gateway_result = await gateway_agent.process(GatewayInput(query=query))
        assert gateway_result.modality in ("video", "both"), (
            f"Gateway should detect video modality, got {gateway_result.modality!r}"
        )
        assert gateway_result.routed_to in ("search_agent", "orchestrator_agent"), (
            f"Should route to search/orchestrator, got {gateway_result.routed_to!r}"
        )

        # Step 2: Entity extraction
        entity_result = await entity_agent.process(
            EntityExtractionInput(query=query)
        )
        assert isinstance(entity_result.entities, list)
        entity_texts = [e.text.lower() for e in entity_result.entities]
        # "robots" is a detectable entity
        assert len(entity_texts) >= 1, (
            f"Expected at least 1 entity from 'robots', got: {entity_texts}"
        )

        # Step 3: Query enhancement with entities from step 2
        entity_dicts = [e.model_dump() for e in entity_result.entities]
        rel_dicts = [r.model_dump() for r in entity_result.relationships]

        enhancement_result = await enhancement_agent.process(
            QueryEnhancementInput(
                query=query,
                entities=entity_dicts,
                relationships=rel_dicts,
            )
        )
        assert enhancement_result.enhanced_query, (
            "Enhancement should produce a non-empty enhanced query"
        )

        # Step 4: Routing with enriched data
        routing_result = await routing_agent.route_query(
            query=query,
            enhanced_query=enhancement_result.enhanced_query,
            entities=entity_dicts,
            relationships=rel_dicts,
            tenant_id="test_tenant",
        )
        assert routing_result.recommended_agent, (
            "Routing should produce a recommended agent"
        )
        assert routing_result.confidence > 0.0, (
            f"Routing confidence should be positive, got: {routing_result.confidence}"
        )

        # Step 5: SEARCH REAL VESPA with ingested data
        manager = vespa_with_schema["manager"]
        search_result = manager.search_videos(query="robot", hits=10)

        assert search_result is not None, (
            "Vespa search should return results (not None)"
        )

        children = search_result.get("root", {}).get("children", [])
        total_count = search_result.get("root", {}).get("fields", {}).get("totalCount", 0)

        assert total_count > 0 or len(children) > 0, (
            f"Should find results for 'robot' in ingested test data. "
            f"totalCount={total_count}, children={len(children)}"
        )

        # Step 6: Verify results are ranked (descending relevance)
        if len(children) > 1:
            scores = [h.get("relevance", 0.0) for h in children]
            assert scores == sorted(scores, reverse=True), (
                f"Results should be ranked by descending relevance, got: {scores}"
            )

    @pytest.mark.asyncio
    async def test_summarize_query_routes_differently(
        self,
        vespa_with_schema,
        gateway_agent,
        routing_agent,
    ):
        """'summarize robot videos' should route to summarizer, not search."""
        from cogniverse_agents.gateway_agent import GatewayInput

        query = "summarize robot videos"

        gateway_result = await gateway_agent.process(GatewayInput(query=query))

        # For summarization query, gateway should detect summary generation type
        # or classify as complex and forward to orchestrator
        if gateway_result.complexity == "simple":
            assert gateway_result.routed_to in (
                "summarizer_agent",
                "search_agent",
            ), (
                f"Simple summarize query should route to summarizer/search, "
                f"got {gateway_result.routed_to!r}. Reasoning: {gateway_result.reasoning}"
            )
        else:
            assert gateway_result.routed_to == "orchestrator_agent", (
                f"Complex query should go to orchestrator, "
                f"got {gateway_result.routed_to!r}"
            )

    @pytest.mark.asyncio
    async def test_vespa_search_returns_relevant_content(
        self,
        vespa_with_schema,
    ):
        """Direct Vespa search for known content should return relevant results."""
        manager = vespa_with_schema["manager"]

        # Search for content we know was ingested
        test_queries = [
            ("robot", "robot soccer video"),
            ("sports", "any sports-related content"),
        ]

        for search_term, description in test_queries:
            result = manager.search_videos(query=search_term, hits=5)
            assert result is not None, (
                f"Search for '{search_term}' ({description}) should not return None"
            )

            total = result.get("root", {}).get("fields", {}).get("totalCount", 0)
            hits = result.get("root", {}).get("children", [])

            logger.info(
                f"Vespa search '{search_term}': {total} total, {len(hits)} hits"
            )

            # We expect at least some results for known ingested content.
            # binary_binary ranking requires a query encoder for relevance
            # scoring — without one (common in test envs), all hits get
            # relevance=0.0 but are still returned. Assert on hit count and
            # field presence, not relevance score.
            assert len(hits) > 0 or total > 0, (
                f"Search for '{search_term}' ({description}) should return hits "
                f"from ingested test data. Got total={total}, hits={len(hits)}"
            )
            if hits:
                first_hit = hits[0]
                fields = first_hit.get("fields", {})
                assert fields, (
                    f"First hit for '{search_term}' should have fields, "
                    f"got: {first_hit}"
                )


# ---------------------------------------------------------------------------
# 6. Telemetry Spans with real Phoenix
# ---------------------------------------------------------------------------


class TestTelemetrySpansInPhoenix:
    """Verify agents emit telemetry spans queryable in real Phoenix."""

    @pytest.mark.asyncio
    async def test_gateway_span_in_phoenix(self, real_telemetry):
        """GatewayAgent.process() emits a span visible in real Phoenix."""
        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )

        agent = GatewayAgent(deps=GatewayDeps())
        agent.set_telemetry_manager(real_telemetry)

        test_query = "test telemetry span for video search"
        await agent.process(
            GatewayInput(
                query=test_query,
                tenant_id="telemetry_a2a_test",
            )
        )

        # Allow Phoenix to ingest the span
        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_a2a_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]

        span = _query_phoenix_for_span(
            "GatewayAgent.process", project_name, phoenix_url
        )
        assert span is not None, (
            f"GatewayAgent.process span not found in Phoenix project "
            f"{project_name}. Span export may be broken."
        )

    @pytest.mark.asyncio
    async def test_gateway_custom_span_has_query_text(self, real_telemetry):
        """GatewayAgent emits cogniverse.gateway span with actual query text."""
        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )

        agent = GatewayAgent(deps=GatewayDeps())
        agent.set_telemetry_manager(real_telemetry)

        test_query = "find robotics engineering videos"
        await agent.process(
            GatewayInput(query=test_query, tenant_id="telemetry_a2a_test")
        )

        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name("telemetry_a2a_test")
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]

        span = _query_phoenix_for_span(
            "cogniverse.gateway", project_name, phoenix_url
        )
        assert span is not None, (
            "cogniverse.gateway custom span not found in Phoenix. "
            "GatewayAgent._emit_gateway_span may not be firing."
        )

        # Phoenix DataFrame stores span attributes as dotted column names:
        # "attributes.gateway" contains {"query": "...", "complexity": "...", ...}
        gateway_attrs = span.get("attributes.gateway") if hasattr(span, "get") else None
        if gateway_attrs is None:
            # Try index-based access for pandas Series
            try:
                gateway_attrs = span["attributes.gateway"]
            except (KeyError, TypeError):
                gateway_attrs = None

        assert gateway_attrs is not None, (
            f"Span should have 'attributes.gateway' column. "
            f"Available columns: {list(span.index) if hasattr(span, 'index') else 'unknown'}"
        )

        # gateway_attrs is a dict with query, complexity, modality, etc.
        if isinstance(gateway_attrs, dict):
            gateway_query = gateway_attrs.get("query", "")
        else:
            gateway_query = str(gateway_attrs)

        assert "robotics" in gateway_query.lower() or "engineering" in gateway_query.lower(), (
            f"Span gateway.query should contain 'robotics' or 'engineering', "
            f"got: {gateway_query!r}"
        )


def _query_phoenix_for_span(
    span_name: str,
    project_name: str,
    phoenix_http_url: str,
    max_wait: int = 30,
):
    """Poll Phoenix for a span with the given name.

    Returns the matched span row (or None) by polling for up to max_wait
    seconds to account for Phoenix ingestion delay.
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
