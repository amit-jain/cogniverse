"""Unit tests for EntityExtractionAgent"""

from unittest.mock import MagicMock, Mock, patch

import dspy
import pytest

from cogniverse_agents.entity_extraction_agent import (
    Entity,
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
    EntityExtractionModule,
    EntityExtractionOutput,
    Relationship,
)
from cogniverse_core.common.tenant_utils import TEST_TENANT_ID


def _make_extraction_agent():
    """Create EntityExtractionAgent with mocked DSPy for use in tests."""
    with patch("dspy.ChainOfThought"):
        deps = EntityExtractionDeps()
        return EntityExtractionAgent(deps=deps, port=8010)


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model"""
    lm = Mock()
    lm.return_value = dspy.Prediction(
        entities="Barack Obama|PERSON|0.95\nChicago|PLACE|0.9",
        entity_types="PERSON, PLACE",
    )
    return lm


@pytest.fixture
def entity_agent():
    """Create EntityExtractionAgent for testing (DSPy fallback mode)."""
    with patch("dspy.ChainOfThought"):
        deps = EntityExtractionDeps()
        agent = EntityExtractionAgent(deps=deps, port=8010)
        # Force DSPy fallback path for existing tests
        agent._gliner_extractor = None
        agent._spacy_analyzer = None
        return agent


class TestEntityExtractionModule:
    """Test DSPy module for entity extraction"""

    def test_module_initialization(self):
        """Test EntityExtractionModule initializes correctly"""
        with patch("dspy.ChainOfThought") as mock_cot:
            module = EntityExtractionModule()
            assert module.extractor is not None
            mock_cot.assert_called_once()

    def test_forward_success(self, mock_dspy_lm):
        """Test successful entity extraction"""
        module = EntityExtractionModule()
        module.extractor = mock_dspy_lm

        result = module.forward(query="Show me Barack Obama in Chicago")

        assert result.entities == "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9"
        assert result.entity_types == "PERSON, PLACE"

    def test_forward_fallback(self):
        """Test fallback when DSPy fails"""
        module = EntityExtractionModule()
        module.extractor = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(query="Show me Barack Obama videos")

        # Fallback should extract capitalized words
        assert "Barack" in result.entities or "Obama" in result.entities
        assert result.entity_types == "CONCEPT"


class TestEntityExtractionAgent:
    """Test EntityExtractionAgent core functionality"""

    def test_agent_initialization(self, entity_agent):
        """Test agent initializes with correct configuration"""
        assert entity_agent.agent_name == "entity_extraction_agent"
        assert "entity_extraction" in entity_agent.capabilities

    @pytest.mark.asyncio
    async def test_process_with_entities(self, entity_agent):
        """Test processing query with entities"""
        # Mock DSPy module
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                entities="Barack Obama|PERSON|0.95\nChicago|PLACE|0.9",
                entity_types="PERSON, PLACE",
            )
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="Show me Barack Obama in Chicago", tenant_id=TEST_TENANT_ID)
        )

        assert isinstance(result, EntityExtractionOutput)
        assert result.query == "Show me Barack Obama in Chicago"
        assert result.entity_count == 2
        assert result.has_entities is True
        assert len(result.entities) == 2
        assert result.entities[0].text == "Barack Obama"
        assert result.entities[0].type == "PERSON"
        assert result.entities[1].text == "Chicago"
        assert result.entities[1].type == "PLACE"

    @pytest.mark.asyncio
    async def test_process_no_entities(self, entity_agent):
        """Test processing query with no entities"""
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(entities="", entity_types="")
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="show me some videos", tenant_id=TEST_TENANT_ID)
        )

        assert result.entity_count == 0
        assert result.has_entities is False
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_process_empty_query(self, entity_agent):
        """Test processing empty query"""
        result = await entity_agent._process_impl(EntityExtractionInput(query="", tenant_id=TEST_TENANT_ID))

        assert result.query == ""
        assert result.entity_count == 0
        assert result.has_entities is False

    @pytest.mark.asyncio
    async def test_process_missing_query(self, entity_agent):
        """Test processing with missing query field (defaults to empty string)"""
        # With typed inputs, we provide an empty query as equivalent to missing
        result = await entity_agent._process_impl(EntityExtractionInput(query="", tenant_id=TEST_TENANT_ID))

        assert result.query == ""
        assert result.entity_count == 0

    def test_parse_entities_valid(self, entity_agent):
        """Test parsing valid entity string"""
        entities_str = "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9"
        query = "Barack Obama in Chicago"

        entities = entity_agent._parse_entities(entities_str, query)

        assert len(entities) == 2
        assert entities[0].text == "Barack Obama"
        assert entities[0].type == "PERSON"
        assert entities[0].confidence == 0.95
        assert entities[1].text == "Chicago"
        assert entities[1].type == "PLACE"
        assert entities[1].confidence == 0.9

    def test_parse_entities_no_confidence(self, entity_agent):
        """Test parsing entities without confidence scores"""
        entities_str = "Apple|ORG\nCalifornia|PLACE"
        query = "Apple in California"

        entities = entity_agent._parse_entities(entities_str, query)

        assert len(entities) == 2
        assert entities[0].confidence == 0.7  # Default confidence
        assert entities[1].confidence == 0.7

    def test_parse_entities_empty(self, entity_agent):
        """Test parsing empty entity string"""
        entities = entity_agent._parse_entities("", "test query")
        assert len(entities) == 0

    def test_extract_context(self, entity_agent):
        """Test context extraction"""
        query = "Show me videos about Barack Obama speaking at the conference"
        entity_text = "Barack Obama"

        context = entity_agent._extract_context(entity_text, query)

        assert "Barack Obama" in context
        assert len(context) <= 80  # Max 30 chars before + entity + 30 chars after

    def test_extract_context_entity_not_found(self, entity_agent):
        """Test context extraction when entity not in query"""
        query = "Show me some videos"
        entity_text = "NonExistent"

        context = entity_agent._extract_context(entity_text, query)

        assert len(context) <= 50  # Fallback to first 50 chars

    @pytest.mark.asyncio
    async def test_dominant_types(self, entity_agent):
        """Test dominant entity types calculation"""
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                entities="Obama|PERSON|0.9\nTrump|PERSON|0.9\nWhite House|PLACE|0.8",
                entity_types="PERSON, PLACE",
            )
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="Obama and Trump at White House", tenant_id=TEST_TENANT_ID)
        )

        assert result.dominant_types[0] == "PERSON"  # Most common type
        assert "PLACE" in result.dominant_types

    def test_dspy_to_a2a_output(self, entity_agent):
        """Test conversion to A2A output format"""
        result = EntityExtractionOutput(
            query="test query",
            entities=[
                Entity(
                    text="Obama",
                    type="PERSON",
                    confidence=0.9,
                    context="about Obama speaking",
                ),
                Entity(
                    text="Chicago",
                    type="PLACE",
                    confidence=0.8,
                    context="Obama in Chicago",
                ),
            ],
            entity_count=2,
            has_entities=True,
            dominant_types=["PERSON", "PLACE"],
        )

        a2a_output = entity_agent._dspy_to_a2a_output(result)

        assert a2a_output["status"] == "success"
        assert a2a_output["agent"] == "entity_extraction_agent"
        assert a2a_output["query"] == "test query"
        assert a2a_output["entity_count"] == 2
        assert a2a_output["has_entities"] is True
        assert len(a2a_output["entities"]) == 2
        assert a2a_output["entities"][0]["text"] == "Obama"

    def test_get_agent_skills(self, entity_agent):
        """Test agent skills definition"""
        skills = entity_agent._get_agent_skills()

        assert len(skills) == 1
        assert skills[0]["name"] == "extract_entities"
        assert "query" in skills[0]["input_schema"]
        assert "entities" in skills[0]["output_schema"]
        assert len(skills[0]["examples"]) > 0

    @pytest.mark.asyncio
    async def test_dspy_fallback_sets_path_used(self, entity_agent):
        """DSPy fallback path sets path_used='dspy' in output."""
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                entities="Obama|PERSON|0.9", entity_types="PERSON"
            )
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="Obama speech", tenant_id=TEST_TENANT_ID)
        )

        assert result.path_used == "dspy"
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_dspy_fallback_output_has_new_fields(self, entity_agent):
        """DSPy fallback output includes relationships (empty) and path_used."""
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(entities="", entity_types="")
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="hello", tenant_id=TEST_TENANT_ID)
        )

        assert isinstance(result.relationships, list)
        assert result.path_used == "dspy"


class TestGLiNERFastPath:
    """Tests for GLiNER + SpaCy fast extraction path."""

    @pytest.fixture
    def fast_agent(self):
        """Create agent with mocked GLiNER + SpaCy extractors."""
        with patch("dspy.ChainOfThought"):
            deps = EntityExtractionDeps()
            agent = EntityExtractionAgent(deps=deps, port=8010)

            mock_gliner = MagicMock()
            mock_spacy = MagicMock()
            agent._gliner_extractor = mock_gliner
            agent._spacy_analyzer = mock_spacy
            return agent

    @pytest.mark.asyncio
    async def test_fast_path_extracts_typed_entities(self, fast_agent):
        """GLiNER fast path converts raw dicts to Entity objects."""
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {"text": "Barack Obama", "label": "PERSON", "confidence": 0.95,
             "start_pos": 0, "end_pos": 12},
            {"text": "Chicago", "label": "LOCATION", "confidence": 0.88,
             "start_pos": 16, "end_pos": 23},
        ]
        fast_agent._spacy_analyzer.extract_semantic_relationships.return_value = []

        result = await fast_agent._process_impl(
            EntityExtractionInput(query="Barack Obama in Chicago", tenant_id=TEST_TENANT_ID)
        )

        assert result.path_used == "fast"
        assert result.entity_count == 2
        assert result.entities[0].text == "Barack Obama"
        assert result.entities[0].type == "PERSON"
        assert result.entities[0].confidence == 0.95
        assert result.entities[1].text == "Chicago"
        assert result.entities[1].type == "LOCATION"

    @pytest.mark.asyncio
    async def test_fast_path_extracts_relationships(self, fast_agent):
        """SpaCy produces Relationship objects when 2+ entities found."""
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {"text": "Obama", "label": "PERSON", "confidence": 0.9,
             "start_pos": 0, "end_pos": 5},
            {"text": "White House", "label": "LOCATION", "confidence": 0.85,
             "start_pos": 14, "end_pos": 25},
        ]
        fast_agent._spacy_analyzer.extract_semantic_relationships.return_value = [
            {"subject": "Obama", "relation": "at", "object": "House",
             "confidence": 0.7, "grammatical_pattern": "prep-at"},
        ]

        result = await fast_agent._process_impl(
            EntityExtractionInput(query="Obama at the White House", tenant_id=TEST_TENANT_ID)
        )

        assert result.path_used == "fast"
        assert len(result.relationships) == 1
        assert result.relationships[0].subject == "Obama"
        assert result.relationships[0].relation == "at"
        assert result.relationships[0].object == "House"
        assert result.relationships[0].confidence == 0.7

    @pytest.mark.asyncio
    async def test_fast_path_skips_relationships_for_single_entity(self, fast_agent):
        """SpaCy relationship extraction skipped when fewer than 2 entities."""
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {"text": "Obama", "label": "PERSON", "confidence": 0.9,
             "start_pos": 0, "end_pos": 5},
        ]

        result = await fast_agent._process_impl(
            EntityExtractionInput(query="Obama speech", tenant_id=TEST_TENANT_ID)
        )

        assert result.path_used == "fast"
        assert result.entity_count == 1
        assert result.relationships == []
        fast_agent._spacy_analyzer.extract_semantic_relationships.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_when_gliner_unavailable(self, fast_agent):
        """When GLiNER is None, DSPy fallback is used."""
        fast_agent._gliner_extractor = None
        fast_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                entities="Obama|PERSON|0.9", entity_types="PERSON"
            )
        )

        result = await fast_agent._process_impl(
            EntityExtractionInput(query="Obama speech", tenant_id=TEST_TENANT_ID)
        )

        assert result.path_used == "dspy"
        assert result.entity_count == 1
        assert result.entities[0].text == "Obama"

    @pytest.mark.asyncio
    async def test_fast_path_runtime_failure_falls_back_to_dspy(self):
        """If GLiNER raises at runtime, should fall back to DSPy."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = MagicMock()
        agent._gliner_extractor.extract_entities.side_effect = RuntimeError("GLiNER OOM")
        agent._spacy_analyzer = MagicMock()

        # Mock DSPy fallback
        mock_result = MagicMock()
        mock_result.entities = "Python|TECHNOLOGY|0.8"
        mock_result.entity_types = "TECHNOLOGY"
        agent.dspy_module = MagicMock()
        agent.dspy_module.forward.return_value = mock_result

        input_data = EntityExtractionInput(query="Python programming", tenant_id=TEST_TENANT_ID)
        result = await agent._process_impl(input_data)

        assert result.entity_count >= 1
        assert result.path_used == "dspy"

    @pytest.mark.asyncio
    async def test_a2a_output_includes_relationships(self, fast_agent):
        """_dspy_to_a2a_output includes relationships and path_used."""
        output = EntityExtractionOutput(
            query="test",
            entities=[Entity(text="X", type="PERSON", confidence=0.9)],
            relationships=[
                Relationship(subject="X", relation="at", object="Y", confidence=0.7)
            ],
            entity_count=1,
            has_entities=True,
            dominant_types=["PERSON"],
            path_used="fast",
        )

        a2a = fast_agent._dspy_to_a2a_output(output)

        assert a2a["path_used"] == "fast"
        assert len(a2a["relationships"]) == 1
        assert a2a["relationships"][0]["subject"] == "X"


class TestTelemetrySpanEmission:
    """Tests for entity extraction telemetry span."""

    @pytest.fixture
    def agent_with_telemetry(self):
        """Create agent with mocked telemetry manager."""
        with patch("dspy.ChainOfThought"):
            deps = EntityExtractionDeps()
            agent = EntityExtractionAgent(deps=deps, port=8010)
            agent._gliner_extractor = None
            agent._spacy_analyzer = None

            mock_tm = MagicMock()
            mock_span = MagicMock()
            mock_tm.span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tm.span.return_value.__exit__ = Mock(return_value=False)
            agent.telemetry_manager = mock_tm
            return agent

    @pytest.mark.asyncio
    async def test_span_emitted(self, agent_with_telemetry):
        """Telemetry span emitted with correct attributes."""
        agent = agent_with_telemetry
        agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                entities="Obama|PERSON|0.9\nChicago|PLACE|0.8",
                entity_types="PERSON, PLACE",
            )
        )

        await agent._process_impl(
            EntityExtractionInput(query="Obama in Chicago", tenant_id="acme")
        )

        agent.telemetry_manager.span.assert_called_once()
        call_kwargs = agent.telemetry_manager.span.call_args
        assert call_kwargs[0][0] == "cogniverse.entity_extraction"
        assert call_kwargs[1]["tenant_id"] == "acme"
        attrs = call_kwargs[1]["attributes"]
        assert attrs["entity_extraction.entity_count"] == 2
        assert attrs["entity_extraction.relationship_count"] == 0
        assert attrs["entity_extraction.path_used"] == "dspy"
        assert "Obama" in attrs["entity_extraction.entities"]

    @pytest.mark.asyncio
    async def test_no_telemetry_manager(self):
        """No telemetry_manager -> no error, no span."""
        with patch("dspy.ChainOfThought"):
            deps = EntityExtractionDeps()
            agent = EntityExtractionAgent(deps=deps, port=8010)
            agent._gliner_extractor = None
            agent._spacy_analyzer = None
            agent.telemetry_manager = None
            agent.dspy_module.forward = Mock(
                return_value=dspy.Prediction(entities="", entity_types="")
            )

            result = await agent._process_impl(
                EntityExtractionInput(query="hello", tenant_id=TEST_TENANT_ID)
            )

            assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, agent_with_telemetry):
        """When tenant_id is missing, _process_impl raises rather than silently
        emitting the span under "default". The telemetry hook in AgentBase now
        calls require_tenant_id."""
        agent = agent_with_telemetry
        agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(entities="", entity_types="")
        )

        with pytest.raises(ValueError, match="tenant_id is required"):
            await agent._process_impl(EntityExtractionInput(query="hello"))

    @pytest.mark.asyncio
    async def test_span_query_truncated(self, agent_with_telemetry):
        """Long queries are truncated to 200 chars in span attributes."""
        agent = agent_with_telemetry
        agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(entities="", entity_types="")
        )

        long_query = "x" * 500
        await agent._process_impl(
            EntityExtractionInput(query=long_query, tenant_id=TEST_TENANT_ID)
        )

        call_kwargs = agent.telemetry_manager.span.call_args
        attrs = call_kwargs[1]["attributes"]
        assert len(attrs["entity_extraction.query"]) <= 200


class TestRelationshipModel:
    """Tests for the Relationship Pydantic model."""

    def test_relationship_defaults(self):
        """Relationship has default confidence of 0.5."""
        r = Relationship(subject="A", relation="knows", object="B")
        assert r.confidence == 0.5

    def test_relationship_custom_confidence(self):
        """Relationship accepts custom confidence."""
        r = Relationship(subject="A", relation="at", object="B", confidence=0.9)
        assert r.confidence == 0.9

    def test_relationship_model_dump(self):
        """Relationship serializes correctly."""
        r = Relationship(subject="X", relation="in", object="Y", confidence=0.7)
        d = r.model_dump()
        assert d == {
            "subject": "X",
            "relation": "in",
            "object": "Y",
            "confidence": 0.7,
        }


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


class TestEntityExtractionArtifactLoading:
    @pytest.mark.asyncio
    async def test_loads_dspy_artifact(self, entity_agent):
        """EntityExtractionAgent should load optimized DSPy module state."""
        import json
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()
        fake_state = {"extractor.predict": {"signature": {"fields": []}, "demos": []}}

        with patch("cogniverse_agents.optimizer.artifact_manager.ArtifactManager") as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(return_value=json.dumps(fake_state))

            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            entity_agent.dspy_module = MagicMock()
            entity_agent._load_artifact()

        entity_agent.dspy_module.load_state.assert_called_once_with(fake_state)

    def test_defaults_without_artifact(self, entity_agent):
        """Agent uses default module when no artifact exists."""
        assert hasattr(entity_agent, "dspy_module")
        assert entity_agent.dspy_module is not None

    def test_no_telemetry_skips_loading(self, entity_agent):
        """_load_artifact is a no-op when telemetry_manager is not set."""
        entity_agent.telemetry_manager = None
        entity_agent._load_artifact()

    @pytest.mark.asyncio
    async def test_artifact_load_failure_uses_defaults(self, entity_agent):
        """_load_artifact falls back to defaults when artifact load fails."""
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()

        with patch("cogniverse_agents.optimizer.artifact_manager.ArtifactManager") as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(side_effect=RuntimeError("connection refused"))
            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            entity_agent._load_artifact()
