"""Unit tests for EntityExtractionAgent"""

from unittest.mock import Mock, patch

import dspy
import pytest
from cogniverse_agents.entity_extraction_agent import (
    Entity,
    EntityExtractionAgent,
    EntityExtractionModule,
    EntityExtractionResult,
)


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
    """Create EntityExtractionAgent for testing"""
    with patch("dspy.ChainOfThought"):
        agent = EntityExtractionAgent(tenant_id="test_tenant", port=8010)
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
        assert entity_agent.tenant_id == "test_tenant"
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

        result = await entity_agent._process(
            {"query": "Show me Barack Obama in Chicago"}
        )

        assert isinstance(result, EntityExtractionResult)
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

        result = await entity_agent._process({"query": "show me some videos"})

        assert result.entity_count == 0
        assert result.has_entities is False
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_process_empty_query(self, entity_agent):
        """Test processing empty query"""
        result = await entity_agent._process({"query": ""})

        assert result.query == ""
        assert result.entity_count == 0
        assert result.has_entities is False

    @pytest.mark.asyncio
    async def test_process_missing_query(self, entity_agent):
        """Test processing with missing query field"""
        result = await entity_agent._process({})

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

        result = await entity_agent._process(
            {"query": "Obama and Trump at White House"}
        )

        assert result.dominant_types[0] == "PERSON"  # Most common type
        assert "PLACE" in result.dominant_types

    def test_dspy_to_a2a_output(self, entity_agent):
        """Test conversion to A2A output format"""
        result = EntityExtractionResult(
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
