"""Unit tests for EntityExtractionAgent"""

import asyncio
import json
import logging
from unittest.mock import MagicMock, Mock, patch

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.entity_extraction_agent import (
    ENTITY_TYPES,
    Entity,
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
    EntityExtractionModule,
    EntityExtractionOutput,
    Relationship,
)
from cogniverse_core.common.tenant_utils import TEST_TENANT_ID
from tests.agents.unit._recording_telemetry import (
    FailingTelemetryManager,
    RecordingTelemetryManager,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


EXPECTED_ENTITY_EXTRACTION_SIGNATURE_INSTRUCTIONS = """Extract named entities from the query.

Allowed types: CONCEPT, EVENT, ORGANIZATION, PERSON, PLACE, TECHNOLOGY. Only emit these labels.

Rules:
- text must be a verbatim span copied from the query.
- Use PERSON for role nouns and people such as man, woman, people, biker.
- Use ORGANIZATION for named organizations or teams.
- Use CONCEPT for physical things such as barbell, car, disk, pipes, and knife.
- Use PLACE for settings such as dirt field, kitchen, and pool area.
- Use TECHNOLOGY for camera and screen.
- Use EVENT for crash.
- Actions are never entities.
- "the video" is never an entity.
- Keep each entity on its own line in text|type|confidence format."""


def _make_extraction_agent():
    """Create EntityExtractionAgent with mocked DSPy for use in tests."""
    with patch("dspy.ChainOfThought"):
        deps = EntityExtractionDeps()
        agent = EntityExtractionAgent(deps=deps, port=8010)
        agent.telemetry_manager = RecordingTelemetryManager()
        return agent


def _messages(caplog, logger_name: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == logger_name]


class _CountingExtractor:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result or []
        self.exc = exc
        self.calls = 0

    def extract_entities(self, query: str):
        del query
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


class _CountingDummyLM(DummyLM):
    def __init__(self, answers):
        super().__init__(answers)
        self.calls = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        return super().__call__(prompt=prompt, messages=messages, **kwargs)


class _RaisingDummyLM(DummyLM):
    def __init__(self, message: str):
        super().__init__([{"reasoning": "unused", "entities": ""}])
        self.calls = 0
        self.message = message

    def __call__(self, prompt=None, messages=None, **kwargs):
        del prompt, messages, kwargs
        self.calls += 1
        raise RuntimeError(self.message)


class _FakeToken:
    def __init__(
        self,
        text: str,
        dep_: str,
        pos_: str,
        idx: int,
        i: int,
    ):
        self.text = text
        self.dep_ = dep_
        self.pos_ = pos_
        self.idx = idx
        self.i = i
        self.head = self
        self.children: list["_FakeToken"] = []
        self.doc = None


class _FakeSpan:
    def __init__(self, doc: "_FakeDoc", start: int, end: int):
        self.doc = doc
        self.start = start
        self.end = end
        self.text = " ".join(token.text for token in doc._tokens[start:end])
        self.start_char = doc._tokens[start].idx
        last_token = doc._tokens[end - 1]
        self.end_char = last_token.idx + len(last_token.text)


class _FakeDoc:
    def __init__(self, tokens: list[_FakeToken], noun_chunks: list[tuple[int, int]]):
        self._tokens = tokens
        self._noun_chunks = [_FakeSpan(self, start, end) for start, end in noun_chunks]
        for token in self._tokens:
            token.doc = self

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.step not in (None, 1):
                raise ValueError("FakeDoc only supports contiguous slices")
            start = 0 if item.start is None else item.start
            end = len(self._tokens) if item.stop is None else item.stop
            return _FakeSpan(self, start, end)
        return self._tokens[item]

    @property
    def noun_chunks(self):
        return list(self._noun_chunks)


def _make_caption_doc() -> _FakeDoc:
    specs = [
        ("A", "det", "DET"),
        ("man", "nsubj", "NOUN"),
        ("wearing", "acl", "VERB"),
        ("a", "det", "DET"),
        ("white", "amod", "ADJ"),
        ("tank", "compound", "NOUN"),
        ("top", "dobj", "NOUN"),
        ("stands", "ROOT", "VERB"),
        ("near", "prep", "ADP"),
        ("a", "det", "DET"),
        ("wire", "compound", "NOUN"),
        ("mesh", "compound", "NOUN"),
        ("fence", "pobj", "NOUN"),
        ("at", "prep", "ADP"),
        ("a", "det", "DET"),
        ("sporting", "compound", "NOUN"),
        ("event", "pobj", "NOUN"),
        (".", "punct", "PUNCT"),
    ]
    tokens = []
    offset = 0
    for i, (text, dep_, pos_) in enumerate(specs):
        token = _FakeToken(text=text, dep_=dep_, pos_=pos_, idx=offset, i=i)
        tokens.append(token)
        offset += len(text)
        if i < len(specs) - 1:
            offset += 1

    head_indices = [
        1,
        7,
        1,
        6,
        5,
        6,
        2,
        7,
        7,
        12,
        11,
        12,
        8,
        12,
        16,
        16,
        13,
        7,
    ]
    for token, head_index in zip(tokens, head_indices, strict=True):
        token.head = tokens[head_index]
    for token in tokens:
        if token.head is not token:
            token.head.children.append(token)

    return _FakeDoc(
        tokens,
        noun_chunks=[
            (0, 2),
            (3, 7),
            (9, 13),
            (14, 17),
        ],
    )


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model"""
    lm = Mock()
    lm.return_value = dspy.Prediction(
        entities="Barack Obama|PERSON|0.95\nChicago|PLACE|0.9",
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
        agent.telemetry_manager = RecordingTelemetryManager()
        return agent


class TestEntityExtractionModule:
    """Test DSPy module for entity extraction"""

    def test_signature_instructions_are_grounded_in_entity_types(self):
        module = EntityExtractionModule()
        instructions = module.extractor.predict.signature.instructions

        for entity_type in ENTITY_TYPES:
            assert entity_type in instructions
        assert "verbatim" in instructions
        assert "PERSON" in instructions

    def test_signature_instructions_match_the_current_contract(self):
        module = EntityExtractionModule()

        assert (
            module.extractor.predict.signature.instructions
            == EXPECTED_ENTITY_EXTRACTION_SIGNATURE_INSTRUCTIONS
        )

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

    def test_forward_raises_when_dspy_fails(self):
        """DSPy failures propagate out of the module."""
        module = EntityExtractionModule()
        module.extractor = Mock(side_effect=RuntimeError("DSPy failed"))

        with pytest.raises(RuntimeError, match="DSPy failed"):
            module.forward(query="Barack Obama videos")


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
            )
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(
                query="Show me Barack Obama in Chicago", tenant_id=TEST_TENANT_ID
            )
        )

        assert isinstance(result, EntityExtractionOutput)
        assert result.query == "Show me Barack Obama in Chicago"
        assert result.entity_count == 2
        assert result.has_entities is True
        assert result.entities == [
            Entity(
                text="Barack Obama",
                type="PERSON",
                confidence=0.95,
                context="Show me Barack Obama in Chicago",
            ),
            Entity(
                text="Chicago",
                type="PLACE",
                confidence=0.9,
                context="Show me Barack Obama in Chicago",
            ),
        ]
        assert result.relationships == []
        assert result.path_used == "dspy"

    @pytest.mark.asyncio
    async def test_process_no_entities(self, entity_agent):
        """Test processing query with no entities"""
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(entities="")
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="show me some videos", tenant_id=TEST_TENANT_ID)
        )

        assert result.query == "show me some videos"
        assert result.entity_count == 0
        assert result.has_entities is False
        assert result.entities == []
        assert result.relationships == []
        assert result.dominant_types == []
        assert result.path_used == "dspy"

    @pytest.mark.asyncio
    async def test_process_empty_query(self, entity_agent):
        """Test processing empty query"""
        result = await entity_agent._process_impl(
            EntityExtractionInput(query="", tenant_id=TEST_TENANT_ID)
        )

        assert result.query == ""
        assert result.entity_count == 0
        assert result.has_entities is False
        assert result.dominant_types == []
        assert result.path_used == "dspy"

    @pytest.mark.asyncio
    async def test_process_missing_query(self, entity_agent):
        """Test processing with missing query field (defaults to empty string)"""
        # With typed inputs, we provide an empty query as equivalent to missing
        result = await entity_agent._process_impl(
            EntityExtractionInput(query="", tenant_id=TEST_TENANT_ID)
        )

        assert result.query == ""
        assert result.entity_count == 0
        assert result.dominant_types == []
        assert result.path_used == "dspy"

    @pytest.mark.asyncio
    async def test_process_uses_dspy_primary_and_skips_gliner(self, entity_agent):
        """DSPy succeeds first; GLiNER is not called."""
        entity_agent.dspy_module = EntityExtractionModule()
        gliner = _CountingExtractor()
        entity_agent._gliner_extractor = gliner
        entity_agent._spacy_analyzer = None
        lm = _CountingDummyLM(
            [
                {
                    "reasoning": "extract the exact query entities",
                    "entities": "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9",
                }
            ]
        )

        with dspy.context(lm=lm):
            result = await entity_agent._process_impl(
                EntityExtractionInput(
                    query="Barack Obama in Chicago", tenant_id=TEST_TENANT_ID
                )
            )

        assert result.model_dump() == {
            "query": "Barack Obama in Chicago",
            "entities": [
                {
                    "text": "Barack Obama",
                    "type": "PERSON",
                    "confidence": 0.95,
                    "context": "Barack Obama in Chicago",
                },
                {
                    "text": "Chicago",
                    "type": "PLACE",
                    "confidence": 0.9,
                    "context": "Barack Obama in Chicago",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["PERSON", "PLACE"],
            "path_used": "dspy",
        }
        assert result.relationships == []
        assert result.path_used == "dspy"
        assert gliner.calls == 0
        assert lm.calls == 1

    @pytest.mark.asyncio
    async def test_process_falls_back_to_fast_path_when_dspy_raises(self, entity_agent):
        """DSPy failure falls through to the real GLiNER + SpaCy path."""
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )

        entity_agent.dspy_module = EntityExtractionModule()
        entity_agent._gliner_extractor = GLiNERRelationshipExtractor()
        entity_agent._spacy_analyzer = SpaCyDependencyAnalyzer()
        lm = _RaisingDummyLM("planned LM failure")

        with dspy.context(lm=lm):
            result = await entity_agent._process_impl(
                EntityExtractionInput(
                    query="Barack Obama in Chicago", tenant_id=TEST_TENANT_ID
                )
            )

        assert result.model_dump() == {
            "query": "Barack Obama in Chicago",
            "entities": [
                {
                    "text": "Barack Obama",
                    "type": "PERSON",
                    # GLiNER forward-pass floats vary in the last digits
                    # across BLAS builds; four significant decimals pin the
                    # model's decision without pinning the hardware.
                    "confidence": pytest.approx(0.99168, rel=1e-4),
                    "context": "Barack Obama in Chicago",
                },
                {
                    "text": "Chicago",
                    "type": "PLACE",
                    "confidence": pytest.approx(0.99024, rel=1e-4),
                    "context": "Barack Obama in Chicago",
                },
            ],
            "relationships": [
                {
                    "subject": "Barack Obama",
                    "relation": "in",
                    "object": "Chicago",
                    "confidence": 0.7,
                }
            ],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["PERSON", "PLACE"],
            "path_used": "fast",
        }
        assert result.relationships == [
            Relationship(
                subject="Barack Obama",
                relation="in",
                object="Chicago",
                confidence=0.7,
            )
        ]
        assert result.entity_count == 2
        assert result.dominant_types == ["PERSON", "PLACE"]
        assert result.entities[0].context == "Barack Obama in Chicago"
        assert result.entities[1].context == "Barack Obama in Chicago"
        assert result.relationships[0].subject == "Barack Obama"
        assert result.relationships[0].object == "Chicago"
        assert result.path_used == "fast"
        assert lm.calls == 2

    @pytest.mark.asyncio
    async def test_relationships_match_between_dspy_and_fast_path(self):
        """The DSPy path uses the same SpaCy grounding as the fast path."""
        from cogniverse_agents.routing.relationship_extraction_tools import (
            GLiNERRelationshipExtractor,
            SpaCyDependencyAnalyzer,
        )

        agent = _make_extraction_agent()
        agent.dspy_module = EntityExtractionModule()
        agent._gliner_extractor = GLiNERRelationshipExtractor()
        agent._spacy_analyzer = SpaCyDependencyAnalyzer()
        query = "Barack Obama in Chicago"

        with patch.object(
            agent, "_extract_dspy_path", side_effect=RuntimeError("LM failed")
        ):
            fast_result = await agent._process_impl(
                EntityExtractionInput(query=query, tenant_id=TEST_TENANT_ID)
            )

        with dspy.context(
            lm=DummyLM(
                [
                    {
                        "reasoning": "extract the exact query entities",
                        "entities": "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9",
                    }
                ]
            )
        ):
            dspy_result = await agent._process_impl(
                EntityExtractionInput(query=query, tenant_id=TEST_TENANT_ID)
            )

        assert [
            (rel.subject, rel.relation, rel.object, rel.confidence)
            for rel in dspy_result.relationships
        ] == [
            (rel.subject, rel.relation, rel.object, rel.confidence)
            for rel in fast_result.relationships
        ]
        assert [(entity.text, entity.type) for entity in dspy_result.entities] == [
            (entity.text, entity.type) for entity in fast_result.entities
        ]
        assert dspy_result.path_used == "dspy"
        assert fast_result.path_used == "fast"

    @pytest.mark.asyncio
    async def test_process_raises_when_both_paths_fail(self, entity_agent):
        """The final exception names both the LM failure and the fast failure."""
        entity_agent.dspy_module = EntityExtractionModule()
        entity_agent._gliner_extractor = _CountingExtractor(
            exc=RuntimeError("gliner boom")
        )
        entity_agent._spacy_analyzer = None
        lm = _RaisingDummyLM("dspy boom")

        with (
            dspy.context(lm=lm),
            pytest.raises(
                RuntimeError,
                match=(
                    r"^Entity extraction failed: DSPy path failed with "
                    r"RuntimeError\('dspy boom'\); fast path failed with "
                    r"RuntimeError\('gliner boom'\)$"
                ),
            ),
        ):
            await entity_agent._process_impl(
                EntityExtractionInput(
                    query="Barack Obama in Chicago", tenant_id=TEST_TENANT_ID
                )
            )

        assert lm.calls == 2
        assert entity_agent._gliner_extractor.calls == 1

    def test_parse_entities_valid(self, entity_agent):
        """Test parsing valid entity string"""
        entities_str = "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9"
        query = "Barack Obama in Chicago"

        entities = entity_agent._parse_entities(entities_str, query)

        assert entities == [
            Entity(
                text="Barack Obama",
                type="PERSON",
                confidence=0.95,
                context="Barack Obama in Chicago",
            ),
            Entity(
                text="Chicago",
                type="PLACE",
                confidence=0.9,
                context="Barack Obama in Chicago",
            ),
        ]

    def test_parse_entities_no_confidence(self, entity_agent):
        """Test parsing entities without confidence scores"""
        entities_str = "Apple|ORGANIZATION\nCalifornia|PLACE"
        query = "Apple in California"

        entities = entity_agent._parse_entities(entities_str, query)

        assert entities == [
            Entity(
                text="Apple",
                type="ORGANIZATION",
                confidence=0.7,
                context="Apple in California",
            ),
            Entity(
                text="California",
                type="PLACE",
                confidence=0.7,
                context="Apple in California",
            ),
        ]

    def test_parse_entities_label_and_percent_confidence(self, entity_agent):
        """LM may emit confidence as a label or percent string. parse_confidence
        maps "high"->0.9 and "85%"->0.85 instead of defaulting to 0.7."""
        entities_str = "Obama|PERSON|high\nChicago|PLACE|85%"
        entities = entity_agent._parse_entities(entities_str, "Obama in Chicago")

        assert entities == [
            Entity(
                text="Obama",
                type="PERSON",
                confidence=0.9,
                context="Obama in Chicago",
            ),
            Entity(
                text="Chicago",
                type="PLACE",
                confidence=0.85,
                context="Obama in Chicago",
            ),
        ]

    def test_parse_entities_empty(self, entity_agent):
        """Test parsing empty entity string"""
        entities = entity_agent._parse_entities("", "test query")
        assert entities == []

    def test_parse_entities_drops_invalid_and_duplicate_entities(
        self, entity_agent, caplog
    ):
        """Invalid entities are dropped; duplicates collapse on casefold/type."""
        entities_str = (
            "Barack Obama|PERSON|0.95\nNotInQuery|CONCEPT|0.7\nBARACK OBAMA|PERSON|0.8"
        )
        query = "Barack Obama visited Chicago and spoke at the conference"

        with caplog.at_level(
            logging.WARNING, logger="cogniverse_agents.entity_extraction_agent"
        ):
            entities = entity_agent._parse_entities(entities_str, query)

        assert entities == [
            Entity(
                text="Barack Obama",
                type="PERSON",
                confidence=0.95,
                context=entity_agent._extract_context("Barack Obama", query),
            )
        ]
        assert query[:50] not in {entity.text for entity in entities}
        assert _messages(caplog, "cogniverse_agents.entity_extraction_agent") == [
            "Dropping invalid entity text='NotInQuery' type='CONCEPT' for query "
            "'Barack Obama visited Chicago and spoke at the conference'",
            "Dropped 1 invalid entity candidates for query "
            "'Barack Obama visited Chicago and spoke at the conference'",
            "Dropped 1 duplicate entity candidates for query "
            "'Barack Obama visited Chicago and spoke at the conference'",
        ]

    def test_extract_context(self, entity_agent):
        """Test context extraction"""
        query = "Show me videos about Barack Obama speaking at the conference"
        entity_text = "Barack Obama"

        context = entity_agent._extract_context(entity_text, query)

        assert "Barack Obama" in context
        assert context == query
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
            )
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(
                query="Obama and Trump at White House", tenant_id=TEST_TENANT_ID
            )
        )

        assert result.dominant_types == ["PERSON", "PLACE"]

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
            return_value=dspy.Prediction(entities="Obama|PERSON|0.9")
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="Obama speech", tenant_id=TEST_TENANT_ID)
        )

        assert result.entities == [
            Entity(
                text="Obama",
                type="PERSON",
                confidence=0.9,
                context="Obama speech",
            )
        ]
        assert result.path_used == "dspy"
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_dspy_fallback_output_has_new_fields(self, entity_agent):
        """DSPy fallback output includes relationships (empty) and path_used."""
        entity_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(entities="")
        )

        result = await entity_agent._process_impl(
            EntityExtractionInput(query="hello", tenant_id=TEST_TENANT_ID)
        )

        assert result.entities == []
        assert result.relationships == []
        assert result.entity_count == 0
        assert result.path_used == "dspy"


class TestGLiNERFastPath:
    """Tests for GLiNER + SpaCy fast extraction path."""

    @pytest.fixture
    def fast_agent(self):
        """Create agent with mocked GLiNER + SpaCy extractors."""
        with patch("dspy.ChainOfThought"):
            deps = EntityExtractionDeps()
            agent = EntityExtractionAgent(deps=deps, port=8010)
            agent.telemetry_manager = RecordingTelemetryManager()

            mock_gliner = MagicMock()
            mock_spacy = MagicMock()
            agent._gliner_extractor = mock_gliner
            agent._spacy_analyzer = mock_spacy
            return agent

    @pytest.mark.asyncio
    async def test_fast_path_extracts_typed_entities(self, fast_agent):
        """GLiNER fast path converts raw dicts to Entity objects."""
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {
                "text": "Barack Obama",
                "label": "PERSON",
                "confidence": 0.95,
                "start_pos": 0,
                "end_pos": 12,
            },
            {
                "text": "Chicago",
                "label": "LOCATION",
                "confidence": 0.88,
                "start_pos": 16,
                "end_pos": 23,
            },
        ]
        fast_agent._spacy_analyzer.extract_semantic_relationships.return_value = []

        with patch.object(
            fast_agent, "_extract_dspy_path", side_effect=RuntimeError("LM failed")
        ):
            result = await fast_agent._process_impl(
                EntityExtractionInput(
                    query="Barack Obama in Chicago", tenant_id=TEST_TENANT_ID
                )
            )

        assert result.path_used == "fast"
        assert result.entity_count == 2
        assert result.entities == [
            Entity(
                text="Barack Obama",
                type="PERSON",
                confidence=0.95,
                context="Barack Obama in Chicago",
            ),
            Entity(
                text="Chicago",
                type="PLACE",
                confidence=0.88,
                context="Barack Obama in Chicago",
            ),
        ]
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_fast_path_extracts_relationships(self, fast_agent):
        """Fast-path relationships are grounded to GLiNER entity texts."""
        query = (
            "A man wearing a white tank top stands near a wire mesh fence "
            "at a sporting event."
        )
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {
                "text": "A man",
                "label": "PERSON",
                "confidence": 0.9,
                "start_pos": 0,
                "end_pos": 5,
            },
            {
                "text": "white tank top",
                "label": "OBJECT",
                "confidence": 0.85,
                "start_pos": 16,
                "end_pos": 30,
            },
            {
                "text": "wire mesh fence",
                "label": "LOCATION",
                "confidence": 0.8,
                "start_pos": 45,
                "end_pos": 60,
            },
            {
                "text": "sporting event",
                "label": "EVENT",
                "confidence": 0.78,
                "start_pos": 66,
                "end_pos": 80,
            },
        ]
        fast_agent._spacy_analyzer.extract_semantic_relationships.return_value = [
            {
                "subject": "stands",
                "relation": "near",
                "object": "fence",
                "confidence": 0.7,
                "grammatical_pattern": "prep-near",
            },
            {
                "subject": "fence",
                "relation": "at",
                "object": "event",
                "confidence": 0.7,
                "grammatical_pattern": "prep-at",
            },
        ]
        fast_agent._spacy_analyzer.nlp = Mock(return_value=_make_caption_doc())

        with patch.object(
            fast_agent, "_extract_dspy_path", side_effect=RuntimeError("LM failed")
        ):
            result = await fast_agent._process_impl(
                EntityExtractionInput(query=query, tenant_id=TEST_TENANT_ID)
            )

        assert result.path_used == "fast"
        assert [
            (rel.subject, rel.relation, rel.object, rel.confidence)
            for rel in result.relationships
        ] == [
            ("A man", "near", "wire mesh fence", 0.7),
            ("wire mesh fence", "at", "sporting event", 0.7),
        ]

    @pytest.mark.asyncio
    async def test_fast_path_relationship_endpoints_are_entity_texts(self, fast_agent):
        """Every emitted relationship endpoint must be one of the entity texts."""
        query = (
            "A man wearing a white tank top stands near a wire mesh fence "
            "at a sporting event."
        )
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {
                "text": "A man",
                "label": "PERSON",
                "confidence": 0.9,
                "start_pos": 0,
                "end_pos": 5,
            },
            {
                "text": "white tank top",
                "label": "OBJECT",
                "confidence": 0.85,
                "start_pos": 16,
                "end_pos": 30,
            },
            {
                "text": "wire mesh fence",
                "label": "LOCATION",
                "confidence": 0.8,
                "start_pos": 45,
                "end_pos": 60,
            },
            {
                "text": "sporting event",
                "label": "EVENT",
                "confidence": 0.78,
                "start_pos": 66,
                "end_pos": 80,
            },
        ]
        fast_agent._spacy_analyzer.extract_semantic_relationships.return_value = [
            {
                "subject": "stands",
                "relation": "near",
                "object": "fence",
                "confidence": 0.7,
                "grammatical_pattern": "prep-near",
            },
            {
                "subject": "fence",
                "relation": "at",
                "object": "event",
                "confidence": 0.7,
                "grammatical_pattern": "prep-at",
            },
        ]
        fast_agent._spacy_analyzer.nlp = Mock(return_value=_make_caption_doc())

        with patch.object(
            fast_agent, "_extract_dspy_path", side_effect=RuntimeError("LM failed")
        ):
            result = await fast_agent._process_impl(
                EntityExtractionInput(query=query, tenant_id=TEST_TENANT_ID)
            )

        entity_texts = {entity.text for entity in result.entities}
        assert entity_texts == {
            "A man",
            "white tank top",
            "wire mesh fence",
            "sporting event",
        }
        assert result.relationships
        for relationship in result.relationships:
            assert relationship.subject in entity_texts
            assert relationship.object in entity_texts

    @pytest.mark.asyncio
    async def test_fast_path_skips_relationships_for_single_entity(self, fast_agent):
        """SpaCy relationship extraction skipped when fewer than 2 entities."""
        fast_agent._gliner_extractor.extract_entities.return_value = [
            {
                "text": "Obama",
                "label": "PERSON",
                "confidence": 0.9,
                "start_pos": 0,
                "end_pos": 5,
            },
        ]

        with patch.object(
            fast_agent, "_extract_dspy_path", side_effect=RuntimeError("LM failed")
        ):
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
            return_value=dspy.Prediction(entities="Obama|PERSON|0.9")
        )

        result = await fast_agent._process_impl(
            EntityExtractionInput(query="Obama speech", tenant_id=TEST_TENANT_ID)
        )

        assert result.path_used == "dspy"
        assert result.entity_count == 1
        assert result.entities[0].text == "Obama"

    @pytest.mark.asyncio
    async def test_dspy_primary_wins_over_gliner_failure(self):
        """DSPy primary wins even if GLiNER would fail."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = MagicMock()
        agent._gliner_extractor.extract_entities.side_effect = RuntimeError(
            "GLiNER OOM"
        )
        agent._spacy_analyzer = MagicMock()

        # Mock DSPy fallback
        mock_result = MagicMock()
        mock_result.entities = "Python|TECHNOLOGY|0.8"
        # call_dspy invokes module(**kwargs) (__call__), not .forward —
        # forward bypasses DSPy's instrumentation.
        agent.dspy_module = MagicMock(return_value=mock_result)

        input_data = EntityExtractionInput(
            query="Python programming", tenant_id=TEST_TENANT_ID
        )
        result = await agent._process_impl(input_data)

        assert result.entities == [
            Entity(
                text="Python",
                type="TECHNOLOGY",
                confidence=0.8,
                context="Python programming",
            )
        ]
        assert result.path_used == "dspy"
        assert agent._gliner_extractor.extract_entities.call_count == 0

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

            agent.telemetry_manager = RecordingTelemetryManager()
            return agent

    @pytest.mark.asyncio
    async def test_span_emitted(self, agent_with_telemetry):
        """Telemetry span emitted with correct attributes."""
        agent = agent_with_telemetry
        agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                entities="Obama|PERSON|0.9\nChicago|PLACE|0.8",
            )
        )

        await agent._process_impl(
            EntityExtractionInput(query="Obama in Chicago", tenant_id="acme")
        )

        import json

        assert len(agent.telemetry_manager.calls) == 1
        assert agent.telemetry_manager.calls == [
            {"name": "cogniverse.entity_extraction", "tenant_id": "acme:acme"}
        ]
        recorded = agent.telemetry_manager.spans[0].attributes
        assert recorded["operation"] == "entity_extraction"
        assert recorded["input.value"] == "Obama in Chicago"
        output = json.loads(recorded["output.value"])
        assert output == {
            "entities": [
                {
                    "text": "Obama",
                    "type": "PERSON",
                    "confidence": 0.9,
                    "context": "Obama in Chicago",
                },
                {
                    "text": "Chicago",
                    "type": "PLACE",
                    "confidence": 0.8,
                    "context": "Obama in Chicago",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "relationship_count": 0,
            "path_used": "dspy",
        }

    @pytest.mark.expects_telemetry_loss_warning
    def test_emit_extraction_span_warns_without_telemetry_manager(self, caplog):
        """No manager: the request continues; the loss is a WARNING, never silent."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = None
        agent._spacy_analyzer = None
        agent.telemetry_manager = None

        with caplog.at_level(
            logging.WARNING, logger="cogniverse_agents.entity_extraction_agent"
        ):
            asyncio.run(
                agent._emit_extraction_span(
                    tenant_id="acme",
                    query="hello",
                    entities=[],
                    relationships=[],
                    path_used="dspy",
                )
            )
        assert _messages(caplog, "cogniverse_agents.entity_extraction_agent") == [
            "EntityExtractionAgent has no telemetry_manager; entity_extraction span "
            "not emitted (tenant=acme)"
        ]

    @pytest.mark.expects_telemetry_loss_warning
    def test_emit_extraction_span_enqueue_failure_warns_and_does_not_raise(
        self, caplog
    ):
        """A telemetry enqueue failure never fails the request; it is a WARNING."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = None
        agent._spacy_analyzer = None
        telemetry = FailingTelemetryManager(RuntimeError("telemetry boom"))
        agent.telemetry_manager = telemetry

        with caplog.at_level(
            logging.WARNING, logger="cogniverse_agents.entity_extraction_agent"
        ):
            asyncio.run(
                agent._emit_extraction_span(
                    tenant_id="acme",
                    query="hello",
                    entities=[],
                    relationships=[],
                    path_used="dspy",
                )
            )

        assert telemetry.calls == [
            {"name": "cogniverse.entity_extraction", "tenant_id": "acme"}
        ]
        assert _messages(caplog, "cogniverse_agents.entity_extraction_agent") == [
            "Failed to emit entity_extraction telemetry: tenant=acme error=telemetry boom"
        ]

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, agent_with_telemetry):
        """When tenant_id is missing, _process_impl raises rather than silently
        emitting the span under "default". The telemetry hook in AgentBase now
        calls require_tenant_id."""
        agent = agent_with_telemetry
        agent.dspy_module.forward = Mock(return_value=dspy.Prediction(entities=""))

        with pytest.raises(ValueError, match="tenant_id is required"):
            await agent._process_impl(EntityExtractionInput(query="hello"))
        assert agent.telemetry_manager.calls == []

    @pytest.mark.asyncio
    async def test_span_records_full_query(self, agent_with_telemetry):
        """The full query is recorded on input.value (no truncation)."""
        agent = agent_with_telemetry
        agent.dspy_module.forward = Mock(return_value=dspy.Prediction(entities=""))

        long_query = "x" * 500
        await agent._process_impl(
            EntityExtractionInput(query=long_query, tenant_id=TEST_TENANT_ID)
        )

        recorded = agent.telemetry_manager.spans[0].attributes
        assert recorded["input.value"] == long_query


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
    async def test_loads_matching_dspy_artifact(self, entity_agent):
        """EntityExtractionAgent should load optimized DSPy module state."""
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()
        artifact_state = json.loads(json.dumps(EntityExtractionModule().dump_state()))
        artifact_state["extractor.predict"]["demos"] = [
            {
                "query": "Barack Obama in Chicago",
                "entities": "Barack Obama|PERSON|0.95\nChicago|PLACE|0.9",
            }
        ]

        with patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager"
        ) as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(return_value=json.dumps(artifact_state))

            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            entity_agent.dspy_module = EntityExtractionModule()
            entity_agent._load_artifact()

        loaded_state = entity_agent.dspy_module.dump_state()["extractor.predict"]
        assert (
            loaded_state["signature"]
            == artifact_state["extractor.predict"]["signature"]
        )
        assert loaded_state["demos"] == artifact_state["extractor.predict"]["demos"]
        assert entity_agent.artifact_load_status == "loaded"

    @pytest.mark.asyncio
    async def test_signature_instructions_mismatch_skips_loading(
        self, entity_agent, caplog
    ):
        """A signature instruction drift must leave the live prompt untouched."""
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()
        artifact_state = json.loads(json.dumps(EntityExtractionModule().dump_state()))
        artifact_state["extractor.predict"]["signature"]["instructions"] = (
            "Extract named entities only from the saved artifact."
        )
        artifact_state["extractor.predict"]["demos"] = [
            {"query": "sentinel", "entities": "sentinel|CONCEPT|1.0"}
        ]

        with patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager"
        ) as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(return_value=json.dumps(artifact_state))
            mock_am.active_blob_version = AsyncMock(return_value=7)

            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            entity_agent.dspy_module = EntityExtractionModule()
            with caplog.at_level(
                logging.WARNING,
                logger="cogniverse_agents.optimizer.artifact_manager",
            ):
                entity_agent._load_artifact()

        assert entity_agent.artifact_load_status == "signature_mismatch"
        assert (
            entity_agent.dspy_module.dump_state()["extractor.predict"]["signature"][
                "instructions"
            ]
            == EntityExtractionModule().extractor.predict.signature.instructions
        )
        assert entity_agent.dspy_module.dump_state()["extractor.predict"]["demos"] == []
        assert any(
            "entity_extraction" in rec.getMessage()
            and "v7" in rec.getMessage()
            and "extractor.predict" in rec.getMessage()
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_field_description_mismatch_skips_loading(self, entity_agent, caplog):
        """A field description drift must leave the live prompt untouched."""
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()
        artifact_state = json.loads(json.dumps(EntityExtractionModule().dump_state()))
        artifact_state["extractor.predict"]["signature"]["fields"][0]["description"] = (
            "User query for artifact compatibility testing"
        )
        artifact_state["extractor.predict"]["demos"] = [
            {"query": "sentinel", "entities": "sentinel|CONCEPT|1.0"}
        ]

        with patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager"
        ) as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(return_value=json.dumps(artifact_state))
            mock_am.active_blob_version = AsyncMock(return_value=8)

            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            entity_agent.dspy_module = EntityExtractionModule()
            with caplog.at_level(
                logging.WARNING,
                logger="cogniverse_agents.optimizer.artifact_manager",
            ):
                entity_agent._load_artifact()

        assert entity_agent.artifact_load_status == "signature_mismatch"
        assert (
            entity_agent.dspy_module.dump_state()["extractor.predict"]["signature"][
                "fields"
            ]
            == EntityExtractionModule().dump_state()["extractor.predict"]["signature"][
                "fields"
            ]
        )
        assert entity_agent.dspy_module.dump_state()["extractor.predict"]["demos"] == []
        assert any(
            "entity_extraction" in rec.getMessage()
            and "v8" in rec.getMessage()
            and "extractor.predict" in rec.getMessage()
            for rec in caplog.records
        )

    def test_defaults_without_artifact(self, entity_agent):
        """Agent uses default module when no artifact exists."""
        assert hasattr(entity_agent, "dspy_module")
        assert entity_agent.dspy_module is not None

    def test_no_telemetry_skips_loading(self, entity_agent):
        """_load_artifact is a no-op when telemetry_manager is not set."""
        entity_agent.telemetry_manager = None
        entity_agent._load_artifact()
        assert entity_agent.artifact_load_status == "no_telemetry"

    @pytest.mark.asyncio
    async def test_absent_artifact_records_no_artifact(self, entity_agent):
        """An absent blob is 'tenant never optimized' — distinct from an
        outage, and serves defaults without noise."""
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()

        with patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager"
        ) as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(return_value=None)
            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            entity_agent._load_artifact()

        assert entity_agent.artifact_load_status == "no_artifact"

    @pytest.mark.asyncio
    async def test_artifact_load_failure_surfaces_error_status(
        self, entity_agent, caplog
    ):
        """An artifact-store OUTAGE must not read as 'never optimized': the
        agent keeps serving on defaults but records status 'error' and logs
        at WARNING instead of swallowing the failure at DEBUG."""
        import logging
        from unittest.mock import AsyncMock

        mock_tm = MagicMock()
        mock_tm.get_provider.return_value = MagicMock()

        with patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager"
        ) as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_blob = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )
            entity_agent.telemetry_manager = mock_tm
            entity_agent._artifact_tenant_id = "test:unit"
            with caplog.at_level(logging.WARNING):
                entity_agent._load_artifact()

        assert entity_agent.artifact_load_status == "error"
        assert (
            "EntityExtractionAgent artifact load failed; using defaults" in caplog.text
        )


@pytest.mark.asyncio
async def test_gliner_fast_path_offloaded_from_event_loop():
    """GLiNER inference + spaCy is sync CPU-heavy work; _process_impl must run it
    off the loop so a concurrent request isn't stalled for the extraction."""
    import asyncio
    import time

    agent = _make_extraction_agent()
    agent._gliner_extractor = object()  # non-None: take the fast path

    def _blocking_fast_path(query):
        time.sleep(0.3)
        return ([Entity(text="Obama", type="PERSON", confidence=0.9)], [], "gliner")

    agent._extract_fast_path = _blocking_fast_path

    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    with patch.object(
        agent, "_extract_dspy_path", side_effect=RuntimeError("LM failed")
    ):
        await agent._process_impl(
            EntityExtractionInput(query="tell me about Obama", tenant_id=TEST_TENANT_ID)
        )
    stop.set()
    await t

    assert ticks >= 10, f"only {ticks} ticks — GLiNER ran on the event loop"
