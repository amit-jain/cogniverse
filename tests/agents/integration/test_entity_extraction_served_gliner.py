"""Entity extraction through the cluster's GLiNER service.

The fast path and the DSPy path of ``EntityExtractionAgent`` against the
served production classifier: what the agent returns when the LM fails, and
that both paths ground relationships the same way. Extraction with no model
at all is pinned by ``tests/agents/unit/test_entity_extraction_agent.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionInput,
    EntityExtractionModule,
    Relationship,
)
from cogniverse_core.common.tenant_utils import TEST_TENANT_ID
from cogniverse_foundation.telemetry.span_contract import (
    ENTITY_EXTRACTION_FALLBACK_ATTRIBUTE,
    ENTITY_EXTRACTION_FALLBACK_ERROR_ATTRIBUTE,
    ENTITY_EXTRACTION_FALLBACK_LM_UNAVAILABLE,
)
from tests.agents.unit.test_entity_extraction_agent import (
    _make_extraction_agent,
    _mention_dicts,
    _messages,
    _RaisingDummyLM,
)
from tests.agents.unit.test_entity_extraction_agent import (
    entity_agent as _entity_agent_fixture,
)

entity_agent = _entity_agent_fixture

pytestmark = pytest.mark.integration


def test_served_gliner_extracts_each_entity_of_a_rich_query(gliner_url):
    from cogniverse_agents.routing.relationship_extraction_tools import (
        GLiNERRelationshipExtractor,
    )

    extractor = GLiNERRelationshipExtractor(inference_url=gliner_url)

    entities = extractor.extract_entities(
        "Apple Inc. develops iPhone using advanced technology"
    )

    assert [
        (entity["text"], entity["label"], entity["start_pos"], entity["end_pos"])
        for entity in entities
    ] == [
        ("Apple Inc.", "ORGANIZATION", 0, 10),
        ("iPhone", "PRODUCT", 20, 26),
        ("advanced technology", "TECHNOLOGY", 33, 52),
    ]
    assert [entity["confidence"] for entity in entities] == pytest.approx(
        [0.94435, 0.98077, 0.90768], rel=1e-4
    )


@pytest.mark.asyncio
async def test_process_falls_back_to_fast_path_when_dspy_raises(
    entity_agent, caplog, gliner_url
):
    """DSPy failure falls through to the real GLiNER + SpaCy path."""
    caplog.set_level(
        logging.WARNING, logger="cogniverse_agents.entity_extraction_agent"
    )
    from cogniverse_agents.routing.relationship_extraction_tools import (
        GLiNERRelationshipExtractor,
        SpaCyDependencyAnalyzer,
    )

    entity_agent.dspy_module = EntityExtractionModule()
    entity_agent._gliner_extractor = GLiNERRelationshipExtractor(
        inference_url=gliner_url
    )
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
    # One LM attempt, not two: EntityExtractionModule binds a JSONAdapter
    # subclass and ChatAdapter.__call__ skips its reformat fallback for
    # those, which is what main.py's LenientJSONAdapter already gave the
    # served agent. The second call only ever happened under pytest's
    # default ChatAdapter.
    assert lm.calls == 1
    # The fall-through says why, so a schema-refusing engine is not read
    # as an LM outage.
    assert _messages(caplog, "cogniverse_agents.entity_extraction_agent") == [
        "DSPy entity extraction failed (lm_unavailable); falling back to "
        "fast path: planned LM failure"
    ]
    ((span,),) = (entity_agent.telemetry_manager.spans,)
    assert span.attributes[ENTITY_EXTRACTION_FALLBACK_ATTRIBUTE] == (
        ENTITY_EXTRACTION_FALLBACK_LM_UNAVAILABLE
    )
    assert span.attributes[ENTITY_EXTRACTION_FALLBACK_ERROR_ATTRIBUTE] == (
        "RuntimeError('planned LM failure')"
    )


@pytest.mark.asyncio
async def test_relationships_match_between_dspy_and_fast_path(gliner_url):
    """The DSPy path uses the same SpaCy grounding as the fast path."""
    from cogniverse_agents.routing.relationship_extraction_tools import (
        GLiNERRelationshipExtractor,
        SpaCyDependencyAnalyzer,
    )

    agent = _make_extraction_agent()
    agent.dspy_module = EntityExtractionModule()
    agent._gliner_extractor = GLiNERRelationshipExtractor(inference_url=gliner_url)
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
                    "entities": _mention_dicts(
                        ("Barack Obama", "PERSON"), ("Chicago", "PLACE")
                    ),
                }
            ],
            adapter=EntityExtractionModule().dspy_adapter,
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
