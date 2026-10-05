"""Topic saliency feeding the served GLiNER entity extraction.

The topics ``extract_topic`` picks from the Big Buck Bunny captions, run
through ``EntityExtractionAgent``'s fast path against the cluster's GLiNER
service. The topic selection itself is pinned without a model in
``tests/synthetic/unit/test_topic_saliency_golden.py``.
"""

from __future__ import annotations

import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_synthetic.topics import TopicSaliency, extract_topic
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
from tests.synthetic.unit.test_topic_saliency_golden import (
    _big_buck_bunny_records,
    _memory_config_manager,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_big_buck_bunny_corpus_pins_each_topics_served_entities(gliner_url):
    records = _big_buck_bunny_records()
    saliency = TopicSaliency.from_records(records)

    zero_record = records[0]
    rich_record = records[20]

    zero_topic = extract_topic(zero_record, saliency=saliency)
    rich_topic = extract_topic(rich_record, saliency=saliency)

    assert zero_topic == "challenging to identify specific colors comprehensively"
    assert rich_topic == "atmospheric conditions such as wildfires causing"

    agent = EntityExtractionAgent(
        deps=EntityExtractionDeps(gliner_inference_url=gliner_url)
    )
    agent.telemetry_manager = RecordingTelemetryManager()
    agent.bind_config_manager(_memory_config_manager())

    zero_result = await agent._process_impl(
        EntityExtractionInput(query=zero_topic, tenant_id="acme")
    )
    # Served at the 0.4 entity threshold every GLiNER path uses, this topic
    # yields three low-scoring CONCEPT spans (0.40-0.50).
    assert zero_result.query == zero_topic
    assert zero_result.entity_count == 3
    assert zero_result.has_entities is True
    assert [
        (entity.text, entity.type, entity.context) for entity in zero_result.entities
    ] == [
        (
            "identify",
            "CONCEPT",
            "challenging to identify specific colors comprehensive",
        ),
        (
            "specific colors",
            "CONCEPT",
            "challenging to identify specific colors comprehensively",
        ),
        (
            "comprehensively",
            "CONCEPT",
            "g to identify specific colors comprehensively",
        ),
    ]
    assert zero_result.relationships == []
    assert zero_result.path_used == "fast"

    rich_result = await agent._process_impl(
        EntityExtractionInput(query=rich_topic, tenant_id="acme")
    )
    assert rich_result.query == rich_topic
    assert rich_result.entity_count == 2
    assert rich_result.has_entities is True
    assert [
        (entity.text, entity.type, entity.context) for entity in rich_result.entities
    ] == [
        (
            "atmospheric conditions",
            "CONCEPT",
            "atmospheric conditions such as wildfires causing",
        ),
        (
            "wildfires",
            "EVENT",
            "tmospheric conditions such as wildfires causing",
        ),
    ]
    assert [
        (relationship.subject, relationship.relation, relationship.object)
        for relationship in rich_result.relationships
    ] == [("atmospheric conditions", "as", "wildfires")]
    assert rich_result.path_used == "fast"
