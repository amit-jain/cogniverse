"""Routing query generation correctness under edge and concurrent inputs."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps, GatewayInput
from cogniverse_core.approval.training_schema import _validate_entities
from cogniverse_foundation.config.unified_config import (
    DSPyModuleConfig,
    OptimizerGenerationConfig,
)
from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)
from cogniverse_synthetic.dspy_modules import (
    EntityQueryValidationError,
    ValidatedEntityQueryGenerator,
)
from cogniverse_synthetic.generators.base import GenerationTracker
from cogniverse_synthetic.generators.entity_extraction import (
    EntityExtractionGenerator,
)
from cogniverse_synthetic.generators.routing import (
    DuplicateLabelFilter,
    RoutingGenerator,
)
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager

pytestmark = pytest.mark.unit


async def _extract_entities(text: str, tenant_id: str):
    entities = []
    for entity_text, entity_type in (
        ("Marie Curie", "PERSON"),
        ("TensorFlow", "TECHNOLOGY"),
        ("PyTorch", "TECHNOLOGY"),
    ):
        if entity_text in text:
            entities.append({"text": entity_text, "type": entity_type})
    return {"query": text, "entities": entities, "relationships": []}


async def _route_query(query: str, tenant_id: str):
    return {
        "query": query,
        "routed_to": "video_search_agent",
        "confidence": 0.73,
    }


def _routing_generator() -> RoutingGenerator:
    return RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=OptimizerGenerationConfig(
            optimizer_type="routing",
            dspy_modules={
                "query_generator": DSPyModuleConfig(
                    signature_class=(
                        "cogniverse_synthetic.dspy_signatures.GenerateEntityQuery"
                    ),
                    module_type="Predict",
                    metadata={"max_retries": 3},
                )
            },
        ),
    )


def _second_routing_sample() -> dict:
    """A second sample for saliency; different content preserves primary topic distinctiveness."""
    return {
        "description": "PyTorch deep learning and neural network training tutorial",
        "content_type": "video",
    }


def test_enhancement_annotates_only_complete_entity_tokens() -> None:
    generator = _routing_generator()
    entity = [{"text": "Go", "type": "TECHNOLOGY"}]

    assert generator._enhance_query("Google Cloud tutorial", entity) == (
        "Google Cloud tutorial"
    )
    assert generator._enhance_query("learn Go today", entity) == (
        "learn Go(TECHNOLOGY) today"
    )


def test_enhancement_annotates_every_complete_entity_occurrence() -> None:
    generator = _routing_generator()
    entity = [{"text": "computer", "type": "CONCEPT"}]

    assert (
        generator._enhance_query(
            "computer vision using a computer",
            entity,
        )
        == "computer(CONCEPT) vision using a computer(CONCEPT)"
    )


def test_enhancement_uses_longest_non_overlapping_entity_match() -> None:
    generator = _routing_generator()
    entities = [
        {"text": "Marie Curie", "type": "PERSON"},
        {"text": "Curie", "type": "SURNAME"},
    ]

    assert generator._enhance_query(
        "Marie Curie discovered radium with Curie notebooks",
        entities,
    ) == ("Marie Curie(PERSON) discovered radium with Curie(SURNAME) notebooks")


def test_enhancement_never_reannotates_inserted_entity_types() -> None:
    generator = _routing_generator()
    entities = [
        {"text": "Python", "type": "TECHNOLOGY"},
        {"text": "technology", "type": "CATEGORY"},
    ]

    assert generator._enhance_query("Python technology", entities) == (
        "Python(TECHNOLOGY) technology(CATEGORY)"
    )


@pytest.mark.asyncio
async def test_generate_extracts_salient_topic_before_labeling_and_query_generation() -> (
    None
):
    """Topic is extracted via saliency (most distinctive 6-word span)."""
    long_caption = {
        "segment_description": (
            "A man wearing a white tank top stands near a wire mesh fence at a "
            "sporting event. Spectators line the field while the athlete prepares "
            "to throw the disc. Rolling hills and a sunny park are visible in the "
            "background."
        )
    }
    # Saliency extracts the most distinctive 6-word span
    expected_topic = "man wearing a white tank top"
    entity_calls = []

    async def label_entities(*, sampled_content, target_count, tenant_id):
        # Filter to just the primary content topic for assertion
        primary_topics = [
            c["topic"]
            for c in sampled_content
            if "white tank top" in c.get("topic", "")
        ]
        entity_calls.append(
            {
                "primary_topic": primary_topics[0] if primary_topics else None,
                "target_count": target_count,
                "tenant_id": tenant_id,
            }
        )
        return [
            SimpleNamespace(
                entities=[{"text": "man", "type": "PERSON"}],
                relationships=[],
            )
        ]

    class _CapturingQueryGenerator:
        max_retries = 3

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                query=f"find {kwargs['topics']}",
                reasoning="Used the supplied entity.",
                _retry_count=0,
                _max_retries=self.max_retries,
            )

    generator = _routing_generator()
    generator.entity_labeler.generate = label_entities
    query_generator = _CapturingQueryGenerator()
    generator.query_generator = query_generator

    examples = await generator.generate(
        [long_caption, _second_routing_sample()],
        target_count=1,
        tenant_id="tenant-a",
    )

    assert entity_calls[0]["primary_topic"] == expected_topic
    assert entity_calls[0]["target_count"] == 1
    assert entity_calls[0]["tenant_id"] == "tenant-a"
    assert query_generator.calls[0]["topics"] == expected_topic
    assert query_generator.calls[0]["entities"] == ["man"]
    assert query_generator.calls[0]["entity_types"] == ["PERSON"]
    assert examples[0].query == f"find {expected_topic}"
    assert examples[0].entities == [{"text": "man", "type": "PERSON"}]


@pytest.mark.asyncio
async def test_generate_dedupes_casefold_entity_texts_before_storage() -> None:
    source = {"segment_description": "sporting event"}
    entity_calls = []

    async def label_entities(*, sampled_content, target_count, tenant_id):
        # Record primary topic for assertion
        primary = [c for c in sampled_content if "sporting" in c.get("topic", "")]
        entity_calls.append(
            {
                "primary_topic": primary[0]["topic"] if primary else None,
                "target_count": target_count,
                "tenant_id": tenant_id,
            }
        )
        return [
            SimpleNamespace(
                entities=[
                    {"text": "man", "type": "PERSON"},
                    {"text": "man", "type": "PERSON"},
                    {"text": "spectators", "type": "PERSON"},
                    {"text": "Spectators", "type": "PERSON"},
                ],
                relationships=[],
            )
        ]

    class _CapturingQueryGenerator:
        max_retries = 3

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                query="find man and spectators at sporting event",
                reasoning="Used the deduped entity set.",
                _retry_count=0,
                _max_retries=self.max_retries,
            )

    generator = _routing_generator()
    generator.entity_labeler.generate = label_entities
    query_generator = _CapturingQueryGenerator()
    generator.query_generator = query_generator

    examples = await generator.generate(
        [source, _second_routing_sample()],
        target_count=1,
        tenant_id="tenant-a",
    )

    assert entity_calls[0]["primary_topic"] == "sporting event"
    assert entity_calls[0]["target_count"] == 1
    assert entity_calls[0]["tenant_id"] == "tenant-a"
    assert query_generator.calls == [
        {
            "topics": "sporting event",
            "entities": ["man", "spectators"],
            "entity_types": ["PERSON", "PERSON"],
        }
    ]
    assert examples[0].query == "find man and spectators at sporting event"
    assert examples[0].entities == [
        {"text": "man", "type": "PERSON"},
        {"text": "spectators", "type": "PERSON"},
    ]


def test_validation_does_not_accept_entity_as_substring_of_another_word() -> None:
    generator = ValidatedEntityQueryGenerator(max_retries=1)
    calls = 0

    def generate(**kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            query="Google Cloud tutorial",
            reasoning="The generated query is unrelated.",
        )

    generator.generate = generate

    with pytest.raises(
        EntityQueryValidationError,
        match=("Entity query validation failed after 1 attempt for entities: Go"),
    ):
        generator.forward(
            topics="cloud",
            entities=["Go"],
            entity_types=["TECHNOLOGY"],
        )

    assert calls == 1


def test_validation_retries_until_every_complete_entity_is_present() -> None:
    generator = ValidatedEntityQueryGenerator(max_retries=2)
    outputs = iter(
        [
            SimpleNamespace(
                query="find practical lessons for PyTorch",
                reasoning="Pluralized one entity.",
            ),
            SimpleNamespace(
                query="find a practical lesson for PyTorch",
                reasoning="Used both exact entities.",
            ),
        ]
    )

    generator.generate = lambda **kwargs: next(outputs)

    result = generator.forward(
        topics="machine learning",
        entities=["practical lesson", "PyTorch"],
        entity_types=["EVENT", "TECHNOLOGY"],
    )

    assert result.query == "find a practical lesson for PyTorch"
    assert result.reasoning == "Used both exact entities."
    assert result._retry_count == 1
    assert result._max_retries == 2


def test_validation_rejects_query_that_omits_one_multiword_entity() -> None:
    generator = ValidatedEntityQueryGenerator(max_retries=1)
    generator.generate = lambda **kwargs: SimpleNamespace(
        query="find Curie notebooks about radium",
        reasoning="Used only part of the person name.",
    )

    with pytest.raises(
        EntityQueryValidationError,
        match=(
            "Entity query validation failed after 1 attempt for entities: "
            "Marie Curie, radium"
        ),
    ):
        generator.forward(
            topics="radioactivity",
            entities=["Marie Curie", "radium"],
            entity_types=["PERSON", "MATERIAL"],
        )


def test_validation_preserves_punctuated_entity_identity_at_dspy_boundary() -> None:
    generator = ValidatedEntityQueryGenerator(max_retries=1)
    observed_inputs = []

    def generate(**kwargs):
        observed_inputs.append(kwargs)
        return SimpleNamespace(
            query="show Washington, D.C. reports from Meta AI",
            reasoning="Used both complete entities.",
        )

    generator.generate = generate

    result = generator.forward(
        topics="policy",
        entities=["Washington, D.C.", "Meta AI"],
        entity_types=["PLACE", "ORGANIZATION"],
    )

    assert result.query == "show Washington, D.C. reports from Meta AI"
    assert observed_inputs == [
        {
            "topics": "policy",
            "entities": '["Washington, D.C.", "Meta AI"]',
            "entity_types": '["PLACE", "ORGANIZATION"]',
        }
    ]


def test_validation_rejects_separated_fragments_of_punctuated_entity() -> None:
    generator = ValidatedEntityQueryGenerator(max_retries=1)
    generator.generate = lambda **kwargs: SimpleNamespace(
        query="compare Washington reports with D.C. reports",
        reasoning="Separated one entity into two fragments.",
    )

    with pytest.raises(
        EntityQueryValidationError,
        match=r"Entity query validation failed after 1 attempt for entities:.*Washington",
    ):
        generator.forward(
            topics="policy",
            entities=["Washington, D.C."],
            entity_types=["PLACE"],
        )


def test_validation_allows_case_change_for_complete_query_span() -> None:
    generator = ValidatedEntityQueryGenerator(max_retries=1)
    generator.generate = lambda **kwargs: SimpleNamespace(
        query="show washington, d.c. policy reports",
        reasoning="Used the complete place name with query casing.",
    )

    result = generator.forward(
        topics="policy",
        entities=["Washington, D.C."],
        entity_types=["PLACE"],
    )

    assert result.query == "show washington, d.c. policy reports"


def test_routing_generator_requires_production_entity_extractor() -> None:
    with pytest.raises(ValueError, match="^entity_extractor is required$"):
        RoutingGenerator(
            entity_extractor=None,
            optimizer_config=_routing_generator().optimizer_config,
        )


def test_query_generator_uses_configured_module_and_retry_limit() -> None:
    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=OptimizerGenerationConfig(
            optimizer_type="routing",
            dspy_modules={
                "query_generator": DSPyModuleConfig(
                    signature_class=(
                        "cogniverse_synthetic.dspy_signatures.GenerateEntityQuery"
                    ),
                    module_type="Predict",
                    metadata={"max_retries": 7},
                )
            },
        ),
    )

    query_generator = generator._get_query_generator()

    assert query_generator.max_retries == 7
    assert query_generator.generate.__class__.__name__ == "Predict"


async def test_generation_rejects_content_without_canonical_topic() -> None:
    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=_routing_generator().optimizer_config,
    )

    # With saliency, need 2+ valid records; then one invalid triggers per-record check
    with pytest.raises(
        ValueError, match="^sampled routing content requires a non-empty topic$"
    ):
        await generator.generate(
            sampled_content=[
                # Invalid record first - selected in attempt 0
                {
                    "schema_name": "document_text",
                    "embedding_type": "single_vector",
                },
                # Valid records needed for saliency (>= 2 with topic text)
                {"topic": "TensorFlow tutorial video"},
                {"topic": "PyTorch deep learning guide"},
            ],
            target_count=1,
            tenant_id="acme:routing",
        )


async def test_query_generation_rejects_missing_source_topic() -> None:
    # With saliency, need 2+ valid records; then one invalid triggers per-record check
    with pytest.raises(
        ValueError,
        match="^sampled routing content requires a non-empty topic$",
    ):
        await RoutingGenerator(
            entity_extractor=_extract_entities,
            routing_decider=_route_query,
            optimizer_config=_routing_generator().optimizer_config,
        ).generate(
            sampled_content=[
                # Invalid record first - selected in attempt 0
                {
                    "schema_name": "document_text",
                    "embedding_type": "single_vector",
                },
                # Valid records needed for saliency (>= 2 with topic text)
                {"topic": "TensorFlow tutorial video"},
                {"topic": "PyTorch deep learning guide"},
            ],
            target_count=1,
            tenant_id="acme:routing",
        )


async def test_generation_uses_canonical_topic_string_for_query_generation() -> None:
    class _SourceQueryGenerator:
        max_retries = 3

        def __init__(self) -> None:
            self.inputs = []

        def __call__(self, **kwargs):
            self.inputs.append(kwargs)
            return SimpleNamespace(
                query=f"find {kwargs['topics']}",
                reasoning=f"Used {kwargs['topics']} from this source item.",
                _retry_count=0,
                _max_retries=3,
            )

    captured_entity_labeler_inputs = []
    captured_entity_queries = []

    async def extract_entities(text: str, tenant_id: str):
        captured_entity_queries.append((text, tenant_id))
        return await _extract_entities(text, tenant_id)

    generator = RoutingGenerator(
        entity_extractor=extract_entities,
        routing_decider=_route_query,
        optimizer_config=_routing_generator().optimizer_config,
    )
    query_generator = _SourceQueryGenerator()
    generator.query_generator = query_generator
    real_entity_labeler_generate = generator.entity_labeler.generate

    async def capture_entity_labeler_generate(
        *, sampled_content, target_count, **kwargs
    ):
        captured_entity_labeler_inputs.append((sampled_content, target_count, kwargs))
        return await real_entity_labeler_generate(
            sampled_content=sampled_content,
            target_count=target_count,
            **kwargs,
        )

    generator.entity_labeler.generate = capture_entity_labeler_generate

    examples = await generator.generate(
        sampled_content=[
            {
                "description": "TensorFlow deployment guide",
                "audio_transcript": "PyTorch and TensorFlow clip",
                "schema_name": "video_segments",
                "embedding_type": "multi_vector",
            },
            _second_routing_sample(),
        ],
        target_count=1,
        tenant_id="acme:routing",
    )

    # Saliency extracts the most distinctive span
    assert (
        captured_entity_labeler_inputs[0][0][0]["topic"]
        == "TensorFlow deployment guide"
    )
    assert captured_entity_labeler_inputs[0][1] == 1
    assert captured_entity_queries[0] == ("TensorFlow deployment guide", "acme:routing")
    assert query_generator.inputs[0]["topics"] == "TensorFlow deployment guide"
    assert query_generator.inputs[0]["entities"] == ["TensorFlow"]
    assert query_generator.inputs[0]["entity_types"] == ["TECHNOLOGY"]
    assert examples[0].query == "find TensorFlow deployment guide"
    assert examples[0].enhanced_query == (
        "find TensorFlow(TECHNOLOGY) deployment guide"
    )


async def test_generation_rejects_content_without_extracted_entities() -> None:
    async def extract_entities(text: str, tenant_id: str):
        return {"query": text, "entities": [], "relationships": []}

    class _UnexpectedQueryGenerator:
        max_retries = 3

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            raise AssertionError("query LM must not receive an empty entity set")

    generator = RoutingGenerator(
        entity_extractor=extract_entities,
        routing_decider=_route_query,
        optimizer_config=_routing_generator().optimizer_config,
    )
    query_generator = _UnexpectedQueryGenerator()
    generator.query_generator = query_generator

    with pytest.raises(ValueError) as error:
        await generator.generate(
            sampled_content=[
                {
                    "title": "Radioactivity research",
                    "description": "A document with no named entities.",
                    "schema_name": "document",
                },
                _second_routing_sample(),
            ],
            target_count=1,
            tenant_id="acme:routing",
        )

    # Budget = 5 draws from 2 items
    assert str(error.value) == (
        "RoutingGenerator generated 0 unique grounded examples but target_count=1; "
        "source_context=5 routing candidate draws from 2 sampled content items"
    )
    assert str(error.value.__cause__) == (
        "EntityExtractionGenerator generated 0 unique grounded examples but "
        "target_count=1; source_context=1 unique source texts, 1 without entities"
    )
    assert query_generator.calls == 0


async def test_generation_keeps_source_query_entities_and_agent_aligned() -> None:
    class _SourceQueryGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            entity_text = " ".join(kwargs["entities"])
            return SimpleNamespace(
                query=f"find {entity_text}",
                reasoning=f"Used only {entity_text} from this source item.",
                _retry_count=0,
                _max_retries=3,
            )

    async def _route_aligned_query(query: str, tenant_id: str):
        assert tenant_id == "acme:routing"
        return {
            "query": query,
            "routed_to": (
                "document_agent" if "TensorFlow" in query else "video_search_agent"
            ),
            "confidence": 0.81 if "TensorFlow" in query else 0.67,
        }

    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_aligned_query,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = _SourceQueryGenerator()

    examples = await generator.generate(
        sampled_content=[
            {
                "topic": "guide to TensorFlow",
                "schema_name": "document_text",
                "embedding_type": "single_vector",
            },
            {
                "topic": "tutorial on PyTorch",
                "schema_name": "video_segments",
                "embedding_type": "multi_vector",
            },
        ],
        target_count=2,
        tenant_id="acme:routing",
    )

    assert [
        {
            "query": example.query,
            "entities": example.entities,
            "enhanced_query": example.enhanced_query,
            "chosen_agent": example.chosen_agent,
            "routing_confidence": example.routing_confidence,
        }
        for example in examples
    ] == [
        {
            "query": "find TensorFlow",
            "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "enhanced_query": "find TensorFlow(TECHNOLOGY)",
            "chosen_agent": "document_agent",
            "routing_confidence": 0.81,
        },
        {
            "query": "find PyTorch",
            "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
            "enhanced_query": "find PyTorch(TECHNOLOGY)",
            "chosen_agent": "video_search_agent",
            "routing_confidence": 0.67,
        },
    ]


async def test_generation_drops_repeated_canonical_routing_label() -> None:
    class _RepeatedQueryGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            return SimpleNamespace(
                query="find TensorFlow",
                reasoning="The deterministic generator repeated one label.",
                _retry_count=0,
                _max_retries=3,
            )

    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = _RepeatedQueryGenerator()
    tracker = GenerationTracker(
        optimizer="routing",
        target_count=2,
        floor_count=1,
    )

    # TensorFlow only in record 0 - saliency preserves it as distinctive.
    # Record 1 has no extractable entity, so entity extraction fails for it.
    # Query generator always returns "find TensorFlow".
    # Iteration: attempt 0 → record 0 → KEPT; attempt 1 → record 1 → fails;
    # attempt 2+ → record 0 → DUPLICATE. Net: 1 example with duplicates dropped.
    examples = await generator.generate(
        sampled_content=[
            {"description": "TensorFlow machine learning framework tutorial"},
            {"description": "cooking recipes and kitchen tips video"},  # No entity
        ],
        target_count=2,
        tenant_id="acme:routing",
        generation_tracker=tracker,
        generation_floor_count=1,
    )

    assert [example.query for example in examples] == ["find TensorFlow"]
    assert [example.chosen_agent for example in examples] == ["video_search_agent"]
    assert tracker.returned_count == 1
    assert tracker.surplus_exhausted is True
    # Find the duplicate-related dropped example (not the no-entity ones)
    duplicate_drops = [d for d in tracker.dropped_examples if "duplicate" in d.reason]
    assert duplicate_drops[0].candidate == "find TensorFlow"
    assert duplicate_drops[0].reason == (
        "RoutingGenerator generated duplicate canonical label "
        "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
        "chosen_agent='video_search_agent')"
    )


@pytest.mark.asyncio
async def test_generation_fills_quota_after_one_duplicate() -> None:
    """Duplicate canonical labels are dropped and replaced from the surplus."""

    class _SequenceQueryGenerator:
        max_retries = 3

        def __init__(self) -> None:
            self.calls = 0
            self.outputs = iter(
                [
                    "find TensorFlow",
                    "find TensorFlow",
                    "find TensorFlow tutorial",
                    "compare TensorFlow benchmarks",
                    "TensorFlow deployment guide",
                    "explain TensorFlow pipelines",
                ]
            )

        def __call__(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                query=next(self.outputs),
                reasoning=f"Sequence attempt {self.calls}.",
                _retry_count=0,
                _max_retries=3,
            )

    query_generator = _SequenceQueryGenerator()
    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = query_generator
    tracker = GenerationTracker(
        optimizer="routing",
        target_count=5,
        floor_count=1,
    )

    # TensorFlow only in record 0 so saliency preserves it (high IDF).
    # Record 1 has no extractable entity, so entity extraction fails for it
    # and the generator cycles back to record 0. The sequence query generator
    # returns "find TensorFlow" twice → duplicate canonical label → dropped.
    examples = await generator.generate(
        sampled_content=[
            {"description": "TensorFlow machine learning framework tutorial"},
            {"description": "cooking recipes and kitchen tips video"},
        ],
        target_count=5,
        tenant_id="acme:routing",
        generation_tracker=tracker,
        generation_floor_count=1,
    )

    assert [example.query for example in examples] == [
        "find TensorFlow",
        "find TensorFlow tutorial",
        "compare TensorFlow benchmarks",
        "TensorFlow deployment guide",
        "explain TensorFlow pipelines",
    ]
    assert [example.enhanced_query for example in examples] == [
        "find TensorFlow(TECHNOLOGY)",
        "find TensorFlow(TECHNOLOGY) tutorial",
        "compare TensorFlow(TECHNOLOGY) benchmarks",
        "TensorFlow(TECHNOLOGY) deployment guide",
        "explain TensorFlow(TECHNOLOGY) pipelines",
    ]
    assert [example.chosen_agent for example in examples] == ["video_search_agent"] * 5
    assert [example.routing_confidence for example in examples] == [0.73] * 5
    assert query_generator.calls == 6
    assert tracker.returned_count == 5
    assert tracker.surplus_exhausted is False
    duplicate_drops = [d for d in tracker.dropped_examples if "duplicate" in d.reason]
    assert [drop.candidate for drop in duplicate_drops] == ["find TensorFlow"]
    assert duplicate_drops[0].reason == (
        "RoutingGenerator generated duplicate canonical label "
        "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
        "chosen_agent='video_search_agent')"
    )


@pytest.mark.asyncio
async def test_generation_stops_after_duplicate_streak_is_exhausted() -> None:
    """After target_count consecutive duplicate canonical labels, the draw stops."""

    class _SequenceQueryGenerator:
        max_retries = 3

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                query="find TensorFlow",
                reasoning=f"Sequence attempt {self.calls}.",
                _retry_count=0,
                _max_retries=3,
            )

    routing_calls = []

    async def _counted_route(query: str, tenant_id: str):
        assert tenant_id == "acme:routing"
        routing_calls.append(query)
        return await _route_query(query, tenant_id)

    query_generator = _SequenceQueryGenerator()
    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_counted_route,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = query_generator
    tracker = GenerationTracker(
        optimizer="routing",
        target_count=5,
        floor_count=1,
    )

    # TensorFlow only in record 0 so saliency preserves it (high IDF).
    # Record 1 has no extractable entity, so entity extraction fails for it.
    # The sequence query generator always returns "find TensorFlow" → every
    # successful attempt after the first is a duplicate. After target_count
    # consecutive duplicates, the generator stops.
    examples = await generator.generate(
        sampled_content=[
            {"description": "TensorFlow machine learning framework tutorial"},
            {"description": "cooking recipes and kitchen tips video"},
        ],
        target_count=5,
        tenant_id="acme:routing",
        generation_tracker=tracker,
        generation_floor_count=1,
    )

    assert [example.query for example in examples] == ["find TensorFlow"]
    assert [example.enhanced_query for example in examples] == [
        "find TensorFlow(TECHNOLOGY)"
    ]
    assert query_generator.calls == 6
    assert routing_calls == ["find TensorFlow"] * 6
    assert tracker.returned_count == 1
    assert tracker.surplus_exhausted is True
    duplicate_drops = [d for d in tracker.dropped_examples if "duplicate" in d.reason]
    assert [drop.candidate for drop in duplicate_drops] == ["find TensorFlow"] * 5
    assert duplicate_drops[-1].reason == (
        "RoutingGenerator generated duplicate canonical label "
        "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
        "chosen_agent='video_search_agent')"
    )


def test_duplicate_label_filter_golden() -> None:
    """Complete golden: duplicate detection is a pure state machine.

    Feed an explicit sequence of canonical labels, assert exact decisions,
    exact kept-label set, and exact drop errors. No LM, no mock, no fixture
    contortion — the contract is expressed directly.
    """
    filter_ = DuplicateLabelFilter()
    target_count = 3

    # Build the canonical label sequence: mix of unique, duplicate, streak
    labels = [
        # Label 0: unique → keep
        ("find TensorFlow", (("TensorFlow", "TECHNOLOGY"),), "video_search_agent"),
        # Label 1: duplicate of 0 → drop, streak=1
        ("find TensorFlow", (("TensorFlow", "TECHNOLOGY"),), "video_search_agent"),
        # Label 2: unique → keep, streak resets
        ("find PyTorch", (("PyTorch", "TECHNOLOGY"),), "video_search_agent"),
        # Label 3: duplicate of 0 → drop, streak=1
        ("find TensorFlow", (("TensorFlow", "TECHNOLOGY"),), "video_search_agent"),
        # Label 4: duplicate of 2 → drop, streak=2
        ("find PyTorch", (("PyTorch", "TECHNOLOGY"),), "video_search_agent"),
        # Label 5: duplicate of 0 → drop, streak=3 (== target_count) → stop
        ("find TensorFlow", (("TensorFlow", "TECHNOLOGY"),), "video_search_agent"),
        # Label 6: never reached
        ("find Marie Curie", (("Marie Curie", "PERSON"),), "research_agent"),
    ]

    # Expected decisions for each label
    expected_decisions = [
        "keep",  # 0: unique
        "drop",  # 1: dup of 0, streak=1
        "keep",  # 2: unique, streak resets
        "drop",  # 3: dup of 0, streak=1
        "drop",  # 4: dup of 2, streak=2
        "stop",  # 5: dup of 0, streak=3 == target_count
    ]

    # Expected error messages (None for keep, exact message for drop/stop)
    expected_errors = [
        None,
        (
            "RoutingGenerator generated duplicate canonical label "
            "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
            "chosen_agent='video_search_agent')"
        ),
        None,
        (
            "RoutingGenerator generated duplicate canonical label "
            "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
            "chosen_agent='video_search_agent')"
        ),
        (
            "RoutingGenerator generated duplicate canonical label "
            "(query='find PyTorch', entities=(('PyTorch', 'TECHNOLOGY'),), "
            "chosen_agent='video_search_agent')"
        ),
        (
            "RoutingGenerator generated duplicate canonical label "
            "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
            "chosen_agent='video_search_agent')"
        ),
    ]

    # Run the filter on all labels up to the stop
    actual_decisions = []
    actual_errors = []
    for label in labels:
        decision, error = filter_.check(label, target_count)
        actual_decisions.append(decision)
        actual_errors.append(str(error) if error else None)
        if decision == "stop":
            break

    # Assert exact decision sequence
    assert actual_decisions == expected_decisions

    # Assert exact error messages
    assert actual_errors == expected_errors

    # Assert exact kept labels (the unique ones)
    assert filter_.seen_count == 2
    assert filter_.seen_labels == {
        ("find TensorFlow", (("TensorFlow", "TECHNOLOGY"),), "video_search_agent"),
        ("find PyTorch", (("PyTorch", "TECHNOLOGY"),), "video_search_agent"),
    }

    # Assert final streak count
    assert filter_.duplicate_streak == 3


async def test_generation_preserves_actual_gateway_routing_decision() -> None:
    class _SourceQueryGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            return SimpleNamespace(
                query="find TensorFlow video",
                reasoning="Used the exact source entity and modality.",
                _retry_count=0,
                _max_retries=3,
            )

    class _VideoEntityModel:
        def predict_entities(self, query, labels, threshold):
            return [{"text": "video", "label": "video_content", "score": 0.91}]

    gateway = GatewayAgent(deps=GatewayDeps())
    gateway.telemetry_manager = RecordingTelemetryManager()
    gateway._gliner_model = _VideoEntityModel()
    gateway_calls = []
    gateway_outputs = []

    async def route_with_gateway(query: str, tenant_id: str):
        gateway_calls.append((query, tenant_id))
        output = await gateway.process(GatewayInput(query=query, tenant_id=tenant_id))
        gateway_outputs.append(output)
        return output

    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=route_with_gateway,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = _SourceQueryGenerator()

    examples = await generator.generate(
        sampled_content=[{"topic": "TensorFlow video"}, _second_routing_sample()],
        target_count=1,
        tenant_id="acme:routing",
    )

    assert gateway_calls == [("find TensorFlow video", "acme:routing")]
    assert examples[0].chosen_agent == "search_agent"
    for example in examples:
        assert example.routing_confidence == gateway_outputs[0].confidence
        assert example.metadata["_outcome_metadata"] == {
            "observed": True,
            "required_field_semantics": {
                "routing_confidence": "observed_gateway_confidence",
                "search_quality": "unobserved_zero_sentinel",
                "agent_success": "unobserved_false_sentinel",
                "processing_time": "unobserved_zero_sentinel",
            },
        }
        assert SyntheticDataConfidenceExtractor().get_confidence_breakdown(
            example.model_dump()
        ) == {
            "schema": "RoutingExperienceSchema",
            "confidence_field": "routing_confidence",
            "final_confidence": gateway_outputs[0].confidence,
            "outcome_observed": True,
            "requires_human_review": False,
        }


@pytest.mark.parametrize(
    ("confidence", "rendered"),
    [
        pytest.param(None, "None", id="missing"),
        pytest.param(True, "True", id="boolean"),
        pytest.param(1, "1", id="integer"),
        pytest.param("0.73", "'0.73'", id="string"),
        pytest.param(float("nan"), "nan", id="non-finite"),
        pytest.param(1.01, "1.01", id="above-range"),
    ],
)
async def test_generation_rejects_noncanonical_gateway_confidence(
    confidence, rendered
) -> None:
    class _SourceQueryGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            return SimpleNamespace(
                query="find TensorFlow video",
                reasoning="Used the exact source entity and modality.",
                _retry_count=0,
                _max_retries=3,
            )

    async def route_with_invalid_confidence(query: str, tenant_id: str):
        decision = {"query": query, "routed_to": "search_agent"}
        if confidence is not None:
            decision["confidence"] = confidence
        return decision

    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=route_with_invalid_confidence,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = _SourceQueryGenerator()

    with pytest.raises(ValueError) as error:
        await generator.generate(
            sampled_content=[{"topic": "TensorFlow video"}, _second_routing_sample()],
            target_count=1,
            tenant_id="acme:routing",
        )

    assert str(error.value) == (
        "routing decision confidence must be a finite float between 0 and 1; "
        f"got {rendered}"
    )


async def test_generation_metadata_matches_canonical_contract() -> None:
    class _MarkedGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            return SimpleNamespace(
                query="find machine learning about TensorFlow",
                reasoning="Used the supplied entity.",
                _retry_count=1,
                _max_retries=3,
            )

    generator = _routing_generator()
    generator.query_generator = _MarkedGenerator()

    query, metadata = await generator._generate_entity_query(
        [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
        "machine learning",
    )

    assert query == "find machine learning about TensorFlow"
    assert metadata == {
        "_generation_metadata": {
            "retry_count": 1,
            "max_retries": 3,
            "reasoning": "Used the supplied entity.",
        }
    }


async def test_routing_passes_entity_texts_and_types_as_structured_sequences() -> None:
    class _CapturingGenerator:
        max_retries = 3

        def __init__(self):
            self.inputs = []

        def __call__(self, **kwargs):
            self.inputs.append(kwargs)
            return SimpleNamespace(
                query="find Washington, D.C. reports from Meta AI",
                reasoning="Used both complete entities.",
                _retry_count=0,
                _max_retries=3,
            )

    query_generator = _CapturingGenerator()
    generator = _routing_generator()
    generator.query_generator = query_generator

    query, _ = await generator._generate_entity_query(
        [
            {"text": "Washington, D.C.", "type": "PLACE"},
            {"text": "Meta AI", "type": "ORGANIZATION"},
        ],
        "policy",
    )

    assert query == "find Washington, D.C. reports from Meta AI"
    assert query_generator.inputs == [
        {
            "topics": "policy",
            "entities": ["Washington, D.C.", "Meta AI"],
            "entity_types": ["PLACE", "ORGANIZATION"],
        }
    ]


@pytest.mark.parametrize(
    ("output", "invalid_field"),
    [
        pytest.param(
            {
                "reasoning": "Used the supplied entity.",
                "_retry_count": 0,
                "_max_retries": 3,
            },
            "query",
            id="missing-query",
        ),
        pytest.param(
            {
                "query": 17,
                "reasoning": "Used the supplied entity.",
                "_retry_count": 0,
                "_max_retries": 3,
            },
            "query",
            id="non-string-query",
        ),
        pytest.param(
            {
                "query": "   ",
                "reasoning": "Used the supplied entity.",
                "_retry_count": 0,
                "_max_retries": 3,
            },
            "query",
            id="blank-query",
        ),
        pytest.param(
            {
                "query": "find TensorFlow tutorials",
                "_retry_count": 0,
                "_max_retries": 3,
            },
            "reasoning",
            id="missing-reasoning",
        ),
        pytest.param(
            {
                "query": "find TensorFlow tutorials",
                "reasoning": 17,
                "_retry_count": 0,
                "_max_retries": 3,
            },
            "reasoning",
            id="non-string-reasoning",
        ),
        pytest.param(
            {
                "query": "find TensorFlow tutorials",
                "reasoning": "   ",
                "_retry_count": 0,
                "_max_retries": 3,
            },
            "reasoning",
            id="blank-reasoning",
        ),
    ],
)
async def test_query_generation_rejects_malformed_boundary_output(
    output: dict[str, object],
    invalid_field: str,
) -> None:
    class _MalformedGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            return SimpleNamespace(**output)

    generator = _routing_generator()
    generator.query_generator = _MalformedGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "Failed to generate valid entity query after 3 retries: "
            f"query generator returned {invalid_field} that is not a non-empty string"
        ),
    ):
        await generator._generate_entity_query(
            [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "machine learning",
        )


async def test_query_generation_rejects_empty_entities() -> None:
    generator = _routing_generator()

    with pytest.raises(
        ValueError,
        match="^entities must contain at least one item$",
    ):
        await generator._generate_entity_query([], "machine learning")


async def test_query_generation_keeps_event_loop_responsive() -> None:
    release = threading.Event()

    class _BlockingGenerator:
        max_retries = 3

        def __init__(self):
            self.released_by_event_loop = False

        def __call__(self, **kwargs):
            self.released_by_event_loop = release.wait(timeout=0.2)
            return SimpleNamespace(
                query="find TensorFlow",
                reasoning="Used exact entity.",
                _retry_count=0,
                _max_retries=3,
            )

    query_generator = _BlockingGenerator()
    generator = _routing_generator()
    generator.query_generator = query_generator

    async def release_generator():
        await asyncio.sleep(0)
        release.set()

    result, _ = await asyncio.gather(
        generator._generate_entity_query(
            [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "machine learning",
        ),
        release_generator(),
    )

    assert query_generator.released_by_event_loop is True
    assert result[0] == "find TensorFlow"


async def test_query_generation_boundary_failure_raises_with_context() -> None:
    class _UnavailableGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            raise TimeoutError("teacher LM timed out")

    generator = _routing_generator()
    generator.query_generator = _UnavailableGenerator()

    with pytest.raises(
        RuntimeError,
        match="^entity query generation failed for entities: TensorFlow$",
    ) as error:
        await generator._generate_entity_query(
            [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "machine learning",
        )

    assert isinstance(error.value.__cause__, TimeoutError)
    assert str(error.value.__cause__) == "teacher LM timed out"


async def test_concurrent_validated_dspy_boundaries_keep_inputs_separate() -> None:
    worker_barrier = threading.Barrier(2)
    boundary_inputs = []

    class _ConcurrentBoundary:
        def __call__(self, **kwargs):
            worker_barrier.wait(timeout=1)
            boundary_inputs.append(kwargs)
            entity_text = " ".join(json.loads(kwargs["entities"]))
            return SimpleNamespace(
                query=f"find {entity_text}",
                reasoning=f"Used {entity_text}.",
            )

    generator = _routing_generator()
    validated_generator = ValidatedEntityQueryGenerator(max_retries=3)
    validated_generator.generate = _ConcurrentBoundary()
    generator.query_generator = validated_generator

    pytorch, tensorflow = await asyncio.gather(
        generator._generate_entity_query(
            [{"text": "PyTorch", "type": "TECHNOLOGY"}],
            "machine learning",
        ),
        generator._generate_entity_query(
            [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "machine learning",
        ),
    )

    assert pytorch[0] == "find PyTorch"
    assert tensorflow[0] == "find TensorFlow"
    assert sorted(boundary_inputs, key=lambda item: item["entities"]) == [
        {
            "topics": "machine learning",
            "entities": '["PyTorch"]',
            "entity_types": '["TECHNOLOGY"]',
        },
        {
            "topics": "machine learning",
            "entities": '["TensorFlow"]',
            "entity_types": '["TECHNOLOGY"]',
        },
    ]


def test_query_generator_cold_build_constructs_once_under_thread_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = threading.Barrier(2)
    second_build_started = threading.Event()
    build_count = 0
    count_lock = threading.Lock()

    class _BlockingValidatedGenerator:
        def __init__(self, max_retries: int):
            nonlocal build_count
            with count_lock:
                build_count += 1
                current_count = build_count
            self.max_retries = max_retries
            if current_count == 1:
                second_build_started.wait(timeout=0.2)
            else:
                second_build_started.set()

    monkeypatch.setattr(
        "cogniverse_synthetic.dspy_modules.ValidatedEntityQueryGenerator",
        _BlockingValidatedGenerator,
    )
    generator = _routing_generator()

    def get_query_generator():
        start.wait(timeout=1)
        return generator._get_query_generator()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(get_query_generator) for _ in range(2)]
        results = [future.result(timeout=1) for future in futures]

    assert build_count == 1
    assert results[0] is results[1]


async def test_query_generation_validation_exhaustion_is_a_value_error() -> None:
    class _ExhaustedGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            raise EntityQueryValidationError(
                "Entity query validation failed after 3 attempts for entities: "
                "TensorFlow"
            )

    generator = _routing_generator()
    generator.query_generator = _ExhaustedGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "^Failed to generate valid entity query after 3 retries: Entity query "
            "validation failed after 3 attempts for entities: TensorFlow$"
        ),
    ) as error:
        await generator._generate_entity_query(
            [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "machine learning",
        )

    assert type(error.value.__cause__) is EntityQueryValidationError


async def test_generation_drops_candidate_when_query_validation_is_exhausted() -> None:
    class _ExhaustThenSucceed:
        max_retries = 3

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise EntityQueryValidationError(
                    "Entity query validation failed after 3 attempts for entities: "
                    "TensorFlow"
                )
            return SimpleNamespace(
                query="find TensorFlow",
                reasoning="Used the supplied entity.",
                _retry_count=0,
                _max_retries=3,
            )

    query_generator = _ExhaustThenSucceed()
    generator = _routing_generator()
    generator.query_generator = query_generator
    tracker = GenerationTracker(optimizer="routing", target_count=1, floor_count=1)

    examples = await generator.generate(
        sampled_content=[{"topic": "TensorFlow"}, _second_routing_sample()],
        target_count=1,
        tenant_id="acme:routing",
        generation_tracker=tracker,
        generation_floor_count=1,
    )

    assert [example.query for example in examples] == ["find TensorFlow"]
    assert query_generator.calls == 2
    assert tracker.returned_count == 1
    assert len(tracker.dropped_examples) == 1
    assert [drop.candidate for drop in tracker.dropped_examples] == ["TensorFlow"]
    assert tracker.dropped_examples[0].reason == (
        "Failed to generate valid entity query after 3 retries: Entity query "
        "validation failed after 3 attempts for entities: TensorFlow"
    )


async def test_generation_drops_candidate_when_labeler_yields_no_example() -> None:
    generator = _routing_generator()
    real_generate = generator.entity_labeler.generate
    labeler_inputs = []

    async def label_entities(sampled_content, target_count, tenant_id):
        labeler_inputs.append(sampled_content)
        if sampled_content == [{"topic": "blank frame"}]:
            raise ValueError(
                "EntityExtractionGenerator generated 0 unique grounded examples "
                "but target_count=1"
            )
        return await real_generate(
            sampled_content=sampled_content,
            target_count=target_count,
            tenant_id=tenant_id,
        )

    generator.entity_labeler.generate = label_entities

    class _Generator:
        max_retries = 3

        def __call__(self, **kwargs):
            return SimpleNamespace(
                query=f"find {kwargs['topics']}",
                reasoning="Used the supplied entity.",
                _retry_count=0,
                _max_retries=3,
            )

    generator.query_generator = _Generator()
    tracker = GenerationTracker(optimizer="routing", target_count=1, floor_count=1)

    examples = await generator.generate(
        sampled_content=[{"topic": "blank frame"}, {"topic": "TensorFlow"}],
        target_count=1,
        tenant_id="acme:routing",
        generation_tracker=tracker,
        generation_floor_count=1,
    )

    assert labeler_inputs == [[{"topic": "blank frame"}], [{"topic": "TensorFlow"}]]
    assert [example.query for example in examples] == ["find TensorFlow"]
    assert tracker.returned_count == 1
    assert [(drop.candidate, drop.reason) for drop in tracker.dropped_examples] == [
        (
            "blank frame",
            "EntityExtractionGenerator generated 0 unique grounded examples "
            "but target_count=1",
        )
    ]


async def test_generation_still_raises_when_query_generator_is_unavailable() -> None:
    class _UnavailableGenerator:
        max_retries = 3

        def __call__(self, **kwargs):
            raise TimeoutError("teacher LM timed out")

    generator = _routing_generator()
    generator.query_generator = _UnavailableGenerator()
    tracker = GenerationTracker(optimizer="routing", target_count=1, floor_count=1)

    with pytest.raises(
        RuntimeError,
        match="^entity query generation failed for entities: TensorFlow$",
    ) as error:
        await generator.generate(
            sampled_content=[{"topic": "TensorFlow"}, _second_routing_sample()],
            target_count=1,
            tenant_id="acme:routing",
            generation_tracker=tracker,
            generation_floor_count=1,
        )

    assert type(error.value.__cause__) is TimeoutError
    assert tracker.dropped_examples == []


def test_to_example_collapses_repeated_relationship_triples() -> None:
    """An exactly-repeated triple collapses to one, mirroring entity handling.

    The entity loop already skips an exact duplicate text/type pair, so a
    teacher LM that restates the same relationship must not produce a second
    identical triple. ``_validate_training_item`` rejects duplicates outright,
    so passing them through fails the whole approval batch.
    """
    text = "Marie Curie isolated radium at the Sorbonne."
    example = EntityExtractionGenerator._to_example(
        text,
        {
            "query": text,
            "entities": [
                {"text": "Marie Curie", "type": "PERSON"},
                {"text": "radium", "type": "SUBSTANCE"},
                {"text": "Sorbonne", "type": "ORGANIZATION"},
            ],
            "relationships": [
                {
                    "subject": "Marie Curie",
                    "relation": "isolated",
                    "object": "radium",
                },
                {
                    "subject": "Marie Curie",
                    "relation": "isolated",
                    "object": "radium",
                },
                {
                    "subject": "Marie Curie",
                    "relation": "worked_at",
                    "object": "Sorbonne",
                },
            ],
        },
    )

    assert example.query == text
    assert example.entities == [
        {"text": "Marie Curie", "type": "PERSON"},
        {"text": "radium", "type": "SUBSTANCE"},
        {"text": "Sorbonne", "type": "ORGANIZATION"},
    ]
    assert example.entity_types == "PERSON,SUBSTANCE,ORGANIZATION"
    assert example.relationships == [
        {"source": "Marie Curie", "target": "radium", "type": "isolated"},
        {"source": "Marie Curie", "target": "Sorbonne", "type": "worked_at"},
    ]


def test_to_example_keeps_distinct_relations_between_one_entity_pair() -> None:
    """Same endpoints under a different relation are separate facts."""
    text = "Marie Curie studied at the Sorbonne and later taught at the Sorbonne."
    example = EntityExtractionGenerator._to_example(
        text,
        {
            "query": text,
            "entities": [
                {"text": "Marie Curie", "type": "PERSON"},
                {"text": "Sorbonne", "type": "ORGANIZATION"},
            ],
            "relationships": [
                {
                    "subject": "Marie Curie",
                    "relation": "studied_at",
                    "object": "Sorbonne",
                },
                {
                    "subject": "Marie Curie",
                    "relation": "taught_at",
                    "object": "Sorbonne",
                },
            ],
        },
    )

    assert example.relationships == [
        {"source": "Marie Curie", "target": "Sorbonne", "type": "studied_at"},
        {"source": "Marie Curie", "target": "Sorbonne", "type": "taught_at"},
    ]


def test_canonicalized_relationships_follow_the_kept_entity_spelling() -> None:
    """Endpoints repoint at the spelling _canonicalize_entities kept.

    Entity canonicalization collapses case variants and keeps the first
    spelling, so a relationship naming a dropped variant would reference an
    entity the item no longer carries. ``_validate_entities`` rejects that,
    which fails the whole approval batch.
    """
    entities = [
        {"text": "Sorbonne", "type": "ORGANIZATION"},
        {"text": "sorbonne", "type": "ORGANIZATION"},
        {"text": "Marie Curie", "type": "PERSON"},
    ]
    canonical_entities = RoutingGenerator._canonicalize_entities(entities)
    canonical_relationships = RoutingGenerator._canonicalize_relationships(
        [{"source": "Marie Curie", "target": "sorbonne", "type": "taught_at"}],
        canonical_entities,
    )

    assert canonical_relationships == [
        {"source": "Marie Curie", "target": "Sorbonne", "type": "taught_at"}
    ]
    _validate_entities(
        {
            "entities": canonical_entities,
            "entity_types": "ORGANIZATION,PERSON",
            "relationships": canonical_relationships,
        },
        "routing_item",
    )


def test_canonicalized_relationships_collapse_triples_made_equal() -> None:
    """Repointing can make two triples identical; keep one."""
    canonical_entities = RoutingGenerator._canonicalize_entities(
        [
            {"text": "Sorbonne", "type": "ORGANIZATION"},
            {"text": "sorbonne", "type": "ORGANIZATION"},
            {"text": "Marie Curie", "type": "PERSON"},
        ]
    )
    canonical_relationships = RoutingGenerator._canonicalize_relationships(
        [
            {"source": "Marie Curie", "target": "Sorbonne", "type": "taught_at"},
            {"source": "Marie Curie", "target": "sorbonne", "type": "taught_at"},
        ],
        canonical_entities,
    )

    assert canonical_relationships == [
        {"source": "Marie Curie", "target": "Sorbonne", "type": "taught_at"}
    ]


def test_canonicalized_relationships_reject_an_unknown_endpoint() -> None:
    """An endpoint naming no canonical entity is an upstream defect."""
    canonical_entities = RoutingGenerator._canonicalize_entities(
        [{"text": "Marie Curie", "type": "PERSON"}]
    )

    with pytest.raises(ValueError) as error:
        RoutingGenerator._canonicalize_relationships(
            [{"source": "Marie Curie", "target": "Sorbonne", "type": "taught_at"}],
            canonical_entities,
        )

    assert str(error.value) == (
        "relationships[0].target 'Sorbonne' is absent from the canonical entities"
    )
