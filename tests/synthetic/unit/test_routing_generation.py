"""Routing query generation correctness under edge and concurrent inputs."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps, GatewayInput
from cogniverse_foundation.config.unified_config import (
    DSPyModuleConfig,
    OptimizerGenerationConfig,
)
from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)
from cogniverse_synthetic.dspy_modules import ValidatedEntityQueryGenerator
from cogniverse_synthetic.generators.base import GenerationTracker
from cogniverse_synthetic.generators.routing import RoutingGenerator

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
        RuntimeError,
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
        RuntimeError,
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
        RuntimeError,
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

    with pytest.raises(
        ValueError, match="^sampled routing content requires a non-empty topic$"
    ):
        await generator.generate(
            sampled_content=[
                {
                    "schema_name": "document_text",
                    "embedding_type": "single_vector",
                }
            ],
            target_count=1,
            tenant_id="acme:routing",
        )


async def test_query_generation_rejects_missing_source_topic() -> None:
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
                {
                    "schema_name": "document_text",
                    "embedding_type": "single_vector",
                }
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

    generator = RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=_routing_generator().optimizer_config,
    )
    query_generator = _SourceQueryGenerator()
    generator.query_generator = query_generator

    examples = await generator.generate(
        sampled_content=[
            {
                "topic": "TensorFlow deployment guide",
                "schema_name": "document_text",
                "embedding_type": "single_vector",
            }
        ],
        target_count=1,
        tenant_id="acme:routing",
    )

    assert query_generator.inputs == [
        {
            "topics": "TensorFlow deployment guide",
            "entities": ["TensorFlow"],
            "entity_types": ["TECHNOLOGY"],
        }
    ]
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
                }
            ],
            target_count=1,
            tenant_id="acme:routing",
        )

    assert str(error.value) == (
        "EntityExtractionGenerator generated 0 unique grounded examples but "
        "target_count=1; source_context=2 unique source texts, 2 without entities"
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

    examples = await generator.generate(
        sampled_content=[
            {"topic": "TensorFlow graph execution"},
            {"topic": "TensorFlow graph optimization"},
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
    assert tracker.dropped_examples[0].candidate == "find TensorFlow"
    assert tracker.dropped_examples[0].reason == (
        "RoutingGenerator generated duplicate canonical label "
        "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
        "chosen_agent='video_search_agent')"
    )


async def test_generation_fills_quota_after_one_duplicate() -> None:
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

    examples = await generator.generate(
        sampled_content=[{"topic": "TensorFlow"}],
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
    assert [drop.candidate for drop in tracker.dropped_examples] == ["find TensorFlow"]
    assert tracker.dropped_examples[0].reason == (
        "RoutingGenerator generated duplicate canonical label "
        "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
        "chosen_agent='video_search_agent')"
    )


async def test_generation_stops_after_duplicate_streak_is_exhausted() -> None:
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

    entity_calls = []
    routing_calls = []

    async def _counted_entities(text: str, tenant_id: str):
        assert tenant_id == "acme:routing"
        entity_calls.append(text)
        return await _extract_entities(text, tenant_id)

    async def _counted_route(query: str, tenant_id: str):
        assert tenant_id == "acme:routing"
        routing_calls.append(query)
        return await _route_query(query, tenant_id)

    query_generator = _SequenceQueryGenerator()
    generator = RoutingGenerator(
        entity_extractor=_counted_entities,
        routing_decider=_counted_route,
        optimizer_config=_routing_generator().optimizer_config,
    )
    generator.query_generator = query_generator
    tracker = GenerationTracker(
        optimizer="routing",
        target_count=5,
        floor_count=1,
    )

    examples = await generator.generate(
        sampled_content=[{"topic": "TensorFlow"}],
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
    assert entity_calls == ["TensorFlow"] * 6
    assert routing_calls == ["find TensorFlow"] * 6
    assert tracker.returned_count == 1
    assert tracker.surplus_exhausted is True
    assert [drop.candidate for drop in tracker.dropped_examples] == [
        "find TensorFlow"
    ] * 5
    assert [drop.reason for drop in tracker.dropped_examples] == [
        (
            "RoutingGenerator generated duplicate canonical label "
            "(query='find TensorFlow', entities=(('TensorFlow', 'TECHNOLOGY'),), "
            "chosen_agent='video_search_agent')"
        )
    ] * 5


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
        sampled_content=[{"topic": "TensorFlow video"}],
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
            sampled_content=[{"topic": "TensorFlow video"}],
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
