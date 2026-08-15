"""Real Vespa content is labelled by the production entity agent."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.generators import EntityExtractionGenerator
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = pytest.mark.integration

HASH_VALUE = "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691"


@pytest.fixture(scope="module")
def ee_service(shared_vespa):
    tenant_id = f"synentity{uuid.uuid4().hex[:8]}:media"
    profile_name = "video_colpali_smol500_mv_frame"
    title = "PyTorch was released by Meta AI in Menlo Park"
    description = "PyTorch was released by Meta AI in Menlo Park"
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        tenant_id=tenant_id,
        profiles={
            profile_name: BackendProfileConfig(
                profile_name=profile_name,
                type="video",
                schema_name=profile_name,
                embedding_type="multi_vector",
                pipeline_config={"generate_descriptions": True},
            )
        },
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant_id})
    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    schema = registry.deploy_schema(
        tenant_id=tenant_id,
        base_schema_name=profile_name,
    )
    feed = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    ).feed_data_point(
        schema=schema,
        data_id="pytorch-meta-segment",
        fields={
            "video_id": "pytorch-meta",
            "video_title": title,
            "source_url": "http://example.test/pytorch-meta",
            "segment_id": 0,
            "segment_description": description,
            "start_time": 0.0,
            "end_time": 9.0,
        },
    )
    assert feed.is_successful(), feed.json

    for _ in range(20):
        indexed = backend.query_metadata_documents(
            schema=schema,
            yql=f"select * from sources {schema} where true limit 1",
            hits=1,
        )
        if indexed:
            assert indexed[0]["video_title"] == title
            assert indexed[0]["segment_description"] == description
            break
        time.sleep(0.5)
    else:
        pytest.fail("PyTorch source document was not indexed by Vespa")

    entity_agent = EntityExtractionAgent(deps=EntityExtractionDeps())
    extraction_paths = []

    async def extract_entities(text: str, tenant_id: str):
        result = await entity_agent.process(
            EntityExtractionInput(query=text, tenant_id=tenant_id)
        )
        extraction_paths.append(result.path_used)
        if result.path_used != "fast":
            raise RuntimeError(
                "production entity extraction did not use the GLiNER fast path"
            )
        return result

    raw_config = json.loads(Path("configs/config.json").read_text())
    generator_config = dict(raw_config["synthetic"])
    generator_config["tenant_id"] = tenant_id
    service = SyntheticDataService(
        backend=backend,
        generator_config=SyntheticGeneratorConfig.from_dict(generator_config),
        backend_config=backend_config,
        agents_config=raw_config["agents"],
        entity_extractor=extract_entities,
    )
    try:
        yield SimpleNamespace(
            service=service,
            tenant_id=tenant_id,
            profile_name=profile_name,
            source_text=description,
            extraction_paths=extraction_paths,
        )
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_service_generates_entity_extraction_examples(ee_service):
    prior_extractions = len(ee_service.extraction_paths)
    request = SyntheticDataRequest(
        tenant_id=ee_service.tenant_id,
        optimizer="entity_extraction",
        count=1,
        vespa_sample_size=1,
        max_profiles=1,
    )
    response = await ee_service.service.generate(request)

    assert response.optimizer == "entity_extraction"
    assert response.schema_name == EntityExtractionExampleSchema.__name__
    assert response.count == 1
    assert len(response.data) == 1
    assert response.selected_profiles == [ee_service.profile_name]
    assert response.metadata["sampled_content_count"] == 1
    assert response.metadata["generation"] == {
        "requested_count": 1,
        "returned_count": 1,
        "shortfall_count": 0,
        "floor_count": 1,
        "surplus_exhausted": False,
        "dropped_count": 0,
        "dropped_examples": [],
    }
    assert ee_service.extraction_paths[prior_extractions:] == ["fast"]

    for item in response.data:
        assert item == {
            "query": ee_service.source_text,
            "entities": [
                {"text": "PyTorch", "type": "TECHNOLOGY"},
                {"text": "Meta AI", "type": "ORGANIZATION"},
                {"text": "Menlo Park", "type": "PLACE"},
            ],
            "entity_types": "TECHNOLOGY,ORGANIZATION,PLACE",
            "relationships": [
                {
                    "source": "Meta AI",
                    "target": "Menlo Park",
                    "type": "in",
                }
            ],
        }


@pytest.mark.asyncio
async def test_generator_extracts_entities_from_content():
    observed_queries = []

    async def extract_entities(text: str, tenant_id: str):
        observed_queries.append((text, tenant_id))
        entities_by_text = {
            "PyTorch was released by Meta AI": [
                {"text": "PyTorch", "type": "TECHNOLOGY"},
                {"text": "Meta AI", "type": "ORGANIZATION"},
            ],
            "TensorFlow is maintained by Google": [
                {"text": "TensorFlow", "type": "TECHNOLOGY"},
                {"text": "Google", "type": "ORGANIZATION"},
            ],
        }
        return {
            "query": text,
            "entities": entities_by_text[text],
            "relationships": [],
        }

    generator = EntityExtractionGenerator(entity_extractor=extract_entities)
    sampled = [
        {"title": "PyTorch was released by Meta AI"},
        {"title": "TensorFlow is maintained by Google"},
    ]
    examples = await generator.generate(
        sampled_content=sampled,
        target_count=2,
        tenant_id="acme:synthetic",
    )

    assert [example.model_dump() for example in examples] == [
        {
            "query": "PyTorch was released by Meta AI",
            "entities": [
                {"text": "PyTorch", "type": "TECHNOLOGY"},
                {"text": "Meta AI", "type": "ORGANIZATION"},
            ],
            "entity_types": "TECHNOLOGY,ORGANIZATION",
            "relationships": [],
        },
        {
            "query": "TensorFlow is maintained by Google",
            "entities": [
                {"text": "TensorFlow", "type": "TECHNOLOGY"},
                {"text": "Google", "type": "ORGANIZATION"},
            ],
            "entity_types": "TECHNOLOGY,ORGANIZATION",
            "relationships": [],
        },
    ]
    assert observed_queries == [
        ("PyTorch was released by Meta AI", "acme:synthetic"),
        ("TensorFlow is maintained by Google", "acme:synthetic"),
    ]


@pytest.mark.parametrize(
    "item",
    [
        {"title": HASH_VALUE},
        {"audio_transcript": "*Screaming*"},
    ],
)
def test_generator_ignores_hash_and_annotation_only_candidate_texts(item):
    async def noop_extractor(text: str, tenant_id: str):
        return {"query": text, "entities": [], "relationships": []}

    generator = EntityExtractionGenerator(entity_extractor=noop_extractor)

    assert generator._candidate_texts([item]) == []


def test_generator_candidate_texts_include_document_fields_and_strip_bom():
    async def noop_extractor(text: str, tenant_id: str):
        return {"query": text, "entities": [], "relationships": []}

    generator = EntityExtractionGenerator(entity_extractor=noop_extractor)

    assert generator._candidate_texts(
        [
            {
                "document_title": "Annual report",
                "full_text": "\ufeffThe video is of people applaud in the arena",
            }
        ]
    ) == ["Annual report", "The video is of people applaud in the arena"]


@pytest.mark.asyncio
async def test_generator_scans_later_fields_and_stops_at_grounded_target():
    calls = []

    async def extract_entities(text: str, tenant_id: str):
        calls.append((text, tenant_id))
        entities = {
            "Marie Curie discovered radium": [
                {"text": "Marie Curie", "type": "PERSON"}
            ],
            "Ada Lovelace wrote the first algorithm": [
                {"text": "Ada Lovelace", "type": "PERSON"}
            ],
            "This source must not be requested": [
                {"text": "This source", "type": "DOCUMENT"}
            ],
        }.get(text, [])
        return {"query": text, "entities": entities, "relationships": []}

    examples = await EntityExtractionGenerator(
        entity_extractor=extract_entities
    ).generate(
        sampled_content=[
            {
                "title": "generic introduction",
                "description": "Marie Curie discovered radium",
            },
            {
                "title": "Ada Lovelace wrote the first algorithm",
                "content": "This source must not be requested",
            },
        ],
        target_count=2,
        tenant_id="acme:synthetic",
    )

    assert calls == [
        ("generic introduction", "acme:synthetic"),
        ("Marie Curie discovered radium", "acme:synthetic"),
        ("Ada Lovelace wrote the first algorithm", "acme:synthetic"),
    ]
    assert [example.model_dump() for example in examples] == [
        {
            "query": "Marie Curie discovered radium",
            "entities": [{"text": "Marie Curie", "type": "PERSON"}],
            "entity_types": "PERSON",
            "relationships": [],
        },
        {
            "query": "Ada Lovelace wrote the first algorithm",
            "entities": [{"text": "Ada Lovelace", "type": "PERSON"}],
            "entity_types": "PERSON",
            "relationships": [],
        },
    ]


@pytest.mark.asyncio
async def test_generator_scans_beyond_one_hundred_entity_free_records():
    calls = []

    async def extract_entities(text: str, tenant_id: str):
        calls.append(text)
        entities = (
            [{"text": "Marie Curie", "type": "PERSON"}]
            if text == "Marie Curie discovered radium"
            else []
        )
        return {"query": text, "entities": entities, "relationships": []}

    sampled_content = [
        {"title": f"entity-free source {index:03d}"} for index in range(100)
    ] + [{"title": "Marie Curie discovered radium"}]
    examples = await EntityExtractionGenerator(
        entity_extractor=extract_entities
    ).generate(
        sampled_content=sampled_content,
        target_count=1,
        tenant_id="acme:synthetic",
    )

    assert len(calls) == 101
    assert calls[-1] == "Marie Curie discovered radium"
    assert examples[0].model_dump() == {
        "query": "Marie Curie discovered radium",
        "entities": [{"text": "Marie Curie", "type": "PERSON"}],
        "entity_types": "PERSON",
        "relationships": [],
    }


def test_generator_accepts_exact_punctuated_source_span():
    source = "The hearing took place in Washington, D.C. yesterday."

    example = EntityExtractionGenerator._to_example(
        source,
        {
            "query": source,
            "entities": [{"text": "Washington, D.C.", "type": "PLACE"}],
            "relationships": [],
        },
    )

    assert example is not None
    assert example.entities == [{"text": "Washington, D.C.", "type": "PLACE"}]


def test_generator_drops_identical_duplicate_entity_text():
    source = "Meta AI released PyTorch."

    example = EntityExtractionGenerator._to_example(
        source,
        {
            "query": source,
            "entities": [
                {"text": "Meta AI", "type": "ORGANIZATION"},
                {"text": "Meta AI", "type": "ORGANIZATION"},
                {"text": "PyTorch", "type": "TECHNOLOGY"},
            ],
            "relationships": [],
        },
    )

    assert example is not None
    assert example.entities == [
        {"text": "Meta AI", "type": "ORGANIZATION"},
        {"text": "PyTorch", "type": "TECHNOLOGY"},
    ]
    assert example.entity_types == "ORGANIZATION,TECHNOLOGY"


def test_generator_rejects_conflicting_types_for_duplicate_entity_text():
    source = "Meta AI released PyTorch."

    with pytest.raises(
        ValueError,
        match=(
            "entity extractor result contains conflicting types for duplicate "
            "entity text 'Meta AI': 'ORGANIZATION' and 'PLACE'"
        ),
    ):
        EntityExtractionGenerator._to_example(
            source,
            {
                "query": source,
                "entities": [
                    {"text": "Meta AI", "type": "ORGANIZATION"},
                    {"text": "Meta AI", "type": "PLACE"},
                ],
                "relationships": [],
            },
        )


@pytest.mark.parametrize(
    ("entity", "field_name"),
    [
        ({"text": " Meta AI", "type": "ORGANIZATION"}, "text"),
        ({"text": "Meta AI", "type": " ORGANIZATION"}, "type"),
    ],
)
def test_generator_rejects_entity_fields_with_surrounding_whitespace(
    entity: dict[str, str],
    field_name: str,
):
    source = "Meta AI released PyTorch."

    with pytest.raises(
        ValueError,
        match=(
            rf"entity extractor entities\[0\]\.{field_name} contains "
            "surrounding whitespace"
        ),
    ):
        EntityExtractionGenerator._to_example(
            source,
            {"query": source, "entities": [entity], "relationships": []},
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("subject", " Meta AI"),
        ("relation", " released"),
        ("object", "PyTorch "),
    ],
)
def test_generator_rejects_relationship_fields_with_surrounding_whitespace(
    field_name: str,
    field_value: str,
):
    source = "Meta AI released PyTorch."
    relationship = {
        "subject": "Meta AI",
        "relation": "released",
        "object": "PyTorch",
    }
    relationship[field_name] = field_value

    with pytest.raises(
        ValueError,
        match=(
            rf"entity extractor relationships\[0\]\.{field_name} contains "
            "surrounding whitespace"
        ),
    ):
        EntityExtractionGenerator._to_example(
            source,
            {
                "query": source,
                "entities": [
                    {"text": "Meta AI", "type": "ORGANIZATION"},
                    {"text": "PyTorch", "type": "TECHNOLOGY"},
                ],
                "relationships": [relationship],
            },
        )


@pytest.mark.parametrize(
    ("source", "entity_text"),
    [
        pytest.param(
            "Meta AI published the model.",
            "Meta A",
            id="partial-multiword-token",
        ),
        pytest.param(
            "A NewYorker profile covered the event.",
            "York",
            id="embedded-substring",
        ),
        pytest.param(
            "PyTorch powers the service.",
            "pytorch",
            id="altered-casing",
        ),
        pytest.param(
            "The hearing took place in Washington, D.C. yesterday.",
            "Washington, D.",
            id="partial-punctuated-span",
        ),
    ],
)
def test_generator_rejects_entity_that_is_not_an_exact_complete_source_span(
    source: str,
    entity_text: str,
):
    with pytest.raises(
        ValueError,
        match=r"entity extractor entities\[0\]\.text must be an exact complete source span",
    ):
        EntityExtractionGenerator._to_example(
            source,
            {
                "query": source,
                "entities": [{"text": entity_text, "type": "PLACE"}],
                "relationships": [],
            },
        )


@pytest.mark.asyncio
async def test_generator_rejects_content_without_entities():
    async def extract_entities(text: str, tenant_id: str):
        return {"query": text, "entities": [], "relationships": []}

    generator = EntityExtractionGenerator(entity_extractor=extract_entities)

    with pytest.raises(ValueError) as error:
        await generator.generate(
            sampled_content=[{"title": "all words are lowercase"}],
            target_count=1,
            tenant_id="acme:synthetic",
        )

    assert str(error.value) == (
        "EntityExtractionGenerator generated 0 unique grounded examples but "
        "target_count=1; source_context=1 unique source texts, 1 without entities"
    )


@pytest.mark.asyncio
async def test_generator_rejects_partial_entity_bearing_source_set():
    async def extract_entities(text: str, tenant_id: str):
        entities = (
            [{"text": "PyTorch", "type": "TECHNOLOGY"}]
            if text == "PyTorch works"
            else []
        )
        return {"query": text, "entities": entities, "relationships": []}

    generator = EntityExtractionGenerator(entity_extractor=extract_entities)

    examples = await generator.generate(
        sampled_content=[
            {"title": "PyTorch works"},
            {"title": "all words are lowercase"},
        ],
        target_count=2,
        tenant_id="acme:synthetic",
    )

    assert [example.model_dump() for example in examples] == [
        {
            "query": "PyTorch works",
            "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
            "entity_types": "TECHNOLOGY",
            "relationships": [],
        }
    ]


@pytest.mark.asyncio
async def test_generator_returns_partial_data_when_extractor_fails():
    calls = []

    async def extract_entities(text: str, tenant_id: str):
        calls.append(text)
        if text == "Meta AI failed":
            raise ConnectionError("GLiNER endpoint closed")
        return {
            "query": text,
            "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
            "relationships": [],
        }

    generator = EntityExtractionGenerator(entity_extractor=extract_entities)

    with pytest.raises(RuntimeError) as error:
        await generator.generate(
            sampled_content=[
                {"title": "PyTorch works"},
                {"title": "Meta AI failed"},
            ],
            target_count=2,
            tenant_id="acme:synthetic",
        )

    assert (
        str(error.value) == "entity extraction failed for source text 'Meta AI failed'"
    )
    assert isinstance(error.value.__cause__, ConnectionError)
    assert str(error.value.__cause__) == "GLiNER endpoint closed"
    assert calls == ["PyTorch works", "Meta AI failed"]


@pytest.mark.asyncio
async def test_generator_raises_when_hung_entity_extraction_yields_no_examples():
    never_finishes = asyncio.Event()

    async def extract_entities(text: str, tenant_id: str):
        await never_finishes.wait()

    generator = EntityExtractionGenerator(
        entity_extractor=extract_entities,
        extraction_timeout_seconds=0.01,
    )

    with pytest.raises(RuntimeError) as error:
        await generator.generate(
            sampled_content=[{"title": "PyTorch works"}],
            target_count=1,
            tenant_id="acme:synthetic",
        )

    assert str(error.value) == (
        "entity extraction timed out after 0.01 seconds for source text 'PyTorch works'"
    )
    assert isinstance(error.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_concurrent_generations_keep_tenant_and_entity_results_isolated():
    gate = asyncio.Event()
    entered = 0

    async def extract_entities(text: str, tenant_id: str):
        nonlocal entered
        entered += 1
        if entered == 2:
            gate.set()
        await gate.wait()
        return {
            "query": text,
            "entities": [{"text": text, "type": tenant_id}],
            "relationships": [],
        }

    generator = EntityExtractionGenerator(entity_extractor=extract_entities)
    first, second = await asyncio.gather(
        generator.generate(
            sampled_content=[{"title": "PyTorch"}],
            target_count=1,
            tenant_id="tenant:first",
        ),
        generator.generate(
            sampled_content=[{"title": "TensorFlow"}],
            target_count=1,
            tenant_id="tenant:second",
        ),
    )

    assert first[0].entities == [{"text": "PyTorch", "type": "tenant:first"}]
    assert second[0].entities == [{"text": "TensorFlow", "type": "tenant:second"}]


@pytest.mark.asyncio
async def test_example_scores_perfectly_against_itself(ee_service):
    """A produced example fed through the finetuning evaluator against itself
    must score correct (F1 = 1.0), proving the entity shape satisfies the
    real ``_check_entity_prediction`` boundary."""
    request = SyntheticDataRequest(
        tenant_id=ee_service.tenant_id,
        optimizer="entity_extraction",
        count=1,
        vespa_sample_size=1,
        max_profiles=1,
    )
    response = await ee_service.service.generate(request)

    for item in response.data:
        payload = {"entities": item["entities"]}
        correct, f1 = AdapterEvaluator._check_entity_prediction(payload, payload)
        assert correct is True
        assert f1 == pytest.approx(1.0)
