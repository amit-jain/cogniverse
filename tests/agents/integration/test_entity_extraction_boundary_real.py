"""Real boundary tests for EntityExtractionAgent routing and telemetry."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.entity_extraction_agent import (
    ENTITY_TYPES,
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
    EntityExtractionModule,
    EntityMention,
)
from cogniverse_foundation.dspy import signature_response_format
from cogniverse_foundation.telemetry.span_contract import read_span_io

pytestmark = pytest.mark.integration


def _entity_output(*rows: tuple[str, str]) -> list[dict[str, str]]:
    """Entities as the enforced schema's JSON carries them."""
    return [{"text": text, "type": entity_type} for text, entity_type in rows]


def _telemetry_capture():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("entity-extraction-boundary-test")

    class _TelemetryManager:
        def span(
            self,
            name,
            *,
            tenant_id,
            project_name=None,
            attributes=None,
            require_export=False,
        ):
            del require_export

            @contextmanager
            def _ctx():
                with tracer.start_as_current_span(name) as span:
                    span.set_attribute("tenant.id", tenant_id)
                    if project_name is not None:
                        span.set_attribute("project.name", project_name)
                    for key, value in (attributes or {}).items():
                        if value is not None:
                            span.set_attribute(key, value)
                    yield span

            return _ctx()

    return SimpleNamespace(exporter=exporter, manager=_TelemetryManager())


@pytest.fixture(scope="module")
def entity_agent():
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    agent = EntityExtractionAgent(deps=EntityExtractionDeps(), port=19150)
    agent.bind_config_manager(ConfigManager(store=store))
    return agent


def _dead_port_lm() -> dspy.LM:
    return dspy.LM(
        model="openai/dead-port-entity-extraction",
        api_base="http://127.0.0.1:29071/v1",
        api_key="not-required",
    )


class _BarrierDummyLM(DummyLM):
    """Answers by query only once every request is inside the engine call.

    The barrier makes the test fail unless the requests really overlap, and
    the recorded response_format shows each concurrent call carried the
    signature's schema.
    """

    def __init__(self, answers, parties: int):
        super().__init__(answers, adapter=EntityExtractionModule().dspy_adapter)
        self._barrier = threading.Barrier(parties, timeout=60)
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.response_formats: list[dict] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.response_formats.append(kwargs["response_format"])
        try:
            self._barrier.wait()
            return super().__call__(prompt=prompt, messages=messages, **kwargs)
        finally:
            with self._lock:
                self._in_flight -= 1


def _dummy_concurrency_lm() -> DummyLM:
    return _BarrierDummyLM(
        {
            "Barack Obama in Chicago": {
                "reasoning": "Return the exact person and place spans.",
                "entities": _entity_output(
                    ("Barack Obama", "PERSON"),
                    ("Chicago", "PLACE"),
                ),
            },
            "Apple in California": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("Apple", "ORGANIZATION"),
                    ("California", "PLACE"),
                ),
            },
            "PyTorch in Menlo Park": {
                "reasoning": "Return the exact technology and place spans.",
                "entities": _entity_output(
                    ("PyTorch", "TECHNOLOGY"),
                    ("Menlo Park", "PLACE"),
                ),
            },
            "Marie Curie in Paris": {
                "reasoning": "Return the exact person and place spans.",
                "entities": _entity_output(
                    ("Marie Curie", "PERSON"),
                    ("Paris", "PLACE"),
                ),
            },
            "Google in London": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("Google", "ORGANIZATION"),
                    ("London", "PLACE"),
                ),
            },
            "NASA in Florida": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("NASA", "ORGANIZATION"),
                    ("Florida", "PLACE"),
                ),
            },
            "Tesla Model 3 in California": {
                "reasoning": "Return the exact technology and place spans.",
                "entities": _entity_output(
                    ("Tesla Model 3", "TECHNOLOGY"),
                    ("California", "PLACE"),
                ),
            },
            "OpenAI in San Francisco": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("OpenAI", "ORGANIZATION"),
                    ("San Francisco", "PLACE"),
                ),
            },
        },
        parties=8,
    )


@pytest.mark.asyncio
async def test_dead_port_lm_falls_back_to_fast_path_and_emits_span(
    entity_agent,
):
    capture = _telemetry_capture()
    entity_agent.set_telemetry_manager(capture.manager)

    with dspy.context(lm=_dead_port_lm()):
        result = await entity_agent._process_impl(
            EntityExtractionInput(
                query="Barack Obama in Chicago",
                tenant_id="entity-boundary-fallback",
            )
        )

    assert result.model_dump() == {
        "query": "Barack Obama in Chicago",
        "entities": [
            {
                "text": "Barack Obama",
                "type": "PERSON",
                "confidence": 0.9916797280311584,
                "context": "Barack Obama in Chicago",
            },
            {
                "text": "Chicago",
                "type": "PLACE",
                "confidence": 0.9902434945106506,
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

    spans = capture.exporter.get_finished_spans()
    assert [span.name for span in spans] == ["cogniverse.entity_extraction"]
    assert read_span_io(dict(spans[0].attributes)) == {
        "input": "Barack Obama in Chicago",
        "output": {
            "entities": [
                {
                    "text": "Barack Obama",
                    "type": "PERSON",
                    "confidence": 0.9916797280311584,
                    "context": "Barack Obama in Chicago",
                },
                {
                    "text": "Chicago",
                    "type": "PLACE",
                    "confidence": 0.9902434945106506,
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
            "relationship_count": 1,
            "path_used": "fast",
        },
        "operation": "entity_extraction",
        "modality": None,
    }


@pytest.mark.asyncio
async def test_concurrent_requests_stay_on_their_own_queries(entity_agent):
    capture = _telemetry_capture()
    entity_agent.set_telemetry_manager(capture.manager)
    entity_agent._spacy_analyzer = None

    queries = [
        "Barack Obama in Chicago",
        "Apple in California",
        "PyTorch in Menlo Park",
        "Marie Curie in Paris",
        "Google in London",
        "NASA in Florida",
        "Tesla Model 3 in California",
        "OpenAI in San Francisco",
    ]
    expected = {
        "Barack Obama in Chicago": {
            "query": "Barack Obama in Chicago",
            "entities": [
                {
                    "text": "Barack Obama",
                    "type": "PERSON",
                    "confidence": None,
                    "context": "Barack Obama in Chicago",
                },
                {
                    "text": "Chicago",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "Barack Obama in Chicago",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["PERSON", "PLACE"],
            "path_used": "dspy",
        },
        "Apple in California": {
            "query": "Apple in California",
            "entities": [
                {
                    "text": "Apple",
                    "type": "ORGANIZATION",
                    "confidence": None,
                    "context": "Apple in California",
                },
                {
                    "text": "California",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "Apple in California",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["ORGANIZATION", "PLACE"],
            "path_used": "dspy",
        },
        "PyTorch in Menlo Park": {
            "query": "PyTorch in Menlo Park",
            "entities": [
                {
                    "text": "PyTorch",
                    "type": "TECHNOLOGY",
                    "confidence": None,
                    "context": "PyTorch in Menlo Park",
                },
                {
                    "text": "Menlo Park",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "PyTorch in Menlo Park",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["TECHNOLOGY", "PLACE"],
            "path_used": "dspy",
        },
        "Marie Curie in Paris": {
            "query": "Marie Curie in Paris",
            "entities": [
                {
                    "text": "Marie Curie",
                    "type": "PERSON",
                    "confidence": None,
                    "context": "Marie Curie in Paris",
                },
                {
                    "text": "Paris",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "Marie Curie in Paris",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["PERSON", "PLACE"],
            "path_used": "dspy",
        },
        "Google in London": {
            "query": "Google in London",
            "entities": [
                {
                    "text": "Google",
                    "type": "ORGANIZATION",
                    "confidence": None,
                    "context": "Google in London",
                },
                {
                    "text": "London",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "Google in London",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["ORGANIZATION", "PLACE"],
            "path_used": "dspy",
        },
        "NASA in Florida": {
            "query": "NASA in Florida",
            "entities": [
                {
                    "text": "NASA",
                    "type": "ORGANIZATION",
                    "confidence": None,
                    "context": "NASA in Florida",
                },
                {
                    "text": "Florida",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "NASA in Florida",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["ORGANIZATION", "PLACE"],
            "path_used": "dspy",
        },
        "Tesla Model 3 in California": {
            "query": "Tesla Model 3 in California",
            "entities": [
                {
                    "text": "Tesla Model 3",
                    "type": "TECHNOLOGY",
                    "confidence": None,
                    "context": "Tesla Model 3 in California",
                },
                {
                    "text": "California",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "Tesla Model 3 in California",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["TECHNOLOGY", "PLACE"],
            "path_used": "dspy",
        },
        "OpenAI in San Francisco": {
            "query": "OpenAI in San Francisco",
            "entities": [
                {
                    "text": "OpenAI",
                    "type": "ORGANIZATION",
                    "confidence": None,
                    "context": "OpenAI in San Francisco",
                },
                {
                    "text": "San Francisco",
                    "type": "PLACE",
                    "confidence": None,
                    "context": "OpenAI in San Francisco",
                },
            ],
            "relationships": [],
            "entity_count": 2,
            "has_entities": True,
            "dominant_types": ["ORGANIZATION", "PLACE"],
            "path_used": "dspy",
        },
    }

    dummy_lm = _dummy_concurrency_lm()
    with dspy.context(lm=dummy_lm):
        results = await asyncio.gather(
            *(
                entity_agent._process_impl(
                    EntityExtractionInput(
                        query=query,
                        tenant_id=f"entity-concurrency-{index}",
                    )
                )
                for index, query in enumerate(queries)
            )
        )

    assert dummy_lm.max_in_flight == 8
    assert (
        dummy_lm.response_formats
        == [
            signature_response_format(
                EntityExtractionModule().extractor.predict.signature
            )
        ]
        * 8
    )
    result_by_query = {result.query: result for result in results}
    assert result_by_query.keys() == expected.keys()
    for query, expected_result in expected.items():
        assert result_by_query[query].model_dump() == expected_result

    spans = capture.exporter.get_finished_spans()
    assert len(spans) == 8
    span_io_by_query = {
        read_span_io(dict(span.attributes))["input"]: read_span_io(
            dict(span.attributes)
        )
        for span in spans
    }
    assert span_io_by_query.keys() == expected.keys()
    for query, expected_result in expected.items():
        assert span_io_by_query[query] == {
            "input": query,
            "output": {
                "entities": expected_result["entities"],
                "relationships": [],
                "entity_count": 2,
                "relationship_count": 0,
                "path_used": "dspy",
            },
            "operation": "entity_extraction",
            "modality": None,
        }


# The typed entities contract against the deployed engines. The input is the
# committed truth row both models answer exactly: three types, a compound span
# and a post-modifier the span must stop before.
HARD_QUERY = "When does the biker ride the dirt bike in the field?"
CONTRACT_CALLS = 20


def _production_lms():
    """Student and teacher exactly as serving and the optimizer build them,
    with response caching off so every call reaches the engine."""
    from cogniverse_foundation.config.llm_factory import (
        create_budgeted_dspy_lm,
        create_dspy_lm,
    )
    from cogniverse_foundation.config.unified_config import LLMConfig
    from tests.utils.llm_config import _load_config

    llm_config = LLMConfig.from_dict(_load_config()["llm_config"])
    student = create_dspy_lm(llm_config.resolve("entity_extraction_agent"))
    teacher = create_budgeted_dspy_lm(llm_config.resolve_teacher())
    student.cache = False
    teacher.cache = False
    return student, teacher


def _raw_responses(lm) -> list[dict]:
    return [json.loads(entry["outputs"][0]) for entry in lm.history]


def _assert_schema_shaped(responses: list[dict]) -> None:
    assert [list(response) for response in responses] == [
        ["reasoning", "entities"]
    ] * CONTRACT_CALLS
    items = [item for response in responses for item in response["entities"]]
    assert {tuple(sorted(item)) for item in items} == {("text", "type")}
    assert {item["type"] for item in items} <= ENTITY_TYPES


@pytest.mark.asyncio
@pytest.mark.requires_teacher_model
async def test_served_student_returns_the_exact_typed_entities(ensure_host_ollama):
    """The served path on the deployed student: every call parses, none is
    refused, and each answer is the exact validated entity list."""
    from unittest.mock import patch

    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    student, _ = _production_lms()
    store = InMemoryConfigStore()
    store.initialize()
    with patch.object(EntityExtractionAgent, "_initialize_extractors"):
        agent = EntityExtractionAgent(deps=EntityExtractionDeps(), port=19151)
    agent.bind_config_manager(ConfigManager(store=store))
    # No fast path: a DSPy failure raises here instead of being served by GLiNER.
    agent._gliner_extractor = None
    agent._spacy_analyzer = None
    agent.set_telemetry_manager(_telemetry_capture().manager)

    async def _one(index: int):
        try:
            return await agent._process_impl(
                EntityExtractionInput(
                    query=HARD_QUERY, tenant_id=f"entity-contract-{index}"
                )
            )
        except Exception as exc:  # recorded, so every call is counted
            return exc

    with dspy.context(lm=student):
        results = await asyncio.gather(*(_one(i) for i in range(CONTRACT_CALLS)))

    assert [
        type(result).__name__ for result in results if isinstance(result, Exception)
    ] == []
    expected = {
        "query": HARD_QUERY,
        "entities": [
            {
                "text": "biker",
                "type": "PERSON",
                "confidence": None,
                "context": "When does the biker ride the dirt bike in the fie",
            },
            {
                "text": "dirt bike",
                "type": "CONCEPT",
                "confidence": None,
                "context": "When does the biker ride the dirt bike in the field?",
            },
            {
                "text": "field",
                "type": "PLACE",
                "confidence": None,
                "context": "ker ride the dirt bike in the field?",
            },
        ],
        "relationships": [],
        "entity_count": 3,
        "has_entities": True,
        "dominant_types": ["PERSON", "CONCEPT", "PLACE"],
        "path_used": "dspy",
    }
    assert [result.model_dump() for result in results] == [expected] * CONTRACT_CALLS
    assert len(student.history) == CONTRACT_CALLS
    _assert_schema_shaped(_raw_responses(student))


@pytest.mark.requires_teacher_model
def test_bootstrap_teacher_answers_inside_the_typed_schema(ensure_host_ollama):
    """The optimizer's teacher runs the same module: every call parses into
    typed mentions and none is refused.

    The teacher samples at its production temperature, so which entities an
    answer names varies call to call; the bootstrap metric scores that content
    and keeps only exact traces. What the schema guarantees is pinned here.
    """
    _, teacher = _production_lms()

    outcomes = []
    mention_classes = set()
    with dspy.context(lm=teacher):
        for _ in range(CONTRACT_CALLS):
            try:
                prediction = EntityExtractionModule()(query=HARD_QUERY)
            except Exception as exc:  # recorded, so every call is counted
                outcomes.append(type(exc).__name__)
                continue
            outcomes.append("ok")
            mention_classes.update(type(mention) for mention in prediction.entities)

    assert outcomes == ["ok"] * CONTRACT_CALLS
    assert mention_classes == {EntityMention}
    assert len(teacher.history) == CONTRACT_CALLS
    _assert_schema_shaped(_raw_responses(teacher))
