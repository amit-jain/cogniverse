"""Real boundary tests for EntityExtractionAgent routing and telemetry."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_foundation.telemetry.span_contract import read_span_io

pytestmark = pytest.mark.integration


def _entity_output(*rows: tuple[str, str, float]) -> str:
    return "\n".join(
        f"{text}|{entity_type}|{confidence}" for text, entity_type, confidence in rows
    )


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
    return EntityExtractionAgent(deps=EntityExtractionDeps(), port=19150)


def _dead_port_lm() -> dspy.LM:
    return dspy.LM(
        model="openai/dead-port-entity-extraction",
        api_base="http://127.0.0.1:29071/v1",
        api_key="not-required",
    )


def _dummy_concurrency_lm() -> DummyLM:
    return DummyLM(
        {
            "Barack Obama in Chicago": {
                "reasoning": "Return the exact person and place spans.",
                "entities": _entity_output(
                    ("Barack Obama", "PERSON", 0.95),
                    ("Chicago", "PLACE", 0.9),
                ),
            },
            "Apple in California": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("Apple", "ORGANIZATION", 0.95),
                    ("California", "PLACE", 0.9),
                ),
            },
            "PyTorch in Menlo Park": {
                "reasoning": "Return the exact technology and place spans.",
                "entities": _entity_output(
                    ("PyTorch", "TECHNOLOGY", 0.95),
                    ("Menlo Park", "PLACE", 0.9),
                ),
            },
            "Marie Curie in Paris": {
                "reasoning": "Return the exact person and place spans.",
                "entities": _entity_output(
                    ("Marie Curie", "PERSON", 0.95),
                    ("Paris", "PLACE", 0.9),
                ),
            },
            "Google in London": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("Google", "ORGANIZATION", 0.95),
                    ("London", "PLACE", 0.9),
                ),
            },
            "NASA in Florida": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("NASA", "ORGANIZATION", 0.95),
                    ("Florida", "PLACE", 0.9),
                ),
            },
            "Tesla Model 3 in California": {
                "reasoning": "Return the exact technology and place spans.",
                "entities": _entity_output(
                    ("Tesla Model 3", "TECHNOLOGY", 0.95),
                    ("California", "PLACE", 0.9),
                ),
            },
            "OpenAI in San Francisco": {
                "reasoning": "Return the exact organization and place spans.",
                "entities": _entity_output(
                    ("OpenAI", "ORGANIZATION", 0.95),
                    ("San Francisco", "PLACE", 0.9),
                ),
            },
        }
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
        },
        "Apple in California": {
            "query": "Apple in California",
            "entities": [
                {
                    "text": "Apple",
                    "type": "ORGANIZATION",
                    "confidence": 0.95,
                    "context": "Apple in California",
                },
                {
                    "text": "California",
                    "type": "PLACE",
                    "confidence": 0.9,
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
                    "confidence": 0.95,
                    "context": "PyTorch in Menlo Park",
                },
                {
                    "text": "Menlo Park",
                    "type": "PLACE",
                    "confidence": 0.9,
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
                    "confidence": 0.95,
                    "context": "Marie Curie in Paris",
                },
                {
                    "text": "Paris",
                    "type": "PLACE",
                    "confidence": 0.9,
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
                    "confidence": 0.95,
                    "context": "Google in London",
                },
                {
                    "text": "London",
                    "type": "PLACE",
                    "confidence": 0.9,
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
                    "confidence": 0.95,
                    "context": "NASA in Florida",
                },
                {
                    "text": "Florida",
                    "type": "PLACE",
                    "confidence": 0.9,
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
                    "confidence": 0.95,
                    "context": "Tesla Model 3 in California",
                },
                {
                    "text": "California",
                    "type": "PLACE",
                    "confidence": 0.9,
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
                    "confidence": 0.95,
                    "context": "OpenAI in San Francisco",
                },
                {
                    "text": "San Francisco",
                    "type": "PLACE",
                    "confidence": 0.9,
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
