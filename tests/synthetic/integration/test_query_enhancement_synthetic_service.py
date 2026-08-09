"""Integration test: full SyntheticDataService dispatch for the
query_enhancement optimizer.

This exercises the request -> registry -> service -> production agent ->
generator -> response flow and asserts the produced examples satisfy the
(query -> enhanced_query) contract consumed by query-enhancement optimization.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

from cogniverse_agents.query_enhancement_agent import (
    QueryEnhancementAgent,
    QueryEnhancementDeps,
    QueryEnhancementInput,
)
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
)
from cogniverse_synthetic.generators import QueryEnhancementGenerator
from cogniverse_synthetic.schemas import (
    QueryEnhancementExampleSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.utils.synthetic_config import video_synthetic_generator_config
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qe_service(shared_vespa):
    tenant_id = f"synquery{uuid.uuid4().hex[:8]}:media"
    profile_name = "video_colpali_smol500_mv_frame"
    title = "transformer attention mechanism"
    description = "encoder decoder architecture improves context windows"
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
        data_id="transformer-attention-segment",
        fields={
            "video_id": "transformer-attention",
            "video_title": title,
            "source_url": "http://example.test/transformer-attention",
            "segment_id": 0,
            "segment_description": description,
            "start_time": 0.0,
            "end_time": 8.0,
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
        pytest.fail("Transformer source document was not indexed by Vespa")

    enhancement_agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())
    enhancement_agent.dspy_module.enhancer = lambda query: dspy.Prediction(
        enhanced_query=f"{query} encoder decoder architecture",
        expansion_terms="encoder, decoder, architecture",
        synonyms="",
        context="",
        confidence="0.91",
        reasoning="Production enhancement preserved the exact source terms.",
    )

    async def enhance_query(query: str, request_tenant_id: str):
        assert request_tenant_id == tenant_id
        return await enhancement_agent.process(
            QueryEnhancementInput(query=query, tenant_id=request_tenant_id)
        )

    agents_config = json.loads(Path("configs/config.json").read_text())["agents"]
    service = SyntheticDataService(
        backend=backend,
        generator_config=video_synthetic_generator_config(tenant_id),
        backend_config=backend_config,
        agents_config=agents_config,
        query_enhancer=enhance_query,
    )
    try:
        yield SimpleNamespace(
            service=service,
            tenant_id=tenant_id,
            profile_name=profile_name,
            title=title,
            expansion_terms=["encoder", "decoder", "architecture"],
            agents_config=agents_config,
        )
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_service_generates_query_enhancement_examples(qe_service):
    request = SyntheticDataRequest(
        tenant_id=qe_service.tenant_id,
        optimizer="query_enhancement",
        count=5,
        vespa_sample_size=1,
        max_profiles=1,
    )
    response = await qe_service.service.generate(request)

    assert response.optimizer == "query_enhancement"
    assert response.schema_name == QueryEnhancementExampleSchema.__name__
    assert response.count == 5
    assert len(response.data) == 5
    assert response.selected_profiles == [qe_service.profile_name]
    assert response.metadata["sampled_content_count"] == 1

    possible_queries = {
        qe_service.title,
        f"find {qe_service.title}",
        f"show me {qe_service.title}",
        f"{qe_service.title} tutorial",
        f"explain {qe_service.title}",
    }
    assert [item["query"] for item in response.data] == [
        qe_service.title,
        f"find {qe_service.title}",
        f"show me {qe_service.title}",
        f"{qe_service.title} tutorial",
        f"explain {qe_service.title}",
    ]

    for item in response.data:
        assert item["query"] in possible_queries
        assert item["enhanced_query"] == (
            f"{item['query']} {' '.join(qe_service.expansion_terms)}"
        )
        assert item["expansion_terms"] == qe_service.expansion_terms
        assert item["synonyms"] == []
        assert item["context"] == qe_service.profile_name
        assert "confidence" not in item
        assert item["reasoning"] == (
            "Production enhancement preserved the exact source terms."
        )


@pytest.mark.requires_lm
@pytest.mark.asyncio
async def test_real_lm_query_agent_rejects_non_source_expansion_terms(
    qe_service,
    ensure_host_ollama,
    dspy_test_lm,
):
    _ = ensure_host_ollama
    agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())

    async def enhance_query(query, tenant_id):
        return await agent.process(
            QueryEnhancementInput(query=query, tenant_id=tenant_id)
        )

    service = SyntheticDataService(
        backend=qe_service.service.backend,
        generator_config=qe_service.service.generator_config,
        backend_config=qe_service.service.backend_config,
        agents_config=qe_service.agents_config,
        query_enhancer=enhance_query,
    )
    with pytest.raises(ValueError) as raised:
        with dspy.context(lm=dspy_test_lm):
            await service.generate(
                SyntheticDataRequest(
                    tenant_id=qe_service.tenant_id,
                    optimizer="query_enhancement",
                    count=1,
                    vespa_sample_size=1,
                    max_profiles=1,
                )
            )

    assert str(raised.value) == (
        "query_enhancement optimizer callback query_enhancer returned "
        "expansion_terms absent from sampled source for "
        f"tenant={qe_service.tenant_id!r} "
        "query='transformer attention mechanism': ['self-attention', "
        "'multi-head attention', 'feed-forward network', "
        "'encoder-decoder structure', 'sequence modeling']"
    )


@pytest.mark.asyncio
async def test_generator_keeps_expansion_terms_with_their_source_item():
    async def enhance_query(query: str, tenant_id: str):
        assert tenant_id == "acme:synthetic"
        terms = (
            ["encoder", "decoder", "architecture"]
            if "transformer attention" in query
            else ["chemistry", "isolation", "laboratory"]
        )
        return {
            "original_query": query,
            "enhanced_query": f"{query} {' '.join(terms)}",
            "expansion_terms": terms,
            "synonyms": [],
            "reasoning": "Production enhancement returned exact grounded terms.",
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)
    sampled = [
        {
            "title": "transformer attention",
            "description": "encoder decoder architecture",
            "content_type": "video",
        },
        {
            "title": "radium discovery",
            "description": "chemistry isolation laboratory",
            "content_type": "document",
        },
    ]
    examples = await generator.generate(
        sampled_content=sampled,
        target_count=10,
        tenant_id="acme:synthetic",
    )

    assert len(examples) == 10
    assert len({example.query for example in examples}) == 10
    expected_terms = {
        "transformer attention": {
            "terms": ["encoder", "decoder", "architecture"],
            "context": "video",
            "unrelated": {"chemistry", "isolation", "laboratory"},
        },
        "radium discovery": {
            "terms": ["chemistry", "isolation", "laboratory"],
            "context": "document",
            "unrelated": {"encoder", "decoder", "architecture"},
        },
    }
    for ex in examples:
        matched_topics = [
            topic
            for topic in expected_terms
            if ex.query
            in {
                topic,
                f"find {topic}",
                f"show me {topic}",
                f"{topic} tutorial",
                f"explain {topic}",
            }
        ]
        assert len(matched_topics) == 1
        topic = matched_topics[0]
        expected = expected_terms[topic]
        assert ex.expansion_terms == expected["terms"]
        assert ex.enhanced_query == f"{ex.query} {' '.join(expected['terms'])}"
        assert set(ex.expansion_terms).isdisjoint(expected["unrelated"])
        assert ex.synonyms == []
        assert ex.context == expected["context"]
        assert "confidence" not in ex.model_dump()
        assert ex.reasoning == ("Production enhancement returned exact grounded terms.")


@pytest.mark.asyncio
async def test_generator_rejects_count_above_unique_grounded_query_capacity():
    calls = []

    async def unexpected_enhancement(query: str, tenant_id: str):
        calls.append((query, tenant_id))
        raise AssertionError("capacity must be checked before enhancement")

    with pytest.raises(ValueError) as error:
        await QueryEnhancementGenerator(query_enhancer=unexpected_enhancement).generate(
            sampled_content=[
                {
                    "title": "transformer attention",
                    "description": "encoder decoder architecture",
                    "content_type": "video",
                }
            ],
            target_count=6,
            tenant_id="acme:synthetic",
        )

    assert calls == []
    assert str(error.value) == (
        "QueryEnhancementGenerator generated 5 unique grounded examples but "
        "target_count=6; source_context=5 unique source-template queries"
    )


@pytest.mark.asyncio
async def test_generator_rejects_topic_without_source_expansion_terms():
    async def unexpected_enhancement(query: str, tenant_id: str):
        pytest.fail(f"unexpected enhancement call: {query}, {tenant_id}")

    generator = QueryEnhancementGenerator(query_enhancer=unexpected_enhancement)

    with pytest.raises(
        ValueError,
        match=(
            "sampled_content contains no expansion terms outside topic "
            "'transformer attention'"
        ),
    ):
        await generator.generate(
            sampled_content=[
                {"title": "transformer attention", "content_type": "video"}
            ],
            target_count=1,
            tenant_id="acme:synthetic",
        )


@pytest.mark.asyncio
async def test_service_response_serializes_to_simba_demo_shape(qe_service):
    """Each demo must round-trip into the ``{"query","enhanced_query",...}``
    dict run_simba_optimization unpacks into a dspy.Example, and must NOT be an
    identity pair (which run_simba skips)."""
    request = SyntheticDataRequest(
        tenant_id=qe_service.tenant_id,
        optimizer="query_enhancement",
        count=4,
        vespa_sample_size=1,
        max_profiles=1,
    )
    response = await qe_service.service.generate(request)

    for item in response.data:
        decoded = json.loads(json.dumps(item, default=str))
        assert decoded["enhanced_query"] == (
            f"{decoded['query']} {' '.join(qe_service.expansion_terms)}"
        )
        assert decoded["expansion_terms"] == qe_service.expansion_terms
