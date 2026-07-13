"""Integration test: full SyntheticDataService dispatch for the
query_enhancement optimizer.

QueryEnhancementGenerator is pattern-based (no LM), so this exercises the
request -> registry -> service -> generator -> response flow end-to-end and
asserts the produced examples satisfy the (query -> enhanced_query) contract
that run_simba_optimization merges. A separate generator-direct test proves the
expansion terms are drawn from the sampled content.
"""

import json

import pytest

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.generators import QueryEnhancementGenerator
from cogniverse_synthetic.schemas import (
    QueryEnhancementExampleSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService


@pytest.fixture
def qe_service():
    return SyntheticDataService(
        generator_config=SyntheticGeneratorConfig(tenant_id="test:qe"),
        backend_config=BackendConfig(profiles={}, tenant_id="test:qe"),
    )


@pytest.mark.asyncio
async def test_service_generates_query_enhancement_examples(qe_service):
    request = SyntheticDataRequest(
        tenant_id="test:qe", optimizer="query_enhancement", count=8
    )
    response = await qe_service.generate(request)

    assert response.optimizer == "query_enhancement"
    assert response.schema_name == QueryEnhancementExampleSchema.__name__
    assert response.count == 8
    assert len(response.data) == 8

    for item in response.data:
        for field in (
            "query",
            "enhanced_query",
            "expansion_terms",
            "synonyms",
            "context",
            "confidence",
            "reasoning",
        ):
            assert field in item, f"missing {field} in {item}"
        # The core property SIMBA trains on: the enhancement is a real rewrite.
        assert item["query"]
        assert item["enhanced_query"] != item["query"]
        assert item["enhanced_query"].startswith(item["query"])
        assert len(item["expansion_terms"]) >= 1
        assert 0.7 <= item["confidence"] <= 0.95


@pytest.mark.asyncio
async def test_generator_draws_expansion_terms_from_content():
    """The generator sources topics and expansion terms from the sampled
    content, not just static defaults."""
    generator = QueryEnhancementGenerator()
    sampled = [
        {"title": "transformer attention mechanism", "content_type": "video"},
        {"title": "self-attention encoder decoder", "content_type": "video"},
    ]
    examples = await generator.generate(sampled_content=sampled, target_count=6)

    assert len(examples) == 6
    content_vocab = {
        w.strip(".,:;!?()") for item in sampled for w in item["title"].lower().split()
    }
    for ex in examples:
        assert ex.enhanced_query != ex.query
        assert ex.expansion_terms
        # Every expansion term is a real word pulled from the sampled content.
        assert all(term in content_vocab for term in ex.expansion_terms)


@pytest.mark.asyncio
async def test_service_response_serializes_to_simba_demo_shape(qe_service):
    """Each demo must round-trip into the ``{"query","enhanced_query",...}``
    dict run_simba_optimization unpacks into a dspy.Example, and must NOT be an
    identity pair (which run_simba skips)."""
    request = SyntheticDataRequest(
        tenant_id="test:qe", optimizer="query_enhancement", count=4
    )
    response = await qe_service.generate(request)

    for item in response.data:
        decoded = json.loads(json.dumps(item, default=str))
        assert decoded["query"].strip()
        assert decoded["enhanced_query"].strip()
        # run_simba drops pairs where query == enhanced_query; these must survive.
        assert decoded["query"].strip() != decoded["enhanced_query"].strip()
