"""Real-Vespa execution coverage for ImageSearchAgent._search_vespa.

Deploys the ``image_colpali_mv`` schema, feeds two ColPali-style image docs
with known multi-vector embeddings, then runs the agent's real Vespa query and
asserts the matching image is retrieved and ranked first. Exercises the schema
name, the ``float_float`` rank profile, the ``query(qt)`` mapped-tensor format,
and the result parse end to end — the path the streaming test (which asserts
only ``results is list``) cannot reach because ``_search_vespa`` swallows a
non-200 into an empty list.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import requests

from cogniverse_agents.image_search_agent import ImageSearchAgent
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "img_rt"
# A query token that aligns with the sunset doc and opposes the cat doc.
# The schema's visual embedding subspace is v[320], so the fed blocks and the
# query(qt) tensor must be 320-dim or the document/v1 feed 400s.
_MATCH = [0.5] * 320
_OPPOSED = [-0.5] * 320


@pytest.fixture(scope="module")
def image_schema(shared_vespa):
    full = deploy_tenant_schema(
        shared_vespa, tenant_id=TENANT, base_schema_name="image_colpali_mv"
    )
    http_port = shared_vespa["http_port"]

    docs = {
        "img_sunset": {
            "image_id": "img_sunset",
            "image_title": "Sunset over mountains",
            "source_url": "s3://corpus/images/sunset.jpg",
            "image_description": "a warm sunset",
            "embedding": {"blocks": {"0": _MATCH}},
        },
        "img_cat": {
            "image_id": "img_cat",
            "image_title": "Cat on a table",
            "source_url": "s3://corpus/images/cat.jpg",
            "image_description": "a grey cat",
            "embedding": {"blocks": {"0": _OPPOSED}},
        },
    }
    for doc_id, fields in docs.items():
        r = requests.post(
            f"http://localhost:{http_port}/document/v1/image/{full}/docid/{doc_id}",
            json={"fields": fields},
            timeout=15,
        )
        assert r.status_code in (200, 201), r.text[:300]

    time.sleep(2)
    yield {"full": full, "http_port": http_port}

    for doc_id in docs:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/image/{full}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


@pytest.fixture
def agent(image_schema):
    a = ImageSearchAgent.__new__(ImageSearchAgent)
    a._tenant_id = TENANT
    a._vespa_endpoint = f"http://localhost:{image_schema['http_port']}"
    return a


def test_schema_name_matches_agent_query_target(image_schema):
    # The agent builds image_colpali_mv_<canonical_tenant> — deploy must
    # produce the same name or every query 404s.
    assert image_schema["full"] == schema_full_name("image_colpali_mv", TENANT)


@pytest.mark.asyncio
async def test_search_vespa_retrieves_and_ranks_matching_image(agent):
    query_embedding = np.array([_MATCH], dtype=np.float32)

    results = await agent._search_vespa(
        query_embedding=query_embedding,
        query_text="sunset over mountains",
        search_mode="semantic",
        limit=5,
        filters=None,
    )

    assert results, "real Vespa image query returned no results"
    ids = [r.image_id for r in results]
    assert "img_sunset" in ids
    # The aligned image must outrank the opposed one.
    assert results[0].image_id == "img_sunset", ids
    assert results[0].relevance_score > 0

    sunset = next(r for r in results if r.image_id == "img_sunset")
    assert sunset.image_url == "s3://corpus/images/sunset.jpg"
    assert sunset.title == "Sunset over mountains"


@pytest.mark.asyncio
async def test_hybrid_search_scores_bm25_match(agent):
    # Hybrid must inject userQuery() so the bm25 second phase sees the query
    # terms; without it every hit's bm25 score is 0.
    query_embedding = np.array([_MATCH], dtype=np.float32)

    results = await agent._search_vespa(
        query_embedding=query_embedding,
        query_text="sunset mountains",
        search_mode="hybrid",
        limit=5,
        filters=None,
    )

    assert results, "hybrid image query returned no results"
    assert results[0].image_id == "img_sunset"
    assert results[0].relevance_score > 0
    assert results[0].relevance_score != 0.0


@pytest.mark.asyncio
async def test_search_images_propagates_vespa_outage():
    # A Vespa outage must surface (500), not be swallowed into an empty 200.
    from types import SimpleNamespace

    a = ImageSearchAgent.__new__(ImageSearchAgent)
    a._tenant_id = TENANT
    a._vespa_endpoint = "http://127.0.0.1:1"  # closed port -> ConnectionError
    a._query_encoder = SimpleNamespace(
        encode=lambda q: np.zeros((1, 320), dtype=np.float32)
    )

    with pytest.raises(requests.RequestException):
        await a.search_images("cat", search_mode="semantic", limit=5)
