"""Real-Vespa coverage for DocumentAgent._search_text.

Deploys document_text, feeds ColBERT-style doc-text embeddings, and runs the
agent's real text query — asserting the matching document is retrieved, ranked,
and parsed. Exercises the tenant-scoped schema name, the hybrid_float_bm25
profile, the query(qt) mapped-tensor format, and the document_path parse — the
path that previously sent a nonexistent profile + wrong embedding and always
404'd to [].
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import requests

from cogniverse_agents.document_agent import DocumentAgent
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "doctext_rt"
_MATCH = [0.1] * 128
_OPPOSED = [-0.1] * 128


@pytest.fixture(scope="module")
def doctext_schema(shared_vespa):
    full = deploy_tenant_schema(
        shared_vespa, tenant_id=TENANT, base_schema_name="document_text"
    )
    http_port = shared_vespa["http_port"]

    docs = {
        "doc_ml": {
            "document_id": "doc_ml",
            "document_title": "Machine Learning Guide",
            "full_text": "an overview of machine learning algorithms and neural networks",
            "document_path": "s3://corpus/docs/ml.pdf",
            "document_type": "pdf",
            "page_count": 10,
            "embedding": {"blocks": {"0": _MATCH}},
        },
        "doc_cook": {
            "document_id": "doc_cook",
            "document_title": "Cooking Recipes",
            "full_text": "assorted recipes for soups and gardening tips",
            "document_path": "s3://corpus/docs/cook.pdf",
            "document_type": "pdf",
            "page_count": 4,
            "embedding": {"blocks": {"0": _OPPOSED}},
        },
    }
    for doc_id, fields in docs.items():
        r = requests.post(
            f"http://localhost:{http_port}/document/v1/doc/{full}/docid/{doc_id}",
            json={"fields": fields},
            timeout=15,
        )
        assert r.status_code in (200, 201), r.text[:300]

    time.sleep(2)
    yield {"full": full, "http_port": http_port}

    for doc_id in docs:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/doc/{full}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


@pytest.fixture
def agent(doctext_schema):
    from types import SimpleNamespace

    a = DocumentAgent.__new__(DocumentAgent)
    a._vespa_endpoint = f"http://localhost:{doctext_schema['http_port']}"
    a._tenant_id = TENANT
    # Controlled ColBERT query embedding aligned with doc_ml.
    a._text_query_encoder = SimpleNamespace(
        encode=lambda q: np.array([_MATCH], dtype=np.float32)
    )
    return a


def test_schema_name_matches_agent_query_target(doctext_schema):
    assert doctext_schema["full"] == schema_full_name("document_text", TENANT)


@pytest.mark.asyncio
async def test_search_text_retrieves_and_parses_document(agent):
    results = await agent._search_text("machine learning algorithms", limit=5)

    assert results, "real Vespa document text query returned no results"
    ids = [r.document_id for r in results]
    assert "doc_ml" in ids
    assert results[0].document_id == "doc_ml", ids

    ml = next(r for r in results if r.document_id == "doc_ml")
    assert ml.document_url == "s3://corpus/docs/ml.pdf"
    assert ml.title == "Machine Learning Guide"
    assert ml.document_type == "pdf"
    assert ml.strategy_used == "text"
    assert ml.relevance_score > 0
