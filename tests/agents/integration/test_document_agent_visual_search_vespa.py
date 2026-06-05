"""Real-Vespa coverage for DocumentAgent._search_visual.

Deploys document_visual, feeds ColPali-style page embeddings, and runs the
agent's real visual query — asserting the matching page is retrieved, ranked,
and parsed. Exercises the tenant-scoped schema name, the float_float profile,
the query(qt) mapped-tensor format, and the document_path parse — the path that
previously sent ``str(flatten())`` against a mis-modeled x[1024] tensor and a
nonexistent ``colpali`` profile, always 404'ing to [].
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import requests

from cogniverse_agents.document_agent import DocumentAgent
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "docvis_rt"
_MATCH = [0.1] * 128
_OPPOSED = [-0.1] * 128


@pytest.fixture(scope="module")
def docvisual_schema(shared_vespa):
    full = deploy_tenant_schema(
        shared_vespa, tenant_id=TENANT, base_schema_name="document_visual"
    )
    http_port = shared_vespa["http_port"]

    docs = {
        "doc_chart_p1": {
            "document_id": "doc_chart_p1",
            "document_title": "Quarterly Revenue Chart",
            "document_path": "s3://corpus/docs/revenue.pdf",
            "document_type": "pdf",
            "page_number": 1,
            "page_count": 12,
            "colpali_embedding": {"blocks": {"0": _MATCH}},
        },
        "doc_blank_p1": {
            "document_id": "doc_blank_p1",
            "document_title": "Empty Cover Page",
            "document_path": "s3://corpus/docs/blank.pdf",
            "document_type": "pdf",
            "page_number": 1,
            "page_count": 3,
            "colpali_embedding": {"blocks": {"0": _OPPOSED}},
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
def agent(docvisual_schema):
    from types import SimpleNamespace

    a = DocumentAgent.__new__(DocumentAgent)
    a._vespa_endpoint = f"http://localhost:{docvisual_schema['http_port']}"
    a._tenant_id = TENANT
    # Controlled ColPali page query embedding aligned with the chart page.
    a._query_encoder = SimpleNamespace(
        encode=lambda q: np.array([_MATCH], dtype=np.float32)
    )
    return a


def test_schema_name_matches_agent_query_target(docvisual_schema):
    assert docvisual_schema["full"] == schema_full_name("document_visual", TENANT)


@pytest.mark.asyncio
async def test_search_visual_retrieves_and_parses_page(agent):
    results = await agent._search_visual("quarterly revenue chart", limit=5)

    assert results, "real Vespa document visual query returned no results"
    ids = [r.document_id for r in results]
    assert "doc_chart_p1" in ids
    assert results[0].document_id == "doc_chart_p1", ids

    chart = next(r for r in results if r.document_id == "doc_chart_p1")
    assert chart.document_url == "s3://corpus/docs/revenue.pdf"
    assert chart.title == "Quarterly Revenue Chart"
    assert chart.page_number == 1
    assert chart.document_type == "pdf"
    assert chart.strategy_used == "visual"
    assert chart.relevance_score > 0
