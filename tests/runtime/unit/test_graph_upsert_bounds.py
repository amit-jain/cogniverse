"""/graph/upsert must bound its node/edge lists so one request can't
materialize millions of items and exhaust memory.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cogniverse_runtime.routers.graph import MAX_UPSERT_ITEMS, NodeDoc, UpsertRequest


def _node(i: int) -> dict:
    return {"name": f"name{i}"}


def test_upsert_accepts_lists_within_limit():
    req = UpsertRequest(
        tenant_id="acme:acme",
        source_doc_id="doc-1",
        nodes=[NodeDoc(**_node(i)) for i in range(5)],
    )
    assert len(req.nodes) == 5


def test_upsert_rejects_nodes_over_limit():
    with pytest.raises(ValidationError):
        UpsertRequest(
            tenant_id="acme:acme",
            source_doc_id="doc-1",
            nodes=[NodeDoc(**_node(i)) for i in range(MAX_UPSERT_ITEMS + 1)],
        )


def test_upsert_rejects_edges_over_limit():
    edge = {
        "src_id": "a",
        "dst_id": "b",
        "relation": "R",
        "evidence_span": "e",
        "segment_id": "s",
        "ts_start": 0.0,
        "ts_end": 1.0,
        "modality": "text",
    }
    with pytest.raises(ValidationError):
        UpsertRequest(
            tenant_id="acme:acme",
            source_doc_id="doc-1",
            edges=[edge for _ in range(MAX_UPSERT_ITEMS + 1)],
        )
