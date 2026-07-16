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


@pytest.mark.asyncio
async def test_upsert_route_returns_502_when_every_feed_fails(monkeypatch):
    """A total feed failure must not report HTTP 200 'upserted' with zero
    counts — the route surfaces 502 so the caller retries instead of losing the
    batch silently."""
    from fastapi import HTTPException

    from cogniverse_runtime.routers import graph as graph_router

    async def _noop_assert(tenant_id):
        return None

    monkeypatch.setattr(
        "cogniverse_core.common.tenant_utils.assert_tenant_exists", _noop_assert
    )

    class _Mgr:
        def upsert(self, result):
            return {
                "nodes_upserted": 0,
                "edges_upserted": 0,
                "failed_ids": ["kg_node_x"],
            }

    monkeypatch.setattr(graph_router, "get_graph_manager", lambda tid: _Mgr())

    req = UpsertRequest(
        tenant_id="acme:acme",
        source_doc_id="doc-1",
        nodes=[NodeDoc(name="alpha")],
    )
    with pytest.raises(HTTPException) as exc:
        await graph_router.upsert(req)
    assert exc.value.status_code == 502
    assert "kg_node_x" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_upsert_route_reports_partial_status(monkeypatch):
    from cogniverse_runtime.routers import graph as graph_router

    async def _noop_assert(tenant_id):
        return None

    monkeypatch.setattr(
        "cogniverse_core.common.tenant_utils.assert_tenant_exists", _noop_assert
    )

    class _Mgr:
        def upsert(self, result):
            return {
                "nodes_upserted": 1,
                "edges_upserted": 0,
                "failed_ids": ["kg_node_y"],
            }

    monkeypatch.setattr(graph_router, "get_graph_manager", lambda tid: _Mgr())

    req = UpsertRequest(
        tenant_id="acme:acme",
        source_doc_id="doc-1",
        nodes=[NodeDoc(name="alpha"), NodeDoc(name="beta")],
    )
    resp = await graph_router.upsert(req)
    assert resp.status == "partially_upserted"
    assert resp.failed_ids == ["kg_node_y"]
