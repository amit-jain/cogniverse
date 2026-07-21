"""HTTP-level coverage for the POST /graph/upsert success body.

The upsert route's 2xx path — building Node/Edge dataclasses from the DTO,
calling the manager, and serializing ``UpsertResponse`` — was only ever run
behind a full-cluster e2e. A direct coroutine call bypasses request-body
binding and ``response_model`` serialization, so a body-shape regression
(status branch, ``failed_ids`` field) would ship green. These drive the
mounted FastAPI app via ASGITransport so the whole wire contract is asserted:
request parse -> manager call -> exact response body. The 404 tenant guard is
kept as a boundary of the same route.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from cogniverse_runtime.routers import graph as graph_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubGraphManager:
    """Faithful double: ``upsert`` takes an ExtractionResult and returns the
    counts dict the real GraphManager returns (nodes/edges upserted +
    failed_ids)."""

    def __init__(self, counts):
        self._counts = counts
        self.received = {}

    def upsert(self, result):
        self.received = {
            "source_doc_id": result.source_doc_id,
            "node_names": [n.name for n in result.nodes],
            "edge_relations": [e.relation for e in result.edges],
            "node_tenant_ids": [n.tenant_id for n in result.nodes],
        }
        return dict(self._counts)


def _build_app(counts, *, tenant_exists=True, monkeypatch):
    if tenant_exists:

        async def _assert(tenant_id):
            return None
    else:

        async def _assert(tenant_id):
            raise HTTPException(
                status_code=404, detail=f"Tenant '{tenant_id}' not registered"
            )

    monkeypatch.setattr(
        "cogniverse_core.common.tenant_utils.assert_tenant_exists", _assert
    )

    stub = _StubGraphManager(counts)
    built = {}

    def _factory(tenant_id, deploy=True):
        built["tenant_id"] = tenant_id
        built["deploy"] = deploy
        return stub

    orig = graph_router._graph_manager_factory
    graph_router.set_graph_manager_factory(_factory)

    app = FastAPI()
    app.include_router(graph_router.router, prefix="/graph")
    return app, stub, built, orig


async def _post(app, body):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.post("/graph/upsert", json=body)


@pytest.mark.asyncio
async def test_upsert_success_returns_exact_response_body(monkeypatch):
    app, stub, built, orig = _build_app(
        {"nodes_upserted": 1, "edges_upserted": 0, "failed_ids": []},
        monkeypatch=monkeypatch,
    )
    try:
        resp = await _post(
            app,
            {
                "tenant_id": "acme",
                "source_doc_id": "x.py",
                "nodes": [{"name": "Foo"}],
                "edges": [],
            },
        )
    finally:
        graph_router._graph_manager_factory = orig

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "upserted",
        "nodes_upserted": 1,
        "edges_upserted": 0,
        "failed_ids": [],
    }
    # Tenant canonicalized before the manager build; write path deploys.
    assert built == {"tenant_id": "acme:acme", "deploy": True}
    assert stub.received == {
        "source_doc_id": "x.py",
        "node_names": ["Foo"],
        "edge_relations": [],
        "node_tenant_ids": ["acme:acme"],
    }


@pytest.mark.asyncio
async def test_upsert_partial_failure_reports_failed_ids(monkeypatch):
    app, stub, built, orig = _build_app(
        {"nodes_upserted": 1, "edges_upserted": 1, "failed_ids": ["Bar"]},
        monkeypatch=monkeypatch,
    )
    try:
        resp = await _post(
            app,
            {
                "tenant_id": "acme:acme",
                "source_doc_id": "x.py",
                "nodes": [{"name": "Foo"}, {"name": "Bar"}],
                "edges": [
                    {
                        "source": "Foo",
                        "target": "Bar",
                        "relation": "has_part",
                        "evidence_span": "the foo has a bar",
                        "segment_id": "seg-1",
                        "ts_start": 0.0,
                        "ts_end": 1.0,
                        "modality": "text",
                    }
                ],
            },
        )
    finally:
        graph_router._graph_manager_factory = orig

    assert resp.status_code == 200, resp.text
    # A non-empty failed_ids flips the status branch and serializes verbatim.
    assert resp.json() == {
        "status": "partially_upserted",
        "nodes_upserted": 1,
        "edges_upserted": 1,
        "failed_ids": ["Bar"],
    }
    assert stub.received["node_names"] == ["Foo", "Bar"]
    assert stub.received["edge_relations"] == ["has_part"]


@pytest.mark.asyncio
async def test_upsert_unregistered_tenant_returns_404(monkeypatch):
    app, stub, built, orig = _build_app(
        {"nodes_upserted": 1, "edges_upserted": 0, "failed_ids": []},
        tenant_exists=False,
        monkeypatch=monkeypatch,
    )
    try:
        resp = await _post(
            app,
            {
                "tenant_id": "unregistered_xyz",
                "source_doc_id": "x.py",
                "nodes": [{"name": "Foo"}],
                "edges": [],
            },
        )
    finally:
        graph_router._graph_manager_factory = orig

    assert resp.status_code == 404, resp.text
    assert "not registered" in resp.text.lower()
    # The guard fires before any manager build.
    assert built == {}
    assert stub.received == {}
