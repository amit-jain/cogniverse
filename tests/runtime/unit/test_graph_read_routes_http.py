"""HTTP-level coverage for the graph read routes (search / path / neighbors /
stats).

Driving them through the mounted FastAPI app exercises query-param binding and
``response_model`` serialization — a direct function call bypasses both, so a
request-parse or response-shape regression would otherwise ship green. The
``/graph/search`` route had no HTTP-level exerciser of any kind, and the
``/graph/path`` length math (``len(path) - 1``, or ``-1`` for no path) was only
ever run behind a full-cluster e2e. Canonicalization and the read-path
``deploy=False`` wiring are asserted through the stack here too.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_runtime.routers import graph as graph_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubGraphManager:
    def __init__(self):
        self.calls = {}

    def search_nodes(self, q, top_k):
        self.calls["search"] = {"q": q, "top_k": top_k}
        return [
            {"name": "Robot", "kind": "entity", "score": 0.9},
            {"name": "Arm", "kind": "entity", "score": 0.7},
        ]

    def get_path(self, source, target, max_depth):
        self.calls["path"] = {
            "source": source,
            "target": target,
            "max_depth": max_depth,
        }
        if source == "Isolated":
            return None
        return ["Robot", "Arm", "Gripper"]

    def get_neighbors(self, node, depth):
        self.calls["neighbors"] = {"node": node, "depth": depth}
        # Real shape: out_edges/in_edges are raw _visit_edges rows keyed
        # source_node_id/target_node_id/relation — the route passes them through
        # unchanged, so a client reads those keys, never "target"/"relation".
        return {
            "node_id": "kg_node_robot",
            "name": node,
            "out_edges": [
                {
                    "source_node_id": "kg_node_robot",
                    "target_node_id": "kg_node_arm",
                    "relation": "has_part",
                }
            ],
            "in_edges": [],
        }

    def get_stats(self):
        self.calls["stats"] = True
        # Real shape: top_nodes entries are {"node_id", "degree"} (graph_manager
        # get_stats), never {"name", "degree"}.
        return {
            "node_count": 12,
            "edge_count": 30,
            "top_nodes": [{"node_id": "kg_node_robot", "degree": 5}],
        }


@pytest.fixture
def graph_env(monkeypatch):
    async def _noop_assert(tenant_id):
        return None

    monkeypatch.setattr(
        "cogniverse_core.common.tenant_utils.assert_tenant_exists", _noop_assert
    )

    stub = _StubGraphManager()
    built = {}

    def _factory(tenant_id, deploy=True):
        built["tenant_id"] = tenant_id
        built["deploy"] = deploy
        return stub

    orig = graph_router._graph_manager_factory
    graph_router.set_graph_manager_factory(_factory)
    app = FastAPI()
    app.include_router(graph_router.router, prefix="/graph")
    try:
        yield app, stub, built
    finally:
        graph_router._graph_manager_factory = orig


async def _get(app, path, **params):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.get(path, params=params)


@pytest.mark.asyncio
async def test_search_returns_nodes_and_count(graph_env):
    app, stub, built = graph_env
    resp = await _get(app, "/graph/search", tenant_id="acme", q="robot", top_k=5)

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [n["name"] for n in body["nodes"]] == ["Robot", "Arm"]
    # Query params bound through FastAPI, tenant canonicalized, read=deploy False.
    assert stub.calls["search"] == {"q": "robot", "top_k": 5}
    assert built["tenant_id"] == "acme:acme"
    assert built["deploy"] is False


@pytest.mark.asyncio
async def test_search_top_k_over_max_is_rejected(graph_env):
    app, _, _ = graph_env
    resp = await _get(app, "/graph/search", tenant_id="acme", q="robot", top_k=101)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_missing_query_is_rejected(graph_env):
    app, _, _ = graph_env
    resp = await _get(app, "/graph/search", tenant_id="acme")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_path_length_is_hops_not_nodes(graph_env):
    app, stub, _ = graph_env
    resp = await _get(
        app,
        "/graph/path",
        tenant_id="acme",
        source="Robot",
        target="Gripper",
        max_depth=3,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == ["Robot", "Arm", "Gripper"]
    # length is edge count (hops), not node count.
    assert body["length"] == 2
    assert stub.calls["path"]["max_depth"] == 3


@pytest.mark.asyncio
async def test_path_absent_reports_length_minus_one(graph_env):
    app, _, _ = graph_env
    resp = await _get(
        app, "/graph/path", tenant_id="acme", source="Isolated", target="Gripper"
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] is None
    assert body["length"] == -1


@pytest.mark.asyncio
async def test_neighbors_serializes_response_model(graph_env):
    app, stub, _ = graph_env
    resp = await _get(app, "/graph/neighbors", tenant_id="acme", node="Robot", depth=2)

    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == "kg_node_robot"
    assert body["name"] == "Robot"
    assert body["out_edges"] == [
        {
            "source_node_id": "kg_node_robot",
            "target_node_id": "kg_node_arm",
            "relation": "has_part",
        }
    ]
    assert body["in_edges"] == []
    assert stub.calls["neighbors"] == {"node": "Robot", "depth": 2}


@pytest.mark.asyncio
async def test_stats_echoes_canonical_tenant_and_counts(graph_env):
    app, _, _ = graph_env
    resp = await _get(app, "/graph/stats", tenant_id="acme")

    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "acme:acme"
    assert body["node_count"] == 12
    assert body["edge_count"] == 30
    assert body["top_nodes"] == [{"node_id": "kg_node_robot", "degree": 5}]
