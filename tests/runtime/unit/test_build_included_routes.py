"""build_included_routes builds every included router's routes up front.

fastapi builds an included router's routes on the first request that matches
against that router. The runtime's lifespan calls this helper so no request
pays for that build.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from fastapi import routing as fastapi_routing
from pydantic import BaseModel

from cogniverse_runtime.main import build_included_routes

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class Payload(BaseModel):
    value: int


def _app(**include_options) -> FastAPI:
    inner = APIRouter()

    @inner.get("/leaf/{leaf_id}")
    def leaf(leaf_id: int) -> Payload:
        return Payload(value=leaf_id)

    outer = APIRouter()

    @outer.post("/items")
    def create(item: Payload) -> Payload:
        return item

    outer.include_router(inner, prefix="/inner")
    app = FastAPI(openapi_url=None)
    app.include_router(outer, prefix="/outer", **include_options)
    return app


@pytest.fixture
def route_builds(monkeypatch):
    builds: list[str] = []
    build = fastapi_routing._populate_api_route_state

    def recorded(route, path, endpoint, **kwargs):
        if isinstance(route, fastapi_routing._EffectiveRouteContext):
            builds.append(path)
        build(route, path, endpoint, **kwargs)

    monkeypatch.setattr(fastapi_routing, "_populate_api_route_state", recorded)
    return builds


def test_builds_nested_routes_so_concurrent_first_requests_build_none(route_builds):
    app = _app()

    assert build_included_routes(app) == 2
    assert sorted(route_builds) == ["/outer/inner/leaf/{leaf_id}", "/outer/items"]

    route_builds.clear()

    async def first_requests():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://app"
        ) as client:
            return await asyncio.gather(
                client.get("/outer/inner/leaf/7"),
                client.post("/outer/items", json={"value": 3}),
                client.get("/unrouted"),
            )

    leaf, item, unrouted = asyncio.run(first_requests())

    assert (leaf.status_code, leaf.json()) == (200, {"value": 7})
    assert (item.status_code, item.json()) == (200, {"value": 3})
    assert unrouted.status_code == 404
    assert route_builds == []


def test_a_route_that_cannot_build_fails_the_call_not_a_later_request():
    app = _app(responses={204: {"model": Payload}})

    with pytest.raises(
        AssertionError, match="^Status code 204 must not have a response body$"
    ):
        build_included_routes(app)
