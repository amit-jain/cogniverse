"""The runtime builds its included routers' routes before it serves.

fastapi builds an included router's routes on the first request that matches
against that router. A fresh runtime process boots the real ``main.app``
through its lifespan and records every route build, so the first requests to
walk all the routers show whether they still build any.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
WALKERS = 8
UNROUTED_PATH = "/__no-router-serves-this__"


async def _boot_and_walk(result_path: str) -> None:
    """Child process: count route builds at startup and on the first requests."""
    from fastapi import routing as fastapi_routing
    from fastapi.routing import APIRoute, iter_route_contexts

    builds: list[str] = []
    build = fastapi_routing._populate_api_route_state

    def recorded(route, path, endpoint, **kwargs):
        if isinstance(route, fastapi_routing._EffectiveRouteContext):
            builds.append(path)
        build(route, path, endpoint, **kwargs)

    fastapi_routing._populate_api_route_state = recorded

    from cogniverse_runtime.main import app, lifespan

    async with lifespan(app):
        at_startup = sorted(builds)
        builds.clear()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            responses = await asyncio.gather(
                *(client.get(UNROUTED_PATH) for _ in range(WALKERS))
            )
        on_first_requests = sorted(builds)
        direct = {id(route) for route in app.routes if isinstance(route, APIRoute)}
        included = sorted(
            context.path
            for context in iter_route_contexts(app.routes)
            if isinstance(context.original_route, APIRoute)
            and id(context.original_route) not in direct
        )

    Path(result_path).write_text(
        json.dumps(
            {
                "at_startup": at_startup,
                "on_first_requests": on_first_requests,
                "statuses": [response.status_code for response in responses],
                "included": included,
            }
        )
    )


def test_the_first_requests_after_startup_build_no_routes(
    workflow_state_redis_url, tmp_path
):
    result_path = tmp_path / "route_builds.json"
    env = {
        **os.environ,
        "REDIS_URL": workflow_state_redis_url,
        "COGNIVERSE_SANDBOX_POLICY": "disabled",
        "COGNIVERSE_MEMORY_LIFECYCLE_DISABLED": "1",
    }
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys\n"
            "from tests.runtime.integration.test_routes_built_at_startup "
            "import _boot_and_walk\n"
            "asyncio.run(_boot_and_walk(sys.argv[1]))",
            str(result_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stderr[-4000:]
    result = json.loads(result_path.read_text())

    assert result["statuses"] == [404] * WALKERS
    assert result["on_first_requests"] == []
    assert result["at_startup"] == result["included"]
    assert {"/admin/harness/keys", "/v1/chat/completions"} <= set(result["included"])
