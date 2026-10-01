"""The retrieval solver runs each requested strategy through ``POST /search``.

The runtime search router is served over a real socket against real Vespa. Two
text strategies order the seeded videos in opposite ways, so each solver
configuration's ranked videos show which rank profile actually served it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_evaluation.core.solvers import create_retrieval_solver
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_runtime.routers import search
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
SCHEMAS_DIR = Path("configs/schemas")
SHIPPED_PROFILE = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
][BASE_SCHEMA]
PROFILE = "solver_frames"
TENANT = f"solverstrat{uuid.uuid4().hex[:8]}:unit"
QUERY = "red kayak river"
SEGMENTS = {
    "vid_described": {
        "video_title": "clip one",
        "segment_description": "a red kayak drifts down the river",
        "audio_transcript": "hello there",
    },
    "vid_spoken": {
        "video_title": "clip two",
        "segment_description": "a person talks to the camera",
        "audio_transcript": "we packed the kayak for the trip",
    },
}


@pytest.fixture(autouse=True)
def _telemetry_disabled():
    """The search route opens spans; keep them off any collector."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if telemetry_manager_module._telemetry_manager is installed:
        telemetry_manager_module._telemetry_manager = None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def runtime_url(shared_vespa):
    import requests

    # The profile's query encoder points at a closed port: text strategies
    # never encode, so any configuration that reached a visual strategy fails.
    config_manager = make_config_manager(
        shared_vespa,
        inference_service_urls={
            SHIPPED_PROFILE["inference_services"]["embedding"]: (
                f"http://127.0.0.1:{_free_port()}"
            )
        },
    )
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(PROFILE, SHIPPED_PROFILE), tenant_id=TENANT
    )
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT,
        base_schema_name=BASE_SCHEMA,
        config_manager=config_manager,
    )
    for video_id, fields in SEGMENTS.items():
        response = requests.post(
            f"http://localhost:{shared_vespa['http_port']}/document/v1/"
            f"content/{schema}/docid/{video_id}_seg_0",
            json={"fields": {"video_id": video_id, "segment_id": 0, **fields}},
            timeout=30,
        )
        assert response.status_code == 200, response.text

    app = FastAPI()
    app.include_router(search.router, prefix="/search")
    app.dependency_overrides[search.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = lambda: (
        FilesystemSchemaLoader(SCHEMAS_DIR)
    )

    async def _tenant_registered(tenant_id: str) -> None:
        assert tenant_id == TENANT

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    with patch(
        "cogniverse_runtime.routers.search.assert_tenant_exists",
        new=_tenant_registered,
    ):
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn did not start"
        yield f"http://127.0.0.1:{port}"
        server.should_exit = True
        thread.join(timeout=20)


class _State:
    def __init__(self, query: str):
        self.input = {"query": query}
        self.metadata: dict = {}
        self.output = None


@pytest.mark.asyncio
async def test_each_configuration_is_served_by_its_own_strategy(runtime_url):
    solver = create_retrieval_solver(
        profiles=[PROFILE],
        strategies=["bm25_only", "bm25_no_description"],
        config={"runtime_url": runtime_url, "tenant_id": TENANT, "top_k": 10},
    )

    state = await solver(_State(QUERY), generate=None)

    configs = state.metadata["search_results"]
    assert {key: config["success"] for key, config in configs.items()} == {
        f"{PROFILE}_bm25_only": True,
        f"{PROFILE}_bm25_no_description": True,
    }
    assert [r["video_id"] for r in configs[f"{PROFILE}_bm25_only"]["results"]] == [
        "vid_described",
        "vid_spoken",
    ]
    assert [
        r["video_id"] for r in configs[f"{PROFILE}_bm25_no_description"]["results"]
    ] == ["vid_spoken", "vid_described"]
