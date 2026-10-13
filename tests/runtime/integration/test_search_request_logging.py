"""``POST /search`` writes one INFO line per request, carrying its stages.

The runtime search route is served over a real socket against real Vespa. A
search's INFO output from the search path's loggers is the single
``Search completed`` line; everything else it logs is DEBUG.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from pathlib import Path

import pytest
import requests

from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    make_config_manager,
    serve_search_route,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
SHIPPED_PROFILE = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
][BASE_SCHEMA]
PROFILE = "reqlog_frames"
TENANT = f"reqlog{uuid.uuid4().hex[:8]}:unit"
QUERY = "red kayak river"
STRATEGY = "bm25_only"
SEARCH_PATH_LOGGERS = (
    "cogniverse_vespa.search_backend",
    "cogniverse_agents.search.service",
    "cogniverse_core.query.encoders",
)
COMPLETED = re.compile(
    r"\[(?P<cid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\] "
    r"Search completed: (?P<results>\d+) results in (?P<total_ms>\d+\.\d{2})ms "
    r"profile=(?P<profile>\S+) schema=(?P<schema>\S+) strategy=(?P<strategy>\S+) "
    r"encode_ms=(?P<encode_ms>\d+\.\d{2}) vespa_ms=(?P<vespa_ms>\d+\.\d{2}) "
    r"query=(?P<query>.+)"
)
# The connection pool logs each connection it opens, which is not per request.
POOL_GROWTH = re.compile(r"Created new connection conn-[0-9a-f]+")


@pytest.fixture(autouse=True)
def _telemetry_disabled():
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


@pytest.fixture(scope="module")
def runtime(shared_vespa):
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()
    config_manager = make_config_manager(shared_vespa)
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(PROFILE, SHIPPED_PROFILE), tenant_id=TENANT
    )
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT,
        base_schema_name=BASE_SCHEMA,
        config_manager=config_manager,
    )
    for video_id, description in (
        ("vid_described", "a red kayak drifts down the river"),
        ("vid_spoken", "a person talks about the kayak trip"),
    ):
        response = requests.post(
            f"http://localhost:{shared_vespa['http_port']}/document/v1/"
            f"content/{schema}/docid/{video_id}_seg_0",
            json={
                "fields": {
                    "video_id": video_id,
                    "segment_id": 0,
                    "video_title": video_id,
                    "segment_description": description,
                }
            },
            timeout=30,
        )
        assert response.status_code == 200, response.text
    with serve_search_route(config_manager, tenants=[TENANT]) as url:
        yield {"url": url, "schema": schema}
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()


def _search(runtime, strategy: str = STRATEGY) -> requests.Response:
    return requests.post(
        f"{runtime['url']}/search/",
        json={
            "query": QUERY,
            "profile": PROFILE,
            "strategy": strategy,
            "tenant_id": TENANT,
            "top_k": 10,
        },
        timeout=60,
    )


def _search_path_records(caplog, level: int) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name in SEARCH_PATH_LOGGERS and record.levelno == level
    ]


def test_a_search_logs_one_info_line_with_its_stage_timings(runtime, caplog):
    assert _search(runtime).status_code == 200  # builds the backend
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        response = _search(runtime)

    assert response.status_code == 200, response.text
    info = _search_path_records(caplog, logging.INFO)
    assert [record.name for record in info] == ["cogniverse_vespa.search_backend"]
    line = COMPLETED.fullmatch(info[0].getMessage())
    assert line, info[0].getMessage()
    assert {
        key: line[key] for key in ("results", "profile", "schema", "strategy", "query")
    } == {
        "results": "2",
        "profile": PROFILE,
        "schema": runtime["schema"],
        "strategy": STRATEGY,
        "query": repr(QUERY),
    }
    assert line["encode_ms"] == "0.00"
    assert float(line["vespa_ms"]) <= float(line["total_ms"])
    debug = [
        record.getMessage()
        for record in _search_path_records(caplog, logging.DEBUG)
        if record.getMessage().startswith(f"[{line['cid']}] ")
    ]
    assert [message.split("] ", 1)[1].split(":")[0] for message in debug[:5]] == [
        "Using requested profile",
        "Applied tenant scoping",
        "Using requested strategy",
        "Query resolved",
        "Search request",
    ]


def test_concurrent_searches_each_log_their_own_line(runtime, caplog):
    assert _search(runtime).status_code == 200
    caplog.clear()
    searches = 8
    barrier = threading.Barrier(searches)
    statuses: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        status = _search(runtime).status_code
        with lock:
            statuses.append(status)

    with caplog.at_level(logging.INFO):
        threads = [threading.Thread(target=worker) for _ in range(searches)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

    assert statuses == [200] * searches
    lines = [
        COMPLETED.fullmatch(record.getMessage())
        for record in _search_path_records(caplog, logging.INFO)
        if not POOL_GROWTH.fullmatch(record.getMessage())
    ]
    assert [bool(line) for line in lines] == [True] * searches
    assert len({line["cid"] for line in lines}) == searches
    assert {line["query"] for line in lines} == {repr(QUERY)}


def test_a_rejected_search_logs_no_completed_line(runtime, caplog):
    with caplog.at_level(logging.INFO):
        response = _search(runtime, strategy="no_such_strategy")

    assert response.status_code == 400, response.text
    assert _search_path_records(caplog, logging.INFO) == []
