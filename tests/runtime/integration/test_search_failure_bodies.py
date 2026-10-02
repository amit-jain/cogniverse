"""``POST /search`` answers failures with typed bodies, never exception text.

The runtime search route is served over a real socket. A configuration store
at a dead port fails every profile read with an error whose text names the
store's URL and document path; a healthy store over real Vespa rejects an
unknown profile. The answers carry stable codes and route-owned values only,
and the cause reaches the runtime log.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import uuid

import pytest
import requests

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.vespa_test_helpers import make_config_manager, serve_search_route

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

PROFILE = "video_colpali_smol500_mv_frame"
TENANT = f"failbody{uuid.uuid4().hex[:8]}:unit"


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


def _dead_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def dead_store_runtime():
    """The route over a configuration store nothing listens on."""
    port = _dead_port()
    config_manager = ConfigManager(
        store=VespaConfigStore(backend_url="http://localhost", backend_port=port)
    )
    with serve_search_route(config_manager, tenants=[TENANT]) as url:
        yield url, port


@pytest.fixture(scope="module")
def healthy_runtime(shared_vespa):
    BackendRegistry.clear_instances()
    with serve_search_route(make_config_manager(shared_vespa), tenants=[TENANT]) as url:
        yield url
    BackendRegistry.clear_instances()


def _search(url: str, **body) -> requests.Response:
    payload = {"query": "red kayak", "tenant_id": TENANT, **body}
    return requests.post(f"{url}/search/", json=payload, timeout=120)


def _failed_body(profile: str) -> dict:
    return {
        "error": "search_failed",
        "message": (
            f"Search with profile '{profile}' failed; the runtime log names the cause."
        ),
        "failure": "RuntimeError",
        "profile": profile,
        "strategy": "bm25_only",
    }


class TestBackendOutage:
    def test_store_outage_is_a_typed_500_and_the_cause_is_logged(
        self, dead_store_runtime, caplog
    ):
        url, port = dead_store_runtime

        with caplog.at_level(logging.ERROR, logger="cogniverse_runtime.http_errors"):
            response = _search(url, profile=PROFILE, strategy="bm25_only")

        assert response.status_code == 500, response.text
        assert response.json() == {"detail": _failed_body(PROFILE)}
        assert f":{port}" not in response.text
        assert "/document/v1" not in response.text
        causes = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.http_errors"
        ]
        assert len(causes) == 1
        assert causes[0].startswith(
            f"search_failed: RuntimeError: Reading backend profile '{PROFILE}' "
            f"for tenant '{TENANT}' failed: "
        )
        assert f"port={port}" in causes[0]

    def test_streamed_store_outage_ends_in_a_typed_error_event(
        self, dead_store_runtime
    ):
        url, port = dead_store_runtime

        response = _search(url, profile=PROFILE, strategy="bm25_only", stream=True)

        assert response.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        body = _failed_body(PROFILE)
        assert events == [
            {"type": "status", "message": "Searching...", "query": "red kayak"},
            {
                "type": "error",
                "error": body["message"],
                "error_type": "RuntimeError",
                "detail": body,
            },
        ]
        assert f":{port}" not in response.text


class TestRejectedRequests:
    def test_missing_tenant_is_a_typed_400(self, dead_store_runtime):
        url, _ = dead_store_runtime

        response = requests.post(
            f"{url}/search/", json={"query": "red kayak"}, timeout=30
        )

        assert response.status_code == 400
        assert response.json() == {
            "detail": {
                "error": "invalid_tenant_id",
                "message": (
                    "tenant_id is required on a search request, as "
                    "'<org>:<tenant>' or '<tenant>'."
                ),
                "failure": "ValueError",
                "tenant_id": None,
            }
        }

    def test_concurrent_unknown_profiles_each_name_their_own_profile(
        self, healthy_runtime
    ):
        profiles = [f"no_such_profile_{i}" for i in range(8)]
        barrier = threading.Barrier(len(profiles))
        answers: dict[str, tuple[int, dict]] = {}
        lock = threading.Lock()

        def worker(profile: str) -> None:
            barrier.wait()
            response = _search(healthy_runtime, profile=profile, strategy="bm25_only")
            with lock:
                answers[profile] = (response.status_code, response.json())

        threads = [threading.Thread(target=worker, args=(p,)) for p in profiles]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert answers == {
            profile: (
                400,
                {
                    "detail": {
                        "error": "invalid_search_request",
                        "message": (
                            f"Search with profile '{profile}' and strategy "
                            "'bm25_only' was rejected; GET /search/profiles and "
                            "GET /search/strategies list what this tenant accepts."
                        ),
                        "failure": "ValueError",
                        "profile": profile,
                        "strategy": "bm25_only",
                    }
                },
            )
            for profile in profiles
        }
