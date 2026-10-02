"""``POST /search`` builds a query encoder only for strategies that encode.

The runtime search route is served over a real socket against real Vespa. The
profile names the shipped ColPali inference service; each test sets whether
that service has no URL in this deployment or points at a closed port, so the
outcome shows whether the request ever built or called the encoder, and how an
encoder failure is answered.
"""

from __future__ import annotations

import dataclasses
import json
import socket
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
ENCODER_SERVICE = SHIPPED_PROFILE["inference_services"]["embedding"]
PROFILE = "encres_frames"
TENANT = f"encres{uuid.uuid4().hex[:8]}:unit"
QUERY = "red kayak river"
TEXT_STRATEGY = "bm25_only"
EMBEDDING_STRATEGY = "float_float"
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
BM25_RANKING = ["vid_described", "vid_spoken"]
NOT_CONFIGURED_BODY = {
    "error": "query_encoder_not_configured",
    "dependency": "query_encoder",
    "profile": PROFILE,
    "strategy": EMBEDDING_STRATEGY,
    "message": (
        f"Strategy '{EMBEDDING_STRATEGY}' needs a query encoder, and profile "
        f"'{PROFILE}' has none configured in this deployment."
    ),
}


def _unavailable_body(failure: str) -> dict:
    return {
        "error": "query_encoder_unavailable",
        "dependency": "query_encoder",
        "profile": PROFILE,
        "strategy": EMBEDDING_STRATEGY,
        "service": ENCODER_SERVICE,
        "failure": failure,
        "retry_after_s": 15,
        "message": (
            f"The query encoder for profile '{PROFILE}' is unavailable: "
            f"inference service '{ENCODER_SERVICE}' did not serve the request "
            f"({failure}). Retry after 15s."
        ),
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


def _closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


@pytest.fixture(scope="module")
def runtime(shared_vespa):
    """The search route over a tenant whose profile's encoder service has no
    URL in this deployment."""
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
    for video_id, fields in SEGMENTS.items():
        response = requests.post(
            f"http://localhost:{shared_vespa['http_port']}/document/v1/"
            f"content/{schema}/docid/{video_id}_seg_0",
            json={"fields": {"video_id": video_id, "segment_id": 0, **fields}},
            timeout=30,
        )
        assert response.status_code == 200, response.text
    with serve_search_route(config_manager, tenants=[TENANT]) as url:
        yield {"url": url, "config_manager": config_manager}
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture
def encoder_service_down(runtime):
    """Point the profile's encoder service at a closed port for one test."""
    config_manager = runtime["config_manager"]
    unconfigured = config_manager.get_system_config()
    config_manager.set_system_config(
        dataclasses.replace(
            unconfigured,
            inference_service_urls={ENCODER_SERVICE: _closed_port_url()},
        )
    )
    yield
    config_manager.set_system_config(unconfigured)
    QueryEncoderFactory._encoder_cache.clear()


def _search(runtime, strategy: str, *, stream: bool = False) -> requests.Response:
    return requests.post(
        f"{runtime['url']}/search/",
        json={
            "query": QUERY,
            "profile": PROFILE,
            "strategy": strategy,
            "tenant_id": TENANT,
            "top_k": 10,
            "stream": stream,
        },
        timeout=60,
    )


def _source_ids(response: requests.Response) -> list[str]:
    return [result["source_id"] for result in response.json()["results"]]


class TestTextStrategyNeedsNoEncoder:
    def test_bm25_search_succeeds_without_an_encoder_service(self, runtime):
        response = _search(runtime, TEXT_STRATEGY)

        assert response.status_code == 200, response.text
        assert _source_ids(response) == BM25_RANKING
        assert dict(QueryEncoderFactory._encoder_cache) == {}

    def test_concurrent_bm25_searches_never_build_an_encoder(self, runtime):
        threads_count = 12
        barrier = threading.Barrier(threads_count)
        outcomes: list[tuple[int, list[str]]] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            response = _search(runtime, TEXT_STRATEGY)
            outcome = (
                response.status_code,
                _source_ids(response) if response.status_code == 200 else [],
            )
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert outcomes == [(200, BM25_RANKING)] * threads_count
        assert dict(QueryEncoderFactory._encoder_cache) == {}


class TestEncoderFailuresAreTyped:
    def test_unconfigured_encoder_is_a_typed_configuration_error(self, runtime):
        response = _search(runtime, EMBEDDING_STRATEGY)

        assert response.status_code == 500, response.text
        assert response.json() == {"detail": NOT_CONFIGURED_BODY}
        assert "retry-after" not in response.headers
        assert "http" not in response.text

    def test_streamed_search_reports_the_typed_configuration_error(self, runtime):
        response = _search(runtime, EMBEDDING_STRATEGY, stream=True)

        assert response.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events == [
            {"type": "status", "message": "Searching...", "query": QUERY},
            {
                "type": "error",
                "error": NOT_CONFIGURED_BODY["message"],
                "error_type": "EncoderNotConfiguredError",
                "detail": NOT_CONFIGURED_BODY,
            },
        ]

    def test_encoder_service_down_is_a_typed_503(self, runtime, encoder_service_down):
        response = _search(runtime, EMBEDDING_STRATEGY)

        assert response.status_code == 503, response.text
        assert response.headers["retry-after"] == "15"
        assert response.json() == {"detail": _unavailable_body("ConnectionError")}
        assert "127.0.0.1" not in response.text

    def test_concurrent_requests_against_a_down_encoder_are_each_a_typed_503(
        self, runtime, encoder_service_down
    ):
        threads_count = 8
        barrier = threading.Barrier(threads_count)
        outcomes: list[tuple[int, str, dict]] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            response = _search(runtime, EMBEDDING_STRATEGY)
            with lock:
                outcomes.append(
                    (
                        response.status_code,
                        response.headers.get("retry-after", ""),
                        response.json(),
                    )
                )

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert len(outcomes) == threads_count
        for status, retry_after, body in outcomes:
            failure = body["detail"]["failure"]
            assert failure in ("ConnectionError", "CircuitOpenError")
            assert (status, retry_after, body) == (
                503,
                "15",
                {"detail": _unavailable_body(failure)},
            )
