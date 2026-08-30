from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Event, Thread
from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import modal
import pytest
from click.testing import CliRunner
from cogniverse_cli.inference_endpoints import (
    EndpointCredentials,
    ResolvedInferenceEndpoint,
)
from cogniverse_cli.main import cli
from cogniverse_cli.modal_inference_lifecycle import (
    ModalInferenceLifecycle,
    ModalLifecycleError,
    QualificationResult,
    ServiceStatus,
)

from cogniverse_foundation.inference_specs import get_inference_service_spec

API_KEY = "modal-lifecycle-secret"
COLPALI = get_inference_service_spec("vllm_colpali")
DENSEON = get_inference_service_spec("denseon")


class _ModalFunction:
    def __init__(
        self,
        web_url: str,
        *,
        active_containers: int = 0,
        scale_up_error: str | None = None,
        release_error: str | None = None,
        status_error: str | None = None,
    ) -> None:
        self.web_url = web_url
        self.active_containers = active_containers
        self.scale_up_error = scale_up_error
        self.release_error = release_error
        self.status_error = status_error
        self.autoscaler_updates: list[dict[str, int]] = []

    def update_autoscaler(
        self,
        *,
        min_containers: int,
        scaledown_window: int,
    ) -> None:
        self.autoscaler_updates.append(
            {
                "min_containers": min_containers,
                "scaledown_window": scaledown_window,
            }
        )
        if min_containers == 0 and self.release_error is not None:
            raise RuntimeError(self.release_error)
        self.active_containers = min_containers
        if min_containers == 1 and self.scale_up_error is not None:
            raise RuntimeError(self.scale_up_error)

    def get_web_url(self) -> str:
        return self.web_url

    def get_current_stats(self):
        if self.status_error is not None:
            raise RuntimeError(self.status_error)
        return SimpleNamespace(
            backlog=0,
            num_total_runners=self.active_containers,
            num_running_inputs=0,
        )


class _ModalApp:
    def __init__(self) -> None:
        self.deploy_calls: list[dict[str, str]] = []

    def deploy(self, *, name: str) -> None:
        self.deploy_calls.append({"name": name})


class _FailingModalApp:
    def __init__(self) -> None:
        self.deploy_calls: list[dict[str, str]] = []

    def deploy(self, *, name: str) -> None:
        self.deploy_calls.append({"name": name})
        raise RuntimeError("controlled build failure")


class _CanonicalModalTransport(httpx.BaseTransport):
    def __init__(self, routes: dict[str, str]) -> None:
        self._routes = routes
        self._transport = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        target = httpx.URL(self._routes[request.url.host]).copy_with(
            path=request.url.path
        )
        if request.url.query:
            target = target.copy_with(query=request.url.query)
        proxied = httpx.Request(
            request.method,
            target,
            headers=request.headers,
            content=request.content,
        )
        return self._transport.handle_request(proxied)

    def close(self) -> None:
        self._transport.close()


def _modal_client(*routes: tuple[str, str]) -> httpx.Client:
    return httpx.Client(
        transport=_CanonicalModalTransport(
            {httpx.URL(canonical).host: target for canonical, target in routes}
        ),
        timeout=2,
    )


@contextmanager
def _inference_server(
    spec,
    *,
    health_status: int = 200,
    health_statuses: tuple[int, ...] = (),
    model_status: int = 200,
    health_started: Event | None = None,
    release_health: Event | None = None,
):
    requests: list[tuple[str, str | None]] = []
    pending_health_statuses = list(health_statuses)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == spec.health_path:
                if health_started is not None:
                    health_started.set()
                if release_health is not None:
                    release_health.wait(timeout=2)
                status = (
                    pending_health_statuses.pop(0)
                    if pending_health_statuses
                    else health_status
                )
                payload = {"status": "ok" if status == 200 else API_KEY}
            elif self.path == spec.models_path:
                status = model_status
                payload = {
                    "data": [
                        {
                            "id": spec.model_id,
                            "revision": spec.model_revision,
                        }
                    ]
                }
            else:
                status = 404
                payload = {"detail": "not found"}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        canonical_url = f"https://{spec.name.replace('_', '-')}.modal.run"
        yield canonical_url, f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _make_lifecycle(
    functions: dict[str, _ModalFunction],
    *,
    client: httpx.Client | None,
    credentials: EndpointCredentials | None = None,
    deployment_loader=None,
    app_stopper=None,
    readiness_timeout: float = 1200,
    readiness_poll_interval: float = 1,
):
    lookup_calls: list[tuple[str, str]] = []
    stop_calls: list[str] = []

    def lookup(app_name: str, object_name: str):
        lookup_calls.append((app_name, object_name))
        for service, spec in (
            ("vllm_colpali", COLPALI),
            ("denseon", DENSEON),
        ):
            if (app_name, object_name) == (spec.modal_app, spec.modal_object):
                return functions[service]
        raise AssertionError((app_name, object_name))

    def stop(app_name: str) -> None:
        stop_calls.append(app_name)

    lifecycle_kwargs = {
        "credentials": credentials or EndpointCredentials(bearer_token=API_KEY),
        "function_from_name": lookup,
        "deployment_loader": deployment_loader,
        "app_stopper": app_stopper or stop,
        "readiness_timeout": readiness_timeout,
        "readiness_poll_interval": readiness_poll_interval,
    }
    if client is not None:
        lifecycle_kwargs["client"] = client
    lifecycle = ModalInferenceLifecycle(
        **lifecycle_kwargs,
    )
    return lifecycle, lookup_calls, stop_calls


def test_deploy_uses_the_canonical_app_name_without_stopping_or_warming():
    function = _ModalFunction("https://colpali.modal.run")
    app = _ModalApp()
    loaded: list[str] = []

    def load(spec):
        loaded.append(spec.name)
        return app

    with httpx.Client(timeout=2) as client:
        lifecycle, lookup_calls, stop_calls = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
            deployment_loader=load,
        )
        result = lifecycle.deploy(["vllm_colpali"])

    assert loaded == ["vllm_colpali"]
    assert app.deploy_calls == [{"name": "cogniverse-vllm-colpali"}]
    assert lookup_calls == [("cogniverse-vllm-colpali", "Inference")]
    assert function.autoscaler_updates == []
    assert stop_calls == []
    assert result == (
        ServiceStatus(
            service="vllm_colpali",
            modal_app="cogniverse-vllm-colpali",
            modal_object="Inference",
            web_url="https://colpali.modal.run",
            active_containers=0,
        ),
    )


def test_deploy_enables_modal_output_for_build_logs(monkeypatch):
    import cogniverse_cli.modal_inference_lifecycle as lifecycle_module

    function = _ModalFunction("https://colpali.modal.run")
    app = _FailingModalApp()
    loaded: list[str] = []
    output_state = SimpleNamespace(entered=0, exited=0)

    class _OutputContext:
        def __enter__(self):
            output_state.entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            output_state.exited += 1
            return False

    def load(spec):
        loaded.append(spec.name)
        return app

    monkeypatch.setattr(lifecycle_module, "modal", modal, raising=False)
    with httpx.Client(timeout=2) as client:
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
            deployment_loader=load,
        )
        with patch.object(
            modal, "enable_output", return_value=_OutputContext()
        ) as enable_output:
            with pytest.raises(
                ModalLifecycleError,
                match="vllm_colpali: Modal deployment failed: controlled build failure",
            ):
                lifecycle.deploy(["vllm_colpali"])

    assert enable_output.call_count == 1
    assert output_state.entered == 1
    assert output_state.exited == 1
    assert loaded == ["vllm_colpali"]
    assert app.deploy_calls == [{"name": "cogniverse-vllm-colpali"}]


def test_warm_retries_cold_health_then_verifies_identity_once_and_releases():
    with _inference_server(
        COLPALI,
        health_statuses=(503, 502, 504, 200),
    ) as (base_url, target_url, requests):
        function = _ModalFunction(base_url)
        with _modal_client((base_url, target_url)) as client:
            lifecycle, _, stop_calls = _make_lifecycle(
                {"vllm_colpali": function},
                client=client,
                readiness_poll_interval=0,
            )
            resolved = lifecycle.warm(["vllm_colpali"])
            fresh_lifecycle, _, _ = _make_lifecycle(
                {"vllm_colpali": function},
                client=client,
            )
            warm_status = fresh_lifecycle.status(["vllm_colpali"])
            released = lifecycle.release(["vllm_colpali"])

    assert requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
    ]
    assert function.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300},
        {"min_containers": 0, "scaledown_window": 300},
    ]
    assert [
        (
            endpoint.service,
            endpoint.provider,
            endpoint.base_url,
            endpoint.model_id,
            endpoint.model_revision,
            dict(endpoint.headers),
        )
        for endpoint in resolved
    ] == [
        (
            "vllm_colpali",
            "modal",
            base_url,
            "TomoroAI/tomoro-colqwen3-embed-4b",
            "bf790bd8780b098b86453444632a184bb770be1a",
            {"Authorization": f"Bearer {API_KEY}"},
        )
    ]
    assert warm_status[0].active_containers == 1
    assert released[0].active_containers == 0
    assert stop_calls == []


def test_warm_retries_timeout_connect_and_gateway_failures_before_identity():
    requests: list[tuple[str, str | None]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, request.headers.get("Authorization")))
        attempt = len(requests)
        if attempt == 1:
            raise httpx.ReadTimeout(
                f"health timed out for {API_KEY}",
                request=request,
            )
        if attempt == 2:
            raise httpx.ConnectError(
                f"health connection failed for {API_KEY}",
                request=request,
            )
        if attempt == 3:
            return httpx.Response(504, json={"detail": API_KEY})
        if request.url.path == COLPALI.health_path:
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": COLPALI.model_id,
                        "revision": COLPALI.model_revision,
                    }
                ]
            },
        )

    function = _ModalFunction("https://colpali.modal.run")
    with httpx.Client(transport=httpx.MockTransport(handle), timeout=2) as client:
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
            readiness_poll_interval=0,
        )
        resolved = lifecycle.warm(["vllm_colpali"])

    assert requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
    ]
    assert [(endpoint.model_id, endpoint.model_revision) for endpoint in resolved] == [
        (
            "TomoroAI/tomoro-colqwen3-embed-4b",
            "bf790bd8780b098b86453444632a184bb770be1a",
        )
    ]


def test_warm_after_release_revalidates_health_and_exact_model_identity():
    with _inference_server(COLPALI) as (base_url, target_url, requests):
        function = _ModalFunction(base_url)
        with _modal_client((base_url, target_url)) as client:
            lifecycle, _, _ = _make_lifecycle(
                {"vllm_colpali": function},
                client=client,
            )
            first = lifecycle.warm(["vllm_colpali"])[0]
            lifecycle.release(["vllm_colpali"])
            second = lifecycle.warm(["vllm_colpali"])[0]

    assert first is not second
    assert requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
    ]
    assert function.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300},
        {"min_containers": 0, "scaledown_window": 300},
        {"min_containers": 1, "scaledown_window": 300},
    ]


def test_concurrent_warm_calls_share_one_autoscaler_update_and_probe():
    with _inference_server(COLPALI) as (base_url, target_url, requests):
        function = _ModalFunction(base_url)
        with _modal_client((base_url, target_url)) as client:
            lifecycle, lookup_calls, _ = _make_lifecycle(
                {"vllm_colpali": function},
                client=client,
            )
            simultaneous_callers = Barrier(8)

            def warm():
                simultaneous_callers.wait(timeout=2)
                return lifecycle.warm(["vllm_colpali"])[0]

            with ThreadPoolExecutor(max_workers=8) as pool:
                resolved = tuple(pool.map(lambda _: warm(), range(8)))

    assert not simultaneous_callers.broken
    assert len({id(endpoint) for endpoint in resolved}) == 1
    assert lookup_calls == [("cogniverse-vllm-colpali", "Inference")]
    assert function.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300}
    ]
    assert requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
    ]


def test_close_waits_for_inflight_warm_then_closes_owned_client():
    health_started = Event()
    release_health = Event()
    with _inference_server(
        COLPALI,
        health_started=health_started,
        release_health=release_health,
    ) as (base_url, target_url, requests):
        function = _ModalFunction(base_url)
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=None,
        )
        lifecycle._client.close()
        lifecycle._client = _modal_client((base_url, target_url))
        owned_client = lifecycle._client
        with ThreadPoolExecutor(max_workers=2) as pool:
            warming = pool.submit(lifecycle.warm, ["vllm_colpali"])
            assert health_started.wait(timeout=2)
            closing = pool.submit(lifecycle.close)
            sleep(0.05)
            assert not closing.done()
            assert not owned_client.is_closed
            release_health.set()
            endpoints = warming.result(timeout=2)
            closing.result(timeout=2)

        assert [
            (endpoint.model_id, endpoint.model_revision) for endpoint in endpoints
        ] == [
            (
                "TomoroAI/tomoro-colqwen3-embed-4b",
                "bf790bd8780b098b86453444632a184bb770be1a",
            )
        ]
        assert owned_client.is_closed
        lifecycle.close()
        with pytest.raises(
            ModalLifecycleError,
            match="Modal inference lifecycle is closed",
        ):
            lifecycle.status(["vllm_colpali"])

    assert requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
    ]


def test_close_does_not_close_injected_client():
    with _inference_server(COLPALI) as (base_url, target_url, requests):
        function = _ModalFunction(base_url)
        with _modal_client((base_url, target_url)) as client:
            lifecycle, _, _ = _make_lifecycle(
                {"vllm_colpali": function},
                client=client,
            )
            lifecycle.warm(["vllm_colpali"])
            lifecycle.close()
            lifecycle.close()
            assert not client.is_closed
            response = client.get(
                f"{base_url}{COLPALI.models_path}",
                headers={"Authorization": "Bearer direct-key"},
            )

    assert response.status_code == 200
    assert requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
        ("/v1/models", "Bearer direct-key"),
    ]


def test_owned_client_close_failure_leaves_lifecycle_retryable():
    class FailsOnceClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise OSError("controlled lifecycle client close failure")

    lifecycle, _, _ = _make_lifecycle({}, client=None)
    lifecycle._client.close()
    failing_client = FailsOnceClient()
    lifecycle._client = failing_client

    with pytest.raises(OSError, match="controlled lifecycle client close failure"):
        lifecycle.close()

    assert lifecycle._closed is False
    assert lifecycle._closing is False
    lifecycle.close()
    assert failing_client.close_calls == 2
    assert lifecycle._closed is True


def test_context_closes_owned_client_after_real_health_fault():
    with _inference_server(COLPALI, health_status=500) as (
        base_url,
        target_url,
        requests,
    ):
        function = _ModalFunction(base_url)
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=None,
        )
        lifecycle._client.close()
        lifecycle._client = _modal_client((base_url, target_url))
        owned_client = lifecycle._client
        with pytest.raises(
            ModalLifecycleError,
            match="authenticated health probe returned HTTP 500",
        ):
            with lifecycle:
                lifecycle.warm(["vllm_colpali"])

    assert owned_client.is_closed
    assert function.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300},
        {"min_containers": 0, "scaledown_window": 300},
    ]
    assert requests == [("/health", f"Bearer {API_KEY}")]


def test_ambiguous_scale_up_failure_is_compensated_to_zero_and_redacted():
    boundary_requests: list[httpx.Request] = []

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        boundary_requests.append(request)
        return httpx.Response(500, request=request)

    function = _ModalFunction(
        "https://colpali.modal.run",
        scale_up_error=f"autoscaler acknowledgement lost for {API_KEY}",
    )
    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        lifecycle, lookup_calls, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
        )
        with pytest.raises(ModalLifecycleError) as caught:
            lifecycle.warm(["vllm_colpali"])

    assert str(caught.value) == (
        "vllm_colpali: Modal autoscaler update to min_containers=1 failed: "
        "autoscaler acknowledgement lost for [redacted]"
    )
    assert function.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300},
        {"min_containers": 0, "scaledown_window": 300},
    ]
    assert function.active_containers == 0
    assert lookup_calls == [
        ("cogniverse-vllm-colpali", "Inference"),
        ("cogniverse-vllm-colpali", "Inference"),
    ]
    assert boundary_requests == []


def test_concurrent_ambiguous_scale_up_failures_all_compensate_to_zero():
    function = _ModalFunction(
        "https://colpali.modal.run",
        scale_up_error="autoscaler acknowledgement lost",
    )
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"unexpected HTTP request: {request.url}")
        )
    ) as client:
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
        )
        simultaneous_callers = Barrier(8)

        def warm() -> str:
            simultaneous_callers.wait(timeout=2)
            with pytest.raises(ModalLifecycleError) as caught:
                lifecycle.warm(["vllm_colpali"])
            return str(caught.value)

        with ThreadPoolExecutor(max_workers=8) as pool:
            errors = tuple(pool.map(lambda _: warm(), range(8)))

    assert not simultaneous_callers.broken
    assert (
        errors
        == (
            "vllm_colpali: Modal autoscaler update to min_containers=1 failed: "
            "autoscaler acknowledgement lost",
        )
        * 8
    )
    assert [
        update
        for update in function.autoscaler_updates
        if update["min_containers"] == 1
    ] == [{"min_containers": 1, "scaledown_window": 300}] * 8
    assert [
        update
        for update in function.autoscaler_updates
        if update["min_containers"] == 0
    ] == [{"min_containers": 0, "scaledown_window": 300}] * 8
    assert function.active_containers == 0


def test_probe_failure_releases_all_warmed_services_and_redacts_cleanup_error():
    with (
        _inference_server(COLPALI) as (
            colpali_url,
            colpali_target,
            colpali_requests,
        ),
        _inference_server(DENSEON, health_status=503) as (
            denseon_url,
            denseon_target,
            denseon_requests,
        ),
    ):
        colpali = _ModalFunction(
            colpali_url,
            release_error=f"release denied for {API_KEY}",
        )
        denseon = _ModalFunction(denseon_url)
        with _modal_client(
            (colpali_url, colpali_target),
            (denseon_url, denseon_target),
        ) as client:
            lifecycle, _, stop_calls = _make_lifecycle(
                {"vllm_colpali": colpali, "denseon": denseon},
                client=client,
                readiness_timeout=0.001,
                readiness_poll_interval=1,
            )
            with pytest.raises(ModalLifecycleError) as caught:
                lifecycle.warm(["vllm_colpali", "denseon"])

    assert str(caught.value) == (
        "denseon: authenticated health did not become ready within 0.001 "
        "seconds; last failure: HTTP 503; "
        "release cleanup failed: vllm_colpali: Modal autoscaler update to "
        "min_containers=0 failed: release denied for [redacted]"
    )
    assert API_KEY not in str(caught.value)
    assert colpali.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300},
        {"min_containers": 0, "scaledown_window": 300},
    ]
    assert denseon.autoscaler_updates == [
        {"min_containers": 1, "scaledown_window": 300},
        {"min_containers": 0, "scaledown_window": 300},
    ]
    assert colpali_requests == [
        ("/health", f"Bearer {API_KEY}"),
        ("/v1/models", f"Bearer {API_KEY}"),
    ]
    assert denseon_requests == [("/health", f"Bearer {API_KEY}")]
    assert stop_calls == []


def test_missing_credentials_fail_before_allocating_a_warm_container():
    function = _ModalFunction("https://colpali.modal.run")
    with httpx.Client(timeout=2) as client:
        lifecycle, lookup_calls, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
            credentials=EndpointCredentials(),
        )
        with pytest.raises(ModalLifecycleError) as caught:
            lifecycle.warm(["vllm_colpali"])

    assert str(caught.value) == (
        "vllm_colpali: endpoint credentials are invalid: "
        "bearer authentication requires a configured API key"
    )
    assert lookup_calls == []
    assert function.autoscaler_updates == []


@pytest.mark.parametrize(
    ("credentials", "raw_secrets", "boundary_error", "expected_error"),
    [
        (
            EndpointCredentials(bearer_token=API_KEY),
            (API_KEY,),
            f"stats denied for {API_KEY}",
            "stats denied for [redacted]",
        ),
        (
            EndpointCredentials(
                modal_key="modal-proxy-key",
                modal_secret="modal-proxy-secret",
            ),
            ("modal-proxy-key", "modal-proxy-secret"),
            "stats denied for modal-proxy-key/modal-proxy-secret",
            "stats denied for [redacted]/[redacted]",
        ),
    ],
)
def test_status_suppresses_raw_secrets_from_formatted_boundary_tracebacks(
    credentials,
    raw_secrets,
    boundary_error,
    expected_error,
):
    function = _ModalFunction("https://colpali.modal.run", status_error=boundary_error)
    with httpx.Client(timeout=2) as client:
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
            credentials=credentials,
        )
        with pytest.raises(ModalLifecycleError) as caught:
            lifecycle.status(["vllm_colpali"])

    assert str(caught.value) == (
        f"vllm_colpali: failed to read Modal function stats: {expected_error}"
    )
    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert all(secret not in formatted for secret in raw_secrets)


def test_qualification_selects_the_earliest_configured_supplied_gpu():
    function = _ModalFunction("https://colpali.modal.run")
    with httpx.Client(timeout=2) as client:
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
        )
        result = lifecycle.qualify("vllm_colpali", ["A10", "L4"])

    assert result == QualificationResult(
        service="vllm_colpali",
        selected_gpu="L4",
        considered_gpus=("L4", "A10"),
    )


def test_qualification_rejects_unknown_or_empty_gpu_candidates():
    function = _ModalFunction("https://colpali.modal.run")
    with httpx.Client(timeout=2) as client:
        lifecycle, _, _ = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
        )
        with pytest.raises(
            ValueError,
            match="vllm_colpali: at least one GPU candidate is required",
        ):
            lifecycle.qualify("vllm_colpali", [])
        with pytest.raises(
            ValueError,
            match=r"vllm_colpali: unsupported GPU candidates \['T4'\]",
        ):
            lifecycle.qualify("vllm_colpali", ["T4"])


def test_undeploy_requires_byte_exact_service_confirmation():
    function = _ModalFunction("https://colpali.modal.run")
    with httpx.Client(timeout=2) as client:
        lifecycle, _, stop_calls = _make_lifecycle(
            {"vllm_colpali": function},
            client=client,
        )
        with pytest.raises(ValueError) as caught:
            lifecycle.undeploy("vllm_colpali", "VLLM_COLPALI")
        lifecycle.undeploy("vllm_colpali", "vllm_colpali")

    assert str(caught.value) == (
        "undeploy confirmation must exactly match service 'vllm_colpali'"
    )
    assert stop_calls == ["cogniverse-vllm-colpali"]


class _ClosingLifecycle:
    def __init__(self, *, warm_error: str | None = None) -> None:
        self.calls: list[str] = []
        self.close_calls = 0
        self.warm_error = warm_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.close_calls += 1

    def _status(self) -> tuple[ServiceStatus, ...]:
        return (
            ServiceStatus(
                service="vllm_colpali",
                modal_app="cogniverse-vllm-colpali",
                modal_object="Inference",
                web_url="https://colpali.modal.run",
                active_containers=1,
            ),
        )

    def deploy(self, services):
        self.calls.append("deploy")
        return self._status()

    def warm(self, services):
        self.calls.append("warm")
        if self.warm_error is not None:
            raise ModalLifecycleError(self.warm_error)
        return (
            ResolvedInferenceEndpoint(
                service="vllm_colpali",
                provider="modal",
                base_url="https://colpali.modal.run",
                headers={},
                model_id=COLPALI.model_id,
                model_revision=COLPALI.model_revision,
            ),
        )

    def release(self, services):
        self.calls.append("release")
        return self._status()

    def status(self, services):
        self.calls.append("status")
        return self._status()

    def qualify(self, service, candidates):
        self.calls.append("qualify")
        return QualificationResult(
            service=service,
            selected_gpu="L4",
            considered_gpus=("L4",),
        )

    def undeploy(self, service, confirmation) -> None:
        self.calls.append("undeploy")


@pytest.mark.parametrize(
    ("arguments", "expected_calls"),
    [
        (["inference", "modal", "deploy", "vllm_colpali"], ["deploy"]),
        (
            ["inference", "modal", "warm", "vllm_colpali"],
            ["warm", "status"],
        ),
        (["inference", "modal", "release", "vllm_colpali"], ["release"]),
        (["inference", "modal", "status", "vllm_colpali"], ["status"]),
        (
            [
                "inference",
                "modal",
                "qualify",
                "vllm_colpali",
                "--gpu",
                "L4",
            ],
            ["qualify"],
        ),
        (
            [
                "inference",
                "modal",
                "undeploy",
                "vllm_colpali",
                "--confirm-service",
                "vllm_colpali",
            ],
            ["undeploy"],
        ),
    ],
)
def test_cli_commands_close_their_lifecycle(
    monkeypatch,
    arguments,
    expected_calls,
):
    lifecycle = _ClosingLifecycle()
    monkeypatch.setattr(
        "cogniverse_cli.main._build_modal_inference_lifecycle",
        lambda: lifecycle,
    )

    result = CliRunner().invoke(cli, arguments)

    assert result.exit_code == 0, result.output
    assert lifecycle.calls == expected_calls
    assert lifecycle.close_calls == 1


def test_cli_closes_lifecycle_when_warm_fails(monkeypatch):
    lifecycle = _ClosingLifecycle(warm_error="vllm_colpali: readiness failed")
    monkeypatch.setattr(
        "cogniverse_cli.main._build_modal_inference_lifecycle",
        lambda: lifecycle,
    )

    result = CliRunner().invoke(
        cli,
        ["inference", "modal", "warm", "vllm_colpali"],
    )

    assert result.exit_code == 1
    assert result.output == "Error: vllm_colpali: readiness failed\n"
    assert lifecycle.calls == ["warm"]
    assert lifecycle.close_calls == 1
