import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Event, Thread
from time import sleep

import httpx
import pytest
from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointAuthenticationError,
    EndpointContractError,
    EndpointCredentials,
    EndpointIdentityEvidence,
    EndpointResolutionError,
    EndpointResolver,
    EndpointServerError,
    EndpointTimeoutError,
    EndpointUnavailableError,
    ModelIdentityError,
    resolve_endpoint,
)
from cogniverse_cli.modal_inference_config import get_inference_service_spec

SPEC = get_inference_service_spec("vllm_colpali")
REVISION = SPEC.model_revision


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _models_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"id": SPEC.model_id, "revision": REVISION}]},
        request=request,
    )


@contextmanager
def _model_server(
    *,
    delay: float = 0.0,
    arrival_barrier: Barrier | None = None,
    request_started: Event | None = None,
    release_response: Event | None = None,
):
    requests: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            if request_started is not None:
                request_started.set()
            if arrival_barrier is not None:
                arrival_barrier.wait(timeout=2)
            if release_response is not None:
                release_response.wait(timeout=2)
            if delay:
                sleep(delay)
            payload = json.dumps(
                {"data": [{"id": SPEC.model_id, "revision": REVISION}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_modal_endpoint_requires_exact_bearer_auth_and_model_revision():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://inference.modal.run/v1/models"
        assert request.headers["Authorization"] == "Bearer modal-key"
        return _models_response(request)

    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run/",
        credentials=EndpointCredentials(bearer_token="modal-key"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )

    resolved = EndpointResolver(_client(handler)).resolve(SPEC, explicit=candidate)

    assert resolved.service == "vllm_colpali"
    assert resolved.provider == "modal"
    assert resolved.base_url == "https://inference.modal.run"
    assert resolved.model_id == SPEC.model_id
    assert resolved.model_revision == REVISION
    assert dict(resolved.headers) == {"Authorization": "Bearer modal-key"}


def test_live_non_modal_boundary_preserves_its_own_endpoint_credentials():
    with _model_server() as (base_url, requests):
        candidate = CandidateEndpoint(
            provider="local",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token="local-endpoint-key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )

        with httpx.Client(timeout=2) as client:
            resolved = EndpointResolver(client).resolve(SPEC, explicit=candidate)

    assert requests == [("/v1/models", "Bearer local-endpoint-key")]
    assert resolved == resolved.__class__(
        service="vllm_colpali",
        provider="local",
        base_url=base_url,
        headers=resolved.headers,
        model_id="TomoroAI/tomoro-colqwen3-embed-4b",
        model_revision="bf790bd8780b098b86453444632a184bb770be1a",
    )
    assert dict(resolved.headers) == {"Authorization": "Bearer local-endpoint-key"}


@pytest.mark.parametrize(
    "candidate_url",
    (
        "http://service.modal.run",
        "https://modal.run",
        "https://service.modal.run.evil.example",
        "https://inference.example",
        "http://127.0.0.1:39001",
    ),
)
def test_modal_provider_rejects_insecure_or_untrusted_hosts_before_resolution(
    candidate_url,
):
    sent_requests: list[httpx.Request] = []

    def record_request(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(500, request=request)

    with httpx.Client(transport=httpx.MockTransport(record_request)) as client:
        with EndpointResolver(client) as resolver:
            with pytest.raises(
                ValueError,
                match=r"Modal endpoint must be an HTTPS \*\.modal\.run root URL",
            ):
                resolver.resolve(
                    SPEC,
                    explicit=CandidateEndpoint(
                        provider="modal",
                        base_url=candidate_url,
                        credentials=EndpointCredentials(
                            bearer_token="must-not-be-transmitted"
                        ),
                    ),
                )

    assert sent_requests == []


def test_cache_key_includes_the_complete_pinned_model_contract():
    updated_revision = "1" * 40
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        revision = REVISION if not calls else updated_revision
        calls.append(revision)
        return httpx.Response(
            200,
            json={"data": [{"id": SPEC.model_id, "revision": revision}]},
            request=request,
        )

    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=EndpointCredentials(bearer_token="key"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )
    resolver = EndpointResolver(_client(handler))

    first = resolver.resolve(SPEC, explicit=candidate)
    second = resolver.resolve(
        replace(SPEC, model_revision=updated_revision),
        explicit=candidate,
    )

    assert calls == [REVISION, updated_revision]
    assert (first.model_revision, second.model_revision) == (
        REVISION,
        updated_revision,
    )


def test_explicit_wrong_model_fails_without_trying_other_candidates():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={"data": [{"id": "wrong/model", "revision": REVISION}]},
            request=request,
        )

    explicit = CandidateEndpoint(
        provider="modal",
        base_url="https://explicit.modal.run",
        credentials=EndpointCredentials(bearer_token="key"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )
    unused = CandidateEndpoint(
        provider="e2e",
        base_url="http://127.0.0.1:33901",
        credentials=EndpointCredentials(bearer_token="e2e-key"),
        model_revision=REVISION,
        identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
    )

    with pytest.raises(ModelIdentityError, match="expected .*TomoroAI"):
        EndpointResolver(_client(handler)).resolve(
            SPEC,
            explicit=explicit,
            candidates=[unused],
        )

    assert calls == ["https://explicit.modal.run/v1/models"]


def test_authentication_failure_is_not_treated_as_unavailability():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(401, json={"detail": "invalid key"}, request=request)

    modal = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=EndpointCredentials(bearer_token="wrong"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )
    e2e = CandidateEndpoint(
        provider="e2e",
        base_url="http://e2e.example",
        credentials=EndpointCredentials(bearer_token="e2e-key"),
        model_revision=REVISION,
        identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
    )

    with pytest.raises(EndpointAuthenticationError, match="vllm_colpali.*401"):
        EndpointResolver(_client(handler)).resolve(SPEC, candidates=[e2e, modal])

    assert calls == ["inference.modal.run"]


def test_only_connection_refusal_advances_through_provider_order():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host in {"inference.modal.run", "e2e.example"}:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200,
            json={"data": [{"id": SPEC.model_id}]},
            request=request,
        )

    candidates = [
        CandidateEndpoint(
            provider="dev",
            base_url="http://dev.example",
            credentials=EndpointCredentials(bearer_token="dev-key"),
            model_revision=REVISION,
            identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
        ),
        CandidateEndpoint(
            provider="e2e",
            base_url="http://e2e.example",
            credentials=EndpointCredentials(bearer_token="e2e-key"),
            model_revision=REVISION,
            identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
        ),
        CandidateEndpoint(
            provider="modal",
            base_url="https://inference.modal.run",
            credentials=EndpointCredentials(bearer_token="key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        ),
    ]

    resolved = EndpointResolver(_client(handler)).resolve(SPEC, candidates=candidates)

    assert calls == ["inference.modal.run", "e2e.example", "dev.example"]
    assert resolved.provider == "dev"


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            lambda request: httpx.Response(500, text="model crashed", request=request),
            EndpointServerError,
        ),
        (
            lambda request: httpx.Response(200, text="not-json", request=request),
            EndpointContractError,
        ),
    ],
)
def test_server_and_contract_failures_stop_resolution(response, error):
    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=EndpointCredentials(bearer_token="key"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )

    with pytest.raises(error):
        EndpointResolver(_client(response)).resolve(SPEC, candidates=[candidate])


def test_deployment_identity_must_be_pinned_before_http_validation():
    candidate = CandidateEndpoint(
        provider="e2e",
        base_url="http://e2e.example",
        credentials=EndpointCredentials(bearer_token="e2e-key"),
        model_revision="wrong-revision",
        identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
    )

    with pytest.raises(ModelIdentityError, match="deployment revision"):
        EndpointResolver(_client(_models_response)).resolve(
            SPEC,
            candidates=[candidate],
        )


def test_modal_candidate_rejects_deployment_only_identity_evidence():
    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=EndpointCredentials(bearer_token="key"),
        model_revision=REVISION,
        identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
    )

    with pytest.raises(
        ModelIdentityError,
        match="modal.*must report its pinned revision",
    ):
        EndpointResolver(
            _client(
                lambda request: httpx.Response(
                    200,
                    json={"data": [{"id": SPEC.model_id}]},
                    request=request,
                )
            )
        ).resolve(SPEC, explicit=candidate)


def test_concurrent_resolution_performs_one_boundary_validation():
    with _model_server(delay=0.05) as (base_url, requests):
        candidate = CandidateEndpoint(
            provider="e2e",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token="key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
        with httpx.Client(timeout=2) as client:
            resolver = EndpointResolver(client)
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = tuple(
                    pool.map(
                        lambda _: resolver.resolve(SPEC, explicit=candidate), range(8)
                    )
                )

    assert requests == [("/v1/models", "Bearer key")]
    assert {result.base_url for result in results} == {base_url}


def test_concurrent_distinct_candidates_validate_without_global_serialization():
    concurrent_requests = Barrier(2)
    with (
        _model_server(arrival_barrier=concurrent_requests) as first,
        _model_server(arrival_barrier=concurrent_requests) as second,
    ):
        candidates = (
            CandidateEndpoint(
                provider="e2e",
                base_url=first[0],
                credentials=EndpointCredentials(bearer_token="first-key"),
                identity_evidence=EndpointIdentityEvidence.ENDPOINT,
            ),
            CandidateEndpoint(
                provider="e2e",
                base_url=second[0],
                credentials=EndpointCredentials(bearer_token="second-key"),
                identity_evidence=EndpointIdentityEvidence.ENDPOINT,
            ),
        )
        with httpx.Client(timeout=2) as client:
            resolver = EndpointResolver(client)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = tuple(
                    pool.map(
                        lambda candidate: resolver.resolve(SPEC, explicit=candidate),
                        candidates,
                    )
                )

    assert first[1] == [("/v1/models", "Bearer first-key")]
    assert second[1] == [("/v1/models", "Bearer second-key")]
    assert not concurrent_requests.broken
    assert {result.base_url for result in results} == {
        first[0],
        second[0],
    }


def test_concurrent_fault_is_shared_by_all_waiters_without_retrying_boundary():
    simultaneous_callers = Barrier(8)
    with _model_server(delay=0.2) as (base_url, requests):
        candidate = CandidateEndpoint(
            provider="e2e",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token="key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
        with httpx.Client(timeout=0.05) as client:
            resolver = EndpointResolver(client)

            def resolve() -> str:
                simultaneous_callers.wait(timeout=2)
                with pytest.raises(EndpointTimeoutError) as caught:
                    resolver.resolve(SPEC, explicit=candidate)
                return str(caught.value)

            with ThreadPoolExecutor(max_workers=8) as pool:
                errors = tuple(pool.map(lambda _: resolve(), range(8)))

    assert requests == [("/v1/models", "Bearer key")]
    assert not simultaneous_callers.broken
    assert set(errors) == {"vllm_colpali: e2e model identity request timed out"}


def test_success_cache_is_bounded_lru_across_distinct_credentials():
    with _model_server() as (base_url, requests):
        candidates = tuple(
            CandidateEndpoint(
                provider="e2e",
                base_url=base_url,
                credentials=EndpointCredentials(bearer_token=f"key-{index}"),
                identity_evidence=EndpointIdentityEvidence.ENDPOINT,
            )
            for index in range(17)
        )
        with EndpointResolver() as resolver:
            for candidate in candidates[:16]:
                resolver.resolve(SPEC, explicit=candidate)
            resolver.resolve(SPEC, explicit=candidates[0])
            resolver.resolve(SPEC, explicit=candidates[16])
            resolver.resolve(SPEC, explicit=candidates[0])
            resolver.resolve(SPEC, explicit=candidates[1])

    assert requests == [
        ("/v1/models", f"Bearer key-{index}") for index in range(16)
    ] + [
        ("/v1/models", "Bearer key-16"),
        ("/v1/models", "Bearer key-1"),
    ]


def test_close_waits_for_inflight_resolution_then_rejects_new_work():
    request_started = Event()
    release_response = Event()
    with _model_server(
        request_started=request_started,
        release_response=release_response,
    ) as (base_url, requests):
        candidate = CandidateEndpoint(
            provider="e2e",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token="key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
        resolver = EndpointResolver()
        owned_client = resolver._client
        with ThreadPoolExecutor(max_workers=2) as pool:
            resolution = pool.submit(resolver.resolve, SPEC, explicit=candidate)
            assert request_started.wait(timeout=2)
            closing = pool.submit(resolver.close)
            sleep(0.05)
            assert not closing.done()
            assert not owned_client.is_closed
            release_response.set()
            resolved = resolution.result(timeout=2)
            closing.result(timeout=2)

        assert resolved.base_url == base_url
        assert owned_client.is_closed
        resolver.close()
        with pytest.raises(
            EndpointResolutionError,
            match="endpoint resolver is closed",
        ):
            resolver.resolve(SPEC, explicit=candidate)

    assert requests == [("/v1/models", "Bearer key")]


def test_close_does_not_close_injected_client():
    with _model_server() as (base_url, requests):
        candidate = CandidateEndpoint(
            provider="e2e",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token="key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
        with httpx.Client(timeout=2) as client:
            resolver = EndpointResolver(client)
            resolver.resolve(SPEC, explicit=candidate)
            resolver.close()
            resolver.close()
            assert not client.is_closed
            response = client.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer direct-key"},
            )

    assert response.status_code == 200
    assert requests == [
        ("/v1/models", "Bearer key"),
        ("/v1/models", "Bearer direct-key"),
    ]


def test_owned_client_close_failure_leaves_resolver_retryable():
    class FailsOnceClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise OSError("controlled endpoint client close failure")

    resolver = EndpointResolver()
    resolver._client.close()
    failing_client = FailsOnceClient()
    resolver._client = failing_client

    with pytest.raises(OSError, match="controlled endpoint client close failure"):
        resolver.close()

    assert resolver._closed is False
    assert resolver._closing is False
    resolver.close()
    assert failing_client.close_calls == 2
    assert resolver._closed is True


def test_one_shot_resolution_closes_its_owned_real_http_client(monkeypatch):
    created_clients: list[httpx.Client] = []
    httpx_client = httpx.Client

    class TrackingClient(httpx_client):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created_clients.append(self)

    with _model_server() as (base_url, requests):
        candidate = CandidateEndpoint(
            provider="local",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token="one-shot-key"),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
        monkeypatch.setattr(
            "cogniverse_cli.inference_endpoints.httpx.Client",
            TrackingClient,
        )

        resolved = resolve_endpoint(SPEC, explicit=candidate)

    assert resolved.base_url == base_url
    assert requests == [("/v1/models", "Bearer one-shot-key")]
    assert len(created_clients) == 1
    assert created_clients[0].is_closed


def test_timeout_is_not_cached_and_the_next_resolution_revalidates():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("model listing timed out", request=request)
        return _models_response(request)

    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=EndpointCredentials(bearer_token="key"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )
    resolver = EndpointResolver(_client(handler))

    with pytest.raises(
        EndpointTimeoutError,
        match="vllm_colpali: modal model identity request timed out",
    ):
        resolver.resolve(SPEC, explicit=candidate)
    resolved = resolver.resolve(SPEC, explicit=candidate)

    assert attempts == 2
    assert (resolved.model_id, resolved.model_revision) == (
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "bf790bd8780b098b86453444632a184bb770be1a",
    )


def test_mixed_or_blank_credentials_are_rejected_before_http_request():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _models_response(request)

    resolver = EndpointResolver(_client(handler))
    errors: list[str] = []
    for credentials in (
        EndpointCredentials(bearer_token="   "),
        EndpointCredentials(
            bearer_token="canonical-key",
            modal_key="proxy-key",
            modal_secret="proxy-secret",
        ),
    ):
        candidate = CandidateEndpoint(
            provider="modal",
            base_url="https://inference.modal.run",
            credentials=credentials,
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
        with pytest.raises(EndpointAuthenticationError) as caught:
            resolver.resolve(SPEC, explicit=candidate)
        errors.append(str(caught.value))

    assert calls == 0
    assert errors == [
        "vllm_colpali: modal bearer authentication requires a configured API key",
        "vllm_colpali: modal bearer authentication rejects Modal proxy credentials",
    ]


@pytest.mark.parametrize(
    "invalid_url",
    [
        "modal.example",
        "ftp://modal.example",
        "https://user:secret@modal.example",
        "https://modal.example?model=wrong",
        "https://modal.example#fragment",
        "https://modal.example/v1",
        "http://[",
    ],
)
def test_candidate_rejects_noncanonical_base_urls(invalid_url):
    with pytest.raises(ValueError, match="base_url"):
        CandidateEndpoint(provider="modal", base_url=invalid_url)


def test_candidate_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider"):
        CandidateEndpoint(provider="staging", base_url="https://model.example")


def test_explicit_connection_failure_is_contextual():
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=EndpointCredentials(bearer_token="key"),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )

    with pytest.raises(
        EndpointUnavailableError,
        match="vllm_colpali: explicit modal endpoint refused a connection",
    ):
        EndpointResolver(_client(unavailable)).resolve(SPEC, explicit=candidate)


def test_credentials_never_appear_in_representations_or_errors():
    secret = "super-secret-value"
    credentials = EndpointCredentials(bearer_token=secret)
    candidate = CandidateEndpoint(
        provider="modal",
        base_url="https://inference.modal.run",
        credentials=credentials,
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )

    with pytest.raises(EndpointAuthenticationError) as caught:
        EndpointResolver(
            _client(
                lambda request: httpx.Response(
                    401,
                    json={"detail": secret},
                    request=request,
                )
            )
        ).resolve(SPEC, explicit=candidate)

    assert secret not in repr(credentials)
    assert secret not in repr(candidate)
    assert secret not in str(caught.value)


def test_all_unreachable_candidates_raise_contextual_error():
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    candidate = CandidateEndpoint(
        provider="local",
        base_url="http://127.0.0.1:39001",
        credentials=EndpointCredentials(bearer_token="local-key"),
        model_revision=REVISION,
        identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
    )

    with pytest.raises(
        EndpointUnavailableError,
        match="vllm_colpali.*modal.*e2e.*dev.*local",
    ):
        EndpointResolver(_client(unavailable)).resolve(SPEC, candidates=[candidate])
