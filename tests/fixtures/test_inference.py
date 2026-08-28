from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Event, Lock, Thread
from types import MappingProxyType

import httpx
import pytest
from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointAuthenticationError,
    EndpointContractError,
    EndpointCredentials,
    EndpointIdentityEvidence,
    ModelIdentityError,
    ResolvedInferenceEndpoint,
)
from cogniverse_cli.modal_inference_config import get_inference_service_spec

from tests.fixtures.inference import (
    TEST_INFERENCE_API_KEY,
    DiscoveredEndpointProvider,
    EndpointValidator,
    InferenceSessionResolver,
    LocalEndpointProvider,
    ModalEndpointProvider,
    ProviderUnavailable,
    collect_required_inference_services,
    explicit_endpoints_from_environment,
    publish_inference_endpoints,
)

COLPALI = get_inference_service_spec("vllm_colpali")
DENSEON = get_inference_service_spec("denseon")
CLAP = get_inference_service_spec("clap_embed")
API_KEY = "shared-inference-secret"


@contextmanager
def _model_server(*, model: str, revision: str | None, token: str | None = None):
    requests: list[tuple[str, str | None]] = []
    request_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with request_lock:
                requests.append((self.path, self.headers.get("Authorization")))
            record = {
                "id": model,
                "object": "model",
            }
            if revision is not None:
                record["revision"] = revision
            payload = {
                "object": "list",
                "data": [record],
            }
            status = 200
            if token is not None and self.headers.get("Authorization") != (
                f"Bearer {token}"
            ):
                status = 401
                payload = {"detail": "unauthorized"}
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
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _vllm_model_server(*, model: str, revision: str | None, token: str | None = None):
    requests: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            if token is not None and self.headers.get("Authorization") != (
                f"Bearer {token}"
            ):
                payload = json.dumps({"detail": "unauthorized"}).encode()
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            record = {
                "id": model,
                "root": model,
                "parent": None,
                "max_model_len": 4096,
            }
            if revision is not None:
                record["revision"] = revision
            payload = json.dumps({"object": "list", "data": [record]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

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


@contextmanager
def _health_server(*, status: str, include_model: bool = True):
    requests: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            payload = {"status": status, "model_revision": CLAP.model_revision}
            if include_model:
                payload["model"] = CLAP.model_id
            body = json.dumps(payload).encode()
            self.send_response(200)
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
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _Provider:
    def __init__(
        self,
        name: str,
        result: ResolvedInferenceEndpoint | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.result = result
        self.error = error
        self.calls: list[str] = []
        self.close_calls = 0

    def resolve(self, spec):
        self.calls.append(spec.name)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.close_calls += 1


def _resolved(service: str, provider: str, url: str, token: str | None = None):
    spec = get_inference_service_spec(service)
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return ResolvedInferenceEndpoint(
        service=service,
        provider=provider,
        base_url=url,
        headers=MappingProxyType(headers),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )


def _candidate(
    base_url: str,
    provider: str = "e2e",
    identity_evidence: EndpointIdentityEvidence = EndpointIdentityEvidence.ENDPOINT,
    model_revision: str | None = None,
) -> CandidateEndpoint:
    return CandidateEndpoint(
        provider=provider,
        base_url=base_url,
        credentials=EndpointCredentials(bearer_token=TEST_INFERENCE_API_KEY),
        identity_evidence=identity_evidence,
        model_revision=model_revision,
    )


def _discovered(base_url: str, model_revision: str | None):
    from tests.utils.vllm_sidecar import _DiscoveredClusterEndpoint

    return _DiscoveredClusterEndpoint(
        base_url=base_url,
        model_revision=model_revision,
    )


@contextmanager
def _blocking_model_server():
    request_started = Event()
    allow_response = Event()
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            request_started.set()
            assert allow_response.wait(timeout=3)
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": COLPALI.model_id,
                            "revision": COLPALI.model_revision,
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
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
        yield (
            f"http://{host}:{port}",
            request_started,
            allow_response,
            requests,
        )
    finally:
        allow_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.unit
def test_validator_close_waits_for_real_request_then_rejects_new_validation():
    validator = EndpointValidator()
    owned_client = validator._client

    with _blocking_model_server() as (
        base_url,
        request_started,
        allow_response,
        requests,
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            validation = pool.submit(validator.validate, COLPALI, _candidate(base_url))
            assert request_started.wait(timeout=3)
            closing = pool.submit(validator.close)
            deadline = time.monotonic() + 3
            while not validator._closed and time.monotonic() < deadline:
                time.sleep(0.001)
            assert validator._closed
            assert not closing.done()
            allow_response.set()
            endpoint = validation.result(timeout=3)
            closing.result(timeout=3)

        with pytest.raises(RuntimeError, match="endpoint validator is closed"):
            validator.validate(COLPALI, _candidate(base_url))

    validator.close()
    assert endpoint.model_id == COLPALI.model_id
    assert endpoint.model_revision == COLPALI.model_revision
    assert requests == ["/v1/models"]
    assert owned_client.is_closed


@pytest.mark.unit
def test_validator_never_closes_injected_client():
    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
    ) as (base_url, requests):
        with httpx.Client(timeout=3) as injected_client:
            validator = EndpointValidator(injected_client)
            endpoint = validator.validate(COLPALI, _candidate(base_url))

            validator.close()
            validator.close()

            assert not injected_client.is_closed
            assert injected_client.get(f"{base_url}/v1/models").json() == {
                "object": "list",
                "data": [
                    {
                        "id": COLPALI.model_id,
                        "object": "model",
                        "revision": COLPALI.model_revision,
                    }
                ],
            }
            with pytest.raises(RuntimeError, match="endpoint validator is closed"):
                validator.validate(COLPALI, _candidate(base_url))

    assert endpoint.model_id == COLPALI.model_id
    assert requests == [
        ("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}"),
        ("/v1/models", None),
    ]


@pytest.mark.unit
def test_discovered_provider_closes_only_its_owned_validator():
    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
    ) as (base_url, requests):
        provider = DiscoveredEndpointProvider(
            "e2e",
            lambda spec: (_discovered(base_url, None),),
        )
        owned_client = provider._validator._client
        endpoint = provider.resolve(COLPALI)

        provider.close()
        provider.close()

    assert endpoint.base_url == base_url
    assert owned_client.is_closed
    assert requests == [("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_discovered_provider_uses_pinned_revision_as_deployment_evidence():
    class RecordingValidator:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def validate(self, spec, candidate):
            self.calls.append((spec, candidate))
            return _resolved(
                spec.name,
                candidate.provider,
                candidate.base_url,
                token=TEST_INFERENCE_API_KEY,
            )

    validator = RecordingValidator()
    provider = DiscoveredEndpointProvider(
        "e2e",
        lambda spec: (_discovered("http://127.0.0.1:34124", COLPALI.model_revision),),
        validator=validator,
    )

    endpoint = provider.resolve(COLPALI)

    assert endpoint == _resolved(
        "vllm_colpali", "e2e", "http://127.0.0.1:34124", TEST_INFERENCE_API_KEY
    )
    assert validator.calls == [
        (
            COLPALI,
            CandidateEndpoint(
                provider="e2e",
                base_url="http://127.0.0.1:34124",
                credentials=EndpointCredentials(bearer_token=TEST_INFERENCE_API_KEY),
                identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
                model_revision=COLPALI.model_revision,
            ),
        )
    ]


@pytest.mark.unit
def test_discovered_provider_rejects_pinned_revision_mismatch():
    wrong_revision = "1" * 40
    provider = DiscoveredEndpointProvider(
        "e2e",
        lambda spec: (_discovered("http://127.0.0.1:34125", wrong_revision),),
    )

    with pytest.raises(ModelIdentityError) as caught:
        provider.resolve(COLPALI)

    assert str(caught.value) == (
        f"vllm_colpali: deployment revision {wrong_revision!r} does not match expected "
        f"{COLPALI.model_revision!r}"
    )


@pytest.mark.unit
def test_discovered_provider_requires_reported_revision_when_discovery_finds_none():
    with _vllm_model_server(
        model=COLPALI.model_id,
        revision=None,
        token=TEST_INFERENCE_API_KEY,
    ) as (base_url, requests):
        provider = DiscoveredEndpointProvider(
            "e2e",
            lambda spec: (_discovered(base_url, None),),
        )

        with pytest.raises(ModelIdentityError) as caught:
            provider.resolve(COLPALI)

    assert str(caught.value) == (
        f"vllm_colpali: expected revision {COLPALI.model_revision!r}, got None"
    )
    assert requests == [("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_discovered_provider_accepts_pinned_revision_against_vllm_models_server():
    with _vllm_model_server(
        model=COLPALI.model_id,
        revision=None,
        token=TEST_INFERENCE_API_KEY,
    ) as (base_url, requests):
        provider = DiscoveredEndpointProvider(
            "e2e",
            lambda spec: (_discovered(base_url, COLPALI.model_revision),),
        )

        endpoint = provider.resolve(COLPALI)

    assert endpoint == _resolved(
        "vllm_colpali",
        "e2e",
        base_url,
        TEST_INFERENCE_API_KEY,
    )
    assert requests == [("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_discovered_custom_server_requires_canonical_ready_health_status():
    with _health_server(status="ready") as (base_url, requests):
        validator = EndpointValidator()
        try:
            endpoint = validator.validate(CLAP, _candidate(base_url))
        finally:
            validator.close()

    assert endpoint == _resolved(
        "clap_embed", "e2e", base_url, token=TEST_INFERENCE_API_KEY
    )
    assert requests == [("/health", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_discovered_custom_server_rejects_obsolete_ok_health_status():
    with _health_server(status="ok") as (base_url, requests):
        validator = EndpointValidator()
        try:
            with pytest.raises(
                EndpointContractError,
                match="^clap_embed: e2e health must report status=ready$",
            ):
                validator.validate(CLAP, _candidate(base_url))
        finally:
            validator.close()

    assert requests == [("/health", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_locally_started_sidecar_health_must_carry_the_model_key():
    """A ``/health`` body without ``model`` is rejected for a locally started
    sidecar, the way the runtime's boot probe rejects it.

    ``LocalEndpointProvider.resolve`` stamps every container/vLLM it starts
    with ``DEPLOYMENT`` evidence, and every non-modal non-vLLM service is
    probed at ``/health``. The runtime's ``_extract_model_from_health`` reads
    ``body["model"]`` and nothing else, so a sidecar that omits the key can
    never be identified and the runtime refuses to boot for every profile
    bound to it.
    """
    with _health_server(status="ready", include_model=False) as (base_url, requests):
        validator = EndpointValidator()
        try:
            with pytest.raises(
                ModelIdentityError,
                match=re.escape(
                    f"clap_embed: expected model {CLAP.model_id!r}, got None"
                ),
            ):
                validator.validate(
                    CLAP,
                    _candidate(
                        base_url,
                        identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
                        model_revision=CLAP.model_revision,
                    ),
                )
        finally:
            validator.close()

    assert requests == [("/health", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_provider_does_not_close_injected_validator():
    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
    ) as (base_url, requests):
        with httpx.Client(timeout=3) as injected_client:
            validator = EndpointValidator(injected_client)
            provider = DiscoveredEndpointProvider(
                "e2e",
                lambda spec: (_discovered(base_url, None),),
                validator=validator,
            )

            first = provider.resolve(COLPALI)
            provider.close()
            second = validator.validate(COLPALI, _candidate(base_url))

            assert not injected_client.is_closed
            validator.close()

    assert first == second
    assert requests == [
        ("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}"),
        ("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}"),
    ]


@pytest.mark.unit
def test_local_provider_closes_validator_when_sidecar_teardown_fails(monkeypatch):
    with _model_server(
        model=COLPALI.model_id,
        revision=None,
    ) as (base_url, requests):
        provider = LocalEndpointProvider()
        owned_client = provider._validator._client
        monkeypatch.setattr(
            provider._vllm,
            "spawn",
            lambda model, *, extra_args, env: base_url,
        )

        def fail_teardown() -> None:
            raise TimeoutError("sidecar teardown timed out")

        monkeypatch.setattr(provider._vllm, "teardown", fail_teardown)
        endpoint = provider.resolve(COLPALI)

        with pytest.raises(
            RuntimeError,
            match="inference cleanup failed: sidecar teardown timed out",
        ):
            provider.close()

    assert endpoint.base_url == base_url
    assert requests == [("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}")]
    assert owned_client.is_closed


@pytest.mark.unit
def test_resolver_closes_explicit_validator_when_another_provider_close_fails(
    monkeypatch,
):
    class FailingCloseProvider(_Provider):
        fail = True

        def close(self) -> None:
            self.close_calls += 1
            if self.fail:
                raise TimeoutError("provider release timed out")

    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
        token=API_KEY,
    ) as (base_url, requests):
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            json.dumps({"vllm_colpali": base_url}),
        )
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        failing = FailingCloseProvider("e2e")
        resolver = InferenceSessionResolver(
            explicit_endpoints=explicit_endpoints_from_environment({"vllm_colpali"}),
            providers=(failing,),
        )
        explicit_client = resolver._explicit._validator._client

        endpoint = resolver.resolve("vllm_colpali")
        with pytest.raises(
            RuntimeError,
            match="e2e: provider release timed out",
        ):
            resolver.close()

        assert explicit_client.is_closed
        failing.fail = False
        resolver.close()

    assert endpoint.base_url == base_url
    assert failing.close_calls == 2
    assert requests == [("/v1/models", f"Bearer {API_KEY}")]


@pytest.mark.unit
def test_explicit_endpoint_publishes_exact_url_and_immutable_headers(monkeypatch):
    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
        token=API_KEY,
    ) as (base_url, requests):
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            json.dumps({"vllm_colpali": base_url}),
        )
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        explicit = explicit_endpoints_from_environment({"vllm_colpali"})
        lower = _Provider(
            "local",
            error=AssertionError(
                "explicit endpoints must not probe fallback providers"
            ),
        )
        resolver = InferenceSessionResolver(
            explicit_endpoints=explicit,
            providers=(lower,),
        )

        endpoints = resolver.resolve_required({"vllm_colpali"})
        with publish_inference_endpoints(endpoints):
            assert json.loads(os.environ["INFERENCE_SERVICE_URLS"]) == {
                "vllm_colpali": base_url
            }
            assert endpoints["vllm_colpali"].base_url == base_url
            assert dict(endpoints["vllm_colpali"].headers) == {
                "Authorization": f"Bearer {API_KEY}"
            }
            assert API_KEY not in repr(endpoints["vllm_colpali"])
            with pytest.raises(TypeError):
                endpoints["vllm_colpali"].headers["Authorization"] = "changed"

    assert requests == [("/v1/models", f"Bearer {API_KEY}")]
    assert lower.calls == []


@pytest.mark.unit
def test_generic_resolution_uses_e2e_dev_local_without_modal():
    calls: list[str] = []

    class Provider(_Provider):
        def resolve(self, spec):
            calls.append(self.name)
            return super().resolve(spec)

    providers = (
        Provider("e2e"),
        Provider("dev"),
        Provider(
            "modal",
            error=AssertionError("generic inference must not warm Modal"),
        ),
        Provider(
            "local",
            _resolved("vllm_colpali", "local", "http://127.0.0.1:34120"),
        ),
    )
    resolver = InferenceSessionResolver(providers=providers)

    endpoint = resolver.resolve("vllm_colpali")

    assert calls == ["e2e", "dev", "local"]
    assert endpoint.provider == "local"
    assert endpoint.base_url == "http://127.0.0.1:34120"


@pytest.mark.unit
def test_api_key_does_not_warm_modal_for_generic_cluster_resolution(monkeypatch):
    from cogniverse_cli import modal_inference_lifecycle

    from tests.fixtures import inference as inference_fixture
    from tests.utils import vllm_sidecar

    events: list[str] = []

    class Lifecycle:
        def __init__(self, *, credentials):
            self.credentials = credentials

        def warm(self, services):
            events.append("modal")
            raise AssertionError("generic inference must not warm Modal")

        def release(self, services):
            raise AssertionError("an unwarmed Modal lifecycle must not be released")

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("borrowed cluster teardown must not invoke subprocess")

    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
        token=API_KEY,
    ) as (base_url, requests):
        # The session fixture publishes INFERENCE_SERVICE_URLS for marked
        # tests; this test exercises the DISCOVERY chain, so ambient
        # published endpoints must not hijack resolution via the explicit
        # provider.
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        monkeypatch.setattr(
            modal_inference_lifecycle,
            "ModalInferenceLifecycle",
            Lifecycle,
        )
        monkeypatch.setattr(
            vllm_sidecar,
            "_discover_e2e_model_urls",
            lambda model: events.append("e2e") or (_discovered(base_url, None),),
        )
        monkeypatch.setattr(
            vllm_sidecar,
            "_discover_dev_model_urls",
            lambda model: events.append("dev") or (),
        )
        resolver = inference_fixture._build_resolver({"vllm_colpali"})
        endpoint = resolver.resolve("vllm_colpali")
        monkeypatch.setattr(inference_fixture.subprocess, "run", unexpected_subprocess)

        resolver.close()

    assert events == ["e2e"]
    assert endpoint.provider == "e2e"
    assert requests == [("/v1/models", f"Bearer {API_KEY}")]


@pytest.mark.unit
def test_explicit_modal_selection_warms_only_modal(monkeypatch):
    from cogniverse_cli import modal_inference_lifecycle

    from tests.fixtures import inference as inference_fixture
    from tests.utils import vllm_sidecar

    events: list[tuple[str, tuple[str, ...]]] = []

    class Lifecycle:
        def __init__(self, *, credentials):
            self.credentials = credentials

        def warm(self, services):
            selected = tuple(services)
            events.append(("warm", selected))
            return (
                _resolved(
                    "vllm_llm_student",
                    "modal",
                    "https://gemma.modal.run",
                    API_KEY,
                ),
            )

        def release(self, services):
            events.append(("release", tuple(services)))

    def unexpected_discovery(model):
        raise AssertionError("explicit Modal selection must not inspect k3d")

    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    monkeypatch.setattr(
        modal_inference_lifecycle,
        "ModalInferenceLifecycle",
        Lifecycle,
    )
    monkeypatch.setattr(
        vllm_sidecar,
        "_discover_e2e_model_urls",
        unexpected_discovery,
    )
    monkeypatch.setattr(
        vllm_sidecar,
        "_discover_dev_model_urls",
        unexpected_discovery,
    )
    resolver = inference_fixture._build_resolver(
        {"vllm_llm_student"},
        modal_services={"vllm_llm_student"},
    )

    endpoint = resolver.resolve("vllm_llm_student")
    resolver.close()

    assert endpoint.provider == "modal"
    assert events == [
        ("warm", ("vllm_llm_student",)),
        ("release", ("vllm_llm_student",)),
    ]


@pytest.mark.unit
def test_bad_explicit_endpoint_raises_without_probing_lower_provider(monkeypatch):
    with _model_server(
        model="wrong/model",
        revision=COLPALI.model_revision,
        token=API_KEY,
    ) as (base_url, requests):
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            json.dumps({"vllm_colpali": base_url}),
        )
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        lower = _Provider(
            "local",
            _resolved("vllm_colpali", "local", "http://127.0.0.1:34121"),
        )
        resolver = InferenceSessionResolver(
            explicit_endpoints=explicit_endpoints_from_environment({"vllm_colpali"}),
            providers=(lower,),
        )

        with pytest.raises(ModelIdentityError, match="expected model"):
            resolver.resolve("vllm_colpali")

    assert requests == [("/v1/models", f"Bearer {API_KEY}")]
    assert lower.calls == []


@pytest.mark.unit
def test_explicit_modal_endpoint_requires_authentication(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        '{"vllm_colpali":"https://cogniverse-vllm-colpali.modal.run"}',
    )
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        EndpointAuthenticationError,
        match="vllm_colpali.*configured API key",
    ):
        explicit_endpoints_from_environment({"vllm_colpali"})


@pytest.mark.unit
def test_explicit_non_modal_endpoint_requires_reported_exact_revision(monkeypatch):
    with _model_server(model=COLPALI.model_id, revision=None) as (base_url, requests):
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            json.dumps({"vllm_colpali": base_url}),
        )
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        resolver = InferenceSessionResolver(
            explicit_endpoints=explicit_endpoints_from_environment({"vllm_colpali"}),
            providers=(),
        )

        with pytest.raises(
            ModelIdentityError,
            match="expected revision.*got None",
        ):
            resolver.resolve("vllm_colpali")

    assert requests == [("/v1/models", f"Bearer {API_KEY}")]


@pytest.mark.unit
def test_explicit_modal_endpoint_rejects_http_before_requesting_with_bearer(
    monkeypatch,
):
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        '{"vllm_colpali":"http://cogniverse-vllm-colpali.modal.run"}',
    )
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)

    with pytest.raises(
        EndpointAuthenticationError,
        match="vllm_colpali.*Modal endpoint requires HTTPS",
    ):
        explicit_endpoints_from_environment({"vllm_colpali"})


@pytest.mark.unit
def test_concurrent_resolution_selects_and_validates_once(monkeypatch):
    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
    ) as (base_url, requests):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        discovery_calls: list[str] = []

        def discover(spec):
            discovery_calls.append(spec.name)
            return (_discovered(base_url, None),)

        endpoint_provider = DiscoveredEndpointProvider(
            "e2e",
            discover,
        )
        resolver = InferenceSessionResolver(providers=(endpoint_provider,))
        simultaneous = Barrier(12)

        def resolve():
            simultaneous.wait(timeout=3)
            return resolver.resolve("vllm_colpali")

        with ThreadPoolExecutor(max_workers=12) as pool:
            endpoints = tuple(pool.map(lambda _: resolve(), range(12)))
        resolver.close()
        resolver.close()

    assert not simultaneous.broken
    assert discovery_calls == ["vllm_colpali"]
    assert requests == [("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}")]
    assert dict(endpoints[0].headers) == {
        "Authorization": f"Bearer {TEST_INFERENCE_API_KEY}"
    }
    assert TEST_INFERENCE_API_KEY not in repr(endpoints[0])
    assert len({id(endpoint) for endpoint in endpoints}) == 1


@pytest.mark.parametrize(
    ("service", "expected_model", "expected_args", "expected_env"),
    [
        (
            "vllm_colpali",
            "TomoroAI/tomoro-colqwen3-embed-4b",
            (
                "--revision",
                "bf790bd8780b098b86453444632a184bb770be1a",
                "--max-model-len",
                "4096",
                "--runner",
                "pooling",
                "--convert",
                "embed",
                "--limit-mm-per-prompt",
                '{"video":0,"image":1}',
            ),
            None,
        ),
        (
            "vllm_llm_student",
            "google/gemma-4-e4b-it",
            (
                "--revision",
                "ee0ef6023621cff504d758262d4e04895a5af4a2",
                "--max-model-len",
                "8192",
                "--enforce-eager",
                "--max-num-seqs",
                "1",
                "--limit-mm-per-prompt",
                '{"video":0,"image":4}',
            ),
            None,
        ),
    ],
)
@pytest.mark.unit
def test_local_vllm_launch_matches_production_contract(
    monkeypatch,
    service,
    expected_model,
    expected_args,
    expected_env,
):
    spec = get_inference_service_spec(service)
    spawn_calls: list[tuple[str, tuple[str, ...], object]] = []
    with _model_server(
        model=spec.model_id,
        revision=None,
    ) as (base_url, requests):
        provider = LocalEndpointProvider()

        def spawn(model, *, extra_args, env):
            spawn_calls.append((model, tuple(extra_args), env))
            return base_url

        monkeypatch.setattr(
            provider._vllm,
            "spawn",
            spawn,
        )
        monkeypatch.setattr(provider._vllm, "teardown", lambda: None)

        endpoint = provider.resolve(spec)
        provider.close()

    assert spawn_calls == [(expected_model, expected_args, expected_env)]
    assert endpoint.base_url == base_url
    assert dict(endpoint.headers) == {
        "Authorization": f"Bearer {TEST_INFERENCE_API_KEY}"
    }
    assert requests == [("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}")]


@pytest.mark.unit
def test_local_llm_provisioning_failure_preserves_error_and_retries_release(
    monkeypatch,
):
    release_calls = 0

    def ensure(model: str, revision: str) -> str:
        raise RuntimeError(f"provisioning failed for {model}@{revision}")

    def release() -> None:
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            raise OSError("controlled LLM release failure")

    provider = LocalEndpointProvider(
        llm_ensurer=ensure,
        llm_active=lambda model, revision: False,
        llm_releaser=release,
    )
    monkeypatch.setattr(provider._vllm, "teardown", lambda: None)

    with pytest.raises(RuntimeError, match="provisioning failed") as caught:
        provider.resolve(get_inference_service_spec("vllm_llm_student"))

    assert caught.value.__notes__ == [
        "vllm_llm_student cleanup failed: OSError: controlled LLM release failure"
    ]
    assert provider._owns_llm is True
    provider.close()
    assert provider._owns_llm is False
    assert release_calls == 2


@pytest.mark.parametrize(
    ("service", "relative_dockerfile", "relative_context"),
    [
        ("gliner", "deploy/gliner/Dockerfile", "."),
        (
            "videoprism_jax",
            "deploy/videoprism/Dockerfile",
            ".",
        ),
        ("clap_embed", "deploy/clap_embed/Dockerfile", "."),
        ("face_embed", "deploy/face_embed/Dockerfile", "."),
        ("colbert_pylate", "deploy/pylate/Dockerfile", "."),
        ("code_colbert_pylate", "deploy/pylate/Dockerfile", "."),
    ],
)
@pytest.mark.unit
def test_local_container_rebuilds_its_declared_docker_context(
    monkeypatch,
    service,
    relative_dockerfile,
    relative_context,
):
    from tests.fixtures import inference as inference_fixture

    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    class Healthy:
        status_code = 200

    monkeypatch.setattr(inference_fixture.subprocess, "run", run)
    monkeypatch.setattr(inference_fixture, "_free_port", lambda: 39123)
    monkeypatch.setattr(
        inference_fixture.httpx, "get", lambda *args, **kwargs: Healthy()
    )
    provider = LocalEndpointProvider()
    spec = get_inference_service_spec(service)

    assert provider._start_container(spec) == "http://127.0.0.1:39123"

    repo = inference_fixture.Path(inference_fixture.__file__).resolve().parents[2]
    assert not any(
        command[:3] == ["docker", "image", "inspect"] for command in commands
    )
    build = next(command for command in commands if command[:2] == ["docker", "build"])
    assert build == [
        "docker",
        "build",
        "-f",
        str(repo / relative_dockerfile),
        "-t",
        inference_fixture._CONTAINER_SPECS[service].image,
        str((repo / relative_context).resolve()),
    ]


@pytest.mark.unit
def test_local_pylate_container_runs_as_invoking_user_with_pinned_model(
    monkeypatch,
):
    """The colbert_pylate container carries the pinned model identity and
    runs as the invoking user against the test-owned cache — container
    writes must never land root-owned files in a cache the host-side
    oracle reads."""
    import os as os_mod

    from tests.fixtures import inference as inference_fixture
    from tests.utils.vllm_sidecar import CONTAINER_HF_CACHE, TEST_HF_CACHE

    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    class Healthy:
        status_code = 200

    monkeypatch.setattr(inference_fixture.subprocess, "run", run)
    monkeypatch.setattr(inference_fixture, "_free_port", lambda: 39123)
    monkeypatch.setattr(
        inference_fixture.httpx, "get", lambda *args, **kwargs: Healthy()
    )
    provider = LocalEndpointProvider()
    spec = get_inference_service_spec("colbert_pylate")

    assert provider._start_container(spec) == "http://127.0.0.1:39123"

    run_command = next(
        command for command in commands if command[:2] == ["docker", "run"]
    )
    container_name = run_command[4]
    assert run_command == [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--label",
        f"cogniverse-test-owner-pid={os_mod.getpid()}",
        "-p",
        "39123:8080",
        "--oom-score-adj=500",
        "--user",
        f"{os_mod.getuid()}:{os_mod.getgid()}",
        "-e",
        f"HOME={CONTAINER_HF_CACHE}",
        "-e",
        f"HF_HOME={CONTAINER_HF_CACHE}",
        "-e",
        "LOGNAME=cogniverse",
        "-e",
        "USER=cogniverse",
        "-v",
        f"{TEST_HF_CACHE}:{CONTAINER_HF_CACHE}",
        "-e",
        "MODEL_NAME=lightonai/LateOn",
        "-e",
        "MODEL_REVISION=c01907b70557ee5c7753680d4819a5cce1674b83",
        "-e",
        "DEVICE=cpu",
        "cogniverse/pylate:0.1.0-dev",
    ]


@pytest.mark.unit
def test_local_container_build_failure_names_service_dockerfile_and_context(
    monkeypatch,
):
    """A build that fails on both attempts surfaces the dockerfile, context,
    and the final cause — a persistent failure is never retried forever."""
    from tests.fixtures import inference as inference_fixture

    build_attempts = 0

    def run(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 1)
        nonlocal build_attempts
        build_attempts += 1
        raise subprocess.CalledProcessError(17, command, stderr="missing server.py")

    monkeypatch.setattr(inference_fixture.subprocess, "run", run)
    provider = LocalEndpointProvider()
    repo = inference_fixture.Path(inference_fixture.__file__).resolve().parents[2]
    expected_command = [
        "docker",
        "build",
        "-f",
        str(repo / "deploy/videoprism/Dockerfile"),
        "-t",
        "cogniverse/videoprism:0.1.0-dev",
        str(repo),
    ]

    with pytest.raises(ProviderUnavailable) as caught:
        provider._start_container(get_inference_service_spec("videoprism_jax"))

    assert build_attempts == 2
    cause = caught.value.__cause__
    assert isinstance(cause, subprocess.CalledProcessError)
    assert cause.cmd == expected_command
    assert cause.returncode == 17
    assert cause.stderr == "missing server.py"
    assert str(caught.value) == (
        "videoprism_jax: Docker image build failed twice using "
        f"{repo / 'deploy/videoprism/Dockerfile'} with context {repo} "
        f"(CalledProcessError: {cause})"
    )


@pytest.mark.unit
def test_local_container_build_retries_once_after_transient_failure(monkeypatch):
    """A single transient registry/PyPI failure mid-build must not sink the
    session: the second attempt runs (cached layers make it cheap) and the
    container starts normally."""
    from tests.fixtures import inference as inference_fixture

    build_attempts = 0
    run_commands: list[list[str]] = []

    def run(command, **kwargs):
        run_commands.append(list(command))
        if command[:2] == ["docker", "build"]:
            nonlocal build_attempts
            build_attempts += 1
            if build_attempts == 1:
                raise subprocess.CalledProcessError(
                    2, command, stderr="Read timed out."
                )
        return subprocess.CompletedProcess(command, 0)

    class Healthy:
        status_code = 200

    monkeypatch.setattr(inference_fixture.subprocess, "run", run)
    monkeypatch.setattr(inference_fixture, "_free_port", lambda: 39124)
    monkeypatch.setattr(
        inference_fixture.httpx, "get", lambda *args, **kwargs: Healthy()
    )
    provider = LocalEndpointProvider()

    url = provider._start_container(get_inference_service_spec("videoprism_jax"))

    assert url == "http://127.0.0.1:39124"
    assert build_attempts == 2
    assert any(command[:2] == ["docker", "run"] for command in run_commands)


@pytest.mark.parametrize(
    "failure",
    [
        "authentication rejected",
        "model identity mismatch",
        "authenticated health probe timed out",
        "autoscaler update failed",
    ],
)
@pytest.mark.unit
def test_modal_lifecycle_failure_propagates_without_provider_fallback(failure):
    from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

    class FailingLifecycle:
        def warm(self, services):
            raise ModalLifecycleError(failure)

    lower = _Provider(
        "e2e",
        _resolved("vllm_colpali", "e2e", "http://127.0.0.1:34122"),
    )
    resolver = InferenceSessionResolver(
        providers=(ModalEndpointProvider(FailingLifecycle()), lower),
        modal_services={"vllm_colpali"},
    )

    with pytest.raises(ModalLifecycleError, match=failure):
        resolver.resolve("vllm_colpali")

    assert lower.calls == []


@pytest.mark.unit
def test_absent_modal_lifecycle_continues_to_the_next_provider():
    lower = _Provider(
        "e2e",
        _resolved("vllm_colpali", "e2e", "http://127.0.0.1:34123"),
    )
    resolver = InferenceSessionResolver(
        providers=(ModalEndpointProvider(), lower),
    )

    endpoint = resolver.resolve("vllm_colpali")

    assert endpoint.provider == "e2e"
    assert lower.calls == ["vllm_colpali"]


@pytest.mark.unit
def test_successful_session_releases_each_provider_once():
    modal = _Provider(
        "modal",
        _resolved("vllm_colpali", "modal", "https://colpali.modal.run", API_KEY),
    )
    resolver = InferenceSessionResolver(
        providers=(modal,),
        modal_services={"vllm_colpali"},
    )

    assert resolver.resolve_required({"vllm_colpali"})["vllm_colpali"].provider == (
        "modal"
    )
    resolver.close()
    resolver.close()

    assert modal.close_calls == 1


@pytest.mark.unit
def test_modal_endpoint_provider_releases_only_the_services_it_warmed():
    class RecordingLifecycle:
        def __init__(self) -> None:
            self.warm_calls: list[tuple[str, ...]] = []
            self.release_calls: list[tuple[str, ...]] = []

        def warm(self, services):
            services = tuple(services)
            self.warm_calls.append(services)
            return tuple(
                _resolved(service, "modal", f"https://{service}.modal.run")
                for service in services
            )

        def release(self, services):
            self.release_calls.append(tuple(services))

    lifecycle = RecordingLifecycle()
    provider = ModalEndpointProvider(lifecycle)
    provider.resolve(get_inference_service_spec("vllm_colpali"))
    provider.close()
    provider.close()

    assert lifecycle.warm_calls == [("vllm_colpali",)]
    assert lifecycle.release_calls == [("vllm_colpali",)]


@pytest.mark.unit
def test_modal_endpoint_provider_close_without_warm_does_not_release():
    class RecordingLifecycle:
        def __init__(self) -> None:
            self.release_calls: list[tuple[str, ...]] = []

        def warm(self, services):
            raise AssertionError("warm should not be called")

        def release(self, services):
            self.release_calls.append(tuple(services))

    lifecycle = RecordingLifecycle()
    provider = ModalEndpointProvider(lifecycle)
    provider.close()

    assert lifecycle.release_calls == []


@pytest.mark.unit
def test_closed_resolver_rejects_resolve_and_resolve_required_without_provider_calls():
    provider = _Provider(
        "local",
        _resolved("vllm_colpali", "local", "http://127.0.0.1:34124"),
    )
    resolver = InferenceSessionResolver(providers=(provider,))
    resolver.close()

    with pytest.raises(RuntimeError, match="inference resolver is closed"):
        resolver.resolve("vllm_colpali")
    with pytest.raises(RuntimeError, match="inference resolver is closed"):
        resolver.resolve_required(("vllm_colpali",))

    assert provider.calls == []


@pytest.mark.unit
def test_close_waits_for_inflight_resolution_and_that_resolution_fails():
    resolve_started = Event()
    allow_resolution = Event()
    events: list[str] = []

    class BlockingProvider(_Provider):
        def resolve(self, spec):
            events.append("resolve-started")
            resolve_started.set()
            assert allow_resolution.wait(timeout=3)
            events.append("resolve-finished")
            return _resolved(spec.name, "local", "http://127.0.0.1:34125")

        def close(self):
            events.append("provider-closed")

    resolver = InferenceSessionResolver(providers=(BlockingProvider("local"),))
    with ThreadPoolExecutor(max_workers=2) as pool:
        resolution = pool.submit(resolver.resolve, "vllm_colpali")
        assert resolve_started.wait(timeout=3)
        teardown = pool.submit(resolver.close)
        deadline = time.monotonic() + 3
        while not resolver._closed and time.monotonic() < deadline:
            time.sleep(0.001)
        assert resolver._closed
        allow_resolution.set()
        with pytest.raises(RuntimeError, match="inference resolver is closed"):
            resolution.result(timeout=3)
        teardown.result(timeout=3)

    assert events == ["resolve-started", "resolve-finished", "provider-closed"]


@pytest.mark.unit
def test_failed_container_cleanup_remains_tracked_and_second_close_retries(
    monkeypatch,
):
    from tests.fixtures import inference as inference_fixture

    provider = LocalEndpointProvider()
    provider._containers.append("cogniverse-face-embed-test-deadbeef")
    monkeypatch.setattr(provider._vllm, "teardown", lambda: None)
    attempts = 0

    def remove(command, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.TimeoutExpired(command, timeout=30)
        return subprocess.CompletedProcess(command, 0, stdout="removed", stderr="")

    monkeypatch.setattr(inference_fixture.subprocess, "run", remove)
    resolver = InferenceSessionResolver(providers=(provider,))

    with pytest.raises(RuntimeError, match="TimeoutExpired"):
        resolver.close()
    assert provider._containers == ["cogniverse-face-embed-test-deadbeef"]

    resolver.close()

    assert attempts == 2
    assert provider._containers == []


@pytest.mark.unit
def test_partial_setup_failure_releases_warmed_provider_once():
    class Modal(_Provider):
        def resolve(self, spec):
            self.calls.append(spec.name)
            if spec.name == "vllm_colpali":
                return _resolved(
                    "vllm_colpali",
                    "modal",
                    "https://colpali.modal.run",
                    API_KEY,
                )
            return None

    modal = Modal("modal")
    unavailable = _Provider("e2e")
    resolver = InferenceSessionResolver(
        providers=(modal, unavailable),
        modal_services={"vllm_colpali"},
    )

    with pytest.raises(
        ProviderUnavailable,
        match="denseon.*e2e",
    ):
        resolver.resolve_required({"vllm_colpali", "denseon"})

    resolver.close()
    assert modal.close_calls == 1
    assert unavailable.close_calls == 1


class _Marker:
    def __init__(self, name: str, *args: str, **kwargs: str) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs


class _Item:
    def __init__(
        self,
        *,
        markers: tuple[_Marker, ...] = (),
        keywords: tuple[str, ...] = (),
        fixturenames: tuple[str, ...] = (),
    ) -> None:
        self.own_markers = list(markers)
        self.keywords = {keyword: True for keyword in keywords}
        self.fixturenames = list(fixturenames)

    def iter_markers_with_node(self, name=None):
        return [
            (self, marker)
            for marker in self.own_markers
            if name is None or marker.name == name
        ]


@pytest.mark.unit
def test_unrelated_collection_requires_no_inference_service():
    assert collect_required_inference_services([_Item()]) == frozenset()


@pytest.mark.unit
def test_collection_uses_only_exact_markers_without_mutating_other_markers():
    class ExactMarkerItem(_Item):
        def iter_markers_with_node(self, name=None):
            if name == "skipif":
                raise AssertionError("collection must not inspect skip markers")
            return super().iter_markers_with_node(name)

    class UnreadableKeywords:
        def __contains__(self, keyword):
            raise AssertionError("collection must not inspect marker keywords")

    unrelated_skip = _Marker("skipif", True, reason="unrelated capability")
    exact_marker = _Marker("requires_inference", "videoprism_jax")
    item = ExactMarkerItem(markers=(unrelated_skip, exact_marker))
    item.keywords = UnreadableKeywords()

    required = collect_required_inference_services([item])

    assert required == frozenset({"videoprism_jax", "vllm_asr"})
    assert item.own_markers == [unrelated_skip, exact_marker]


@pytest.mark.unit
def test_collection_records_explicit_modal_service_as_required():
    item = _Item(markers=(_Marker("requires_modal_inference", "vllm_llm_student"),))

    required = collect_required_inference_services([item])

    assert required == frozenset({"vllm_llm_student"})
    assert item._cogniverse_modal_inference_services == frozenset({"vllm_llm_student"})


@pytest.mark.unit
def test_unknown_named_service_fails_collection_instead_of_skipping():
    with pytest.raises(pytest.UsageError, match="unknown inference service 'missing'"):
        collect_required_inference_services(
            [_Item(markers=(_Marker("requires_inference", "missing"),))]
        )


@pytest.mark.unit
def test_borrowed_cluster_endpoint_is_not_mutated_during_teardown(monkeypatch):
    from tests.fixtures import inference as inference_fixture
    from tests.utils import vllm_sidecar

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("borrowed cluster teardown must not invoke subprocess")

    with _model_server(
        model=COLPALI.model_id,
        revision=COLPALI.model_revision,
    ) as (base_url, requests):
        # The session fixture publishes INFERENCE_SERVICE_URLS for marked
        # tests; this test exercises the DISCOVERY chain, so ambient
        # published endpoints must not hijack resolution via the explicit
        # provider.
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        monkeypatch.setattr(
            vllm_sidecar,
            "_discover_e2e_model_urls",
            lambda model: (_discovered(base_url, None),),
        )
        monkeypatch.setattr(
            vllm_sidecar,
            "_discover_dev_model_urls",
            lambda model: (),
        )
        resolver = inference_fixture._build_resolver({"vllm_colpali"})
        endpoint = resolver.resolve("vllm_colpali")
        monkeypatch.setattr(inference_fixture.subprocess, "run", unexpected_subprocess)

        resolver.close()

    assert endpoint.provider == "e2e"
    assert requests == [
        ("/v1/models", f"Bearer {TEST_INFERENCE_API_KEY}"),
    ]


@pytest.mark.integration
def test_shared_vespa_uses_tmpfs_for_search_db_and_round_trips(
    shared_vespa,
):
    """The shared Vespa keeps its search DB on a container tmpfs so feeds
    keep working regardless of host root-disk fill (Vespa blocks feeds when
    its data filesystem crosses its disk-usage limit). Pins the exact tmpfs
    mount and proves a config write→read round-trip lands on it."""
    from cogniverse_sdk.interfaces.config_store import ConfigScope
    from cogniverse_vespa.config.config_store import VespaConfigStore

    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            shared_vespa["container_name"],
            "--format",
            "{{json .HostConfig.Tmpfs}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(inspect.stdout) == {
        "/opt/vespa/var/db/vespa/search": "rw,size=8g,uid=1000,gid=1000,mode=0755"
    }

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_vespa["http_port"],
    )
    try:
        written = store.set_config(
            tenant_id="tmpfs:roundtrip",
            scope=ConfigScope.SYSTEM,
            service="inference-fixture",
            config_key="disk-boundary",
            config_value={"location": "tmpfs", "host_used_percent": 82},
        )
        loaded = store.get_config(
            tenant_id="tmpfs:roundtrip",
            scope=ConfigScope.SYSTEM,
            service="inference-fixture",
            config_key="disk-boundary",
        )
    finally:
        store.close()

    assert written.version == 1
    assert written.tenant_id == "tmpfs:roundtrip"
    assert written.scope is ConfigScope.SYSTEM
    assert written.service == "inference-fixture"
    assert written.config_key == "disk-boundary"
    assert written.config_value == {"location": "tmpfs", "host_used_percent": 82}
    assert loaded is not None
    assert loaded.scope is ConfigScope.SYSTEM
    assert loaded.version == 1
    assert loaded.config_value == {"location": "tmpfs", "host_used_percent": 82}


@pytest.mark.unit
def test_telemetry_guard_disables_export_without_collector(monkeypatch):
    """Without a test-owned collector, the ingestion conftest guard installs
    a disabled telemetry singleton so pipeline/worker spans no-op instead of
    exporting to the unreachable default localhost:4317, and it uninstalls
    only its own instance on teardown."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import (
        NoOpSpan,
        get_telemetry_manager,
    )
    from tests.ingestion.integration import conftest as ingestion_conftest

    monkeypatch.delenv("TELEMETRY_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(telemetry_manager_module, "_telemetry_manager", None)

    guard = ingestion_conftest._test_owned_telemetry.__wrapped__()
    next(guard)
    try:
        manager = telemetry_manager_module._telemetry_manager
        assert manager is not None
        assert manager.config.enabled is False
        assert get_telemetry_manager() is manager
        with manager.span(
            "pipeline.worker.process_job",
            tenant_id="guard:tenant",
            component="pipeline",
        ) as span:
            assert isinstance(span, NoOpSpan)
    finally:
        with pytest.raises(StopIteration):
            next(guard)
    assert telemetry_manager_module._telemetry_manager is None


@pytest.mark.unit
def test_telemetry_guard_defers_to_configured_collector(monkeypatch):
    """With TELEMETRY_OTLP_ENDPOINT set (a collector was provided, e.g. by
    phoenix_container), the guard must not install anything — the next
    telemetry build picks up the configured endpoint."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from tests.ingestion.integration import conftest as ingestion_conftest

    monkeypatch.setenv("TELEMETRY_OTLP_ENDPOINT", "http://localhost:24317")
    monkeypatch.setattr(telemetry_manager_module, "_telemetry_manager", None)

    guard = ingestion_conftest._test_owned_telemetry.__wrapped__()
    next(guard)
    try:
        assert telemetry_manager_module._telemetry_manager is None
    finally:
        with pytest.raises(StopIteration):
            next(guard)


@pytest.mark.unit
def test_ingestion_config_keeps_the_existing_inference_plugin_registration():
    from tests.ingestion.integration import conftest as ingestion_conftest

    class PluginManager:
        def __init__(self):
            self.registrations = []

        def hasplugin(self, name):
            return name == "tests.fixtures.inference"

        def register(self, plugin, name):
            self.registrations.append((plugin, name))

    plugin_manager = PluginManager()

    ingestion_conftest._register_inference_plugin(plugin_manager)

    assert plugin_manager.registrations == []


@pytest.mark.unit
def test_ingestion_config_registers_inference_when_it_is_the_test_root():
    from tests.fixtures import inference
    from tests.ingestion.integration import conftest as ingestion_conftest

    class PluginManager:
        def __init__(self):
            self.registrations = []

        def hasplugin(self, name):
            return False

        def register(self, plugin, name):
            self.registrations.append((plugin, name))

    plugin_manager = PluginManager()

    ingestion_conftest._register_inference_plugin(plugin_manager)

    assert plugin_manager.registrations == [(inference, "tests.fixtures.inference")]


def _validator_with_transport(handler) -> EndpointValidator:
    return EndpointValidator(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_validate_maps_a_closed_connection_to_provider_unavailable():
    """A k3d NodePort whose Service has no endpoints accepts the TCP connection
    and closes it: httpx raises RemoteProtocolError, not ConnectError. The
    validator must report it as the provider being unavailable, naming the
    service, exactly as it does for a refused connection."""

    def handler(request):
        raise httpx.RemoteProtocolError(
            "Server disconnected without sending a response.", request=request
        )

    validator = _validator_with_transport(handler)
    with pytest.raises(ProviderUnavailable) as caught:
        validator.validate(COLPALI, _candidate("http://127.0.0.1:33905"))
    assert str(caught.value) == "vllm_colpali: e2e closed the connection"
    assert isinstance(caught.value.__cause__, httpx.RemoteProtocolError)
    validator.close()


def test_validate_maps_a_refused_connection_to_provider_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    validator = _validator_with_transport(handler)
    with pytest.raises(ProviderUnavailable) as caught:
        validator.validate(COLPALI, _candidate("http://127.0.0.1:1"))
    assert str(caught.value) == "vllm_colpali: e2e refused a connection"
    assert isinstance(caught.value.__cause__, httpx.ConnectError)
    validator.close()
