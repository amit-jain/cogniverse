"""Collection-driven exact inference services for integration tests."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import httpx
import pytest
from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointAuthenticationError,
    EndpointContractError,
    EndpointCredentials,
    EndpointIdentityEvidence,
    EndpointServerError,
    EndpointTimeoutError,
    ModelIdentityError,
    ResolvedInferenceEndpoint,
)
from cogniverse_cli.modal_inference_config import (
    INFERENCE_SERVICE_SPECS,
    InferenceServiceSpec,
    get_inference_service_spec,
)

_PROVIDER_ORDER = ("e2e", "dev", "modal", "local")
_REQUIRED_SERVICES_ATTR = "_cogniverse_required_inference_services"
_MODAL_SERVICES_ATTR = "_cogniverse_modal_inference_services"
TEST_INFERENCE_API_KEY = "cogniverse-test-inference"
_SERVICE_DEPENDENCIES = {
    "vllm_colpali": frozenset({"vllm_asr"}),
    "videoprism_jax": frozenset({"vllm_asr"}),
}
_VLLM_ARGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "vllm_colpali": (
            "--max-model-len",
            "4096",
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--limit-mm-per-prompt",
            '{"video":0,"image":1}',
        ),
        "denseon": (
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--dtype",
            "float32",
        ),
        "vllm_llm_student": (
            "--max-model-len",
            "8192",
            "--enforce-eager",
            "--max-num-seqs",
            "1",
            "--limit-mm-per-prompt",
            '{"video":0,"image":4}',
        ),
        "vllm_asr": ("--runner", "generate", "--max-model-len", "448"),
    }
)


class ProviderUnavailable(RuntimeError):
    """A named provider has no reachable exact endpoint for the service."""


def _immutable_headers(candidate: CandidateEndpoint, spec: InferenceServiceSpec):
    return candidate.credentials.headers(spec.auth)


class EndpointValidator:
    """Validate one test endpoint against the canonical model contract."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=10)
        self._condition = Condition(Lock())
        self._inflight = 0
        self._closed = False
        self._closing = False
        self._client_released = False

    def validate(
        self,
        spec: InferenceServiceSpec,
        candidate: CandidateEndpoint,
    ) -> ResolvedInferenceEndpoint:
        with self._condition:
            if self._closed:
                raise RuntimeError("endpoint validator is closed")
            self._inflight += 1
        try:
            return self._validate_open(spec, candidate)
        finally:
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def _validate_open(
        self,
        spec: InferenceServiceSpec,
        candidate: CandidateEndpoint,
    ) -> ResolvedInferenceEndpoint:
        if (
            candidate.provider == "modal"
            and httpx.URL(candidate.base_url).scheme != "https"
        ):
            raise EndpointAuthenticationError(
                f"{spec.name}: Modal endpoint requires HTTPS"
            )
        if (
            candidate.identity_evidence is EndpointIdentityEvidence.DEPLOYMENT
            and candidate.model_revision != spec.model_revision
        ):
            raise ModelIdentityError(
                f"{spec.name}: deployment revision {candidate.model_revision!r} "
                f"does not match expected {spec.model_revision!r}"
            )
        headers = _immutable_headers(candidate, spec)
        path = spec.models_path
        if candidate.provider != "modal" and spec.name not in _VLLM_ARGS:
            path = spec.health_path
        try:
            response = self._client.get(
                f"{candidate.base_url}{path}",
                headers=headers,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(
                f"{spec.name}: {candidate.provider} refused a connection"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EndpointTimeoutError(
                f"{spec.name}: {candidate.provider} validation timed out"
            ) from exc
        if response.status_code in {401, 403}:
            raise EndpointAuthenticationError(
                f"{spec.name}: {candidate.provider} authentication failed with "
                f"HTTP {response.status_code}"
            )
        if response.status_code >= 500:
            raise EndpointServerError(
                f"{spec.name}: {candidate.provider} validation failed with "
                f"HTTP {response.status_code}"
            )
        if response.status_code != 200:
            raise EndpointContractError(
                f"{spec.name}: {candidate.provider} validation returned "
                f"HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise EndpointContractError(
                f"{spec.name}: {candidate.provider} returned non-JSON identity"
            ) from exc

        if path == spec.models_path:
            self._validate_models(spec, candidate, payload)
        else:
            self._validate_health(spec, candidate, payload)
        return ResolvedInferenceEndpoint(
            service=spec.name,
            provider=candidate.provider,
            base_url=candidate.base_url,
            headers=headers,
            model_id=spec.model_id,
            model_revision=spec.model_revision,
        )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            while self._inflight:
                self._condition.wait()
            while self._closing:
                self._condition.wait()
            if self._client_released:
                return
            if not self._owns_client:
                self._client_released = True
                return
            self._closing = True
        try:
            self._client.close()
        except Exception as exc:
            with self._condition:
                self._closing = False
                self._condition.notify_all()
            raise RuntimeError(
                f"endpoint validator client close failed: {exc}"
            ) from exc
        with self._condition:
            self._client_released = True
            self._closing = False
            self._condition.notify_all()

    @staticmethod
    def _validate_models(spec, candidate, payload) -> None:
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != 1:
            raise EndpointContractError(
                f"{spec.name}: {candidate.provider} must report exactly one model"
            )
        record = rows[0]
        actual = record.get("id") if isinstance(record, dict) else None
        if actual != spec.model_id:
            raise ModelIdentityError(
                f"{spec.name}: expected model {spec.model_id!r}, got {actual!r}"
            )
        revision = record.get("revision")
        if (
            candidate.identity_evidence is EndpointIdentityEvidence.ENDPOINT
            and revision != spec.model_revision
        ):
            raise ModelIdentityError(
                f"{spec.name}: expected revision {spec.model_revision!r}, "
                f"got {revision!r}"
            )
        if revision is not None and revision != spec.model_revision:
            raise ModelIdentityError(
                f"{spec.name}: endpoint revision {revision!r} conflicts with "
                f"{spec.model_revision!r}"
            )

    @staticmethod
    def _validate_health(spec, candidate, payload) -> None:
        if not isinstance(payload, dict) or payload.get("status") != "ready":
            raise EndpointContractError(
                f"{spec.name}: {candidate.provider} health must report status=ready"
            )
        # Read the same key the runtime's boot probe reads, and only that key,
        # and demand it unconditionally the way the probe does — an alias, or a
        # payload that simply omits the key, lets a server ship something the
        # runtime cannot identify while every test stays green.
        actual_model = payload.get("model")
        if actual_model != spec.model_id:
            raise ModelIdentityError(
                f"{spec.name}: expected model {spec.model_id!r}, got {actual_model!r}"
            )
        revision = payload.get("model_revision")
        if (
            candidate.identity_evidence is EndpointIdentityEvidence.ENDPOINT
            and revision != spec.model_revision
        ):
            raise ModelIdentityError(
                f"{spec.name}: expected revision {spec.model_revision!r}, "
                f"got {revision!r}"
            )
        if revision is not None and revision != spec.model_revision:
            raise ModelIdentityError(
                f"{spec.name}: expected revision {spec.model_revision!r}, "
                f"got {revision!r}"
            )


class ExplicitEndpointProvider:
    name = "explicit"

    def __init__(
        self,
        endpoints: Mapping[str, CandidateEndpoint],
        validator: EndpointValidator | None = None,
    ) -> None:
        self._endpoints = MappingProxyType(dict(endpoints))
        self._owns_validator = validator is None
        self._validator = validator if validator is not None else EndpointValidator()

    def has_service(self, service: str) -> bool:
        return service in self._endpoints

    def resolve(self, spec: InferenceServiceSpec):
        return self._validator.validate(spec, self._endpoints[spec.name])

    def close(self) -> None:
        if self._owns_validator:
            self._validator.close()


class DiscoveredEndpointProvider:
    """Validate URLs found for one automatic provider."""

    def __init__(
        self,
        name: str,
        discover: Callable[[InferenceServiceSpec], Sequence[str]],
        validator: EndpointValidator | None = None,
        credentials: EndpointCredentials | None = None,
    ) -> None:
        if name not in _PROVIDER_ORDER:
            raise ValueError(f"unknown inference provider {name!r}")
        self.name = name
        self._discover = discover
        self._owns_validator = validator is None
        self._validator = validator if validator is not None else EndpointValidator()
        self._credentials = credentials or EndpointCredentials(
            bearer_token=TEST_INFERENCE_API_KEY
        )

    def resolve(self, spec: InferenceServiceSpec):
        failures: list[str] = []
        for url in self._discover(spec):
            candidate = CandidateEndpoint(
                provider=self.name,
                base_url=url,
                credentials=self._credentials,
                identity_evidence=EndpointIdentityEvidence.ENDPOINT,
            )
            try:
                endpoint = self._validator.validate(spec, candidate)
            except ProviderUnavailable as exc:
                failures.append(str(exc))
            else:
                return endpoint
        if failures:
            raise ProviderUnavailable("; ".join(failures))
        return None

    def close(self) -> None:
        errors: list[str] = []
        if self._owns_validator:
            try:
                self._validator.close()
            except Exception as exc:
                errors.append(f"endpoint validator: {exc}")
        if errors:
            raise RuntimeError(
                f"{self.name} inference release failed: {'; '.join(errors)}"
            )


class ModalEndpointProvider:
    name = "modal"

    def __init__(self, lifecycle=None) -> None:
        self._lifecycle = lifecycle
        self._warmed: set[str] = set()

    def resolve(self, spec: InferenceServiceSpec):
        if self._lifecycle is None:
            return None
        endpoint = self._lifecycle.warm((spec.name,))[0]
        self._warmed.add(spec.name)
        return endpoint

    def close(self) -> None:
        if self._lifecycle is not None and self._warmed:
            self._lifecycle.release(tuple(sorted(self._warmed)))
            self._warmed.clear()


@dataclass(frozen=True, slots=True)
class _ContainerSpec:
    image: str
    dockerfile: str
    build_context: str
    port: int
    environment: Mapping[str, str]


_CONTAINER_SPECS = {
    "gliner": _ContainerSpec(
        "cogniverse/gliner:0.1.0-dev",
        "deploy/gliner/Dockerfile",
        ".",
        8080,
        MappingProxyType({"MODEL_NAME": INFERENCE_SERVICE_SPECS["gliner"].model_id}),
    ),
    "videoprism_jax": _ContainerSpec(
        "cogniverse/videoprism:0.1.0-dev",
        "deploy/videoprism/Dockerfile",
        ".",
        7999,
        MappingProxyType(
            {
                "MODEL_NAME": INFERENCE_SERVICE_SPECS["videoprism_jax"].model_id,
                "JAX_PLATFORM_NAME": "cpu",
                "JAX_PLATFORMS": "cpu",
            }
        ),
    ),
    "clap_embed": _ContainerSpec(
        "cogniverse/clap-embed:0.1.0-dev",
        "deploy/clap_embed/Dockerfile",
        ".",
        8080,
        MappingProxyType(
            {"CLAP_EMBED_MODEL": INFERENCE_SERVICE_SPECS["clap_embed"].model_id}
        ),
    ),
    "face_embed": _ContainerSpec(
        "cogniverse/face-embed:0.1.0-dev",
        "deploy/face_embed/Dockerfile",
        ".",
        8080,
        MappingProxyType(
            {"FACE_EMBED_MODEL": INFERENCE_SERVICE_SPECS["face_embed"].model_id}
        ),
    ),
    # Both LateOn services run the same PyLate image with their own pinned
    # model; the server performs PyLate's exact query expansion, which the
    # vLLM /pooling path cannot reproduce (no attention-mask input).
    "colbert_pylate": _ContainerSpec(
        "cogniverse/pylate:0.1.0-dev",
        "deploy/pylate/Dockerfile",
        ".",
        8080,
        MappingProxyType(
            {
                "MODEL_NAME": INFERENCE_SERVICE_SPECS["colbert_pylate"].model_id,
                "MODEL_REVISION": (
                    INFERENCE_SERVICE_SPECS["colbert_pylate"].model_revision
                ),
                "DEVICE": "cpu",
            }
        ),
    ),
    "code_colbert_pylate": _ContainerSpec(
        "cogniverse/pylate:0.1.0-dev",
        "deploy/pylate/Dockerfile",
        ".",
        8080,
        MappingProxyType(
            {
                "MODEL_NAME": INFERENCE_SERVICE_SPECS["code_colbert_pylate"].model_id,
                "MODEL_REVISION": (
                    INFERENCE_SERVICE_SPECS["code_colbert_pylate"].model_revision
                ),
                "DEVICE": "cpu",
            }
        ),
    ),
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LocalEndpointProvider:
    name = "local"

    def __init__(
        self,
        validator: EndpointValidator | None = None,
        credentials: EndpointCredentials | None = None,
        llm_ensurer: Callable[[str, str], str] | None = None,
        llm_active: Callable[[str, str], bool] | None = None,
        llm_releaser: Callable[[], None] | None = None,
    ) -> None:
        from tests.utils.vllm_sidecar import VllmSidecarFactory

        if llm_ensurer is not None and (llm_active is None or llm_releaser is None):
            raise ValueError(
                "custom Gemma provisioning requires active and release callbacks"
            )

        self._owns_validator = validator is None
        self._validator = validator if validator is not None else EndpointValidator()
        self._credentials = credentials or EndpointCredentials(
            bearer_token=TEST_INFERENCE_API_KEY
        )
        self._llm_ensurer = llm_ensurer
        self._llm_active = llm_active
        self._llm_releaser = llm_releaser
        self._owns_llm = False
        self._vllm = VllmSidecarFactory(configured_urls=())
        self._containers: list[str] = []

    def resolve(self, spec: InferenceServiceSpec):
        if spec.name == "vllm_llm_student" and self._llm_ensurer is not None:
            self._owns_llm = not self._llm_active(
                spec.model_id,
                spec.model_revision,
            )
            try:
                url = self._llm_ensurer(
                    spec.model_id,
                    spec.model_revision,
                ).rstrip("/")
            except Exception as exc:
                try:
                    self._release_owned_llm()
                except Exception as cleanup_exc:
                    exc.add_note(
                        "vllm_llm_student cleanup failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                raise
            if url.endswith("/v1"):
                url = url[: -len("/v1")]
        elif spec.name in _VLLM_ARGS:
            url = self._vllm.spawn(
                spec.model_id,
                extra_args=[
                    "--revision",
                    spec.model_revision,
                    *_VLLM_ARGS[spec.name],
                ],
                env=None,
            )
        else:
            url = self._start_container(spec)
        candidate = CandidateEndpoint(
            provider="local",
            base_url=url,
            credentials=self._credentials,
            identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
            model_revision=spec.model_revision,
        )
        return self._validator.validate(spec, candidate)

    def _release_owned_llm(self) -> None:
        if not self._owns_llm:
            return
        assert self._llm_releaser is not None
        self._llm_releaser()
        self._owns_llm = False

    def _start_container(self, spec: InferenceServiceSpec) -> str:
        try:
            container_spec = _CONTAINER_SPECS[spec.name]
        except KeyError as exc:
            raise ProviderUnavailable(
                f"{spec.name}: no exact test-owned local service is defined"
            ) from exc
        repo = Path(__file__).resolve().parents[2]
        dockerfile = repo / container_spec.dockerfile
        build_context = (repo / container_spec.build_context).resolve()
        # One bounded retry: image builds download hundreds of MB of wheels,
        # and a single transient registry/PyPI read-timeout must not sink the
        # whole session (every already-built layer stays cached, so the retry
        # only repeats the step that failed).
        build_error: Exception | None = None
        for attempt in range(2):
            try:
                subprocess.run(
                    [
                        "docker",
                        "build",
                        "-f",
                        str(dockerfile),
                        "-t",
                        container_spec.image,
                        str(build_context),
                    ],
                    check=True,
                    timeout=1800,
                )
                build_error = None
                break
            except (OSError, subprocess.SubprocessError) as exc:
                build_error = exc
        if build_error is not None:
            raise ProviderUnavailable(
                f"{spec.name}: Docker image build failed twice using {dockerfile} "
                f"with context {build_context} "
                f"({type(build_error).__name__}: {build_error})"
            ) from build_error
        from tests.utils.vllm_sidecar import (
            CONTAINER_HF_CACHE,
            writable_test_hf_cache,
        )

        port = _free_port()
        container = f"cogniverse-{spec.name}-test-{uuid.uuid4().hex[:8]}"
        # Run as the invoking user against the test-owned cache so model
        # downloads stay user-owned on the host. HOME inside the mount keeps
        # every ~-derived cache path writable for the arbitrary uid;
        # LOGNAME/USER keep getpass.getuser() working for a uid with no
        # container passwd entry (torch inductor derives its cache dir from
        # it and crashes on KeyError otherwise).
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:{container_spec.port}",
            "--oom-score-adj=500",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"HOME={CONTAINER_HF_CACHE}",
            "-e",
            f"HF_HOME={CONTAINER_HF_CACHE}",
            "-e",
            "LOGNAME=cogniverse",
            "-e",
            "USER=cogniverse",
            "-v",
            f"{writable_test_hf_cache()}:{CONTAINER_HF_CACHE}",
        ]
        for key, value in container_spec.environment.items():
            command.extend(("-e", f"{key}={value}"))
        command.append(container_spec.image)
        try:
            subprocess.run(command, check=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            logs = self._container_logs(container)
            cleanup_error = self._remove_container(container)
            detail = getattr(exc, "stderr", None) or str(exc)
            cleanup_context = (
                f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
            )
            raise RuntimeError(
                f"{spec.name}: local model {spec.model_id} launch failed: "
                f"{detail}; logs: {logs}{cleanup_context}"
            ) from exc
        self._containers.append(container)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 1800
        last_probe_error = "no health response"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{url}{spec.health_path}", timeout=5)
                if response.status_code == 200:
                    return url
                last_probe_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_probe_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
        logs = self._container_logs(container)
        cleanup_error = self._remove_container(container)
        cleanup_context = f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
        raise RuntimeError(
            f"{spec.name}: local model {spec.model_id} did not become ready in "
            f"1800s: {last_probe_error}; logs: {logs}{cleanup_context}"
        )

    @staticmethod
    def _container_logs(container: str) -> str:
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", "200", container],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable ({type(exc).__name__}: {exc})"
        detail = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        ).strip()
        return detail or "empty"

    def _remove_container(self, container: str) -> str | None:
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
            if result.returncode != 0:
                detail = "\n".join(
                    part for part in (result.stdout, result.stderr) if part
                ).strip()
                error = f"docker exited {result.returncode}: {detail}"
        if error is None and container in self._containers:
            self._containers.remove(container)
        return error

    def close(self) -> None:
        errors: list[str] = []
        try:
            self._vllm.teardown()
        except Exception as exc:
            errors.append(str(exc))
        try:
            self._release_owned_llm()
        except Exception as exc:
            errors.append(f"vllm_llm_student: {exc}")
        for container in tuple(self._containers):
            cleanup_error = self._remove_container(container)
            if cleanup_error:
                errors.append(f"{container}: {cleanup_error}")
        if self._owns_validator:
            try:
                self._validator.close()
            except Exception as exc:
                errors.append(f"endpoint validator: {exc}")
        if errors:
            raise RuntimeError("inference cleanup failed: " + "; ".join(errors))


class InferenceSessionResolver:
    """Resolve each service once and close every provider once."""

    def __init__(
        self,
        *,
        providers: Sequence[object],
        explicit_endpoints: Mapping[str, CandidateEndpoint] | None = None,
        modal_services: Iterable[str] = (),
    ) -> None:
        self._providers = tuple(providers)
        self._explicit = ExplicitEndpointProvider(explicit_endpoints or {})
        self._modal_services = frozenset(modal_services)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._resolved: dict[str, ResolvedInferenceEndpoint] = {}
        self._inflight: dict[str, Future[ResolvedInferenceEndpoint]] = {}
        self._closed = False
        self._closing = False
        self._pending_close = list(reversed((*self._providers, self._explicit)))

    def resolve(self, service: str) -> ResolvedInferenceEndpoint:
        try:
            spec = get_inference_service_spec(service)
        except KeyError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        with self._condition:
            if self._closed:
                raise RuntimeError("inference resolver is closed")
            cached = self._resolved.get(service)
            if cached is not None:
                return cached
            future = self._inflight.get(service)
            owner = future is None
            if future is None:
                future = Future()
                self._inflight[service] = future
        if not owner:
            return future.result()
        try:
            endpoint = self._resolve_once(spec)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            with self._condition:
                if self._closed:
                    error = RuntimeError("inference resolver is closed")
                    future.set_exception(error)
                else:
                    error = None
                    self._resolved[service] = endpoint
                    future.set_result(endpoint)
            if error is not None:
                raise error
            return endpoint
        finally:
            with self._condition:
                self._inflight.pop(service, None)
                self._condition.notify_all()

    def _resolve_once(self, spec: InferenceServiceSpec):
        if self._explicit.has_service(spec.name):
            return self._explicit.resolve(spec)
        failures: list[str] = []
        providers = tuple(
            provider
            for provider in self._providers
            if (getattr(provider, "name", None) == "modal")
            == (spec.name in self._modal_services)
        )
        for provider in providers:
            try:
                endpoint = provider.resolve(spec)
            except ProviderUnavailable as exc:
                failures.append(str(exc))
                continue
            if endpoint is not None:
                return endpoint
        detail = f": {'; '.join(failures)}" if failures else ""
        provider_order = " -> ".join(
            getattr(provider, "name", type(provider).__name__) for provider in providers
        )
        raise ProviderUnavailable(
            f"{spec.name}: no exact endpoint in provider order {provider_order}{detail}"
        )

    def resolve_required(
        self,
        services: Iterable[str],
    ) -> Mapping[str, ResolvedInferenceEndpoint]:
        try:
            resolved = {service: self.resolve(service) for service in services}
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"inference cleanup failed: {cleanup_error}")
            raise
        return MappingProxyType(resolved)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            while self._inflight:
                self._condition.wait()
            while self._closing:
                self._condition.wait()
            if not self._pending_close:
                return
            self._closing = True
            pending = tuple(self._pending_close)
        errors: list[str] = []
        failed: list[object] = []
        for provider in pending:
            try:
                provider.close()
            except Exception as exc:
                failed.append(provider)
                errors.append(
                    f"{getattr(provider, 'name', type(provider).__name__)}: {exc}"
                )
        with self._condition:
            self._pending_close = failed
            self._closing = False
            self._condition.notify_all()
        if errors:
            raise RuntimeError("; ".join(errors))


def explicit_endpoints_from_environment(
    required: Iterable[str],
) -> Mapping[str, CandidateEndpoint]:
    raw = os.environ.get("INFERENCE_SERVICE_URLS")
    if raw is None:
        return MappingProxyType({})
    try:
        configured = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "INFERENCE_SERVICE_URLS must be a JSON object of service URLs"
        ) from exc
    if not isinstance(configured, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in configured.items()
    ):
        raise ValueError("INFERENCE_SERVICE_URLS must be a JSON object of service URLs")
    unknown = sorted(set(configured) - set(INFERENCE_SERVICE_SPECS))
    if unknown:
        raise ValueError(
            f"INFERENCE_SERVICE_URLS contains unknown services {unknown!r}"
        )
    token = os.environ.get("COGNIVERSE_INFERENCE_API_KEY")
    endpoints: dict[str, CandidateEndpoint] = {}
    for service in required:
        url = configured.get(service)
        if url is None:
            continue
        parsed = httpx.URL(url)
        provider = (
            "modal" if parsed.host and parsed.host.endswith(".modal.run") else "local"
        )
        if provider == "modal" and parsed.scheme != "https":
            raise EndpointAuthenticationError(
                f"{service}: explicit Modal endpoint requires HTTPS"
            )
        if provider == "modal" and not token:
            raise EndpointAuthenticationError(
                f"{service}: explicit Modal endpoint requires a configured API key"
            )
        endpoints[service] = CandidateEndpoint(
            provider=provider,
            base_url=url,
            credentials=EndpointCredentials(
                bearer_token=token or TEST_INFERENCE_API_KEY
            ),
            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
        )
    return MappingProxyType(endpoints)


def collect_required_inference_services(items) -> frozenset[str]:
    required: set[str] = set()
    for item in items:
        item_required = set(
            getattr(item, "_cogniverse_required_inference_services", ())
        )
        for marker in item.iter_markers_with_node(name="requires_inference"):
            _, requirement = marker
            if len(requirement.args) != 1 or not isinstance(requirement.args[0], str):
                raise pytest.UsageError(
                    "requires_inference must name exactly one inference service"
                )
            item_required.add(requirement.args[0])
        modal_services: set[str] = set()
        for marker in item.iter_markers_with_node(name="requires_modal_inference"):
            _, requirement = marker
            if len(requirement.args) != 1 or not isinstance(requirement.args[0], str):
                raise pytest.UsageError(
                    "requires_modal_inference must name exactly one inference service"
                )
            modal_services.add(requirement.args[0])
            item_required.add(requirement.args[0])
        unknown = sorted(item_required - set(INFERENCE_SERVICE_SPECS))
        if unknown:
            raise pytest.UsageError(f"unknown inference service {unknown[0]!r}")
        for service in tuple(item_required):
            item_required.update(_SERVICE_DEPENDENCIES.get(service, ()))
        item._cogniverse_required_inference_services = frozenset(item_required)
        setattr(item, _MODAL_SERVICES_ATTR, frozenset(modal_services))
        required.update(item_required)
    return frozenset(required)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_inference(service): require one exact named inference service",
    )
    config.addinivalue_line(
        "markers",
        "requires_modal_inference(service): require one exact named Modal service",
    )


def pytest_collection_modifyitems(config, items) -> None:
    setattr(config, _REQUIRED_SERVICES_ATTR, collect_required_inference_services(items))


def _build_resolver(
    required: Iterable[str],
    *,
    modal_services: Iterable[str] = (),
) -> InferenceSessionResolver:
    from cogniverse_cli.modal_inference_lifecycle import ModalInferenceLifecycle

    from tests.utils.vllm_sidecar import (
        _discover_dev_model_urls,
        _discover_e2e_model_urls,
    )

    required = frozenset(required)
    modal_services = frozenset(modal_services)
    unexpected_modal_services = sorted(modal_services - required)
    if unexpected_modal_services:
        raise ValueError(
            "Modal services must also be required inference services: "
            f"{unexpected_modal_services!r}"
        )
    token = os.environ.get("COGNIVERSE_INFERENCE_API_KEY")
    shared_credentials = EndpointCredentials(
        bearer_token=token or TEST_INFERENCE_API_KEY
    )
    lifecycle = (
        ModalInferenceLifecycle(credentials=EndpointCredentials(bearer_token=token))
        if token and modal_services
        else None
    )
    return InferenceSessionResolver(
        explicit_endpoints=explicit_endpoints_from_environment(required),
        providers=(
            DiscoveredEndpointProvider(
                "e2e",
                lambda spec: _discover_e2e_model_urls(spec.model_id),
                credentials=shared_credentials,
            ),
            DiscoveredEndpointProvider(
                "dev",
                lambda spec: _discover_dev_model_urls(spec.model_id),
                credentials=shared_credentials,
            ),
            ModalEndpointProvider(lifecycle),
            LocalEndpointProvider(credentials=shared_credentials),
        ),
        modal_services=modal_services,
    )


@contextmanager
def publish_inference_endpoints(
    endpoints: Mapping[str, ResolvedInferenceEndpoint],
):
    original_urls = os.environ.get("INFERENCE_SERVICE_URLS")
    original_key = os.environ.get("COGNIVERSE_INFERENCE_API_KEY")
    authorizations = {
        endpoint.headers.get("Authorization") for endpoint in endpoints.values()
    }
    if len(authorizations) != 1:
        raise RuntimeError(
            "resolved inference endpoints must share one bearer credential"
        )
    authorization = authorizations.pop()
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise RuntimeError("resolved inference endpoints require bearer authentication")
    os.environ["INFERENCE_SERVICE_URLS"] = json.dumps(
        {service: endpoint.base_url for service, endpoint in sorted(endpoints.items())},
        separators=(",", ":"),
        sort_keys=True,
    )
    os.environ["COGNIVERSE_INFERENCE_API_KEY"] = authorization.removeprefix("Bearer ")
    try:
        yield endpoints
    finally:
        if original_urls is None:
            os.environ.pop("INFERENCE_SERVICE_URLS", None)
        else:
            os.environ["INFERENCE_SERVICE_URLS"] = original_urls
        if original_key is None:
            os.environ.pop("COGNIVERSE_INFERENCE_API_KEY", None)
        else:
            os.environ["COGNIVERSE_INFERENCE_API_KEY"] = original_key


@pytest.fixture(scope="session", autouse=True)
def requested_inference_services(request):
    required = frozenset(
        service
        for item in request.session.items
        for service in getattr(
            item,
            "_cogniverse_required_inference_services",
            (),
        )
    )
    if not required:
        yield MappingProxyType({})
        return
    modal_services = frozenset(
        service
        for item in request.session.items
        for service in getattr(item, _MODAL_SERVICES_ATTR, ())
    )
    resolver = _build_resolver(required, modal_services=modal_services)
    try:
        endpoints = resolver.resolve_required(sorted(required))
        with publish_inference_endpoints(endpoints):
            yield endpoints
    finally:
        resolver.close()


@pytest.fixture(scope="session")
def resolved_inference_endpoints(requested_inference_services):
    return requested_inference_services
