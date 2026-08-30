"""Deployment and autoscaler lifecycle for Modal inference services."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from threading import Condition, Lock
from time import monotonic, sleep
from types import TracebackType
from typing import Callable, Iterable, Sequence

import httpx
import modal

from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointAuthenticationError,
    EndpointCredentials,
    EndpointIdentityEvidence,
    EndpointResolutionError,
    EndpointResolver,
    ResolvedInferenceEndpoint,
)
from cogniverse_cli.modal_inference_config import (
    InferenceServiceSpec,
    get_inference_service_spec,
)

_DEPLOYMENT_MODULES = {
    "vllm_colpali": "cogniverse_cli.modal_inference.vllm_colpali",
    "colbert_pylate": "cogniverse_cli.modal_inference.lateon",
    "code_colbert_pylate": "cogniverse_cli.modal_inference.code_colbert_pylate",
    "denseon": "cogniverse_cli.modal_inference.denseon",
    "gliner": "cogniverse_cli.modal_inference.gliner",
    "videoprism_jax": "cogniverse_cli.modal_inference.videoprism",
    "vllm_llm_student": "cogniverse_cli.modal_inference.gemma",
    "vllm_llm_teacher": "cogniverse_cli.modal_inference.teacher",
    "vllm_asr": "cogniverse_cli.modal_inference.whisper",
    "clap_embed": "cogniverse_cli.modal_inference.clap",
    "face_embed": "cogniverse_cli.modal_inference.face",
}
_RETRYABLE_HEALTH_STATUSES = frozenset({502, 503, 504})


class ModalLifecycleError(RuntimeError):
    """A Modal lifecycle operation failed without exposing credentials."""


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Observed Modal endpoint and live runner count."""

    service: str
    modal_app: str
    modal_object: str
    web_url: str
    active_containers: int


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """Deterministic GPU choice constrained by a service specification."""

    service: str
    selected_gpu: str
    considered_gpus: tuple[str, ...]


class _WarmFailure(ModalLifecycleError):
    """A failure after a service's minimum container count was raised."""


def _load_deployment(spec: InferenceServiceSpec):
    module = import_module(_DEPLOYMENT_MODULES[spec.name])
    try:
        return module.app
    except AttributeError:
        raise ModalLifecycleError(
            f"{spec.name}: deployment module does not expose a Modal app"
        ) from None


def _lookup_function(app_name: str, object_name: str):
    import modal

    return modal.Function.from_name(app_name, object_name)


def _stop_app(app_name: str) -> None:
    from modal.cli.app import stop

    stop(app_name)


def _lifecycle_operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._begin_operation()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._finish_operation()

    return guarded


class ModalInferenceLifecycle:
    """Manage Modal deployments without stopping apps during normal use."""

    def __init__(
        self,
        *,
        credentials: EndpointCredentials,
        client: httpx.Client | None = None,
        function_from_name: Callable[[str, str], object] = _lookup_function,
        deployment_loader: Callable[[InferenceServiceSpec], object] = _load_deployment,
        app_stopper: Callable[[str], None] = _stop_app,
        readiness_timeout: float = 1200,
        readiness_poll_interval: float = 1,
    ) -> None:
        self._credentials = credentials
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=30)
        self._function_from_name = function_from_name
        self._deployment_loader = deployment_loader
        self._app_stopper = app_stopper
        self._readiness_timeout = readiness_timeout
        self._readiness_poll_interval = readiness_poll_interval
        self._state_lock = Lock()
        self._lifecycle_condition = Condition(Lock())
        self._closing = False
        self._closed = False
        self._active_operations = 0
        self._service_locks: dict[str, Lock] = {}
        self._warm_endpoints: dict[str, ResolvedInferenceEndpoint] = {}

    def __enter__(self) -> ModalInferenceLifecycle:
        with self._lifecycle_condition:
            if self._closing or self._closed:
                raise ModalLifecycleError("Modal inference lifecycle is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Wait for active operations and close only an owned HTTP client."""

        with self._lifecycle_condition:
            if self._closed:
                return
            while self._closing:
                self._lifecycle_condition.wait_for(lambda: not self._closing)
                if self._closed:
                    return
            self._closing = True
            self._lifecycle_condition.wait_for(lambda: self._active_operations == 0)

        try:
            if self._owns_client:
                self._client.close()
        except BaseException:
            with self._lifecycle_condition:
                self._closing = False
                self._lifecycle_condition.notify_all()
            raise
        else:
            with self._lifecycle_condition:
                self._closed = True
                self._closing = False
                self._lifecycle_condition.notify_all()

    @_lifecycle_operation
    def deploy(self, services: Sequence[str]) -> tuple[ServiceStatus, ...]:
        """Deploy each canonical Modal app without allocating a warm replica."""

        specs = self._service_specs(services)
        try:
            with modal.enable_output():
                for spec in specs:
                    app = self._deployment_loader(spec)
                    app.deploy(name=spec.modal_app)
        except Exception as exc:
            raise ModalLifecycleError(
                f"{spec.name}: Modal deployment failed: {self._redact(exc)}"
            ) from None
        return self._statuses(specs)

    @_lifecycle_operation
    def warm(
        self,
        services: Sequence[str],
    ) -> tuple[ResolvedInferenceEndpoint, ...]:
        """Warm services and verify authenticated health and model identity."""

        specs = self._service_specs(services)
        touched: list[InferenceServiceSpec] = []
        resolved: list[ResolvedInferenceEndpoint] = []
        try:
            for spec in specs:
                try:
                    endpoint = self._warm_one(spec)
                except _WarmFailure:
                    touched.append(spec)
                    raise
                touched.append(spec)
                resolved.append(endpoint)
        except ModalLifecycleError as exc:
            cleanup_errors = self._release_specs(touched)
            if cleanup_errors:
                raise ModalLifecycleError(
                    f"{exc}; release cleanup failed: {'; '.join(cleanup_errors)}"
                ) from None
            raise
        return tuple(resolved)

    @_lifecycle_operation
    def release(self, services: Sequence[str]) -> tuple[ServiceStatus, ...]:
        """Return services to scale-to-zero without stopping their Modal apps."""

        specs = self._service_specs(services)
        errors = self._release_specs(specs)
        if errors:
            raise ModalLifecycleError("; ".join(errors))
        return self._statuses(specs)

    @_lifecycle_operation
    def status(self, services: Sequence[str]) -> tuple[ServiceStatus, ...]:
        """Read Modal web URLs and live runner counts."""

        return self._statuses(self._service_specs(services))

    @_lifecycle_operation
    def qualify(
        self,
        service: str,
        candidates: Sequence[str],
    ) -> QualificationResult:
        """Select the first configured GPU among strictly valid candidates."""

        spec = self._service_spec(service)
        if not candidates:
            raise ValueError(f"{service}: at least one GPU candidate is required")
        unsupported = [gpu for gpu in candidates if gpu not in spec.gpu_candidates]
        if unsupported:
            raise ValueError(f"{service}: unsupported GPU candidates {unsupported!r}")
        considered = tuple(gpu for gpu in spec.gpu_candidates if gpu in candidates)
        return QualificationResult(
            service=service,
            selected_gpu=considered[0],
            considered_gpus=considered,
        )

    @_lifecycle_operation
    def undeploy(self, service: str, confirmation: str) -> None:
        """Stop one Modal app only after byte-exact service confirmation."""

        spec = self._service_spec(service)
        if confirmation != service:
            raise ValueError(
                f"undeploy confirmation must exactly match service {service!r}"
            )
        try:
            self._app_stopper(spec.modal_app)
        except Exception as exc:
            raise ModalLifecycleError(
                f"{service}: Modal undeploy failed: {self._redact(exc)}"
            ) from None

    def _warm_one(
        self,
        spec: InferenceServiceSpec,
    ) -> ResolvedInferenceEndpoint:
        try:
            headers = self._credentials.headers(spec.auth)
        except EndpointAuthenticationError as exc:
            raise ModalLifecycleError(
                f"{spec.name}: endpoint credentials are invalid: {exc}"
            ) from None

        with self._service_lock(spec.name):
            with self._state_lock:
                cached = self._warm_endpoints.get(spec.name)
            if cached is not None:
                return cached

            function = self._function(spec)
            try:
                function.update_autoscaler(
                    min_containers=1,
                    scaledown_window=spec.scaledown_window,
                )
            except Exception as exc:
                raise _WarmFailure(
                    f"{spec.name}: Modal autoscaler update to min_containers=1 "
                    f"failed: {self._redact(exc)}"
                ) from None

            try:
                web_url = self._web_url(spec, function)
                self._probe_health(spec, web_url, headers)
                with EndpointResolver(self._client) as resolver:
                    endpoint = resolver.resolve(
                        spec,
                        explicit=CandidateEndpoint(
                            provider="modal",
                            base_url=web_url,
                            credentials=self._credentials,
                            identity_evidence=EndpointIdentityEvidence.ENDPOINT,
                        ),
                    )
            except ModalLifecycleError as exc:
                raise _WarmFailure(str(exc)) from None
            except (EndpointResolutionError, ValueError) as exc:
                raise _WarmFailure(self._redact(exc)) from None

            with self._state_lock:
                self._warm_endpoints[spec.name] = endpoint
            return endpoint

    def _probe_health(
        self,
        spec: InferenceServiceSpec,
        web_url: str,
        headers,
    ) -> None:
        health_url = f"{web_url.rstrip('/')}{spec.health_path}"
        deadline = monotonic() + self._readiness_timeout
        last_failure = "no response"
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                self._raise_readiness_timeout(spec, last_failure)
            try:
                response = self._client.get(
                    health_url,
                    headers=headers,
                    timeout=remaining,
                )
            except httpx.TimeoutException:
                last_failure = "request timed out"
            except httpx.ConnectError:
                last_failure = "connection failed"
            except httpx.HTTPError as exc:
                raise ModalLifecycleError(
                    f"{spec.name}: authenticated health probe failed: "
                    f"{self._redact(exc)}"
                ) from None
            else:
                if response.status_code == 200:
                    return
                if response.status_code not in _RETRYABLE_HEALTH_STATUSES:
                    raise ModalLifecycleError(
                        f"{spec.name}: authenticated health probe returned "
                        f"HTTP {response.status_code}"
                    )
                last_failure = f"HTTP {response.status_code}"

            remaining = deadline - monotonic()
            if remaining <= 0:
                self._raise_readiness_timeout(spec, last_failure)
            sleep(min(self._readiness_poll_interval, remaining))

    def _raise_readiness_timeout(
        self,
        spec: InferenceServiceSpec,
        last_failure: str,
    ) -> None:
        raise ModalLifecycleError(
            f"{spec.name}: authenticated health did not become ready within "
            f"{self._readiness_timeout:g} seconds; last failure: {last_failure}"
        ) from None

    def _release_specs(self, specs: Iterable[InferenceServiceSpec]) -> list[str]:
        errors: list[str] = []
        for spec in specs:
            with self._service_lock(spec.name):
                try:
                    function = self._function(spec)
                    function.update_autoscaler(
                        min_containers=0,
                        scaledown_window=spec.scaledown_window,
                    )
                except ModalLifecycleError as exc:
                    errors.append(str(exc))
                    continue
                except Exception as exc:
                    errors.append(
                        f"{spec.name}: Modal autoscaler update to "
                        "min_containers=0 failed: "
                        f"{self._redact(exc)}"
                    )
                    continue
                with self._state_lock:
                    self._warm_endpoints.pop(spec.name, None)
        return errors

    def _statuses(
        self,
        specs: Iterable[InferenceServiceSpec],
    ) -> tuple[ServiceStatus, ...]:
        statuses: list[ServiceStatus] = []
        for spec in specs:
            function = self._function(spec)
            web_url = self._web_url(spec, function)
            try:
                active_containers = function.get_current_stats().num_total_runners
            except Exception as exc:
                raise ModalLifecycleError(
                    f"{spec.name}: failed to read Modal function stats: "
                    f"{self._redact(exc)}"
                ) from None
            statuses.append(
                ServiceStatus(
                    service=spec.name,
                    modal_app=spec.modal_app,
                    modal_object=spec.modal_object,
                    web_url=web_url,
                    active_containers=active_containers,
                )
            )
        return tuple(statuses)

    def _function(self, spec: InferenceServiceSpec):
        try:
            return self._function_from_name(spec.modal_app, spec.modal_object)
        except Exception as exc:
            raise ModalLifecycleError(
                f"{spec.name}: failed to look up Modal function: {self._redact(exc)}"
            ) from None

    def _web_url(self, spec: InferenceServiceSpec, function) -> str:
        try:
            web_url = function.get_web_url()
        except Exception as exc:
            raise ModalLifecycleError(
                f"{spec.name}: failed to read Modal endpoint: {self._redact(exc)}"
            ) from None
        if not isinstance(web_url, str) or not web_url:
            raise ModalLifecycleError(
                f"{spec.name}: Modal function does not expose a web endpoint"
            )
        try:
            return CandidateEndpoint(
                provider="modal",
                base_url=web_url,
                credentials=self._credentials,
            ).base_url
        except ValueError as exc:
            raise ModalLifecycleError(
                f"{spec.name}: Modal function exposes an invalid web endpoint: {exc}"
            ) from None

    def _service_lock(self, service: str) -> Lock:
        with self._state_lock:
            return self._service_locks.setdefault(service, Lock())

    def _begin_operation(self) -> None:
        with self._lifecycle_condition:
            if self._closing or self._closed:
                raise ModalLifecycleError("Modal inference lifecycle is closed")
            self._active_operations += 1

    def _finish_operation(self) -> None:
        with self._lifecycle_condition:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._lifecycle_condition.notify_all()

    def _service_specs(
        self,
        services: Sequence[str],
    ) -> tuple[InferenceServiceSpec, ...]:
        if not services:
            raise ValueError("at least one inference service is required")
        if len(set(services)) != len(services):
            raise ValueError("inference service names must be unique")
        return tuple(self._service_spec(service) for service in services)

    @staticmethod
    def _service_spec(service: str) -> InferenceServiceSpec:
        try:
            return get_inference_service_spec(service)
        except KeyError:
            raise ValueError(f"unknown inference service {service!r}") from None

    def _redact(self, error: object) -> str:
        message = str(error)
        for secret in (
            self._credentials.bearer_token,
            self._credentials.modal_key,
            self._credentials.modal_secret,
        ):
            if secret:
                message = message.replace(secret, "[redacted]")
        return message
