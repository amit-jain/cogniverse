"""Strict authenticated discovery for exact inference endpoints."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Condition, Lock
from types import MappingProxyType, TracebackType
from typing import Literal, Mapping, Sequence

import httpx

from cogniverse_foundation.config.inference_auth import is_modal_inference_url
from cogniverse_foundation.inference_specs import (
    EndpointAuth,
    InferenceServiceSpec,
)

EndpointProvider = Literal["modal", "e2e", "dev", "local"]
_PROVIDER_ORDER: tuple[EndpointProvider, ...] = ("modal", "e2e", "dev", "local")
_RESOLVED_CACHE_CAPACITY = 16


class EndpointResolutionError(RuntimeError):
    """Base error for endpoint selection and validation failures."""


class EndpointAuthenticationError(EndpointResolutionError):
    """The selected endpoint rejected its configured credentials."""


class EndpointContractError(EndpointResolutionError):
    """The selected endpoint returned a malformed or unsupported contract."""


class EndpointServerError(EndpointResolutionError):
    """The selected endpoint reported an inference service failure."""


class EndpointTimeoutError(EndpointResolutionError):
    """The selected endpoint exceeded its model-identity request deadline."""


class EndpointUnavailableError(EndpointResolutionError):
    """No automatically discovered endpoint accepted a connection."""


class ModelIdentityError(EndpointResolutionError):
    """The endpoint does not serve the pinned production model revision."""


class EndpointIdentityEvidence(StrEnum):
    """Where the immutable model revision was verified."""

    ENDPOINT = "endpoint"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True, slots=True)
class EndpointCredentials:
    """Secret endpoint credentials whose representation is always redacted."""

    bearer_token: str | None = field(default=None, repr=False)
    modal_key: str | None = field(default=None, repr=False)
    modal_secret: str | None = field(default=None, repr=False)

    def headers(self, auth: EndpointAuth) -> Mapping[str, str]:
        """Build the exact authentication headers required by ``auth``."""

        if auth is EndpointAuth.BEARER:
            if not self.bearer_token or self.bearer_token != self.bearer_token.strip():
                raise EndpointAuthenticationError(
                    "bearer authentication requires a configured API key"
                )
            if self.modal_key is not None or self.modal_secret is not None:
                raise EndpointAuthenticationError(
                    "bearer authentication rejects Modal proxy credentials"
                )
            return MappingProxyType({"Authorization": f"Bearer {self.bearer_token}"})
        if self.bearer_token is not None:
            raise EndpointAuthenticationError(
                "Modal proxy authentication rejects bearer credentials"
            )
        if (
            not self.modal_key
            or self.modal_key != self.modal_key.strip()
            or not self.modal_secret
            or self.modal_secret != self.modal_secret.strip()
        ):
            raise EndpointAuthenticationError(
                "Modal proxy authentication requires both key and secret"
            )
        return MappingProxyType(
            {
                "Modal-Key": self.modal_key,
                "Modal-Secret": self.modal_secret,
            }
        )


@dataclass(frozen=True, slots=True)
class CandidateEndpoint:
    """One provider endpoint plus its independently verified identity evidence."""

    provider: EndpointProvider
    base_url: str
    credentials: EndpointCredentials = field(
        default_factory=EndpointCredentials,
        repr=False,
    )
    identity_evidence: EndpointIdentityEvidence = EndpointIdentityEvidence.ENDPOINT
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDER_ORDER:
            raise ValueError(f"unknown endpoint provider {self.provider!r}")
        parsed = httpx.URL(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or b"%" in parsed.raw_host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("endpoint base_url must be a root HTTP(S) URL")
        if self.provider == "modal" and not is_modal_inference_url(self.base_url):
            raise ValueError("Modal endpoint must be an HTTPS *.modal.run root URL")
        normalized = str(parsed).rstrip("/")
        object.__setattr__(self, "base_url", normalized)


@dataclass(frozen=True, slots=True)
class ResolvedInferenceEndpoint:
    """Validated endpoint ready for a production Cogniverse client."""

    service: str
    provider: EndpointProvider
    base_url: str
    headers: Mapping[str, str] = field(repr=False)
    model_id: str
    model_revision: str


class EndpointResolver:
    """Resolve endpoints once per exact service/candidate contract."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=10)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._closing = False
        self._closed = False
        self._active_resolutions = 0
        self._resolved: OrderedDict[
            tuple[InferenceServiceSpec, CandidateEndpoint], ResolvedInferenceEndpoint
        ] = OrderedDict()
        self._inflight: dict[
            tuple[InferenceServiceSpec, CandidateEndpoint],
            Future[ResolvedInferenceEndpoint],
        ] = {}

    def __enter__(self) -> EndpointResolver:
        with self._condition:
            if self._closing or self._closed:
                raise EndpointResolutionError("endpoint resolver is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Wait for active resolutions and close only an owned HTTP client."""

        with self._condition:
            if self._closed:
                return
            while self._closing:
                self._condition.wait_for(lambda: not self._closing)
                if self._closed:
                    return
            self._closing = True
            self._condition.wait_for(lambda: self._active_resolutions == 0)

        try:
            if self._owns_client:
                self._client.close()
        except BaseException:
            with self._condition:
                self._closing = False
                self._condition.notify_all()
            raise
        else:
            with self._condition:
                self._closed = True
                self._closing = False
                self._condition.notify_all()

    def resolve(
        self,
        spec: InferenceServiceSpec,
        *,
        explicit: CandidateEndpoint | None = None,
        candidates: Sequence[CandidateEndpoint] = (),
    ) -> ResolvedInferenceEndpoint:
        """Validate an explicit endpoint or discover one in provider order."""

        self._begin_resolution()
        try:
            return self._resolve(
                spec,
                explicit=explicit,
                candidates=candidates,
            )
        finally:
            self._finish_resolution()

    def _resolve(
        self,
        spec: InferenceServiceSpec,
        *,
        explicit: CandidateEndpoint | None,
        candidates: Sequence[CandidateEndpoint],
    ) -> ResolvedInferenceEndpoint:

        if explicit is not None:
            try:
                return self._resolve_once(spec, explicit)
            except httpx.ConnectError as exc:
                raise EndpointUnavailableError(
                    f"{spec.name}: explicit {explicit.provider} endpoint "
                    "refused a connection"
                ) from exc

        ordered = sorted(
            candidates,
            key=lambda candidate: _PROVIDER_ORDER.index(candidate.provider),
        )
        connection_failures: list[str] = []
        for candidate in ordered:
            try:
                return self._resolve_once(spec, candidate)
            except httpx.ConnectError as exc:
                connection_failures.append(f"{candidate.provider}: {exc}")

        detail = "; ".join(connection_failures) or "no candidates discovered"
        order = " -> ".join(_PROVIDER_ORDER)
        raise EndpointUnavailableError(
            f"{spec.name}: no endpoint available in provider order {order}: {detail}"
        )

    def _begin_resolution(self) -> None:
        with self._condition:
            if self._closing or self._closed:
                raise EndpointResolutionError("endpoint resolver is closed")
            self._active_resolutions += 1

    def _finish_resolution(self) -> None:
        with self._condition:
            self._active_resolutions -= 1
            if self._active_resolutions == 0:
                self._condition.notify_all()

    def _resolve_once(
        self,
        spec: InferenceServiceSpec,
        candidate: CandidateEndpoint,
    ) -> ResolvedInferenceEndpoint:
        cache_key = (spec, candidate)
        with self._lock:
            cached = self._resolved.get(cache_key)
            if cached is not None:
                self._resolved.move_to_end(cache_key)
                return cached
            future = self._inflight.get(cache_key)
            owner = future is None
            if future is None:
                future = Future()
                self._inflight[cache_key] = future

        if not owner:
            return future.result()

        try:
            resolved = validate_endpoint(spec, candidate, client=self._client)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            with self._lock:
                self._resolved[cache_key] = resolved
                self._resolved.move_to_end(cache_key)
                if len(self._resolved) > _RESOLVED_CACHE_CAPACITY:
                    self._resolved.popitem(last=False)
            future.set_result(resolved)
            return resolved
        finally:
            with self._lock:
                if self._inflight.get(cache_key) is future:
                    del self._inflight[cache_key]


def validate_endpoint(
    spec: InferenceServiceSpec,
    candidate: CandidateEndpoint,
    *,
    client: httpx.Client,
) -> ResolvedInferenceEndpoint:
    """Verify authentication and exact model identity at one live endpoint."""

    if (
        candidate.provider == "modal"
        and candidate.identity_evidence is not EndpointIdentityEvidence.ENDPOINT
    ):
        raise ModelIdentityError(
            f"{spec.name}: modal must report its pinned revision at the endpoint"
        )
    if candidate.identity_evidence is EndpointIdentityEvidence.DEPLOYMENT:
        if candidate.model_revision != spec.model_revision:
            raise ModelIdentityError(
                f"{spec.name}: deployment revision {candidate.model_revision!r} "
                f"does not match expected {spec.model_revision!r}"
            )

    try:
        headers = candidate.credentials.headers(spec.auth)
    except EndpointAuthenticationError as exc:
        raise EndpointAuthenticationError(
            f"{spec.name}: {candidate.provider} {exc}"
        ) from exc
    try:
        response = client.get(
            f"{candidate.base_url}{spec.models_path}",
            headers=headers,
        )
    except httpx.TimeoutException as exc:
        raise EndpointTimeoutError(
            f"{spec.name}: {candidate.provider} model identity request timed out"
        ) from exc
    if response.status_code in {401, 403}:
        raise EndpointAuthenticationError(
            f"{spec.name}: {candidate.provider} authentication failed "
            f"with HTTP {response.status_code}"
        )
    if response.status_code >= 500:
        raise EndpointServerError(
            f"{spec.name}: {candidate.provider} model identity request failed "
            f"with HTTP {response.status_code}"
        )
    if response.status_code != 200:
        raise EndpointContractError(
            f"{spec.name}: {candidate.provider} model identity request returned "
            f"HTTP {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise EndpointContractError(
            f"{spec.name}: {candidate.provider} returned non-JSON model identity"
        ) from exc

    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != 1:
        raise EndpointContractError(
            f"{spec.name}: {candidate.provider} must report exactly one model"
        )
    record = records[0]
    if not isinstance(record, dict) or record.get("id") != spec.model_id:
        actual = record.get("id") if isinstance(record, dict) else None
        raise ModelIdentityError(
            f"{spec.name}: expected model {spec.model_id!r}, got {actual!r}"
        )

    reported_revision = record.get("revision")
    if candidate.identity_evidence is EndpointIdentityEvidence.ENDPOINT:
        if reported_revision != spec.model_revision:
            raise ModelIdentityError(
                f"{spec.name}: expected revision {spec.model_revision!r}, "
                f"got {reported_revision!r}"
            )
    elif reported_revision is not None and reported_revision != spec.model_revision:
        raise ModelIdentityError(
            f"{spec.name}: endpoint revision {reported_revision!r} conflicts "
            f"with deployment revision {spec.model_revision!r}"
        )

    return ResolvedInferenceEndpoint(
        service=spec.name,
        provider=candidate.provider,
        base_url=candidate.base_url,
        headers=headers,
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )


def resolve_endpoint(
    spec: InferenceServiceSpec,
    *,
    explicit: CandidateEndpoint | None = None,
    candidates: Sequence[CandidateEndpoint] = (),
    client: httpx.Client | None = None,
) -> ResolvedInferenceEndpoint:
    """Resolve one endpoint without retaining cross-call cache state."""

    with EndpointResolver(client) as resolver:
        return resolver.resolve(
            spec,
            explicit=explicit,
            candidates=candidates,
        )
