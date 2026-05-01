"""Startup validation that inference services serve the models profiles expect.

Closes the silent-wrong-embedding failure mode: if a profile says
``embedding_model: lightonai/LateOn`` and the deployed inference service
actually serves ``lightonai/Reason-ModernColBERT``, both will happily produce
128-dim token vectors, fed to Vespa, returned at query time — but retrieval
quality will be silently wrong. This module probes each service at boot and
refuses to start if any profile references a service serving a different
model than expected.

Two server shapes are probed:
- pylate sidecar exposes ``GET /health`` → ``{"status": "ok", "model": "..."}``.
- vLLM exposes ``GET /v1/models`` → ``{"data": [{"id": "...", ...}]}``.

The probe tries both and uses whichever responds.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_BOOT_DEADLINE_SECONDS = 120.0
DEFAULT_RETRY_INTERVAL_SECONDS = 5.0

# In-cluster service-account paths. Present when the runtime runs in K8s
# with default automountServiceAccountToken; absent in unit tests and
# outside-cluster invocations.
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_SA_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_SA_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
_K8S_API_HOST = "https://kubernetes.default.svc"


def k8s_service_has_endpoints(service_url: str) -> Optional[bool]:
    """Return whether the K8s Service backing ``service_url`` has live endpoints.

    Looks up the Endpoints object for the service host via the
    in-cluster K8s API. Returns:
      * True  — at least one ready endpoint address.
      * False — Endpoints exists with no addresses (the Service is
        defined but its Deployment is at 0 replicas, all backing pods
        are unready, or every pod is terminating).
      * None  — cannot determine (no in-cluster credentials, K8s API
        unreachable, or the Service host doesn't look like a cluster
        DNS name). Caller should fall back to probing.
    """
    if not (os.path.exists(_SA_TOKEN_PATH) and os.path.exists(_SA_NAMESPACE_PATH)):
        return None

    host = urlparse(service_url).hostname
    if not host:
        return None
    # Cluster DNS shapes: "svc", "svc.namespace", "svc.namespace.svc",
    # "svc.namespace.svc.cluster.local". Take the leftmost label as the
    # service name; the namespace comes from the runtime's own SA.
    service_name = host.split(".", 1)[0]

    try:
        with open(_SA_TOKEN_PATH) as f:
            token = f.read().strip()
        with open(_SA_NAMESPACE_PATH) as f:
            namespace = f.read().strip()
    except OSError:
        return None

    api_url = f"{_K8S_API_HOST}/api/v1/namespaces/{namespace}/endpoints/{service_name}"
    try:
        resp = requests.get(
            api_url,
            headers={"Authorization": f"Bearer {token}"},
            verify=_SA_CA_PATH if os.path.exists(_SA_CA_PATH) else True,
            timeout=3,
        )
    except requests.RequestException as exc:
        logger.debug("k8s endpoints lookup for %s failed: %s", service_name, exc)
        return None
    if resp.status_code == 404:
        # No Endpoints object — Service may exist without selector, or
        # it doesn't exist at all. Either way, fall back to probing.
        return None
    if not resp.ok:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    subsets = body.get("subsets") or []
    return any(s.get("addresses") for s in subsets)


class InferenceServiceMismatch(RuntimeError):
    """A profile references a service serving the wrong model."""


@dataclass(frozen=True)
class ProfileBinding:
    """A profile's expectation about an inference service."""

    profile_name: str
    service_name: str
    expected_model: str


def probe_service_model(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Return the model id the service at ``url`` reports it is serving.

    Returns ``None`` when the service is unreachable or reports no model.
    Tries the pylate sidecar ``/health`` shape first, then vLLM ``/v1/models``.
    """
    sess = session or requests.Session()

    for path, extract in (
        ("/health", _extract_model_from_health),
        ("/v1/models", _extract_model_from_v1_models),
    ):
        try:
            resp = sess.get(f"{url.rstrip('/')}{path}", timeout=timeout_seconds)
        except requests.RequestException as exc:
            logger.debug("Probe %s%s failed: %s", url, path, exc)
            continue
        if not resp.ok:
            continue
        try:
            body = resp.json()
        except ValueError:
            continue
        model = extract(body)
        if model:
            return model
    return None


def _extract_model_from_health(body: Any) -> Optional[str]:
    if isinstance(body, dict):
        value = body.get("model")
        if isinstance(value, str) and value:
            return value
    return None


def _extract_model_from_v1_models(body: Any) -> Optional[str]:
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                value = first.get("id")
                if isinstance(value, str) and value:
                    return value
    return None


def collect_profile_bindings(profiles: dict[str, dict]) -> list[ProfileBinding]:
    """Extract (profile, service, expected_model) tuples from profile configs.

    Only includes profiles that declare ``inference_service`` (i.e., those
    configured for remote inference). Profiles without the field fall back to
    local loading and are not validated.
    """
    bindings: list[ProfileBinding] = []
    for profile_name, profile_body in profiles.items():
        if not isinstance(profile_body, dict):
            continue
        service_name = (profile_body.get("inference_services") or {}).get("embedding")
        if not service_name:
            continue
        # ``semantic_model`` wins over ``embedding_model`` for hybrid profiles
        # where the main embedding is an acoustic/visual model but a separate
        # text model handles semantic retrieval (e.g. CLAP + ColBERT audio).
        expected_model = profile_body.get("semantic_model") or profile_body.get(
            "embedding_model"
        )
        if not expected_model:
            continue
        bindings.append(
            ProfileBinding(
                profile_name=profile_name,
                service_name=service_name,
                expected_model=expected_model,
            )
        )
    return bindings


def validate_inference_services(
    bindings: Iterable[ProfileBinding],
    service_urls: dict[str, str],
    *,
    probe: Callable[[str], Optional[str]] = probe_service_model,
    boot_deadline_seconds: float = DEFAULT_BOOT_DEADLINE_SECONDS,
    retry_interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    has_endpoints: Callable[[str], Optional[bool]] = k8s_service_has_endpoints,
) -> None:
    """Raise if any profile's service serves the wrong model.

    Retries unreachable services until ``boot_deadline_seconds`` elapses —
    the runtime often starts before inference pods are ready. A service that
    never responds is a deployment failure and must fail loud.

    Services whose K8s Endpoints are empty (Deployment scaled to 0 by
    an operator or test fixture) are treated as parked: log a warning
    and skip the probe instead of blocking startup. ``has_endpoints``
    can be overridden in tests; the default in-cluster implementation
    returns ``None`` outside K8s, in which case we always probe.
    """
    bindings = list(bindings)
    if not bindings:
        logger.info("No profiles reference an inference_service; skipping probe.")
        return

    # Filter out bindings whose service isn't deployed here. A profile
    # conflict on an undeployed service is not a startup failure — the profile
    # will fail on first use with a clear error. Only conflicts on services
    # that ARE deployed matter, because those change the meaning of requests
    # the runtime will actually route.
    referenced = {b.service_name for b in bindings}
    missing = sorted(referenced - service_urls.keys())
    if missing:
        logger.warning(
            "Profiles reference inference services not deployed here: %s. "
            "Available: %s. Profiles bound to missing services will fail on "
            "first use.",
            missing,
            sorted(service_urls),
        )
    bindings = [b for b in bindings if b.service_name in service_urls]

    # Dedupe — many profiles may share one service; probe it once. Raise on
    # conflicts only among deployed services.
    unique: dict[str, str] = {}
    for b in bindings:
        existing = unique.get(b.service_name)
        if existing and existing != b.expected_model:
            raise InferenceServiceMismatch(
                f"Profiles disagree on service {b.service_name!r}: one wants "
                f"{existing!r}, another wants {b.expected_model!r}. A single "
                f"service can only serve one model."
            )
        unique[b.service_name] = b.expected_model

    deadline = now() + boot_deadline_seconds
    for service_name, expected_model in unique.items():
        url = service_urls[service_name]
        endpoints_present = has_endpoints(url)
        if endpoints_present is False:
            logger.warning(
                "Inference service %r at %s has no live endpoints "
                "(Deployment likely scaled to 0). Skipping probe; "
                "profiles bound to this service will fail on first use.",
                service_name,
                url,
            )
            continue
        served_model = _probe_until_reachable(
            probe, service_name, url, deadline, retry_interval_seconds, sleep, now
        )
        if served_model != expected_model:
            raise InferenceServiceMismatch(
                f"Inference service {service_name!r} at {url} serves "
                f"{served_model!r}, but profiles expect {expected_model!r}. "
                f"Update the Helm values ``inference.{service_name}.model`` or "
                f"change the profile's ``embedding_model``."
            )
        logger.info(
            "Inference service %r at %s verified as %s",
            service_name,
            url,
            served_model,
        )


def _probe_until_reachable(
    probe: Callable[[str], Optional[str]],
    service_name: str,
    url: str,
    deadline: float,
    retry_interval_seconds: float,
    sleep: Callable[[float], None],
    now: Callable[[], float],
) -> str:
    attempt = 0
    while True:
        attempt += 1
        served_model = probe(url)
        if served_model:
            return served_model
        remaining = deadline - now()
        if remaining <= 0:
            raise InferenceServiceMismatch(
                f"Inference service {service_name!r} at {url} did not respond "
                f"with a model identifier within the boot deadline. Check that "
                f"the pod is running and its /health or /v1/models endpoint is "
                f"reachable."
            )
        logger.info(
            "Inference service %r at %s not ready yet (attempt %d); retrying in %.1fs",
            service_name,
            url,
            attempt,
            retry_interval_seconds,
        )
        sleep(min(retry_interval_seconds, remaining))
