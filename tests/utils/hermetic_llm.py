"""Exact chat LLMs for integration tests, served by Modal.

``ensure_llm()`` resolves a role's model (``MODEL``, the primary student, or
``TEACHER_MODEL``) to the service that serves it: an explicit
``INFERENCE_SERVICE_URLS`` entry for that service when one is set, otherwise
the service's Modal deployment, read through the repo's Modal lifecycle. The
endpoint is accepted only when its authenticated model list names the exact
model and revision. No model is ever started on this host: an absent
deployment or a wrong model raises a typed ``LlmUnavailableError`` naming what
to deploy. Each model resolves once per process; every decision is recorded
for the terminal summary. ``activate_llms()`` then writes the selected roles
into the session config.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Callable

import httpx
from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointCredentials,
    EndpointIdentityEvidence,
    EndpointResolutionError,
    ResolvedInferenceEndpoint,
    resolve_endpoint,
)

from cogniverse_foundation.config.inference_auth import is_modal_inference_url
from cogniverse_foundation.inference_specs import (
    INFERENCE_SERVICE_SPECS,
    InferenceServiceSpec,
)
from cogniverse_runtime.inference_services import parse_inference_service_urls
from tests.utils.model_resolution import ModelResolution, record

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = REPO_ROOT / "configs" / "config.json"
CHAT_SERVICES = ("vllm_llm_student", "vllm_llm_teacher")


def _role_model(role: str) -> str:
    """Return the exact model the shipped config serves for ``role``."""
    model = json.loads(SOURCE_CONFIG.read_text())["llm_config"][role]["model"]
    return model[len("openai/") :] if model.startswith("openai/") else model


MODEL = _role_model("primary")
TEACHER_MODEL = _role_model("teacher")
HERMETIC_CONFIG_DIR = REPO_ROOT / "outputs" / ".hermetic"


class LlmUnavailableError(RuntimeError):
    """No remote endpoint serves the requested chat model exactly."""

    def __init__(self, service: str, model: str, detail: str, remedy: str) -> None:
        self.service = service
        self.model = model
        self.detail = detail
        super().__init__(
            f"{service} ({model}): {detail}. Tests never start a model on this "
            f"host; {remedy}."
        )


class ModalLlmNotDeployedError(LlmUnavailableError):
    """Modal could not name an endpoint for the service."""


class LlmEndpointMismatchError(LlmUnavailableError):
    """An endpoint answered but did not serve the exact model and revision."""


def service_for_model(model: str) -> InferenceServiceSpec:
    """The chat service whose spec serves ``model`` exactly."""
    for service in CHAT_SERVICES:
        spec = INFERENCE_SERVICE_SPECS[service]
        if spec.model_id == model:
            return spec
    raise ValueError(
        f"No chat inference service serves {model!r}; served models: "
        f"{sorted(INFERENCE_SERVICE_SPECS[s].model_id for s in CHAT_SERVICES)}"
    )


def _deploy_remedy(service: str) -> str:
    return (
        f"deploy it with `uv run cogniverse inference modal deploy {service}` "
        f"(check with `uv run cogniverse inference modal status {service}`)"
    )


def _modal_web_url(spec: InferenceServiceSpec, credentials: EndpointCredentials):
    from cogniverse_cli.modal_inference_lifecycle import ModalInferenceLifecycle

    with ModalInferenceLifecycle(credentials=credentials) as lifecycle:
        return lifecycle.status((spec.name,))[0].web_url


def _http_client(spec: InferenceServiceSpec) -> httpx.Client:
    # A scaled-to-zero Modal app answers its first request after booting, so
    # the identity request gets the deployment's own cold-start budget.
    return httpx.Client(timeout=spec.boot_deadline_seconds)


class LlmResolver:
    """Resolve each chat model once per process; concurrent callers share it."""

    def __init__(
        self,
        *,
        modal_web_url: Callable[
            [InferenceServiceSpec, EndpointCredentials], str
        ] = _modal_web_url,
        http_client: Callable[[InferenceServiceSpec], httpx.Client] = _http_client,
    ) -> None:
        self._modal_web_url = modal_web_url
        self._http_client = http_client
        self._lock = threading.Lock()
        self._outcomes: dict[str, Future[ResolvedInferenceEndpoint]] = {}

    def resolve(self, model: str) -> ResolvedInferenceEndpoint:
        spec = service_for_model(model)
        with self._lock:
            future = self._outcomes.get(model)
            owner = future is None
            if owner:
                future = Future()
                self._outcomes[model] = future
        if owner:
            try:
                future.set_result(self._resolve_once(spec))
            except BaseException as exc:
                future.set_exception(exc)
        return future.result()

    def _resolve_once(self, spec: InferenceServiceSpec) -> ResolvedInferenceEndpoint:
        subject = f"LM {spec.model_id}"
        token = os.environ.get("COGNIVERSE_INFERENCE_API_KEY")
        credentials = EndpointCredentials(bearer_token=token)
        explicit = (
            parse_inference_service_urls(os.environ.get("INFERENCE_SERVICE_URLS")) or {}
        ).get(spec.name)
        source = (
            "INFERENCE_SERVICE_URLS"
            if explicit is not None
            else f"modal {spec.modal_app}"
        )
        try:
            if not token:
                raise ModalLlmNotDeployedError(
                    spec.name,
                    spec.model_id,
                    "COGNIVERSE_INFERENCE_API_KEY is not set, so no Modal "
                    "endpoint can be authenticated",
                    "put the key in .env/COGNIVERSE_INFERENCE_API_KEY.env",
                )
            if explicit is not None:
                base_url = explicit
            else:
                try:
                    base_url = self._modal_web_url(spec, credentials)
                except Exception as exc:
                    raise ModalLlmNotDeployedError(
                        spec.name,
                        spec.model_id,
                        f"Modal names no endpoint for {spec.modal_app}: {exc}",
                        _deploy_remedy(spec.name),
                    ) from exc
            try:
                candidate = CandidateEndpoint(
                    provider="modal" if is_modal_inference_url(base_url) else "local",
                    base_url=base_url,
                    credentials=credentials,
                    identity_evidence=EndpointIdentityEvidence.ENDPOINT,
                )
                with self._http_client(spec) as client:
                    endpoint = resolve_endpoint(spec, explicit=candidate, client=client)
            except (EndpointResolutionError, ValueError, httpx.HTTPError) as exc:
                raise LlmEndpointMismatchError(
                    spec.name,
                    spec.model_id,
                    f"{source} endpoint {base_url} does not serve it exactly: "
                    f"{type(exc).__name__}: {exc}",
                    _deploy_remedy(spec.name),
                ) from exc
        except LlmUnavailableError as exc:
            record(ModelResolution(subject, "refused", None, (source,), exc.detail))
            raise
        record(
            ModelResolution(subject, "resolved-remote", endpoint.base_url, (source,))
        )
        return endpoint


_RESOLVER = LlmResolver()


def ensure_llm_endpoint(model: str = MODEL) -> ResolvedInferenceEndpoint:
    """The verified remote endpoint serving ``model`` exactly."""
    return _RESOLVER.resolve(model)


def ensure_llm(model: str = MODEL) -> str:
    """The OpenAI base URL (``…/v1``) of the endpoint serving ``model``."""
    return f"{ensure_llm_endpoint(model).base_url}/v1"


def _write_session_config(
    primary_api_base: str,
    teacher_api_base: str | None,
    *,
    source_config: Path = SOURCE_CONFIG,
) -> Path:
    config = json.loads(source_config.read_text())
    llm = config.setdefault("llm_config", {})
    primary = llm.setdefault("primary", {})
    primary["model"] = f"openai/{MODEL}"
    primary["api_base"] = primary_api_base
    # LLMConfig.from_dict requires a teacher entry, so config load must
    # succeed even in primary-only sessions. Point an unprovisioned teacher
    # at the dead sentinel port (nothing ever listens there — see the
    # BACKEND_PORT fixture in tests/conftest.py) so any teacher call outside
    # a requires_teacher_model test fails at connect.
    teacher = llm.setdefault("teacher", {})
    teacher["model"] = f"openai/{TEACHER_MODEL}"
    teacher["api_base"] = teacher_api_base or "http://127.0.0.1:29071/v1"
    HERMETIC_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = HERMETIC_CONFIG_DIR / f"config-{os.getpid()}.json"
    pending = config_path.with_name(f".{config_path.name}.{threading.get_ident()}.tmp")
    try:
        pending.write_text(json.dumps(config, indent=2))
        os.replace(pending, config_path)
    finally:
        pending.unlink(missing_ok=True)
    return config_path


def _server_base(url: str) -> str:
    base = url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def activate_llms(
    primary_api_base: str,
    teacher_api_base: str | None = None,
    *,
    source_config: Path = SOURCE_CONFIG,
) -> Path:
    """Publish the verified exact LM roles selected for this process."""
    primary_api_base = f"{_server_base(primary_api_base)}/v1"
    if teacher_api_base is not None:
        teacher_api_base = f"{_server_base(teacher_api_base)}/v1"
    config_path = _write_session_config(
        primary_api_base,
        teacher_api_base,
        source_config=source_config,
    )
    os.environ["COGNIVERSE_CONFIG"] = str(config_path)
    os.environ["TEST_LLM_API_BASE"] = primary_api_base
    os.environ["TEST_LLM_MODEL"] = MODEL
    os.environ.setdefault("OPENAI_API_KEY", "not-required")
    return config_path
