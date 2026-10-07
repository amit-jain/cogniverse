"""How ``ensure_llm`` resolves a chat model: Modal or an explicit URL, never local."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

from cogniverse_foundation.inference_specs import get_inference_service_spec
from tests.utils import hermetic_llm
from tests.utils.hermetic_llm import (
    MODEL,
    TEACHER_MODEL,
    LlmEndpointMismatchError,
    LlmResolver,
    ModalLlmNotDeployedError,
)

STUDENT = get_inference_service_spec("vllm_llm_student")
TEACHER = get_inference_service_spec("vllm_llm_teacher")
API_KEY = "modal-inference-secret"
MODAL_URL = "https://amit--cogniverse-vllm-llm-student-inference.modal.run"
REMEDY = (
    "Tests never start a model on this host; deploy it with "
    "`uv run cogniverse inference modal deploy vllm_llm_student` (check with "
    "`uv run cogniverse inference modal status vllm_llm_student`)."
)


def _model_list(model_id: str, revision: str | None) -> dict:
    row = {"id": model_id, "object": "model", "owned_by": "cogniverse"}
    if revision is not None:
        row["revision"] = revision
    return {"object": "list", "data": [row]}


class _ModalServing:
    """The Modal deployment boundary: the web URL lookup and the HTTP app."""

    def __init__(
        self,
        model_id: str = STUDENT.model_id,
        revision: str | None = STUDENT.model_revision,
        *,
        lookup_error: Exception | None = None,
    ) -> None:
        self.payload = _model_list(model_id, revision)
        self.lookup_error = lookup_error
        self.lookups: list[str] = []
        self.requests: list[tuple[str, str | None]] = []
        self.clients: list[float | None] = []
        self._lock = threading.Lock()

    def web_url(self, spec, credentials) -> str:
        with self._lock:
            self.lookups.append(spec.modal_app)
        if self.lookup_error is not None:
            raise self.lookup_error
        return MODAL_URL

    def client(self, spec) -> httpx.Client:
        def answer(request: httpx.Request) -> httpx.Response:
            with self._lock:
                self.requests.append(
                    (str(request.url), request.headers.get("Authorization"))
                )
            return httpx.Response(200, json=self.payload)

        self.clients.append(spec.boot_deadline_seconds)
        return httpx.Client(
            transport=httpx.MockTransport(answer),
            timeout=spec.boot_deadline_seconds,
        )

    def resolver(self) -> LlmResolver:
        return LlmResolver(modal_web_url=self.web_url, http_client=self.client)


@contextmanager
def _identity_server(model_id: str, revision: str | None):
    """A real local OpenAI model-list endpoint for the explicit override."""
    body = json.dumps(_model_list(model_id, revision)).encode()
    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def modal_env(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)


class TestRoleModelsDeriveFromShippedConfig:
    """The LM roles the tests resolve are whatever configs/config.json serves."""

    @staticmethod
    def _shipped(role: str) -> str:
        model = json.loads(hermetic_llm.SOURCE_CONFIG.read_text())["llm_config"][role][
            "model"
        ]
        return model[len("openai/") :] if model.startswith("openai/") else model

    def test_primary_role_model_is_not_restated(self) -> None:
        assert MODEL == self._shipped("primary")

    def test_teacher_role_model_is_not_restated(self) -> None:
        assert TEACHER_MODEL == self._shipped("teacher")

    def test_each_role_maps_to_its_chat_service(self) -> None:
        assert hermetic_llm.service_for_model(MODEL) == STUDENT
        assert hermetic_llm.service_for_model(TEACHER_MODEL) == TEACHER

    def test_a_model_no_chat_service_serves_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"No chat inference service serves 'x/y'"):
            hermetic_llm.service_for_model("x/y")


class TestModalResolution:
    def test_the_modal_deployment_serving_the_exact_model_is_returned(
        self, modal_env
    ) -> None:
        modal = _ModalServing()

        endpoint = modal.resolver().resolve(MODEL)

        assert (
            endpoint.service,
            endpoint.provider,
            endpoint.base_url,
            dict(endpoint.headers),
            endpoint.model_id,
            endpoint.model_revision,
        ) == (
            "vllm_llm_student",
            "modal",
            MODAL_URL,
            {"Authorization": f"Bearer {API_KEY}"},
            STUDENT.model_id,
            STUDENT.model_revision,
        )
        assert modal.lookups == ["cogniverse-vllm-llm-student"]
        assert modal.requests == [(f"{MODAL_URL}/v1/models", f"Bearer {API_KEY}")]
        # The identity request gets the deployment's cold-start budget.
        assert modal.clients == [STUDENT.boot_deadline_seconds]

    def test_the_teacher_role_resolves_its_own_deployment(self, modal_env) -> None:
        modal = _ModalServing(TEACHER.model_id, TEACHER.model_revision)

        endpoint = modal.resolver().resolve(TEACHER_MODEL)

        assert (endpoint.service, endpoint.model_id) == (
            "vllm_llm_teacher",
            TEACHER.model_id,
        )
        assert modal.lookups == ["cogniverse-vllm-llm-teacher"]

    def test_a_deployment_serving_another_model_is_refused(self, modal_env) -> None:
        modal = _ModalServing(TEACHER.model_id, TEACHER.model_revision)

        with pytest.raises(LlmEndpointMismatchError) as caught:
            modal.resolver().resolve(MODEL)

        assert str(caught.value) == (
            f"vllm_llm_student ({MODEL}): modal cogniverse-vllm-llm-student "
            f"endpoint {MODAL_URL} does not serve it exactly: ModelIdentityError: "
            f"vllm_llm_student: expected model {MODEL!r}, got {TEACHER.model_id!r}. "
            + REMEDY
        )

    def test_a_deployment_serving_another_revision_is_refused(self, modal_env) -> None:
        modal = _ModalServing(STUDENT.model_id, "0" * 40)

        with pytest.raises(LlmEndpointMismatchError) as caught:
            modal.resolver().resolve(MODEL)

        assert caught.value.detail == (
            f"modal cogniverse-vllm-llm-student endpoint {MODAL_URL} does not "
            "serve it exactly: ModelIdentityError: vllm_llm_student: expected "
            f"revision {STUDENT.model_revision!r}, got {'0' * 40!r}"
        )

    def test_an_undeployed_app_raises_naming_the_deploy_command(
        self, modal_env
    ) -> None:
        modal = _ModalServing(
            lookup_error=ModalLifecycleError(
                "vllm_llm_student: failed to read Modal endpoint: App "
                "'cogniverse-vllm-llm-student' not found in environment 'main'."
            )
        )

        with pytest.raises(ModalLlmNotDeployedError) as caught:
            modal.resolver().resolve(MODEL)

        assert str(caught.value) == (
            f"vllm_llm_student ({MODEL}): Modal names no endpoint for "
            "cogniverse-vllm-llm-student: vllm_llm_student: failed to read Modal "
            "endpoint: App 'cogniverse-vllm-llm-student' not found in environment "
            "'main'.. " + REMEDY
        )
        assert (caught.value.service, caught.value.model) == (
            "vllm_llm_student",
            MODEL,
        )
        assert modal.clients == []

    def test_a_missing_inference_key_raises_before_asking_modal(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        modal = _ModalServing()

        with pytest.raises(ModalLlmNotDeployedError) as caught:
            modal.resolver().resolve(MODEL)

        assert str(caught.value) == (
            f"vllm_llm_student ({MODEL}): COGNIVERSE_INFERENCE_API_KEY is not "
            "set, so no Modal endpoint can be authenticated. Tests never start a "
            "model on this host; put the key in "
            ".env/COGNIVERSE_INFERENCE_API_KEY.env."
        )
        assert modal.lookups == []


class TestExplicitOverride:
    def test_an_explicit_url_is_used_instead_of_modal(self, monkeypatch) -> None:
        modal = _ModalServing()
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        with _identity_server(STUDENT.model_id, STUDENT.model_revision) as (
            url,
            seen,
        ):
            monkeypatch.setenv(
                "INFERENCE_SERVICE_URLS", json.dumps({"vllm_llm_student": url})
            )
            endpoint = LlmResolver(modal_web_url=modal.web_url).resolve(MODEL)

        assert (endpoint.provider, endpoint.base_url) == ("local", url)
        assert seen == [("/v1/models", f"Bearer {API_KEY}")]
        assert modal.lookups == []

    def test_a_wrong_explicit_url_fails_without_falling_back_to_modal(
        self, monkeypatch
    ) -> None:
        modal = _ModalServing()
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
        with _identity_server(TEACHER.model_id, TEACHER.model_revision) as (url, _):
            monkeypatch.setenv(
                "INFERENCE_SERVICE_URLS", json.dumps({"vllm_llm_student": url})
            )
            with pytest.raises(LlmEndpointMismatchError) as caught:
                LlmResolver(modal_web_url=modal.web_url).resolve(MODEL)

        assert caught.value.detail == (
            f"INFERENCE_SERVICE_URLS endpoint {url} does not serve it exactly: "
            f"ModelIdentityError: vllm_llm_student: expected model {MODEL!r}, "
            f"got {TEACHER.model_id!r}"
        )
        assert modal.lookups == []


class TestConcurrentResolution:
    def test_concurrent_callers_share_one_resolution(self, modal_env) -> None:
        modal = _ModalServing()
        resolver = modal.resolver()
        callers = 16
        barrier = threading.Barrier(callers)

        def resolve(_):
            barrier.wait(timeout=10)
            return resolver.resolve(MODEL)

        with ThreadPoolExecutor(max_workers=callers) as pool:
            endpoints = list(pool.map(resolve, range(callers)))

        assert len({id(endpoint) for endpoint in endpoints}) == 1
        assert endpoints[0].base_url == MODAL_URL
        assert modal.lookups == ["cogniverse-vllm-llm-student"]
        assert len(modal.requests) == 1

    def test_concurrent_callers_all_get_the_one_typed_failure(self, modal_env) -> None:
        modal = _ModalServing(lookup_error=ModalLifecycleError("App not found"))
        resolver = modal.resolver()
        callers = 16
        barrier = threading.Barrier(callers)

        def resolve(_):
            barrier.wait(timeout=10)
            try:
                resolver.resolve(MODEL)
            except ModalLlmNotDeployedError as exc:
                return exc
            raise AssertionError("resolution must fail")

        with ThreadPoolExecutor(max_workers=callers) as pool:
            errors = list(pool.map(resolve, range(callers)))

        assert len({id(error) for error in errors}) == 1
        assert modal.lookups == ["cogniverse-vllm-llm-student"]
        assert modal.clients == []

    def test_each_role_resolves_independently(self, modal_env) -> None:
        student = _ModalServing()
        teacher = _ModalServing(TEACHER.model_id, TEACHER.model_revision)

        def web_url(spec, credentials):
            serving = student if spec.name == "vllm_llm_student" else teacher
            return serving.web_url(spec, credentials)

        def client(spec):
            serving = student if spec.name == "vllm_llm_student" else teacher
            return serving.client(spec)

        resolver = LlmResolver(modal_web_url=web_url, http_client=client)
        barrier = threading.Barrier(2)

        def resolve(model):
            barrier.wait(timeout=10)
            return resolver.resolve(model)

        with ThreadPoolExecutor(max_workers=2) as pool:
            primary, distilled = pool.map(resolve, (MODEL, TEACHER_MODEL))

        assert (primary.service, distilled.service) == (
            "vllm_llm_student",
            "vllm_llm_teacher",
        )
        assert student.lookups == ["cogniverse-vllm-llm-student"]
        assert teacher.lookups == ["cogniverse-vllm-llm-teacher"]


def test_ensure_llm_returns_the_openai_base_of_the_resolved_endpoint(
    monkeypatch, modal_env
) -> None:
    monkeypatch.setattr(hermetic_llm, "_RESOLVER", _ModalServing().resolver())

    assert hermetic_llm.ensure_llm() == f"{MODAL_URL}/v1"
    assert hermetic_llm.ensure_llm_endpoint(MODEL).base_url == MODAL_URL


class TestNothingIsStartedLocally:
    def test_the_resolver_has_no_container_or_process_path(self) -> None:
        source = hermetic_llm.__loader__.get_source(hermetic_llm.__name__)
        assert re.findall(r"\b(subprocess|docker|Popen|vllm serve)\b", source) == []


class TestDecisionsReachTheTerminalSummaryOfAPassingRun:
    """A real nested pytest session prints each decision, whatever the capture."""

    def test_resolved_and_refused_lines_are_printed(self, tmp_path) -> None:
        with _identity_server(STUDENT.model_id, STUDENT.model_revision) as (
            serving,
            _,
        ):
            session_dir = tmp_path / "session"
            session_dir.mkdir()
            (session_dir / "conftest.py").write_text(
                'pytest_plugins = ["tests.fixtures.sidecars"]\n'
            )
            (session_dir / "test_resolution.py").write_text(
                textwrap.dedent(
                    f"""
                    import json

                    import pytest

                    from tests.utils import hermetic_llm


                    def test_serving(monkeypatch):
                        monkeypatch.setenv(
                            "INFERENCE_SERVICE_URLS",
                            json.dumps({{"vllm_llm_student": "{serving}"}}),
                        )
                        assert hermetic_llm.ensure_llm() == "{serving}/v1"


                    def test_wrong_model(monkeypatch):
                        monkeypatch.setenv(
                            "INFERENCE_SERVICE_URLS",
                            json.dumps({{"vllm_llm_teacher": "{serving}"}}),
                        )
                        with pytest.raises(hermetic_llm.LlmEndpointMismatchError):
                            hermetic_llm.ensure_llm(hermetic_llm.TEACHER_MODEL)
                    """
                )
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if key != "INFERENCE_SERVICE_URLS"
            }
            env.update(
                PYTHONPATH=str(hermetic_llm.REPO_ROOT),
                COGNIVERSE_INFERENCE_API_KEY=API_KEY,
            )
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
                cwd=session_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        lines = output.splitlines()
        starts = [
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"=+ test sidecars =+", line)
        ]
        assert len(starts) == 1, output
        section = []
        for line in lines[starts[0] + 1 :]:
            if line.startswith("="):
                break
            section.append(line)
        assert [line for line in section if line] == [
            f"LM {MODEL}: resolved-remote {serving} "
            "[candidates: INFERENCE_SERVICE_URLS]",
            f"LM {TEACHER_MODEL}: refused (INFERENCE_SERVICE_URLS endpoint {serving} "
            "does not serve it exactly: ModelIdentityError: vllm_llm_teacher: "
            f"expected model {TEACHER_MODEL!r}, got {MODEL!r}) "
            "[candidates: INFERENCE_SERVICE_URLS]",
        ]
