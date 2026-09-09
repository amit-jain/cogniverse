"""Tests for the runtime gate on LM-backed integration cases."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests import conftest as root_conftest
from tests.fixtures import llm as llm_fixtures
from tests.fixtures.markers import enforce_lm_gate
from tests.utils import hermetic_llm

pytest_plugins = ["pytester"]

_CHAIN_FLAGS = ("--tb=long", "-q", "-p", "no:cacheprovider")
DEAD_ROOT = "http://127.0.0.1:29071"

_GATE_CONFTEST = """
import pytest

from tests.fixtures.markers import enforce_lm_gate


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_lm: needs the configured test LM")


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item):
    enforce_lm_gate(item)
"""

_GATED_MODULE = """
import pytest

pytestmark = pytest.mark.requires_lm


def test_gated():
    assert 1 == 1
"""


def unreachable_message(root: str, source: str) -> str:
    """The exact gate failure line for a probe root and endpoint source."""
    return (
        "Exact configured LLM endpoint not reachable. Probed "
        f"{root}/api/tags, {root}/v1/models (endpoint from {source})"
    )


class _GateConfig:
    def __init__(self, setupplan: bool):
        self._setupplan = setupplan

    def getoption(self, name, default=None):
        return self._setupplan if name == "setupplan" else default


class _MarkedItem:
    def __init__(self, marked: bool = True, setupplan: bool = False):
        self._marked = marked
        self.config = _GateConfig(setupplan)

    def get_closest_marker(self, name):
        return object() if self._marked and name == "requires_lm" else None


def test_lm_fixture_rejects_missing_config_without_legacy_default(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(missing))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_base_url()

    assert str(error.value) == f"Test LM config file does not exist: {missing}"


def test_lm_fixture_rejects_malformed_config(monkeypatch, tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json")
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(malformed))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_bare_model()

    assert str(error.value) == f"Test LM config file is not valid JSON: {malformed}"


def test_lm_fixture_rejects_incomplete_primary_config(monkeypatch, tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps({"llm_config": {"primary": {"model": hermetic_llm.MODEL}}})
    )
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(incomplete))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_base_url()

    assert str(error.value) == (
        f"Test LM config requires non-empty llm_config.primary.api_base and model: "
        f"{incomplete}"
    )


def test_lm_fixture_requires_complete_explicit_environment(monkeypatch):
    monkeypatch.setenv("TEST_LLM_API_BASE", "http://127.0.0.1:29110/v1")
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_base_url()

    assert str(error.value) == (
        "Test LM environment requires both TEST_LLM_API_BASE and TEST_LLM_MODEL"
    )


def test_lm_fixture_explicit_environment_overrides_missing_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("TEST_LLM_API_BASE", "http://127.0.0.1:29110/v1")
    monkeypatch.setenv("TEST_LLM_MODEL", hermetic_llm.MODEL)

    assert llm_fixtures.resolve_base_url() == "http://127.0.0.1:29110/v1"
    assert llm_fixtures.resolve_bare_model() == hermetic_llm.MODEL


def test_gate_runs_after_session_fixture_setup(monkeypatch):
    monkeypatch.setattr(llm_fixtures, "is_test_lm_available", lambda: True)

    root_conftest.pytest_runtest_setup(_MarkedItem())

    assert root_conftest.pytest_runtest_setup.pytest_impl["trylast"] is True


def test_gate_fails_with_exact_endpoint_after_unsuccessful_provision(monkeypatch):
    """Unreachable endpoint on a ``requires_lm`` test FAILS — never skips."""
    monkeypatch.setenv("TEST_LLM_API_BASE", "http://127.0.0.1:29999/v1")
    monkeypatch.setenv("TEST_LLM_MODEL", "m")
    monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(pytest.fail.Exception) as error:
        root_conftest.pytest_runtest_setup(_MarkedItem())

    assert str(error.value) == unreachable_message(
        "http://127.0.0.1:29999", "TEST_LLM_API_BASE"
    )


def test_unmarked_test_is_never_gated(monkeypatch):
    monkeypatch.setenv("TEST_LLM_API_BASE", f"{DEAD_ROOT}/v1")
    monkeypatch.setenv("TEST_LLM_MODEL", "m")

    assert root_conftest.pytest_runtest_setup(_MarkedItem(marked=False)) is None


class TestSetupPlanIsNotGated:
    """``--setup-plan`` executes no fixture, so there is no endpoint to gate."""

    def test_plan_only_run_reports_no_gate_error(self, pytester, monkeypatch):
        monkeypatch.setenv("TEST_LLM_API_BASE", f"{DEAD_ROOT}/v1")
        monkeypatch.setenv("TEST_LLM_MODEL", "m")
        pytester.makeconftest(_GATE_CONFTEST)
        pytester.makepyfile(test_gated=_GATED_MODULE)

        result = pytester.runpytest("--setup-plan", *_CHAIN_FLAGS)

        assert result.ret == 0
        assert [line for line in result.outlines if "not reachable" in line] == []

    def test_executing_run_still_fails_naming_both_probe_urls(
        self, pytester, monkeypatch
    ):
        monkeypatch.setenv("TEST_LLM_API_BASE", f"{DEAD_ROOT}/v1")
        monkeypatch.setenv("TEST_LLM_MODEL", "m")
        monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        pytester.makeconftest(_GATE_CONFTEST)
        pytester.makepyfile(test_gated=_GATED_MODULE)

        result = pytester.runpytest(*_CHAIN_FLAGS)

        result.assert_outcomes(errors=1, passed=0, failed=0, skipped=0)
        assert unreachable_message(DEAD_ROOT, "TEST_LLM_API_BASE") in result.outlines

    def test_the_direct_gate_call_short_circuits_under_setup_plan(self, monkeypatch):
        monkeypatch.setenv("TEST_LLM_API_BASE", f"{DEAD_ROOT}/v1")
        monkeypatch.setenv("TEST_LLM_MODEL", "m")

        assert enforce_lm_gate(_MarkedItem(setupplan=True)) is None


@contextmanager
def _authenticated_lm_server(*model_ids: str, require_bearer: str | None = None):
    """A real OpenAI-compatible endpoint that may demand a bearer token."""
    payload = {
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in model_ids],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if (
                require_bearer is not None
                and self.headers.get("Authorization") != f"Bearer {require_bearer}"
            ):
                self.send_response(401)
                self.end_headers()
                return
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestApiKeyFollowsTheEndpoint:
    """An authenticated endpoint must be given the inference API key."""

    def test_explicit_test_key_wins(self, monkeypatch):
        monkeypatch.setenv("TEST_LLM_API_KEY", "explicit")
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "inference")
        assert llm_fixtures.resolve_api_key() == "explicit"

    def test_inference_key_is_used_when_no_test_key_is_set(self, monkeypatch):
        monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "inference")
        assert llm_fixtures.resolve_api_key() == "inference"

    def test_unauthenticated_endpoints_still_get_the_sentinel(self, monkeypatch):
        monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        assert llm_fixtures.resolve_api_key() == "not-required"


class TestReachabilityProbeAuthenticates:
    """The requires_lm gate must not read 401 as 'endpoint unreachable'."""

    def test_authenticated_endpoint_is_reachable_with_the_key(self, monkeypatch):
        monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "s3cr3t")
        with _authenticated_lm_server("m", require_bearer="s3cr3t") as base_url:
            monkeypatch.setenv("TEST_LLM_API_BASE", f"{base_url}/v1")
            monkeypatch.setenv("TEST_LLM_MODEL", "m")
            assert llm_fixtures.is_test_lm_available() is True

    def test_wrong_key_reports_unreachable(self, monkeypatch):
        monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "wrong")
        with _authenticated_lm_server("m", require_bearer="s3cr3t") as base_url:
            monkeypatch.setenv("TEST_LLM_API_BASE", f"{base_url}/v1")
            monkeypatch.setenv("TEST_LLM_MODEL", "m")
            assert llm_fixtures.is_test_lm_available() is False

    def test_unauthenticated_endpoint_still_works(self, monkeypatch):
        monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        with _authenticated_lm_server("m") as base_url:
            monkeypatch.setenv("TEST_LLM_API_BASE", f"{base_url}/v1")
            monkeypatch.setenv("TEST_LLM_MODEL", "m")
            assert llm_fixtures.is_test_lm_available() is True


@contextmanager
def _failing_lm_server(status: int):
    """A real endpoint that answers every probe with ``status``."""
    requested: list[str] = []
    record_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with record_lock:
                requested.append(self.path)
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requested
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _point_at(monkeypatch, root: str) -> None:
    monkeypatch.setenv("TEST_LLM_API_BASE", f"{root}/v1")
    monkeypatch.setenv("TEST_LLM_MODEL", "m")
    monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)


class TestDegradedEndpointFaultContract:
    """A discovered-but-degraded endpoint fails naming what was probed."""

    def test_503_endpoint_fails_with_both_probed_urls(self, monkeypatch):
        with _failing_lm_server(503) as (root, requested):
            _point_at(monkeypatch, root)

            with pytest.raises(pytest.fail.Exception) as error:
                enforce_lm_gate(_MarkedItem())

        assert str(error.value) == unreachable_message(root, "TEST_LLM_API_BASE")
        assert requested == ["/api/tags", "/v1/models"]

    def test_probe_targets_match_the_urls_the_endpoint_receives(self, monkeypatch):
        with _failing_lm_server(503) as (root, requested):
            _point_at(monkeypatch, root)
            targets = llm_fixtures.lm_probe_targets()

            assert llm_fixtures.is_test_lm_available() is False

        assert targets == (f"{root}/api/tags", f"{root}/v1/models")
        assert [f"{root}{path}" for path in requested] == list(targets)


class TestConcurrentGateEvaluations:
    """Concurrent gate evaluations reach one verdict and share no state."""

    def test_eight_threads_agree_and_each_probes_the_endpoint(self, monkeypatch):
        thread_count = 8
        messages: list[str] = []
        collect_lock = threading.Lock()
        barrier = threading.Barrier(thread_count)

        with _failing_lm_server(503) as (root, requested):
            _point_at(monkeypatch, root)

            def evaluate():
                barrier.wait(timeout=30)
                try:
                    enforce_lm_gate(_MarkedItem())
                except pytest.fail.Exception as exc:
                    with collect_lock:
                        messages.append(str(exc))

            threads = [threading.Thread(target=evaluate) for _ in range(thread_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)

        assert [thread.is_alive() for thread in threads] == [False] * thread_count
        assert (
            messages == [unreachable_message(root, "TEST_LLM_API_BASE")] * thread_count
        )
        assert sorted(requested) == sorted(["/api/tags", "/v1/models"] * thread_count)
