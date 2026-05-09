"""Unit tests for D.4 — sandbox lifecycle event telemetry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_runtime.sandbox_manager import (
    SandboxManager,
    SandboxPolicy,
    _classify_exec_failure,
)


@pytest.fixture
def captured_spans(monkeypatch):
    """Install a TracerProvider that captures spans for assertion.

    The SandboxManager calls ``trace.get_tracer(__name__)`` which reads
    the global TracerProvider. Earlier versions of this fixture mutated
    ``trace._TRACER_PROVIDER`` directly, but that left the global in a
    state that could confuse other test files importing the OTel
    tracer (causing a ``RecursionError`` in ``ProxyTracerProvider``
    when a subsequent test re-entered ``get_tracer``). Instead, patch
    ``trace.get_tracer`` for the duration of the test — monkeypatch
    cleans it up automatically and never touches the underlying
    proxy chain.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        trace, "get_tracer", lambda *_a, **_kw: provider.get_tracer("test")
    )
    yield exporter


def _build_manager_with_fake_client(client) -> SandboxManager:
    """Construct a SandboxManager whose _client is the supplied stub.

    Uses policy=DISABLED to short-circuit the boot connect, then sets the
    client + _available manually so exec_in_sandbox proceeds.
    """
    mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
    mgr._client = client
    mgr._available = True
    return mgr


@pytest.fixture(autouse=True)
def _disable_pool_for_d4_telemetry_tests(monkeypatch):
    """D.4 tests assert per-call create/wait/delete spans — that's the
    un-pooled lifecycle. The session pool (D.5) reuses sessions across
    calls so create/wait fire only on first checkout and delete fires on
    eviction. Disable the pool here so each test sees the original
    per-call shape; pool-specific tests live in test_sandbox_pool.py.
    """
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POOL_ENABLED", "false")
    yield


class _FakeSession:
    def __init__(self, exit_code=0, stderr="", stdout="ok", name="sandbox-1"):
        self.id = name
        self.sandbox = MagicMock()
        self.sandbox.name = name
        self._result = MagicMock(exit_code=exit_code, stderr=stderr, stdout=stdout)
        self.exec_calls = []
        self.delete_calls = 0

    def exec(self, command, timeout_seconds=60):
        self.exec_calls.append((tuple(command), timeout_seconds))
        return self._result

    def delete(self):
        self.delete_calls += 1


class _FakeClient:
    def __init__(self, session):
        self._session = session
        self.create_count = 0
        self.wait_count = 0

    def create_session(self):
        self.create_count += 1
        return self._session

    def wait_ready(self, name, timeout_seconds):
        self.wait_count += 1


class TestClassifyExecFailure:
    @pytest.mark.parametrize(
        "exit_code,stderr,expected_oom,expected_denied",
        [
            (0, "", False, False),
            (137, "", True, False),  # SIGKILL exit
            (139, "", True, False),  # SIGSEGV
            (1, "Killed", True, False),
            (1, "Process OOMKilled by cgroup", True, False),
            (
                1,
                "open('/etc/shadow'): permission denied",
                False,
                True,
            ),
            (1, "syscall denied for openat", False, True),
            (1, "blocked by policy: egress to example.com", False, True),
        ],
    )
    def test_classification(self, exit_code, stderr, expected_oom, expected_denied):
        out = _classify_exec_failure(exit_code, stderr)
        assert out["openshell.oom"] is expected_oom
        assert out["openshell.policy_denied"] is expected_denied


class TestSpanEmission:
    def test_successful_exec_emits_full_lifecycle_spans(self, captured_spans):
        session = _FakeSession(exit_code=0, stderr="", stdout="hello")
        client = _FakeClient(session)
        mgr = _build_manager_with_fake_client(client)

        result = mgr.exec_in_sandbox(
            "search_agent", ["echo", "hello"], timeout_seconds=10
        )
        assert result == {"stdout": "hello", "stderr": "", "exit_code": 0}

        spans = captured_spans.get_finished_spans()
        names = [s.name for s in spans]
        # Lifecycle phases all present.
        assert "sandbox.create_session" in names
        assert "sandbox.wait_ready" in names
        assert "sandbox.exec" in names
        assert "sandbox.delete" in names
        assert "sandbox.exec_in_sandbox" in names

        # exec span carries exit_code + non-positive wall_ms (skips on
        # CI where monotonic granularity is coarse, so use >= 0).
        exec_span = next(s for s in spans if s.name == "sandbox.exec")
        attrs = dict(exec_span.attributes)
        assert attrs["openshell.exit_code"] == 0
        assert attrs["openshell.oom"] is False
        assert attrs["openshell.policy_denied"] is False
        assert attrs["openshell.command_first"] == "echo"
        assert attrs["openshell.wall_ms"] >= 0

        # Parent span has the same exit_code + classification mirrored.
        parent_span = next(s for s in spans if s.name == "sandbox.exec_in_sandbox")
        parent_attrs = dict(parent_span.attributes)
        assert parent_attrs["openshell.agent_type"] == "search_agent"
        assert parent_attrs["openshell.exit_code"] == 0

    def test_oom_exec_marks_oom_attribute(self, captured_spans):
        session = _FakeSession(exit_code=137, stderr="Killed", stdout="")
        client = _FakeClient(session)
        mgr = _build_manager_with_fake_client(client)

        mgr.exec_in_sandbox("coding_agent", ["python", "memhog.py"])

        spans = captured_spans.get_finished_spans()
        exec_span = next(s for s in spans if s.name == "sandbox.exec")
        attrs = dict(exec_span.attributes)
        assert attrs["openshell.oom"] is True
        assert attrs["openshell.policy_denied"] is False

    def test_policy_denied_marks_policy_denied_attribute(self, captured_spans):
        session = _FakeSession(
            exit_code=1, stderr="open('/etc/shadow'): permission denied", stdout=""
        )
        client = _FakeClient(session)
        mgr = _build_manager_with_fake_client(client)

        mgr.exec_in_sandbox("coding_agent", ["cat", "/etc/shadow"])

        spans = captured_spans.get_finished_spans()
        exec_span = next(s for s in spans if s.name == "sandbox.exec")
        attrs = dict(exec_span.attributes)
        assert attrs["openshell.policy_denied"] is True

    def test_session_deleted_even_on_exec_exception(self, captured_spans):
        session = _FakeSession()

        def raising_exec(*a, **kw):
            raise RuntimeError("simulated exec failure")

        session.exec = raising_exec
        client = _FakeClient(session)
        mgr = _build_manager_with_fake_client(client)

        result = mgr.exec_in_sandbox("search_agent", ["x"])
        # exec_in_sandbox swallows the exception → returns dict with exit -1.
        assert result["exit_code"] == -1
        # Delete still called.
        assert session.delete_calls == 1
        # Span recorded the exception type.
        spans = captured_spans.get_finished_spans()
        parent = next(s for s in spans if s.name == "sandbox.exec_in_sandbox")
        assert dict(parent.attributes).get("openshell.error") == "RuntimeError"

    def test_no_spans_when_sandbox_unavailable(self, captured_spans):
        mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
        assert mgr._available is False
        assert mgr.exec_in_sandbox("x", ["echo"]) is None
        assert len(captured_spans.get_finished_spans()) == 0
