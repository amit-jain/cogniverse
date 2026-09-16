"""Unit tests for OpenShell mTLS cert rotation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_runtime.openshell_cert_rotator import (
    CertRotator,
    _gateway_dir,
    diff_mtimes,
    snapshot_mtimes,
)


@pytest.fixture
def gateway_dir(tmp_path: Path, monkeypatch) -> Path:
    """Build a fake OpenShell config dir layout under tmp."""
    base = tmp_path / "openshell"
    gw_name = "test-gateway"
    gw_dir = base / "gateways" / gw_name
    (gw_dir / "mtls").mkdir(parents=True)
    (gw_dir / "metadata.json").write_text('{"name":"test-gateway"}', encoding="utf-8")
    (gw_dir / "mtls" / "ca.crt").write_text("ORIGINAL_CA", encoding="utf-8")
    (gw_dir / "mtls" / "tls.crt").write_text("ORIGINAL_TLS_CRT", encoding="utf-8")
    (gw_dir / "mtls" / "tls.key").write_text("ORIGINAL_TLS_KEY", encoding="utf-8")
    (base / "active_gateway").write_text(gw_name, encoding="utf-8")
    monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(base))
    return gw_dir


@pytest.fixture
def recording_tracer() -> trace.Tracer:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    tracer._exporter = exporter  # type: ignore[attr-defined]
    return tracer


def _spans_named(tracer: trace.Tracer, name: str):
    """Return all recorded spans matching ``name`` (in order)."""
    exporter: InMemorySpanExporter = tracer._exporter  # type: ignore[attr-defined]
    return [s for s in exporter.get_finished_spans() if s.name == name]


class TestGatewayDir:
    def test_resolves_active_gateway(self, gateway_dir: Path):
        gw = _gateway_dir(None)
        assert gw == gateway_dir

    def test_explicit_cluster_overrides(self, gateway_dir: Path):
        gw = _gateway_dir(cluster="test-gateway")
        assert gw == gateway_dir

    def test_unknown_cluster_returns_none(self, gateway_dir: Path):
        gw = _gateway_dir(cluster="does-not-exist")
        assert gw is None

    def test_no_config_dir_returns_none(self, monkeypatch, tmp_path: Path):
        # Override env to a directory with no active_gateway file.
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path / "empty"))
        gw = _gateway_dir(None)
        assert gw is None


class TestSnapshotAndDiff:
    def test_snapshot_includes_all_four_files(self, gateway_dir: Path):
        snap = snapshot_mtimes(gateway_dir)
        # 4 watched files: metadata, ca.crt, tls.crt, tls.key.
        assert len(snap) == 4
        for path in snap:
            assert snap[path] > 0  # all exist on disk

    def test_diff_when_unchanged_is_empty(self, gateway_dir: Path):
        a = snapshot_mtimes(gateway_dir)
        b = snapshot_mtimes(gateway_dir)
        assert diff_mtimes(a, b) == []

    def test_diff_detects_modified_file(self, gateway_dir: Path):
        before = snapshot_mtimes(gateway_dir)
        # Wait long enough that mtime resolution catches the change.
        os.utime(gateway_dir / "mtls" / "tls.crt", (1_000_000_000, 1_000_000_000))
        after = snapshot_mtimes(gateway_dir)
        changed = diff_mtimes(before, after)
        assert any(p.endswith("tls.crt") for p in changed)

    def test_diff_detects_disappeared_file(self, gateway_dir: Path):
        before = snapshot_mtimes(gateway_dir)
        (gateway_dir / "mtls" / "tls.key").unlink()
        after = snapshot_mtimes(gateway_dir)
        changed = diff_mtimes(before, after)
        assert any(p.endswith("tls.key") for p in changed)


class TestCertRotatorProbe:
    def test_first_probe_captures_baseline_no_reconnect(
        self, gateway_dir: Path, recording_tracer
    ):
        mgr = MagicMock()
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        changed, paths = rotator.probe_once()
        assert changed is False
        assert paths == []
        mgr.reconnect.assert_not_called()
        spans = _spans_named(recording_tracer, "openshell.cert_rotation")
        assert len(spans) == 1
        assert (
            spans[0].attributes["openshell.cert_rotation_reason"] == "baseline_capture"
        )

    def test_unchanged_probe_is_quiet(self, gateway_dir: Path, recording_tracer):
        mgr = MagicMock()
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        rotator.probe_once()  # baseline
        changed, _ = rotator.probe_once()  # second tick, nothing changed
        assert changed is False
        mgr.reconnect.assert_not_called()
        spans = _spans_named(recording_tracer, "openshell.cert_rotation")
        # baseline + unchanged = 2 spans
        assert len(spans) == 2
        assert spans[1].attributes["openshell.cert_rotation_reason"] == "unchanged"

    def test_rotation_triggers_reconnect_and_records_paths(
        self, gateway_dir: Path, recording_tracer
    ):
        mgr = MagicMock()
        mgr.reconnect.return_value = True
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        rotator.probe_once()  # baseline

        # Rotate the certs.
        os.utime(gateway_dir / "mtls" / "ca.crt", (2_000_000_000, 2_000_000_000))
        os.utime(gateway_dir / "mtls" / "tls.crt", (2_000_000_000, 2_000_000_000))
        os.utime(gateway_dir / "mtls" / "tls.key", (2_000_000_000, 2_000_000_000))

        changed, paths = rotator.probe_once()
        assert changed is True
        assert len(paths) == 3
        assert all(p.endswith((".crt", ".key")) for p in paths)
        mgr.reconnect.assert_called_once()
        # last_rotation_at recorded.
        assert rotator.last_rotation_at is not None
        # Span emitted with rotation_detected reason and path list.
        rotation_spans = [
            s
            for s in _spans_named(recording_tracer, "openshell.cert_rotation")
            if s.attributes["openshell.cert_rotation_detected"] == 1
        ]
        assert len(rotation_spans) == 1
        attrs = rotation_spans[0].attributes
        assert attrs["openshell.cert_rotation_reason"] == "rotation_detected"
        assert "tls.crt" in attrs["openshell.cert_rotation_changed_paths"]

    def test_rotation_only_fires_once_per_change(
        self, gateway_dir: Path, recording_tracer
    ):
        mgr = MagicMock()
        mgr.reconnect.return_value = True
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        rotator.probe_once()  # baseline
        os.utime(gateway_dir / "mtls" / "tls.crt", (2_000_000_000, 2_000_000_000))
        rotator.probe_once()  # detects rotation, reconnects
        rotator.probe_once()  # nothing new — must not reconnect again
        assert mgr.reconnect.call_count == 1

    def test_reconnect_failure_is_logged_not_raised(
        self, gateway_dir: Path, recording_tracer, caplog
    ):
        mgr = MagicMock()
        mgr.reconnect.side_effect = RuntimeError("bad cluster")
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        rotator.probe_once()
        os.utime(gateway_dir / "mtls" / "tls.crt", (2_000_000_000, 2_000_000_000))
        # Must not raise even when reconnect blows up.
        rotator.probe_once()
        # Warning logged.
        assert any("reconnect raised" in rec.message for rec in caplog.records)

    def test_no_gateway_dir_records_span_and_does_nothing(
        self, monkeypatch, tmp_path: Path, recording_tracer
    ):
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path / "empty"))
        mgr = MagicMock()
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        changed, _ = rotator.probe_once()
        assert changed is False
        mgr.reconnect.assert_not_called()
        spans = _spans_named(recording_tracer, "openshell.cert_rotation")
        assert spans[0].attributes["openshell.cert_rotation_reason"] == "no_gateway_dir"


class TestAuthFailureTrigger:
    def test_auth_failure_triggers_reconnect(self, gateway_dir: Path, recording_tracer):
        mgr = MagicMock()
        mgr.reconnect.return_value = True
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        attempted = rotator.trigger_on_auth_failure("AuthError: x509: bad cert")
        assert attempted is True
        mgr.reconnect.assert_called_once()

    def test_burst_is_rate_limited(self, gateway_dir: Path, recording_tracer):
        mgr = MagicMock()
        mgr.reconnect.return_value = True
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        first = rotator.trigger_on_auth_failure("auth_error")
        second = rotator.trigger_on_auth_failure("auth_error")
        third = rotator.trigger_on_auth_failure("auth_error")
        assert first is True
        assert second is False
        assert third is False
        # Only the first reconnect actually fired.
        assert mgr.reconnect.call_count == 1

    def test_auth_trigger_refreshes_snapshot(self, gateway_dir: Path, recording_tracer):
        mgr = MagicMock()
        mgr.reconnect.return_value = True
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        # First baseline.
        rotator.probe_once()
        # Modify a cert AND trigger an auth-failure reconnect.
        os.utime(gateway_dir / "mtls" / "tls.crt", (2_000_000_000, 2_000_000_000))
        rotator.trigger_on_auth_failure("auth_error")
        # The next polling tick must NOT double-fire even though the file
        # mtime is now different from the original baseline — the auth
        # trigger should have refreshed the snapshot.
        before_count = mgr.reconnect.call_count
        rotator.probe_once()
        assert mgr.reconnect.call_count == before_count


@pytest.mark.asyncio
class TestLifecycle:
    async def test_start_runs_probe_then_stop_cleans_up(
        self, gateway_dir: Path, recording_tracer
    ):
        mgr = MagicMock()
        mgr.reconnect.return_value = True
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=1, tracer=recording_tracer
        )
        rotator.start()
        # First probe runs immediately after start; give it a beat.
        await asyncio.sleep(0.1)
        assert rotator.last_snapshot is not None  # baseline captured
        await rotator.stop()

    async def test_double_start_is_idempotent(
        self, gateway_dir: Path, recording_tracer
    ):
        mgr = MagicMock()
        rotator = CertRotator(
            sandbox_manager=mgr, interval_seconds=60, tracer=recording_tracer
        )
        rotator.start()
        rotator.start()  # no-op
        await rotator.stop()


def test_invalid_interval_rejected():
    mgr = MagicMock()
    with pytest.raises(ValueError, match=">= 1"):
        CertRotator(sandbox_manager=mgr, interval_seconds=0.5)


class _FakeSession:
    def __init__(self, name: str = "sandbox-1"):
        self.id = name
        self.sandbox = MagicMock()
        self.sandbox.name = name
        self.delete_count = 0

    def exec(self, command, timeout_seconds):
        raise PermissionError("x509: certificate has expired")

    def delete(self):
        self.delete_count += 1


class _ReadyClient:
    def __init__(self, session):
        self._session = session

    def create_session(self):
        return self._session

    def wait_ready(self, name, timeout_seconds):
        return None


def _manager_with_client(client):
    from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

    mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
    mgr._client = client
    mgr._available = True
    return mgr


class TestTaskSessionWiring:
    """Round-trip: a task sandbox's auth failure → rotator trigger fires."""

    @pytest.fixture(autouse=True)
    def _reset_breakers(self):
        from cogniverse_core.common.utils.circuit_breaker import CircuitBreaker

        CircuitBreaker.reset_registry()
        yield
        CircuitBreaker.reset_registry()

    @pytest.mark.asyncio
    async def test_auth_error_creating_the_sandbox_triggers_rotator(self):
        class _BoomClient:
            def create_session(self):
                raise PermissionError("x509: certificate has expired")

        mgr = _manager_with_client(_BoomClient())
        rotator = MagicMock()
        rotator.trigger_on_auth_failure.return_value = True
        mgr.attach_cert_rotator(rotator)

        with pytest.raises(PermissionError, match="x509: certificate has expired"):
            async with mgr.task_session("test_agent", "prodfixagents:certs"):
                pass

        assert rotator.trigger_on_auth_failure.call_count == 1
        assert "x509" in rotator.trigger_on_auth_failure.call_args.args[0]

    @pytest.mark.asyncio
    async def test_auth_error_during_exec_triggers_rotator(self):
        session = _FakeSession()
        mgr = _manager_with_client(_ReadyClient(session))
        rotator = MagicMock()
        rotator.trigger_on_auth_failure.return_value = True
        mgr.attach_cert_rotator(rotator)

        with pytest.raises(PermissionError, match="x509: certificate has expired"):
            async with mgr.task_session("test_agent", "prodfixagents:certs") as owned:
                await owned.exec(["echo", "hi"], timeout_seconds=5)

        assert rotator.trigger_on_auth_failure.call_count == 1
        assert "x509" in rotator.trigger_on_auth_failure.call_args.args[0]
        assert session.delete_count == 1

    @pytest.mark.asyncio
    async def test_non_auth_error_does_not_trigger_rotator(self):
        class _BoomClient:
            def create_session(self):
                raise RuntimeError("OOM: container killed")

        mgr = _manager_with_client(_BoomClient())
        rotator = MagicMock()
        mgr.attach_cert_rotator(rotator)

        with pytest.raises(RuntimeError, match="OOM: container killed"):
            async with mgr.task_session("test_agent", "prodfixagents:certs"):
                pass

        assert rotator.trigger_on_auth_failure.call_count == 0

    @pytest.mark.asyncio
    async def test_no_rotator_attached_surfaces_the_original_error(self):
        class _BoomClient:
            def create_session(self):
                raise PermissionError("x509: bad cert")

        mgr = _manager_with_client(_BoomClient())

        with pytest.raises(PermissionError, match="x509: bad cert"):
            async with mgr.task_session("test_agent", "prodfixagents:certs"):
                pass
