"""Unit tests for the Phoenix reachability probe.

The probe must ACTUALLY check the OTLP collector is reachable. Emitting a span
proved nothing — TelemetryManager.span() swallows tracer/export errors and
yields a NoOpSpan, so the old span-based probe never raised and always logged
"OK" even with Phoenix down, so TELEMETRY_REQUIRED=true never blocked startup
and dashboards silently went empty. These drive the probe against REAL sockets:
a live listener (reachable) and a closed port (unreachable).
"""

import contextlib
import socket

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_runtime.main import _probe_phoenix_reachability

_MGR = "cogniverse_foundation.telemetry.manager.get_telemetry_manager"


@pytest.fixture
def disable_telemetry_required(monkeypatch):
    monkeypatch.delenv("TELEMETRY_REQUIRED", raising=False)


@contextlib.contextmanager
def _listening():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        host, port = srv.getsockname()
        yield f"{host}:{port}"
    finally:
        srv.close()


def _closed_endpoint() -> str:
    """A 127.0.0.1 endpoint nothing is listening on (connect will be refused)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"127.0.0.1:{port}"


def _tm(endpoint: str, enabled: bool = True) -> MagicMock:
    tm = MagicMock()
    tm.config.enabled = enabled
    tm.config.otlp_endpoint = endpoint
    return tm


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPhoenixProbe:
    def test_probe_ok_when_collector_reachable(
        self, disable_telemetry_required, caplog
    ):
        with _listening() as endpoint:
            with patch(_MGR, return_value=_tm(endpoint)), caplog.at_level("INFO"):
                _probe_phoenix_reachability()
        assert any("Phoenix reachability probe OK" in r.message for r in caplog.records)

    def test_probe_skipped_when_telemetry_disabled(
        self, disable_telemetry_required, caplog
    ):
        tm = _tm("phoenix:4317", enabled=False)
        with patch(_MGR, return_value=tm), caplog.at_level("INFO"):
            _probe_phoenix_reachability()
        assert any(
            "skipping Phoenix reachability probe" in r.message for r in caplog.records
        )

    def test_probe_warns_on_unreachable_collector_when_optional(
        self, disable_telemetry_required, caplog
    ):
        """Collector down + TELEMETRY_REQUIRED unset: warn, do not raise."""
        with patch(_MGR, return_value=_tm(_closed_endpoint())):
            with caplog.at_level("WARNING"):
                _probe_phoenix_reachability()  # must not raise
        assert any(
            "Phoenix reachability probe FAILED" in r.message for r in caplog.records
        )

    def test_probe_raises_when_required_and_unreachable(self, monkeypatch):
        """Collector down + TELEMETRY_REQUIRED=true: refuse to start."""
        monkeypatch.setenv("TELEMETRY_REQUIRED", "true")
        with patch(_MGR, return_value=_tm(_closed_endpoint())):
            with pytest.raises(RuntimeError, match="TELEMETRY_REQUIRED=true"):
                _probe_phoenix_reachability()

    def test_probe_treats_empty_required_as_false(self, monkeypatch):
        monkeypatch.setenv("TELEMETRY_REQUIRED", "")
        with patch(_MGR, return_value=_tm(_closed_endpoint())):
            _probe_phoenix_reachability()  # must not raise
