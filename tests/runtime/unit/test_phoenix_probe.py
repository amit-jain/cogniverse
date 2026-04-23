"""Unit tests for the Phoenix reachability probe (audit fix #11).

Verifies that ``_probe_phoenix_reachability``:
1. Logs success when the global TelemetryManager can emit a span.
2. Logs a warning (but does not raise) when the probe fails and
   ``TELEMETRY_REQUIRED`` is unset.
3. Raises ``RuntimeError`` when the probe fails AND ``TELEMETRY_REQUIRED``
   is set — fail-fast for production deployments.
4. Skips the probe entirely when telemetry is disabled in config.

Before this fix the dispatcher silently fell back to NoOpSpan whenever
Phoenix was unreachable, leaving observability dashboards empty without
any startup-time hint at the cause.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_runtime.main import _probe_phoenix_reachability


@pytest.fixture
def disable_telemetry_required(monkeypatch):
    """Ensure TELEMETRY_REQUIRED is unset for the warning-path tests."""
    monkeypatch.delenv("TELEMETRY_REQUIRED", raising=False)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPhoenixProbe:
    def test_probe_succeeds_when_span_emits(self, disable_telemetry_required, caplog):
        mock_tm = MagicMock()
        mock_tm.config.enabled = True
        mock_tm.config.otlp_endpoint = "phoenix:4317"
        mock_span = MagicMock()
        mock_tm.span.return_value.__enter__.return_value = mock_span
        mock_tm.span.return_value.__exit__.return_value = None

        with patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=mock_tm,
        ):
            with caplog.at_level("INFO"):
                _probe_phoenix_reachability()

        # Span was attempted with the right name and the reserved system
        # identity (cluster-wide identity for runtime-internal telemetry).
        from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

        mock_tm.span.assert_called_once_with(
            "startup.probe", tenant_id=SYSTEM_TENANT_ID
        )
        # Success was logged.
        assert any("Phoenix reachability probe OK" in r.message for r in caplog.records)

    def test_probe_skipped_when_telemetry_disabled(
        self, disable_telemetry_required, caplog
    ):
        mock_tm = MagicMock()
        mock_tm.config.enabled = False

        with patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=mock_tm,
        ):
            with caplog.at_level("INFO"):
                _probe_phoenix_reachability()

        # span() should never be called when telemetry is disabled.
        mock_tm.span.assert_not_called()
        assert any(
            "skipping Phoenix reachability probe" in r.message for r in caplog.records
        )

    def test_probe_logs_warning_on_failure_when_optional(
        self, disable_telemetry_required, caplog
    ):
        """When TELEMETRY_REQUIRED is unset, a probe failure must log a
        warning but never raise — local dev should still start up."""
        mock_tm = MagicMock()
        mock_tm.config.enabled = True
        mock_tm.config.otlp_endpoint = "phoenix:4317"
        # Simulate the span context manager raising on entry.
        mock_tm.span.side_effect = ConnectionError("phoenix unreachable")

        with patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=mock_tm,
        ):
            with caplog.at_level("WARNING"):
                # Should NOT raise.
                _probe_phoenix_reachability()

        assert any(
            "Phoenix reachability probe FAILED" in r.message for r in caplog.records
        )

    def test_probe_raises_when_required_and_failing(self, monkeypatch, caplog):
        """When TELEMETRY_REQUIRED=true, a probe failure must raise
        RuntimeError so the runtime refuses to start with broken telemetry."""
        monkeypatch.setenv("TELEMETRY_REQUIRED", "true")

        mock_tm = MagicMock()
        mock_tm.config.enabled = True
        mock_tm.config.otlp_endpoint = "phoenix:4317"
        mock_tm.span.side_effect = ConnectionError("phoenix unreachable")

        with patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=mock_tm,
        ):
            with pytest.raises(RuntimeError, match="TELEMETRY_REQUIRED=true"):
                _probe_phoenix_reachability()

    def test_probe_treats_empty_required_as_false(self, monkeypatch, caplog):
        """An empty TELEMETRY_REQUIRED env var (set to '') must be treated
        as 'not required' — Helm sometimes injects empty defaults."""
        monkeypatch.setenv("TELEMETRY_REQUIRED", "")

        mock_tm = MagicMock()
        mock_tm.config.enabled = True
        mock_tm.config.otlp_endpoint = "phoenix:4317"
        mock_tm.span.side_effect = ConnectionError("phoenix unreachable")

        with patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=mock_tm,
        ):
            # Should NOT raise.
            _probe_phoenix_reachability()
