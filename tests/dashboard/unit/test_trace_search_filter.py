class TestFetchTenantTracesSafely:
    """Caller-side wiring for PhoenixAnalytics.get_traces raising on outage:
    the Analytics tab shows the failure instead of the indistinguishable
    'No traces found' empty state."""

    def _args(self):
        from datetime import datetime, timezone

        return (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            None,
        )

    def test_outage_maps_to_error_message(self):
        from unittest.mock import MagicMock

        from cogniverse_dashboard.utils.traces import fetch_tenant_traces_safely

        analytics = MagicMock()
        analytics.get_traces.side_effect = ConnectionError("phoenix down")

        traces, error = fetch_tenant_traces_safely(
            analytics, "acme:acme", *self._args()
        )

        assert traces == []
        assert error == (
            "Failed to fetch traces from the telemetry backend: phoenix down"
        )

    def test_success_passes_traces_through_with_tenant_project(self):
        from unittest.mock import MagicMock

        from cogniverse_dashboard.utils.traces import fetch_tenant_traces_safely

        analytics = MagicMock()
        analytics.get_traces.return_value = ["t1", "t2"]

        traces, error = fetch_tenant_traces_safely(
            analytics, "acme:acme", *self._args()
        )

        assert traces == ["t1", "t2"]
        assert error is None
        assert (
            analytics.get_traces.call_args.kwargs["project_name"]
            == "cogniverse-acme:acme"
        )
