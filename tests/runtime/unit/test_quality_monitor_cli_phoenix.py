"""quality_monitor_cli builds its PhoenixProvider on the deployment's endpoints.

``_build_phoenix_provider`` takes both endpoints from the caller, which reads
them from TELEMETRY_OTLP_ENDPOINT and TELEMETRY_HTTP_ENDPOINT (or
``--phoenix-url``); a missing endpoint stops the process instead of reaching
for a Phoenix on localhost.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_runtime.quality_monitor_cli import _build_phoenix_provider


@pytest.mark.unit
@pytest.mark.ci_fast
class TestBuildPhoenixProvider:
    def test_builds_the_provider_on_the_given_endpoints(self, caplog):
        with caplog.at_level(logging.INFO, "cogniverse_runtime.quality_monitor_cli"):
            provider = _build_phoenix_provider(
                tenant_id="acme:acme",
                http_endpoint="http://cogniverse-phoenix:6006",
                grpc_endpoint="cogniverse-phoenix:4317",
            )

        assert type(provider).__name__ == "PhoenixProvider"
        assert provider._http_endpoint == "http://cogniverse-phoenix:6006"
        assert provider.traces.http_endpoint == "http://cogniverse-phoenix:6006"
        assert provider.datasets.http_endpoint == "http://cogniverse-phoenix:6006"
        assert [
            r.getMessage()
            for r in caplog.records
            if r.name == "cogniverse_runtime.quality_monitor_cli"
        ] == [
            "PhoenixProvider initialized for QualityMonitor (tenant=acme:acme, "
            "http=http://cogniverse-phoenix:6006, grpc=cogniverse-phoenix:4317)"
        ]

    def test_ignores_phoenix_env_overrides(self, monkeypatch):
        monkeypatch.setenv("PHOENIX_GRPC_ENDPOINT", "phoenix-otlp:14317")
        monkeypatch.setenv("PHOENIX_HTTP_ENDPOINT", "http://phoenix-env:16006")

        with patch(
            "cogniverse_telemetry_phoenix.provider.PhoenixProvider.initialize"
        ) as initialize:
            _build_phoenix_provider(
                tenant_id="acme:acme",
                http_endpoint="http://cogniverse-phoenix:6006",
                grpc_endpoint="cogniverse-phoenix:4317",
            )

        assert initialize.call_args.args == (
            {
                "tenant_id": "acme:acme",
                "http_endpoint": "http://cogniverse-phoenix:6006",
                "grpc_endpoint": "cogniverse-phoenix:4317",
            },
        )

    @pytest.mark.parametrize(
        ("http_endpoint", "grpc_endpoint", "missing"),
        [
            ("", "cogniverse-phoenix:4317", "http_endpoint"),
            ("http://cogniverse-phoenix:6006", "", "grpc_endpoint"),
        ],
    )
    def test_an_empty_endpoint_raises_instead_of_degrading(
        self, http_endpoint, grpc_endpoint, missing
    ):
        with pytest.raises(ValueError) as excinfo:
            _build_phoenix_provider(
                tenant_id="acme:acme",
                http_endpoint=http_endpoint,
                grpc_endpoint=grpc_endpoint,
            )

        assert str(excinfo.value) == (
            f"{missing} required in Phoenix provider config. Got config: "
            f"{{'tenant_id': 'acme:acme', 'http_endpoint': {http_endpoint!r}, "
            f"'grpc_endpoint': {grpc_endpoint!r}}}"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMissingEndpointsStopTheMonitor:
    @pytest.mark.parametrize(
        ("env", "argv_extra", "message"),
        [
            (
                {"TELEMETRY_HTTP_ENDPOINT": "http://cogniverse-phoenix:6006"},
                [],
                "TELEMETRY_OTLP_ENDPOINT must name the deployment's Phoenix for "
                "the quality monitor",
            ),
            (
                {"TELEMETRY_OTLP_ENDPOINT": "cogniverse-phoenix:4317"},
                [],
                "TELEMETRY_HTTP_ENDPOINT or --phoenix-url must name the "
                "deployment's Phoenix for the quality monitor",
            ),
            (
                {},
                ["--phoenix-url", "http://cogniverse-phoenix:6006"],
                "TELEMETRY_OTLP_ENDPOINT must name the deployment's Phoenix for "
                "the quality monitor",
            ),
        ],
    )
    def test_exits_2_naming_the_missing_endpoint(
        self, monkeypatch, caplog, env, argv_extra, message
    ):
        from cogniverse_runtime import quality_monitor_cli

        for name in ("TELEMETRY_OTLP_ENDPOINT", "TELEMETRY_HTTP_ENDPOINT"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        fake_manager = MagicMock()
        fake_manager.config.provider_config = {}
        built = []
        argv = [
            "quality_monitor_cli",
            "--tenant-id",
            "acme",
            "--llm-model",
            "test-model",
            "--annotation-cycle",
            *argv_extra,
        ]
        with (
            patch(
                "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
                return_value=fake_manager,
            ),
            patch.object(
                quality_monitor_cli,
                "_build_phoenix_provider",
                side_effect=lambda **kwargs: built.append(kwargs),
            ),
            patch("sys.argv", argv),
            caplog.at_level(logging.ERROR, "cogniverse_runtime.quality_monitor_cli"),
        ):
            with pytest.raises(SystemExit) as excinfo:
                quality_monitor_cli.main()

        assert excinfo.value.code == 2
        assert [
            r.getMessage()
            for r in caplog.records
            if r.name == "cogniverse_runtime.quality_monitor_cli"
        ] == [message]
        assert built == []
        assert fake_manager.config.provider_config == {}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPhoenixUrlReachesTelemetryManager:
    def test_annotation_cycle_gets_phoenix_url_override(self, monkeypatch):
        """--phoenix-url must reach the readers built via get_telemetry_manager
        (AnnotationStorage/AnnotationAgent) — they otherwise derive the HTTP
        endpoint from TELEMETRY_OTLP_ENDPOINT's host on the fixed :6006 port.
        The override must land before the cycle constructs them."""
        from cogniverse_runtime import quality_monitor_cli

        monkeypatch.delenv("TELEMETRY_HTTP_ENDPOINT", raising=False)
        monkeypatch.setenv("TELEMETRY_OTLP_ENDPOINT", "cogniverse-phoenix:4317")
        fake_manager = MagicMock()
        fake_manager.config.provider_config = {}

        seen = {}

        async def _fake_cycle(**kwargs):
            seen["http_endpoint"] = fake_manager.config.provider_config.get(
                "http_endpoint"
            )
            return {"identified": 0, "already_annotated": 0, "enqueued": 0}

        argv = [
            "quality_monitor_cli",
            "--tenant-id",
            "acme",
            "--llm-model",
            "test-model",
            "--phoenix-url",
            "http://example:26006",
            "--annotation-cycle",
        ]
        with (
            patch(
                "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
                return_value=fake_manager,
            ),
            patch.object(
                quality_monitor_cli, "_build_phoenix_provider", return_value=None
            ),
            patch.object(
                quality_monitor_cli, "run_annotation_cycle", side_effect=_fake_cycle
            ),
            patch("cogniverse_evaluation.quality_monitor.QualityMonitor"),
            patch("sys.argv", argv),
        ):
            with pytest.raises(SystemExit) as excinfo:
                quality_monitor_cli.main()

        assert excinfo.value.code == 0
        assert seen["http_endpoint"] == "http://example:26006"
        assert (
            fake_manager.config.provider_config["http_endpoint"]
            == "http://example:26006"
        )
