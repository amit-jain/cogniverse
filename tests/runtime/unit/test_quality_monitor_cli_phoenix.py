"""Unit tests for quality_monitor_cli's PhoenixProvider injection (audit fix #15).

Verifies that ``_build_phoenix_provider``:
1. Builds a real PhoenixProvider with HTTP and gRPC endpoints derived from
   the ``--phoenix-url`` argument.
2. Honors the ``PHOENIX_GRPC_ENDPOINT`` env var override for non-default
   gRPC ports.
3. Returns ``None`` (and logs a warning) when PhoenixProvider construction
   fails — letting the QualityMonitor degrade to naive verdicts instead of
   crashing the sidecar.

Before this fix the CLI never injected ``telemetry_provider``, leaving
the XGBoost training-decision gate in QualityMonitor.check_thresholds as
unreachable dead code 100% of the time.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_runtime.quality_monitor_cli import _build_phoenix_provider


@pytest.mark.unit
@pytest.mark.ci_fast
class TestBuildPhoenixProvider:
    def test_builds_provider_with_http_and_default_grpc(self, monkeypatch):
        """When PHOENIX_GRPC_ENDPOINT is unset, the helper derives the gRPC
        endpoint from the HTTP endpoint host on the standard OTLP port 4317."""
        monkeypatch.delenv("PHOENIX_GRPC_ENDPOINT", raising=False)

        with patch(
            "cogniverse_telemetry_phoenix.provider.PhoenixProvider"
        ) as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            result = _build_phoenix_provider(
                tenant_id="acme",
                http_endpoint="http://phoenix:6006",
            )

        assert result is mock_instance
        mock_instance.initialize.assert_called_once()
        config = mock_instance.initialize.call_args[0][0]
        assert config["tenant_id"] == "acme"
        assert config["http_endpoint"] == "http://phoenix:6006"
        # Default gRPC: same host, port 4317.
        assert config["grpc_endpoint"] == "phoenix:4317"

    def test_env_var_overrides_grpc_endpoint(self, monkeypatch):
        """A custom PHOENIX_GRPC_ENDPOINT env var must override the default."""
        monkeypatch.setenv("PHOENIX_GRPC_ENDPOINT", "phoenix-otlp:14317")

        with patch(
            "cogniverse_telemetry_phoenix.provider.PhoenixProvider"
        ) as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            _build_phoenix_provider(
                tenant_id="acme",
                http_endpoint="http://phoenix:6006",
            )

        config = mock_instance.initialize.call_args[0][0]
        assert config["grpc_endpoint"] == "phoenix-otlp:14317"

    def test_returns_none_on_initialization_failure(self, monkeypatch, caplog):
        """If PhoenixProvider.initialize() raises (e.g., wrong endpoint),
        the helper must return None and log a warning — not crash the
        sidecar at startup."""
        monkeypatch.delenv("PHOENIX_GRPC_ENDPOINT", raising=False)

        with patch(
            "cogniverse_telemetry_phoenix.provider.PhoenixProvider"
        ) as MockProvider:
            mock_instance = MagicMock()
            mock_instance.initialize.side_effect = ConnectionError("nope")
            MockProvider.return_value = mock_instance

            with caplog.at_level("WARNING"):
                result = _build_phoenix_provider(
                    tenant_id="acme",
                    http_endpoint="http://phoenix:6006",
                )

        assert result is None
        assert any(
            "Failed to build PhoenixProvider" in r.message for r in caplog.records
        )

    def test_returns_none_on_import_failure(self, monkeypatch):
        """If the cogniverse_telemetry_phoenix package isn't installed
        (e.g., a stripped-down test image), the helper must degrade
        gracefully rather than ImportError-ing the sidecar at startup."""
        monkeypatch.delenv("PHOENIX_GRPC_ENDPOINT", raising=False)

        # Patch the import path to raise ImportError on import.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cogniverse_telemetry_phoenix.provider":
                raise ImportError("module not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            result = _build_phoenix_provider(
                tenant_id="acme",
                http_endpoint="http://phoenix:6006",
            )

        assert result is None

    def test_falls_back_to_localhost_for_unparseable_http_endpoint(self, monkeypatch):
        """If urlparse fails to extract a hostname (e.g. malformed URL),
        the helper must still build a workable gRPC endpoint."""
        monkeypatch.delenv("PHOENIX_GRPC_ENDPOINT", raising=False)

        with patch(
            "cogniverse_telemetry_phoenix.provider.PhoenixProvider"
        ) as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            _build_phoenix_provider(
                tenant_id="acme",
                http_endpoint="not-a-url",
            )

        config = mock_instance.initialize.call_args[0][0]
        # urlparse("not-a-url").hostname is None, so falls back to "localhost"
        assert config["grpc_endpoint"] == "localhost:4317"
