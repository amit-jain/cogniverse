"""Span-export config knobs must reach the exporter, not just round-trip.

``BatchExportConfig`` (max_queue_size / max_export_batch_size /
export_timeout_millis / schedule_delay_millis) and the resource attributes
(``service_version`` + ``extra_resource_attributes``) were serialized and
deserialized but never forwarded to ``phoenix.otel.register`` — every
TracerProvider ran with SDK-default batch settings and no version resource.
These tests build real Phoenix TracerProviders (no network I/O happens at
construction) and assert the knobs land on the live processor + resource.
"""

from unittest.mock import MagicMock

from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider


def _single_processor(tracer_provider):
    processors = tracer_provider._active_span_processor._span_processors
    assert len(processors) == 1
    return processors[0]


def test_batch_knobs_reach_the_span_processor():
    provider = PhoenixProvider()
    batch = BatchExportConfig(
        max_queue_size=777,
        max_export_batch_size=99,
        export_timeout_millis=12_345,
        schedule_delay_millis=250,
    )

    tracer_provider = provider.configure_span_export(
        endpoint="localhost:14317",
        project_name="cogniverse-acme-search",
        use_batch_export=True,
        batch_config=batch,
        resource_attributes=None,
    )
    try:
        processor = _single_processor(tracer_provider)
        assert type(processor).__name__ == "BatchSpanProcessor"
        internals = processor._batch_processor
        assert internals._max_queue_size == 777
        assert internals._max_export_batch_size == 99
        assert internals._export_timeout_millis == 12_345
        assert internals._schedule_delay_millis == 250
        assert type(internals._exporter).__name__ == "GRPCSpanExporter"
        assert internals._exporter._endpoint == "localhost:14317"
    finally:
        tracer_provider.shutdown()


def test_resource_attributes_reach_the_tracer_provider():
    provider = PhoenixProvider()

    tracer_provider = provider.configure_span_export(
        endpoint="localhost:14317",
        project_name="cogniverse-acme-search",
        use_batch_export=True,
        batch_config=BatchExportConfig(),
        resource_attributes={
            "service.name": "video-search",
            "service.version": "9.9.9",
            "deployment.env": "ci",
        },
    )
    try:
        attrs = dict(tracer_provider.resource.attributes)
        # register() merges the Phoenix project attribute into ours.
        assert attrs["openinference.project.name"] == "cogniverse-acme-search"
        assert attrs["service.name"] == "video-search"
        assert attrs["service.version"] == "9.9.9"
        assert attrs["deployment.env"] == "ci"
    finally:
        tracer_provider.shutdown()


def test_sync_export_keeps_simple_processor():
    provider = PhoenixProvider()

    tracer_provider = provider.configure_span_export(
        endpoint="localhost:14317",
        project_name="cogniverse-acme-sync",
        use_batch_export=False,
        batch_config=BatchExportConfig(max_queue_size=777),
        resource_attributes=None,
    )
    try:
        processor = _single_processor(tracer_provider)
        assert type(processor).__name__ == "SimpleSpanProcessor"
    finally:
        tracer_provider.shutdown()


def test_manager_forwards_batch_config_and_resource_attributes():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        otlp_endpoint="localhost:4317",
        service_name="video-search",
        service_version="2.5.0",
        extra_resource_attributes={"deployment.env": "staging"},
        batch_config=BatchExportConfig(max_queue_size=64),
    )
    manager._project_configs = {}
    fake_provider = MagicMock()
    manager.get_provider = MagicMock(return_value=fake_provider)

    result = manager._create_tenant_provider_for_project("acme", "search")

    fake_provider.configure_span_export.assert_called_once_with(
        endpoint="localhost:4317",
        project_name="cogniverse-acme-search",
        use_batch_export=True,
        batch_config=manager.config.batch_config,
        resource_attributes={
            "service.name": "video-search",
            "service.version": "2.5.0",
            "deployment.env": "staging",
        },
    )
    assert result is fake_provider.configure_span_export.return_value


def test_manager_forwards_sync_mode_from_registered_project():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        otlp_endpoint="localhost:4317",
        service_version="2.5.0",
    )
    manager._project_configs = {
        "acme:search": {
            "otlp_endpoint": "localhost:24317",
            "use_sync_export": True,
        }
    }
    fake_provider = MagicMock()
    manager.get_provider = MagicMock(return_value=fake_provider)

    manager._create_tenant_provider_for_project("acme", "search")

    kwargs = fake_provider.configure_span_export.call_args.kwargs
    assert kwargs["endpoint"] == "localhost:24317"
    assert kwargs["use_batch_export"] is False
    assert kwargs["batch_config"] is manager.config.batch_config
