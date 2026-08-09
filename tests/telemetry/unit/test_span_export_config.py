"""Span-export config knobs must reach the exporter, not just round-trip.

``BatchExportConfig`` (max_queue_size / max_export_batch_size /
export_timeout_millis / schedule_delay_millis) and the resource attributes
(``service_version`` + ``extra_resource_attributes``) were serialized and
deserialized but never forwarded to ``phoenix.otel.register`` — every
TracerProvider ran with SDK-default batch settings and no version resource.
These tests build real Phoenix TracerProviders (no network I/O happens at
construction) and assert the knobs land on the live processor + resource.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, get_ident
from unittest.mock import MagicMock

import pytest
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_telemetry_phoenix.provider import (
    PhoenixProvider,
    _CheckedSynchronousSpanProcessor,
)


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
        raise_on_export_failure=False,
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
        raise_on_export_failure=False,
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


def test_optional_sync_export_uses_standard_processor():
    provider = PhoenixProvider()

    tracer_provider = provider.configure_span_export(
        endpoint="localhost:14317",
        project_name="cogniverse-acme-sync",
        use_batch_export=False,
        batch_config=BatchExportConfig(max_queue_size=777),
        resource_attributes=None,
        raise_on_export_failure=False,
    )
    try:
        processor = _single_processor(tracer_provider)
        assert type(processor).__name__ == "SimpleSpanProcessor"
    finally:
        tracer_provider.shutdown()


def test_required_sync_export_uses_checked_processor():
    provider = PhoenixProvider()
    batch_config = BatchExportConfig(
        max_queue_size=777,
        export_timeout_millis=2500,
    )

    tracer_provider = provider.configure_span_export(
        endpoint="localhost:14317",
        project_name="cogniverse-acme-required",
        use_batch_export=False,
        raise_on_export_failure=True,
        batch_config=batch_config,
        resource_attributes=None,
    )
    try:
        processor = _single_processor(tracer_provider)
        assert type(processor).__name__ == "_CheckedSynchronousSpanProcessor"
        assert processor._exporter._timeout == 2.0
    finally:
        tracer_provider.shutdown()


def test_required_export_requires_explicit_positive_deadline():
    provider = PhoenixProvider()

    with pytest.raises(
        ValueError,
        match="raise_on_export_failure requires an explicit batch_config deadline",
    ):
        provider.configure_span_export(
            endpoint="localhost:14317",
            project_name="cogniverse-acme-required",
            use_batch_export=False,
            raise_on_export_failure=True,
            batch_config=None,
            resource_attributes=None,
        )

    with pytest.raises(
        ValueError,
        match="export_timeout_millis must be positive",
    ):
        provider.configure_span_export(
            endpoint="localhost:14317",
            project_name="cogniverse-acme-required",
            use_batch_export=False,
            raise_on_export_failure=True,
            batch_config=BatchExportConfig(export_timeout_millis=0),
            resource_attributes=None,
        )


def test_required_export_rejects_batch_processing():
    provider = PhoenixProvider()

    with pytest.raises(
        ValueError,
        match="raise_on_export_failure requires synchronous span export",
    ):
        provider.configure_span_export(
            endpoint="localhost:14317",
            project_name="cogniverse-acme-required",
            use_batch_export=True,
            raise_on_export_failure=True,
            batch_config=BatchExportConfig(),
            resource_attributes=None,
        )


def test_required_export_uses_always_on_sampling(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    provider = PhoenixProvider()

    tracer_provider = provider.configure_span_export(
        endpoint="localhost:14317",
        project_name="cogniverse-acme-required",
        use_batch_export=False,
        raise_on_export_failure=True,
        batch_config=BatchExportConfig(),
        resource_attributes=None,
    )
    try:
        assert tracer_provider.sampler is ALWAYS_ON
    finally:
        tracer_provider.shutdown()


def test_failed_required_processor_replacement_closes_partial_provider(
    monkeypatch,
):
    import phoenix.otel

    tracer_provider = MagicMock()
    tracer_provider.add_span_processor.side_effect = ValueError(
        "processor replacement failed"
    )
    exporter = MagicMock()
    monkeypatch.setattr(
        phoenix.otel,
        "register",
        MagicMock(return_value=tracer_provider),
    )
    monkeypatch.setattr(
        phoenix.otel,
        "GRPCSpanExporter",
        MagicMock(return_value=exporter),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Phoenix span export configuration failed: processor replacement failed"
        ),
    ) as exc_info:
        PhoenixProvider().configure_span_export(
            endpoint="localhost:14317",
            project_name="cogniverse-acme-required",
            use_batch_export=False,
            batch_config=BatchExportConfig(export_timeout_millis=2500),
            resource_attributes=None,
            raise_on_export_failure=True,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    tracer_provider.shutdown.assert_called_once_with()
    exporter.shutdown.assert_called_once_with()


def test_checked_sync_processor_surfaces_rejection():
    exporter = MagicMock()
    exporter.export.return_value = SpanExportResult.FAILURE
    span = MagicMock()
    span.context.trace_flags.sampled = True

    with pytest.raises(
        RuntimeError,
        match=(
            "Phoenix rejected required span export: "
            "project=cogniverse-acme-required endpoint=localhost:14317"
        ),
    ):
        _CheckedSynchronousSpanProcessor(
            exporter,
            endpoint="localhost:14317",
            project_name="cogniverse-acme-required",
        ).on_end(span)

    exporter.export.assert_called_once_with((span,))


def test_checked_sync_processor_suppresses_instrumentation_and_restores_context():
    exporter = MagicMock()
    observed_suppression = []

    def export(_batch):
        observed_suppression.append(get_value(_SUPPRESS_INSTRUMENTATION_KEY))
        raise ConnectionError("collector disconnected")

    exporter.export.side_effect = export
    span = MagicMock()
    span.context.trace_flags.sampled = True
    processor = _CheckedSynchronousSpanProcessor(
        exporter,
        endpoint="localhost:14317",
        project_name="cogniverse-acme-required",
    )

    with pytest.raises(RuntimeError, match="cogniverse-acme-required") as error:
        processor.on_end(span)

    assert isinstance(error.value.__cause__, ConnectionError)
    assert observed_suppression == [True]
    assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is None


def test_checked_sync_processor_exports_each_concurrent_span_once():
    worker_count = 4
    barrier = Barrier(worker_count)
    lock = Lock()
    exported_names = []
    exporter = MagicMock()

    def export(batch):
        barrier.wait(timeout=5)
        with lock:
            exported_names.append(batch[0].name)
        return SpanExportResult.SUCCESS

    exporter.export.side_effect = export
    processor = _CheckedSynchronousSpanProcessor(
        exporter,
        endpoint="localhost:14317",
        project_name="cogniverse-acme-required",
    )
    spans = []
    for index in range(worker_count):
        span = MagicMock()
        span.context.trace_flags.sampled = True
        span.name = f"span-{index}"
        spans.append(span)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(processor.on_end, spans))

    assert sorted(exported_names) == [f"span-{index}" for index in range(worker_count)]
    assert exporter.export.call_count == worker_count


def test_checked_tracer_setup_preserves_tenant_project_endpoint_and_cause():
    TelemetryManager.reset()
    manager = TelemetryManager(
        TelemetryConfig(
            otlp_endpoint="localhost:14317",
            batch_config=BatchExportConfig(use_sync_export=False),
        )
    )
    provider = MagicMock()
    provider.configure_span_export.side_effect = ValueError("invalid endpoint")
    manager.get_provider = MagicMock(return_value=provider)

    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to create required telemetry exporter: tenant=acme "
                "project=cogniverse-acme-search endpoint=localhost:14317"
            ),
        ) as error:
            manager._create_checked_tracer_for_project("acme", "search")
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == "invalid endpoint"
    finally:
        TelemetryManager.reset()


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
        raise_on_export_failure=False,
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
    assert kwargs["raise_on_export_failure"] is False


def test_manager_can_force_checked_export_for_required_spans():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        otlp_endpoint="localhost:4317",
        batch_config=BatchExportConfig(use_sync_export=False),
    )
    manager._project_configs = {}
    fake_provider = MagicMock()
    manager.get_provider = MagicMock(return_value=fake_provider)

    manager._create_tenant_provider_for_project(
        "acme",
        "search",
        force_sync_export=True,
    )

    kwargs = fake_provider.configure_span_export.call_args.kwargs
    assert kwargs["use_batch_export"] is False
    assert kwargs["raise_on_export_failure"] is True


def test_required_span_bypasses_optional_component_filter():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(level=TelemetryLevel.BASIC)
    manager._project_configs = {}
    tracer = MagicMock()
    provider = MagicMock()
    emitted_span = MagicMock()
    emitted_span.is_recording.return_value = True
    emitted_span.get_span_context.return_value.trace_flags.sampled = True
    tracer.start_as_current_span.return_value.__enter__.return_value = emitted_span
    manager._create_checked_tracer_for_project = MagicMock(
        return_value=(provider, tracer)
    )

    with manager.span(
        "cogniverse.orchestration",
        tenant_id="acme",
        component="agents",
        require_export=True,
    ) as span:
        assert span is emitted_span

    manager._create_checked_tracer_for_project.assert_called_once_with("acme", None)
    provider.shutdown.assert_called_once_with()


def test_required_span_rejects_missing_tracer_without_noop_fallback():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        otlp_endpoint="localhost:14317",
        level=TelemetryLevel.VERBOSE,
    )
    manager._project_configs = {}
    provider = MagicMock()
    manager._create_checked_tracer_for_project = MagicMock(
        return_value=(provider, None)
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Required telemetry exporter returned no tracer: tenant=acme "
            "project=cogniverse-acme-search endpoint=localhost:14317"
        ),
    ):
        with manager.span(
            "cogniverse.orchestration",
            tenant_id="acme",
            project_name="search",
            require_export=True,
        ):
            pass

    provider.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_required_span_ends_off_loop_and_times_out_with_context():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        service_name="test-service",
        environment="test",
        otlp_endpoint="localhost:14317",
        batch_config=BatchExportConfig(export_timeout_millis=50),
    )
    manager._project_configs = {}
    tracer = MagicMock()
    span = MagicMock()
    span.is_recording.return_value = True
    span.get_span_context.return_value.trace_flags.sampled = True
    tracer.start_span.return_value = span
    provider = MagicMock()
    caller_thread = get_ident()
    setup_thread = None

    def create_lease(tenant_id, project_name):
        nonlocal setup_thread
        assert (tenant_id, project_name) == ("acme", "search")
        setup_thread = get_ident()
        return provider, tracer

    manager._create_checked_tracer_for_project = MagicMock(side_effect=create_lease)
    export_entered = Event()
    export_finished = Event()
    release_export = Event()

    def end_span():
        export_entered.set()
        release_export.wait(timeout=5)
        export_finished.set()

    span.end.side_effect = end_span
    provider.shutdown.side_effect = release_export.set
    manager._tenant_providers = {}
    manager._tenant_tracers = {}
    loop_ticked = asyncio.Event()

    async def tick_loop():
        await asyncio.sleep(0.01)
        loop_ticked.set()

    ticker = asyncio.create_task(tick_loop())
    try:
        with pytest.raises(
            TimeoutError,
            match=(
                "Required telemetry export timed out: tenant=acme "
                "project=cogniverse-acme-search endpoint=localhost:14317 "
                "timeout_seconds=0.05"
            ),
        ):
            async with manager.required_span(
                "cogniverse.orchestration",
                tenant_id="acme",
                project_name="search",
                attributes={"run.id": "run-123"},
            ) as emitted_span:
                assert emitted_span is span
        await ticker
        assert loop_ticked.is_set()
        assert setup_thread == caller_thread
        assert export_entered.is_set()
        assert export_finished.is_set()
        provider.shutdown.assert_called_once_with()
        span.set_attribute.assert_any_call("tenant.id", "acme")
        span.set_attribute.assert_any_call("service.name", "test-service")
        span.set_attribute.assert_any_call("environment", "test")
        span.set_attribute.assert_any_call("run.id", "run-123")
    finally:
        release_export.set()


@pytest.mark.asyncio
async def test_concurrent_required_spans_use_independent_exporter_leases():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        service_name="test-service",
        environment="test",
        otlp_endpoint="localhost:14317",
        batch_config=BatchExportConfig(export_timeout_millis=50),
    )
    manager._project_configs = {}
    manager._tenant_providers = {}
    manager._tenant_tracers = {}

    hung_provider = MagicMock()
    successful_provider = MagicMock()
    hung_tracer = MagicMock()
    successful_tracer = MagicMock()
    hung_span = MagicMock()
    successful_span = MagicMock()
    for span in (hung_span, successful_span):
        span.is_recording.return_value = True
        span.get_span_context.return_value.trace_flags.sampled = True
    hung_tracer.start_span.return_value = hung_span
    successful_tracer.start_span.return_value = successful_span
    export_entered = Event()
    export_finished = Event()
    release_export = Event()

    def end_hung_span():
        export_entered.set()
        release_export.wait(timeout=5)
        export_finished.set()

    hung_span.end.side_effect = end_hung_span
    hung_provider.shutdown.side_effect = release_export.set

    def create_lease(tenant_id, project_name):
        assert project_name == "search"
        if tenant_id == "hung":
            return hung_provider, hung_tracer
        if tenant_id == "successful":
            return successful_provider, successful_tracer
        raise AssertionError(tenant_id)

    manager._create_checked_tracer_for_project = MagicMock(side_effect=create_lease)

    async def emit(tenant_id):
        async with manager.required_span(
            f"required-{tenant_id}",
            tenant_id=tenant_id,
            project_name="search",
        ):
            pass

    try:
        results = await asyncio.gather(
            emit("hung"),
            emit("successful"),
            return_exceptions=True,
        )
        assert isinstance(results[0], TimeoutError)
        assert results[1] is None
        assert export_entered.is_set()
        assert export_finished.is_set()
        hung_provider.shutdown.assert_called_once_with()
        successful_provider.shutdown.assert_called_once_with()
        successful_span.end.assert_called_once_with()
        assert manager._tenant_providers == {}
        assert manager._tenant_tracers == {}
    finally:
        release_export.set()


@pytest.mark.asyncio
async def test_required_span_cancellation_joins_export_worker():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig(
        service_name="test-service",
        environment="test",
        otlp_endpoint="localhost:14317",
        batch_config=BatchExportConfig(export_timeout_millis=50),
    )
    manager._project_configs = {}
    manager._tenant_providers = {}
    manager._tenant_tracers = {}
    provider = MagicMock()
    tracer = MagicMock()
    span = MagicMock()
    span.is_recording.return_value = True
    span.get_span_context.return_value.trace_flags.sampled = True
    tracer.start_span.return_value = span
    manager._create_checked_tracer_for_project = MagicMock(
        return_value=(provider, tracer)
    )
    export_entered = Event()
    export_finished = Event()
    release_export = Event()

    def end_span():
        export_entered.set()
        release_export.wait(timeout=5)
        export_finished.set()

    span.end.side_effect = end_span
    provider.shutdown.side_effect = release_export.set

    async def emit():
        async with manager.required_span(
            "required-cancelled",
            tenant_id="cancelled",
            project_name="search",
        ):
            pass

    task = asyncio.create_task(emit())
    try:
        assert await asyncio.to_thread(export_entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert export_finished.is_set()
        provider.shutdown.assert_called_once_with()
        assert manager._tenant_providers == {}
        assert manager._tenant_tracers == {}
    finally:
        release_export.set()


def test_sync_required_span_rejects_disabled_otel_sdk(
    monkeypatch,
    unused_tcp_port,
):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    TelemetryManager.reset()
    endpoint = f"localhost:{unused_tcp_port}"
    manager = TelemetryManager(
        TelemetryConfig(
            level=TelemetryLevel.VERBOSE,
            otlp_endpoint=endpoint,
            batch_config=BatchExportConfig(use_sync_export=False),
        )
    )

    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Required telemetry span was not sampled: tenant=acme "
                "project=cogniverse-acme-search "
                f"endpoint={endpoint}"
            ),
        ):
            with manager.span(
                "cogniverse.orchestration",
                tenant_id="acme",
                project_name="search",
                require_export=True,
            ):
                pass
    finally:
        TelemetryManager.reset()


@pytest.mark.asyncio
async def test_async_required_span_rejects_disabled_otel_sdk(
    monkeypatch,
    unused_tcp_port,
):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    TelemetryManager.reset()
    endpoint = f"localhost:{unused_tcp_port}"
    manager = TelemetryManager(
        TelemetryConfig(
            level=TelemetryLevel.VERBOSE,
            otlp_endpoint=endpoint,
            batch_config=BatchExportConfig(use_sync_export=False),
        )
    )

    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Required telemetry span was not sampled: tenant=acme "
                "project=cogniverse-acme-search "
                f"endpoint={endpoint}"
            ),
        ):
            async with manager.required_span(
                "cogniverse.orchestration",
                tenant_id="acme",
                project_name="search",
            ):
                pass
    finally:
        TelemetryManager.reset()


@pytest.mark.asyncio
async def test_sync_required_span_rejects_event_loop_use():
    manager = object.__new__(TelemetryManager)
    manager.config = TelemetryConfig()

    with pytest.raises(
        RuntimeError,
        match="Use TelemetryManager.required_span from async code",
    ):
        with manager.span(
            "cogniverse.orchestration",
            tenant_id="acme",
            require_export=True,
        ):
            pass
