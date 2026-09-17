"""Runtime Phoenix clients reach the Phoenix the deployment configures.

The ingestion artifact store and the quality monitor's provider are built on
the endpoints the deployment names, against a real Phoenix container; a
half-configured pair fails naming what is set instead of reaching for a
Phoenix on localhost.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.quality_monitor_cli import _build_phoenix_provider
from cogniverse_runtime.routers.ingestion import _lookup_artifact_manager
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.integration]


def _config_manager(telemetry_url: str, collector_endpoint: str) -> ConfigManager:
    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_system_config(
        SystemConfig(
            telemetry_url=telemetry_url,
            telemetry_collector_endpoint=collector_endpoint,
        )
    )
    return manager


def test_ingestion_artifact_store_round_trips_through_the_configured_phoenix(
    phoenix_container,
):
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    tenant_id = f"epres{uuid.uuid4().hex[:8]}:ingest"
    content = f'{{"marker": "{uuid.uuid4().hex}"}}'
    manager = _lookup_artifact_manager(
        tenant_id,
        _config_manager(
            phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
        ),
    )
    asyncio.run(manager.save_blob("endpoint_resolution", "probe", content))

    reader_provider = PhoenixProvider()
    reader_provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        }
    )
    reader = ArtifactManager(telemetry_provider=reader_provider, tenant_id=tenant_id)

    assert manager._provider._http_endpoint == phoenix_container["http_endpoint"]
    assert asyncio.run(reader.load_blob("endpoint_resolution", "probe")) == content


@pytest.mark.parametrize(
    ("telemetry_url", "collector_endpoint"),
    [("http://cogniverse-phoenix:6006", ""), ("", "cogniverse-phoenix:4317")],
)
def test_ingestion_artifact_store_refuses_a_half_configured_phoenix(
    telemetry_url, collector_endpoint
):
    with pytest.raises(ValueError) as excinfo:
        _lookup_artifact_manager(
            "acme:acme", _config_manager(telemetry_url, collector_endpoint)
        )

    assert str(excinfo.value) == (
        "artifact store for tenant 'acme:acme' needs both Phoenix endpoints; "
        f"SystemConfig has telemetry_url={telemetry_url!r}, "
        f"telemetry_collector_endpoint={collector_endpoint!r}"
    )


def test_quality_monitor_provider_reads_spans_from_the_configured_phoenix(
    phoenix_container,
):
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tenant_id = f"epres{uuid.uuid4().hex[:8]}:qm"
    project = f"cogniverse-{tenant_id}"
    span_name = f"endpoint.resolution.{uuid.uuid4().hex[:8]}"
    tracer_provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": project})
    )
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(
            OTLPSpanExporter(endpoint=phoenix_container["otlp_endpoint"], insecure=True)
        )
    )
    with tracer_provider.get_tracer("endpoint-resolution").start_as_current_span(
        span_name
    ):
        pass
    tracer_provider.shutdown()

    provider = _build_phoenix_provider(
        tenant_id=tenant_id,
        http_endpoint=phoenix_container["http_endpoint"],
        grpc_endpoint=phoenix_container["grpc_endpoint"],
    )
    start = datetime.now(timezone.utc) - timedelta(minutes=10)
    deadline = time.monotonic() + 60
    names: list[str] = []
    while time.monotonic() < deadline:
        spans = asyncio.run(
            provider.traces.get_spans(
                project=project,
                start_time=start,
                end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                filters={"name": span_name},
                limit=10,
            )
        )
        names = list(spans["name"]) if not spans.empty else []
        if names:
            break
        time.sleep(1)

    assert names == [span_name]
