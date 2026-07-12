"""Real-Phoenix round-trip for TraceToInstructionConverter.convert().

Emits one routing span and one approved annotation. On the real Phoenix
annotations frame span_id is the INDEX (no column), so the old
``annotations_df['span_id']`` raised KeyError('span_id') from
_create_instruction_examples. This pins the index-aware lookup.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_finetuning.dataset.trace_converter import TraceToInstructionConverter

pytestmark = pytest.mark.integration


@pytest.fixture
def telemetry_manager(phoenix_container):
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    yield manager
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_convert_builds_example_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = "sft_rt"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )

    with telemetry_manager.span(
        name="routing_agent",
        tenant_id=tenant_id,
        project_name=project_name,
        attributes={
            "input.query": "find sunset videos",
            "output.response": "default route",
        },
    ):
        pass
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )

    span_id = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=full_project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=1000,
        )
        if spans is not None and not spans.empty and "name" in spans.columns:
            match = spans[spans["name"] == "routing_agent"]
            if not match.empty:
                span_id = match.iloc[0]["context.span_id"]
                break
        await asyncio.sleep(2)
    assert span_id is not None, f"routing_agent span not found in {full_project}"

    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_approval",
        label="approved",
        score=1.0,
        metadata={},
        project=full_project,
    )

    converter = TraceToInstructionConverter(provider)

    # convert raises ValueError while annotations lag (insufficient approved).
    # On the OLD code, once the approved annotation lands it raises
    # KeyError('span_id') — NOT caught here — so the test fails on old code.
    dataset = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            end = datetime.now(timezone.utc)
            dataset = await converter.convert(
                project=full_project,
                agent_type="routing",
                min_annotations=1,
                start_time=end - timedelta(hours=1),
                end_time=end,
            )
            break
        except ValueError:
            await asyncio.sleep(2)
    assert dataset is not None, "convert() never produced the example"

    assert len(dataset.examples) == 1
    ex = dataset.examples[0]
    assert (
        ex.instruction == "Route the following query to the appropriate modality agent."
    )
    assert ex.input == "find sunset videos"
    assert ex.output == "default route"
    assert ex.metadata["span_id"] == span_id
    assert dataset.metadata["approved_annotations"] == 1
    assert dataset.metadata["agent_type"] == "routing"
