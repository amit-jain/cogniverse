"""Real-Phoenix round-trip for TraceToInstructionConverter.convert().

Emits one routing span through the canonical production span writer and adds
one approved annotation. The converter reads the real Phoenix frame, projects
the routing payload onto exact model-facing JSON, and resolves the annotation
span ID from Phoenix's index-backed shape.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_finetuning.dataset.trace_converter import (
    TraceToInstructionConverter,
    TraceToTrajectoryConverter,
)
from cogniverse_foundation.telemetry.span_contract import OP_ROUTING, record_span_io

pytestmark = pytest.mark.integration


async def _wait_for_named_spans(provider, project, names):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=1000,
        )
        if spans is not None and not spans.empty and "name" in spans.columns:
            matches = spans[spans["name"].isin(names)]
            if set(matches["name"]) == set(names):
                return matches
        await asyncio.sleep(2)
    raise AssertionError(f"spans {sorted(names)} not found in {project}")


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
    ) as span:
        record_span_io(
            span,
            input_value="find sunset videos",
            output={"recommended_agent": "video_search"},
            operation=OP_ROUTING,
        )
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

    # The approved annotation and span can become visible on different polls.
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
    assert ex.output == '{"recommended_agent":"video_search"}'
    assert ex.metadata["span_id"] == span_id
    assert dataset.metadata["approved_annotations"] == 1
    assert dataset.metadata["agent_type"] == "routing"


@pytest.mark.asyncio
async def test_convert_rejects_conflicting_annotation_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = "sft_conflicting_annotation"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"
    span_name = "routing_agent.conflicting_review"
    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )

    with telemetry_manager.span(
        name=span_name,
        tenant_id=tenant_id,
        project_name=project_name,
    ) as span:
        record_span_io(
            span,
            input_value="find the product launch recording",
            output={"recommended_agent": "video_search"},
            operation=OP_ROUTING,
        )
    telemetry_manager.force_flush(timeout_millis=10000)
    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )
    source_spans = await _wait_for_named_spans(provider, full_project, {span_name})
    span_id = source_spans.iloc[0]["context.span_id"]

    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_approval",
        label="rejected",
        score=1.0,
        metadata={"feedback": "The route was rejected by the reviewer."},
        project=full_project,
    )

    annotations = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        annotations = await provider.annotations.get_annotations(
            spans_df=source_spans,
            project=full_project,
            annotation_names=["human_approval"],
        )
        if len(annotations) == 1:
            break
        await asyncio.sleep(2)
    assert annotations is not None
    assert [
        (
            row["result.label"],
            row["result.score"],
            row["metadata"]["feedback"],
        )
        for _, row in annotations.iterrows()
    ] == [
        (
            "rejected",
            1.0,
            "The route was rejected by the reviewer.",
        )
    ]

    with pytest.raises(
        ValueError,
        match="^Insufficient approved annotations: 0 < 1$",
    ):
        await TraceToInstructionConverter(provider).convert(
            project=full_project,
            agent_type="routing",
            min_annotations=1,
        )


@pytest.mark.asyncio
async def test_convert_builds_canonical_trajectory_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = "trajectory_rt"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"
    session_id = "routing-session-canonical"
    span_specs = [
        (
            "routing_agent.trajectory_first",
            "find sunset videos",
            {
                "recommended_agent": "video_search",
                "confidence": 0.99,
                "reasoning": "The request asks for videos.",
            },
        ),
        (
            "routing_agent.trajectory_second",
            "find launch documents",
            {
                "recommended_agent": "document_agent",
                "confidence": 0.97,
                "reasoning": "The request asks for documents.",
            },
        ),
    ]

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )
    for name, query, output in span_specs:
        with telemetry_manager.session_span(
            name=name,
            tenant_id=tenant_id,
            session_id=session_id,
            project_name=project_name,
        ) as span:
            record_span_io(
                span,
                input_value=query,
                output=output,
                operation=OP_ROUTING,
            )
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )
    source_spans = await _wait_for_named_spans(
        provider, full_project, {name for name, _, _ in span_specs}
    )

    dataset = await TraceToTrajectoryConverter(provider).convert(
        project=full_project,
        agent_type="routing",
        min_turns_per_session=2,
    )

    assert len(dataset.trajectories) == 1
    trajectory = dataset.trajectories[0]
    assert trajectory.session_id == session_id
    assert [turn.query for turn in trajectory.turns] == [
        "find sunset videos",
        "find launch documents",
    ]
    assert [turn.response for turn in trajectory.turns] == [
        '{"recommended_agent":"video_search"}',
        '{"recommended_agent":"document_agent"}',
    ]
    assert {turn.span_id for turn in trajectory.turns} == set(
        source_spans["context.span_id"]
    )
    assert dataset.metadata["total_sessions"] == 1
    assert dataset.metadata["total_turns"] == 2


@pytest.mark.asyncio
async def test_convert_rejects_malformed_trajectory_span_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = "trajectory_malformed_rt"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"
    session_id = "routing-session-malformed"
    span_specs = [
        (
            "routing_agent.valid_turn",
            "find sunset videos",
            {"recommended_agent": "video_search"},
        ),
        (
            "routing_agent.malformed_turn",
            "find launch documents",
            {"confidence": 0.99},
        ),
    ]

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )
    for name, query, output in span_specs:
        with telemetry_manager.session_span(
            name=name,
            tenant_id=tenant_id,
            session_id=session_id,
            project_name=project_name,
        ) as span:
            record_span_io(
                span,
                input_value=query,
                output=output,
                operation=OP_ROUTING,
            )
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )
    source_spans = await _wait_for_named_spans(
        provider, full_project, {name for name, _, _ in span_specs}
    )
    malformed_span_id = source_spans[
        source_spans["name"] == "routing_agent.malformed_turn"
    ].iloc[0]["context.span_id"]

    with pytest.raises(
        ValueError,
        match=(
            rf"routing trajectory span {malformed_span_id} turn 2 requires exactly "
            r"the recommended_agent field"
        ),
    ):
        await TraceToTrajectoryConverter(provider).convert(
            project=full_project,
            agent_type="routing",
            min_turns_per_session=2,
        )
