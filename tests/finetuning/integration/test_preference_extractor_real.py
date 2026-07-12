"""Real-Phoenix round-trip for PreferencePairExtractor.extract().

Emits a routing-agent span (input.query / output.response) to a real Phoenix
container, attaches approved + rejected human annotations carrying the edited
responses, then runs extract() end to end and asserts the reconstructed
preference pair. This exercises the real span + annotation query/flatten
contract (``attributes.input.query``, ``result.label``, ``metadata.response``)
that a mocked provider can only assume.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_finetuning.dataset.preference_extractor import PreferencePairExtractor

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
async def test_extract_builds_pair_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = "pref_rt"
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

    # Recover the emitted span's id from Phoenix (the canonical id the
    # annotation API keys on).
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

    # The human reviewer's edited responses ride on the annotation metadata —
    # this is what distinguishes chosen from rejected for the same span.
    # Distinct annotation names keep both on the span (Phoenix upserts by
    # span_id + annotation_name + identifier).
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_review_approved",
        label="approved",
        score=1.0,
        metadata={"response": "good route"},
        project=full_project,
    )
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_review_rejected",
        label="rejected",
        score=0.0,
        metadata={"response": "bad route"},
        project=full_project,
    )

    extractor = PreferencePairExtractor(provider=provider)

    dataset = None
    last_err = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            end = datetime.now(timezone.utc)
            dataset = await extractor.extract(
                project=full_project,
                agent_type="routing",
                min_pairs=1,
                start_time=end - timedelta(hours=1),
                end_time=end,
            )
            break
        except ValueError as e:
            last_err = e
            await asyncio.sleep(2)
    assert dataset is not None, f"extract() never produced the pair: {last_err}"

    assert len(dataset.pairs) == 1
    pair = dataset.pairs[0]
    assert pair.prompt == "find sunset videos"
    assert pair.chosen == "good route"
    assert pair.rejected == "bad route"
    assert pair.metadata["span_id"] == span_id
    assert dataset.metadata["agent_type"] == "routing"
    assert dataset.metadata["preference_pairs"] == 1


@pytest.mark.asyncio
async def test_extract_matches_gateway_named_span(phoenix_container, telemetry_manager):
    """A GatewayAgent.process span must be recognized as a routing span.

    The routing keyword map lacked 'gateway' (its two siblings carry it), so
    DPO was recommended on gateway spans then extract() found zero and raised.
    """
    tenant_id = "pref_gw_rt"
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
        name="GatewayAgent.process",
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
            match = spans[spans["name"] == "GatewayAgent.process"]
            if not match.empty:
                span_id = match.iloc[0]["context.span_id"]
                break
        await asyncio.sleep(2)
    assert span_id is not None, f"gateway span not found in {full_project}"

    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_review_approved",
        label="approved",
        score=1.0,
        metadata={"response": "good route"},
        project=full_project,
    )
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_review_rejected",
        label="rejected",
        score=0.0,
        metadata={"response": "bad route"},
        project=full_project,
    )

    extractor = PreferencePairExtractor(provider=provider)

    dataset = None
    last_err = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            end = datetime.now(timezone.utc)
            dataset = await extractor.extract(
                project=full_project,
                agent_type="routing",
                min_pairs=1,
                start_time=end - timedelta(hours=1),
                end_time=end,
            )
            break
        except ValueError as e:
            last_err = e
            await asyncio.sleep(2)
    assert dataset is not None, f"extract() never produced the pair: {last_err}"

    assert len(dataset.pairs) == 1
    pair = dataset.pairs[0]
    assert pair.prompt == "find sunset videos"
    assert pair.chosen == "good route"
    assert pair.rejected == "bad route"
    assert pair.metadata["span_id"] == span_id
    assert dataset.metadata["agent_type"] == "routing"
    assert dataset.metadata["preference_pairs"] == 1
