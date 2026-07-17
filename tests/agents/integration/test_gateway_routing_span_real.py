"""Real GatewayAgent -> gateway + routing spans -> real consumers.

Drives the REAL GatewayAgent (only GLiNER is mocked to avoid a model
download), lets it emit its real cogniverse.gateway and cogniverse.routing
spans into real Phoenix, then asserts the real consumers read them back
through the canonical span contract: the gateway-threshold reader sees the
decision on output.value, and RoutingEvaluator returns the same chosen agent.

A mock-telemetry unit test confirms only the calibration arithmetic; it cannot
confirm that a real gateway emits a span these consumers can actually read.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingEvaluator
from cogniverse_foundation.telemetry.span_contract import read_span_io

pytestmark = pytest.mark.integration


async def _fetch_named_span(real_telemetry, tenant_id, span_name):
    canonical = canonical_tenant_id(tenant_id)
    project = real_telemetry.config.get_project_name(canonical)
    provider = real_telemetry.get_provider(tenant_id=canonical, project_name=project)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=1000,
        )
        if spans is not None and not spans.empty and "name" in spans.columns:
            hit = spans[spans["name"] == span_name]
            if not hit.empty:
                return hit.iloc[0], provider
        await asyncio.sleep(2)
    return None, None


@pytest.mark.asyncio
async def test_real_gateway_and_routing_spans_read_by_consumers(real_telemetry):
    from cogniverse_agents.gateway_agent import (
        GatewayAgent,
        GatewayDeps,
        GatewayInput,
    )

    tenant_id = "gw-route-real"
    agent = GatewayAgent(deps=GatewayDeps(), port=19114)
    agent._gliner_model = MagicMock()
    agent._gliner_model.predict_entities.return_value = [
        {"text": "videos", "label": "video_content", "score": 0.92},
    ]
    agent.set_telemetry_manager(real_telemetry)

    await agent.process(GatewayInput(query="cooking videos", tenant_id=tenant_id))

    gateway_row, _ = await _fetch_named_span(
        real_telemetry, tenant_id, "cogniverse.gateway"
    )
    routing_row, provider = await _fetch_named_span(
        real_telemetry, tenant_id, "cogniverse.routing"
    )
    assert gateway_row is not None, "cogniverse.gateway span not indexed"
    assert routing_row is not None, "cogniverse.routing span not indexed"

    # Gateway decision, read off the canonical output.value slot.
    gw_out = read_span_io(gateway_row)["output"]
    assert isinstance(gw_out, dict)
    assert gw_out["routed_to"] == "search_agent"
    assert gw_out["modality"] == "video"
    assert isinstance(gw_out["confidence"], float)

    # Every routing decision records the calibration it ran under, so
    # serving state is assertable from telemetry alone (the runtime pod
    # emits no INFO logs to scrape).
    routing_out = read_span_io(routing_row)["output"]
    assert (
        routing_out["fast_path_confidence_threshold"]
        == agent.deps.fast_path_confidence_threshold
    )
    assert routing_out["gliner_threshold"] == agent.deps.gliner_threshold

    # RoutingEvaluator reads the same decision from the routing span.
    outcome, metrics = RoutingEvaluator(provider).evaluate_routing_decision(
        routing_row.to_dict()
    )
    assert metrics["chosen_agent"] == gw_out["routed_to"]
    assert metrics["confidence"] == gw_out["confidence"]
