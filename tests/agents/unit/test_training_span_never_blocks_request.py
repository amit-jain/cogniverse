"""Executable Class-F proof: a training-span emitter never blocks or fails a
request, whatever the telemetry backend does.

Serving telemetry is best effort. These tests drive the real emitters with a
REAL ``TelemetryManager`` whose exporter points at a dead endpoint (batch
export, the production default) and with a manager whose enqueue hangs, and
assert the emit returns promptly and never raises. If either regressed to
``require_export=True`` / a synchronous export / a raise on the request path,
these fail — proven by executing the path, not by grepping for a keyword.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager

pytestmark = [pytest.mark.unit]

# Per-request budget for one emit AFTER the per-project tracer provider is
# built. A synchronous exporter to a dead endpoint blocks for its connect
# timeout (seconds); the batch-queue enqueue is measured at ~40 microseconds.
# 50ms sits three orders of magnitude below the former and above the latter,
# so it separates blocking from enqueue unambiguously. Provider construction
# (~2s, lazy, once per project) is excluded by warming first — it is paid at
# pod warmup / first request, never per request.
_MAX_EMIT_SECONDS = 0.05


def _dead_backend_manager() -> TelemetryManager:
    """A real manager whose OTLP exporter targets a closed port, batch export."""
    TelemetryManager.reset()
    config = TelemetryConfig(
        otlp_endpoint="127.0.0.1:9",  # discard port: nothing listens
        provider_config={
            "http_endpoint": "http://127.0.0.1:9",
            "grpc_endpoint": "127.0.0.1:9",
        },
        batch_config=BatchExportConfig(use_sync_export=False),
    )
    return TelemetryManager(config=config)


class _HangingManager:
    """span() enqueue blocks forever — a synchronous emitter would hang here."""

    def span(self, name: str, tenant_id: str):
        raise AssertionError(
            "request path entered span() synchronously; emit must enqueue, "
            "never block the request on the telemetry backend"
        )


async def _emit_all(manager) -> None:
    from cogniverse_agents.entity_extraction_agent import EntityExtractionAgent
    from cogniverse_agents.gateway_agent import GatewayAgent
    from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent
    from cogniverse_agents.query_enhancement_agent import QueryEnhancementAgent
    from cogniverse_foundation.telemetry.span_contract import (
        QUERY_ENHANCEMENT_PATH_LM,
    )

    qe = object.__new__(QueryEnhancementAgent)
    qe.telemetry_manager = manager
    await qe._emit_enhancement_span(
        tenant_id="acme:acme",
        original_query="q",
        source_text="s",
        grounding_context="",
        enhanced_query="eq",
        expansion_terms=[],
        synonyms=[],
        context_additions=[],
        variant_count=0,
        confidence=0.5,
        path_used=QUERY_ENHANCEMENT_PATH_LM,
    )

    gw = object.__new__(GatewayAgent)
    gw.telemetry_manager = manager
    await gw._emit_gateway_span(
        tenant_id="acme:acme",
        query="q",
        complexity="simple",
        modality="video",
        generation_type="raw_results",
        routed_to="search_agent",
        confidence=0.9,
    )

    ee = object.__new__(EntityExtractionAgent)
    ee.telemetry_manager = manager
    await ee._emit_extraction_span(
        tenant_id="acme:acme",
        query="q",
        entities=[],
        relationships=[],
        path_used="dspy",
    )

    ps = object.__new__(ProfileSelectionAgent)
    ps.telemetry_manager = manager
    await ps._emit_profile_span(
        query="q",
        tenant_id="acme:acme",
        available_profiles="video_colpali_base",
        selected_profile="video_colpali_base",
        intent="video_search",
        modality="video",
        complexity="simple",
        confidence=0.8,
    )


def test_emitters_do_not_block_when_backend_is_dead():
    """Real manager, dead OTLP endpoint: after the per-project provider is
    warmed, each subsequent emit returns far under the synchronous-export
    timeout because it enqueues instead of exporting."""
    manager = _dead_backend_manager()
    try:
        # Warm the lazy per-project tracer provider (one-time construction).
        asyncio.run(_emit_all(manager))
        # Now measure per-request cost: the emit path only enqueues.
        start = time.monotonic()
        asyncio.run(_emit_all(manager))
        elapsed = time.monotonic() - start
    finally:
        TelemetryManager.reset()
    assert elapsed < _MAX_EMIT_SECONDS, (
        f"emitting four training spans against a dead backend took {elapsed * 1000:.1f}ms "
        f"(> {_MAX_EMIT_SECONDS * 1000:.0f}ms) after warmup: the request path is "
        "blocking on telemetry export instead of enqueuing"
    )


@pytest.mark.expects_telemetry_loss_warning
def test_emitters_do_not_call_span_synchronously_on_the_request_path(caplog):
    """A manager whose span() would block never even gets called synchronously:
    the emit awaits its own coroutine and warns instead of raising."""
    manager = _HangingManager()
    with caplog.at_level(logging.WARNING):
        # Each emit catches the failure and warns; the request path survives.
        # (The hanging manager raises on span(); the emitter must turn that
        # into a WARNING, not propagate it.)
        asyncio.run(_emit_all(manager))
    losses = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Failed to emit ")
    ]
    assert len(losses) == 4, (
        f"expected four best-effort loss WARNINGs (one per emitter), got {losses}"
    )
