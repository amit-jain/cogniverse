"""Dispatcher process-span coverage with a real in-memory exporter.

These tests pin the request-trace seam the dispatcher now owns:
`AgentBase.process()` must be the root span for orchestrator and search
dispatches, gateway must stay on the direct `_process_impl()` hop, and a
session id must flow through the provider's `session_context()` onto the
whole request trace when one is present.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openinference.instrumentation import TracerProvider as OITracerProvider
from openinference.instrumentation import using_session
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_agents.gateway_agent import GatewayAgent as RealGatewayAgent
from cogniverse_agents.gateway_agent import GatewayInput
from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent as RealOrchestratorAgent,
)
from cogniverse_agents.orchestrator_agent import OrchestratorInput
from cogniverse_agents.search_agent import SearchAgent as RealSearchAgent
from cogniverse_agents.search_agent import SearchInput
from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher, _GatewayAgentEntry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    TelemetryManager.reset()
    yield
    TelemetryManager.reset()


class _SessionProvider:
    @contextmanager
    def session_context(self, session_id: str):
        with using_session(session_id):
            yield


def _recording_telemetry_manager():
    config = TelemetryConfig(
        enabled=True,
        otlp_endpoint="http://localhost:4317",
        service_name="dispatcher-span-test",
        environment="test",
        level=TelemetryLevel.VERBOSE,
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)

    provider = OITracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("dispatcher-span-test")

    manager._get_tracer_for_project = lambda tenant_id, project_name=None: tracer  # type: ignore[assignment]
    manager.get_provider = lambda tenant_id, project_name=None: _SessionProvider()  # type: ignore[assignment]

    return manager, exporter


class SearchAgent(RealSearchAgent):
    def __init__(self):
        self.telemetry_manager = None
        self._input_rails = None
        self._output_rails = None

    async def _process_impl(self, input_data: SearchInput):
        with self.telemetry_manager.span(
            "search.execute", tenant_id=input_data.tenant_id
        ) as span:
            span.set_attribute("operation", "search")
            span.set_attribute("query", input_data.query)
        return SimpleNamespace(
            results=[{"id": "doc-1", "score": 1.0}],
            enhanced_query=None,
            profile="video_colpali_smol500_mv_frame",
            search_mode="single_profile",
        )


class OrchestratorAgent(RealOrchestratorAgent):
    def __init__(self):
        self.telemetry_manager = None
        self._input_rails = None
        self._output_rails = None

    async def _process_impl(self, input_data: OrchestratorInput):
        with self.telemetry_manager.span(
            "cogniverse.orchestration", tenant_id=input_data.tenant_id
        ) as span:
            span.set_attribute("operation", "orchestration")
            span.set_attribute("query", input_data.query)
        return SimpleNamespace(model_dump=lambda: {"workflow_id": "wf-123"})


class GatewayAgent(RealGatewayAgent):
    def __init__(self):
        self.telemetry_manager = None
        self._input_rails = None
        self._output_rails = None

    async def _process_impl(self, input_data: GatewayInput):
        with self.telemetry_manager.span(
            "cogniverse.gateway", tenant_id=input_data.tenant_id
        ) as span:
            span.set_attribute("operation", "gateway")
            span.set_attribute("query", input_data.query)
        return SimpleNamespace(
            complexity="simple",
            modality="video",
            generation_type="raw_results",
            routed_to="search_agent",
            confidence=0.9,
            fast_path_confidence_threshold=0.4,
            gliner_threshold=0.3,
            detected_modalities=["video"],
            reasoning="stub",
        )


def _dispatcher():
    registry = MagicMock()
    config_manager = MagicMock()
    schema_loader = MagicMock()
    dispatcher = AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    return dispatcher


def _span(spans, name):
    for span in spans:
        if span.name == name:
            return span
    raise AssertionError(f"missing span {name!r}: {[span.name for span in spans]}")


def _assert_parent_child(parent, child):
    assert parent.parent is None
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_orchestrator_dispatch_roots_process_and_children(monkeypatch):
    manager, exporter = _recording_telemetry_manager()
    dispatcher = _dispatcher()
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: manager,
    )

    agent = OrchestratorAgent()
    agent.set_telemetry_manager(manager)

    async def _build_orchestrator(tenant_id):
        return agent

    dispatcher._get_or_build_orchestrator = _build_orchestrator
    dispatcher._apply_artefact_overlay = lambda *a, **k: None

    await dispatcher._execute_orchestration_task(
        "compare videos and summarize them",
        {"tenant_id": "acme:prod"},
        "acme:prod",
    )

    spans = exporter.get_finished_spans()
    root = _span(spans, "OrchestratorAgent.process")
    child = _span(spans, "cogniverse.orchestration")
    _assert_parent_child(root, child)


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_search_dispatch_roots_process_and_children(monkeypatch):
    manager, exporter = _recording_telemetry_manager()
    dispatcher = _dispatcher()
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: manager,
    )

    fake_config = MagicMock()
    fake_config.get = lambda key, default=None: (
        "video_colpali_smol500_mv_frame" if key == "active_video_profile" else default
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda **kwargs: fake_config,
    )

    agent = SearchAgent()
    agent.set_telemetry_manager(manager)

    def _build_search_agent(profile):
        return agent

    dispatcher._get_search_agent = _build_search_agent
    dispatcher._apply_artefact_overlay = lambda *a, **k: None
    dispatcher.consult_egress_policy = lambda *a, **k: None
    dispatcher._verify_search_egress = lambda *a, **k: None

    await dispatcher._execute_search_task("find cats", "acme:prod", top_k=5)

    spans = exporter.get_finished_spans()
    root = _span(spans, "SearchAgent.process")
    child = _span(spans, "search.execute")
    _assert_parent_child(root, child)


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_gateway_dispatch_stays_off_process_wrapper(monkeypatch):
    manager, exporter = _recording_telemetry_manager()
    dispatcher = _dispatcher()
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: manager,
    )

    agent = GatewayAgent()
    agent.set_telemetry_manager(manager)
    dispatcher._gateway_agents.set(
        "acme:prod", _GatewayAgentEntry(agent=agent, loaded_at=0.0)
    )
    dispatcher.consult_egress_policy = lambda *a, **k: None
    dispatcher._verify_routing_egress = lambda *a, **k: None
    dispatcher._get_rail_chains = lambda tenant_id: None

    async def _execute_downstream_agent(*a, **k):
        return {"status": "success", "agent": "search_agent"}

    dispatcher._execute_downstream_agent = _execute_downstream_agent

    await dispatcher._execute_gateway_task(
        "find cats",
        {"tenant_id": "acme:prod"},
        "acme:prod",
    )

    names = [span.name for span in exporter.get_finished_spans()]
    assert "GatewayAgent.process" not in names
    assert "cogniverse.gateway" in names


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_orchestrator_session_id_reaches_root_and_child(monkeypatch):
    manager, exporter = _recording_telemetry_manager()
    dispatcher = _dispatcher()
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: manager,
    )

    agent = OrchestratorAgent()
    agent.set_telemetry_manager(manager)

    async def _build_orchestrator(tenant_id):
        return agent

    dispatcher._get_or_build_orchestrator = _build_orchestrator
    dispatcher._apply_artefact_overlay = lambda *a, **k: None

    await dispatcher._execute_orchestration_task(
        "compare videos and summarize them",
        {"tenant_id": "acme:prod", "session_id": "session-123"},
        "acme:prod",
    )

    spans = exporter.get_finished_spans()
    root = _span(spans, "OrchestratorAgent.process")
    child = _span(spans, "cogniverse.orchestration")
    _assert_parent_child(root, child)
    # The session context is active before AgentBase starts the root span, so
    # both spans inherit the same session id from OpenInference.
    assert dict(root.attributes)["session.id"] == "session-123"
    assert dict(child.attributes)["session.id"] == "session-123"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_orchestrator_without_session_id_sets_none_nowhere(monkeypatch):
    manager, exporter = _recording_telemetry_manager()
    dispatcher = _dispatcher()
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: manager,
    )

    agent = OrchestratorAgent()
    agent.set_telemetry_manager(manager)

    async def _build_orchestrator(tenant_id):
        return agent

    dispatcher._get_or_build_orchestrator = _build_orchestrator
    dispatcher._apply_artefact_overlay = lambda *a, **k: None

    await dispatcher._execute_orchestration_task(
        "compare videos and summarize them",
        {"tenant_id": "acme:prod"},
        "acme:prod",
    )

    spans = exporter.get_finished_spans()
    assert "session.id" not in {
        key for span in spans for key in dict(span.attributes).keys()
    }
