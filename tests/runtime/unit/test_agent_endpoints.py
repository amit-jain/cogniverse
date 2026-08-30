"""
Unit tests for agent router endpoints.

Tests the gateway→orchestration handoff via AgentDispatcher, and HTTP-level
round-trip tests for the annotation queue endpoints.
"""

import time
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_runtime.agent_dispatcher import AgentDispatcher, _GatewayAgentEntry
from cogniverse_runtime.routers import agents as agents_router


@pytest.fixture
def mock_telemetry_manager():
    """Create mock telemetry manager."""
    manager = MagicMock()

    @contextmanager
    def fake_span(*args, **kwargs):
        yield MagicMock()

    manager.span = fake_span
    return manager


@pytest.fixture
def dispatcher():
    """Create an AgentDispatcher with mock dependencies."""
    registry = MagicMock()
    config_manager = MagicMock()
    from cogniverse_foundation.config.unified_config import BackendConfig

    # Production dispatch reads backend_url + backend_port from
    # get_system_config() to build URLs; bare MagicMocks would yield
    # a MagicMock port that urllib.parse can't cast to int.
    sys_cfg = MagicMock()
    sys_cfg.backend_url = "http://localhost"
    sys_cfg.backend_port = 8080
    sys_cfg.search_backend = "vespa"
    # _resolve_gliner_url reads inference_service_urls; a bare MagicMock
    # would return another MagicMock from .get("gliner"), which then fails
    # GatewayDeps pydantic validation (Optional[str] rejects a MagicMock).
    sys_cfg.inference_service_urls = None
    config_manager.get_system_config.return_value = sys_cfg
    config_manager.get_backend_config.side_effect = (
        lambda tenant_id, service="backend": BackendConfig(tenant_id=tenant_id)
    )
    schema_loader = MagicMock()
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


def _make_gateway_output(
    *,
    complexity="simple",
    modality="video",
    generation_type="raw_results",
    routed_to="search_agent",
    confidence=0.9,
    fast_path_confidence_threshold=0.4,
    gliner_threshold=0.3,
):
    """Build a mock GatewayOutput for tests."""
    output = Mock()
    output.complexity = complexity
    output.modality = modality
    output.generation_type = generation_type
    output.routed_to = routed_to
    output.confidence = confidence
    output.fast_path_confidence_threshold = fast_path_confidence_threshold
    output.gliner_threshold = gliner_threshold
    output.reasoning = "test reasoning"
    return output


@pytest.mark.unit
class TestGatewayOrchestrationHandoff:
    """Test that AgentDispatcher routes through GatewayAgent for triage."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_simple_query_routes_to_downstream(self, dispatcher):
        """Simple query via GatewayAgent dispatches directly to execution agent."""
        gateway_output = _make_gateway_output(
            complexity="simple",
            routed_to="search_agent",
            modality="video",
        )

        # Registry: gateway_agent has ["gateway"],
        # search_agent has ["search"] for downstream
        gateway_ep = MagicMock()
        gateway_ep.capabilities = ["gateway"]
        search_ep = MagicMock()
        search_ep.capabilities = ["search"]

        def get_agent_by_name(name):
            if name == "gateway_agent":
                return gateway_ep
            if name == "search_agent":
                return search_ep
            return None

        dispatcher._registry.get_agent.side_effect = get_agent_by_name

        mock_downstream = {
            "status": "success",
            "agent": "search_agent",
            "message": "Found 3 results",
            "results_count": 3,
            "results": [],
            "profile": "test_profile",
        }

        with (
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent._process_impl",
                new_callable=AsyncMock,
                return_value=gateway_output,
            ),
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent.__init__",
                return_value=None,
            ),
            patch.object(
                dispatcher,
                "_execute_downstream_agent",
                new_callable=AsyncMock,
                return_value=mock_downstream,
            ),
        ):
            result = await dispatcher.dispatch(
                agent_name="gateway_agent",
                query="find videos of cats",
                context={"tenant_id": "test_tenant"},
            )

        assert result["status"] == "success"
        assert result["agent"] == "gateway_agent"
        assert result["gateway"] == {
            "complexity": "simple",
            "modality": "video",
            "generation_type": "raw_results",
            "routed_to": "search_agent",
            "confidence": 0.9,
            "fast_path_confidence_threshold": 0.4,
            "gliner_threshold": 0.3,
        }
        assert result["gateway"]["confidence"] == 0.9
        # Assert on the SHAPE the dispatcher must build, not on the literal
        # dict the test injected — the latter would pass even if the
        # dispatcher silently dropped fields.
        ds = result["downstream_result"]
        assert ds["status"] == "success"
        assert ds["agent"] == "search_agent"
        assert "results_count" in ds and isinstance(ds["results_count"], int)
        assert "profile" in ds

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_gateway_simple_persists_downstream_answer_not_breadcrumb(
        self, dispatcher
    ):
        """A gateway 'simple' route persists the downstream agent's user-facing
        answer as the assistant turn — the value the response path renders —
        not the routing breadcrumb. The breadcrumb, stored as the prior reply,
        was fed to the anaphora rewriter on the next turn and shown by the
        messaging display in place of the real answer.
        """
        gateway_output = _make_gateway_output(
            complexity="simple", routed_to="search_agent", modality="video"
        )
        # Force a simple classification without building the real GatewayAgent:
        # seed the per-tenant gateway cache with a fake whose _process_impl
        # returns the simple triage. The real _get_or_build_gateway_agent
        # returns it on the cache hit.
        fake_gateway = SimpleNamespace(
            _process_impl=AsyncMock(return_value=gateway_output)
        )
        dispatcher._gateway_agents.set(
            "acme:acme",
            _GatewayAgentEntry(agent=fake_gateway, loaded_at=time.monotonic()),
        )

        gateway_ep = MagicMock()
        gateway_ep.capabilities = ["gateway"]
        search_ep = MagicMock()
        search_ep.capabilities = ["search"]

        def _get_agent(name):
            return {"gateway_agent": gateway_ep, "search_agent": search_ep}.get(name)

        dispatcher._registry.get_agent.side_effect = _get_agent

        # The downstream agent's answer at its execution boundary — the exact
        # shape _execute_search_task returns for three hits.
        downstream_answer = {
            "status": "success",
            "agent": "search_agent",
            "message": "Found 3 results for 'find videos of cats'",
            "results_count": 3,
            "results": [
                {"document_id": "v1"},
                {"document_id": "v2"},
                {"document_id": "v3"},
            ],
            "profile": "video_colpali_smol500_mv_frame",
            "search_mode": "hybrid",
        }
        dispatcher._execute_downstream_agent = AsyncMock(return_value=downstream_answer)

        # Capture the turns the runtime persists via the conversation-store seam.
        stored: list[tuple[str, str]] = []

        class _CaptureStore:
            def get_history(self, ctx):
                return []

            def store_turn(self, ctx, role, content):
                stored.append((role, content))

        dispatcher._conversation_store_factory = lambda tenant: _CaptureStore()

        result = await dispatcher.dispatch(
            agent_name="gateway_agent",
            query="find videos of cats",
            context={"tenant_id": "acme:acme", "context_id": "chat-1"},
        )

        # The persisted assistant turn is the answer, not the breadcrumb.
        assert ("user", "find videos of cats") in stored
        assistant_turns = [c for (r, c) in stored if r == "assistant"]
        assert assistant_turns == ["Found 3 results for 'find videos of cats'"]

        # The response the caller/display consumes carries the answer as
        # `message` and surfaces the hits at top level, gateway triage kept.
        assert result["message"] == "Found 3 results for 'find videos of cats'"
        assert not result["message"].startswith("Routed")
        assert result["agent"] == "gateway_agent"
        assert result["gateway"] == {
            "complexity": "simple",
            "modality": "video",
            "generation_type": "raw_results",
            "routed_to": "search_agent",
            "confidence": 0.9,
            "fast_path_confidence_threshold": 0.4,
            "gliner_threshold": 0.3,
        }
        assert result["results_count"] == 3
        assert result["results"] == downstream_answer["results"]
        assert result["downstream_result"] == downstream_answer

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_complex_query_routes_to_orchestrator(self, dispatcher):
        """Complex query via GatewayAgent forwards to OrchestratorAgent."""
        gateway_output = _make_gateway_output(
            complexity="complex",
            routed_to="orchestrator_agent",
            modality="both",
            confidence=0.4,
            fast_path_confidence_threshold=0.55,
            gliner_threshold=0.35,
        )

        gateway_ep = MagicMock()
        gateway_ep.capabilities = ["gateway"]
        dispatcher._registry.get_agent.return_value = gateway_ep

        mock_orch_result = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated 'find robots' via A2A pipeline",
            "orchestration_result": {"workflow_id": "wf_test"},
            "gateway_context": {
                "modality": "both",
                "generation_type": "raw_results",
                "confidence": 0.4,
            },
        }

        with (
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent._process_impl",
                new_callable=AsyncMock,
                return_value=gateway_output,
            ),
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent.__init__",
                return_value=None,
            ),
            patch.object(
                dispatcher,
                "_execute_orchestration_task",
                new_callable=AsyncMock,
                return_value=mock_orch_result,
            ),
        ):
            result = await dispatcher.dispatch(
                agent_name="gateway_agent",
                query="find robots then summarize and create report",
                context={"tenant_id": "test_tenant"},
            )

        assert result["status"] == "success"
        assert result["agent"] == "orchestrator_agent"
        # The gateway triage is stamped identically on both routing paths so a
        # caller never needs to know which branch answered.
        assert result["gateway"] == {
            "complexity": "complex",
            "modality": "both",
            "generation_type": "raw_results",
            "routed_to": "orchestrator_agent",
            "confidence": 0.4,
            "fast_path_confidence_threshold": 0.55,
            "gliner_threshold": 0.35,
        }
        assert result["gateway_context"]["modality"] == "both"
        # Strong shape assertion: the dispatcher must thread through every
        # field the orchestrator round-trip is contracted to surface.
        assert result["gateway_context"]["generation_type"] == "raw_results"
        assert 0.0 <= result["gateway_context"]["confidence"] <= 1.0
        assert "orchestration_result" in result
        assert "workflow_id" in result["orchestration_result"]

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_complex_query_carries_detected_modalities(self, dispatcher):
        """The dispatcher must keep the gateway's full modality set in the
        orchestration handoff payload."""
        gateway_output = _make_gateway_output(
            complexity="complex",
            routed_to="orchestrator_agent",
            modality="both",
            confidence=0.4,
            fast_path_confidence_threshold=0.55,
            gliner_threshold=0.35,
        )
        gateway_output.detected_modalities = ["video", "document"]

        gateway_ep = MagicMock()
        gateway_ep.capabilities = ["gateway"]
        dispatcher._registry.get_agent.return_value = gateway_ep

        captured = {}

        async def _spy_orchestration(query, context, tenant_id, gateway_context):
            captured.update(
                query=query,
                context=context,
                tenant_id=tenant_id,
                gateway_context=gateway_context,
            )
            return {
                "status": "success",
                "agent": "orchestrator_agent",
                "answer": "orchestrated",
            }

        with (
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent._process_impl",
                new_callable=AsyncMock,
                return_value=gateway_output,
            ),
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent.__init__",
                return_value=None,
            ),
        ):
            dispatcher._execute_orchestration_task = _spy_orchestration
            await dispatcher._execute_gateway_task(
                "compare videos and documents about neural networks",
                {"tenant_id": "acme:prod", "top_k": 10},
                "acme:prod",
            )

        assert captured["context"]["detected_modalities"] == [
            "video",
            "document",
        ]
        assert captured["gateway_context"] == {
            "modality": "both",
            "generation_type": "raw_results",
            "confidence": 0.4,
        }

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_routing_capability_triggers_gateway(self, dispatcher):
        """Agent with 'routing' capability also routes through gateway (backward compat)."""
        routing_ep = MagicMock()
        routing_ep.capabilities = ["routing"]
        dispatcher._registry.get_agent.return_value = routing_ep

        mock_gateway_result = {
            "status": "success",
            "agent": "gateway_agent",
            "message": "Routed 'test' to search_agent (simple)",
            "gateway": {"complexity": "simple"},
            "downstream_result": {},
        }

        with patch.object(
            dispatcher,
            "_execute_gateway_task",
            new_callable=AsyncMock,
            return_value=mock_gateway_result,
        ) as mock_gw:
            result = await dispatcher.dispatch(
                agent_name="legacy_routing_capability_agent",
                query="test",
                context={"tenant_id": "t1"},
            )

        mock_gw.assert_called_once()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_orchestration_capability_dispatches_directly(self, dispatcher):
        """Agent with 'orchestration' capability dispatches to orchestration task."""
        orch_ep = MagicMock()
        orch_ep.capabilities = ["orchestration"]
        dispatcher._registry.get_agent.return_value = orch_ep

        mock_orch_result = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated 'complex q' via A2A pipeline",
            "orchestration_result": {},
            "gateway_context": None,
        }

        with patch.object(
            dispatcher,
            "_execute_orchestration_task",
            new_callable=AsyncMock,
            return_value=mock_orch_result,
        ) as mock_orch:
            result = await dispatcher.dispatch(
                agent_name="orchestrator_agent",
                query="complex q",
                context={"tenant_id": "t1"},
            )

        mock_orch.assert_called_once()
        assert result["agent"] == "orchestrator_agent"


@pytest.mark.unit
class TestAgentDispatcherCapabilityRouting:
    """Test that dispatch routes to the correct _execute_* method by capability."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_unknown_agent_raises(self, dispatcher):
        """Unknown agent name raises ValueError."""
        dispatcher._registry.get_agent.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await dispatcher.dispatch(agent_name="nonexistent", query="test")

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_unregistered_agent_raises(self, dispatcher):
        """Agent not in AGENT_CLASSES falls through to generic dispatch and raises."""
        agent_ep = MagicMock()
        agent_ep.capabilities = ["unknown_capability"]
        dispatcher._registry.get_agent.return_value = agent_ep

        with pytest.raises(ValueError, match="no supported execution path"):
            await dispatcher.dispatch(
                agent_name="weird_agent",
                query="test",
                context={"tenant_id": "test:unit"},
            )

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_search_capability(self, dispatcher):
        """Agent with 'search' capability routes to search handler."""
        agent_ep = MagicMock()
        agent_ep.capabilities = ["search"]
        dispatcher._registry.get_agent.return_value = agent_ep

        with patch.object(
            dispatcher,
            "_execute_search_task",
            new_callable=AsyncMock,
            return_value={"status": "success", "agent": "search_agent"},
        ) as mock_search:
            result = await dispatcher.dispatch(
                agent_name="search_agent",
                query="find cats",
                context={"tenant_id": "t1"},
            )

        # require_tenant_id canonicalizes "t1" → "t1:t1" for the tenant_id
        # positional arg, while context dict is not mutated.
        mock_search.assert_called_once_with(
            "find cats",
            "t1:t1",
            10,
            conversation_history=[],
            enrichment=None,
            context={"tenant_id": "t1"},
        )
        assert result["status"] == "success"


@pytest.mark.unit
class TestOrchestrationUsesOrchestratorAgent:
    """MultiAgentOrchestrator was replaced by OrchestratorAgent. Proven by
    EXECUTING the orchestration dispatch path — it constructs OrchestratorAgent
    and initializes that agent's memory under "orchestrator_agent" — and by
    asserting the removed optimizer-lookup methods are gone. Not by grepping
    the dispatcher source (which would pass even on a stale, never-run path)."""

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_orchestration_path_wires_orchestrator_agent(
        self, dispatcher, monkeypatch
    ):
        monkeypatch.setattr(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_agents.orchestrator_agent.OrchestratorDeps",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            lambda: None,
        )

        class _StopAfterInit(Exception):
            pass

        seen = []

        def _spy(agent, name, tenant):
            seen.append((name, tenant))
            raise _StopAfterInit

        dispatcher._init_agent_memory = _spy
        with pytest.raises(_StopAfterInit):
            await dispatcher._execute_orchestration_task("q", {}, "acme:prod")
        assert seen == [("orchestrator_agent", "acme:prod")]

    @pytest.mark.ci_fast
    def test_optimizer_lookup_methods_removed(self):
        """Optimizer lookup moved to Argo — the dispatcher must not carry the
        old _get_optimizer / get_routing_statistics methods."""
        assert not hasattr(AgentDispatcher, "_get_optimizer")
        assert not hasattr(AgentDispatcher, "get_routing_statistics")


@pytest.mark.unit
class TestModalitySearchDispatchSerialization:
    """image/audio/document dispatch returns Pydantic result objects, which must
    be serialized with model_dump() — dataclasses.asdict() raises TypeError on
    them. These exercise the dispatch path with REAL result objects (a Mock
    result would serialize fine and hide the bug)."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_image_search_dispatch_serializes_results(
        self, dispatcher, monkeypatch
    ):
        from cogniverse_agents.image_search_agent import ImageResult

        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        stub = MagicMock()
        stub.search_images = AsyncMock(
            return_value=[ImageResult(image_id="img1", image_url="http://x/1.jpg")]
        )
        monkeypatch.setattr(
            "cogniverse_agents.image_search_agent.ImageSearchAgent",
            lambda *a, **k: stub,
        )

        result = await dispatcher._execute_image_search_task("cats", "acme:prod", 5)

        assert result["status"] == "success"
        assert result["results_count"] == 1
        assert result["results"][0]["image_id"] == "img1"
        assert result["results"][0]["image_url"] == "http://x/1.jpg"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_audio_search_dispatch_serializes_results(
        self, dispatcher, monkeypatch
    ):
        from cogniverse_agents.audio_analysis_agent import (
            AudioAnalysisDeps,
            AudioResult,
        )

        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        backend = MagicMock()
        backend.schema_exists = MagicMock(return_value=True)
        monkeypatch.setattr(
            "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
        )
        stub = MagicMock()
        stub.search_audio = AsyncMock(
            return_value=[AudioResult(audio_id="aud1", audio_url="http://x/1.mp3")]
        )
        captured = {}

        def build_agent(*, deps, **kwargs):
            captured["deps"] = deps
            return stub

        monkeypatch.setattr(
            "cogniverse_agents.audio_analysis_agent.AudioAnalysisAgent",
            build_agent,
        )

        result = await dispatcher._execute_audio_search_task("speech", "acme:prod", 5)

        assert backend.schema_exists.call_args_list == [
            call("audio_content", "acme:prod")
        ]
        deps = captured["deps"]
        assert isinstance(deps, AudioAnalysisDeps)
        assert deps.backend_type == "vespa"
        assert deps.config_manager is dispatcher._config_manager
        assert deps.schema_loader is dispatcher._schema_loader
        assert deps.backend_config["url"] == "http://localhost"
        assert deps.backend_config["port"] == 8080
        assert deps.backend_config["schema_name"] == "audio_content"
        assert "profile" not in deps.backend_config
        assert deps.backend_config["backend"]["type"] == "vespa"
        assert result["status"] == "success"
        assert result["results_count"] == 1
        assert result["results"][0]["audio_id"] == "aud1"
        assert result["results"][0]["audio_url"] == "http://x/1.mp3"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_audio_search_dispatch_is_empty_when_tenant_has_no_audio_schema(
        self, dispatcher, monkeypatch
    ):
        """Tenant schemas deploy on first ingest; a tenant that never ingested
        audio has no audio_content schema, and a query for it is a real
        no-content answer, not a Vespa error."""
        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        backend = MagicMock()
        backend.schema_exists = MagicMock(return_value=False)
        monkeypatch.setattr(
            "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
        )

        def _must_not_build(*a, **k):
            raise AssertionError("audio agent must not be built without a schema")

        monkeypatch.setattr(
            "cogniverse_agents.audio_analysis_agent.AudioAnalysisAgent",
            _must_not_build,
        )

        result = await dispatcher._execute_audio_search_task("speech", "acme:prod", 5)

        assert backend.schema_exists.call_args_list == [
            call("audio_content", "acme:prod")
        ]
        assert result == {
            "status": "success",
            "agent": "audio_analysis_agent",
            "message": "No audio content indexed for tenant 'acme:prod'",
            "results_count": 0,
            "results": [],
        }

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_audio_search_dispatch_raises_when_schema_lookup_fails(
        self, dispatcher, monkeypatch
    ):
        """A registry outage is not "no audio content": it must surface."""
        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        backend = MagicMock()
        backend.schema_exists = MagicMock(
            side_effect=RuntimeError("schema registry unavailable")
        )
        monkeypatch.setattr(
            "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
        )

        def _must_not_build(*a, **k):
            raise AssertionError("audio agent must not be built when lookup fails")

        monkeypatch.setattr(
            "cogniverse_agents.audio_analysis_agent.AudioAnalysisAgent",
            _must_not_build,
        )

        with pytest.raises(RuntimeError, match="^schema registry unavailable$"):
            await dispatcher._execute_audio_search_task("speech", "acme:prod", 5)

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_document_search_dispatch_serializes_results(
        self, dispatcher, monkeypatch
    ):
        from cogniverse_agents.document_agent import DocumentResult

        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        monkeypatch.setattr(dispatcher, "_init_agent_memory", lambda *a, **k: None)
        stub = MagicMock()
        stub.search_documents = AsyncMock(
            return_value=[
                DocumentResult(
                    document_id="doc1", document_url="http://x/1.pdf", title="Doc One"
                )
            ]
        )
        backend = MagicMock()
        schema_calls = []

        def schema_exists(base_schema, tenant_id):
            schema_calls.append((base_schema, tenant_id))
            return True

        backend.schema_exists = schema_exists
        monkeypatch.setattr(
            "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
        )
        captured = {}

        def build_agent(*, deps, **kwargs):
            captured["deps"] = deps
            return stub

        monkeypatch.setattr(
            "cogniverse_agents.document_agent.DocumentAgent", build_agent
        )

        result = await dispatcher._execute_document_search_task(
            "report", "acme:prod", 5
        )

        assert result["status"] == "success"
        assert result["results_count"] == 1
        assert result["results"][0]["document_id"] == "doc1"
        assert result["results"][0]["title"] == "Doc One"
        # The tenant's deployed document schemas are read once each and handed
        # to the agent, which narrows its search strategy to them.
        assert schema_calls == [
            ("document_text", "acme:prod"),
            ("document_visual", "acme:prod"),
        ]
        assert captured["deps"].deployed_document_schemas == (
            "document_text",
            "document_visual",
        )

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_document_dispatch_propagates_degraded(self, dispatcher, monkeypatch):
        """A Vespa soft-timeout raised by the agent must reach dispatch callers —
        the dispatcher must not flatten it into a success-shaped empty result."""
        from cogniverse_agents.search.vespa_query import VespaSearchDegraded

        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        monkeypatch.setattr(dispatcher, "_init_agent_memory", lambda *a, **k: None)
        stub = MagicMock()
        stub.search_documents = AsyncMock(
            side_effect=VespaSearchDegraded("Vespa query returned errors: [code 12]")
        )
        backend = MagicMock()
        backend.schema_exists = MagicMock(return_value=True)
        monkeypatch.setattr(
            "cogniverse_runtime.admin.tenant_manager.get_backend", lambda: backend
        )
        monkeypatch.setattr(
            "cogniverse_agents.document_agent.DocumentAgent", lambda *a, **k: stub
        )

        with pytest.raises(VespaSearchDegraded, match="code 12"):
            await dispatcher._execute_document_search_task("report", "acme:prod", 5)


@pytest.mark.unit
class TestProcessRouteDegradedMapping:
    """POST /agents/{name}/process maps VespaSearchDegraded to 503, not a bare 500."""

    def test_process_route_maps_degraded_to_503(self, monkeypatch):
        from cogniverse_agents.search.vespa_query import VespaSearchDegraded

        stub_dispatcher = MagicMock()
        stub_dispatcher.dispatch = AsyncMock(
            side_effect=VespaSearchDegraded("Vespa query returned errors: [code 12]")
        )
        monkeypatch.setattr(
            agents_router, "_ensure_dispatcher", lambda: stub_dispatcher
        )

        test_app = FastAPI()
        test_app.include_router(agents_router.router, prefix="/agents")
        with TestClient(test_app) as client:
            resp = client.post(
                "/agents/document_agent/process",
                json={
                    "agent_name": "document_agent",
                    "query": "quarterly report",
                    "context": {"tenant_id": "acme:prod"},
                },
            )

        assert resp.status_code == 503
        assert "code 12" in resp.json()["detail"]

    def test_app_level_handler_maps_degraded_to_503(self):
        """Routes without their own mapping (graph, wiki) get the app-level
        VespaSearchDegraded -> 503 handler registered by main.py."""
        from cogniverse_agents.search.vespa_query import VespaSearchDegraded
        from cogniverse_runtime.main import register_degraded_search_handler

        test_app = FastAPI()
        register_degraded_search_handler(test_app)

        @test_app.get("/boom")
        async def boom():
            raise VespaSearchDegraded("Vespa query returned errors: [code 12]")

        with TestClient(test_app) as client:
            resp = client.get("/boom")

        assert resp.status_code == 503
        assert "code 12" in resp.json()["detail"]


@pytest.mark.unit
class TestStreamingAgentConstruction:
    """create_streaming_agent must build (agent, typed_input) for every
    streamable capability. Previously 7 capabilities (incl. the default
    orchestrator, image/audio/document search, entity extraction, query
    enhancement, profile selection) raised "streaming not configured" and
    every such A2A stream returned only an error event. The agents stream via
    the shared A2AAgent base; these branches just needed construction."""

    @pytest.mark.parametrize(
        "agent_name,capability,agent_type,input_type",
        [
            (
                "image_search_agent",
                "image_search",
                "ImageSearchAgent",
                "ImageSearchInput",
            ),
            (
                "audio_analysis_agent",
                "audio_analysis",
                "AudioAnalysisAgent",
                "AudioSearchInput",
            ),
            (
                "document_agent",
                "document_analysis",
                "DocumentAgent",
                "DocumentSearchInput",
            ),
            (
                "entity_extraction_agent",
                "entity_extraction",
                "EntityExtractionAgent",
                "EntityExtractionInput",
            ),
            (
                "query_enhancement_agent",
                "query_enhancement",
                "QueryEnhancementAgent",
                "QueryEnhancementInput",
            ),
            (
                "profile_selection_agent",
                "profile_selection",
                "ProfileSelectionAgent",
                "ProfileSelectionInput",
            ),
            (
                "orchestrator_agent",
                "orchestration",
                "OrchestratorAgent",
                "OrchestratorInput",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_create_streaming_agent_builds_each_capability(
        self, dispatcher, monkeypatch, agent_name, capability, agent_type, input_type
    ):
        monkeypatch.setattr(dispatcher, "_get_vespa_endpoint", lambda t: "http://vespa")
        # Orchestration resolves WorkflowIntelligence from the telemetry manager;
        # None keeps construction env-independent (workflow_intelligence=None). The
        # orchestrator build also inits memory and loads its artifact (it routes
        # through the per-tenant cache like dispatch) — stub both so construction
        # stays env-independent while still building a real OrchestratorAgent.
        monkeypatch.setattr(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            lambda: None,
        )
        monkeypatch.setattr(dispatcher, "_init_agent_memory", lambda *a, **k: None)
        monkeypatch.setattr(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent._load_artifact",
            lambda self: None,
            raising=False,
        )
        entry = MagicMock()
        entry.capabilities = [capability]
        dispatcher._registry.get_agent.return_value = entry

        agent, typed_input = await dispatcher.create_streaming_agent(
            agent_name, "find robots", "acme:prod"
        )

        assert type(agent).__name__ == agent_type
        assert type(typed_input).__name__ == input_type
        if capability == "query_enhancement":
            # Interactive callers carry no sampled source; the input is built
            # with an empty source_text and grounding applies only when a
            # caller supplies one.
            assert typed_input.source_text == ""
            _, sourced_input = await dispatcher.create_streaming_agent(
                agent_name,
                "find robots",
                "acme:prod",
                context={"source_text": "Robots assemble cars on a factory floor."},
            )
            assert sourced_input.source_text == (
                "Robots assemble cars on a factory floor."
            )


# ── Annotation Queue HTTP endpoints ──────────────────────────────────────


def _make_annotation_request(
    span_id: str = "span-http-1",
    priority: AnnotationPriority = AnnotationPriority.MEDIUM,
) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=span_id,
        timestamp=datetime.now(),
        query="http test query",
        chosen_agent="search_agent",
        routing_confidence=0.5,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=priority,
        reason="http test",
        context={},
    )


@pytest.fixture
def annotation_client():
    """
    TestClient with agents router mounted and a fresh AnnotationQueue injected.

    Overrides the module-level _annotation_queue singleton so each test
    gets an isolated queue — no cross-test state leakage.
    """
    test_app = FastAPI()
    test_app.include_router(agents_router.router, prefix="/agents")

    fresh_queue = AnnotationQueue()
    # Patch the module-level singleton directly for the duration of the test
    original = agents_router._annotation_queue
    agents_router._annotation_queue = fresh_queue
    try:
        with TestClient(test_app) as client:
            yield client, fresh_queue
    finally:
        agents_router._annotation_queue = original


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAnnotationQueueEndpoints:
    """Round-trip HTTP tests for the annotation queue API."""

    def test_get_empty_queue(self, annotation_client):
        """GET /agents/annotations/queue on empty queue returns zero statistics."""
        client, _ = annotation_client
        resp = client.get("/agents/annotations/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["statistics"]["total"] == 0
        assert data["pending"] == []
        assert data["assigned"] == []
        assert data["expired"] == []

    def test_get_queue_shows_pending_items(self, annotation_client):
        """GET /agents/annotations/queue reflects items enqueued directly in queue."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-a"))
        queue.enqueue(_make_annotation_request("span-b", AnnotationPriority.HIGH))

        resp = client.get("/agents/annotations/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["statistics"]["total"] == 2
        assert data["statistics"]["by_status"]["pending"] == 2
        # HIGH priority item must appear first in the sorted pending list
        assert data["pending"][0]["span_id"] == "span-b"
        assert data["pending"][1]["span_id"] == "span-a"

    def test_assign_endpoint_round_trip(self, annotation_client):
        """POST /agents/annotations/queue/{span_id}/assign transitions PENDING→ASSIGNED."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-assign"))

        resp = client.post(
            "/agents/annotations/queue/span-assign/assign",
            json={"reviewer": "alice", "sla_hours": 8},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "assigned"
        ann = data["annotation"]
        assert ann["span_id"] == "span-assign"
        assert ann["status"] == "assigned"
        assert ann["assigned_to"] == "alice"
        assert ann["assigned_at"] is not None
        assert ann["sla_deadline"] is not None

        # Verify queue state is actually updated
        assert queue.get("span-assign").status == AnnotationStatus.ASSIGNED

    def test_assign_missing_span_returns_404(self, annotation_client):
        """POST assign on unknown span_id returns 404."""
        client, _ = annotation_client
        resp = client.post(
            "/agents/annotations/queue/nonexistent/assign",
            json={"reviewer": "bob"},
        )
        assert resp.status_code == 404

    def test_assign_already_assigned_returns_400(self, annotation_client):
        """POST assign on already-ASSIGNED span returns 400."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-dup"))
        queue.assign("span-dup", reviewer="alice")

        resp = client.post(
            "/agents/annotations/queue/span-dup/assign",
            json={"reviewer": "bob"},
        )
        assert resp.status_code == 400

    def test_complete_endpoint_round_trip(self, annotation_client):
        """POST complete transitions ASSIGNED→COMPLETED and persists label."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-complete"))
        queue.assign("span-complete", reviewer="alice")

        resp = client.post(
            "/agents/annotations/queue/span-complete/complete",
            json={"label": "correct_routing"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        ann = data["annotation"]
        assert ann["status"] == "completed"
        assert ann["completed_at"] is not None

        # Verify queue state is actually updated
        assert queue.get("span-complete").status == AnnotationStatus.COMPLETED

    def test_complete_missing_span_returns_404(self, annotation_client):
        """POST complete on unknown span_id returns 404."""
        client, _ = annotation_client
        resp = client.post(
            "/agents/annotations/queue/ghost/complete",
            json={},
        )
        assert resp.status_code == 404

    def test_complete_already_completed_returns_400(self, annotation_client):
        """POST complete on already-COMPLETED span returns 400."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-done"))
        queue.complete("span-done")

        resp = client.post(
            "/agents/annotations/queue/span-done/complete",
            json={},
        )
        assert resp.status_code == 400

    def test_full_lifecycle_via_http(self, annotation_client):
        """Full PENDING→ASSIGNED→COMPLETED lifecycle exercised through HTTP."""
        client, queue = annotation_client
        queue.enqueue(
            _make_annotation_request("span-lifecycle", AnnotationPriority.HIGH)
        )

        # Step 1: Verify appears in queue as PENDING
        resp = client.get("/agents/annotations/queue")
        assert resp.status_code == 200
        assert len(resp.json()["pending"]) == 1

        # Step 2: Assign via HTTP
        resp = client.post(
            "/agents/annotations/queue/span-lifecycle/assign",
            json={"reviewer": "reviewer1"},
        )
        assert resp.status_code == 200
        assert resp.json()["annotation"]["status"] == "assigned"

        # Step 3: Verify moved to assigned in GET response
        resp = client.get("/agents/annotations/queue")
        data = resp.json()
        assert len(data["pending"]) == 0
        assert len(data["assigned"]) == 1

        # Step 4: Complete via HTTP
        resp = client.post(
            "/agents/annotations/queue/span-lifecycle/complete",
            json={"label": "correct"},
        )
        assert resp.status_code == 200
        assert resp.json()["annotation"]["status"] == "completed"

        # Step 5: Verify no longer in pending/assigned
        resp = client.get("/agents/annotations/queue")
        data = resp.json()
        assert len(data["pending"]) == 0
        assert len(data["assigned"]) == 0
        assert data["statistics"]["total"] == 1
        assert data["statistics"]["by_status"]["completed"] == 1


class _StubAnnotationStorage:
    """Captures persistence calls the complete endpoint makes."""

    instances: list = []

    def __init__(self, tenant_id, agent_type="routing"):
        self.tenant_id = tenant_id
        self.agent_type = agent_type
        self.stored: list = []
        _StubAnnotationStorage.instances.append(self)

    async def store_human_annotation(
        self, span_id, label, reasoning, suggested_agent=None, annotator_id="human"
    ):
        self.stored.append((span_id, label, reasoning, annotator_id))
        return True


class _FailingAnnotationStorage(_StubAnnotationStorage):
    async def store_human_annotation(self, *args, **kwargs):
        raise RuntimeError("telemetry backend down")


class TestAnnotationEnqueueAndPersist:
    """The queue has an ingress endpoint and completion persists durably.

    Before this, nothing could enqueue over HTTP (the queue was always empty
    in production) and a completed label lived only in process memory — a
    restart lost every human annotation.
    """

    def _payload(self, span_id, tenant_id="acme:acme", agent_type="routing"):
        return {
            "span_id": span_id,
            "timestamp": "2026-07-17T10:00:00+00:00",
            "query": "find robot videos",
            "chosen_agent": "video_search",
            "routing_confidence": 0.4,
            "outcome": "ambiguous",
            "priority": "medium",
            "reason": "low confidence",
            "context": {},
            "agent_type": agent_type,
            "tenant_id": tenant_id,
        }

    def test_enqueue_batch_dedupes_by_span_id(self, annotation_client):
        client, queue = annotation_client
        resp = client.post(
            "/agents/annotations/queue/enqueue",
            json={
                "requests": [
                    self._payload("span-e1"),
                    self._payload("span-e2"),
                    self._payload("span-e1"),
                ]
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enqueued"] == 2
        assert data["skipped"] == 1
        assert queue.get("span-e1") is not None
        assert queue.get("span-e2") is not None
        assert queue.get("span-e1").tenant_id == "acme:acme"
        assert queue.get("span-e1").agent_type == "routing"

    def test_complete_persists_label_durably(self, annotation_client, monkeypatch):
        client, queue = annotation_client
        _StubAnnotationStorage.instances = []
        monkeypatch.setattr(
            "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
            _StubAnnotationStorage,
        )

        client.post(
            "/agents/annotations/queue/enqueue",
            json={"requests": [self._payload("span-p1", agent_type="summary")]},
        )
        resp = client.post(
            "/agents/annotations/queue/span-p1/complete",
            json={"label": "correct", "reasoning": "matches the meeting notes"},
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["persisted"] is True
        assert data["annotation"]["label"] == "correct"
        assert data["annotation"]["status"] == "completed"

        storage = _StubAnnotationStorage.instances[-1]
        assert storage.tenant_id == "acme:acme"
        assert storage.agent_type == "summary"
        assert len(storage.stored) == 1
        span_id, label, reasoning, annotator = storage.stored[0]
        assert span_id == "span-p1"
        assert label.value == "correct"
        assert reasoning == "matches the meeting notes"

    def test_complete_without_tenant_completes_but_flags_unpersisted(
        self, annotation_client
    ):
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-nt"))

        resp = client.post(
            "/agents/annotations/queue/span-nt/complete",
            json={"label": "correct_routing"},
        )
        assert resp.status_code == 200
        assert resp.json()["persisted"] is False
        assert queue.get("span-nt").status == AnnotationStatus.COMPLETED

    def test_complete_persist_failure_is_502_and_item_retryable(
        self, annotation_client, monkeypatch
    ):
        client, queue = annotation_client
        monkeypatch.setattr(
            "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
            _FailingAnnotationStorage,
        )
        client.post(
            "/agents/annotations/queue/enqueue",
            json={"requests": [self._payload("span-f1")]},
        )

        resp = client.post(
            "/agents/annotations/queue/span-f1/complete",
            json={"label": "correct"},
        )
        assert resp.status_code == 502
        # The label was NOT silently dropped: the item stays open for retry.
        assert queue.get("span-f1").status == AnnotationStatus.PENDING

    def test_complete_unknown_label_is_400(self, annotation_client):
        client, queue = annotation_client
        client.post(
            "/agents/annotations/queue/enqueue",
            json={"requests": [self._payload("span-b1")]},
        )
        resp = client.post(
            "/agents/annotations/queue/span-b1/complete",
            json={"label": "bogus_label"},
        )
        assert resp.status_code == 400
        assert queue.get("span-b1").status == AnnotationStatus.PENDING


@pytest.mark.unit
class TestGetAgentCard:
    """GET /agents/{name}/card builds the A2A card from the registry entry."""

    @pytest.fixture
    def card_client(self):
        from cogniverse_core.common.agent_models import AgentEndpoint

        entry = AgentEndpoint(
            name="video_search_agent",
            url="http://agents.svc:9001",
            capabilities=["video_search", "summarization"],
            health_endpoint="/healthz",
            process_endpoint="/tasks/process",
            health_status="healthy",
        )
        registry = MagicMock(name="agent_registry")
        registry.get_agent.side_effect = lambda name: (
            entry if name == "video_search_agent" else None
        )

        saved_registry = agents_router._agent_registry
        agents_router.set_agent_registry(registry)
        test_app = FastAPI()
        test_app.include_router(agents_router.router, prefix="/agents")
        try:
            with TestClient(test_app) as client:
                yield client, registry
        finally:
            agents_router._agent_registry = saved_registry
            agents_router._dispatcher = None

    def test_card_body_maps_registry_entry_exactly(self, card_client):
        client, registry = card_client
        resp = client.get("/agents/video_search_agent/card")
        assert resp.status_code == 200
        assert resp.json() == {
            "name": "video_search_agent",
            "url": "http://agents.svc:9001",
            "version": "1.0",
            "capabilities": ["video_search", "summarization"],
            "endpoints": {
                "health": "/healthz",
                "process": "/tasks/process",
                "info": "/agents/video_search_agent",
            },
        }
        registry.get_agent.assert_called_once_with("video_search_agent")

    def test_unknown_agent_card_returns_404(self, card_client):
        client, _ = card_client
        resp = client.get("/agents/ghost_agent/card")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Agent 'ghost_agent' not found"}


@pytest.mark.unit
class TestAudioTextSearchBackendContract:
    """Audio text search uses the backend even when clap_embed is unset."""

    def test_audio_dispatch_succeeds_without_clap_embed_for_text_search(
        self, dispatcher, monkeypatch
    ):
        import importlib.util as _importlib_util

        real_find_spec = _importlib_util.find_spec

        def _no_torch(name, *args, **kwargs):
            if name == "torch":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(_importlib_util, "find_spec", _no_torch)
        search_backend = MagicMock()
        search_backend.search = MagicMock(return_value=[])
        monkeypatch.setattr(
            "cogniverse_agents.audio_analysis_agent.AudioAnalysisAgent._get_backend",
            lambda self: search_backend,
        )

        # The tenant has an audio schema; the text-search path should not
        # require clap_embed or in-process CLAP.
        tenant_backend = MagicMock()
        tenant_backend.schema_exists = MagicMock(return_value=True)
        monkeypatch.setattr(
            "cogniverse_runtime.admin.tenant_manager.get_backend",
            lambda: tenant_backend,
        )

        # The deployed runtime carried no clap_embed entry at all.
        assert (
            dispatcher._config_manager.get_system_config().inference_service_urls or {}
        ).get("clap_embed") is None

        import asyncio as _asyncio

        result = _asyncio.run(
            dispatcher._execute_audio_search_task(
                "listen to podcasts about deep learning run 4", "acme:prod", 3
            )
        )

        assert result == {
            "status": "success",
            "agent": "audio_analysis_agent",
            "message": "Found 0 audio results for 'listen to podcasts about deep learning run 4'",
            "results_count": 0,
            "results": [],
        }
        search_backend.search.assert_called_once_with(
            {
                "query": "listen to podcasts about deep learning run 4",
                "type": "audio",
                "strategy": "phased_semantic",
                "tenant_id": "acme:prod",
                "top_k": 3,
            }
        )

    def test_process_route_maps_unconfigured_service_to_503(self, monkeypatch):
        """The route names the missing service instead of a bare 500."""
        from cogniverse_foundation.config.inference_service import (
            InferenceServiceUnavailableError,
        )

        stub_dispatcher = MagicMock()
        stub_dispatcher.dispatch = AsyncMock(
            side_effect=InferenceServiceUnavailableError(
                "clap_embed",
                "clap_embed inference service is not configured and its "
                "in-process backend is unavailable in this image (no module "
                "named 'torch').",
                module="torch",
            )
        )
        monkeypatch.setattr(
            agents_router, "_ensure_dispatcher", lambda: stub_dispatcher
        )

        test_app = FastAPI()
        test_app.include_router(agents_router.router, prefix="/agents")
        with TestClient(test_app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "listen to podcasts about deep learning run 4",
                    "context": {"tenant_id": "acme:prod"},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "clap_embed" in detail
        assert "torch" in detail

    def test_process_route_maps_unreachable_pooling_sidecar_to_503(self, monkeypatch):
        """A configured sidecar that died mid-request surfaces as 503, not 500."""
        from cogniverse_foundation.config.inference_service import (
            InferenceServiceUnavailableError,
        )

        message = (
            "remote ColBERT pooling sidecar unreachable for model "
            "'lightonai/LateOn' at http://cogniverse-colbert-pylate:8000"
        )
        stub_dispatcher = MagicMock()
        stub_dispatcher.dispatch = AsyncMock(
            side_effect=InferenceServiceUnavailableError("colbert_pooling", message)
        )
        monkeypatch.setattr(
            agents_router, "_ensure_dispatcher", lambda: stub_dispatcher
        )

        test_app = FastAPI()
        test_app.include_router(agents_router.router, prefix="/agents")
        with TestClient(test_app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "find PDF documents about Python run 5",
                    "context": {"tenant_id": "acme:prod"},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 503
        assert resp.json()["detail"] == message

    def test_process_route_500_names_stage_without_leaking_detail(self, monkeypatch):
        """An unexpected failure returns a JSON body naming agent + error type.

        The raw exception text may carry a backend URL with credentials, so
        it must not reach the client; the request id ties the response to the
        server-side traceback.
        """
        stub_dispatcher = MagicMock()
        stub_dispatcher.dispatch = AsyncMock(
            side_effect=RuntimeError(
                "connect to http://admin:sekrit@vespa:8080 refused"
            )
        )
        monkeypatch.setattr(
            agents_router, "_ensure_dispatcher", lambda: stub_dispatcher
        )

        test_app = FastAPI()
        test_app.include_router(agents_router.router, prefix="/agents")
        with TestClient(test_app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "listen to podcasts about deep learning run 4",
                    "context": {"tenant_id": "acme:prod"},
                    "session_id": "sess-42",
                },
            )

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "gateway_agent" in detail
        assert "RuntimeError" in detail
        assert "sess-42" in detail
        assert "sekrit" not in resp.text
        assert "Traceback" not in resp.text
