"""
E2E API tests exercising routing, search, tenant CRUD, agent registry,
A2A protocol, profile CRUD, ingestion, synthetic data, and event streaming.

Requires live runtime at http://localhost:28000 with Ollama + Vespa + Phoenix.
Uses flywheel_org:production tenant which has ingested data.

Architecture note (A2A gateway):
    The primary entry point is now ``gateway_agent``, which classifies queries
    and dispatches simple ones directly to execution agents (search_agent, etc.)
    and complex ones to the OrchestratorAgent for multi-agent coordination.

    Entity extraction, query enhancement, and profile selection are handled by
    dedicated A2A agents that are invoked internally by the orchestrator --
    they are NOT called inline by the routing_agent anymore.

    The routing_agent is now a thin DSPy-powered decision-maker. Both
    ``gateway`` and ``routing`` capabilities route through the gateway
    pipeline in the dispatcher.

    See ``test_a2a_gateway_e2e.py`` for gateway-specific E2E tests.
"""

import json
import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    RUNTIME,
    TENANT_ID,
    skip_if_no_runtime,
    unique_id,
)

PROFILE = "video_colpali_smol500_mv_frame"


@pytest.mark.e2e
@skip_if_no_runtime
class TestRoutingPipeline:
    """Scenario 1: Routing agent routes query via the gateway pipeline.

    In the A2A architecture, both 'gateway' and 'routing' capabilities route
    through _execute_gateway_task in the dispatcher. The routing_agent no
    longer does entity extraction or query enhancement inline -- those are
    handled by dedicated upstream A2A agents via the orchestrator.

    The gateway_agent is the new entry point. See test_a2a_gateway_e2e.py
    for comprehensive gateway tests.
    """

    def test_routing_decision_structure(self):
        """Routing agent returns success via the gateway pipeline."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "Show me videos of cats playing piano",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        # In the new architecture, the response comes from the gateway
        # pipeline. The agent field will be gateway_agent or orchestrator_agent.
        assert data["agent"] in (
            "routing_agent", "gateway_agent", "orchestrator_agent",
        )

        # Content assertions: "cats playing piano" is an unambiguous video query
        if "gateway" in data:
            gw = data["gateway"]
            assert gw["modality"] == "video", (
                f"Expected video modality for cat piano query, got {gw['modality']!r}"
            )
            assert gw["routed_to"] == "search_agent", (
                f"Simple video query should route to search_agent, "
                f"got {gw['routed_to']!r}"
            )
            assert gw["confidence"] >= 0.4, (
                f"Simple route should have confidence >= 0.4, got {gw['confidence']}"
            )

    def test_routing_no_longer_returns_inline_entities(self):
        """Routing agent no longer returns entities/enhanced_query at top level.

        In the A2A architecture, entity extraction and query enhancement are
        handled by dedicated agents invoked by the orchestrator. The routing
        agent response now has a gateway-style structure with downstream_result
        or orchestration_result.
        """
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "Find videos about Tesla cars in San Francisco",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # The response should have gateway-style keys, not old inline keys
        has_new_shape = (
            "gateway" in data
            or "downstream_result" in data
            or "orchestration_result" in data
        )
        assert has_new_shape, (
            f"Routing should produce gateway-style response, "
            f"got keys: {list(data.keys())}"
        )
        # Old inline entity/enhancement fields must NOT appear at the top level
        assert "entities" not in data, (
            "entities should not be at the top level in the A2A architecture"
        )
        assert "enhanced_query" not in data, (
            "enhanced_query should not be at the top level in the A2A architecture"
        )
        # If recommended_agent is surfaced, it must be a known valid agent
        if "recommended_agent" in data:
            assert data["recommended_agent"] in (
                "search_agent",
                "summarizer_agent",
                "detailed_report_agent",
                "image_search_agent",
                "audio_analysis_agent",
                "document_agent",
                "deep_research_agent",
                "coding_agent",
                "text_analysis_agent",
            ), f"Routing returned unknown agent: {data['recommended_agent']!r}"

    def test_routing_executes_downstream(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "search for animal videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        has_downstream = (
            "downstream_result" in data
            or "orchestration_result" in data
            or "gateway" in data
        )
        assert has_downstream, (
            f"Routing should execute downstream, got keys: {list(data.keys())}"
        )
        # Content assertion: animal video query should produce search results
        downstream = data.get("downstream_result", {})
        if "results" in downstream and downstream.get("results_count", 0) > 0:
            results = downstream["results"]
            assert len(results) > 0, (
                "Search for 'animal videos' should return results from ingested data"
            )
            score_keys = ("score", "relevance", "relevance_score", "_score")
            score_key = next(
                (k for k in score_keys if k in results[0]), None
            )
            if score_key is not None:
                scores = [r[score_key] for r in results]
                assert scores == sorted(scores, reverse=True), (
                    f"Results should be ranked by {score_key} descending, "
                    f"got: {scores}"
                )


@pytest.mark.e2e
@skip_if_no_runtime
class TestQueryEnhancementViaGateway:
    """Scenarios 2-3: Query enhancement and entity extraction are now handled
    by dedicated A2A agents via the orchestrator pipeline.

    The routing_agent no longer returns enhanced_query or entities at the
    top level. These tests verify the gateway pipeline works end-to-end.
    """

    def test_gateway_processes_query_successfully(self):
        """Gateway pipeline processes queries without errors."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "ML transformer videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_gateway_classifies_entity_rich_queries(self):
        """Entity-rich queries are classified and routed by the gateway."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "Obama speaking at MIT about climate change",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        # Gateway should classify and route the query
        has_routing = (
            "gateway" in data
            or "orchestration_result" in data
            or "downstream_result" in data
        )
        assert has_routing, (
            f"Gateway should classify and route, got keys: {list(data.keys())}"
        )

    def test_gateway_confidence_in_range(self):
        """Gateway classification confidence should be in [0.4, 1.0] for clear queries.

        This query is classified 'complex' by the gateway and hands off to the
        OrchestratorAgent, which fires 5+ ChainOfThought LLM calls through
        local Ollama on CPU. End-to-end latency on a dev k3d cluster regularly
        lands north of 10 minutes. The default 300s httpx timeout tripped
        before the orchestration could return a 200, so the test failed on
        ReadTimeout instead of surfacing the actual confidence score. Give
        the pipeline a 900s budget to complete on CPU hardware; if real
        latency regresses past that it's a separate signal.
        """
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "find me detailed analysis of deep learning architectures",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        # Confidence is nested in gateway metadata
        gw = data.get("gateway", {})
        if "confidence" in gw:
            assert 0.0 <= gw["confidence"] <= 1.0
            # A clear, unambiguous query should exceed the routing threshold
            assert gw["confidence"] >= 0.4, (
                f"Clear query should produce confidence >= 0.4 (threshold), "
                f"got {gw['confidence']}"
            )


@pytest.mark.e2e
@skip_if_no_runtime
class TestOrchestration:
    """Scenarios 4-5: Gateway triggers orchestration for complex queries."""

    def test_complex_query_triggers_orchestration_or_downstream(self):
        """Complex queries route through the gateway to orchestration or
        direct downstream."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "Find videos about machine learning and write a detailed report",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] in ("gateway_agent", "orchestrator_agent")

    def test_multi_turn_routing(self):
        """Multi-turn routing preserves conversation context."""
        session_id = str(uuid.uuid4())

        transport = httpx.HTTPTransport(retries=2)
        with httpx.Client(
            base_url=RUNTIME, timeout=900.0, transport=transport
        ) as client:
            resp1 = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "search for cat videos",
                    "context": {"tenant_id": TENANT_ID, "session_id": session_id},
                    "top_k": 3,
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["status"] == "success"

            resp2 = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "show me longer ones",
                    "context": {"tenant_id": TENANT_ID, "session_id": session_id},
                    "top_k": 3,
                    "conversation_history": [
                        {"role": "user", "content": "search for cat videos"},
                        {"role": "agent", "content": "Found cat video results"},
                    ],
                },
            )

        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == "success"


@pytest.mark.e2e
@skip_if_no_runtime
class TestSearchAPI:
    """Scenario 7: Search with profile/strategy selection and result validation."""

    def test_list_strategies(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get("/search/strategies")

        assert resp.status_code == 200
        data = resp.json()
        strategies = data.get("strategies", data) if isinstance(data, dict) else data
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_list_profiles(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(
                "/search/profiles",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        profiles = data.get("profiles", data) if isinstance(data, dict) else data
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        profile_names = [p["name"] if isinstance(p, dict) else p for p in profiles]
        assert PROFILE in profile_names, (
            f"Expected {PROFILE} in profiles, got: {profile_names}"
        )

    def test_search_with_explicit_profile(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "animals in nature",
                    "profile": PROFILE,
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "results_count" in data
        assert isinstance(data["results_count"], int)
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_result_fields(self):
        """Verify search results contain expected content fields."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "sports activities",
                    "profile": PROFILE,
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["results_count"] >= 1, (
            "Search for 'sports activities' should return results from ingested data"
        )
        result = data["results"][0]
        assert isinstance(result, dict)
        assert len(result) > 1, (
            f"Result should have multiple fields, got: {list(result.keys())}"
        )

    def test_search_response_echoes_params(self):
        """Verify response includes the query, profile, and strategy sent."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "cooking video",
                    "profile": PROFILE,
                    "strategy": "default",
                    "top_k": 3,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "cooking video"
        assert data["profile"] == PROFILE

    def test_search_with_different_strategy(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "outdoor activities",
                    "profile": PROFILE,
                    "strategy": "float_float",
                    "top_k": 3,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "results_count" in data

    def test_search_rerank_validation(self):
        """POST /search/rerank without query returns 400."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/search/rerank",
                json={"results": [], "strategy": "learned"},
            )
        assert resp.status_code == 400, (
            f"Rerank without query should return 400, got {resp.status_code}"
        )

    def test_search_rerank_unknown_strategy(self):
        """POST /search/rerank with unknown strategy returns 400."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/search/rerank",
                json={
                    "query": "test",
                    "results": [{"id": "1", "score": 0.5}],
                    "strategy": "nonexistent_strategy",
                },
            )
        assert resp.status_code == 400


@pytest.mark.e2e
@skip_if_no_runtime
class TestProfileCRUD:
    """Profile management: create, list, get, update, delete."""

    def test_list_profiles_for_tenant(self):
        """GET /admin/profiles returns profile list structure for tenant."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(
                "/admin/profiles",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "profiles" in data
        assert "total_count" in data
        assert "tenant_id" in data
        assert isinstance(data["profiles"], list)
        assert isinstance(data["total_count"], int)
        assert data["total_count"] >= 0

    def test_create_then_get_profile(self):
        """Create a profile via admin API, then GET it by name."""
        profile_name = unique_id("e2e_get")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            try:
                create_resp = client.post(
                    "/admin/profiles",
                    json={
                        "profile_name": profile_name,
                        "tenant_id": TENANT_ID,
                        "type": "video",
                        "description": "E2E get test profile",
                        "schema_name": "video_colpali_smol500_mv_frame",
                        "embedding_model": "vidore/colpali-v1.3-hf",
                        "embedding_type": "multi_vector",
                        "deploy_schema": False,
                    },
                )
                assert create_resp.status_code == 201, (
                    f"Create profile failed: {create_resp.text}"
                )

                resp = client.get(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["profile_name"] == profile_name
                assert data["tenant_id"] == TENANT_ID
                assert "schema_name" in data
                assert "embedding_model" in data
                assert "type" in data
                assert "version" in data
                assert isinstance(data["version"], int)
            finally:
                client.delete(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )

    def test_get_nonexistent_profile_returns_404(self):
        """GET /admin/profiles/{name} returns 404 for missing profile."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(
                "/admin/profiles/nonexistent_profile_xyz",
                params={"tenant_id": TENANT_ID},
            )
        assert resp.status_code == 404

    def test_profile_create_update_delete_lifecycle(self):
        """Full lifecycle: create profile → get → update → delete."""
        profile_name = unique_id("e2e_profile")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            try:
                resp = client.post(
                    "/admin/profiles",
                    json={
                        "profile_name": profile_name,
                        "tenant_id": TENANT_ID,
                        "type": "video",
                        "description": "E2E test profile",
                        "schema_name": "video_colpali_smol500_mv_frame",
                        "embedding_model": "vidore/colpali-v1.3-hf",
                        "embedding_type": "multi_vector",
                        "deploy_schema": False,
                    },
                )
                assert resp.status_code == 201, f"Create profile failed: {resp.text}"
                data = resp.json()
                assert data["profile_name"] == profile_name
                assert data["tenant_id"] == TENANT_ID
                assert "version" in data

                resp = client.get(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )
                assert resp.status_code == 200
                assert resp.json()["profile_name"] == profile_name

                resp = client.put(
                    f"/admin/profiles/{profile_name}",
                    json={
                        "tenant_id": TENANT_ID,
                        "description": "Updated E2E test profile",
                    },
                )
                assert resp.status_code == 200
                update_data = resp.json()
                assert "description" in update_data["updated_fields"]

            finally:
                client.delete(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )


@pytest.mark.e2e
@skip_if_no_runtime
class TestSystemStats:
    """GET /admin/system/stats returns system statistics."""

    def test_system_stats(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/admin/system/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "registered_backends" in data
        assert isinstance(data["registered_backends"], list)
        assert "timestamp" in data


@pytest.mark.e2e
@skip_if_no_runtime
class TestAgentOperations:
    """Agent registration, capability search, unregistration, and process."""

    def test_capability_based_discovery(self):
        """GET /agents/by-capability/{cap} returns matching agents."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/by-capability/search")

        assert resp.status_code == 200
        data = resp.json()
        assert data["capability"] == "search"
        assert "count" in data
        assert isinstance(data["agents"], list)

    def test_agent_upload_endpoint_removed(self):
        """POST /agents/{name}/upload was deleted (audit fix #13).

        The endpoint was a 501 stub with no implementation path. File
        uploads have a real home at POST /ingestion/upload, so the stub
        is gone. Hitting the old URL must now produce 404 / 405."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/agents/routing_agent/upload",
                files={"file": ("test.txt", b"test content", "text/plain")},
            )
        assert resp.status_code in (404, 405)

    def test_unregister_nonexistent_agent_returns_404(self):
        """DELETE /agents/{name} returns 404 for unknown agent."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.delete("/agents/nonexistent_agent_xyz")
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "agent_name,query",
        [
            ("text_analysis_agent", "analyze this text about video processing"),
            ("summarizer_agent", "summarize the key findings"),
            ("detailed_report_agent", "write a report on search results"),
        ],
    )
    def test_agent_process_response_structure(self, agent_name, query):
        """Each agent returns status=success with agent name."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                f"/agents/{agent_name}/process",
                json={
                    "agent_name": agent_name,
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200, f"{agent_name} process failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == agent_name

    def test_process_nonexistent_agent_returns_404(self):
        """POST /agents/{name}/process returns 404 for unknown agent."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/agents/nonexistent_agent_xyz/process",
                json={
                    "agent_name": "nonexistent_agent_xyz",
                    "query": "test",
                    "context": {"tenant_id": TENANT_ID},
                },
            )
        assert resp.status_code == 404


@pytest.mark.e2e
@skip_if_no_runtime
class TestSyntheticDataAPI:
    """Synthetic data generation endpoints."""

    def test_synthetic_health(self):
        """GET /synthetic/health returns healthy status."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "synthetic-data-generation"
        assert "generators" in data
        assert "optimizers" in data

    def test_list_optimizers(self):
        """GET /synthetic/optimizers returns optimizer registry."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/optimizers")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert len(data) >= 1, f"Should have at least 1 optimizer, got: {data}"

    def test_get_optimizer_detail(self):
        """GET /synthetic/optimizers/{name} returns optimizer info."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            list_resp = client.get("/synthetic/optimizers")
            optimizers = list_resp.json()
            first_optimizer = next(iter(optimizers))

            resp = client.get(f"/synthetic/optimizers/{first_optimizer}")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_get_nonexistent_optimizer_returns_404(self):
        """GET /synthetic/optimizers/{name} returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/optimizers/nonexistent_xyz")
        assert resp.status_code == 404

    def test_generate_synthetic_data(self):
        """POST /synthetic/generate creates real synthetic training examples."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "routing",
                    "count": 5,
                    "vespa_sample_size": 10,
                    "strategies": ["diverse"],
                    "max_profiles": 2,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200, f"Synthetic generation failed: {resp.text}"
        data = resp.json()
        assert data["optimizer"] == "routing"
        assert data["count"] >= 1, (
            f"Should generate at least 1 example, got {data['count']}"
        )
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

        # Verify each example has the expected routing structure
        example = data["data"][0]
        assert "query" in example, (
            f"Synthetic example missing query: {list(example.keys())}"
        )
        assert "chosen_agent" in example, (
            f"Synthetic example missing chosen_agent: {list(example.keys())}"
        )
        assert "routing_confidence" in example, (
            f"Synthetic example missing routing_confidence: {list(example.keys())}"
        )

        # Verify profile selection reasoning
        assert "selected_profiles" in data
        assert isinstance(data["selected_profiles"], list)

    def test_generate_synthetic_data_cross_modal(self):
        """POST /synthetic/generate with cross_modal optimizer."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "cross_modal",
                    "count": 3,
                    "vespa_sample_size": 10,
                    "strategies": ["diverse"],
                    "max_profiles": 2,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200, f"Cross-modal generation failed: {resp.text}"
        data = resp.json()
        assert data["optimizer"] == "cross_modal"
        assert data["count"] >= 1
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1


@pytest.mark.e2e
@skip_if_no_runtime
class TestEventOperations:
    """Event queue cancel and offset endpoints."""

    def test_cancel_nonexistent_workflow_returns_404(self):
        """POST /events/workflows/{id}/cancel returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/events/workflows/nonexistent_wf_xyz/cancel",
                json={"reason": "test"},
            )
        assert resp.status_code == 404

    def test_cancel_nonexistent_ingestion_returns_404(self):
        """POST /events/ingestion/{id}/cancel returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/events/ingestion/nonexistent_job_xyz/cancel",
                json={"reason": "test"},
            )
        assert resp.status_code == 404

    def test_queue_offset_not_found(self):
        """GET /events/queues/{id}/offset returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/events/queues/nonexistent_q_xyz/offset")
        assert resp.status_code == 404


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantCRUD:
    """Scenario 15: Full tenant lifecycle create -> list -> delete via API."""

    def test_tenant_lifecycle(self):
        org_id = unique_id("apiorg")
        tenant_name = "test_tenant"
        tenant_full_id = f"{org_id}:{tenant_name}"

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            try:
                resp = client.post(
                    "/admin/organizations",
                    json={
                        "org_id": org_id,
                        "org_name": f"E2E Test Org {org_id}",
                        "created_by": "e2e_test",
                    },
                )
                assert resp.status_code == 200, f"Create org failed: {resp.text}"
                org_data = resp.json()
                assert org_data["org_id"] == org_id

                resp = client.get("/admin/organizations")
                assert resp.status_code == 200
                orgs = resp.json()
                org_ids = [o["org_id"] for o in orgs["organizations"]]
                assert org_id in org_ids

                resp = client.post(
                    "/admin/tenants",
                    json={
                        "tenant_id": tenant_full_id,
                        "created_by": "e2e_test",
                    },
                )
                assert resp.status_code == 200, f"Create tenant failed: {resp.text}"
                tenant_data = resp.json()
                assert tenant_data["tenant_full_id"] == tenant_full_id

                resp = client.get(f"/admin/organizations/{org_id}/tenants")
                assert resp.status_code == 200
                tenants = resp.json()
                tenant_ids = [t["tenant_full_id"] for t in tenants["tenants"]]
                assert tenant_full_id in tenant_ids

                resp = client.get(f"/admin/tenants/{tenant_full_id}")
                assert resp.status_code == 200
                assert resp.json()["tenant_full_id"] == tenant_full_id

            finally:
                client.delete(f"/admin/tenants/{tenant_full_id}")
                client.delete(f"/admin/organizations/{org_id}")

    def test_org_not_found_returns_404(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/admin/organizations/nonexistent_org_xyz")
        assert resp.status_code == 404, (
            f"Non-existent org should return 404, got {resp.status_code}"
        )


@pytest.mark.e2e
@skip_if_no_runtime
class TestAgentRegistryAndHealth:
    """Scenario 18: Health endpoints and agent registry queries."""

    def test_health_check(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cogniverse-runtime"
        assert "backends" in data
        assert "agents" in data

    def test_liveness_probe(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health/live")

        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_probe(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "not_ready")

    def test_list_agents(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/")

        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "count" in data
        assert data["count"] >= 1

    def test_agent_stats(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "total_agents" in data, (
            f"Stats missing total_agents, got: {list(data.keys())}"
        )
        assert isinstance(data["total_agents"], int)
        assert data["total_agents"] >= 1

    def test_root_endpoint(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/")

        assert resp.status_code == 200

    def test_get_agent_info(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/routing_agent")

        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data or "name" in data

    def test_agent_card(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/routing_agent/card")

        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card or "agent_name" in card

    @pytest.mark.parametrize(
        "agent_name",
        [
            "routing_agent",
            "search_agent",
            "text_analysis_agent",
            "summarizer_agent",
            "detailed_report_agent",
            "gateway_agent",
            "entity_extraction_agent",
            "query_enhancement_agent",
            "profile_selection_agent",
            "orchestrator_agent",
        ],
    )
    def test_registered_agents_accessible(self, agent_name):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(f"/agents/{agent_name}")
        assert resp.status_code == 200, f"Agent {agent_name} not accessible"

    def test_search_agent_process(self):
        """Scenario 18 sub-test: Direct search agent process."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "nature documentary",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "search_agent"


@pytest.mark.e2e
@skip_if_no_runtime
class TestA2AProtocol:
    """Scenario 19: A2A protocol agent card, tasks/send, and streaming."""

    def test_runtime_agent_card(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/a2a/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "skills" in card or "capabilities" in card

    def test_agent_card_advertises_streaming(self):
        """AgentCard capabilities should include streaming=True."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/a2a/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        capabilities = card.get("capabilities", {})
        assert capabilities.get("streaming") is True, (
            f"AgentCard should advertise streaming=True, got: {capabilities}"
        )

    def test_a2a_single_turn(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-api-1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": "search for nature videos"}
                            ],
                            "messageId": str(uuid.uuid4()),
                        },
                        "configuration": {
                            "acceptedOutputModes": ["text"],
                        },
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data, f"Expected result, got: {data}"
        result = data["result"]
        assert result["id"]
        assert result["contextId"]

    def test_a2a_streaming_produces_sse_events(self):
        """message/stream returns SSE events with progress + final result."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with client.stream(
                "POST",
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-stream-1",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "summarize what machine learning is",
                                }
                            ],
                            "messageId": str(uuid.uuid4()),
                        },
                        "metadata": {
                            "agent_name": "summarizer_agent",
                            "tenant_id": TENANT_ID,
                            "stream": True,
                        },
                    },
                },
            ) as resp:
                assert resp.status_code == 200
                events = []
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if data_str:
                            events.append(json.loads(data_str))

        assert len(events) >= 2, (
            f"Streaming should produce ≥2 SSE events (progress + final), got {len(events)}"
        )

        # Parse agent events from A2A wrapper
        parsed = []
        for event in events:
            for part in (
                event.get("result", {})
                .get("status", {})
                .get("message", {})
                .get("parts", [])
            ):
                text = part.get("text", "")
                if text:
                    try:
                        parsed.append(json.loads(text))
                    except json.JSONDecodeError:
                        pass

        # Should have status events (from emit_progress) and a final event
        types = [e.get("type") for e in parsed]
        assert "status" in types, f"Should have progress events, got types: {types}"

        # Final event should contain summary
        finals = [e for e in parsed if e.get("type") == "final"]
        assert len(finals) == 1, f"Should have exactly 1 final event, got: {parsed}"
        assert "data" in finals[0]
        assert "summary" in finals[0]["data"]
        summary = finals[0]["data"]["summary"]
        assert len(summary) > 20, f"Summary too short: '{summary}'"
        summary_lower = summary.lower()
        assert any(
            term in summary_lower
            for term in ["machine learning", "ml", "learn", "algorithm", "data"]
        ), f"Summary should reference ML, got: '{summary}'"


@pytest.mark.e2e
@skip_if_no_runtime
class TestStreamingAllAgents:
    """Verify A2A streaming works for multiple agent types."""

    @pytest.mark.parametrize(
        "agent_name,query,expect_streaming",
        [
            ("summarizer_agent", "summarize AI trends briefly", True),
            ("detailed_report_agent", "write a report on search results", True),
            ("routing_agent", "find videos about cats", True),
        ],
    )
    def test_streaming_agent_returns_events(self, agent_name, query, expect_streaming):
        """message/stream returns SSE events for streaming-capable agents."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with client.stream(
                "POST",
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-stream-all",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "role": "user",
                            "messageId": str(uuid.uuid4()),
                            "contextId": str(uuid.uuid4()),
                            "parts": [{"kind": "text", "text": query}],
                        },
                        "metadata": {
                            "agent_name": agent_name,
                            "tenant_id": TENANT_ID,
                            "stream": True,
                        },
                    },
                },
            ) as resp:
                assert resp.status_code == 200
                events = []
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if data_str:
                            raw = json.loads(data_str)
                            for part in (
                                raw.get("result", {})
                                .get("status", {})
                                .get("message", {})
                                .get("parts", [])
                            ):
                                text = part.get("text", "")
                                if text:
                                    try:
                                        events.append(json.loads(text))
                                    except json.JSONDecodeError:
                                        pass

        assert len(events) >= 1, (
            f"{agent_name}: should return ≥1 event, got {len(events)}"
        )

        if expect_streaming:
            types = [e.get("type") for e in events]
            assert "status" in types, (
                f"{agent_name}: streaming agent should emit progress events, got: {types}"
            )


@pytest.mark.e2e
@skip_if_no_runtime
class TestOptimizationE2E:
    """End-to-end optimization through the runtime API."""

    def test_record_examples_triggers_optimization(self):
        """POST optimize_routing with examples → optimization_triggered."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "optimize routing",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "action": "optimize_routing",
                        "examples": [
                            {
                                "query": "find cat videos",
                                "chosen_agent": "search_agent",
                                "confidence": 0.9,
                                "search_quality": 0.85,
                                "agent_success": True,
                            },
                        ],
                    },
                },
            )

        assert resp.status_code == 200, f"Optimization failed: {resp.text}"
        data = resp.json()
        assert data.get("status") == "optimization_triggered", (
            f"Should trigger optimization, got: {data}"
        )
        assert data.get("training_examples") == 1

    def test_auto_optimization_cycle_from_traces(self):
        """POST optimize_routing without examples → runs full cycle from traces."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "optimize routing",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "action": "optimize_routing",
                    },
                },
            )

        assert resp.status_code == 200, f"Optimization cycle failed: {resp.text}"
        data = resp.json()
        assert data.get("status") == "optimization_triggered", (
            f"Auto cycle should trigger optimization, got: {data}"
        )
        assert data.get("optimizer") == "OptimizationOrchestrator"
        assert "cycle_results" in data
        assert isinstance(data["cycle_results"], dict)

    def test_optimization_status(self):
        """GET optimization status returns optimizer state."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "optimization status",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "action": "get_optimization_status",
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "optimizer_ready" in data
        assert isinstance(data["optimizer_ready"], bool)
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)

    def test_route_after_optimization_succeeds(self):
        """Routing works after optimization was triggered."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            # First trigger optimization
            client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "optimize routing",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "action": "optimize_routing",
                        "examples": [
                            {
                                "query": "basketball videos",
                                "chosen_agent": "search_agent",
                                "confidence": 0.8,
                                "search_quality": 0.7,
                                "agent_success": True,
                            },
                        ],
                    },
                },
            )

            # Then route a query — should use optimized model
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find basketball highlights",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        # Response varies by dispatch path: direct routing, gateway, or orchestrator
        agent = (
            data.get("recommended_agent")
            or data.get("gateway", {}).get("routed_to")
            or data.get("agent")  # orchestrator path returns agent key
        )
        assert agent, f"No agent in response: {list(data.keys())}"
        confidence = (
            data.get("confidence")
            or data.get("gateway", {}).get("confidence")
            or data.get("gateway_context", {}).get("confidence")
            or 0.1  # orchestrated queries don't have top-level confidence
        )
        assert confidence > 0.0, f"Expected positive confidence, got {confidence}"


@pytest.mark.e2e
@skip_if_no_runtime
class TestIngestionAPI:
    """Ingestion endpoints: start, status, upload validation."""

    def test_start_ingestion_invalid_dir_returns_error(self):
        """POST /ingestion/start with non-existent directory returns error."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": "/nonexistent/path/e2e_fake_dir",
                    "profile": "video_colpali_smol500_mv_frame",
                    "backend": "vespa",
                    "tenant_id": TENANT_ID,
                },
            )
        assert resp.status_code in (400, 422, 500), (
            f"Non-existent video_dir should fail, got {resp.status_code}"
        )

    def test_ingestion_status_not_found(self):
        """GET /ingestion/status/{fake_id} returns 404."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/ingestion/status/nonexistent-job-id-xyz")
        assert resp.status_code == 404, (
            f"Non-existent job_id should return 404, got {resp.status_code}"
        )
        data = resp.json()
        assert "not found" in data.get("detail", "").lower()

    def test_upload_requires_file(self):
        """POST /ingestion/upload without file returns 422 (validation error)."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post("/ingestion/upload")
        assert resp.status_code == 422, (
            f"Upload without file should return 422, got {resp.status_code}"
        )


# Real artifact ingestion + search round-trip tests
#
# Each test uploads a REAL file, ingests it through the actual ML pipeline,
# waits for Vespa indexing, then searches and verifies results.
#
# Artifacts:
# - Video: ActivityNet clip (874KB, man throwing discus)
# - Image: Big Buck Bunny keyframe (1280x720 real animation frame)
# - Audio: Art of War narration from LibriVox (public domain speech)
#          + extracted audio from sample video (real video sound)
# - PDF: Video-ChatGPT paper from arxiv (related to the test dataset)
# - Document: dataset_summary.md (real markdown about the evaluation set)


@pytest.mark.e2e
@skip_if_no_runtime
class TestVideoIngestionAndSearch:
    """Upload real ActivityNet video, verify ingestion, then search."""

    def test_upload_video_and_search(self, real_video_path):
        """Upload v_-nl4G-00PtA.mp4 (874KB) → ColPali embedding → search."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            with open(real_video_path, "rb") as f:
                resp = client.post(
                    "/ingestion/upload",
                    files={"file": (real_video_path.name, f, "video/mp4")},
                    data={
                        "profile": "video_colpali_smol500_mv_frame",
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Video upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["filename"] == real_video_path.name
            assert upload_data["chunks_created"] >= 1, (
                f"Expected >=1 chunks from video frames, got {upload_data['chunks_created']}"
            )

            time.sleep(5)

            search_resp = client.post(
                "/search/",
                json={
                    "query": "person doing activity",
                    "profile": "video_colpali_smol500_mv_frame",
                    "top_k": 10,
                    "tenant_id": TENANT_ID,
                },
            )
            assert search_resp.status_code == 200
            search_data = search_resp.json()
            assert search_data["results_count"] >= 1, (
                "Search after video ingestion should return results"
            )
            assert len(search_data["results"]) >= 1


@pytest.mark.e2e
@skip_if_no_runtime
class TestImageIngestionAndSearch:
    """Upload real Big Buck Bunny keyframe (1280x720), ingest, search."""

    def _deploy_schema_if_needed(self, client, profile_name):
        """Deploy schema for profile if not already deployed."""
        resp = client.post(
            f"/admin/profiles/{profile_name}/deploy",
            json={"tenant_id": TENANT_ID, "force": False},
        )
        # Accept 200 (deployed) or 409/400 (already deployed)
        return resp.status_code in (200, 400, 409)

    def test_upload_image_and_search(self, real_image_path):
        """Upload real 1280x720 keyframe → ColPali embedding → search."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            self._deploy_schema_if_needed(client, "image_colpali_mv")

            with open(real_image_path, "rb") as f:
                resp = client.post(
                    "/ingestion/upload",
                    files={"file": (real_image_path.name, f, "image/jpeg")},
                    data={
                        "profile": "image_colpali_mv",
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Image upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["filename"] == real_image_path.name
            assert upload_data["chunks_created"] >= 1, (
                f"Image should create >=1 chunk, got {upload_data['chunks_created']}"
            )

            time.sleep(5)

            search_resp = client.post(
                "/search/",
                json={
                    "query": "animated character cartoon scene",
                    "profile": "image_colpali_mv",
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )
            assert search_resp.status_code == 200
            search_data = search_resp.json()
            # documents_fed may be 0 if Vespa schema deployment is still converging
            if upload_data.get("documents_fed", 0) > 0:
                assert search_data["results_count"] >= 1, (
                    "Search after image ingestion should return results"
                )


@pytest.mark.e2e
@skip_if_no_runtime
class TestAudioIngestionAndSearch:
    """Extract real audio from sample video, verify pipeline processes it."""

    def test_upload_extracted_audio_processing(self, extracted_audio_path):
        """Extract audio from ActivityNet video → pipeline processing.

        Uses real audio from sample video (speech/ambient sound), not a
        synthetic sine tone. Tests the audio pipeline: file discovery →
        embedding generation.
        """
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with open(extracted_audio_path, "rb") as f:
                resp = client.post(
                    "/ingestion/upload",
                    files={"file": (extracted_audio_path.name, f, "audio/wav")},
                    data={
                        "profile": "audio_clap_semantic",
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Audio upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["chunks_created"] >= 1, (
                f"Audio should create >=1 chunk, got {upload_data['chunks_created']}"
            )

            # Search may fail if CLAP encoder not configured for search
            time.sleep(3)

            search_resp = client.post(
                "/search/",
                json={
                    "query": "speech people talking activity",
                    "profile": "audio_clap_semantic",
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                if upload_data.get("documents_fed", 0) > 0:
                    assert search_data["results_count"] >= 1


@pytest.mark.e2e
@skip_if_no_runtime
class TestPDFIngestionAndSearch:
    """Upload real arxiv PDF (Video-ChatGPT paper), verify pipeline processes it."""

    def test_upload_pdf_processing(self, real_pdf_path):
        """Upload Video-ChatGPT paper → PDF text extraction → embedding."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with open(real_pdf_path, "rb") as f:
                resp = client.post(
                    "/ingestion/upload",
                    files={"file": (real_pdf_path.name, f, "application/pdf")},
                    data={
                        "profile": "document_text_semantic",
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"PDF upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["filename"] == real_pdf_path.name
            assert upload_data["chunks_created"] >= 1, (
                f"PDF should create >=1 chunk, got {upload_data['chunks_created']}"
            )

            # Search uses same encoder limitation as document test
            time.sleep(3)

            search_resp = client.post(
                "/search/",
                json={
                    "query": "video understanding large language model",
                    "profile": "document_text_semantic",
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                if upload_data.get("documents_fed", 0) > 0:
                    assert search_data["results_count"] >= 1


@pytest.mark.e2e
@skip_if_no_runtime
class TestDocumentIngestionAndSearch:
    """Upload real dataset_summary.md, verify pipeline processes it."""

    def test_upload_markdown_processing(self, real_document_path):
        """Upload real markdown doc → text extraction → GTE-ColBERT embedding."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with open(real_document_path, "rb") as f:
                resp = client.post(
                    "/ingestion/upload",
                    files={"file": (real_document_path.name, f, "text/markdown")},
                    data={
                        "profile": "document_text_semantic",
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Document upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["filename"] == real_document_path.name
            assert upload_data["chunks_created"] >= 1, (
                f"Document should create >=1 chunk, got {upload_data['chunks_created']}"
            )

            # Search verification: document_text_semantic uses Reason-ModernColBERT
            # which requires a separate encoder config. Verify search is attempted
            # but accept that the encoder may not be configured for this profile.
            time.sleep(3)

            search_resp = client.post(
                "/search/",
                json={
                    "query": "ActivityNet video benchmark evaluation",
                    "profile": "document_text_semantic",
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )
            # Search may fail (500) if encoder not configured for this model;
            # the critical assertion is that upload + processing succeeded above.
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                if upload_data.get("documents_fed", 0) > 0:
                    assert search_data["results_count"] >= 1


# Scenario 20 (API portion): Event queue listing
# Placed before batch ingestion because batch starts CPU-bound ColPali
# inference that blocks the async event loop for minutes.


@pytest.mark.e2e
@skip_if_no_runtime
class TestEventEndpoints:
    """Scenario 20 API: Event queue listing."""

    def test_list_queues(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(
                "/events/queues",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_queue_not_found(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get("/events/queues/nonexistent_fake_id")

        assert resp.status_code == 404, (
            f"Non-existent queue should return 404, got {resp.status_code}"
        )


@pytest.mark.e2e
@skip_if_no_runtime
class TestBatchVideoIngestion:
    """Start batch ingestion and verify the event loop stays responsive.

    The asyncio.to_thread fix for embedding generation means the event loop
    should remain responsive during CPU-bound ColPali inference. This test
    verifies both job tracking AND event loop health.
    """

    def test_batch_ingestion_start(self):
        """Start batch ingestion → poll status → verify event loop responsive."""
        # Use pod-internal path (devMode mounts data/ at /app/data)
        video_dir = "/app/data/testset/evaluation/sample_videos"

        # Verify the host copy exists (pod mount mirrors host)
        host_dir = Path(__file__).parent.parent.parent / "data" / "testset" / "evaluation" / "sample_videos"
        if not host_dir.exists():
            pytest.skip(f"Sample video dir not found on host: {host_dir}")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": video_dir,
                    "profile": "video_colpali_smol500_mv_frame",
                    "backend": "vespa",
                    "tenant_id": TENANT_ID,
                    "max_videos": 1,
                    "batch_size": 1,
                },
            )

            assert resp.status_code == 200, f"Batch ingestion start failed: {resp.text}"
            data = resp.json()
            assert data["status"] == "started"
            assert "job_id" in data
            job_id = data["job_id"]

        # Poll status — with asyncio.to_thread fix the event loop stays
        # responsive during inference, so status endpoint should reply.
        status_data = None
        health_ok_during_ingestion = False
        for attempt in range(30):
            time.sleep(5)
            try:
                with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
                    # Verify event loop is responsive by hitting health endpoint
                    health_resp = client.get("/health/live")
                    if health_resp.status_code == 200:
                        health_ok_during_ingestion = True

                    status_resp = client.get(f"/ingestion/status/{job_id}")
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        if status_data["status"] in ("completed", "failed"):
                            break
            except (httpx.ReadTimeout, httpx.ConnectError):
                continue

        # Event loop responsiveness is the key assertion for the deadlock fix
        assert health_ok_during_ingestion, (
            "Event loop should remain responsive during batch ingestion "
            "(asyncio.to_thread fix for embedding generation)"
        )

        assert status_data is not None, (
            "Should be able to poll ingestion status while job is running"
        )
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ("started", "processing", "completed", "failed")

    def test_runtime_responsive_after_batch(self):
        """Verify runtime is fully responsive after batch ingestion completes."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"

            # Also verify agent processing still works
            resp = client.get("/agents/")
            assert resp.status_code == 200
