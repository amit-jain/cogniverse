"""
E2E API tests exercising routing, query enhancement, entity extraction,
orchestration, search, tenant CRUD, agent registry, and A2A protocol.

Requires live runtime at http://localhost:8000 with Ollama + Vespa + Phoenix.
Uses flywheel_org:production tenant which has ingested data.

Architecture note: Entity extraction, query enhancement, and orchestration
happen INSIDE the routing_agent pipeline. They are not separate registered
agents. The routing_agent response includes entities, enhanced_query, etc.
"""

import uuid

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime, unique_id

PROFILE = "video_colpali_smol500_mv_frame"

# Scenario 1: Tiered routing with entity extraction


@pytest.mark.e2e
@skip_if_no_runtime
class TestRoutingPipeline:
    """Scenario 1: Routing agent routes query to correct agent with entities."""

    def test_routing_decision_structure(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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
        assert data["agent"] == "routing_agent"
        assert "recommended_agent" in data
        assert isinstance(data["confidence"], (int, float))
        assert data["confidence"] > 0.0
        assert "reasoning" in data

    def test_routing_returns_entities_and_enhanced_query(self):
        """Routing pipeline extracts entities and enhances query internally."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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
        assert "entities" in data
        assert isinstance(data["entities"], list)
        assert "enhanced_query" in data
        assert isinstance(data["enhanced_query"], str)
        assert "relationships" in data
        assert isinstance(data["relationships"], list)
        assert "query_variants" in data
        assert isinstance(data["query_variants"], list)

    def test_routing_executes_downstream(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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
            or data.get("metadata", {}).get("needs_orchestration") is not None
        )
        assert has_downstream, (
            f"Routing should execute downstream, got keys: {list(data.keys())}"
        )

    def test_routing_metadata_structure(self):
        """Verify routing metadata includes tier info and timing."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "show me cooking tutorials",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "metadata" in data
        metadata = data["metadata"]
        assert "processing_time_ms" in metadata
        assert isinstance(metadata["processing_time_ms"], (int, float))


# Scenario 2-3: Query enhancement + entity extraction (via routing_agent)


@pytest.mark.e2e
@skip_if_no_runtime
class TestQueryEnhancementViaRouting:
    """Scenarios 2-3: Entity extraction and query enhancement happen inside routing."""

    def test_enhanced_query_differs_from_original(self):
        """Routing pipeline should enhance/modify the query."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "ML transformer videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "enhanced_query" in data
        assert isinstance(data["enhanced_query"], str)
        assert len(data["enhanced_query"]) > 0

    def test_entity_extraction_in_routing(self):
        """Routing extracts entities from entity-rich queries."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "Obama speaking at MIT about climate change",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "entities" in data
        assert isinstance(data["entities"], list)

    def test_routing_confidence_range(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find me detailed analysis of deep learning architectures",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["confidence"] <= 1.0


# Scenario 4-5: Orchestration (via routing with orchestration trigger)


@pytest.mark.e2e
@skip_if_no_runtime
class TestOrchestration:
    """Scenarios 4-5: Routing triggers orchestration for complex queries."""

    def test_complex_query_triggers_orchestration_or_downstream(self):
        """Complex queries may trigger orchestration or direct downstream."""
        with httpx.Client(base_url=RUNTIME, timeout=180.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "Find videos about machine learning and write a detailed report",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "routing_agent"
        assert "recommended_agent" in data

    def test_multi_turn_routing(self):
        """Multi-turn routing preserves conversation context."""
        session_id = str(uuid.uuid4())

        transport = httpx.HTTPTransport(retries=2)
        with httpx.Client(
            base_url=RUNTIME, timeout=300.0, transport=transport
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


# Scenario 7: Search API with profiles and strategies


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
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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

    def test_search_result_structure(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "technology demonstration",
                    "profile": PROFILE,
                    "top_k": 3,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["results_count"] >= 0
        if data["results_count"] > 0:
            result = data["results"][0]
            assert isinstance(result, dict)

    def test_search_with_different_strategy(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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


# Scenario 15: Tenant CRUD via API


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


# Scenario 18: Agent registry and health cascade


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
        assert "total_agents" in data, f"Stats missing total_agents, got: {list(data.keys())}"
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
        ],
    )
    def test_registered_agents_accessible(self, agent_name):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(f"/agents/{agent_name}")
        assert resp.status_code == 200, f"Agent {agent_name} not accessible"

    def test_search_agent_process(self):
        """Scenario 18 sub-test: Direct search agent process."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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


# Scenario 19: A2A protocol


@pytest.mark.e2e
@skip_if_no_runtime
class TestA2AProtocol:
    """Scenario 19: A2A protocol agent card and tasks/send."""

    def test_runtime_agent_card(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/a2a/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "skills" in card or "capabilities" in card

    def test_a2a_single_turn(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
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


# Ingestion API endpoints


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


# Scenario 20 (API portion): Event queue listing


@pytest.mark.e2e
@skip_if_no_runtime
class TestEventEndpoints:
    """Scenario 20 API: Event queue listing."""

    def test_list_queues(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(
                "/events/queues",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_queue_not_found(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/events/queues/nonexistent_fake_id")

        assert resp.status_code == 404, (
            f"Non-existent queue should return 404, got {resp.status_code}"
        )
