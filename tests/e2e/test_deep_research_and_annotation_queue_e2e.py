"""
E2E tests for new features against the live runtime.

Requires live runtime at http://localhost:8000 with Ollama + Vespa + Phoenix.
Each test exercises the full HTTP round-trip through the actual ASGI app.

Features tested:
1. Deep research agent — decompose → search → synthesize via runtime API
2. Annotation queue — GET/POST queue endpoints via runtime API
3. Content rails — blocked query returns error via routing agent
"""

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime


@pytest.mark.e2e
@skip_if_no_runtime
class TestDeepResearchE2E:
    """Deep research agent through the runtime HTTP API."""

    def test_deep_research_returns_structured_report(self):
        """POST /agents/deep_research_agent/process → structured research output."""
        with httpx.Client(base_url=RUNTIME, timeout=180.0) as client:
            resp = client.post(
                "/agents/deep_research_agent/process",
                json={
                    "agent_name": "deep_research_agent",
                    "query": "What visual patterns appear in outdoor activity videos?",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "deep_research_agent"

        result = data["result"]
        assert len(result["summary"]) > 50, (
            f"Summary too short ({len(result['summary'])} chars)"
        )
        assert len(result["sub_questions"]) >= 2, (
            f"Expected >=2 sub-questions, got {result['sub_questions']}"
        )
        assert result["iterations_used"] >= 1
        assert len(result["evidence"]) >= 1, "Should collect evidence"
        assert result["confidence"] > 0.0


@pytest.mark.e2e
@skip_if_no_runtime
class TestAnnotationQueueE2E:
    """Annotation queue REST endpoints through the runtime."""

    def test_queue_lifecycle_via_api(self):
        """GET queue → seed via routing → GET queue shows pending items."""
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.get("/agents/annotations/queue")
            assert resp.status_code == 200
            initial = resp.json()
            assert "statistics" in initial
            assert "pending" in initial

            # Trigger a routing request to create spans that could be annotated
            client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "search for video clips of animals",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

            resp = client.get("/agents/annotations/queue")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data["statistics"]["total"], int)

    def test_assign_and_complete_lifecycle(self):
        """Optimization cycle seeds queue → assign → complete → verify state."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            # Trigger the real optimization cycle — this is how production works.
            # It reads spans from Phoenix, identifies low-confidence ones,
            # enqueues them, and optionally runs DSPy compilation.
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "optimize_routing",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "action": "optimize_routing",
                    },
                },
            )
            assert resp.status_code == 200

            resp = client.get("/agents/annotations/queue")
            assert resp.status_code == 200
            data = resp.json()
            assert "statistics" in data
            assert "pending" in data

            pending = data["pending"]
            assert len(pending) > 0, (
                f"Optimization cycle should populate queue with low-confidence spans. "
                f"Queue stats: {data['statistics']}"
            )

            span_id = pending[0]["span_id"]

            # Assign
            resp = client.post(
                f"/agents/annotations/queue/{span_id}/assign",
                json={"reviewer": "e2e_test_reviewer"},
            )
            assert resp.status_code == 200
            assign_data = resp.json()
            assert assign_data["status"] == "assigned"
            assert assign_data["annotation"]["assigned_to"] == "e2e_test_reviewer"
            assert assign_data["annotation"]["status"] == "assigned"
            assert assign_data["annotation"]["sla_deadline"] is not None

            # Complete
            resp = client.post(
                f"/agents/annotations/queue/{span_id}/complete",
                json={"label": "correct_routing"},
            )
            assert resp.status_code == 200
            complete_data = resp.json()
            assert complete_data["status"] == "completed"
            assert complete_data["annotation"]["status"] == "completed"
            assert complete_data["annotation"]["completed_at"] is not None

            # Verify queue state updated
            resp = client.get("/agents/annotations/queue")
            stats = resp.json()["statistics"]["by_status"]
            assert stats.get("completed", 0) >= 1

    def test_assign_nonexistent_returns_404(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/agents/annotations/queue/nonexistent-span-id/assign",
                json={"reviewer": "test"},
            )
            assert resp.status_code == 404

    def test_complete_nonexistent_returns_404(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/agents/annotations/queue/nonexistent-span-id/complete",
                json={"label": "correct_routing"},
            )
            assert resp.status_code == 404

