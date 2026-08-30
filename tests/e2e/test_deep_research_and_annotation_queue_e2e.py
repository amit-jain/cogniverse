"""
E2E tests for new features against the live runtime.

Requires live runtime at http://localhost:33000 with LM + Vespa + Phoenix.
Each test exercises the full HTTP round-trip through the actual ASGI app.

Features tested:
1. Deep research agent — decompose → search → synthesize via runtime API
2. Annotation queue — GET/POST queue endpoints via runtime API
3. Content rails — blocked query returns error via routing agent
"""

import uuid
from datetime import datetime, timezone

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID


@pytest.mark.e2e
class TestDeepResearchE2E:
    """Deep research agent through the runtime HTTP API."""

    def test_deep_research_returns_structured_report(self):
        """POST /agents/deep_research_agent/process → structured research output."""
        # Deep research chains 3 DSPy calls (decompose → evaluate → synthesize)
        # and each is 60-80s on CPU-served LM, so the 180s default was tight.
        query = "What visual patterns appear in outdoor activity videos?"
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            resp = client.post(
                "/agents/deep_research_agent/process",
                json={
                    "agent_name": "deep_research_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
        )
        data = resp.json()
        assert set(data) == {"status", "agent", "message", "result"}, data
        assert data["status"] == "success"
        assert data["agent"] == "deep_research_agent"
        assert data["message"] == f"Research complete for '{query}'"

        # DeepResearchOutput.model_dump(): the summary, the sub-questions and
        # the confidence are LM-chosen and bounded structurally; the evidence
        # rows are one search per sub-question in decomposition order, so the
        # first iteration's rows are pinned to the sub-question list exactly.
        result = data["result"]
        assert set(result) == {
            "summary",
            "sub_questions",
            "evidence",
            "citations",
            "iterations_used",
            "gaps_remaining",
            "confidence",
            "rlm_synthesis",
            "rlm_telemetry",
        }, result
        assert len(result["summary"]) > 50, (
            f"Summary too short ({len(result['summary'])} chars)"
        )
        assert len(result["sub_questions"]) >= 2, (
            f"Expected >=2 sub-questions, got {result['sub_questions']}"
        )
        assert len(result["sub_questions"]) <= 5, result["sub_questions"]
        assert result["iterations_used"] >= 1
        assert result["iterations_used"] <= 3, result["iterations_used"]
        assert len(result["evidence"]) >= 1, "Should collect evidence"
        sub_question_count = len(result["sub_questions"])
        assert [
            row["question"] for row in result["evidence"][:sub_question_count]
        ] == result["sub_questions"], result["evidence"]
        for row in result["evidence"]:
            assert set(row) <= {"question", "results", "source", "error"}, row
            assert row["source"] == "search", row
            assert isinstance(row["results"], list), row
        assert result["confidence"] > 0.0
        assert result["confidence"] <= 1.0, result["confidence"]
        assert result["rlm_synthesis"] is None, result
        assert result["rlm_telemetry"] is None, result


@pytest.mark.e2e
class TestAnnotationQueueE2E:
    """Annotation queue REST endpoints through the runtime."""

    def test_queue_lifecycle_via_api(self):
        """Enqueue one owned request → pending → assign → complete → gone."""
        span_id = f"e2e-queue-{uuid.uuid4().hex}"
        timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        request = {
            "span_id": span_id,
            "timestamp": timestamp.isoformat(),
            "query": "search for video clips of animals",
            "chosen_agent": "search_agent",
            "routing_confidence": 0.42,
            "outcome": "ambiguous",
            "priority": "medium",
            "reason": "e2e annotation queue lifecycle",
            "context": {"source": "test_queue_lifecycle_via_api"},
            "status": "pending",
            "assigned_to": None,
            "assigned_at": None,
            "sla_deadline": None,
            "completed_at": None,
            "label": None,
            "agent_type": "routing",
            "tenant_id": None,
        }
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.get("/agents/annotations/queue")
            assert resp.status_code == 200
            initial = resp.json()
            assert set(initial) == {"statistics", "pending", "assigned", "expired"}
            assert set(initial["statistics"]) == {"total", "by_status", "by_priority"}
            initial_total = initial["statistics"]["total"]
            assert span_id not in {r["span_id"] for r in initial["pending"]}

            enqueue = client.post(
                "/agents/annotations/queue/enqueue", json={"requests": [request]}
            )
            assert enqueue.status_code == 200, enqueue.text[:300]
            assert enqueue.json() == {
                "enqueued": 1,
                "skipped": 0,
                "queue_total": initial_total + 1,
            }

            resp = client.get("/agents/annotations/queue")
            assert resp.status_code == 200
            data = resp.json()
            assert data["statistics"]["total"] == initial_total + 1
            mine = [r for r in data["pending"] if r["span_id"] == span_id]
            assert mine == [request], mine

            assign = client.post(
                f"/agents/annotations/queue/{span_id}/assign",
                json={"reviewer": "e2e-reviewer", "sla_hours": 1},
            )
            assert assign.status_code == 200, assign.text[:300]
            assigned = assign.json()
            assert set(assigned) == {"status", "annotation"}
            assert assigned["status"] == "assigned"
            assert assigned["annotation"]["span_id"] == span_id
            assert assigned["annotation"]["status"] == "assigned"
            assert assigned["annotation"]["assigned_to"] == "e2e-reviewer"
            assert datetime.fromisoformat(assigned["annotation"]["assigned_at"])
            assert datetime.fromisoformat(assigned["annotation"]["sla_deadline"])

            complete = client.post(
                f"/agents/annotations/queue/{span_id}/complete",
                json={"reasoning": "e2e lifecycle", "annotator": "e2e"},
            )
            assert complete.status_code == 200, complete.text[:300]
            completed = complete.json()
            assert set(completed) == {"status", "persisted", "annotation"}
            assert completed["status"] == "completed"
            assert completed["persisted"] is False
            assert completed["annotation"]["span_id"] == span_id
            assert completed["annotation"]["status"] == "completed"
            assert completed["annotation"]["label"] is None
            assert datetime.fromisoformat(completed["annotation"]["completed_at"])

            resp = client.get("/agents/annotations/queue")
            assert resp.status_code == 200
            final = resp.json()
            assert span_id not in {r["span_id"] for r in final["pending"]}
            assert span_id not in {r["span_id"] for r in final["assigned"]}

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
