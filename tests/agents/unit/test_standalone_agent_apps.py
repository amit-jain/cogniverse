"""HTTP-level coverage for the standalone agent FastAPI apps.

Each of ``summarizer_agent``, ``detailed_report_agent``, and ``search_agent``
defines a module-level ``app = FastAPI(...)`` and an ``if __name__ ==
"__main__"`` block, so they can be deployed as standalone services. In
production traffic flows through the runtime dispatcher, but the
standalone HTTP surface — the 503-on-uninit guards, the 400-on-missing-
tenant guards, and the response envelopes — also has to behave.

These tests drive each app via ``fastapi.testclient.TestClient`` against
both the uninitialised state (no agent singleton yet) and the
initialised happy path (a mocked agent singleton).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from cogniverse_agents import detailed_report_agent as dr_module
from cogniverse_agents import search_agent as sa_module
from cogniverse_agents import summarizer_agent as sm_module


@pytest.fixture
def summarizer_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(sm_module, "summarizer_agent", None)
    yield TestClient(sm_module.app)
    monkeypatch.setattr(sm_module, "summarizer_agent", None)


@pytest.fixture
def detailed_report_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(dr_module, "detailed_report_agent", None)
    yield TestClient(dr_module.app)
    monkeypatch.setattr(dr_module, "detailed_report_agent", None)


# ---------------------------------------------------------------------------
# summarizer_agent app
# ---------------------------------------------------------------------------


def test_summarizer_health_initializing_when_no_singleton(
    summarizer_client: TestClient,
) -> None:
    r = summarizer_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "initializing", "agent": "summarizer_agent"}


def test_summarizer_health_healthy_when_initialised(
    summarizer_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sm_module, "summarizer_agent", MagicMock())
    r = summarizer_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["agent"] == "summarizer_agent"
    assert "summarization" in body["capabilities"]


def test_summarizer_process_503_when_not_initialised(
    summarizer_client: TestClient,
) -> None:
    r = summarizer_client.post("/process", json={"query": "hi"})
    assert r.status_code == 503
    assert "not initialized" in r.json()["detail"].lower()


def test_summarizer_summarize_503_when_not_initialised(
    summarizer_client: TestClient,
) -> None:
    r = summarizer_client.post("/summarize", params={"query": "hi"}, json=[])
    assert r.status_code == 503


def test_summarizer_process_happy_path_returns_envelope(
    summarizer_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake.summarize = AsyncMock(
        return_value=MagicMock(
            summary="the answer",
            key_points=["a", "b"],
            visual_insights=[],
            confidence_score=0.8,
            metadata={"x": 1},
            thinking_phase=MagicMock(
                key_themes=["t1"],
                content_categories=["c1"],
                reasoning="because",
            ),
        )
    )
    monkeypatch.setattr(sm_module, "summarizer_agent", fake)
    r = summarizer_client.post("/process", json={"query": "what?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["summary"] == "the answer"
    assert body["key_points"] == ["a", "b"]
    assert body["confidence_score"] == 0.8
    assert body["thinking_process"]["reasoning"] == "because"


# ---------------------------------------------------------------------------
# detailed_report_agent app
# ---------------------------------------------------------------------------


def test_detailed_report_health_initializing_when_no_singleton(
    detailed_report_client: TestClient,
) -> None:
    r = detailed_report_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "initializing", "agent": "detailed_report_agent"}


def test_detailed_report_health_healthy_when_initialised(
    detailed_report_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dr_module, "detailed_report_agent", MagicMock())
    r = detailed_report_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["agent"] == "detailed_report_agent"
    assert "detailed_report" in body["capabilities"]


def test_detailed_report_process_503_when_not_initialised(
    detailed_report_client: TestClient,
) -> None:
    r = detailed_report_client.post("/process", json={"query": "x"})
    assert r.status_code == 503


def test_detailed_report_generate_endpoint_503_when_not_initialised(
    detailed_report_client: TestClient,
) -> None:
    r = detailed_report_client.post("/generate_report", params={"query": "x"}, json=[])
    assert r.status_code == 503


def test_detailed_report_agent_card_skills_empty_when_no_singleton(
    detailed_report_client: TestClient,
) -> None:
    r = detailed_report_client.get("/agent.json")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "DetailedReportAgent"
    assert body["protocol"] == "a2a"
    # When the singleton is missing the card returns an empty skills list
    # rather than 500ing.
    assert body["skills"] == []


# ---------------------------------------------------------------------------
# search_agent app
# ---------------------------------------------------------------------------


@pytest.fixture
def search_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(sa_module, "search_agent", None)
    yield TestClient(sa_module.app)
    monkeypatch.setattr(sa_module, "search_agent", None)


def test_search_health_initializing_when_no_singleton(
    search_client: TestClient,
) -> None:
    r = search_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "initializing", "agent": "search_agent"}


def test_search_health_healthy_when_initialised(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake.embedding_type = "frame_based"
    monkeypatch.setattr(sa_module, "search_agent", fake)
    r = search_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["agent"] == "search_agent"
    assert body["embedding_type"] == "frame_based"


def test_search_process_503_when_not_initialised(search_client: TestClient) -> None:
    r = search_client.post("/process", json={"query": "x"})
    assert r.status_code == 503


def test_search_upload_video_503_when_not_initialised(
    search_client: TestClient,
) -> None:
    r = search_client.post(
        "/upload/video",
        params={"tenant_id": "acme"},
        files={"file": ("clip.mp4", b"\x00\x00\x00\x00", "video/mp4")},
    )
    assert r.status_code == 503


def test_search_upload_image_400_when_tenant_id_missing(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant_id is a query-string default ``""``, so the route's own guard
    is what surfaces the 400 — not FastAPI validation."""
    monkeypatch.setattr(sa_module, "search_agent", MagicMock())
    r = search_client.post(
        "/upload/image",
        files={"file": ("x.png", b"\x00", "image/png")},
    )
    assert r.status_code == 400
    assert "tenant_id" in r.json()["detail"].lower()


def test_search_agent_card(search_client: TestClient) -> None:
    r = search_client.get("/agent.json")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "SearchAgent"
    assert body["protocol"] == "a2a"
    assert "multi_modal_search" in body["capabilities"]
