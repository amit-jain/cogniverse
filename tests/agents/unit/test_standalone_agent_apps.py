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
        return_value=sm_module.SummaryResult(
            summary="the answer",
            key_points=["a", "b"],
            visual_insights=[],
            confidence_score=0.8,
            thinking_phase=sm_module.ThinkingPhase(
                key_themes=["t1"],
                content_categories=["c1"],
                relevance_scores={"r1": 0.5},
                visual_elements=[],
                reasoning="because",
            ),
            metadata={"x": 1},
        )
    )
    monkeypatch.setattr(sm_module, "summarizer_agent", fake)
    r = summarizer_client.post("/process", json={"query": "what?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["summary"] == "the answer"
    assert body["key_points"] == ["a", "b"]
    assert body["visual_insights"] == []
    assert body["confidence_score"] == 0.8
    assert body["metadata"] == {"x": 1}
    assert body["thinking_process"] == {
        "themes": ["t1"],
        "categories": ["c1"],
        "reasoning": "because",
    }


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


def test_detailed_report_process_happy_path_returns_envelope(
    detailed_report_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    result = SimpleNamespace(
        executive_summary="exec",
        detailed_findings="findings",
        visual_analysis="visuals",
        technical_details="tech",
        recommendations=["r1", "r2"],
        confidence_assessment="high",
        metadata={"k": "v"},
        thinking_phase=SimpleNamespace(
            content_analysis="ca",
            technical_findings="tf",
            patterns_identified=["p1", "p2"],
            reasoning="because",
        ),
    )
    fake = MagicMock()
    fake.generate_report = AsyncMock(return_value=result)
    monkeypatch.setattr(dr_module, "detailed_report_agent", fake)

    r = detailed_report_client.post(
        "/process",
        json={"query": "quarterly review", "search_results": [{"id": "d1"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["executive_summary"] == "exec"
    assert body["detailed_findings"] == "findings"
    assert body["recommendations"] == ["r1", "r2"]
    assert body["confidence_assessment"] == "high"
    assert body["metadata"] == {"k": "v"}
    # patterns_identified is renamed to "patterns" in the envelope.
    assert body["thinking_process"] == {
        "content_analysis": "ca",
        "technical_findings": "tf",
        "patterns": ["p1", "p2"],
        "reasoning": "because",
    }
    # The handler built a ReportRequest from the dict and passed it through.
    req = fake.generate_report.call_args.args[0]
    assert req.query == "quarterly review"
    assert req.search_results == [{"id": "d1"}]


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


def test_summarizer_agent_card_url_is_absolute(
    summarizer_client: TestClient,
) -> None:
    """A2A clients resolve the card url — a bare path is unresolvable."""
    r = summarizer_client.get("/agent.json")
    assert r.status_code == 200
    assert r.json()["url"] == "http://localhost:8003"


def test_detailed_report_agent_card_url_is_absolute(
    detailed_report_client: TestClient,
) -> None:
    """A2A clients resolve the card url — a bare path is unresolvable."""
    r = detailed_report_client.get("/agent.json")
    assert r.status_code == 200
    assert r.json()["url"] == "http://localhost:8004"


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


def test_search_agent_card_url_is_absolute(search_client: TestClient) -> None:
    # A remote A2A client cannot resolve a bare "/process"; the card must
    # advertise an absolute base like its summarizer / detailed-report siblings.
    r = search_client.get("/agent.json")
    assert r.json()["url"] == "http://localhost:8002"


def test_summarizer_summarize_happy_path_envelope(
    summarizer_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake.summarize = AsyncMock(
        return_value=sm_module.SummaryResult(
            summary="condensed",
            key_points=["p1"],
            visual_insights=["v1"],
            confidence_score=0.7,
            thinking_phase=sm_module.ThinkingPhase(
                key_themes=[],
                content_categories=[],
                relevance_scores={},
                visual_elements=[],
                reasoning="r",
            ),
            metadata={"k": "v"},
        )
    )
    monkeypatch.setattr(sm_module, "summarizer_agent", fake)
    r = summarizer_client.post(
        "/summarize",
        params={"query": "what happened?"},
        json=[{"id": "r1"}],
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "summary": "condensed",
        "key_points": ["p1"],
        "visual_insights": ["v1"],
        "confidence_score": 0.7,
        "metadata": {"k": "v"},
    }


def test_detailed_report_generate_happy_path_envelope(
    detailed_report_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake.generate_report = AsyncMock(
        return_value=dr_module.ReportResult(
            executive_summary="exec",
            detailed_findings=[{"finding": "f1"}],
            visual_analysis=[{"visual": "v1"}],
            technical_details=[{"tech": "t1"}],
            recommendations=["do x"],
            confidence_assessment={"overall": 0.9},
            thinking_phase=dr_module.ThinkingPhase(
                content_analysis={},
                visual_assessment={},
                technical_findings=[],
                patterns_identified=[],
                gaps_and_limitations=[],
                reasoning="r",
            ),
            metadata={"m": 1},
        )
    )
    monkeypatch.setattr(dr_module, "detailed_report_agent", fake)
    r = detailed_report_client.post(
        "/generate_report",
        params={"query": "analyze"},
        json=[{"id": "r1"}],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["executive_summary"] == "exec"
    assert body["detailed_findings"] == [{"finding": "f1"}]
    assert body["recommendations"] == ["do x"]
    assert body["confidence_assessment"] == {"overall": 0.9}
    assert set(body.keys()) == {
        "executive_summary",
        "detailed_findings",
        "visual_analysis",
        "technical_details",
        "recommendations",
        "confidence_assessment",
        "metadata",
    }


def test_search_upload_video_happy_path_envelope(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake._search_by_video = MagicMock(return_value=[{"id": "v1"}, {"id": "v2"}])
    monkeypatch.setattr(sa_module, "search_agent", fake)
    r = search_client.post(
        "/upload/video",
        params={"tenant_id": "acme", "top_k": 5},
        files={"file": ("clip.mp4", b"\x00\x01", "video/mp4")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["search_type"] == "video"
    assert body["filename"] == "clip.mp4"
    assert body["total_results"] == 2
    assert body["results"] == [{"id": "v1"}, {"id": "v2"}]


def test_search_upload_image_happy_path_envelope(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake._search_by_image = MagicMock(return_value=[{"id": "i1"}])
    monkeypatch.setattr(sa_module, "search_agent", fake)
    r = search_client.post(
        "/upload/image",
        params={"tenant_id": "acme"},
        files={"file": ("pic.png", b"\x00", "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["search_type"] == "image"
    assert body["filename"] == "pic.png"
    assert body["total_results"] == 1


def test_search_enhanced_happy_path_returns_backend_result(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake.search_with_relationship_context = MagicMock(
        return_value={"hits": [{"id": "h1"}], "total": 1}
    )
    monkeypatch.setattr(sa_module, "search_agent", fake)
    r = search_client.post(
        "/search/enhanced",
        json={"query": "graphs of revenue", "tenant_id": "acme"},
    )
    assert r.status_code == 200
    assert r.json() == {"hits": [{"id": "h1"}], "total": 1}
    # the route must pass the resolved original_query (falls back to query)
    _, kwargs = fake.search_with_relationship_context.call_args
    assert kwargs["tenant_id"] == "acme"
    ctx = fake.search_with_relationship_context.call_args[0][0]
    assert ctx.original_query == "graphs of revenue"


# --- Real-lifespan / real-agent coverage (the MagicMock fixtures above never
# run the real startup, so they hid: the lifespan crash from a missing
# config_manager, and the AttributeError from the undefined get_agent_skills()
# on /agent.json). These drive the actual lifespan with an in-memory (but real)
# ConfigManager and assert /agent.json serves real skill descriptors. ---


def _in_memory_config_manager():
    """Real ConfigManager over an in-memory store, seeded with a default
    SystemConfig — enough for the agents' __init__ (DSPy LM is lazy, no
    network) without depending on BACKEND_URL / a live config store."""
    from datetime import datetime, timezone

    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigStore

    class _Store(ConfigStore):
        def __init__(self):
            self._d = {}

        def _k(self, t, s, svc, k):
            return f"{t}:{s.value}:{svc}:{k}"

        def initialize(self):
            pass

        def set_config(self, tenant_id, scope, service, config_key, config_value):
            now = datetime.now(timezone.utc)
            e = ConfigEntry(
                tenant_id=tenant_id,
                scope=scope,
                service=service,
                config_key=config_key,
                config_value=config_value,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._d[self._k(tenant_id, scope, service, config_key)] = e
            return e

        def get_config(self, tenant_id, scope, service, config_key, version=None):
            return self._d.get(self._k(tenant_id, scope, service, config_key))

        def get_config_history(self, tenant_id, scope, service, config_key, limit=10):
            e = self.get_config(tenant_id, scope, service, config_key)
            return [e] if e else []

        def list_configs(self, tenant_id, scope=None, service=None):
            return [
                e
                for e in self._d.values()
                if e.tenant_id == tenant_id
                and (scope is None or e.scope == scope)
                and (service is None or e.service == service)
            ]

        def list_all_configs(self):
            return list(self._d.values())

        def delete_config(self, tenant_id, scope, service, config_key):
            return (
                self._d.pop(self._k(tenant_id, scope, service, config_key), None)
                is not None
            )

        def export_configs(self, tenant_id, include_history=False):
            return {"configs": []}

        def import_configs(self, tenant_id, configs):
            return 0

        def get_stats(self):
            return {"total": len(self._d)}

        def health_check(self):
            return True

    cm = ConfigManager(store=_Store())
    cm.set_system_config(SystemConfig())
    return cm


def test_summarizer_lifespan_starts_and_agent_card_lists_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        _in_memory_config_manager,
    )
    monkeypatch.setattr(sm_module, "summarizer_agent", None)
    # TestClient as a context manager runs the REAL lifespan (startup must not
    # crash on the missing config_manager).
    with TestClient(sm_module.app) as client:
        assert sm_module.summarizer_agent is not None  # lifespan built it
        card = client.get("/agent.json")
        assert card.status_code == 200
        skills = card.json()["skills"]
        assert [s["name"] for s in skills] == ["process"]
        assert "input_schema" in skills[0] and "output_schema" in skills[0]
        health = client.get("/health")
        assert health.json()["status"] == "healthy"
    monkeypatch.setattr(sm_module, "summarizer_agent", None)


def test_detailed_report_lifespan_starts_and_agent_card_lists_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        _in_memory_config_manager,
    )
    monkeypatch.setattr(dr_module, "detailed_report_agent", None)
    with TestClient(dr_module.app) as client:
        assert dr_module.detailed_report_agent is not None
        card = client.get("/agent.json")
        assert card.status_code == 200
        skills = card.json()["skills"]
        assert [s["name"] for s in skills] == ["process"]
        assert "input_schema" in skills[0] and "output_schema" in skills[0]
    monkeypatch.setattr(dr_module, "detailed_report_agent", None)


def test_search_lifespan_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entering the app lifespan builds the real SearchAgent. Pre-fix the
    lifespan imported cogniverse_core.schemas.schema_loader (nonexistent) and
    crashed with ModuleNotFoundError before the agent was ever built."""
    import cogniverse_foundation.config.utils as config_utils

    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", "8080")
    monkeypatch.setattr(
        config_utils, "create_default_config_manager", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(sa_module, "search_agent", None)

    with TestClient(sa_module.app):
        assert sa_module.search_agent is not None

    monkeypatch.setattr(sa_module, "search_agent", None)
