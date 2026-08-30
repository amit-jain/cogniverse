"""A Vespa soft-timeout (HTTP 200 + root.errors) is surfaced, not swallowed.

The media search agents read root.children after only a status check, so a
degraded query (200 + root.errors code 12 + partial children) returned
incomplete results as if complete. vespa_search_children raises instead —
mirroring the base search_backend._process_results contract.
"""

from __future__ import annotations

import pytest
from vespa.exceptions import VespaError

from cogniverse_agents.search.vespa_query import (
    VespaSearchDegraded,
    vespa_search_children,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_soft_timeout_body_raises():
    degraded = {
        "root": {
            "errors": [{"code": 12, "summary": "Timeout", "message": "timed out"}],
            "children": [],  # partial/empty on a soft timeout
        }
    }
    with pytest.raises(VespaSearchDegraded, match="errors"):
        vespa_search_children(degraded)


def test_clean_body_returns_children():
    ok = {"root": {"children": [{"fields": {"id": "d1"}}, {"fields": {"id": "d2"}}]}}
    hits = vespa_search_children(ok)
    assert [h["fields"]["id"] for h in hits] == ["d1", "d2"]


def test_empty_result_is_not_an_error():
    assert vespa_search_children({"root": {"children": []}}) == []
    assert vespa_search_children({}) == []


def test_non_dict_coverage_shape_does_not_crash():
    body = {"root": {"coverage": "full", "children": [{"fields": {"id": "d1"}}]}}
    assert len(vespa_search_children(body)) == 1


def test_degraded_coverage_warns_but_returns_hits(caplog):
    import logging

    body = {
        "root": {
            "coverage": {"degraded": {"timeout": True}},
            "children": [{"fields": {"id": "d1"}}],
        }
    }
    with caplog.at_level(logging.WARNING):
        hits = vespa_search_children(body)
    assert len(hits) == 1
    assert any("degraded" in r.getMessage().lower() for r in caplog.records)


class _DegradedHTTPResponse:
    """HTTP 200 whose body carries root.errors — a Vespa soft-timeout."""

    status_code = 200
    text = "ok"

    hits = []

    def get_json(self):
        return {
            "root": {
                "errors": [{"code": 12, "summary": "Timeout", "message": "timed out"}],
                "coverage": {"degraded": {"timeout": True}},
                "children": [],
            }
        }

    def json(self):
        return self.get_json()


@pytest.mark.asyncio
async def test_document_text_search_raises_on_degraded_body(monkeypatch):
    """search_documents must surface a soft-timeout, not flatten it to []."""
    import numpy as np

    from cogniverse_agents.document_agent import DocumentAgent, DocumentAgentDeps

    monkeypatch.setattr(
        "cogniverse_agents.search.vespa_query.vespa_search_post",
        lambda *a, **k: _DegradedHTTPResponse(),
    )
    agent = DocumentAgent(
        deps=DocumentAgentDeps(
            tenant_id="t1",
            vespa_endpoint="http://localhost:1",
            deployed_document_schemas=("document_text",),
        )
    )
    # Bypass the lazy ColBERT load — the seam under test is the Vespa response.
    agent._text_query_encoder = type(
        "_Enc", (), {"encode": staticmethod(lambda q: np.zeros((4, 128), np.float32))}
    )()
    with pytest.raises(VespaSearchDegraded, match="errors"):
        await agent.search_documents(query="quarterly report", strategy="text", limit=3)


@pytest.mark.asyncio
async def test_audio_transcript_search_raises_on_degraded_body(monkeypatch):
    """search_audio must surface a soft-timeout through both except layers."""
    from pathlib import Path

    from cogniverse_agents.audio_analysis_agent import (
        AudioAnalysisAgent,
        AudioAnalysisDeps,
    )
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.search_backend import VespaSearchBackend
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    config_manager.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=8080)
    )
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    agent = AudioAnalysisAgent(
        deps=AudioAnalysisDeps(
            tenant_id="t1",
            vespa_endpoint="http://localhost:1",
            config_manager=config_manager,
            schema_loader=schema_loader,
            backend_config={
                "backend": {
                    "profiles": {
                        "audio_transcript_profile": {
                            "type": "audio",
                            "schema_name": "audio_content",
                        }
                    },
                    "default_profiles": {
                        "audio": {
                            "profile": "audio_transcript_profile",
                            "strategy": "transcript_search",
                        }
                    },
                }
            },
        )
    )
    backend = agent._get_backend()
    search_backend = VespaSearchBackend(
        config=backend.config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    class _Conn:
        def __init__(self, response):
            self._response = response

        def query(self, body):
            return self._response

    class _ConnCtx:
        def __init__(self, response):
            self._response = response

        def __enter__(self):
            return _Conn(self._response)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        search_backend.pool,
        "get_connection",
        lambda: _ConnCtx(_DegradedHTTPResponse()),
    )
    search_backend._tenant_schema_exists = lambda *a, **k: True
    backend._vespa_search_backend = search_backend
    with pytest.raises(VespaError, match="errors"):
        await agent.search_audio(
            query="keynote speech", search_mode="transcript", limit=3
        )
