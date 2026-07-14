"""A Vespa soft timeout (HTTP 200 + root.errors) must fail the search.

Vespa reports query timeouts and container errors as HTTP 200 with a
``root.errors`` list and partial or empty ``root.children``. pyvespa's
``VespaQueryResponse.hits`` is just ``root.children`` and its
``raise_for_status`` returns early on 200, so consuming hits without
checking ``root.errors`` turned a degraded backend into "no results"
recorded as a SUCCESSFUL search — no retry, no breaker signal, green
metrics. Every ranking strategy defaults to a 2s query timeout, so any
slow query hits this path.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse

from cogniverse_core.common.utils.circuit_breaker import CircuitBreaker
from cogniverse_vespa import search_backend as sb_module
from cogniverse_vespa.search_backend import VespaSearchBackend


@pytest.fixture(autouse=True)
def _reset_breakers():
    CircuitBreaker.reset_registry()
    yield
    CircuitBreaker.reset_registry()


def _soft_timeout_body(children: list | None = None) -> dict:
    return {
        "root": {
            "id": "toplevel",
            "relevance": 1.0,
            "errors": [
                {
                    "code": 12,
                    "summary": "Timed out",
                    "source": "content",
                    "message": "Query timed out after 2.0s.",
                }
            ],
            "coverage": {
                "coverage": 40,
                "documents": 123,
                "degraded": {"timeout": True},
            },
            "children": children or [],
        }
    }


def _hit(doc_id: str, relevance: float) -> dict:
    return {
        "id": f"id:video:video_test::{doc_id}",
        "relevance": relevance,
        "fields": {"video_id": doc_id, "text": "some content"},
    }


def _clean_body() -> dict:
    return {
        "root": {
            "id": "toplevel",
            "relevance": 1.0,
            "fields": {"totalCount": 2},
            "coverage": {"coverage": 100, "documents": 2},
            "children": [_hit("doc1", 0.9), _hit("doc2", 0.8)],
        }
    }


def _response(body: dict) -> VespaQueryResponse:
    return VespaQueryResponse(
        json=body, status_code=200, url="http://localhost:8080/search/"
    )


@pytest.mark.parametrize(
    "children",
    [[], [_hit("partial1", 0.5)]],
    ids=["empty_children", "partial_children"],
)
def test_process_results_raises_on_root_errors(children):
    """200 + root.errors is a failed query, whether or not partial hits came back."""
    backend = object.__new__(VespaSearchBackend)

    with pytest.raises(VespaError) as exc_info:
        backend._process_results(_response(_soft_timeout_body(children)), "corr-1")

    assert "Timed out" in str(exc_info.value)


def test_process_results_parses_clean_response():
    """Pin the happy path: 200 with children and no errors parses fully."""
    backend = object.__new__(VespaSearchBackend)

    results = backend._process_results(_response(_clean_body()), "corr-2")

    assert [r.score for r in results] == [0.9, 0.8]
    assert [r.document.id for r in results] == ["doc1", "doc2"]


def test_process_results_warns_on_degraded_coverage_without_errors(caplog):
    """Degraded coverage without errors returns results but is not silent."""
    body = _clean_body()
    body["root"]["coverage"] = {
        "coverage": 60,
        "documents": 2,
        "degraded": {"timeout": True},
    }
    backend = object.__new__(VespaSearchBackend)

    with caplog.at_level(logging.WARNING):
        results = backend._process_results(_response(body), "corr-3")

    assert len(results) == 2
    assert any("degraded" in rec.message.lower() for rec in caplog.records)


def test_search_retries_soft_timeout_and_records_failures(monkeypatch):
    """The full search path treats a soft timeout as a transient failure:
    retried max_attempts times, every attempt recorded as a failed search,
    never as a success."""
    backend = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": 8080,
            "profiles": {"p1": {"type": "video", "schema_name": "video_test"}},
        },
        enable_connection_pool=False,
    )
    backend.vespa = MagicMock()
    backend.vespa.query.return_value = _response(_soft_timeout_body())

    monkeypatch.setattr(
        sb_module,
        "_RANKING_STRATEGIES_CACHE",
        {"video_test": {"bm25": {"needs_text_query": True}}},
    )
    import cogniverse_core.common.utils.retry as retry_module

    monkeypatch.setattr(retry_module.time, "sleep", lambda _s: None)

    with pytest.raises(VespaError):
        backend.search(
            {
                "query": "cat videos",
                "type": "video",
                "tenant_id": "acme",
                "strategy": "bm25",
            }
        )

    assert backend.vespa.query.call_count == 3
    assert backend.metrics.total_searches == 3
    assert backend.metrics.failed_searches == 3
    assert backend.metrics.successful_searches == 0
    assert backend.metrics.error_types["VespaError"] == 3
