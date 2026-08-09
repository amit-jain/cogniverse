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

from unittest.mock import MagicMock

import pytest
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse

from cogniverse_core.common.utils.circuit_breaker import CircuitBreaker
from cogniverse_core.common.utils.retry import RetryConfig
from cogniverse_vespa import search_backend as sb_module
from cogniverse_vespa.search_backend import VespaConnection, VespaSearchBackend


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
        backend._process_results(
            _response(_soft_timeout_body(children)), "corr-1", "video"
        )

    assert "Timed out" in str(exc_info.value)


def test_process_results_parses_clean_response():
    """Pin the happy path: 200 with children and no errors parses fully."""
    backend = object.__new__(VespaSearchBackend)

    results = backend._process_results(_response(_clean_body()), "corr-2", "video")

    assert [r.score for r in results] == [0.9, 0.8]
    assert [r.document.id for r in results] == ["doc1", "doc2"]


@pytest.mark.parametrize(
    ("query_type", "expected"),
    [
        ("video", "video"),
        ("document", "document"),
        ("memory", "document"),
        ("wiki", "document"),
    ],
)
def test_process_results_preserves_query_content_type(query_type, expected):
    backend = object.__new__(VespaSearchBackend)

    results = backend._process_results(
        _response(_clean_body()), "corr-type", query_type
    )

    assert [result.document.content_type.value for result in results] == [
        expected,
        expected,
    ]


def test_process_results_raises_on_degraded_coverage_without_errors():
    """Partial content-node coverage is not a successful search result."""
    body = _clean_body()
    body["root"]["coverage"] = {
        "coverage": 60,
        "documents": 2,
        "degraded": {"timeout": True},
    }
    backend = object.__new__(VespaSearchBackend)

    with pytest.raises(VespaError, match="coverage degraded"):
        backend._process_results(_response(body), "corr-3", "video")


@pytest.mark.parametrize(
    ("hit", "message"),
    [
        ({"relevance": 0.7, "fields": {"text": "missing id"}}, "non-empty id"),
        (
            {"id": "id:content:video_test::doc1", "relevance": 0.7, "fields": []},
            "fields must be a mapping",
        ),
    ],
)
def test_process_results_rejects_malformed_hits(hit, message):
    body = _clean_body()
    body["root"]["children"] = [hit]
    backend = object.__new__(VespaSearchBackend)

    with pytest.raises(ValueError, match=message):
        backend._process_results(_response(body), "corr-malformed", "video")


def test_process_results_rejects_missing_response():
    backend = object.__new__(VespaSearchBackend)

    with pytest.raises(VespaError, match="missing a hits collection"):
        backend._process_results(None, "corr-empty", "video")


def test_process_results_rejects_response_without_json_body():
    backend = object.__new__(VespaSearchBackend)
    response = MagicMock(spec=["hits"])
    response.hits = []

    with pytest.raises(VespaError, match="missing get_json"):
        backend._process_results(response, "corr-no-json", "video")


def test_process_results_rejects_hit_without_relevance():
    body = _clean_body()
    body["root"]["children"] = [
        {
            "id": "id:content:video_test::doc1",
            "fields": {"text": "missing relevance"},
        }
    ]
    backend = object.__new__(VespaSearchBackend)

    with pytest.raises(ValueError, match="relevance"):
        backend._process_results(_response(body), "corr-no-relevance", "video")


@pytest.mark.parametrize(
    "body",
    [
        _soft_timeout_body(),
        {
            "root": {
                "coverage": {
                    "coverage": 50,
                    "documents": 2,
                    "degraded": {"timeout": True},
                },
                "children": [],
            }
        },
    ],
    ids=["root_errors", "degraded_coverage"],
)
def test_connection_health_rejects_soft_failures(body):
    connection = object.__new__(VespaConnection)
    connection.connection_id = "health-probe"
    connection.is_healthy = True
    connection._sync = MagicMock()
    connection._sync.query.return_value = _response(body)

    assert connection.health_check() is False
    assert connection.is_healthy is False


@pytest.mark.parametrize(
    "response",
    [
        MagicMock(
            spec=["get_json"],
            get_json=MagicMock(return_value=_clean_body()),
        ),
        MagicMock(
            spec=["status_code", "get_json"],
            status_code=200,
            get_json=MagicMock(return_value={}),
        ),
        MagicMock(spec=["status_code"], status_code=200),
    ],
    ids=["missing_status", "missing_root", "missing_get_json"],
)
def test_connection_health_rejects_malformed_responses(response):
    connection = object.__new__(VespaConnection)
    connection.connection_id = "health-probe"
    connection.is_healthy = True
    connection._sync = MagicMock()
    connection._sync.query.return_value = response

    assert connection.health_check() is False
    assert connection.is_healthy is False


def test_backend_health_rejects_false_connection_probe():
    connection = MagicMock()
    connection.health_check.return_value = False
    checkout = MagicMock()
    checkout.__enter__.return_value = connection
    backend = object.__new__(VespaSearchBackend)
    backend.pool = MagicMock()
    backend.pool.get_connection.return_value = checkout
    backend.schema_name = "video_acme_acme"
    backend.get_metrics = MagicMock(return_value={})

    health = backend.health_check()

    assert health["status"] == "degraded"
    assert health["components"]["vespa"] == "unhealthy: health probe returned false"


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


def test_search_honors_constructor_retry_configuration(monkeypatch):
    backend = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": 8080,
            "profiles": {"p1": {"type": "video", "schema_name": "video_test"}},
        },
        retry_config=RetryConfig(
            max_attempts=2,
            initial_delay=0,
            jitter=False,
            exceptions=(VespaError,),
        ),
        enable_connection_pool=False,
    )
    backend.vespa = MagicMock()
    backend.vespa.query.return_value = _response(_soft_timeout_body())
    monkeypatch.setattr(
        sb_module,
        "_RANKING_STRATEGIES_CACHE",
        {"video_test": {"bm25": {"needs_text_query": True}}},
    )

    with pytest.raises(VespaError):
        backend.search(
            {
                "query": "cat videos",
                "type": "video",
                "tenant_id": "acme",
                "strategy": "bm25",
            }
        )

    assert backend.vespa.query.call_count == 2
    assert backend.metrics.total_searches == 2
