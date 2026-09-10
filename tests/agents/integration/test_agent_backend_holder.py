"""Agent searches survive the registry closing the backend they resolved.

The BackendRegistry owns the lifetime of every instance it hands out and
closes it on capacity eviction, an overwriting set and ``clear_instances()``.
SearchAgent, AudioAnalysisAgent and SearchService each cached one on the
object for the process's lifetime, so the first close turned every later
search into ``BackendClosedError`` until the process restarted. Each now
resolves through the registry per search and holds a checkout for the call.

Real Vespa, real registry, real agent objects: seed one wiki page, search it,
close the resolved instance the way production does, search again.
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document
from cogniverse_sdk.interfaces.backend import BackendClosedError
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    load_raw_schema_json,
    make_config_manager,
    schema_full_name,
)

pytestmark = pytest.mark.integration

_WIKI_NAMESPACE = "wiki_content"


def _embedding_dim(base_schema_name: str, field_name: str) -> int:
    """The field's declared tensor dimension, read from the shipped schema."""
    schema = load_raw_schema_json(base_schema_name)
    field = next(f for f in schema["fields"] if f["name"] == field_name)
    return int(re.search(r"\[(\d+)\]", field["type"]).group(1))


PAGE_TITLE = "Holder Page"
PAGE_TEXT = "the registry owns the backend lifetime"


class _StubEncoder:
    """Stands in for the remote query encoder (not the code under test)."""

    def encode(self, query: str):
        return np.zeros((1, 128), dtype=np.float32)


@pytest.fixture(scope="module")
def wiki_tenant(shared_vespa):
    """A tenant with wiki_pages deployed and one page fed, on real Vespa."""
    tenant_id = f"holder{uuid.uuid4().hex[:8]}:wiki"
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant_id,
        base_schema_name="wiki_pages",
        config_manager=config_manager,
    )
    schema_name = schema_full_name("wiki_pages", tenant_id)
    dim = _embedding_dim("wiki_pages", "embedding")

    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config={
            "url": "http://localhost",
            "port": shared_vespa["http_port"],
            "config_port": shared_vespa["config_port"],
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    doc_id = f"page_{uuid.uuid4().hex[:8]}"
    backend.put_document(
        Document(
            id=doc_id,
            content_id=doc_id,
            content_type=ContentType.TEXT,
            title=PAGE_TITLE,
            text_content=PAGE_TEXT,
            metadata={
                "tenant_id": tenant_id,
                "page_type": "entity",
                "slug": "holder_page",
                "entities": "[]",
                "sources": "[]",
                "cross_references": "[]",
                "update_count": 1,
                "embedding": [0.0] * dim,
            },
        ),
        schema_name=schema_name,
        base_schema_name="wiki_pages",
        namespace=_WIKI_NAMESPACE,
    )
    yield {
        "tenant_id": tenant_id,
        "doc_id": doc_id,
        "dim": dim,
        "config_manager": config_manager,
        "schema_loader": schema_loader,
        "http_port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
    }
    BackendRegistry.get_instance().clear_instances()


def _query_dict(wiki_tenant):
    return {
        "query": PAGE_TEXT,
        "type": "wiki",
        "top_k": 5,
        "tenant_id": wiki_tenant["tenant_id"],
        "strategy": "hybrid",
        "query_embeddings": np.zeros(wiki_tenant["dim"], dtype=np.float32),
    }


def _rows(results):
    """(id, title) for every hit, in rank order."""
    return [(r.document.id, r.document.metadata.get("title", "")) for r in results]


@pytest.fixture
def search_agent(wiki_tenant):
    with patch(
        "cogniverse_agents.search_agent.QueryEncoderFactory.create_encoder",
        return_value=_StubEncoder(),
    ):
        agent = SearchAgent(
            deps=SearchAgentDeps(
                tenant_id=wiki_tenant["tenant_id"],
                backend_url="http://localhost",
                backend_port=wiki_tenant["http_port"],
                backend_config_port=wiki_tenant["config_port"],
                auto_create_memory_schema=False,
            ),
            schema_loader=wiki_tenant["schema_loader"],
            config_manager=wiki_tenant["config_manager"],
            port=8034,
        )
    agent.query_encoder = _StubEncoder()
    return agent


def _expected_rows(wiki_tenant):
    return [(wiki_tenant["doc_id"], PAGE_TITLE)]


def test_search_agent_serves_after_the_registry_closes_that_instance(
    search_agent, wiki_tenant
):
    """The production sequence: resolve, registry clear (which closes), search."""
    query = _query_dict(wiki_tenant)
    resolved = search_agent._get_backend()
    assert _rows(search_agent._search_backend(query)) == _expected_rows(wiki_tenant)

    BackendRegistry.get_instance().clear_instances()

    # The instance the agent resolved is genuinely dead now: this is the
    # exact error the runtime pod served on every search after a clear.
    with pytest.raises(BackendClosedError) as closed:
        resolved.search(query)
    assert "is closed" in str(closed.value)

    assert _rows(search_agent._search_backend(query)) == _expected_rows(wiki_tenant)
    # Resolution went back to the registry rather than to a held handle.
    assert search_agent._get_backend() is not resolved


def test_a_search_in_flight_is_closed_only_after_it_releases(
    search_agent, wiki_tenant, monkeypatch
):
    """Barrier-executed interleaving: clear() lands mid-search.

    The checkout defers the close, so the search finishes on a live pool and
    the instance closes exactly once, after the release.
    """
    query = _query_dict(wiki_tenant)
    backend = search_agent._get_backend()

    closed_instances: list[int] = []
    real_close = type(backend).close

    def counting_close(self):
        closed_instances.append(id(self))
        return real_close(self)

    monkeypatch.setattr(type(backend), "close", counting_close, raising=True)

    search_entered = threading.Event()
    clear_returned = threading.Event()
    real_search = backend.search
    closes_seen_during_search: list[int] = []

    def barrier_search(qd):
        search_entered.set()
        assert clear_returned.wait(30), "the evictor never ran"
        return real_search(qd)

    monkeypatch.setattr(backend, "search", barrier_search, raising=True)

    def evict():
        assert search_entered.wait(30), "the search never reached the backend"
        BackendRegistry.get_instance().clear_instances()
        closes_seen_during_search.extend(closed_instances)
        clear_returned.set()

    evictor = threading.Thread(target=evict, name="evictor")
    evictor.start()
    try:
        results = search_agent._search_backend(query)
    finally:
        evictor.join(30)
        assert evictor.is_alive() is False

    assert _rows(results) == _expected_rows(wiki_tenant)
    # Nothing closed while the search held its checkout...
    assert closes_seen_during_search == []
    # ...and the deferred close ran exactly once on release.
    assert closed_instances == [id(backend)]


def test_concurrent_searches_share_one_instance_and_all_serve(
    search_agent, wiki_tenant
):
    """N concurrent searches: one cached instance, every search the exact row."""
    query = _query_dict(wiki_tenant)
    resolved = search_agent._get_backend()
    rows: list[list[tuple[str, str]]] = [[] for _ in range(8)]

    def run(slot):
        rows[slot] = _rows(search_agent._search_backend(dict(query)))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert [t.is_alive() for t in threads] == [False] * 8

    assert rows == [_expected_rows(wiki_tenant)] * 8
    assert search_agent._get_backend() is resolved


def test_the_agent_keeps_no_backend_attribute_between_searches(search_agent):
    """The removed holder: no attribute survives a search to go stale."""
    assert hasattr(search_agent, "_shared_backend") is False
    assert hasattr(search_agent, "_shared_backend_lock") is False
