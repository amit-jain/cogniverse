"""The SDK backend ABCs must declare the contract the real backend implements.

The SearchBackend/IngestionBackend ABCs previously declared method signatures
(search(query_embeddings, query_text, ...) -> List[Dict]) that the only real
backend (Vespa) does not implement — it uses search(query_dict) -> SearchResult
list. Registry 'compliance' mocks then certified the fiction. These tests pin
the ABC signatures to the real contract so a regression to the fiction fails.
"""

from __future__ import annotations

import inspect

from cogniverse_sdk.interfaces.backend import IngestionBackend, SearchBackend
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.search_backend import VespaSearchBackend


def _params(func) -> list[str]:
    return [p for p in inspect.signature(func).parameters if p != "self"]


def test_search_abc_declares_query_dict_contract():
    assert _params(SearchBackend.search) == ["query_dict"]


def test_ingest_documents_abc_declares_operation_type():
    params = _params(IngestionBackend.ingest_documents)
    assert params == ["documents", "schema_name", "operation_type"]


def test_update_document_abc_declares_schema_name():
    params = _params(IngestionBackend.update_document)
    assert params == ["document_id", "document", "schema_name"]


def test_ingest_stream_abc_declares_schema_name():
    assert _params(IngestionBackend.ingest_stream) == ["documents", "schema_name"]


def test_real_vespa_search_matches_abc_signature():
    """The real backends' search must accept the same positional shape the ABC
    declares — no more certifying a fiction."""
    assert _params(VespaSearchBackend.search) == _params(SearchBackend.search)
    assert _params(VespaBackend.search) == _params(SearchBackend.search)


def test_real_vespa_ingest_matches_abc_signature():
    assert _params(VespaBackend.ingest_documents) == _params(
        IngestionBackend.ingest_documents
    )


def test_real_vespa_ingest_stream_accepts_schema_name():
    # The impl may add optional trailing params (batch_size) but must lead with
    # the ABC's positional contract so ingest_documents receives the schema.
    assert _params(VespaBackend.ingest_stream)[:2] == _params(
        IngestionBackend.ingest_stream
    )
