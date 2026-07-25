"""The SDK backend ABCs must declare the contract the real backend implements.

The SearchBackend/IngestionBackend ABCs previously declared method signatures
(search(query_embeddings, query_text, ...) -> List[Dict]) that the only real
backend (Vespa) does not implement — it uses search(query_dict) -> SearchResult
list. Registry 'compliance' mocks then certified the fiction. These tests pin
the ABC signatures to the real contract so a regression to the fiction fails.
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_sdk.interfaces.backend import Backend, IngestionBackend, SearchBackend
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


def test_runtime_profile_mutation_is_required_and_matches_real_backends():
    assert {"add_profile", "remove_profile"} <= SearchBackend.__abstractmethods__
    assert _params(VespaSearchBackend.add_profile) == _params(SearchBackend.add_profile)
    assert _params(VespaSearchBackend.remove_profile) == _params(
        SearchBackend.remove_profile
    )
    assert _params(VespaBackend.add_profile) == _params(SearchBackend.add_profile)
    assert _params(VespaBackend.remove_profile) == _params(SearchBackend.remove_profile)


class _InitializationProbe:
    def __init__(self, hook):
        Backend.__init__(self, "probe")
        self._hook = hook

    def initialize(self, config):
        Backend.initialize(self, config)

    def _initialize_backend(self, config):
        self._hook(config)


def test_backend_initialize_invokes_hook_once_under_concurrency():
    worker_count = 8
    start = threading.Barrier(worker_count + 1)
    release_hook = threading.Event()
    duplicate_hook = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def hook(config):
        nonlocal calls
        assert config == {"endpoint": "vespa"}
        with call_lock:
            calls += 1
            if calls > 1:
                duplicate_hook.set()
        assert release_hook.wait(timeout=2)

    backend = _InitializationProbe(hook)

    def initialize():
        start.wait()
        backend.initialize({"endpoint": "vespa"})

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(initialize) for _ in range(worker_count)]
        start.wait()
        duplicate_seen = duplicate_hook.wait(timeout=1)
        release_hook.set()
        for future in futures:
            future.result(timeout=2)

    assert duplicate_seen is False
    assert calls == 1
    assert backend._initialized is True


def test_backend_initialize_failure_is_retryable():
    seen_configs = []

    def hook(config):
        seen_configs.append(config)
        if len(seen_configs) == 1:
            raise RuntimeError("connection refused")

    backend = _InitializationProbe(hook)

    with pytest.raises(RuntimeError, match="connection refused"):
        backend.initialize({"attempt": 1})

    assert backend._initialized is False

    backend.initialize({"attempt": 2})

    assert seen_configs == [{"attempt": 1}, {"attempt": 2}]
    assert backend._initialized is True


def test_backend_instances_initialize_independently():
    hooks_entered = threading.Barrier(2, timeout=2)

    def hook(config):
        hooks_entered.wait()

    backends = [_InitializationProbe(hook), _InitializationProbe(hook)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(backend.initialize, {"instance": index})
            for index, backend in enumerate(backends)
        ]
        for future in futures:
            future.result(timeout=3)

    assert [backend._initialized for backend in backends] == [True, True]
