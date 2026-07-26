"""VespaBackend caches the pyvespa app for metadata ops.

Each metadata call previously built a fresh ``make_vespa_app`` (a new
connection pool); the app is now cached and rebuilt only when url/port change.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from cogniverse_vespa.backend import VespaBackend


def _bare_backend(url="http://localhost", port=8080) -> VespaBackend:
    b = object.__new__(VespaBackend)
    b._url = url
    b._port = port
    b._metadata_app = None
    b._metadata_app_key = None
    return b


def test_metadata_app_is_cached_across_calls():
    b = _bare_backend()
    assert b._metadata_vespa_app() is b._metadata_vespa_app()


def test_metadata_app_rebuilt_when_url_changes():
    b = _bare_backend()
    first = b._metadata_vespa_app()
    b._url = "http://other-host"  # deploy-time override
    assert b._metadata_vespa_app() is not first


def test_rebuild_on_url_change_closes_stale_session():
    """The displaced client holds a live HTTP session — rebuilding without
    releasing it leaks the pool."""
    from unittest.mock import MagicMock

    b = _bare_backend()
    first = b._metadata_vespa_app()
    first._sync = MagicMock()

    b._url = "http://other-host"
    second = b._metadata_vespa_app()

    assert second is not first
    first._sync._close_http_client.assert_called_once()


def test_metadata_ops_share_one_persistent_session():
    """Metadata CRUD must reuse one HTTP session — pyvespa's per-op
    VespaSync costs a fresh TCP(+TLS) handshake per call."""
    from unittest.mock import MagicMock, patch

    from vespa.application import VespaSync

    b = _bare_backend()
    with patch("vespa.application.VespaSync", wraps=VespaSync) as spy:
        client = b._metadata_vespa_app()
        assert spy.call_count == 1

        client._sync = MagicMock()
        client.query(yql="select * from sources * where true")
        client.get_data(schema="s", data_id="d")
    assert spy.call_count == 1


def test_backend_close_releases_metadata_session():
    b = _bare_backend()
    b._vespa_ingestion_clients = {}
    b._async_ingestion_clients = {}
    b._vespa_search_backend = None
    client = b._metadata_vespa_app()
    client._sync = MagicMock()

    b.close()

    client._sync._close_http_client.assert_called_once()


def test_backend_close_releases_every_owned_client_once():
    backend = _bare_backend()
    search = MagicMock()
    ingestion = [MagicMock(), MagicMock()]
    metadata = MagicMock()
    backend._vespa_search_backend = search
    backend._vespa_ingestion_clients = {
        "agent_memories_one": ingestion[0],
        "agent_memories_two": ingestion[1],
    }
    backend._metadata_app = metadata

    backend.close()
    backend.close()

    search.close.assert_called_once_with()
    for client in ingestion:
        client.close.assert_called_once_with()
    metadata.close.assert_called_once_with()
    assert backend._vespa_search_backend is None
    assert backend._vespa_ingestion_clients == {}
    assert backend._metadata_app is None


def test_backend_close_is_single_execution_under_concurrency():
    backend = _bare_backend()
    search = MagicMock()
    entered = threading.Event()
    release = threading.Event()

    def delayed_close():
        entered.set()
        assert release.wait(timeout=2)

    search.close.side_effect = delayed_close
    backend._vespa_search_backend = search
    backend._vespa_ingestion_clients = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(backend.close) for _ in range(8)]
        assert entered.wait(timeout=2)
        release.set()
        for future in futures:
            future.result(timeout=2)

    search.close.assert_called_once_with()


def test_backend_close_attempts_every_client_before_reporting_failure():
    backend = _bare_backend()
    search = MagicMock()
    ingestion = MagicMock()
    metadata = MagicMock()
    search.close.side_effect = OSError("search pool stuck")
    ingestion.close.side_effect = RuntimeError("ingestion close failed")
    backend._vespa_search_backend = search
    backend._vespa_ingestion_clients = {"agent_memories": ingestion}
    backend._metadata_app = metadata

    with pytest.raises(RuntimeError) as exc_info:
        backend.close()

    assert str(exc_info.value) == (
        "Failed to close Vespa backend resources: search backend: search pool stuck; "
        "ingestion client agent_memories: ingestion close failed"
    )
    search.close.assert_called_once_with()
    ingestion.close.assert_called_once_with()
    metadata.close.assert_called_once_with()
    assert backend._vespa_search_backend is None
    assert backend._vespa_ingestion_clients == {}
    assert backend._metadata_app is None
