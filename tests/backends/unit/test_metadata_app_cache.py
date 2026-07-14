"""VespaBackend caches the pyvespa app for metadata ops.

Each metadata call previously built a fresh ``make_vespa_app`` (a new
connection pool); the app is now cached and rebuilt only when url/port change.
"""

from __future__ import annotations

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
    from unittest.mock import MagicMock

    b = _bare_backend()
    b._vespa_ingestion_clients = {}
    b._async_ingestion_clients = {}
    client = b._metadata_vespa_app()
    client._sync = MagicMock()

    b.close()

    client._sync._close_http_client.assert_called_once()
