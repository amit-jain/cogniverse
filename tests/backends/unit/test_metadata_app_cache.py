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
