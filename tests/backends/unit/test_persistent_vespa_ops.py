"""PersistentVespaOps reuses ONE HTTP session across data-plane operations.

pyvespa's ``Vespa.query()``/``feed_data_point()``/``get_data()``/
``delete_data()`` each run ``with VespaSync(self, pool_maxsize=1)`` — a
fresh connection pool and TCP(+TLS) handshake per operation. Long-lived
callers (config/adapter stores, the backend's metadata client) must route
those ops through a persistent session instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vespa.application import VespaSync

from cogniverse_vespa._vespa_factory import (
    PersistentVespaOps,
    make_persistent_vespa_ops,
    make_vespa_app,
)


def test_construction_builds_exactly_one_sync_session():
    with patch("vespa.application.VespaSync", wraps=VespaSync) as spy:
        ops = make_persistent_vespa_ops(url="http://localhost", port=1)
    assert spy.call_count == 1
    assert ops.url == "http://localhost:1"


def test_ops_route_through_the_persistent_session_without_new_syncs():
    with patch("vespa.application.VespaSync", wraps=VespaSync) as spy:
        ops = make_persistent_vespa_ops(url="http://localhost", port=1)
        session = MagicMock()
        ops._sync = session

        ops.query(yql="select * from sources * where true")
        ops.feed_data_point(schema="s", data_id="d", fields={})
        ops.get_data(schema="s", data_id="d")
        ops.delete_data(schema="s", data_id="d")
        ops.update_data(schema="s", data_id="d", fields={})

    assert spy.call_count == 1
    assert session.query.call_count == 1
    assert session.query.call_args.kwargs["yql"] == (
        "select * from sources * where true"
    )
    assert session.feed_data_point.call_count == 1
    assert session.get_data.call_count == 1
    assert session.delete_data.call_count == 1
    assert session.update_data.call_count == 1


def test_close_releases_the_http_client():
    ops = PersistentVespaOps(make_vespa_app(url="http://localhost", port=1))
    session = MagicMock()
    ops._sync = session

    ops.close()

    session._close_http_client.assert_called_once()


def test_stores_expose_close_that_releases_their_session():
    from cogniverse_vespa.config.config_store import VespaConfigStore
    from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

    for store_cls in (VespaConfigStore, VespaAdapterStore):
        store = store_cls(backend_url="http://localhost", backend_port=1)
        session = MagicMock()
        store.vespa_app._sync = session

        store.close()

        session._close_http_client.assert_called_once()


def test_store_close_is_noop_for_injected_plain_app():
    from cogniverse_vespa.config.config_store import VespaConfigStore

    injected = MagicMock(spec=[])  # no close attribute at all
    store = VespaConfigStore(vespa_app=injected)

    store.close()  # must not raise
