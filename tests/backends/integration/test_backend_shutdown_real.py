"""Real Vespa lifecycle coverage for the unified backend."""

from __future__ import annotations

from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.search_backend import (
    ConnectionPoolConfig,
    VespaSearchBackend,
)


def test_backend_close_stops_real_search_pool_before_service_teardown(shared_vespa):
    search = VespaSearchBackend(
        backend_url="http://localhost",
        backend_port=shared_vespa["http_port"],
        schema_name="config_metadata",
        pool_config=ConnectionPoolConfig(
            min_connections=1,
            max_connections=1,
            health_check_interval=0.05,
        ),
    )
    backend = object.__new__(VespaBackend)
    backend._vespa_search_backend = search
    backend._vespa_ingestion_clients = {}
    backend._metadata_app = None

    health = search.health_check()
    assert health["status"] == "healthy"
    assert health["components"] == {"vespa": "healthy"}
    assert len(search.pool._connections) == 1
    assert search.pool._health_check_thread.is_alive()

    backend.close()

    assert backend._vespa_search_backend is None
    assert search.pool._connections == []
    assert search.pool._available == []
    assert not search.pool._health_check_thread.is_alive()
