"""Dependency-not-wired helpers must surface HTTP 503, not 500.

The three router files (`agents`, `search`, `ingestion`) carry FastAPI
dependency helpers that raise when ``app.dependency_overrides`` has not
yet installed the real ConfigManager / SchemaLoader / registry. Naked
``RuntimeError`` was bubbling to FastAPI's default 500. A partial-startup
window should look like "Service Unavailable, retry" to clients, not
"Internal Server Error".
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cogniverse_runtime.routers import agents as agents_router
from cogniverse_runtime.routers import ingestion as ingestion_router
from cogniverse_runtime.routers import search as search_router


def test_search_config_manager_dep_raises_503() -> None:
    with pytest.raises(HTTPException) as excinfo:
        search_router.get_config_manager_dependency()
    assert excinfo.value.status_code == 503
    assert "ConfigManager" in excinfo.value.detail


def test_search_schema_loader_dep_raises_503() -> None:
    with pytest.raises(HTTPException) as excinfo:
        search_router.get_schema_loader_dependency()
    assert excinfo.value.status_code == 503
    assert "SchemaLoader" in excinfo.value.detail


def test_ingestion_config_manager_dep_raises_503() -> None:
    with pytest.raises(HTTPException) as excinfo:
        ingestion_router.get_config_manager_dependency()
    assert excinfo.value.status_code == 503
    assert "ConfigManager" in excinfo.value.detail


def test_ingestion_schema_loader_dep_raises_503() -> None:
    with pytest.raises(HTTPException) as excinfo:
        ingestion_router.get_schema_loader_dependency()
    assert excinfo.value.status_code == 503
    assert "SchemaLoader" in excinfo.value.detail


def test_agents_dispatcher_partial_startup_raises_503() -> None:
    # Force the "not wired" branch by clearing the module-level state.
    saved = (
        agents_router._dispatcher,
        agents_router._agent_registry,
        agents_router._config_manager,
        agents_router._schema_loader,
    )
    agents_router._dispatcher = None
    agents_router._agent_registry = None
    agents_router._config_manager = None
    agents_router._schema_loader = None
    try:
        with pytest.raises(HTTPException) as excinfo:
            agents_router._ensure_dispatcher()
        assert excinfo.value.status_code == 503
        assert "Agent dependencies" in excinfo.value.detail
    finally:
        (
            agents_router._dispatcher,
            agents_router._agent_registry,
            agents_router._config_manager,
            agents_router._schema_loader,
        ) = saved
