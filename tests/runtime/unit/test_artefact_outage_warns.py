"""An artefact-store outage is a named overlay state, not a silent default.

``resolve_artefact_for_request`` keeps serving the request on default prompts
— raising would fail every request for the outage's duration — but the overlay
it returns says ``store_unavailable``/``error``, so "the store is down" is
never read as "this tenant promoted nothing and selected no variant".
"""

import logging
from unittest.mock import AsyncMock

import pytest

from cogniverse_agents.optimizer.artifact_manager import (
    ARTIFACT_LOAD_ERROR,
    ARTIFACT_LOAD_STORE_UNAVAILABLE,
)
from cogniverse_foundation.telemetry.providers.base import (
    DatasetStoreUnavailableError,
)
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _dispatcher(factory):
    d = object.__new__(AgentDispatcher)
    d._artifact_manager_factory = factory
    return d


@pytest.mark.asyncio
async def test_factory_outage_returns_store_unavailable_overlay(caplog):
    def _boom(tenant_id):
        raise DatasetStoreUnavailableError(
            "store down", endpoint="http://phoenix:6006", dataset="dspy-model-acme-x"
        )

    d = _dispatcher(_boom)
    with caplog.at_level(logging.WARNING):
        result = await d.resolve_artefact_for_request(
            "search_agent", "acme:acme", "seed-1"
        )

    assert result == {
        "prompts": None,
        "served_from": "default",
        "version": None,
        "variant_id": None,
        "artifact_load_status": ARTIFACT_LOAD_STORE_UNAVAILABLE,
        "error": "DatasetStoreUnavailableError: store down",
    }
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "Artefact resolution for agent=search_agent tenant=acme:acme failed "
        "(store_unavailable) — serving default prompts: store down"
    ]


@pytest.mark.asyncio
async def test_load_failure_returns_error_overlay_with_the_resolved_variant(
    caplog, monkeypatch
):
    am = AsyncMock()
    am.load_for_request = AsyncMock(side_effect=RuntimeError("Phoenix read failed"))

    d = _dispatcher(lambda tenant_id: am)
    monkeypatch.setattr(
        admin_router, "load_signature_variants", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        d, "_resolve_signature_variant", lambda tenant_id, agent_name: "variant-b"
    )

    with caplog.at_level(logging.WARNING):
        result = await d.resolve_artefact_for_request(
            "search_agent", "acme:acme", "seed-1"
        )

    assert result == {
        "prompts": None,
        "served_from": "default",
        "version": None,
        "variant_id": "variant-b",
        "artifact_load_status": ARTIFACT_LOAD_ERROR,
        "error": "RuntimeError: Phoenix read failed",
    }
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "Artefact resolution for agent=search_agent tenant=acme:acme failed "
        "(error) — serving default prompts: Phoenix read failed"
    ]


@pytest.mark.asyncio
async def test_no_factory_configured_returns_no_overlay():
    assert (
        await _dispatcher(None).resolve_artefact_for_request(
            "search_agent", "acme:acme", "seed-1"
        )
        is None
    )
