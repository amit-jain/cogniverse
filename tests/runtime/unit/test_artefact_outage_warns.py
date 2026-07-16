"""A telemetry/Phoenix outage during artefact resolution must warn, not hide.

resolve_artefact_for_request returns None on any failure so the request falls
back to the default (un-optimized) prompts — an acceptable degrade. But it
logged only at DEBUG, so an operator never saw that optimized prompts had
silently stopped being served. These pin a WARNING on both failure paths.
"""

import logging
from unittest.mock import AsyncMock

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _dispatcher(factory):
    d = object.__new__(AgentDispatcher)
    d._artifact_manager_factory = factory
    return d


@pytest.mark.asyncio
async def test_factory_failure_warns_and_returns_none(caplog):
    def _boom(tenant_id):
        raise RuntimeError("Phoenix unreachable")

    d = _dispatcher(_boom)
    with caplog.at_level(logging.WARNING):
        result = await d.resolve_artefact_for_request(
            "search_agent", "acme:acme", "seed-1"
        )

    assert result is None
    assert any(
        "default prompts" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_load_failure_warns_and_returns_none(caplog, monkeypatch):
    am = AsyncMock()
    am.load_for_request = AsyncMock(side_effect=RuntimeError("Phoenix read failed"))

    d = _dispatcher(lambda tenant_id: am)
    monkeypatch.setattr(
        d, "_resolve_signature_variant", lambda tenant_id, agent_name: "default"
    )

    with caplog.at_level(logging.WARNING):
        result = await d.resolve_artefact_for_request(
            "search_agent", "acme:acme", "seed-1"
        )

    assert result is None
    assert any(
        "default prompts" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )
