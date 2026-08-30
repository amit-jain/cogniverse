from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.optimization_cli import run_monthly_reports


class _FakeTraces:
    async def get_all_spans(self, **kwargs):
        project = kwargs["project"]
        raise RuntimeError(
            f"Failed to query every span from Phoenix project {project}"
        ) from ConnectionError("phoenix unreachable")


class _FakeProvider:
    traces = _FakeTraces()


class _FakeTelemetryManager:
    config = SimpleNamespace(get_project_name=lambda tid: f"proj-{tid}")

    def get_provider(self, tenant_id=None):
        return _FakeProvider()


@pytest.mark.asyncio
async def test_monthly_reports_error_serializes_chained_phoenix_cause(
    monkeypatch, tmp_path
):
    async def _list_organizations_internal():
        return ["acme"]

    async def _list_tenants_for_org_internal(_org_id):
        return [
            SimpleNamespace(
                tenant_full_id="acme:prod",
                tenant_name="prod",
                status="active",
                schemas_deployed=["agent_memories"],
            )
        ]

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: MagicMock(name="config_manager"),
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.set_schema_loader",
        lambda loader: None,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.list_organizations_internal",
        _list_organizations_internal,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.list_tenants_for_org_internal",
        _list_tenants_for_org_internal,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda otlp_endpoint=None: _FakeTelemetryManager(),
    )

    result = await run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    error = result["failed_details"]["acme:prod"]
    assert "Failed to query every span from Phoenix project proj-acme:prod" in error
    assert "phoenix unreachable" in error
    assert result["failed"] == ["acme:prod"]
