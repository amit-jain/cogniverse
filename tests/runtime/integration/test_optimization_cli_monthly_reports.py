"""Integration test for ``optimization_cli --mode monthly-reports``
against the real Vespa + Phoenix stack.

The cron writes ``usage-YYYYMM.json`` + ``performance-YYYYMM.json`` to
the configured output dir; both files must be valid JSON with the
exact schema downstream (billing, dashboards) consumes. The test
seeds real org/tenant rows in the live metadata schemas, runs the
function, and parses the files — no mocks of the filesystem, Phoenix
client, or tenant_manager helpers.
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

import pytest

from cogniverse_runtime.optimization_cli import run_monthly_reports

pytestmark = pytest.mark.integration


class TestRunMonthlyReportsWritesUsageAndPerformanceFiles:
    @pytest.mark.asyncio
    async def test_writes_two_files_with_real_org_tenant_data(
        self, memory_manager, vespa_instance, config_manager, tmp_path
    ):
        """Seeds 1 org + 2 tenants in the live metadata schemas, runs the
        report, and asserts both output files exist with the expected
        shape — including the exact tenant ids the report enumerated.
        """
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.admin import tenant_manager as tm

        tm.set_config_manager(config_manager)
        tm.set_schema_loader(FilesystemSchemaLoader(Path("configs/schemas")))
        backend = tm.get_backend()

        org_id = "monthly_rep_org"
        tenant_ids = [f"{org_id}:alpha", f"{org_id}:beta"]
        backend.create_metadata_document(
            schema="organization_metadata",
            doc_id=org_id,
            fields={
                "org_id": org_id,
                "org_name": "monthly-reports-integration",
                "created_at": int(_time.time() * 1000),
                "created_by": "integration-test",
                "status": "active",
                "tenant_count": len(tenant_ids),
            },
        )
        for tid in tenant_ids:
            backend.create_metadata_document(
                schema="tenant_metadata",
                doc_id=tid,
                fields={
                    "tenant_full_id": tid,
                    "org_id": org_id,
                    "tenant_name": tid.split(":", 1)[1],
                    "created_at": int(_time.time() * 1000),
                    "created_by": "integration-test",
                    "status": "active",
                    "schemas_deployed": ["agent_memories"],
                },
            )

        try:
            result = await run_monthly_reports(
                output_dir=str(tmp_path / "reports"),
                lookback_hours=1.0,
            )

            # Top-level summary contract.
            assert result["period"], "result.period must be set"
            assert len(result["files_written"]) == 2
            assert result["summary"]["org_count"] >= 1
            assert result["summary"]["tenant_count"] >= 2

            # Both files exist on disk.
            usage_path = Path(result["files_written"][0])
            perf_path = Path(result["files_written"][1])
            assert usage_path.exists() and usage_path.suffix == ".json"
            assert perf_path.exists() and perf_path.suffix == ".json"
            assert usage_path.name.startswith("usage-")
            assert perf_path.name.startswith("performance-")

            # Usage report content: seeded org+tenants are present with
            # exact metadata the chart relies on.
            usage = json.loads(usage_path.read_text())
            assert usage["period"] == result["period"]
            assert org_id in usage["organizations"]
            org_view = usage["organizations"][org_id]
            assert org_view["tenant_count"] == 2
            seen_tids = sorted(t["tenant_full_id"] for t in org_view["tenants"])
            assert seen_tids == sorted(tenant_ids), (
                f"usage report must enumerate exactly the seeded tenants; "
                f"got {seen_tids!r}, expected {sorted(tenant_ids)!r}"
            )
            for t_view in org_view["tenants"]:
                assert t_view["status"] == "active"
                assert "agent_memories" in t_view["schemas_deployed"]
                assert t_view["schema_count"] == 1

            # Performance report content: exact tenant set + numeric shape.
            perf = json.loads(perf_path.read_text())
            assert perf["period"] == result["period"]
            assert perf["lookback_hours"] == 1.0
            for tid in tenant_ids:
                assert tid in perf["tenants"], (
                    f"perf report must include tenant {tid!r}; "
                    f"got {sorted(perf['tenants'])!r}"
                )
                entry = perf["tenants"][tid]
                # Either we have a span_count int (Phoenix returned data
                # for an empty project = 0) or an error string. Both
                # are valid shapes; ZERO is acceptable for a fresh
                # tenant with no traffic.
                assert "span_count" in entry or "error" in entry, (
                    f"perf entry for {tid} missing required keys: {entry!r}"
                )
                if "span_count" in entry:
                    assert isinstance(entry["span_count"], int)
                    assert entry["span_count"] >= 0
                    assert entry["error_rate"] >= 0.0
        finally:
            for tid in tenant_ids:
                try:
                    backend.delete_metadata_document(
                        schema="tenant_metadata", doc_id=tid
                    )
                except Exception:
                    pass
            try:
                backend.delete_metadata_document(
                    schema="organization_metadata", doc_id=org_id
                )
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_phoenix_outage_surfaces_as_failed_and_nonzero_exit(
        self, memory_manager, vespa_instance, config_manager, tmp_path, monkeypatch
    ):
        """A per-tenant Phoenix outage must surface at the TOP level
        (result['failed']) so the cron's _run_failed gate exits non-zero.
        Previously the 'phoenix query failed' string lived only in the written
        file and evaded _run_failed's failed:/error: prefix, so a total outage
        wrote an all-errors report yet reported Succeeded and the dropped
        monthly reports were never regenerated. Org/tenant listing stays real
        (Vespa); only the Phoenix span read is faulted."""
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.admin import tenant_manager as tm
        from cogniverse_runtime.optimization_cli import _run_failed

        tm.set_config_manager(config_manager)
        tm.set_schema_loader(FilesystemSchemaLoader(Path("configs/schemas")))
        backend = tm.get_backend()

        org_id = "monthly_rep_fault_org"
        tenant_ids = [f"{org_id}:alpha", f"{org_id}:beta"]
        backend.create_metadata_document(
            schema="organization_metadata",
            doc_id=org_id,
            fields={
                "org_id": org_id,
                "org_name": "monthly-reports-fault",
                "created_at": int(_time.time() * 1000),
                "created_by": "integration-test",
                "status": "active",
                "tenant_count": len(tenant_ids),
            },
        )
        for tid in tenant_ids:
            backend.create_metadata_document(
                schema="tenant_metadata",
                doc_id=tid,
                fields={
                    "tenant_full_id": tid,
                    "org_id": org_id,
                    "tenant_name": tid.split(":", 1)[1],
                    "created_at": int(_time.time() * 1000),
                    "created_by": "integration-test",
                    "status": "active",
                    "schemas_deployed": ["agent_memories"],
                },
            )

        class _Traces:
            async def get_spans(self, **kwargs):
                raise ConnectionError("phoenix unreachable")

        class _Provider:
            traces = _Traces()

        class _Config:
            @staticmethod
            def get_project_name(tid):
                return f"proj-{tid}"

        class _PhoenixDownManager:
            config = _Config()

            def get_provider(self, tenant_id=None):
                return _Provider()

        monkeypatch.setattr(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            lambda: _PhoenixDownManager(),
        )

        try:
            result = await run_monthly_reports(
                output_dir=str(tmp_path / "reports"),
                lookback_hours=1.0,
            )
            assert result.get("failed"), (
                "a Phoenix outage on every tenant must surface at the top level"
            )
            assert sorted(result["failed"]) == sorted(tenant_ids), (
                f"failed must name the errored tenants; got {result.get('failed')!r}"
            )
            assert _run_failed(result) is True, (
                "the cron exit gate must treat a Phoenix outage as a failure"
            )
        finally:
            for tid in tenant_ids:
                try:
                    backend.delete_metadata_document(
                        schema="tenant_metadata", doc_id=tid
                    )
                except Exception:
                    pass
            try:
                backend.delete_metadata_document(
                    schema="organization_metadata", doc_id=org_id
                )
            except Exception:
                pass
