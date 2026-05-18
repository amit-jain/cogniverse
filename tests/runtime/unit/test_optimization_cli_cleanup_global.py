"""Unit tests for ``optimization_cli --mode cleanup`` global-cleanup path.

The daily-cleanup CronWorkflow was exiting 2 because the real CLI
declared ``--tenant-id`` ``required=True`` and the workflow has no
tenant context. These tests pin the contract that cleanup mode runs
globally when ``--tenant-id`` is omitted (enumerates every tenant in
every org and sweeps each one), and that the global iteration
isolates per-tenant failures so one bad tenant doesn't abort the
sweep across the rest.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from cogniverse_runtime.optimization_cli import run_cleanup


class _FakeMem:
    """Records every cleanup invocation across instances."""

    calls: list[tuple[str, int]] = []
    fail_for: set[str] = set()

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    def cleanup(self, retention_days: int) -> None:
        if self.tenant_id in self.fail_for:
            raise RuntimeError(f"boom on {self.tenant_id}")
        type(self).calls.append((self.tenant_id, retention_days))


@pytest.fixture(autouse=True)
def _reset_fake_mem():
    _FakeMem.calls = []
    _FakeMem.fail_for = set()
    yield
    _FakeMem.calls = []
    _FakeMem.fail_for = set()


class TestRunCleanupTenantScoped:
    def test_single_tenant_runs_once_and_returns_completed(self):
        with mock.patch("cogniverse_core.memory.manager.Mem0MemoryManager", _FakeMem):
            result = asyncio.run(
                run_cleanup("acme:prod", log_retention_days=7, memory_retention_days=30)
            )

        assert _FakeMem.calls == [("acme:prod", 30)]
        assert result["memory_cleanup"] == {"acme:prod": "completed"}
        assert result["log_retention_days"] == 7
        assert result["memory_retention_days"] == 30
        assert "tenants_processed" not in result

    def test_single_tenant_failure_is_captured_not_raised(self):
        _FakeMem.fail_for = {"acme:prod"}
        with mock.patch("cogniverse_core.memory.manager.Mem0MemoryManager", _FakeMem):
            result = asyncio.run(
                run_cleanup("acme:prod", log_retention_days=7, memory_retention_days=30)
            )
        assert result["memory_cleanup"] == {"acme:prod": "failed: boom on acme:prod"}


class TestRunCleanupGlobal:
    """tenant_id=None → enumerate orgs → enumerate tenants → cleanup each."""

    def _patches(self, orgs, tenants_per_org):
        async def fake_list_orgs():
            return list(orgs)

        async def fake_list_tenants(org_id):
            return [
                SimpleNamespace(tenant_full_id=tid)
                for tid in tenants_per_org.get(org_id, [])
            ]

        return (
            mock.patch("cogniverse_core.memory.manager.Mem0MemoryManager", _FakeMem),
            mock.patch(
                "cogniverse_runtime.admin.tenant_manager.list_organizations_internal",
                fake_list_orgs,
            ),
            mock.patch(
                "cogniverse_runtime.admin.tenant_manager.list_tenants_for_org_internal",
                fake_list_tenants,
            ),
        )

    def test_iterates_every_tenant_in_every_org(self):
        orgs = ["org_a", "org_b"]
        tenants_per_org = {
            "org_a": ["org_a:t1", "org_a:t2"],
            "org_b": ["org_b:t1"],
        }

        with (
            self._patches(orgs, tenants_per_org)[0],
            self._patches(orgs, tenants_per_org)[1],
            self._patches(orgs, tenants_per_org)[2],
        ):
            result = asyncio.run(
                run_cleanup(None, log_retention_days=7, memory_retention_days=30)
            )

        assert sorted(t for t, _ in _FakeMem.calls) == [
            "org_a:t1",
            "org_a:t2",
            "org_b:t1",
        ]
        assert all(days == 30 for _, days in _FakeMem.calls)
        assert result["tenants_processed"] == 3
        assert result["memory_cleanup"] == {
            "org_a:t1": "completed",
            "org_a:t2": "completed",
            "org_b:t1": "completed",
        }

    def test_per_tenant_failure_does_not_abort_sweep(self):
        orgs = ["org_a"]
        tenants_per_org = {"org_a": ["org_a:good1", "org_a:bad", "org_a:good2"]}
        _FakeMem.fail_for = {"org_a:bad"}

        with (
            self._patches(orgs, tenants_per_org)[0],
            self._patches(orgs, tenants_per_org)[1],
            self._patches(orgs, tenants_per_org)[2],
        ):
            result = asyncio.run(
                run_cleanup(None, log_retention_days=7, memory_retention_days=30)
            )

        assert result["memory_cleanup"]["org_a:good1"] == "completed"
        assert result["memory_cleanup"]["org_a:good2"] == "completed"
        assert result["memory_cleanup"]["org_a:bad"].startswith("failed: ")
        assert result["tenants_processed"] == 3

    def test_no_orgs_returns_empty_summary(self):
        with (
            self._patches([], {})[0],
            self._patches([], {})[1],
            self._patches([], {})[2],
        ):
            result = asyncio.run(
                run_cleanup(None, log_retention_days=7, memory_retention_days=30)
            )
        assert result["memory_cleanup"] == {}
        assert result["tenants_processed"] == 0
        assert _FakeMem.calls == []


class TestCleanupCliExitCode:
    """Pin the exit-code-2 regression: argparse must accept cleanup without --tenant-id."""

    def test_cleanup_without_tenant_id_does_not_exit_2_on_argparse(self):
        """Run the real CLI with --help variant that exercises argparse.

        We invoke the parser directly via subprocess with ``--mode cleanup
        --log-retention-days 1 --memory-retention-days 1`` and mock out
        the actual work via PYTHONPATH-injected stubs would be heavy.
        Instead, this test asserts argparse.parse_args succeeds — the
        narrow regression the prior code missed (argparse returning rc=2
        because --tenant-id was required).
        """
        from cogniverse_runtime import optimization_cli

        # Re-import-safe: build the same parser the CLI builds, then
        # parse a cleanup invocation with no tenant. Prior to the fix
        # this raised SystemExit(2). After the fix, it parses cleanly
        # with tenant_id=None.
        argv = [
            "--mode",
            "cleanup",
            "--log-retention-days",
            "7",
            "--memory-retention-days",
            "30",
        ]
        with mock.patch.object(sys, "argv", ["optimization_cli"] + argv):
            with mock.patch.object(optimization_cli, "asyncio") as fake_asyncio:
                fake_asyncio.run = mock.MagicMock(return_value={"ok": True})
                with mock.patch.object(optimization_cli, "sys") as fake_sys:
                    fake_sys.exit = mock.MagicMock()
                    fake_sys.argv = ["optimization_cli"] + argv
                    optimization_cli.main()
                    fake_sys.exit.assert_called_once_with(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
