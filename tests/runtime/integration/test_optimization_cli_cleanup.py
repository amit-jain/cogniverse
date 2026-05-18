"""Integration tests for ``optimization_cli --mode cleanup`` against the
real Vespa + Mem0 stack.

Pin the contract that the daily-cleanup CronWorkflow exercises end to
end: schema-driven retention is enforced per kind, soft-delete fires at
TTL, hard-delete fires at 2× TTL, ``PERMANENT`` kinds survive
indefinitely. No mocks — the CLI runs against the live ``memory_manager``
fixture (real Mem0, real DenseOn, real LM, real Vespa).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_runtime.optimization_cli import run_cleanup

pytestmark = pytest.mark.integration


def _seed(
    mm: Mem0MemoryManager,
    *,
    kind: str,
    content: str,
    age_days: float = 0.0,
) -> str:
    """Add one memory with deterministic age via metadata.created_at.

    Mem0 stamps ``created_at`` from a C-level clock that ignores Python
    monkeypatching, so the only deterministic age primitive is to pass
    ``created_at`` in metadata at write time — Mem0 honours it. Same
    pattern the schema e2e tests use.
    """
    meta: dict = {"kind": kind}
    if age_days > 0:
        meta["created_at"] = (
            datetime.now(timezone.utc) - timedelta(days=age_days)
        ).isoformat()
    return mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name="cleanup_integration_writer",
        metadata=meta,
        infer=False,
    )


def _resolve(mm: Mem0MemoryManager, mid: str) -> dict | None:
    """Fetch a memory by id directly from the live backend."""
    try:
        return mm.memory.get(memory_id=mid)
    except Exception:
        return None


class TestRunCleanupEnforcesSchemaRetention:
    """The exact retention contract a daily-cleanup workflow must honour."""

    @pytest.mark.asyncio
    async def test_per_tenant_cleanup_keeps_fresh_archives_aging_hard_deletes_stale(
        self, memory_manager
    ):
        """Real Mem0: fresh / soft-delete-window / hard-delete-window /
        PERMANENT each verified by exact post-state — no wiring-only checks.

        ``conversation_turn`` retention is 14 days (per
        ``build_default_registry``). The cleanup hook soft-deletes
        memories whose age is between [TTL, 2×TTL] (flips
        metadata.archived=True), hard-deletes memories whose age is >
        2×TTL, and never touches ``PERMANENT`` kinds regardless of age.
        """
        tenant_id = memory_manager.tenant_id

        fresh_id = _seed(
            memory_manager, kind="conversation_turn", content="fresh-turn", age_days=0
        )
        soft_id = _seed(
            memory_manager, kind="conversation_turn", content="soft-turn", age_days=20
        )
        hard_id = _seed(
            memory_manager, kind="conversation_turn", content="hard-turn", age_days=40
        )
        permanent_id = _seed(
            memory_manager,
            kind="tenant_instruction",
            content="rule-stays-forever",
            age_days=999,
        )

        result = await run_cleanup(
            tenant_id=tenant_id,
            log_retention_days=7,
            memory_retention_days=30,
        )

        # Wiring: cleanup ran for exactly this tenant, reported completion.
        assert tenant_id in result["memory_cleanup"], (
            f"per-tenant cleanup must report this tenant; got {result['memory_cleanup']!r}"
        )
        outcome = result["memory_cleanup"][tenant_id]
        assert outcome.startswith("completed:"), (
            f"per-tenant cleanup must succeed; got {outcome!r}"
        )

        # Outcome: exact survivor set + soft-delete state.
        fresh_doc = _resolve(memory_manager, fresh_id)
        soft_doc = _resolve(memory_manager, soft_id)
        hard_doc = _resolve(memory_manager, hard_id)
        perm_doc = _resolve(memory_manager, permanent_id)

        assert fresh_doc is not None, "fresh conversation_turn must survive"
        assert (fresh_doc.get("metadata") or {}).get("archived") in (None, False), (
            f"fresh memory must not be archived; got {fresh_doc!r}"
        )

        assert soft_doc is not None, (
            "conversation_turn aged 20d (between 14d TTL and 28d 2×TTL) "
            "must soft-delete, not hard-delete"
        )
        assert (soft_doc.get("metadata") or {}).get("archived") is True, (
            f"soft-delete window memory must have metadata.archived=True; "
            f"got {soft_doc!r}"
        )

        assert hard_doc is None, (
            f"conversation_turn aged 40d (past 28d 2×TTL) must hard-delete; "
            f"resolver returned {hard_doc!r}"
        )

        assert perm_doc is not None, (
            "tenant_instruction (PERMANENT) must never be touched by cleanup, "
            "even with age 999d"
        )

    @pytest.mark.asyncio
    async def test_global_cleanup_enforces_schema_retention_for_all_real_tenants(
        self, memory_manager, vespa_instance, config_manager
    ):
        """tenant_id=None: real backend enumeration via tenant_manager
        helpers; cleanup applied per discovered tenant.

        Seeds a real organization + tenant row in the live metadata
        schemas, plants a hard-delete-window memory under that tenant,
        runs the global cleanup, and asserts the memory is gone —
        outcome, not wiring. Pre-fix the daily-cleanup workflow never
        reached this branch (exited 2 in argparse).
        """
        import time as _time

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.admin import tenant_manager as tm

        tm.set_config_manager(config_manager)
        tm.set_schema_loader(FilesystemSchemaLoader(Path("configs/schemas")))
        backend = tm.get_backend()

        org_id = "cli_cleanup_org"
        seeded_full_id = f"{org_id}:cli_cleanup_t"
        backend.create_metadata_document(
            schema="organization_metadata",
            doc_id=org_id,
            fields={
                "org_id": org_id,
                "org_name": "cleanup-integration",
                "created_at": int(_time.time() * 1000),
                "created_by": "integration-test",
                "status": "active",
                "tenant_count": 1,
            },
        )
        backend.create_metadata_document(
            schema="tenant_metadata",
            doc_id=seeded_full_id,
            fields={
                "tenant_full_id": seeded_full_id,
                "org_id": org_id,
                "tenant_name": "cli_cleanup_t",
                "created_at": int(_time.time() * 1000),
                "created_by": "integration-test",
                "status": "active",
                "schemas_deployed": [],
            },
        )

        # Plant a hard-deletable memory under the seeded tenant via a
        # live Mem0 instance — the module-scoped ``memory_manager``
        # belongs to a different tenant ("test:unit").
        from tests.utils.llm_config import get_llm_base_url, get_llm_model

        sys_cfg = config_manager.get_system_config()
        embedder_base_url = sys_cfg.inference_service_urls.get("denseon")
        assert embedder_base_url, (
            "config_manager fixture must seed inference_service_urls['denseon']"
        )

        seeded_mm = Mem0MemoryManager(tenant_id=seeded_full_id)
        seeded_mm.initialize(
            backend_host="http://localhost",
            backend_port=vespa_instance["http_port"],
            backend_config_port=vespa_instance["config_port"],
            llm_model=get_llm_model(),
            embedding_model="lightonai/DenseOn",
            llm_base_url=get_llm_base_url(),
            embedder_base_url=embedder_base_url,
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            auto_create_schema=True,
        )
        hard_id = _seed(
            seeded_mm,
            kind="conversation_turn",
            content="global-cleanup-victim",
            age_days=40,
        )

        try:
            result = await run_cleanup(
                tenant_id=None,
                log_retention_days=1,
                memory_retention_days=1,
            )

            assert result["tenants_processed"] == len(result["memory_cleanup"])
            assert seeded_full_id in result["memory_cleanup"], (
                f"global cleanup must include the seeded tenant "
                f"({seeded_full_id!r}); got {sorted(result['memory_cleanup'])!r}"
            )
            assert result["memory_cleanup"][seeded_full_id].startswith("completed:"), (
                "global cleanup must succeed on the seeded tenant; "
                f"got {result['memory_cleanup'][seeded_full_id]!r}"
            )

            # Real outcome: the hard-delete-window memory is gone.
            assert _resolve(seeded_mm, hard_id) is None, (
                "global cleanup must have hard-deleted the 40d-old "
                "conversation_turn under the seeded tenant"
            )
        finally:
            try:
                backend.delete_metadata_document(
                    schema="tenant_metadata", doc_id=seeded_full_id
                )
            except Exception:
                pass
            try:
                backend.delete_metadata_document(
                    schema="organization_metadata", doc_id=org_id
                )
            except Exception:
                pass


class TestRunCleanupCoversLogsTempAndConfigVacuum:
    """The daily-cleanup workflow absorbed log rotation, temp file purge,
    and config_metadata version vacuum from the standalone daily-cleanup
    cron the chart didn't previously cover. These tests pin each section
    against real filesystem + real Vespa."""

    @pytest.mark.asyncio
    async def test_log_cleanup_deletes_files_older_than_retention(
        self, memory_manager, tmp_path, monkeypatch
    ):
        """Files older than ``log_retention_days`` are removed; fresh files survive.

        Real filesystem, real os.unlink — no mock of pathlib or os. The
        test seeds two files with deterministic mtimes via os.utime,
        runs cleanup, then asserts exact survivor set on disk.
        """
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        stale = log_dir / "stale.log"
        fresh = log_dir / "fresh.log"
        stale.write_text("stale entry")
        fresh.write_text("fresh entry")

        # Backdate stale to 10 days ago; fresh stays "now".
        ten_days_ago = time.time() - 10 * 86400
        os.utime(stale, (ten_days_ago, ten_days_ago))

        monkeypatch.setenv("LOG_DIR", str(log_dir))
        # Disable temp/vacuum so this test only asserts log behaviour.
        monkeypatch.setenv("TEMP_DIR", str(tmp_path / "nonexistent_temp"))
        monkeypatch.setenv("CONFIG_KEEP_VERSIONS", "10")

        result = await run_cleanup(
            tenant_id=memory_manager.tenant_id,
            log_retention_days=7,
            memory_retention_days=30,
        )

        assert result["log_cleanup"]["path"] == str(log_dir)
        assert result["log_cleanup"]["scanned"] == 2
        assert result["log_cleanup"]["deleted"] == 1
        assert result["log_cleanup"]["errors"] == []
        assert not stale.exists(), "stale.log (10d old) must be deleted"
        assert fresh.exists(), "fresh.log must survive"

    @pytest.mark.asyncio
    async def test_log_cleanup_skips_missing_directory(
        self, memory_manager, tmp_path, monkeypatch
    ):
        """LOG_DIR pointing at a non-existent path is a logged skip, not a failure."""
        missing = tmp_path / "no_logs_here"
        monkeypatch.setenv("LOG_DIR", str(missing))
        monkeypatch.setenv("TEMP_DIR", str(tmp_path / "no_tmp_here"))

        result = await run_cleanup(
            tenant_id=memory_manager.tenant_id,
            log_retention_days=7,
            memory_retention_days=30,
        )

        assert "skipped" in result["log_cleanup"], (
            f"missing log dir must report skipped, not crash; "
            f"got {result['log_cleanup']!r}"
        )
        assert result["log_cleanup"]["scanned"] == 0
        assert result["log_cleanup"]["deleted"] == 0

    @pytest.mark.asyncio
    async def test_temp_cleanup_deletes_old_temp_files(
        self, memory_manager, tmp_path, monkeypatch
    ):
        """TEMP_DIR sweep removes files older than TEMP_RETENTION_DAYS."""
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        old_tmp = temp_dir / "old.tmp"
        new_tmp = temp_dir / "new.tmp"
        old_tmp.write_text("x")
        new_tmp.write_text("y")
        two_days_ago = time.time() - 2 * 86400
        os.utime(old_tmp, (two_days_ago, two_days_ago))

        monkeypatch.setenv("LOG_DIR", str(tmp_path / "no_logs"))
        monkeypatch.setenv("TEMP_DIR", str(temp_dir))
        monkeypatch.setenv("TEMP_RETENTION_DAYS", "1")

        result = await run_cleanup(
            tenant_id=memory_manager.tenant_id,
            log_retention_days=7,
            memory_retention_days=30,
        )

        assert result["temp_cleanup"]["path"] == str(temp_dir)
        assert result["temp_cleanup"]["scanned"] == 2
        assert result["temp_cleanup"]["deleted"] == 1
        assert not old_tmp.exists()
        assert new_tmp.exists()

    @pytest.mark.asyncio
    async def test_config_vacuum_prunes_excess_versions(
        self, memory_manager, vespa_instance, monkeypatch
    ):
        """The config_vacuum section drives ``VespaConfigStore.prune_all_configs``.

        Pre-seed N=15 versions of one config_id with keep_versions=5,
        run cleanup, assert (a) the vacuum section reports the
        dropped count >= 10, and (b) the on-disk version count for
        that config_id is now <= 5.
        """
        from cogniverse_sdk.interfaces.config_store import ConfigScope
        from cogniverse_vespa.config.config_store import VespaConfigStore

        store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            keep_versions=100,  # write all 15 without per-write pruning
        )
        for i in range(1, 16):
            store.set_config(
                tenant_id="vac_test",
                scope=ConfigScope.BACKEND,
                service="vacuum_probe",
                config_key="kv",
                config_value={"i": i},
            )

        config_id = store._create_document_id(
            "vac_test", ConfigScope.BACKEND, "vacuum_probe", "kv"
        )
        pre_resp = store.vespa_app.query(
            yql=(
                f"select version from config_metadata "
                f'where config_id contains "{config_id}" '
                f"order by version desc limit 100"
            )
        )
        pre_count = len(list(pre_resp.hits or []))
        assert pre_count == 15, (
            f"setup precondition: expected 15 versions seeded, found {pre_count}"
        )

        monkeypatch.setenv("LOG_DIR", "/no_logs_here_either")
        monkeypatch.setenv("TEMP_DIR", "/no_tmp_here_either")
        monkeypatch.setenv("CONFIG_KEEP_VERSIONS", "5")

        try:
            result = await run_cleanup(
                tenant_id=memory_manager.tenant_id,
                log_retention_days=7,
                memory_retention_days=30,
            )

            vacuum = result["config_vacuum"]
            assert vacuum.get("keep_versions") == 5, (
                f"vacuum should respect CONFIG_KEEP_VERSIONS=5; got {vacuum!r}"
            )
            assert vacuum.get("dropped", 0) >= 10, (
                f"15-5=10 stale rows must be dropped; got {vacuum!r}"
            )

            post_resp = store.vespa_app.query(
                yql=(
                    f"select version from config_metadata "
                    f'where config_id contains "{config_id}" '
                    f"order by version desc limit 100"
                )
            )
            surviving = sorted(h["fields"]["version"] for h in post_resp.hits or [])
            assert surviving == [11, 12, 13, 14, 15], (
                f"vacuum must retain exactly the latest 5 versions; got {surviving!r}"
            )
        finally:
            store.delete_config(
                tenant_id="vac_test",
                scope=ConfigScope.BACKEND,
                service="vacuum_probe",
                config_key="kv",
            )
