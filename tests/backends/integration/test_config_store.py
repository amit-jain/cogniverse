"""Integration tests for ``VespaConfigStore`` against a real Vespa instance.

The read path uses the Document v1 visit API (read-after-write consistent
with feeds) rather than a YQL ``/search/`` query (eventually consistent),
so cross-process schema_registry writes become visible immediately on the
next read. Version allocation feeds with a conditional write so two
concurrent writers can never persist the same version number. These tests
pin both contracts against a real Vespa instance.
"""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

import cogniverse_vespa.config.config_store as config_store_module
from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_config_store(vespa_instance):
    """``VespaConfigStore`` against the module's Vespa instance.

    The shared ``vespa_instance`` fixture (in conftest.py) deploys the
    metadata schemas before yielding, so the ``config_metadata`` schema
    is already up by the time we connect.
    """
    return VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )


@pytest.mark.integration
@pytest.mark.ci_fast
class TestVespaConfigStoreListAllConfigs:
    """Pins read-after-write semantics on ``list_all_configs``.

    Vespa's ``/search/`` indexing is eventually consistent — a freshly
    fed document can take hundreds of milliseconds to surface in YQL
    results. The Document v1 visit endpoint is consistent with the
    feed; writes made via ``set_config`` (which feeds via Document v1)
    are visible on the next visit immediately.
    """

    def test_set_then_list_returns_new_entry_without_delay(self, vespa_config_store):
        """Feed a config, immediately list, find it. No sleep."""
        store = vespa_config_store
        entry = store.set_config(
            tenant_id="cs_rw_a",
            scope=ConfigScope.BACKEND,
            service="probe",
            config_key="rw_marker",
            config_value={"hello": "world"},
        )
        try:
            results = store.list_all_configs(scope=ConfigScope.BACKEND, service="probe")
            matches = [r for r in results if r.tenant_id == "cs_rw_a"]
            assert matches, (
                "list_all_configs did not see a config that set_config just "
                "wrote — read-after-write contract broken (the symptom that "
                "drove the YQL→Document v1 visit switch)"
            )
            assert matches[0].config_value == {"hello": "world"}
            assert matches[0].version == entry.version
        finally:
            store.delete_config(
                tenant_id="cs_rw_a",
                scope=ConfigScope.BACKEND,
                service="probe",
                config_key="rw_marker",
            )

    def test_set_then_get_and_filtered_list_are_immediately_consistent(
        self, vespa_config_store
    ):
        store = vespa_config_store
        tenant, service, key = "cs_rw_point_a", "point_probe", "exact_key"
        first = store.set_config(
            tenant_id=tenant,
            scope=ConfigScope.SYSTEM,
            service=service,
            config_key=key,
            config_value={"revision": 1},
        )
        second = store.set_config(
            tenant_id=tenant,
            scope=ConfigScope.SYSTEM,
            service=service,
            config_key=key,
            config_value={"revision": 2},
        )
        try:
            latest = store.get_config(
                tenant,
                ConfigScope.SYSTEM,
                service,
                key,
            )
            version_one = store.get_config(
                tenant,
                ConfigScope.SYSTEM,
                service,
                key,
                version=1,
            )
            listed = store.list_configs(
                tenant,
                scope=ConfigScope.SYSTEM,
                service=service,
            )

            assert (latest.version, latest.config_value) == (
                second.version,
                {"revision": 2},
            )
            assert (version_one.version, version_one.config_value) == (
                first.version,
                {"revision": 1},
            )
            assert [
                (entry.config_key, entry.version, entry.config_value)
                for entry in listed
            ] == [(key, second.version, {"revision": 2})]
        finally:
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.SYSTEM,
                service=service,
                config_key=key,
            )

    def test_point_and_filtered_reads_propagate_visit_timeout(
        self, vespa_config_store, monkeypatch
    ):
        expected_message = (
            "Failed to read Vespa config visit after 5 attempts over 3.751s: "
            "Timeout: Document visit timed out"
        )
        monotonic_values = iter([100.0, 103.751, 200.0, 203.751])
        monkeypatch.setattr(
            config_store_module.time, "monotonic", lambda: next(monotonic_values)
        )
        monkeypatch.setattr(config_store_module.time, "sleep", lambda *_: None)
        monkeypatch.setattr(
            requests,
            "get",
            Mock(side_effect=requests.Timeout("Document visit timed out")),
        )

        with pytest.raises(RuntimeError) as error:
            vespa_config_store.get_config(
                "cs_timeout_a",
                ConfigScope.SYSTEM,
                "runtime",
                "key",
            )
        assert str(error.value) == expected_message
        assert requests.get.call_count == 5

        with pytest.raises(RuntimeError) as error:
            vespa_config_store.list_configs(
                "cs_timeout_a",
                scope=ConfigScope.SYSTEM,
                service="runtime",
            )
        assert str(error.value) == expected_message
        assert requests.get.call_count == 10

    def test_concurrent_writers_allocate_distinct_versions(self, vespa_config_store):
        store = vespa_config_store
        tenant, service, key = "cs_concurrent_a", "version_probe", "same_key"
        writer_count = 8
        barrier = threading.Barrier(writer_count)
        store.delete_config(
            tenant_id=tenant,
            scope=ConfigScope.BACKEND,
            service=service,
            config_key=key,
        )

        def write(writer: int):
            barrier.wait()
            return store.set_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
                config_value={"writer": writer},
            )

        try:
            with ThreadPoolExecutor(max_workers=writer_count) as executor:
                entries = list(executor.map(write, range(writer_count)))

            history = store.get_config_history(
                tenant,
                ConfigScope.BACKEND,
                service,
                key,
            )
            assert sorted(entry.version for entry in entries) == list(
                range(1, writer_count + 1)
            )
            assert [entry.version for entry in history] == list(
                range(writer_count, 0, -1)
            )
            assert {entry.config_value["writer"] for entry in history} == set(
                range(writer_count)
            )
        finally:
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
            )

    def test_delete_config_raises_after_a_mid_delete_failure(
        self, vespa_config_store, monkeypatch
    ):
        store = vespa_config_store
        tenant, service, key = "cs_delete_fault_a", "delete_probe", "same_key"
        entries = [
            store.set_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
                config_value={"revision": revision},
            )
            for revision in range(1, 4)
        ]
        config_id = store._create_document_id(tenant, ConfigScope.BACKEND, service, key)
        failed_id = f"{store.schema_name}::{config_id}::2"
        original_delete = store.vespa_app.delete_data
        attempted_ids = []

        def fail_second_version(*, schema, data_id):
            attempted_ids.append(data_id)
            if data_id == failed_id:
                raise ConnectionError("Vespa disconnected during config deletion")
            return original_delete(schema=schema, data_id=data_id)

        monkeypatch.setattr(store.vespa_app, "delete_data", fail_second_version)
        try:
            with pytest.raises(
                RuntimeError,
                match="Failed to delete 1 of 3 versions.*version 2",
            ):
                store.delete_config(
                    tenant_id=tenant,
                    scope=ConfigScope.BACKEND,
                    service=service,
                    config_key=key,
                )

            assert attempted_ids == [
                f"{store.schema_name}::{config_id}::{entry.version}"
                for entry in entries[::-1]
            ]
            for entry in entries:
                response = store.vespa_app.get_data(
                    schema=store.schema_name,
                    data_id=f"{store.schema_name}::{config_id}::{entry.version}",
                    raise_on_not_found=False,
                )
                assert response.status_code == (200 if entry.version == 2 else 404)
        finally:
            monkeypatch.setattr(store.vespa_app, "delete_data", original_delete)
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
            )

    def test_set_config_raises_on_version_query_failure_preserving_v1(
        self, vespa_config_store
    ):
        """A Vespa query-API outage during set_config must NOT be flattened to
        version 0 — that treats a live config as brand-new and rewrites its v1
        row. Seed v1..v3, break the query API, assert the write raises and every
        version is intact."""
        store = vespa_config_store
        tenant, service, key = "cs_verr_a", "verr_probe", "k1"

        store.delete_config(
            tenant_id=tenant,
            scope=ConfigScope.BACKEND,
            service=service,
            config_key=key,
        )

        for i in (1, 2, 3):
            store.set_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
                config_value={"seed": i},
            )
        try:
            assert store.get_config(
                tenant, ConfigScope.BACKEND, service, key, version=1
            ).config_value == {"seed": 1}
            assert (
                store.get_config(tenant, ConfigScope.BACKEND, service, key).version == 3
            )

            # Inject a query-API outage during the write; the feed path and all
            # verification reads stay on the real Vespa.
            real_query = store.vespa_app.query

            def boom(*args, **kwargs):
                raise ConnectionError("simulated Vespa query outage")

            store.vespa_app.query = boom
            try:
                with pytest.raises(ConnectionError):
                    store.set_config(
                        tenant_id=tenant,
                        scope=ConfigScope.BACKEND,
                        service=service,
                        config_key=key,
                        config_value={"clobber": True},
                    )
            finally:
                store.vespa_app.query = real_query

            # v1 untouched, latest unchanged, no spurious/rewritten row.
            assert store.get_config(
                tenant, ConfigScope.BACKEND, service, key, version=1
            ).config_value == {"seed": 1}
            latest = store.get_config(tenant, ConfigScope.BACKEND, service, key)
            assert latest.version == 3
            assert latest.config_value == {"seed": 3}
            assert (
                len(store.get_config_history(tenant, ConfigScope.BACKEND, service, key))
                == 3
            )
        finally:
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
            )

    def test_scope_filter_excludes_other_scopes(self, vespa_config_store):
        """``scope=`` arg is enforced server-side via the selection clause."""
        store = vespa_config_store
        store.set_config(
            tenant_id="cs_scope_a",
            scope=ConfigScope.BACKEND,
            service="filter_probe",
            config_key="tenant_entry",
            config_value={"k": 1},
        )
        store.set_config(
            tenant_id="cs_scope_a",
            scope=ConfigScope.SYSTEM,
            service="filter_probe",
            config_key="system_entry",
            config_value={"k": 2},
        )
        try:
            tenant_results = store.list_all_configs(
                scope=ConfigScope.BACKEND, service="filter_probe"
            )
            tenant_keys = {r.config_key for r in tenant_results}
            assert "tenant_entry" in tenant_keys
            assert "system_entry" not in tenant_keys, (
                "scope filter did not exclude SYSTEM-scoped entry — selection "
                "clause is not being applied"
            )
        finally:
            store.delete_config(
                tenant_id="cs_scope_a",
                scope=ConfigScope.BACKEND,
                service="filter_probe",
                config_key="tenant_entry",
            )
            store.delete_config(
                tenant_id="cs_scope_a",
                scope=ConfigScope.SYSTEM,
                service="filter_probe",
                config_key="system_entry",
            )

    def test_service_filter_excludes_other_services(self, vespa_config_store):
        """``service=`` arg is enforced server-side via the selection clause."""
        store = vespa_config_store
        store.set_config(
            tenant_id="cs_svc_a",
            scope=ConfigScope.BACKEND,
            service="svc_in",
            config_key="x",
            config_value={"k": 1},
        )
        store.set_config(
            tenant_id="cs_svc_a",
            scope=ConfigScope.BACKEND,
            service="svc_out",
            config_key="x",
            config_value={"k": 2},
        )
        try:
            results = store.list_all_configs(
                scope=ConfigScope.BACKEND, service="svc_in"
            )
            services = {r.service for r in results}
            assert "svc_in" in services
            assert "svc_out" not in services, (
                "service filter did not exclude svc_out — selection clause "
                "is not being applied"
            )
        finally:
            store.delete_config(
                tenant_id="cs_svc_a",
                scope=ConfigScope.BACKEND,
                service="svc_in",
                config_key="x",
            )
            store.delete_config(
                tenant_id="cs_svc_a",
                scope=ConfigScope.BACKEND,
                service="svc_out",
                config_key="x",
            )

    def test_set_config_prunes_to_keep_versions_window(self, vespa_instance):
        """``set_config`` retains exactly ``keep_versions`` rows per config_id.

        Pins the bound that fixes the config_metadata bloat (~5800 rows
        observed in dev after a few days of e2e churn). Writes
        ``keep+overflow`` versions, then asserts the surviving version
        set is exactly ``[keep+overflow-keep+1 .. keep+overflow]`` and
        the row count is exactly ``keep``.
        """
        keep = 3
        overflow = 5
        store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            keep_versions=keep,
        )
        tenant = "cs_prune_a"
        try:
            written = []
            for i in range(1, keep + overflow + 1):
                entry = store.set_config(
                    tenant_id=tenant,
                    scope=ConfigScope.BACKEND,
                    service="prune_probe",
                    config_key="k1",
                    config_value={"i": i},
                )
                written.append(entry.version)

            assert written == list(range(1, keep + overflow + 1)), (
                f"set_config must monotonically increment version per key; "
                f"got {written}"
            )

            config_id = store._create_document_id(
                tenant, ConfigScope.BACKEND, "prune_probe", "k1"
            )
            response = store.vespa_app.query(
                yql=(
                    f"select version from config_metadata "
                    f'where config_id contains "{config_id}" '
                    f"order by version desc limit 100"
                )
            )
            surviving = sorted(h["fields"]["version"] for h in response.hits)
            expected = list(range(overflow + 1, keep + overflow + 1))
            assert surviving == expected, (
                f"pruning did not retain exactly the latest {keep} versions; "
                f"expected {expected}, got {surviving}"
            )

            latest = store.get_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service="prune_probe",
                config_key="k1",
            )
            assert latest is not None
            assert latest.version == keep + overflow
            assert latest.config_value == {"i": keep + overflow}
        finally:
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service="prune_probe",
                config_key="k1",
            )

    def test_set_config_does_not_prune_below_keep_window(self, vespa_instance):
        """Fewer than ``keep_versions`` writes → no rows pruned."""
        store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            keep_versions=10,
        )
        tenant = "cs_prune_b"
        try:
            for i in range(1, 4):
                store.set_config(
                    tenant_id=tenant,
                    scope=ConfigScope.BACKEND,
                    service="prune_probe",
                    config_key="k_small",
                    config_value={"i": i},
                )
            config_id = store._create_document_id(
                tenant, ConfigScope.BACKEND, "prune_probe", "k_small"
            )
            response = store.vespa_app.query(
                yql=(
                    f"select version from config_metadata "
                    f'where config_id contains "{config_id}" '
                    f"order by version desc limit 100"
                )
            )
            surviving = sorted(h["fields"]["version"] for h in response.hits)
            assert surviving == [1, 2, 3], (
                f"writes below keep_versions must not prune; got surviving={surviving}"
            )
        finally:
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service="prune_probe",
                config_key="k_small",
            )

    def test_returns_only_latest_version(self, vespa_config_store):
        """Multiple writes to the same key — list returns only the latest."""
        store = vespa_config_store
        store.set_config(
            tenant_id="cs_ver_a",
            scope=ConfigScope.BACKEND,
            service="ver_probe",
            config_key="versioned",
            config_value={"v": 1},
        )
        v2 = store.set_config(
            tenant_id="cs_ver_a",
            scope=ConfigScope.BACKEND,
            service="ver_probe",
            config_key="versioned",
            config_value={"v": 2},
        )
        try:
            results = store.list_all_configs(
                scope=ConfigScope.BACKEND, service="ver_probe"
            )
            matches = [r for r in results if r.tenant_id == "cs_ver_a"]
            assert len(matches) == 1, (
                f"list_all_configs returned multiple versions for one key: "
                f"{[(r.config_key, r.version) for r in matches]}"
            )
            assert matches[0].version == v2.version
            assert matches[0].config_value == {"v": 2}
        finally:
            store.delete_config(
                tenant_id="cs_ver_a",
                scope=ConfigScope.BACKEND,
                service="ver_probe",
                config_key="versioned",
            )


class TestPersistentSession:
    def test_store_ops_create_no_new_sync_sessions(self, vespa_config_store):
        """Config CRUD reuses the store's persistent session — pyvespa's
        default per-op VespaSync costs a fresh TCP(+TLS) handshake per
        operation, multiplied across every config cache-miss read and
        metadata write."""
        from unittest.mock import patch

        from vespa.application import VespaSync

        store = vespa_config_store
        with patch("vespa.application.VespaSync", wraps=VespaSync) as spy:
            store.set_config(
                tenant_id="cs_sess:cs_sess",
                scope=ConfigScope.SYSTEM,
                service="svc",
                config_key="sess_k",
                config_value={"v": 1},
            )
            entry = store.get_config(
                tenant_id="cs_sess:cs_sess",
                scope=ConfigScope.SYSTEM,
                service="svc",
                config_key="sess_k",
            )
            history = store.get_config_history(
                tenant_id="cs_sess:cs_sess",
                scope=ConfigScope.SYSTEM,
                service="svc",
                config_key="sess_k",
            )

        assert entry is not None and entry.config_value == {"v": 1}
        assert len(history) == 1
        assert spy.call_count == 0


@pytest.mark.integration
class TestExportImportRoundTrip:
    """export_configs → import_configs round-trips real configs through real
    Vespa — the dashboard's backup/restore path had zero test reach (and
    ConfigManager.export_configs is a DIFFERENT implementation, so this store
    pair was never exercised anywhere)."""

    def test_export_then_import_onto_a_new_tenant(self, vespa_config_store):
        import time
        import uuid

        store = vespa_config_store
        src = f"exp_src_{uuid.uuid4().hex[:6]}"
        dst = f"exp_dst_{uuid.uuid4().hex[:6]}"

        store.set_config(
            tenant_id=src,
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="feature_flags",
            config_value={"beta": True, "limit": 5},
        )
        store.set_config(
            tenant_id=src,
            scope=ConfigScope.AGENT,
            service="summarizer_agent",
            config_key="agent_config",
            config_value={"thinking_enabled": False},
        )
        time.sleep(1)  # visibility

        exported = store.export_configs(src)
        assert exported["tenant_id"] == src
        assert exported["include_history"] is False
        by_key = {c["config_key"]: c for c in exported["configs"]}
        assert by_key["feature_flags"]["config_value"] == {"beta": True, "limit": 5}
        assert by_key["feature_flags"]["scope"] == "system"
        assert by_key["agent_config"]["config_value"] == {"thinking_enabled": False}
        assert by_key["agent_config"]["service"] == "summarizer_agent"

        imported = store.import_configs(dst, exported)
        assert imported == 2
        time.sleep(1)

        restored = store.get_config(
            tenant_id=dst,
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="feature_flags",
        )
        assert restored is not None
        assert restored.config_value == {"beta": True, "limit": 5}
        restored_agent = store.get_config(
            tenant_id=dst,
            scope=ConfigScope.AGENT,
            service="summarizer_agent",
            config_key="agent_config",
        )
        assert restored_agent.config_value == {"thinking_enabled": False}

    @staticmethod
    def _fault_rows() -> dict:
        return {
            "configs": [
                {
                    "scope": "system",
                    "service": "runtime",
                    "config_key": key,
                    "config_value": {"position": position},
                }
                for position, key in enumerate(("first", "second", "third"), start=1)
            ]
        }

    @staticmethod
    def _feeds_key(method: str, path: str, body: bytes, key: str) -> bool:
        return (
            method == "POST"
            and path.startswith("/document/v1/")
            and json.loads(body)["fields"]["config_key"] == key
        )

    @staticmethod
    def _history(store, tenant: str, key: str) -> list:
        return [
            (entry.version, entry.config_value)
            for entry in store.get_config_history(
                tenant_id=tenant,
                scope=ConfigScope.SYSTEM,
                service="runtime",
                config_key=key,
            )
        ]

    @staticmethod
    def _delete_tenant(store, tenant: str) -> None:
        for key in ("first", "second", "third"):
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.SYSTEM,
                service="runtime",
                config_key=key,
            )

    def test_a_store_failure_mid_import_removes_every_version_it_wrote(
        self, vespa_instance, vespa_config_store
    ):
        from tests.utils.http_fault_proxy import InterceptFaultProxy

        store = vespa_config_store
        tenant = f"exp_fault_{uuid.uuid4().hex[:6]}"
        store.set_config(
            tenant_id=tenant,
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="first",
            config_value={"position": 0},
        )

        def refuse_third(method, path, body):
            if self._feeds_key(method, path, body, "third"):
                return 500, {"message": "injected feed failure"}
            return None

        try:
            with InterceptFaultProxy(
                f"http://localhost:{vespa_instance['http_port']}", refuse_third
            ) as proxy:
                importer = VespaConfigStore(
                    backend_url="http://127.0.0.1",
                    backend_port=proxy.port,
                    keep_versions=1,
                )
                try:
                    with pytest.raises(RuntimeError) as raised:
                        importer.import_configs(tenant, self._fault_rows())
                finally:
                    importer.close()

            assert str(raised.value) == (
                f"Configuration import for tenant {tenant} failed at row 3 of 3 "
                "(runtime/third): injected feed failure; removed 2 of the 2 "
                "versions it had written"
            )
            assert type(raised.value.__cause__).__name__ == "VespaError"
            # The two written versions are deleted newest first, and nothing
            # was pruned before the import failed.
            assert [
                path.split("/docid/", 1)[1]
                for method, path, _body in proxy.requests
                if method == "DELETE"
            ] == [
                f"config_metadata%3A%3A{tenant}%3Asystem%3Aruntime%3Asecond%3A%3A1",
                f"config_metadata%3A%3A{tenant}%3Asystem%3Aruntime%3Afirst%3A%3A2",
            ]
            # The tenant holds exactly what it held before the import.
            assert self._history(store, tenant, "first") == [(1, {"position": 0})]
            assert self._history(store, tenant, "second") == []
            assert self._history(store, tenant, "third") == []
        finally:
            self._delete_tenant(store, tenant)

    def test_versions_a_failed_rollback_leaves_are_named(
        self, vespa_instance, vespa_config_store
    ):
        from tests.utils.http_fault_proxy import InterceptFaultProxy

        store = vespa_config_store
        tenant = f"exp_stuck_{uuid.uuid4().hex[:6]}"

        def refuse_third_and_deletes(method, path, body):
            if method == "DELETE" or self._feeds_key(method, path, body, "third"):
                return 500, {"message": "injected storage failure"}
            return None

        try:
            with InterceptFaultProxy(
                f"http://localhost:{vespa_instance['http_port']}",
                refuse_third_and_deletes,
            ) as proxy:
                importer = VespaConfigStore(
                    backend_url="http://127.0.0.1", backend_port=proxy.port
                )
                try:
                    with pytest.raises(RuntimeError) as raised:
                        importer.import_configs(tenant, self._fault_rows())
                finally:
                    importer.close()

            assert str(raised.value) == (
                f"Configuration import for tenant {tenant} failed at row 3 of 3 "
                "(runtime/third): injected storage failure; removed 0 of the 2 "
                "versions it had written; still stored: runtime/second v1: "
                "injected storage failure; runtime/first v1: injected storage failure"
            )
            assert [
                method for method, _path, _body in proxy.requests if method == "DELETE"
            ] == ["DELETE", "DELETE"]
            # The versions the message names are the versions still stored.
            assert self._history(store, tenant, "first") == [(1, {"position": 1})]
            assert self._history(store, tenant, "second") == [(1, {"position": 2})]
        finally:
            self._delete_tenant(store, tenant)

    def test_a_rollback_removes_only_the_imports_own_versions(
        self, vespa_instance, vespa_config_store
    ):
        """A writer that lands a version of the same key while the import is
        in flight keeps it: the rollback deletes the import's versions by
        number, never the key."""
        from tests.utils.http_fault_proxy import InterceptFaultProxy

        store = vespa_config_store
        tenant = f"exp_race_{uuid.uuid4().hex[:6]}"
        store.set_config(
            tenant_id=tenant,
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="first",
            config_value={"position": 0},
        )

        def concurrent_write_then_refuse_third(method, path, body):
            if self._feeds_key(method, path, body, "third"):
                store.set_config(
                    tenant_id=tenant,
                    scope=ConfigScope.SYSTEM,
                    service="runtime",
                    config_key="first",
                    config_value={"concurrent": True},
                )
                return 500, {"message": "injected feed failure"}
            return None

        try:
            with InterceptFaultProxy(
                f"http://localhost:{vespa_instance['http_port']}",
                concurrent_write_then_refuse_third,
            ) as proxy:
                importer = VespaConfigStore(
                    backend_url="http://127.0.0.1", backend_port=proxy.port
                )
                try:
                    with pytest.raises(RuntimeError) as raised:
                        importer.import_configs(tenant, self._fault_rows())
                finally:
                    importer.close()

            assert str(raised.value) == (
                f"Configuration import for tenant {tenant} failed at row 3 of 3 "
                "(runtime/third): injected feed failure; removed 2 of the 2 "
                "versions it had written"
            )
            assert self._history(store, tenant, "first") == [
                (3, {"concurrent": True}),
                (1, {"position": 0}),
            ]
            assert self._history(store, tenant, "second") == []
        finally:
            self._delete_tenant(store, tenant)

    def test_a_complete_import_prunes_only_after_every_row_is_written(
        self, vespa_instance, vespa_config_store
    ):
        from tests.utils.http_fault_proxy import InterceptFaultProxy

        store = vespa_config_store
        tenant = f"exp_prune_{uuid.uuid4().hex[:6]}"
        store.set_config(
            tenant_id=tenant,
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="first",
            config_value={"position": 0},
        )

        try:
            with InterceptFaultProxy(
                f"http://localhost:{vespa_instance['http_port']}"
            ) as proxy:
                importer = VespaConfigStore(
                    backend_url="http://127.0.0.1",
                    backend_port=proxy.port,
                    keep_versions=1,
                )
                try:
                    assert importer.import_configs(tenant, self._fault_rows()) == 3
                finally:
                    importer.close()

            methods = [
                (method, path.startswith("/document/v1/"))
                for method, path, _body in proxy.requests
            ]
            feeds = [i for i, call in enumerate(methods) if call == ("POST", True)]
            deletes = [i for i, call in enumerate(methods) if call[0] == "DELETE"]
            assert (len(feeds), len(deletes)) == (3, 1), methods
            assert max(feeds) < min(deletes), methods
            assert self._history(store, tenant, "first") == [(2, {"position": 1})]
            assert self._history(store, tenant, "third") == [(1, {"position": 3})]
        finally:
            self._delete_tenant(store, tenant)

    def test_export_with_history_returns_all_versions(self, vespa_config_store):
        import time
        import uuid

        store = vespa_config_store
        tenant = f"exp_hist_{uuid.uuid4().hex[:6]}"
        for i in range(3):
            store.set_config(
                tenant_id=tenant,
                scope=ConfigScope.SYSTEM,
                service="runtime",
                config_key="versioned_key",
                config_value={"rev": i},
            )
        time.sleep(1)

        exported = store.export_configs(tenant, include_history=True)
        assert exported["include_history"] is True
        versions = sorted(
            c["version"]
            for c in exported["configs"]
            if c["config_key"] == "versioned_key"
        )
        assert versions == [1, 2, 3]
        values = {c["version"]: c["config_value"] for c in exported["configs"]}
        assert values[3] == {"rev": 2}


@pytest.mark.integration
class TestCountVersionRows:
    def test_counts_every_version_row_per_config_id(self, vespa_config_store):
        """count_version_rows sees ALL version rows (visit), not just latest —
        the prune dry-run needs the full per-id row counts."""
        import time
        import uuid

        store = vespa_config_store
        tenant = f"vc_{uuid.uuid4().hex[:6]}"
        for i in range(3):
            store.set_config(
                tenant_id=tenant,
                scope=ConfigScope.SYSTEM,
                service="runtime",
                config_key="multi",
                config_value={"rev": i},
            )
        store.set_config(
            tenant_id=tenant,
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="single",
            config_value={"rev": 0},
        )
        time.sleep(1)

        counts = store.count_version_rows()

        multi_id = f"{tenant}:system:runtime:multi"
        single_id = f"{tenant}:system:runtime:single"
        assert counts[multi_id] == 3
        assert counts[single_id] == 1


@pytest.fixture(scope="module")
def export_history_corpus(vespa_config_store):
    """Retained configurations and a foreign tenant with overlapping keys."""
    from vespa.application import Vespa

    store = vespa_config_store
    rows = json.loads(
        (Path(__file__).parents[1] / "fixtures/config_history.json").read_text()
    )
    assert len(rows) == 413
    # The recording carries exactly the fields the production writer stores, so
    # a renamed or dropped ConfigEntry field makes this corpus go red instead of
    # quietly exporting a shape nothing writes any more.
    written_fields = frozenset(
        ConfigEntry(
            tenant_id="t",
            scope=ConfigScope.SYSTEM,
            service="s",
            config_key="k",
            config_value={},
            version=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ).to_dict()
    ) | {"config_id"}
    assert {frozenset(row["fields"]) for row in rows} == {written_fields}
    responses = []
    Vespa(url=store.vespa_app.url).feed_iterable(
        rows,
        schema=store.schema_name,
        namespace=store.schema_name,
        callback=lambda response, doc_id: responses.append(
            (doc_id, response.status_code)
        ),
    )
    assert sorted(responses) == sorted((row["id"], 200) for row in rows)
    deadline = time.monotonic() + 30
    while True:
        response = store.vespa_app.query(
            yql="select * from config_metadata where tenant_id contains 'export_history' limit 0"
        )
        coverage = response.json["root"].get("coverage", {})
        if response.json["root"]["fields"]["totalCount"] == 410 and coverage.get(
            "full"
        ):
            break
        if time.monotonic() >= deadline:
            pytest.fail(f"Config corpus did not converge: {response.json}")
        time.sleep(0.1)
    return store


def _export_coordinates(exported):
    return sorted(
        (
            row["tenant_id"],
            row["scope"],
            row["service"],
            row["config_key"],
            row["version"],
            row["config_value"]["key"],
            row["config_value"]["revision"],
        )
        for row in exported["configs"]
    )


def _expected_history(tenant="export_history", keys=41, versions=range(1, 11)):
    return [
        (tenant, "system", "runtime", f"key_{key:02}", version, key, version)
        for key in range(keys)
        for version in versions
    ]


class _VisitProxy:
    """Forward to owned Vespa, bound pages and interrupt a later response."""

    def __init__(self, upstream, *, failure_status=None, barrier=None):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlencode, urlsplit

        self.pages = []
        self.failures = 0
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                response = requests.post(
                    upstream + self.path,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                self.send_response(response.status_code)
                self.end_headers()
                self.wfile.write(response.content)

            def do_GET(self):
                parsed = urlsplit(self.path)
                params = parse_qs(parsed.query)
                visiting = parsed.path.startswith("/document/v1/")
                if visiting:
                    params["wantedDocumentCount"] = ["20"]
                if barrier and "continuation" not in params:
                    barrier.wait(timeout=20)
                if visiting and "continuation" in params and failure_status:
                    proxy.failures += 1
                    self.send_response(failure_status)
                    self.end_headers()
                    self.wfile.write(b"visit interrupted")
                    return
                response = requests.get(
                    upstream + parsed.path,
                    params=urlencode(params, doseq=True),
                    timeout=30,
                )
                if visiting:
                    proxy.pages.append(response.json())
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.content)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.store = VespaConfigStore(
            backend_url="http://127.0.0.1",
            backend_port=self.server.server_port,
        )
        return self

    def __exit__(self, *args):
        self.store.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.mark.integration
class TestCompleteHistoryExport:
    def test_export_every_retained_version_and_exact_latest(
        self, export_history_corpus
    ):
        store = export_history_corpus
        exported = store.export_configs("export_history", include_history=True)
        assert (exported["tenant_id"], exported["include_history"]) == (
            "export_history",
            True,
        )
        assert (
            _export_coordinates(json.loads(json.dumps(exported))) == _expected_history()
        )
        latest = store.export_configs("export_history")
        assert _export_coordinates(latest) == _expected_history(versions=[10])
        assert (latest["tenant_id"], latest["include_history"]) == (
            "export_history",
            False,
        )

    def test_concurrent_exports_keep_exact_tenant_histories(
        self, export_history_corpus
    ):
        store = export_history_corpus
        barrier = threading.Barrier(2)
        with _VisitProxy(store.vespa_app.url, barrier=barrier) as proxy:
            with ThreadPoolExecutor(max_workers=2) as executor:
                exports = list(
                    executor.map(
                        lambda tenant: proxy.store.export_configs(
                            tenant, include_history=True
                        ),
                        ["export_history", "export_foreign"],
                    )
                )
        assert _export_coordinates(exports[0]) == _expected_history()
        assert _export_coordinates(exports[1]) == _expected_history(
            tenant="export_foreign",
            keys=1,
            versions=range(1, 4),
        )

    @pytest.mark.parametrize("status", [503, 404])
    def test_later_visit_failure_cannot_return_partial_export(
        self,
        export_history_corpus,
        status,
    ):
        store = export_history_corpus
        with _VisitProxy(store.vespa_app.url, failure_status=status) as proxy:
            with pytest.raises((RuntimeError, requests.HTTPError)) as caught:
                proxy.store.export_configs("export_history", include_history=True)
            assert str(status) in str(caught.value)
            assert len(proxy.pages) == 1
            assert proxy.failures == (5 if status == 503 else 1)
            assert proxy.pages[0]["continuation"] != ""
        assert (
            _export_coordinates(
                store.export_configs("export_history", include_history=True)
            )
            == _expected_history()
        )

    def test_export_orders_every_row_by_config_id_then_version(
        self, export_history_corpus
    ):
        """The artifact is ordered, so replaying it restores the exported
        latest: ``import_configs`` replays rows in file order through
        ``set_config``, which ignores each row's version."""
        store = export_history_corpus
        exported = store.export_configs("export_history", include_history=True)
        assert [
            (
                f"{row['tenant_id']}:{row['scope']}:{row['service']}:"
                f"{row['config_key']}",
                row["version"],
            )
            for row in exported["configs"]
        ] == [
            (f"export_history:system:runtime:key_{key:02}", version)
            for key in range(41)
            for version in range(1, 11)
        ]

    def test_history_export_restores_the_exported_latest(self, vespa_config_store):
        store = vespa_config_store
        source = f"export_roundtrip_{uuid.uuid4().hex[:8]}"
        target = f"export_restored_{uuid.uuid4().hex[:8]}"
        for revision in range(1, 4):
            for key in ("alpha", "beta", "gamma"):
                store.set_config(
                    tenant_id=source,
                    scope=ConfigScope.SYSTEM,
                    service="runtime",
                    config_key=key,
                    config_value={"revision": revision},
                )

        exported = store.export_configs(source, include_history=True)
        assert [(row["config_key"], row["version"]) for row in exported["configs"]] == [
            ("alpha", 1),
            ("alpha", 2),
            ("alpha", 3),
            ("beta", 1),
            ("beta", 2),
            ("beta", 3),
            ("gamma", 1),
            ("gamma", 2),
            ("gamma", 3),
        ]

        assert store.import_configs(target, exported) == 9
        restored = {
            key: store.get_config(target, ConfigScope.SYSTEM, "runtime", key)
            for key in ("alpha", "beta", "gamma")
        }
        assert {key: entry.config_value for key, entry in restored.items()} == {
            "alpha": {"revision": 3},
            "beta": {"revision": 3},
            "gamma": {"revision": 3},
        }
        assert {key: entry.version for key, entry in restored.items()} == {
            "alpha": 3,
            "beta": 3,
            "gamma": 3,
        }

    def test_dashboard_download_contains_every_retained_version(
        self, export_history_corpus
    ):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""
import streamlit as st
from types import SimpleNamespace
from streamlit.runtime import get_instance
from cogniverse_dashboard.tabs.config_management import render_import_export_ui
from cogniverse_vespa.config.config_store import VespaConfigStore
store = VespaConfigStore(
    backend_url=st.session_state.endpoint,
    backend_port=st.session_state.port,
)
render_import_export_ui(SimpleNamespace(store=store), "export_history")
st.session_state.media = get_instance().media_file_mgr._storage
""")
        url = export_history_corpus.vespa_app.url
        app.session_state.endpoint, _, port = url.rpartition(":")
        app.session_state.port = int(port)
        app.run(timeout=30)
        app.checkbox[0].check()
        app.button[0].click().run(timeout=30)
        assert [error.message for error in app.exception] == []
        assert [notice.value for notice in app.error] == []
        assert [notice.value for notice in app.success] == [
            "Exported 410 configurations"
        ]
        [download] = app.get("download_button")
        media = app.session_state.media.get_file(download.proto.url.rsplit("/", 1)[-1])
        assert media.mimetype == "application/json"
        assert _export_coordinates(json.loads(media.content)) == _expected_history()
