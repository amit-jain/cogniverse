"""Integration tests for ``VespaConfigStore.list_all_configs`` against a
real Vespa instance.

The read path was switched from a YQL ``/search/`` query (eventually
consistent) to the Document v1 visit API (read-after-write consistent
with feeds) so cross-process schema_registry writes become visible
immediately on the next read. These tests pin that contract: a feed
followed by a visit must observe the freshly written document with no
sleep / retry.
"""

import logging

import pytest

from cogniverse_sdk.interfaces.config_store import ConfigScope
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

    def test_set_config_raises_on_version_query_failure_preserving_v1(
        self, vespa_config_store
    ):
        """A Vespa query-API outage during set_config must NOT be flattened to
        version 0 — that treats a live config as brand-new and rewrites its v1
        row. Seed v1..v3, break the query API, assert the write raises and every
        version is intact."""
        store = vespa_config_store
        tenant, service, key = "cs_verr_a", "verr_probe", "k1"

        # Guarantee exact version counting even if a prior run aborted.
        try:
            store.delete_config(
                tenant_id=tenant,
                scope=ConfigScope.BACKEND,
                service=service,
                config_key=key,
            )
        except Exception:
            pass

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
