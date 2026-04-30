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
