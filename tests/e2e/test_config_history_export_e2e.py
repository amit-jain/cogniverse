"""Configuration export must carry the tenant's whole history.

The history branch of the export read a single bounded query, so a tenant
past that bound got a shorter artifact that reported itself as a successful
backup — the records it dropped could never be restored from it. The shipped
form walks the same complete traversal the listing uses.

The corpus has to be written live: the boundary is the property under test,
so no committed fixture can stand in for it.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import pytest
from playwright.sync_api import expect

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import (
    DASHBOARD,
    active_tab_panel,
    click_sub_tab,
    click_top_tab,
    register_tenant_and_wait,
    set_tenant,
    unique_id,
    wait_for_script_idle,
    wait_for_streamlit,
)

pytestmark = pytest.mark.e2e

VESPA_HOST = "http://localhost"
VESPA_QUERY_PORT = 33080

# One past the bound the old single-query read stopped at, so an export that
# still stops there is short by exactly the row this adds.
HISTORY_RECORDS = 401
CONFIG_SERVICE = "prode2e_history_export"


def _store() -> VespaConfigStore:
    return VespaConfigStore(backend_url=VESPA_HOST, backend_port=VESPA_QUERY_PORT)


def _record_fields() -> set[str]:
    """The field set an exported record carries, from the entry that emits it."""
    now = datetime.now(timezone.utc)
    return set(
        ConfigEntry(
            tenant_id="t",
            scope=ConfigScope.SYSTEM,
            service="s",
            config_key="k",
            config_value={},
            version=1,
            created_at=now,
            updated_at=now,
        )
        .to_dict()
        .keys()
    )


@pytest.fixture(scope="module")
def history_corpus(request):
    """A tenant carrying more configuration records than the old bound read."""
    tenant_id = canonical_tenant_id(unique_id("prode2epipe") + ":t1")
    register_tenant_and_wait(tenant_id, created_by="e2e-test")
    store = _store()

    baseline_history = len(
        store.export_configs(tenant_id, include_history=True)["configs"]
    )
    baseline_latest = len(
        store.export_configs(tenant_id, include_history=False)["configs"]
    )

    run = uuid.uuid4().hex[:8]
    # Zero-padded so the exported (config_id, version) order is the order the
    # sequence numbers already imply.
    written = {
        f"{run}_row_{index:04d}": {"sequence": index}
        for index in range(HISTORY_RECORDS)
    }

    def release() -> None:
        for config_key in written:
            store.delete_config(
                tenant_id=tenant_id,
                scope=ConfigScope.SYSTEM,
                service=CONFIG_SERVICE,
                config_key=config_key,
            )
        store.close()

    request.addfinalizer(release)

    for config_key, config_value in written.items():
        entry = store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service=CONFIG_SERVICE,
            config_key=config_key,
            config_value=config_value,
        )
        assert entry.version == 1, entry
    return {
        "tenant_id": tenant_id,
        "store": store,
        "written": written,
        "baseline_history": baseline_history,
        "baseline_latest": baseline_latest,
    }


def test_the_export_carries_every_record_past_the_old_bound(history_corpus):
    """Every written record is in the history export, in (key, version) order.

    The old read stopped at 400 records and reported the artifact as
    complete, so the tail was silently unrecoverable.
    """
    tenant_id = history_corpus["tenant_id"]
    written = history_corpus["written"]
    store = history_corpus["store"]

    export = store.export_configs(tenant_id, include_history=True)
    assert set(export) == {"tenant_id", "include_history", "configs", "exported_at"}
    assert export["tenant_id"] == tenant_id
    assert export["include_history"] is True

    records = export["configs"]
    assert len(records) == history_corpus["baseline_history"] + HISTORY_RECORDS
    for record in records:
        assert set(record) == _record_fields(), record

    mine = [record for record in records if record["service"] == CONFIG_SERVICE]
    assert len(mine) == HISTORY_RECORDS
    assert [record["config_key"] for record in mine] == sorted(written)
    assert [record["version"] for record in mine] == [1] * HISTORY_RECORDS
    assert {record["config_key"]: record["config_value"] for record in mine} == written
    assert {record["scope"] for record in mine} == {ConfigScope.SYSTEM.value}
    assert {record["tenant_id"] for record in mine} == {tenant_id}

    # The latest-only branch reads the same traversal, so it returns the same
    # rows: the boundary was the history read alone.
    latest = store.export_configs(tenant_id, include_history=False)
    assert latest["include_history"] is False
    assert len(latest["configs"]) == history_corpus["baseline_latest"] + HISTORY_RECORDS


@pytest.mark.browser
def test_the_dashboard_export_reports_every_record(page, history_corpus):
    """The Import/Export tab's success alert counts the whole history.

    The alert is the operator's only signal that the downloaded artifact is
    the backup they asked for.
    """
    tenant_id = history_corpus["tenant_id"]
    expected = history_corpus["baseline_history"] + HISTORY_RECORDS

    page.goto(DASHBOARD, timeout=30_000)
    wait_for_streamlit(page)
    set_tenant(page, tenant_id)
    click_top_tab(page, "Configuration")
    wait_for_script_idle(page)
    click_sub_tab(page, "Import/Export")
    wait_for_script_idle(page)

    panel = active_tab_panel(page)
    history_checkbox = panel.get_by_role("checkbox", name="Include Version History")
    expect(history_checkbox).to_have_count(1, timeout=30_000)
    history_checkbox.check()
    wait_for_script_idle(page)
    expect(history_checkbox).to_be_checked(timeout=30_000)

    export_button = active_tab_panel(page).get_by_role(
        "button", name="📥 Export Configurations", exact=True
    )
    expect(export_button).to_have_count(1, timeout=30_000)
    export_button.first.click()
    wait_for_script_idle(page)

    panel = active_tab_panel(page)
    failure = panel.locator('[data-testid="stAlert"]:has-text("Export failed")')
    assert failure.count() == 0, failure.first.inner_text()
    alert = panel.locator('[data-testid="stAlert"]:has-text("Exported")')
    expect(alert).to_have_count(1, timeout=60_000)
    assert f"Exported {expected} configurations" in alert.first.inner_text()
    assert panel.locator('[data-testid="stDownloadButton"]').count() == 1


def test_a_second_export_of_the_same_corpus_is_identical(history_corpus):
    """Two exports of an unchanged tenant carry the same records.

    The traversal pages through the store, so an export whose page boundary
    moved between runs would return different subsets of one corpus.
    """
    tenant_id = history_corpus["tenant_id"]
    store = history_corpus["store"]

    first = store.export_configs(tenant_id, include_history=True)["configs"]
    time.sleep(1)
    second = store.export_configs(tenant_id, include_history=True)["configs"]
    assert first == second
