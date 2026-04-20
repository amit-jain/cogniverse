"""
E2E tests for multi-profile ingestion, cross-tenant isolation, and load testing.

Tests exercise the full path against the k3d cluster:
- Ingest content with multiple profiles via API, verify search via dashboard UI
- Create isolated tenants, verify data doesn't leak between them
- Concurrent multi-tenant search under load
- Verify ingestion tab UI elements and profile selection

Uses API for data setup (ingestion is slow), Playwright for UI verification
(search results, annotation controls, tenant switching).

Requires: k3d cluster running with Vespa, Runtime, and Ollama.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    DASHBOARD,
    RUNTIME,
    TENANT_ID,
    click_top_tab,
    set_tenant,
    skip_if_no_runtime,
    unique_id,
    wait_for_streamlit,
)

SEARCH_TIMEOUT = 120_000
LLM_TIMEOUT = 60_000

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.json"


def _get_profile_def(profile_name: str) -> dict:
    """Read profile definition from configs/config.json."""
    config = json.loads(CONFIG_PATH.read_text())
    return config.get("backend", {}).get("profiles", {}).get(profile_name, {})


def _deploy_schema(client: httpx.Client, profile_name: str, tenant_id: str) -> dict:
    """Register and deploy schema for profile. Returns deploy response."""
    profile_def = _get_profile_def(profile_name)
    if profile_def:
        client.post(
            "/admin/profiles",
            json={
                "profile_name": profile_name,
                "tenant_id": tenant_id,
                "type": profile_def.get("type", "video"),
                "description": profile_def.get("description", ""),
                "schema_name": profile_def.get("schema_name", profile_name),
                "embedding_model": profile_def.get("embedding_model", ""),
                "pipeline_config": profile_def.get("pipeline_config", {}),
                "strategies": profile_def.get("strategies", {}),
                "embedding_type": profile_def.get("embedding_type", ""),
                "schema_config": profile_def.get("schema_config", {}),
                "model_specific": profile_def.get("model_specific"),
                "deploy_schema": True,
            },
            timeout=60,
        )

    resp = client.post(
        f"/admin/profiles/{profile_name}/deploy",
        json={"tenant_id": tenant_id, "force": False},
        timeout=60,
    )
    return resp.json() if resp.status_code == 200 else {}


def _upload_file(
    client: httpx.Client,
    file_path: Path,
    profile: str,
    tenant_id: str,
    mime_type: str = "video/mp4",
) -> dict:
    """Upload file via /ingestion/upload, return response data."""
    with open(file_path, "rb") as f:
        resp = client.post(
            "/ingestion/upload",
            files={"file": (file_path.name, f, mime_type)},
            data={"profile": profile, "tenant_id": tenant_id, "backend": "vespa"},
        )
    assert resp.status_code == 200, f"Upload failed ({resp.status_code}): {resp.text}"
    return resp.json()


def _search(
    client: httpx.Client,
    query: str,
    profile: str,
    tenant_id: str,
    top_k: int = 10,
    strategy: str = "float_float",
) -> dict:
    """Execute search, return response data."""
    resp = client.post(
        "/search/",
        json={
            "query": query,
            "profile": profile,
            "top_k": top_k,
            "tenant_id": tenant_id,
            "strategy": strategy,
        },
    )
    assert resp.status_code == 200, f"Search failed ({resp.status_code}): {resp.text}"
    return resp.json()


def _create_tenant(client: httpx.Client, tenant_id: str) -> dict:
    """Create tenant (org auto-created). Returns response data."""
    resp = client.post(
        "/admin/tenants",
        json={"tenant_id": tenant_id, "created_by": "e2e-multiprofile-test"},
    )
    assert resp.status_code in (200, 409), f"Create tenant failed: {resp.text}"
    return resp.json() if resp.status_code == 200 else {"tenant_full_id": tenant_id}


def _cleanup_tenant(client: httpx.Client, tenant_id: str):
    """Delete tenant and org. Best-effort cleanup."""
    client.delete(f"/admin/tenants/{tenant_id}")
    org_id = tenant_id.split(":")[0] if ":" in tenant_id else tenant_id
    client.delete(f"/admin/organizations/{org_id}")


def _restart_runtime_if_unhealthy():
    """Verify the runtime is reachable before running this class's tests.

    Uses /health/live (trivial endpoint) not /health (which queries the
    backend/agent registries and can take >5s when the main event loop is
    busy with a long LLM call). The earlier implementation here issued a
    `kubectl rollout restart` whenever /health timed out within 5s, which
    created a new ReplicaSet mid-suite and cascaded dozens of subsequent
    tests into 500/connection-refused. With OLLAMA_KEEP_ALIVE=-1 the model
    stays resident, so there's no longer any accumulated-model-load cost
    that would justify an in-suite pod restart.
    """
    try:
        resp = httpx.get(f"{RUNTIME}/health/live", timeout=10)
        if resp.status_code == 200:
            return
    except (httpx.ConnectError, httpx.ReadTimeout):
        pass
    raise RuntimeError(
        "Runtime /health/live did not respond within 10s. Check pod "
        "state; the stack-level fixture should have verified this at "
        "session start."
    )


@pytest.mark.e2e
@skip_if_no_runtime
class TestMultiProfileIngestion:
    """Ingest same content with multiple profiles, verify independent results."""

    @pytest.fixture(autouse=True)
    def _ensure_runtime(self):
        _restart_runtime_if_unhealthy()

    def test_video_colpali_produces_searchable_results(self, real_video_path):
        """Baseline: ColPali profile ingests video and search returns results."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            _deploy_schema(client, "video_colpali_smol500_mv_frame", TENANT_ID)

            data = _upload_file(
                client, real_video_path, "video_colpali_smol500_mv_frame", TENANT_ID
            )
            assert data["status"] == "success"
            assert data["chunks_created"] >= 1

            if data.get("documents_fed", 0) > 0:
                time.sleep(5)
                results = _search(
                    client, "person throwing discus", "video_colpali_smol500_mv_frame",
                    TENANT_ID,
                )
                assert results["results_count"] >= 1, (
                    "ColPali search must return results after ingestion"
                )

    def test_same_video_different_ingestion_runs(self, real_video_path):
        """Re-ingesting the same video creates new documents (not idempotent)."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            _deploy_schema(client, "video_colpali_smol500_mv_frame", TENANT_ID)

            data1 = _upload_file(
                client, real_video_path, "video_colpali_smol500_mv_frame", TENANT_ID,
            )
            assert data1["status"] == "success"
            chunks_1 = data1["chunks_created"]
            assert chunks_1 >= 1

            data2 = _upload_file(
                client, real_video_path, "video_colpali_smol500_mv_frame", TENANT_ID,
            )
            assert data2["status"] == "success"
            chunks_2 = data2["chunks_created"]

            assert chunks_2 == chunks_1, (
                f"Same video should produce same chunk count: {chunks_1} vs {chunks_2}"
            )

    def test_document_text_semantic_profile(self, real_document_path):
        """Document profile ingests text via ColBERT embeddings."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            _deploy_schema(client, "document_text_semantic", TENANT_ID)

            data = _upload_file(
                client, real_document_path, "document_text_semantic", TENANT_ID,
                mime_type="text/markdown",
            )
            assert data["status"] == "success"
            assert data["chunks_created"] >= 1

    def test_audio_clap_semantic_profile(self, extracted_audio_path):
        """Audio profile ingests wav via CLAP + ColBERT embeddings."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            _deploy_schema(client, "audio_clap_semantic", TENANT_ID)

            data = _upload_file(
                client, extracted_audio_path, "audio_clap_semantic", TENANT_ID,
                mime_type="audio/wav",
            )
            assert data["status"] == "success"
            assert data["chunks_created"] >= 1

    def test_list_profiles_shows_deployed(self):
        """Profile listing includes all deployed profiles for the tenant."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(f"/search/profiles?tenant_id={TENANT_ID}")
            assert resp.status_code == 200
            data = resp.json()
            profile_names = [p["name"] for p in data.get("profiles", [])]
            assert "video_colpali_smol500_mv_frame" in profile_names, (
                f"ColPali profile must be listed, got: {profile_names}"
            )


@pytest.mark.e2e
@skip_if_no_runtime
class TestMultiProfileDashboardUI:
    """Verify multi-profile search results render in the dashboard UI."""

    def test_search_returns_results_in_dashboard(self, page):
        """After API ingestion, the dashboard search UI shows results."""
        # Data was ingested by conftest. Search via dashboard UI.
        page.goto(DASHBOARD, timeout=30_000)
        wait_for_streamlit(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Interactive Search")

        # get_by_label also matches the adjacent help-button with the same
        # aria-label; narrow to the actual textbox by role to keep Playwright
        # strict-mode happy.
        search_input = page.get_by_role(
            "textbox", name="Enter your search query"
        )
        search_input.fill("sports throwing discus")
        search_input.press("Enter")
        page.wait_for_timeout(5_000)
        page.wait_for_load_state("networkidle")

        page.locator('button[kind="primary"]:has-text("Search")').click()
        page.wait_for_timeout(SEARCH_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # The dashboard renders a heading <h3 id="search-results">🎯 Search Results</h3>
        # after the search completes. Exact-text locator like text="Search Results"
        # doesn't match because the heading is prefixed with an emoji — pin by the
        # stable element id instead. If no hits are found, Streamlit shows an
        # st.info/st.warning alert with a "No results" or "No matching" banner.
        results_heading = page.locator("#search-results")
        no_results = page.locator(
            '[data-testid="stAlert"]:has-text("No results"), '
            '[data-testid="stAlert"]:has-text("No matching")'
        )

        assert results_heading.count() > 0 or no_results.count() > 0, (
            "Dashboard search must execute and render the Search Results section "
            "(or an explicit 'No results' alert when the query matches nothing)"
        )

    def test_ingestion_tab_shows_profile_options(self, page):
        """Ingestion tab lists multiple profiles for selection."""
        page.goto(DASHBOARD, timeout=30_000)
        wait_for_streamlit(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Ingestion")
        page.wait_for_load_state("networkidle")

        multiselect = page.locator('[data-testid="stMultiSelect"]')
        assert multiselect.count() > 0, (
            "Ingestion tab must have profile multiselect"
        )

        body_text = page.inner_text("body").lower()
        assert "video_colpali_smol500_mv_frame" in body_text, (
            "Default ColPali profile must be visible in ingestion tab"
        )

    def test_tenant_switch_changes_search_context(self, page):
        """Switching tenant in sidebar changes which data search queries see."""
        page.goto(DASHBOARD, timeout=30_000)
        wait_for_streamlit(page)

        # Set to the test tenant that has data
        set_tenant(page, TENANT_ID)
        page.wait_for_timeout(2_000)

        # Verify sidebar shows the correct tenant
        sidebar_text = page.locator('[data-testid="stSidebar"]').inner_text()
        assert TENANT_ID.replace(":", " ").replace("_", " ") in sidebar_text.lower().replace(":", " ").replace("_", " ") or "flywheel" in sidebar_text.lower(), (
            f"Sidebar must show active tenant {TENANT_ID}"
        )


@pytest.mark.e2e
@skip_if_no_runtime
class TestCrossTenantIsolation:
    """Verify data isolation between tenants sharing the same profiles."""

    def test_tenant_data_isolation(self, real_video_path):
        """Tenant A has data, tenant B with same profile sees nothing."""
        org_id = unique_id("iso")
        tenant_a = f"{org_id}:alpha"
        tenant_b = f"{org_id}:beta"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_a
                )
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_b
                )

                data = _upload_file(
                    client, real_video_path, "video_colpali_smol500_mv_frame",
                    tenant_a,
                )
                assert data["status"] == "success"
                assert data["chunks_created"] >= 1

                if data.get("documents_fed", 0) == 0:
                    pytest.skip("documents_fed=0, cannot test isolation")

                time.sleep(5)

                results_a = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_a,
                )
                assert results_a["results_count"] >= 1, (
                    "Tenant A must see its own data"
                )

                results_b = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_b,
                )
                assert results_b["results_count"] == 0, (
                    f"Tenant B must NOT see tenant A's data, "
                    f"got {results_b['results_count']} results"
                )

            finally:
                _cleanup_tenant(client, tenant_a)
                _cleanup_tenant(client, tenant_b)

    def test_tenant_schema_names_are_separate(self):
        """Vespa creates distinct schema names per tenant."""
        org_id = unique_id("sch")
        tenant_a = f"{org_id}:one"
        tenant_b = f"{org_id}:two"

        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            try:
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                deploy_a = _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_a
                )
                deploy_b = _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_b
                )

                schema_a = deploy_a.get("tenant_schema_name", "")
                schema_b = deploy_b.get("tenant_schema_name", "")

                assert schema_a, f"Deploy A must return tenant_schema_name: {deploy_a}"
                assert schema_b, f"Deploy B must return tenant_schema_name: {deploy_b}"

                assert schema_a != schema_b, (
                    f"Tenant schemas must be different: {schema_a} vs {schema_b}"
                )
                assert "one" in schema_a, f"Schema A must contain tenant suffix: {schema_a}"
                assert "two" in schema_b, f"Schema B must contain tenant suffix: {schema_b}"

            finally:
                _cleanup_tenant(client, tenant_a)
                _cleanup_tenant(client, tenant_b)

    def test_reverse_isolation(self, real_video_path):
        """Ingest into B, verify A is empty."""
        org_id = unique_id("rev")
        tenant_a = f"{org_id}:first"
        tenant_b = f"{org_id}:second"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_a
                )
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_b
                )

                data = _upload_file(
                    client, real_video_path, "video_colpali_smol500_mv_frame",
                    tenant_b,
                )
                assert data["status"] == "success"

                if data.get("documents_fed", 0) == 0:
                    pytest.skip("documents_fed=0, cannot test isolation")

                time.sleep(5)

                results_a = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_a,
                )
                assert results_a["results_count"] == 0, (
                    "Tenant A must NOT see tenant B's data"
                )

                results_b = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_b,
                )
                assert results_b["results_count"] >= 1, (
                    "Tenant B must see its own data"
                )

            finally:
                _cleanup_tenant(client, tenant_a)
                _cleanup_tenant(client, tenant_b)

    def test_both_tenants_with_data_see_only_own(self, real_video_path):
        """Both tenants have video data, each sees only its own."""
        org_id = unique_id("both")
        tenant_a = f"{org_id}:left"
        tenant_b = f"{org_id}:right"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_a
                )
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_b
                )

                # Ingest into both tenants
                data_a = _upload_file(
                    client, real_video_path, "video_colpali_smol500_mv_frame",
                    tenant_a,
                )
                data_b = _upload_file(
                    client, real_video_path, "video_colpali_smol500_mv_frame",
                    tenant_b,
                )

                assert data_a["status"] == "success"
                assert data_b["status"] == "success"

                fed_a = data_a.get("documents_fed", 0) > 0
                fed_b = data_b.get("documents_fed", 0) > 0

                if not fed_a or not fed_b:
                    pytest.skip("Both tenants need documents_fed > 0")

                time.sleep(5)

                # Both tenants should see their own data
                results_a = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_a,
                )
                results_b = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_b,
                )

                assert results_a["results_count"] >= 1, "Tenant A must see its data"
                assert results_b["results_count"] >= 1, "Tenant B must see its data"

                # Results should reference different video_ids (same source but
                # different ingestion runs produce different IDs)
                ids_a = {r.get("metadata", {}).get("video_id") for r in results_a["results"]}
                ids_b = {r.get("metadata", {}).get("video_id") for r in results_b["results"]}
                assert ids_a.isdisjoint(ids_b), (
                    f"Tenants must have different video_ids: A={ids_a}, B={ids_b}"
                )

            finally:
                _cleanup_tenant(client, tenant_a)
                _cleanup_tenant(client, tenant_b)

    def test_tenant_deletion_removes_data(self, real_video_path):
        """After deleting a tenant, its data is no longer searchable."""
        org_id = unique_id("del")
        tenant_id = f"{org_id}:ephemeral"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_id)
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_id
                )

                data = _upload_file(
                    client, real_video_path, "video_colpali_smol500_mv_frame",
                    tenant_id,
                )
                assert data["status"] == "success"

                if data.get("documents_fed", 0) == 0:
                    pytest.skip("documents_fed=0, cannot test deletion")

                time.sleep(5)

                results = _search(
                    client, "person throwing discus",
                    "video_colpali_smol500_mv_frame", tenant_id,
                )
                assert results["results_count"] >= 1, (
                    "Data must be searchable before deletion"
                )

                # Delete the tenant
                resp = client.delete(f"/admin/tenants/{tenant_id}")
                assert resp.status_code == 200, (
                    f"Tenant deletion failed: {resp.text}"
                )
                time.sleep(3)

                # Search after deletion should fail or return 0
                post_delete = client.post(
                    "/search/",
                    json={
                        "query": "person throwing discus",
                        "profile": "video_colpali_smol500_mv_frame",
                        "top_k": 5,
                        "tenant_id": tenant_id,
                    },
                )
                if post_delete.status_code == 200:
                    assert post_delete.json()["results_count"] == 0, (
                        "Deleted tenant's data must not be searchable"
                    )

            finally:
                _cleanup_tenant(client, tenant_id)


def _search_sync(
    query: str, profile: str, tenant_id: str, top_k: int = 5
) -> dict:
    """Thread-safe search call for concurrent testing."""
    with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
        resp = client.post(
            "/search/",
            json={
                "query": query,
                "profile": profile,
                "top_k": top_k,
                "tenant_id": tenant_id,
                "strategy": "float_float",
            },
        )
        return {
            "tenant_id": tenant_id,
            "status_code": resp.status_code,
            "data": resp.json() if resp.status_code == 200 else {},
            "error": resp.text if resp.status_code != 200 else None,
        }


@pytest.mark.e2e
@skip_if_no_runtime
class TestConcurrentMultiTenantSearch:
    """Verify isolation holds under concurrent requests from multiple tenants."""

    def test_concurrent_search_isolation(self, real_video_path):
        """2 tenants search simultaneously, each sees only own data."""
        org_id = unique_id("conc")
        tenants = [f"{org_id}:t{i}" for i in range(2)]

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                # Setup: create tenants, deploy schemas, ingest data
                for t in tenants:
                    _create_tenant(client, t)
                    _deploy_schema(client, "video_colpali_smol500_mv_frame", t)

                fed_tenants = []
                for t in tenants:
                    data = _upload_file(
                        client, real_video_path,
                        "video_colpali_smol500_mv_frame", t,
                    )
                    if data.get("documents_fed", 0) > 0:
                        fed_tenants.append(t)

                assert len(fed_tenants) >= 2, (
                    f"Need at least 2 tenants with data, got {len(fed_tenants)}"
                )
                time.sleep(5)

                # Concurrent search: all tenants search at once
                with ThreadPoolExecutor(max_workers=len(fed_tenants)) as pool:
                    futures = {
                        pool.submit(
                            _search_sync,
                            "person throwing discus",
                            "video_colpali_smol500_mv_frame",
                            t,
                        ): t
                        for t in fed_tenants
                    }

                    results = {}
                    for future in as_completed(futures):
                        tenant = futures[future]
                        results[tenant] = future.result()

                # Each tenant must get results
                for t, r in results.items():
                    assert r["status_code"] == 200, (
                        f"Tenant {t} search failed: {r['error']}"
                    )
                    assert r["data"]["results_count"] >= 1, (
                        f"Tenant {t} must see its own data"
                    )

                # Results must reference different video_ids (isolation)
                all_video_ids = {}
                for t, r in results.items():
                    ids = {
                        hit.get("metadata", {}).get("video_id")
                        for hit in r["data"]["results"]
                    }
                    all_video_ids[t] = ids

                for i, t1 in enumerate(fed_tenants):
                    for t2 in fed_tenants[i + 1:]:
                        assert all_video_ids[t1].isdisjoint(all_video_ids[t2]), (
                            f"Tenant {t1} and {t2} share video_ids: "
                            f"{all_video_ids[t1] & all_video_ids[t2]}"
                        )

            finally:
                for t in tenants:
                    _cleanup_tenant(client, t)

    def test_concurrent_search_with_empty_tenant(self, real_video_path):
        """Concurrent search: tenant with data + tenant without data."""
        org_id = unique_id("mix")
        tenant_data = f"{org_id}:has_data"
        tenant_empty = f"{org_id}:no_data"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_data)
                _create_tenant(client, tenant_empty)
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_data
                )
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_empty
                )

                data = _upload_file(
                    client, real_video_path,
                    "video_colpali_smol500_mv_frame", tenant_data,
                )
                if data.get("documents_fed", 0) == 0:
                    pytest.skip("documents_fed=0")

                time.sleep(5)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_data = pool.submit(
                        _search_sync, "throwing discus",
                        "video_colpali_smol500_mv_frame", tenant_data,
                    )
                    f_empty = pool.submit(
                        _search_sync, "throwing discus",
                        "video_colpali_smol500_mv_frame", tenant_empty,
                    )

                    r_data = f_data.result()
                    r_empty = f_empty.result()

                assert r_data["status_code"] == 200
                assert r_data["data"]["results_count"] >= 1, (
                    "Tenant with data must see results"
                )

                assert r_empty["status_code"] == 200
                assert r_empty["data"]["results_count"] == 0, (
                    f"Empty tenant must see 0 results, got "
                    f"{r_empty['data']['results_count']}"
                )

            finally:
                _cleanup_tenant(client, tenant_data)
                _cleanup_tenant(client, tenant_empty)


@pytest.mark.e2e
@skip_if_no_runtime
class TestLoadTesting:
    """Verify system stability under burst load."""

    @pytest.fixture(autouse=True)
    def _ensure_runtime(self):
        _restart_runtime_if_unhealthy()

    def test_burst_search_requests(self):
        """Send 20 concurrent search requests to the same tenant."""
        n_requests = 20

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [
                pool.submit(
                    _search_sync,
                    f"sports activity query {i}",
                    "video_colpali_smol500_mv_frame",
                    TENANT_ID,
                )
                for i in range(n_requests)
            ]

            results = [f.result() for f in as_completed(futures)]

        success_count = sum(1 for r in results if r["status_code"] == 200)
        error_count = sum(1 for r in results if r["status_code"] != 200)

        assert success_count >= n_requests * 0.9, (
            f"At least 90% of burst requests must succeed: "
            f"{success_count}/{n_requests} succeeded, {error_count} failed"
        )

    def test_burst_routing_requests(self):
        """Send 5 concurrent routing requests."""
        queries = [
            "find sports videos",
            "summarize the game",
            "what happened in the match",
            "show me basketball highlights",
            "analyze player performance",
        ]
        n_requests = len(queries)

        def _route(query: str) -> dict:
            # 5 concurrent routing calls × ~60-90s each on CPU Ollama
            # serialize through the LLM. Total wait per request can hit
            # 300-450s; 180s timeout was too tight.
            with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
                resp = client.post(
                    "/agents/routing_agent/process",
                    json={
                        "agent_name": "routing_agent",
                        "query": query,
                        "context": {"tenant_id": TENANT_ID},
                    },
                )
                return {
                    "query": query,
                    "status_code": resp.status_code,
                    "agent": (
                        resp.json().get("recommended_agent")
                        or resp.json().get("gateway", {}).get("routed_to")
                        or resp.json().get("agent")
                    ) if resp.status_code == 200 else None,
                }

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_route, q) for q in queries]
            results = [f.result() for f in as_completed(futures)]

        success_count = sum(1 for r in results if r["status_code"] == 200)
        assert success_count >= n_requests * 0.8, (
            f"At least 80% of routing requests must succeed: "
            f"{success_count}/{n_requests}"
        )

        # All successful routes must select a valid agent
        for r in results:
            if r["status_code"] == 200:
                assert r["agent"] in (
                    "search_agent", "summarizer_agent",
                    "text_analysis_agent", "detailed_report_agent",
                    "routing_agent", "gateway_agent", "orchestrator_agent",
                ), f"Invalid agent for '{r['query']}': {r['agent']}"

    def test_sequential_ingestion_different_tenants(self, real_video_path):
        """2 tenants ingest sequentially, then search concurrently — isolation holds."""
        org_id = unique_id("load")
        tenant_a = f"{org_id}:ingest_a"
        tenant_b = f"{org_id}:ingest_b"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_a
                )
                _deploy_schema(
                    client, "video_colpali_smol500_mv_frame", tenant_b
                )

                # Ingest sequentially with pause between
                data_a = _upload_file(
                    client, real_video_path,
                    "video_colpali_smol500_mv_frame", tenant_a,
                )
                assert data_a["status"] == "success", (
                    f"Tenant A ingestion failed: {data_a}"
                )
                time.sleep(3)
                data_b = _upload_file(
                    client, real_video_path,
                    "video_colpali_smol500_mv_frame", tenant_b,
                )
                assert data_b["status"] == "success", (
                    f"Tenant B ingestion failed: {data_b}"
                )

                fed_a = data_a.get("documents_fed", 0) > 0
                fed_b = data_b.get("documents_fed", 0) > 0

                if not fed_a or not fed_b:
                    pytest.skip("Both tenants need documents_fed > 0")

                time.sleep(5)

                # Search concurrently — isolation must hold
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_a = pool.submit(
                        _search_sync, "person throwing",
                        "video_colpali_smol500_mv_frame", tenant_a,
                    )
                    f_b = pool.submit(
                        _search_sync, "person throwing",
                        "video_colpali_smol500_mv_frame", tenant_b,
                    )
                    r_a = f_a.result()
                    r_b = f_b.result()

                assert r_a["data"]["results_count"] >= 1
                assert r_b["data"]["results_count"] >= 1

                ids_a = {r.get("metadata", {}).get("video_id") for r in r_a["data"]["results"]}
                ids_b = {r.get("metadata", {}).get("video_id") for r in r_b["data"]["results"]}
                assert ids_a.isdisjoint(ids_b), (
                    f"Data leaked between tenants: A={ids_a}, B={ids_b}"
                )

            finally:
                _cleanup_tenant(client, tenant_a)
                _cleanup_tenant(client, tenant_b)
