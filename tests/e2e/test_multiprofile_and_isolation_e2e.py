"""
E2E tests for multi-profile ingestion, cross-tenant isolation, and load testing.

Tests exercise the full path against the k3d cluster:
- Ingest content with multiple profiles via API, verify search via the web client
- Create isolated tenants, verify data doesn't leak between them
- Concurrent multi-tenant search under load
- Verify ingestion tab UI elements and profile selection

Uses API for data setup (ingestion is slow), Playwright for UI verification
(search results, annotation controls, tenant switching).

Requires: k3d cluster running with Vespa, Runtime, and the configured LM.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from tests.e2e.cluster import RUNTIME, TENANT_ID
from tests.e2e.conftest import (
    GATEWAY_VIDEO_QUERIES,
    SAMPLE_VIDEO_CONTENT_ID,
    expected_gateway_routing,
)
from tests.e2e.tenants import register_tenant_and_wait, unique_id
from tests.e2e.test_api_e2e import (
    AUDIO_PROFILE,
    DOCUMENT_PROFILE,
    PROFILE,
    _served_document_windows,
)
from tests.e2e.web_client import (
    RUN_TIMEOUT_MS,
    VIEW_TIMEOUT_MS,
    choose_tenant,
    ensure_web_tenant_corpus,
    minted_tenant,
    open_agent,
    open_view,
    result_cards,
    run_agent_and_capture_state,
)
from tests.utils.profile_payload import profile_create_payload

SEARCH_TIMEOUT = 120_000
LLM_TIMEOUT = 60_000

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.json"

# Second tracked clip: video_id is the content sha256, so the same file
# yields the same id in every tenant. Isolation tests therefore give each
# tenant DISTINCT content — disjoint ids then prove isolation (a leak would
# surface the other tenant's content id in the results).
SECOND_VIDEO_PATH = (
    Path(__file__).resolve().parent.parent
    / "system"
    / "resources"
    / "videos"
    / "v_-6dz6tBH77I.mp4"
)


def _get_profile_def(profile_name: str) -> dict:
    """Read profile definition from configs/config.json."""
    config = json.loads(CONFIG_PATH.read_text())
    return config.get("backend", {}).get("profiles", {}).get(profile_name, {})


def _expected_video_documents_fed(video_path: Path, profile_name: str) -> int:
    """Return the exact keyframe/doc count for a tracked video/profile pair."""
    profile_def = _get_profile_def(profile_name)
    pipeline_config = profile_def.get("pipeline_config", {})
    target_fps = pipeline_config.get("keyframe_fps")
    if not isinstance(target_fps, (int, float)) or target_fps <= 0:
        target_fps = (
            profile_def.get("strategies", {})
            .get("segmentation", {})
            .get("params", {})
            .get("fps", 0.5)
        )

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total_frames <= 0:
        raise AssertionError(
            f"Could not determine frame count for tracked video {video_path!r}: "
            f"fps={video_fps!r}, frames={total_frames!r}"
        )
    frame_interval = int(video_fps / target_fps) if video_fps > target_fps else 1
    return sum(
        1 for frame_idx in range(total_frames) if frame_idx % frame_interval == 0
    )


def _deploy_schema(client: httpx.Client, profile_name: str, tenant_id: str) -> dict:
    """Register and deploy schema for profile. Returns deploy response."""
    profile_def = _get_profile_def(profile_name)
    if profile_def:
        client.post(
            "/admin/profiles",
            json=profile_create_payload(profile_name, profile_def, tenant_id),
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
    *,
    force: bool = False,
) -> dict:
    """Upload via /ingestion/upload (wait=true so the response shape is
    synchronous; isolation tests assert status=='success').

    ``force=True`` bypasses the (source_url, profile, tenant) idempotency
    record so the response is a fresh ingest, not an echo of a run a prior
    session already completed against the same cluster.
    """
    with open(file_path, "rb") as f:
        resp = client.post(
            f"/ingestion/upload?wait=true&wait_timeout=600&force={str(force).lower()}",
            files={"file": (file_path.name, f, mime_type)},
            data={"profile": profile, "tenant_id": tenant_id, "backend": "vespa"},
        )
    assert resp.status_code == 200, f"Upload failed ({resp.status_code}): {resp.text}"
    return resp.json()


# Every key a terminal ``/ingestion/upload?wait=true`` response carries. A
# failed run adds ``error`` and ``error_type``; pinning the set on the success
# path proves the run completed AND nothing else leaked onto the response.
_TERMINAL_UPLOAD_KEYS = frozenset(
    {
        "ingest_id",
        "sha",
        "state",
        "existing",
        "filename",
        "source_url",
        "video_id",
        "chunks_created",
        "documents_fed",
        "status",
        "wait_timed_out",
        "graph_nodes",
        "graph_edges",
    }
)


def _assert_upload_completed(data: dict) -> None:
    """Pin the terminal success shape; on a failed run the message carries the
    worker's ``error`` and ``error_type`` instead of a bare status diff."""
    assert (data["status"], data["state"]) == ("success", "complete"), data
    assert set(data) == _TERMINAL_UPLOAD_KEYS, data


def _expected_hits(top_k: int, sources_fed: int) -> int:
    """Search collapses to one hit per source video, so a tenant holding
    ``sources_fed`` videos returns that many hits (bounded by ``top_k``)."""
    return min(top_k, sources_fed)


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
    """Create tenant (org auto-created). Returns the persisted tenant row."""
    tenant_row = register_tenant_and_wait(tenant_id, created_by="e2e-multiprofile-test")
    assert (
        tenant_row["tenant_full_id"],
        tenant_row["status"],
        tenant_row["created_by"],
    ) == (
        canonical_tenant_id(tenant_id),
        "active",
        "e2e-multiprofile-test",
    )
    return tenant_row


def _cleanup_tenant(client: httpx.Client, tenant_id: str):
    """Delete the tenant. Best-effort cleanup — schemas redeploy without it.

    Deliberately does NOT call DELETE /admin/organizations: the cascade
    inside delete_organization races against Vespa's tenant_metadata
    propagation, so list_tenants_for_org_internal can return [] while
    the peer tenant's schema is still in Vespa, leaving an orphan that
    blocks subsequent deploys with BackendDeploymentError. Each test's
    finally block calls _cleanup_tenant for *every* tenant it created,
    so the explicit per-tenant DELETE handles both cleanups; the org
    record gets reaped by garbage-collection rather than the test
    pulling it. If the test wants to drop the org explicitly, use
    _cleanup_org once after all tenants are gone.
    """
    client.delete(f"/admin/tenants/{tenant_id}")


def _cleanup_org(client: httpx.Client, tenant_id: str):
    """Drop the org record. Call after all tenants in it are deleted."""
    org_id = tenant_id.split(":")[0] if ":" in tenant_id else tenant_id
    client.delete(f"/admin/organizations/{org_id}")


def _restart_runtime_if_unhealthy():
    """Verify the runtime is reachable before running this class's tests.

    Uses /health/live (trivial endpoint) not /health (which queries the
    backend/agent registries and can take >5s when the main event loop is
    busy with a long LLM call). The earlier implementation here issued a
    `kubectl rollout restart` whenever /health timed out within 5s, which
    created a new ReplicaSet mid-suite and cascaded dozens of subsequent
    tests into 500/connection-refused. With long keep-alive on the LM the model
    stays resident, so there's no longer any accumulated-model-load cost
    that would justify an in-suite pod restart.
    """
    try:
        resp = httpx.get(f"{RUNTIME}/health/live", timeout=10)
        if resp.status_code == 200:
            return
    except httpx.HTTPError:
        pass
    raise RuntimeError(
        "Runtime /health/live did not respond within 10s. Check pod "
        "state; the stack-level fixture should have verified this at "
        "session start."
    )


@pytest.mark.e2e
class TestMultiProfileIngestion:
    """Ingest same content with multiple profiles, verify independent results."""

    @pytest.fixture(autouse=True)
    def _ensure_runtime(self):
        _restart_runtime_if_unhealthy()

    def test_video_colpali_produces_searchable_results(self, real_video_path):
        """Baseline: ColPali profile ingests video and search returns results."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            expected_documents_fed = _expected_video_documents_fed(
                real_video_path, PROFILE
            )
            _deploy_schema(client, PROFILE, TENANT_ID)

            data = _upload_file(client, real_video_path, PROFILE, TENANT_ID, force=True)
            _assert_upload_completed(data)
            assert data["status"] == "success"
            assert data["existing"] is False, data
            assert data["chunks_created"] == expected_documents_fed
            assert data["documents_fed"] == expected_documents_fed

            time.sleep(5)
            results = _search(
                client,
                "person throwing discus",
                PROFILE,
                TENANT_ID,
            )
            assert results["results_count"] >= 1, (
                "ColPali search must return results after ingestion"
            )

    def test_same_video_resubmit_returns_existing_run(self, real_video_path):
        """A second identical upload echoes the completed run, not a new one."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            expected_documents_fed = _expected_video_documents_fed(
                real_video_path, PROFILE
            )
            _deploy_schema(client, PROFILE, TENANT_ID)

            data1 = _upload_file(
                client, real_video_path, PROFILE, TENANT_ID, force=True
            )
            _assert_upload_completed(data1)
            assert data1["status"] == "success"
            assert data1["existing"] is False, data1
            assert data1["chunks_created"] == expected_documents_fed
            assert data1["documents_fed"] == expected_documents_fed

            data2 = _upload_file(client, real_video_path, PROFILE, TENANT_ID)
            _assert_upload_completed(data2)
            assert data2["status"] == "success"
            assert data2["existing"] is True, data2
            assert data2["state"] == "complete"
            assert data2["ingest_id"] == data1["ingest_id"]
            assert data2["sha"] == data1["sha"]
            assert data2["video_id"] == data1["video_id"]
            assert data2["chunks_created"] == expected_documents_fed
            assert data2["documents_fed"] == expected_documents_fed

    def test_document_profile(self, real_document_path):
        """Document profile ingests text via ColBERT embeddings."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            _deploy_schema(client, DOCUMENT_PROFILE, TENANT_ID)
            windows, _ = _served_document_windows(
                ProcessingStrategySet._extract_document_text(real_document_path)
            )
            assert len(windows) > 1, windows

            data = _upload_file(
                client,
                real_document_path,
                DOCUMENT_PROFILE,
                TENANT_ID,
                mime_type="text/markdown",
                force=True,
            )
            _assert_upload_completed(data)
            assert data["status"] == "success"
            assert data["existing"] is False, data
            assert data["chunks_created"] == len(windows)
            assert data["documents_fed"] == len(windows)

    def test_audio_profile(self, extracted_audio_path):
        """Audio profile ingests wav via CLAP + ColBERT embeddings."""
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            _deploy_schema(client, AUDIO_PROFILE, TENANT_ID)

            data = _upload_file(
                client,
                extracted_audio_path,
                AUDIO_PROFILE,
                TENANT_ID,
                mime_type="audio/wav",
                force=True,
            )
            _assert_upload_completed(data)
            assert data["status"] == "success"
            assert data["existing"] is False, data
            assert data["chunks_created"] == 1
            assert data["documents_fed"] == 1

    def test_list_profiles_shows_deployed(self):
        """Profile listing includes all deployed profiles for the tenant."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(f"/search/profiles?tenant_id={TENANT_ID}")
            assert resp.status_code == 200
            data = resp.json()
            profile_names = [p["name"] for p in data.get("profiles", [])]
            assert PROFILE in profile_names, (
                f"ColPali profile must be listed, got: {profile_names}"
            )


@pytest.mark.e2e
@pytest.mark.browser
class TestMultiProfileWebUI:
    """Verify search results and tenant context in the web client."""

    def test_search_returns_results_in_the_web_client(self, page):
        """A search run's hits render as the result cards, one per hit."""
        from playwright.sync_api import expect

        query = "sports throwing discus"
        ensure_web_tenant_corpus()
        open_agent(page, "search_agent")
        state = run_agent_and_capture_state(page, "search_agent", query)
        hits = state["result"]["results"]
        assert hits != [], state["result"]
        assert [SAMPLE_VIDEO_CONTENT_ID in json.dumps(hit) for hit in hits] == [
            True
        ] * len(hits)
        cards = result_cards(state)
        results = page.get_by_role("complementary", name="Results")
        expect(results.locator(".result-title")).to_have_text(
            [card["title"] for card in cards], timeout=RUN_TIMEOUT_MS
        )
        # The search's one group states what it found for the query.
        plural = "" if len(cards) == 1 else "s"
        expect(results.locator(".result-found")).to_have_text(
            [f"Found {len(cards)} result{plural} for '{query}'."]
        )

    def test_ingestion_view_takes_the_profile_to_ingest_with(self, page):
        """The upload form names its tenant, offers the tenant's upload
        profiles with the tenant's default chosen, and ingests with the
        profiles the operator checks."""
        from playwright.sync_api import expect

        listed = httpx.get(
            f"{RUNTIME}/ingestion/profiles",
            params={"tenant_id": TENANT_ID},
            timeout=60.0,
        )
        assert listed.status_code == 200, listed.text
        targets = listed.json()
        assert targets["tenant_id"] == canonical_tenant_id(TENANT_ID), targets
        by_name = {entry["name"]: entry for entry in targets["profiles"]}
        assert by_name[PROFILE]["kind"] == "video", targets
        names = [entry["name"] for entry in targets["profiles"]]
        defaults = [name for name in names if name == targets["default_profile"]]

        open_view(page, "ingestion")
        choose_tenant(page, TENANT_ID, "Use tenant")
        expect(page.get_by_role("region", name=f"Upload to {TENANT_ID}")).to_be_visible(
            timeout=VIEW_TIMEOUT_MS
        )
        form = page.get_by_role("form", name="Upload content")
        expect(
            form.get_by_text(f"Backend: {targets['backend']}", exact=True)
        ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        boxes = form.get_by_role("group", name="Profiles").get_by_role("checkbox")
        expect(boxes).to_have_count(len(names))

        def checked() -> list[str]:
            return [
                box.get_attribute("aria-label")
                for box in boxes.all()
                if box.is_checked()
            ]

        assert [box.get_attribute("aria-label") for box in boxes.all()] == names
        assert checked() == defaults

        def box(name: str):
            return form.get_by_role("group", name="Profiles").get_by_role(
                "checkbox", name=name, exact=True
            )

        for name in defaults:
            if name != PROFILE:
                box(name).uncheck()
        box(PROFILE).check()
        expect(box(PROFILE)).to_be_checked()
        assert checked() == [PROFILE]
        expect(form.get_by_label("File", exact=True)).to_have_attribute(
            "accept", ",".join(by_name[PROFILE]["extensions"])
        )

    def test_tenant_switch_changes_the_views_context(self, page):
        """Choosing another tenant replaces every panel of the previous one."""
        from playwright.sync_api import expect

        # The views act for a registered tenant only.
        other = minted_tenant("isoview")
        open_view(page, "memory")
        choose_tenant(page, TENANT_ID, "Show memories")
        expect(
            page.get_by_role(
                "region", name=f"Memories of _user_memories in {TENANT_ID}"
            )
        ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        choose_tenant(page, other, "Show memories")
        expect(
            page.get_by_role("region", name=f"Memories of _user_memories in {other}")
        ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        expect(page.get_by_text(TENANT_ID, exact=False)).to_have_count(0)


@pytest.mark.e2e
class TestCrossTenantIsolation:
    """Verify data isolation between tenants sharing the same profiles."""

    def test_tenant_data_isolation(self, real_video_path):
        """Tenant A has data, tenant B with same profile sees nothing."""
        org_id = unique_id("iso")
        tenant_a = f"{org_id}:alpha"
        tenant_b = f"{org_id}:beta"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                expected_documents_fed = _expected_video_documents_fed(
                    real_video_path, PROFILE
                )
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                _deploy_schema(client, PROFILE, tenant_a)
                _deploy_schema(client, PROFILE, tenant_b)

                data = _upload_file(
                    client,
                    real_video_path,
                    PROFILE,
                    tenant_a,
                )
                _assert_upload_completed(data)
                assert data["status"] == "success"
                assert data["chunks_created"] == expected_documents_fed
                assert data["documents_fed"] == expected_documents_fed

                time.sleep(5)

                results_a = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_a,
                )
                assert results_a["results_count"] == _expected_hits(
                    10, sources_fed=1
                ), "Tenant A must see exactly its own segments"
                assert {r["metadata"]["video_id"] for r in results_a["results"]} == {
                    data["video_id"]
                }, results_a["results"]

                results_b = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_b,
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

                deploy_a = _deploy_schema(client, PROFILE, tenant_a)
                deploy_b = _deploy_schema(client, PROFILE, tenant_b)

                schema_a = deploy_a.get("tenant_schema_name", "")
                schema_b = deploy_b.get("tenant_schema_name", "")

                assert schema_a, f"Deploy A must return tenant_schema_name: {deploy_a}"
                assert schema_b, f"Deploy B must return tenant_schema_name: {deploy_b}"

                assert schema_a != schema_b, (
                    f"Tenant schemas must be different: {schema_a} vs {schema_b}"
                )
                # VespaSchemaManager.get_tenant_schema_name:
                # ``{base_schema}_{canonical org_tenant}``.
                base_schema = _get_profile_def(PROFILE)["schema_name"]
                assert schema_a == (
                    f"{base_schema}_{canonical_tenant_id(tenant_a).replace(':', '_')}"
                ), schema_a
                assert schema_b == (
                    f"{base_schema}_{canonical_tenant_id(tenant_b).replace(':', '_')}"
                ), schema_b

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
                expected_documents_fed = _expected_video_documents_fed(
                    real_video_path, PROFILE
                )
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                _deploy_schema(client, PROFILE, tenant_a)
                _deploy_schema(client, PROFILE, tenant_b)

                data = _upload_file(
                    client,
                    real_video_path,
                    PROFILE,
                    tenant_b,
                )
                _assert_upload_completed(data)
                assert data["status"] == "success"
                assert data["chunks_created"] == expected_documents_fed
                assert data["documents_fed"] == expected_documents_fed

                time.sleep(5)

                results_a = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_a,
                )
                assert results_a["results_count"] == 0, (
                    "Tenant A must NOT see tenant B's data"
                )

                results_b = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_b,
                )
                assert results_b["results_count"] == _expected_hits(
                    10, sources_fed=1
                ), "Tenant B must see exactly its own segments"
                assert {r["metadata"]["video_id"] for r in results_b["results"]} == {
                    data["video_id"]
                }, results_b["results"]

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
                expected_a_documents_fed = _expected_video_documents_fed(
                    real_video_path, PROFILE
                )
                expected_b_documents_fed = _expected_video_documents_fed(
                    SECOND_VIDEO_PATH, PROFILE
                )
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)

                _deploy_schema(client, PROFILE, tenant_a)
                _deploy_schema(client, PROFILE, tenant_b)

                # Ingest into both tenants
                data_a = _upload_file(
                    client,
                    real_video_path,
                    PROFILE,
                    tenant_a,
                )
                data_b = _upload_file(
                    client,
                    SECOND_VIDEO_PATH,
                    PROFILE,
                    tenant_b,
                )

                assert data_a["status"] == "success"
                assert data_b["status"] == "success"
                assert data_a["chunks_created"] == expected_a_documents_fed
                assert data_a["documents_fed"] == expected_a_documents_fed
                assert data_b["chunks_created"] == expected_b_documents_fed
                assert data_b["documents_fed"] == expected_b_documents_fed

                time.sleep(5)

                # Both tenants should see their own data
                results_a = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_a,
                )
                results_b = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_b,
                )

                assert results_a["results_count"] == _expected_hits(
                    10, sources_fed=1
                ), "Tenant A must see exactly its own segments"
                assert results_b["results_count"] == _expected_hits(
                    10, sources_fed=1
                ), "Tenant B must see exactly its own segments"

                # video_id is the content sha256 and each tenant
                # ingested a different clip, so each tenant's hits carry
                # exactly its own id
                ids_a = {
                    r.get("metadata", {}).get("video_id") for r in results_a["results"]
                }
                ids_b = {
                    r.get("metadata", {}).get("video_id") for r in results_b["results"]
                }
                assert ids_a == {data_a["video_id"]}, results_a["results"]
                assert ids_b == {data_b["video_id"]}, results_b["results"]
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
                expected_documents_fed = _expected_video_documents_fed(
                    real_video_path, PROFILE
                )
                _create_tenant(client, tenant_id)
                _deploy_schema(client, PROFILE, tenant_id)

                data = _upload_file(
                    client,
                    real_video_path,
                    PROFILE,
                    tenant_id,
                )
                _assert_upload_completed(data)
                assert data["status"] == "success"
                assert data["chunks_created"] == expected_documents_fed
                assert data["documents_fed"] == expected_documents_fed

                time.sleep(5)

                results = _search(
                    client,
                    "person throwing discus",
                    PROFILE,
                    tenant_id,
                )
                assert results["results_count"] == _expected_hits(10, sources_fed=1), (
                    "Data must be searchable before deletion"
                )
                assert {r["metadata"]["video_id"] for r in results["results"]} == {
                    data["video_id"]
                }, results["results"]

                # Delete the tenant
                resp = client.delete(f"/admin/tenants/{tenant_id}")
                assert resp.status_code == 200, f"Tenant deletion failed: {resp.text}"
                time.sleep(3)

                # Search after deletion should fail or return 0
                post_delete = client.post(
                    "/search/",
                    json={
                        "query": "person throwing discus",
                        "profile": PROFILE,
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


def _search_sync(query: str, profile: str, tenant_id: str, top_k: int = 5) -> dict:
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
class TestConcurrentMultiTenantSearch:
    """Verify isolation holds under concurrent requests from multiple tenants."""

    def test_concurrent_search_isolation(self, real_video_path):
        """2 tenants search simultaneously, each sees only own data."""
        org_id = unique_id("conc")
        tenants = [f"{org_id}:t{i}" for i in range(2)]

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                expected_documents_fed = {
                    real_video_path: _expected_video_documents_fed(
                        real_video_path, PROFILE
                    ),
                    SECOND_VIDEO_PATH: _expected_video_documents_fed(
                        SECOND_VIDEO_PATH, PROFILE
                    ),
                }
                # Setup: create tenants, deploy schemas, ingest data
                for t in tenants:
                    _create_tenant(client, t)
                    _deploy_schema(client, PROFILE, t)

                # Distinct content per tenant: content-addressed
                # video_ids from the same file would collide across
                # tenants.
                tenant_videos = dict(zip(tenants, (real_video_path, SECOND_VIDEO_PATH)))
                fed_tenants = []
                for t in tenants:
                    data = _upload_file(
                        client,
                        tenant_videos[t],
                        PROFILE,
                        t,
                    )
                    assert (
                        data["chunks_created"]
                        == expected_documents_fed[tenant_videos[t]]
                    )
                    assert (
                        data["documents_fed"]
                        == expected_documents_fed[tenant_videos[t]]
                    )
                    fed_tenants.append(t)

                assert len(fed_tenants) == 2, (
                    f"Need at least 2 tenants with data, got {len(fed_tenants)}"
                )
                time.sleep(5)

                # Concurrent search: all tenants search at once
                with ThreadPoolExecutor(max_workers=len(fed_tenants)) as pool:
                    futures = {
                        pool.submit(
                            _search_sync,
                            "person throwing discus",
                            PROFILE,
                            t,
                        ): t
                        for t in fed_tenants
                    }

                    results = {}
                    for future in as_completed(futures):
                        tenant = futures[future]
                        results[tenant] = future.result()

                # Each tenant must get exactly its own segments (top_k=5)
                assert sorted(results) == sorted(fed_tenants), results
                for t, r in results.items():
                    assert r["status_code"] == 200, (
                        f"Tenant {t} search failed: {r['error']}"
                    )
                    assert r["data"]["results_count"] == _expected_hits(
                        5, sources_fed=1
                    ), f"Tenant {t} must see exactly its own segments"

                # Results must reference different video_ids (isolation)
                all_video_ids = {}
                for t, r in results.items():
                    ids = {
                        hit.get("metadata", {}).get("video_id")
                        for hit in r["data"]["results"]
                    }
                    all_video_ids[t] = ids

                for i, t1 in enumerate(fed_tenants):
                    for t2 in fed_tenants[i + 1 :]:
                        assert all_video_ids[t1].isdisjoint(all_video_ids[t2]), (
                            f"Tenant {t1} and {t2} share video_ids: "
                            f"{all_video_ids[t1] & all_video_ids[t2]}"
                        )

            finally:
                for t in tenants:
                    _cleanup_tenant(client, t)

    def test_a_cancelled_cold_build_does_not_poison_the_tenant(self):
        """A client that hangs up during a tenant's first gateway dispatch
        must not take its peers with it.

        The first dispatch for a tenant builds that tenant's gateway agent,
        and every concurrent dispatch for the same tenant waits on that one
        build. Abandoning the request that started it leaves the build to run
        for the requests still waiting: each of them, and a later one, must
        answer normally.
        """
        org_id = unique_id("coldbuild")
        tenant_id = f"{org_id}:t1"
        query = GATEWAY_VIDEO_QUERIES[0]

        def _dispatch(timeout_s: float) -> httpx.Response:
            return httpx.post(
                f"{RUNTIME}/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": tenant_id},
                    "top_k": 3,
                },
                timeout=timeout_s,
            )

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                _create_tenant(client, tenant_id)
                _deploy_schema(client, PROFILE, tenant_id)

                with ThreadPoolExecutor(max_workers=3) as pool:
                    # The build owner hangs up while the build is in flight.
                    abandoned = pool.submit(_dispatch, 0.5)
                    time.sleep(0.2)
                    waiters = [pool.submit(_dispatch, 900.0) for _ in range(2)]
                    with pytest.raises(httpx.ReadTimeout):
                        abandoned.result()
                    settled = [waiter.result() for waiter in waiters]

                # One more after the abandoned build settled, to prove the
                # tenant's cache is usable and not holding a dead entry.
                settled.append(_dispatch(900.0))

                assert [response.status_code for response in settled] == [
                    200,
                    200,
                    200,
                ], [response.text[:200] for response in settled]
                bodies = [response.json() for response in settled]
                assert [body["status"] for body in bodies] == [
                    "success",
                    "success",
                    "success",
                ], bodies
                for body in bodies:
                    gw = body["gateway"]
                    assert (
                        gw["complexity"],
                        gw["routed_to"],
                    ) == expected_gateway_routing(query, gw), gw
                    assert body["downstream_result"]["status"] == "success", body
                    assert body["downstream_result"]["agent"] == gw["routed_to"], body
            finally:
                _cleanup_tenant(client, tenant_id)
                _cleanup_org(client, tenant_id)

    def test_concurrent_search_with_empty_tenant(self, real_video_path):
        """Concurrent search: tenant with data + tenant without data."""
        org_id = unique_id("mix")
        tenant_data = f"{org_id}:has_data"
        tenant_empty = f"{org_id}:no_data"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                expected_documents_fed = _expected_video_documents_fed(
                    real_video_path, PROFILE
                )
                _create_tenant(client, tenant_data)
                _create_tenant(client, tenant_empty)
                _deploy_schema(client, PROFILE, tenant_data)
                _deploy_schema(client, PROFILE, tenant_empty)

                data = _upload_file(
                    client,
                    real_video_path,
                    PROFILE,
                    tenant_data,
                )
                assert data["chunks_created"] == expected_documents_fed
                assert data["documents_fed"] == expected_documents_fed

                time.sleep(5)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_data = pool.submit(
                        _search_sync,
                        "throwing discus",
                        PROFILE,
                        tenant_data,
                    )
                    f_empty = pool.submit(
                        _search_sync,
                        "throwing discus",
                        PROFILE,
                        tenant_empty,
                    )

                    r_data = f_data.result()
                    r_empty = f_empty.result()

                assert r_data["status_code"] == 200
                assert r_data["data"]["results_count"] == _expected_hits(
                    5, sources_fed=1
                ), "Tenant with data must see exactly its own segments"
                assert {
                    r["metadata"]["video_id"] for r in r_data["data"]["results"]
                } == {data["video_id"]}, r_data["data"]["results"]

                assert r_empty["status_code"] == 200
                assert r_empty["data"]["results_count"] == 0, (
                    f"Empty tenant must see 0 results, got "
                    f"{r_empty['data']['results_count']}"
                )

            finally:
                _cleanup_tenant(client, tenant_data)
                _cleanup_tenant(client, tenant_empty)


@pytest.mark.e2e
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
                    PROFILE,
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
        """Send 3 concurrent routing requests (laptop CPU can't sustain 5).

        Originally 5 concurrent — but on a CPU-served LM with k3d nginx
        loadbalancer in front of one runtime pod, ~3 of 5 connections
        were dropped with httpx.RemoteProtocolError before reaching
        uvicorn. Three concurrent fits comfortably; the test still
        proves "burst routing routes correctly under contention".
        """
        queries = [
            "find sports videos",
            "summarize the game",
            "show me basketball highlights",
        ]
        n_requests = len(queries)

        def _route(query: str) -> dict:
            # Concurrent routing calls serialise through one CPU-served LM
            # worker. Each call is 90-180s; the queued tail can sit for
            # ~10 min on a laptop. 1800s timeout covers the worst case.
            with httpx.Client(base_url=RUNTIME, timeout=1800.0) as client:
                resp = client.post(
                    "/agents/gateway_agent/process",
                    json={
                        "agent_name": "gateway_agent",
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
                    )
                    if resp.status_code == 200
                    else None,
                }

        with ThreadPoolExecutor(max_workers=n_requests) as pool:
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
                    "search_agent",
                    "summarizer_agent",
                    "text_analysis_agent",
                    "detailed_report_agent",
                    "gateway_agent",
                    "orchestrator_agent",
                ), f"Invalid agent for '{r['query']}': {r['agent']}"

    def test_sequential_ingestion_different_tenants(self, real_video_path):
        """2 tenants ingest sequentially, then search concurrently — isolation holds."""
        org_id = unique_id("load")
        tenant_a = f"{org_id}:ingest_a"
        tenant_b = f"{org_id}:ingest_b"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            try:
                expected_a_documents_fed = _expected_video_documents_fed(
                    real_video_path, PROFILE
                )
                expected_b_documents_fed = _expected_video_documents_fed(
                    SECOND_VIDEO_PATH, PROFILE
                )
                _create_tenant(client, tenant_a)
                _create_tenant(client, tenant_b)
                _deploy_schema(client, PROFILE, tenant_a)
                _deploy_schema(client, PROFILE, tenant_b)

                # Ingest sequentially with pause between
                data_a = _upload_file(
                    client,
                    real_video_path,
                    PROFILE,
                    tenant_a,
                )
                assert data_a["status"] == "success", (
                    f"Tenant A ingestion failed: {data_a}"
                )
                assert data_a["chunks_created"] == expected_a_documents_fed
                assert data_a["documents_fed"] == expected_a_documents_fed
                time.sleep(3)
                data_b = _upload_file(
                    client,
                    SECOND_VIDEO_PATH,
                    PROFILE,
                    tenant_b,
                )
                assert data_b["status"] == "success", (
                    f"Tenant B ingestion failed: {data_b}"
                )
                assert data_b["chunks_created"] == expected_b_documents_fed
                assert data_b["documents_fed"] == expected_b_documents_fed

                time.sleep(5)

                # Search concurrently — isolation must hold
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_a = pool.submit(
                        _search_sync,
                        "person throwing",
                        PROFILE,
                        tenant_a,
                    )
                    f_b = pool.submit(
                        _search_sync,
                        "person throwing",
                        PROFILE,
                        tenant_b,
                    )
                    r_a = f_a.result()
                    r_b = f_b.result()

                assert r_a["status_code"] == 200, r_a["error"]
                assert r_b["status_code"] == 200, r_b["error"]
                assert r_a["data"]["results_count"] == _expected_hits(
                    5, sources_fed=1
                ), r_a["data"]
                assert r_b["data"]["results_count"] == _expected_hits(
                    5, sources_fed=1
                ), r_b["data"]

                ids_a = {
                    r.get("metadata", {}).get("video_id")
                    for r in r_a["data"]["results"]
                }
                ids_b = {
                    r.get("metadata", {}).get("video_id")
                    for r in r_b["data"]["results"]
                }
                assert ids_a == {data_a["video_id"]}, r_a["data"]["results"]
                assert ids_b == {data_b["video_id"]}, r_b["data"]["results"]
                assert ids_a.isdisjoint(ids_b), (
                    f"Data leaked between tenants: A={ids_a}, B={ids_b}"
                )

            finally:
                _cleanup_tenant(client, tenant_a)
                _cleanup_tenant(client, tenant_b)
