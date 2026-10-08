"""The web client's Ingestion view, driven in Chromium.

The built client's Node server forwards uploads to the runtime's
``/ingestion/upload`` route on a real uvicorn socket. Files land in a real
MinIO, jobs go through a real Redis queue to the ingestion worker's claim
loop, and the view follows each job on the real status stream. The worker's
processor localises each upload from MinIO through the production media
locator and stands in for the model pipeline, which needs inference services
this suite does not run; its result has the pipeline's envelope, so the
worker's own success, failure and summary handling shape every event.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_core.common.media import MediaLocator
from cogniverse_foundation.config.utils import get_config, resolve_default_profile
from cogniverse_runtime.ingestion_worker.worker import _media_config_from_defaults
from tests.system.minio_test_manager import MinIOTestManager
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "web-ops-harness-key"
DEPLOY_TIMEOUT_MS = 240_000
BUCKET = "web-ops-ingest"
# Held jobs wait on this; a test releases them once it has seen them run.
_release = threading.Event()


@pytest.fixture(scope="module")
def minio():
    manager = MinIOTestManager()
    with manager.lifecycle(name_prefix="web-ops-ingest") as instance:
        instance.boto3_client().create_bucket(Bucket=BUCKET)
        yield instance


@pytest.fixture(scope="module")
def ingest_redis_url(workflow_state_redis_url):
    # Its own database, so the queue never meets the cluster-events keys.
    return workflow_state_redis_url.rsplit("/", 1)[0] + "/5"


async def _pipeline(job, *, endpoint: str) -> dict:
    """Localise the upload and return a pipeline envelope: one keyframe and
    one fed document per line, none fed when the first line is ``unfed``; an
    empty file, or one whose first line is ``fail:<the job's profile>``, fails
    as the pipeline fails."""
    locator = MediaLocator(
        tenant_id=job.tenant_id,
        config=_media_config_from_defaults({"minio_endpoint": endpoint}),
    )
    local_path = Path(await asyncio.to_thread(locator.localize, job.source_url))
    lines = local_path.read_text().splitlines()
    name = Path(job.source_url).name
    if lines and lines[0] == "hold":
        await asyncio.to_thread(_release.wait, 120)
    if not lines or lines[0] == f"fail:{job.profile}":
        return {
            "status": "failed",
            "error": "no keyframes extracted",
            "errors": [f"{name} has no frames"],
        }
    return {
        "video_id": Path(name).stem,
        "results": {
            "keyframes": [{"frame": index} for index in range(len(lines))],
            "embeddings": {"documents_fed": 0 if lines[0] == "unfed" else len(lines)},
        },
    }


@pytest.fixture(scope="module")
def runtime_url(
    config_manager, schema_loader, workflow_state_redis_url, minio, ingest_redis_url
):
    system = config_manager.get_system_config()
    previous = (system.redis_url, system.minio_endpoint)
    system.redis_url = ingest_redis_url
    system.minio_endpoint = minio.endpoint
    config_manager.set_system_config(system)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("REDIS_URL", ingest_redis_url)
        env.setenv("MINIO_ENDPOINT", minio.endpoint)
        env.setenv("MINIO_ACCESS_KEY", minio.access_key)
        env.setenv("MINIO_SECRET_KEY", minio.secret_key)
        env.setenv("MINIO_DEFAULT_BUCKET", BUCKET)
        env.setenv("AWS_ACCESS_KEY_ID", minio.access_key)
        env.setenv("AWS_SECRET_ACCESS_KEY", minio.secret_key)
        env.setenv("INGEST_IDEMPOTENCY_TTL_SECONDS", "3600")

        async def processor(job):
            return await _pipeline(job, endpoint=minio.endpoint)

        try:
            with serve_ops_runtime(
                config_manager,
                schema_loader,
                workflow_state_redis_url,
                ingest_processor=processor,
            ) as url:
                yield url
        finally:
            system = config_manager.get_system_config()
            system.redis_url, system.minio_endpoint = previous
            config_manager.set_system_config(system)


@pytest.fixture(scope="module")
def tenant(runtime_url):
    tenant_id = f"webingest{uuid.uuid4().hex[:8]}:main"
    created = httpx.post(
        f"{runtime_url}/admin/tenants",
        json={"tenant_id": tenant_id, "created_by": "web-ops-test"},
        timeout=DEPLOY_TIMEOUT_MS / 1000,
    )
    assert created.status_code == 200, created.text
    yield tenant_id
    deleted = httpx.delete(
        f"{runtime_url}/admin/tenants/{tenant_id}", timeout=DEPLOY_TIMEOUT_MS / 1000
    )
    assert deleted.status_code == 200, deleted.text


@pytest.fixture()
def web_url(built_client, runtime_url):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, KEY, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


def _ingestion_view(page: Page, web_url: str, tenant: str) -> None:
    page.goto(f"{web_url}/#/ops/ingestion")
    expect(page.get_by_role("heading", name="Ingestion", level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Use tenant").click()
    expect(page.get_by_role("region", name=f"Upload to {tenant}")).to_be_visible()


def _upload(page: Page, path: Path, *, force: bool = False):
    form = page.get_by_role("form", name="Upload content")
    form.get_by_label("File", exact=True).set_input_files(str(path))
    if force:
        form.get_by_label("Ingest again even if already ingested").check()
    form.get_by_role("button", name="Upload and ingest").click()
    return form


def _row(page: Page, ingest_id: str):
    return (
        page.get_by_role("region", name="Ingests")
        .get_by_role("row")
        .filter(has=page.get_by_role("cell", name=ingest_id, exact=True))
    )


def _status(runtime_url: str, ingest_id: str) -> dict:
    response = httpx.get(f"{runtime_url}/ingestion/{ingest_id}/status")
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_state(runtime_url: str, ingest_id: str, state: str, timeout=60.0) -> str:
    """The ingest's state once it is ``state``, or its last state at the timeout."""
    deadline = time.monotonic() + timeout
    while True:
        current = _status(runtime_url, ingest_id)["state"]
        if current == state or time.monotonic() > deadline:
            return current
        time.sleep(0.5)


def _stored_keys(minio, tenant: str) -> list:
    listed = minio.boto3_client().list_objects_v2(Bucket=BUCKET, Prefix=f"{tenant}/")
    return sorted(item["Key"] for item in listed.get("Contents", []))


def _ingest_id_from_notice(page: Page, filename: str) -> str:
    notice = page.get_by_role("status")
    expect(notice).to_contain_text(f"Queued {filename} as ingest ")
    return notice.inner_text().removeprefix(f"Queued {filename} as ingest ").rstrip(".")


class TestIngestion:
    def test_an_upload_is_followed_live_to_its_result(
        self, page, web_url, runtime_url, tenant, config_manager, minio, tmp_path
    ):
        _release.clear()
        clip = tmp_path / f"clip-{uuid.uuid4().hex[:6]}.mp4"
        clip.write_text("hold\nframe two\nframe three\n")
        _ingestion_view(page, web_url, tenant)
        _upload(page, clip)
        ingest_id = _ingest_id_from_notice(page, clip.name)

        row = _row(page, ingest_id)
        profile = resolve_default_profile(get_config(tenant, config_manager))
        expect(row.get_by_role("cell")).to_have_text(
            [ingest_id, clip.name, profile, "running", ""]
        )
        _release.set()

        status = _status(runtime_url, ingest_id)
        source_url = status["history"][0]["source_url"]
        video_id = Path(source_url).stem
        expect(row.get_by_role("cell")).to_have_text(
            [
                ingest_id,
                clip.name,
                profile,
                "complete",
                f"{video_id}: 3 chunks, 3 documents fed.",
            ]
        )
        status = _status(runtime_url, ingest_id)
        assert [event["state"] for event in status["history"]] == [
            "queued",
            "running",
            "complete",
        ]
        assert status["latest"]["result"] == {
            "video_id": video_id,
            "keyframes": 3,
            "documents_fed": 3,
            "chunks": 3,
        }
        # The bytes the page sent are the object the worker read.
        bucket, key = source_url.removeprefix("s3://").split("/", 1)
        stored = minio.boto3_client().get_object(Bucket=bucket, Key=key)
        assert stored["Body"].read() == clip.read_bytes()
        assert key.startswith(f"{tenant}/")

        # The same bytes again are the same ingest, unless forced.
        _upload(page, clip)
        expect(page.get_by_role("status")).to_have_text(
            f"{clip.name} matches ingest {ingest_id} (complete); following it."
        )
        expect(
            page.get_by_role("region", name="Ingests").get_by_role("row")
        ).to_have_count(2)
        _upload(page, clip, force=True)
        forced_id = _ingest_id_from_notice(page, clip.name)
        assert forced_id != ingest_id
        expect(_row(page, forced_id).get_by_role("cell").nth(3)).to_have_text(
            "complete"
        )

    def test_a_failed_pipeline_shows_the_workers_error(
        self, page, web_url, runtime_url, tenant, tmp_path
    ):
        empty = tmp_path / f"empty-{uuid.uuid4().hex[:6]}.mp4"
        empty.write_text("")
        _ingestion_view(page, web_url, tenant)
        _upload(page, empty)
        ingest_id = _ingest_id_from_notice(page, empty.name)
        row = _row(page, ingest_id)
        expect(row.get_by_role("cell").nth(3)).to_have_text("failed")
        status = _status(runtime_url, ingest_id)
        error = (
            "no keyframes extracted "
            f"[{Path(status['history'][0]['source_url']).name} has no frames]"
        )
        assert (
            status["latest"]["state"],
            status["latest"]["error_type"],
            status["latest"]["error"],
        ) == ("failed", "IngestPipelineError", error)
        expect(row.get_by_role("cell").nth(4)).to_have_text(
            f"IngestPipelineError: {error}"
        )

    def test_following_an_ingest_by_id(
        self, page, web_url, runtime_url, tenant, config_manager, tmp_path
    ):
        clip = tmp_path / f"follow-{uuid.uuid4().hex[:6]}.mp4"
        clip.write_text("one\n")
        _ingestion_view(page, web_url, tenant)
        _upload(page, clip)
        ingest_id = _ingest_id_from_notice(page, clip.name)
        expect(_row(page, ingest_id).get_by_role("cell").nth(3)).to_have_text(
            "complete"
        )

        other = page.context.new_page()
        _ingestion_view(other, web_url, tenant)
        follow = other.get_by_role("form", name="Follow an ingest")
        follow.get_by_label("Ingest ID").fill("no-such-ingest")
        follow.get_by_role("button", name="Follow").click()
        expect(follow.get_by_role("alert")).to_have_text(
            "No status events for ingest_id='no-such-ingest'"
        )
        follow.get_by_label("Ingest ID").fill(ingest_id)
        follow.get_by_role("button", name="Follow").click()
        video_id = Path(
            _status(runtime_url, ingest_id)["history"][0]["source_url"]
        ).stem
        expect(_row(other, ingest_id).get_by_role("cell")).to_have_text(
            [
                ingest_id,
                clip.name,
                resolve_default_profile(get_config(tenant, config_manager)),
                "complete",
                f"{video_id}: 1 chunks, 1 documents fed.",
            ]
        )

    def test_a_file_the_profile_cannot_read_is_refused_with_the_reason(
        self, page, web_url, tenant, config_manager, minio, tmp_path
    ):
        """A text file sent to the tenant's default video profile is refused
        when it is uploaded, with the reason, and nothing is stored or
        queued."""
        stored_before = _stored_keys(minio, tenant)
        text = tmp_path / "zephyr_kangaroo.txt"
        text.write_text("A red kangaroo named Zephyr plays chess.\n")
        profile = resolve_default_profile(get_config(tenant, config_manager))
        _ingestion_view(page, web_url, tenant)

        form = _upload(page, text)

        expect(form.get_by_role("alert")).to_have_text(
            f"zephyr_kangaroo.txt is a .txt file; profile '{profile}' ingests "
            "video files (.avi, .mkv, .mov, .mp4, .webm)."
        )
        expect(
            page.get_by_role("region", name="Ingests").get_by_role("table")
        ).to_have_count(0)
        assert _stored_keys(minio, tenant) == stored_before

    def test_a_completion_that_fed_nothing_reads_as_a_failure(
        self, page, web_url, runtime_url, tenant, tmp_path
    ):
        clip = tmp_path / f"unfed-{uuid.uuid4().hex[:6]}.mp4"
        clip.write_text("unfed\nframe two\n")
        _ingestion_view(page, web_url, tenant)
        _upload(page, clip)
        ingest_id = _ingest_id_from_notice(page, clip.name)
        row = _row(page, ingest_id)
        expect(row.get_by_role("cell").nth(3)).to_have_text("complete")
        status = _status(runtime_url, ingest_id)
        video_id = Path(status["history"][0]["source_url"]).stem
        assert status["latest"]["result"]["documents_fed"] == 0
        expect(row.get_by_role("cell").nth(4).get_by_role("alert")).to_have_text(
            f"{video_id}: completed without feeding any documents."
        )

    def test_a_cancelled_ingest_shows_its_reason_and_stops_being_followed(
        self, page, web_url, runtime_url, tenant, tmp_path
    ):
        _release.clear()
        held = tmp_path / f"held-{uuid.uuid4().hex[:6]}.mp4"
        held.write_text("hold\n")
        queued = tmp_path / f"queued-{uuid.uuid4().hex[:6]}.mp4"
        queued.write_text("frame\n")
        event_reads = []
        page.on(
            "request",
            lambda request: (
                event_reads.append(request.url) if "/events" in request.url else None
            ),
        )
        _ingestion_view(page, web_url, tenant)
        _upload(page, held)
        held_id = _ingest_id_from_notice(page, held.name)
        expect(_row(page, held_id).get_by_role("cell").nth(3)).to_have_text("running")
        _upload(page, queued)
        queued_id = _ingest_id_from_notice(page, queued.name)
        row = _row(page, queued_id)
        expect(row.get_by_role("cell").nth(3)).to_have_text("queued")

        cancelled = httpx.post(
            f"{runtime_url}/events/ingestion/{queued_id}/cancel",
            json={"reason": "operator stopped it"},
        )
        assert cancelled.status_code == 200, cancelled.text
        _release.set()

        expect(row.get_by_role("cell").nth(3)).to_have_text("cancelled")
        expect(row.get_by_role("cell").nth(4).get_by_role("alert")).to_have_text(
            "Cancelled: operator stopped it"
        )
        assert _status(runtime_url, queued_id)["latest"]["state"] == "cancelled"
        # The stream ended on the terminal event and is not opened again.
        page.wait_for_timeout(3000)
        assert [url for url in event_reads if f"/{queued_id}/events" in url] == [
            f"{web_url}/ui-api/runtime/ingestion/{queued_id}/events"
        ]


def _upload_targets(runtime_url: str, tenant: str) -> dict:
    response = httpx.get(
        f"{runtime_url}/ingestion/profiles", params={"tenant_id": tenant}, timeout=60
    )
    assert response.status_code == 200, response.text
    return response.json()


def _profile_box(page: Page, name: str):
    return (
        page.get_by_role("form", name="Upload content")
        .get_by_role("group", name="Profiles")
        .get_by_role("checkbox", name=name, exact=True)
    )


def _second_video_profile(targets: dict) -> str:
    """A video profile of the tenant other than its default."""
    others = [
        profile["name"]
        for profile in targets["profiles"]
        if profile["kind"] == "video" and profile["name"] != targets["default_profile"]
    ]
    assert others, targets
    return others[0]


class TestUploadChoices:
    def test_the_form_offers_the_tenants_profiles_and_the_files_they_read(
        self, page, web_url, runtime_url, tenant, config_manager
    ):
        targets = _upload_targets(runtime_url, tenant)
        default = resolve_default_profile(get_config(tenant, config_manager))
        assert targets["tenant_id"] == tenant
        assert targets["backend"] == "vespa"
        assert targets["default_profile"] == default
        by_name = {profile["name"]: profile for profile in targets["profiles"]}
        assert by_name[default]["kind"] == "video"
        assert by_name[default]["extensions"] == [
            ".avi",
            ".mkv",
            ".mov",
            ".mp4",
            ".webm",
        ]
        _ingestion_view(page, web_url, tenant)
        form = page.get_by_role("form", name="Upload content")

        boxes = form.get_by_role("group", name="Profiles").get_by_role("checkbox")
        expect(boxes).to_have_count(len(targets["profiles"]))
        assert [box.get_attribute("aria-label") for box in boxes.all()] == sorted(
            by_name
        )
        assert [
            box.get_attribute("aria-label") for box in boxes.all() if box.is_checked()
        ] == [default]
        expect(form.get_by_text("Backend: vespa", exact=True)).to_be_visible()
        file_input = form.get_by_label("File", exact=True)
        expect(file_input).to_have_attribute("accept", ".avi,.mkv,.mov,.mp4,.webm")

        second = _second_video_profile(targets)
        _profile_box(page, second).check()
        expect(file_input).to_have_attribute(
            "accept",
            ",".join(
                dict.fromkeys(
                    by_name[default]["extensions"] + by_name[second]["extensions"]
                )
            ),
        )
        _profile_box(page, default).uncheck()
        _profile_box(page, second).uncheck()
        expect(form.get_by_role("button", name="Upload and ingest")).to_be_disabled()
        expect(
            form.get_by_text("Choose at least one profile.", exact=True)
        ).to_be_visible()

    def test_one_file_goes_to_each_chosen_profile_and_the_batch_reads_as_one(
        self, page, web_url, runtime_url, tenant, config_manager, tmp_path
    ):
        default = resolve_default_profile(get_config(tenant, config_manager))
        second = _second_video_profile(_upload_targets(runtime_url, tenant))
        good = tmp_path / f"pair-{uuid.uuid4().hex[:6]}.mp4"
        good.write_text(f"frame {uuid.uuid4().hex}\nframe two\n")
        half = tmp_path / f"half-{uuid.uuid4().hex[:6]}.mp4"
        half.write_text(f"fail:{second}\nframe {uuid.uuid4().hex}\n")
        _ingestion_view(page, web_url, tenant)
        _profile_box(page, second).check()

        rows = {}
        for clip in (good, half):
            _upload(page, clip)
            notice = page.get_by_role("status")
            expect(notice).to_contain_text(f"Queued {clip.name} as ingest ")
            ids = re.findall(
                rf"Queued {re.escape(clip.name)} as ingest (\S+) \(([^)]+)\)\.",
                notice.inner_text(),
            )
            assert [profile for _, profile in ids] == [default, second]
            rows[clip.name] = dict((profile, ingest) for ingest, profile in ids)

        for profile, ingest_id in rows[good.name].items():
            video_id = Path(
                _status(runtime_url, ingest_id)["history"][0]["source_url"]
            ).stem
            expect(_row(page, ingest_id).get_by_role("cell")).to_have_text(
                [
                    ingest_id,
                    good.name,
                    profile,
                    "complete",
                    f"{video_id}: 2 chunks, 2 documents fed.",
                ]
            )
        expect(
            _row(page, rows[half.name][second]).get_by_role("cell").nth(3)
        ).to_have_text("failed")
        expect(
            page.get_by_role("region", name="Batches").get_by_role("listitem")
        ).to_have_text(
            [
                f"{half.name}: ingestion failed for {second} (1 of 2 profiles).",
                f"{good.name}: all 2 profiles ingested.",
            ]
        )

    def test_an_upload_answer_without_an_ingest_id_is_an_error(
        self, page, web_url, tenant, config_manager, tmp_path
    ):
        """A runtime answer that names no ingest is not followed as one."""
        clip = tmp_path / f"noid-{uuid.uuid4().hex[:6]}.mp4"
        clip.write_text("frame\n")
        page.route(
            "**/ui-api/runtime/ingestion/upload",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {"state": "queued", "existing": False, "filename": clip.name}
                ),
            ),
        )
        _ingestion_view(page, web_url, tenant)
        form = _upload(page, clip)

        expect(form.get_by_role("alert")).to_have_text(
            f"The runtime accepted {clip.name} but answered no ingest ID."
        )
        expect(
            page.get_by_role("region", name="Ingests").get_by_role("table")
        ).to_have_count(0)
        expect(
            page.get_by_role("region", name="Batches").get_by_role("listitem")
        ).to_have_text(
            [
                f"{clip.name}: ingestion failed for "
                f"{resolve_default_profile(get_config(tenant, config_manager))} "
                "(1 of 1 profiles)."
            ]
        )

    def test_an_ingest_that_never_ends_is_given_up_after_the_deadline(
        self, page, web_url, runtime_url, tenant, tmp_path
    ):
        _release.clear()
        held = tmp_path / f"stuck-{uuid.uuid4().hex[:6]}.mp4"
        held.write_text(f"hold\n{uuid.uuid4().hex}\n")
        page.clock.install()
        _ingestion_view(page, web_url, tenant)
        _upload(page, held)
        ingest_id = _ingest_id_from_notice(page, held.name)
        row = _row(page, ingest_id)
        expect(row.get_by_role("cell").nth(3)).to_have_text("running")

        page.clock.fast_forward("15:00")

        expect(row.get_by_role("cell").nth(4).get_by_role("alert")).to_have_text(
            "No terminal state within 900 s; the last state was running. "
            "Follow it by ID to keep watching."
        )
        _release.set()
        assert _wait_for_state(runtime_url, ingest_id, "complete") == "complete"
        # The page stopped following it at the deadline.
        page.wait_for_timeout(2000)
        expect(row.get_by_role("cell").nth(3)).to_have_text("running")

    def test_followed_ingests_and_batches_survive_a_reload(
        self, page, web_url, runtime_url, tenant, config_manager, tmp_path
    ):
        clip = tmp_path / f"kept-{uuid.uuid4().hex[:6]}.mp4"
        clip.write_text(f"frame {uuid.uuid4().hex}\n")
        _ingestion_view(page, web_url, tenant)
        _upload(page, clip)
        ingest_id = _ingest_id_from_notice(page, clip.name)
        video_id = Path(
            _status(runtime_url, ingest_id)["history"][0]["source_url"]
        ).stem
        expected = [
            ingest_id,
            clip.name,
            resolve_default_profile(get_config(tenant, config_manager)),
            "complete",
            f"{video_id}: 1 chunks, 1 documents fed.",
        ]
        expect(_row(page, ingest_id).get_by_role("cell")).to_have_text(expected)

        page.reload()
        expect(page.get_by_role("heading", name="Ingestion", level=1)).to_be_visible()

        expect(_row(page, ingest_id).get_by_role("cell")).to_have_text(expected)
        expect(
            page.get_by_role("region", name="Batches").get_by_role("listitem")
        ).to_have_text([f"{clip.name}: all 1 profile ingested."])
        page.get_by_role("region", name="Ingests").get_by_role(
            "button", name="Clear list"
        ).click()
        page.reload()
        expect(
            page.get_by_role("region", name="Ingests").get_by_role("table")
        ).to_have_count(0)
        expect(page.get_by_role("region", name="Batches")).to_have_count(0)


class TestConcurrency:
    def test_two_uploads_at_once_each_follow_their_own_job(
        self, browser, web_url, runtime_url, tenant, tmp_path
    ):
        files = []
        for lines in (2, 4):
            path = tmp_path / f"clip{lines}-{uuid.uuid4().hex[:6]}.mp4"
            path.write_text("".join(f"frame {n}\n" for n in range(lines)))
            files.append((path, lines))
        contexts = [browser.new_context() for _ in files]
        pages = [context.new_page() for context in contexts]
        try:
            for page in pages:
                _ingestion_view(page, web_url, tenant)
            for page, (path, _) in zip(pages, files):
                form = page.get_by_role("form", name="Upload content")
                form.get_by_label("File", exact=True).set_input_files(str(path))
            for page in pages:
                page.get_by_role("form", name="Upload content").get_by_role(
                    "button", name="Upload and ingest"
                ).click(no_wait_after=True)
            for page, (path, lines) in zip(pages, files):
                ingest_id = _ingest_id_from_notice(page, path.name)
                video_id = Path(
                    _status(runtime_url, ingest_id)["history"][0]["source_url"]
                ).stem
                expect(_row(page, ingest_id).get_by_role("cell").nth(4)).to_have_text(
                    f"{video_id}: {lines} chunks, {lines} documents fed."
                )
                expect(
                    page.get_by_role("region", name="Ingests").get_by_role("row")
                ).to_have_count(2)
        finally:
            for context in contexts:
                context.close()


class TestFaultContract:
    def test_a_down_runtime_is_reported_before_anything_is_uploaded(
        self, page, built_client, tmp_path
    ):
        """The view asks the runtime what it accepts before it offers the
        upload; a runtime that does not answer leaves nothing to submit."""
        dead_runtime = f"http://127.0.0.1:{free_port()}"
        uploads = []
        page.on(
            "request",
            lambda request: (
                uploads.append(request.url)
                if "/ingestion/upload" in request.url
                else None
            ),
        )
        with recording_telemetry_sink() as (sink_url, _):
            with serve_web(
                built_client, dead_runtime, KEY, telemetry_url=sink_url, built=True
            ) as url:
                _ingestion_view(page, url, "acme:production")
                region = page.get_by_role("region", name="Upload to acme:production")
                expect(region.get_by_role("alert")).to_have_text(
                    "The runtime cannot take uploads now: The Cogniverse runtime "
                    f"at {dead_runtime} did not answer (TypeError)."
                )
                expect(
                    region.get_by_role("button", name="Upload and ingest")
                ).to_be_disabled()
                expect(
                    page.get_by_role("region", name="Ingests").get_by_role("table")
                ).to_have_count(0)
        assert uploads == []
