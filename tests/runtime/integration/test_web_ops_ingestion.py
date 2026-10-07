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
import threading
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
    build_web_client,
    free_port,
    install_web_client,
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
def built_client(tmp_path_factory):
    return build_web_client(install_web_client(tmp_path_factory.mktemp("web_ops")))


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
    one fed document per line; an empty file fails as the pipeline fails."""
    locator = MediaLocator(
        tenant_id=job.tenant_id,
        config=_media_config_from_defaults({"minio_endpoint": endpoint}),
    )
    local_path = Path(await asyncio.to_thread(locator.localize, job.source_url))
    lines = local_path.read_text().splitlines()
    name = Path(job.source_url).name
    if lines and lines[0] == "hold":
        await asyncio.to_thread(_release.wait, 120)
    if not lines:
        return {
            "status": "failed",
            "error": "no keyframes extracted",
            "errors": [f"{name} has no frames"],
        }
    return {
        "video_id": Path(name).stem,
        "results": {
            "keyframes": [{"frame": index} for index in range(len(lines))],
            "embeddings": {"documents_fed": len(lines)},
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


def _ingest_id_from_notice(page: Page, filename: str) -> str:
    notice = page.get_by_role("status")
    expect(notice).to_contain_text(f"Queued {filename} as ingest ")
    return notice.inner_text().removeprefix(f"Queued {filename} as ingest ").rstrip(".")


class TestIngestion:
    def test_an_upload_is_followed_live_to_its_result(
        self, page, web_url, runtime_url, tenant, config_manager, minio, tmp_path
    ):
        _release.clear()
        clip = tmp_path / f"clip-{uuid.uuid4().hex[:6]}.txt"
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
        empty = tmp_path / f"empty-{uuid.uuid4().hex[:6]}.txt"
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
        clip = tmp_path / f"follow-{uuid.uuid4().hex[:6]}.txt"
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
        source_name = Path(
            _status(runtime_url, ingest_id)["history"][0]["source_url"]
        ).name
        expect(_row(other, ingest_id).get_by_role("cell")).to_have_text(
            [
                ingest_id,
                source_name,
                resolve_default_profile(get_config(tenant, config_manager)),
                "complete",
                f"{Path(source_name).stem}: 1 chunks, 1 documents fed.",
            ]
        )


class TestConcurrency:
    def test_two_uploads_at_once_each_follow_their_own_job(
        self, browser, web_url, runtime_url, tenant, tmp_path
    ):
        files = []
        for lines in (2, 4):
            path = tmp_path / f"clip{lines}-{uuid.uuid4().hex[:6]}.txt"
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
    def test_a_down_runtime_refuses_the_upload_with_its_reason(
        self, page, built_client, tmp_path
    ):
        dead_runtime = f"http://127.0.0.1:{free_port()}"
        clip = tmp_path / "clip.txt"
        clip.write_text("frame\n")
        with recording_telemetry_sink() as (sink_url, _):
            with serve_web(
                built_client, dead_runtime, KEY, telemetry_url=sink_url, built=True
            ) as url:
                _ingestion_view(page, url, "acme:production")
                form = _upload(page, clip)
                expect(form.get_by_role("alert")).to_have_text(
                    f"The Cogniverse runtime at {dead_runtime} did not answer "
                    "(TypeError)."
                )
                expect(
                    page.get_by_role("region", name="Ingests").get_by_role("table")
                ).to_have_count(0)
