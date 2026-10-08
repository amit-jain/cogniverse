"""The web client's Backend profiles view, driven in Chromium against real Vespa.

The built client's Node server forwards to the runtime's profile and tenant
admin routers on a real uvicorn socket, over the real config store and schema
deployment. Each action is taken through the page and its outcome is read
back from the runtime and from Vespa, not from the page alone.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.validation.profile_validator import ProfileValidator
from cogniverse_foundation.common.tenant_utils import SYSTEM_TENANT_ID
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "web-ops-harness-key"
DEPLOY_TIMEOUT_MS = 240_000
SHIPPED_PROFILES = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)["backend"]["profiles"]
BASE_SCHEMA = "video_colpali_smol500_mv_frame"
MODEL = SHIPPED_PROFILES[BASE_SCHEMA]["embedding_model"]
PIPELINE = {"extract_keyframes": True, "keyframe_fps": 1.0}
STRATEGIES = {
    "segmentation": {"class": "FrameSegmentationStrategy", "params": {"fps": 1.0}},
    "embedding": {"class": "MultiVectorEmbeddingStrategy", "params": {}},
}
SCHEMA_CONFIG = {"embedding_dim": 320, "num_patches": 1024}


@pytest.fixture(scope="module")
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        config_manager, schema_loader, workflow_state_redis_url
    ) as url:
        yield url


@pytest.fixture(scope="module")
def tenant(runtime_url):
    """A registered tenant with the runtime's base schemas deployed."""
    tenant_id = f"webprof{uuid.uuid4().hex[:8]}:main"
    created = httpx.post(
        f"{runtime_url}/admin/tenants",
        json={"tenant_id": tenant_id, "created_by": "web-ops-test"},
        timeout=DEPLOY_TIMEOUT_MS / 1000,
    )
    assert created.status_code == 200, created.text
    assert BASE_SCHEMA in created.json()["schemas_deployed"]
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


def _profiles_view(page: Page, web_url: str, tenant: str) -> None:
    page.goto(f"{web_url}/#/ops/profiles")
    expect(
        page.get_by_role("heading", name="Backend profiles", level=1)
    ).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Show profiles").click()
    expect(page.get_by_role("region", name=f"Profiles of {tenant}")).to_be_visible()


def _fill_create(
    page: Page,
    name: str,
    *,
    type_: str = "video",
    schema: str = BASE_SCHEMA,
    model: str = MODEL,
    description: str = "",
    pipeline: str = "",
    strategies: str = "",
    schema_config: str = "",
    loader: str = "colpali",
    deploy: bool = False,
):
    form = page.get_by_role("form", name="Create profile")
    form.get_by_label("Profile name").fill(name)
    form.get_by_label("Type", exact=True).fill(type_)
    form.get_by_label("Schema name").fill(schema)
    form.get_by_label("Embedding model").fill(model)
    form.get_by_label("Embedding type").select_option("multi_vector")
    form.get_by_label("Model loader").fill(loader)
    form.get_by_label("Description").fill(description)
    form.get_by_label("Pipeline config", exact=True).fill(pipeline)
    form.get_by_label("Strategies", exact=True).fill(strategies)
    form.get_by_label("Schema config", exact=True).fill(schema_config)
    if deploy:
        form.get_by_label("Deploy the schema now").check()
    form.get_by_role("button", name="Create profile").click()
    return form


def _detail(runtime_url: str, tenant: str, name: str) -> httpx.Response:
    return httpx.get(
        f"{runtime_url}/admin/profiles/{name}", params={"tenant_id": tenant}
    )


def _schema_deployed(config_manager, schema_loader, tenant: str, schema: str) -> bool:
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    return backend.schema_exists(schema_name=schema, tenant_id=tenant)


def _create_through_api(runtime_url: str, tenant: str, name: str) -> None:
    created = httpx.post(
        f"{runtime_url}/admin/profiles",
        json={
            "profile_name": name,
            "tenant_id": tenant,
            "schema_name": BASE_SCHEMA,
            "embedding_model": MODEL,
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
            "strategies": STRATEGIES,
        },
    )
    assert created.status_code == 201, created.text


def _delete_through_api(runtime_url: str, tenant: str, name: str) -> None:
    deleted = httpx.delete(
        f"{runtime_url}/admin/profiles/{name}", params={"tenant_id": tenant}
    )
    assert deleted.status_code == 200, deleted.text


class TestProfileLifecycle:
    def test_an_operator_creates_edits_deploys_and_deletes_a_profile(
        self, page, web_url, runtime_url, tenant, config_manager, schema_loader
    ):
        name = f"web_{uuid.uuid4().hex[:8]}"
        _profiles_view(page, web_url, tenant)
        expect(page.locator(f'#known-tenants option[value="{tenant}"]')).to_have_count(
            1
        )
        profiles = page.get_by_role("region", name=f"Profiles of {tenant}")
        expect(
            profiles.get_by_text(f"No profiles created for {tenant}.")
        ).to_be_visible()

        _fill_create(
            page,
            name,
            description="Web ops profile",
            pipeline=json.dumps(PIPELINE),
            strategies=json.dumps(STRATEGIES),
            schema_config=json.dumps(SCHEMA_CONFIG),
        )
        notice = page.get_by_role("status")
        expect(notice).to_contain_text(f"Created profile {name} ")
        created = _detail(runtime_url, tenant, name).json()
        assert {
            key: created[key]
            for key in (
                "type",
                "description",
                "schema_name",
                "embedding_model",
                "embedding_type",
                "model_loader",
                "pipeline_config",
                "strategies",
                "schema_config",
                "model_specific",
            )
        } == {
            "type": "video",
            "description": "Web ops profile",
            "schema_name": BASE_SCHEMA,
            "embedding_model": MODEL,
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
            "pipeline_config": PIPELINE,
            "strategies": STRATEGIES,
            "schema_config": SCHEMA_CONFIG,
            "model_specific": {},
        }
        expect(notice).to_have_text(
            f"Created profile {name} (config version {created['version']})."
        )
        row = profiles.get_by_role("row").filter(
            has=page.get_by_role("button", name=name, exact=True)
        )
        expect(row.get_by_role("cell")).to_have_text(
            [name, "video", BASE_SCHEMA, MODEL, "yes", "Web ops profile"]
        )

        detail = page.get_by_role("region", name=f"Profile {name}")
        expect(detail.locator("dt:text-is('Deployed as') + dd")).to_have_text(
            created["tenant_schema_name"]
        )
        expect(detail.locator("dt:text-is('Config version') + dd")).to_have_text(
            str(created["version"])
        )

        edit = page.get_by_role("form", name=f"Edit profile {name}")
        edit.get_by_label("Description").fill("Edited in the web view")
        edited_pipeline = {**PIPELINE, "keyframe_fps": 2.0}
        edit.get_by_label("Pipeline config", exact=True).fill(
            json.dumps(edited_pipeline)
        )
        edit.get_by_role("button", name="Save changes").click()
        expect(notice).to_contain_text(f"Saved pipeline_config, description of {name}")
        edited = _detail(runtime_url, tenant, name).json()
        expect(notice).to_have_text(
            f"Saved pipeline_config, description of {name} (config version "
            f"{edited['version']})."
        )
        assert (
            edited["description"],
            edited["pipeline_config"],
            edited["strategies"],
        ) == ("Edited in the web view", edited_pipeline, STRATEGIES)
        expect(detail.locator("dt:text-is('Config version') + dd")).to_have_text(
            str(edited["version"])
        )

        edit = page.get_by_role("form", name=f"Edit profile {name}")
        expect(edit.get_by_label("Description")).to_have_value("Edited in the web view")
        edit.get_by_role("button", name="Save changes").click()
        expect(edit.get_by_role("alert")).to_have_text(
            "Nothing to save; no field changed."
        )
        edit.get_by_label("Strategies", exact=True).fill('{"segmentation": ')
        edit.get_by_role("button", name="Save changes").click()
        expect(edit.get_by_role("alert")).to_have_text("Strategies is not valid JSON.")
        assert _detail(runtime_url, tenant, name).json()["version"] == edited["version"]

        detail.get_by_role("group", name="Deploy schema").get_by_role(
            "button", name="Deploy schema"
        ).click()
        expect(notice).to_have_text(
            f"Schema {BASE_SCHEMA} is already deployed as "
            f"{created['tenant_schema_name']}.",
            timeout=DEPLOY_TIMEOUT_MS,
        )

        delete = detail.get_by_role("group", name="Delete profile")
        delete.get_by_role("button", name="Delete").click()
        delete.get_by_label(f"Type {name} to delete this profile").fill(name)
        delete.get_by_role("button", name="Delete profile").click()
        expect(notice).to_have_text(f"Deleted profile {name}.")
        assert _detail(runtime_url, tenant, name).status_code == 404
        expect(page.get_by_role("region", name=f"Profile {name}")).to_have_count(0)
        expect(
            profiles.get_by_text(f"No profiles created for {tenant}.")
        ).to_be_visible()
        # The tenant's base schema stays when only the profile goes.
        assert _schema_deployed(config_manager, schema_loader, tenant, BASE_SCHEMA)

    def test_a_profile_deploys_its_schema_on_create_and_takes_it_on_delete(
        self, page, web_url, runtime_url, tenant, config_manager, schema_loader
    ):
        schema = "lateon_mv"
        shipped = SHIPPED_PROFILES[schema]
        name = f"web_{uuid.uuid4().hex[:8]}"
        assert not _schema_deployed(config_manager, schema_loader, tenant, schema)
        _profiles_view(page, web_url, tenant)

        _fill_create(
            page,
            name,
            type_=shipped["type"],
            schema=schema,
            model=shipped["embedding_model"],
            strategies=json.dumps(shipped["strategies"]),
            loader=shipped["model_loader"],
            deploy=True,
        )
        notice = page.get_by_role("status")
        expect(notice).to_contain_text(
            f"Created profile {name} ", timeout=DEPLOY_TIMEOUT_MS
        )
        created = _detail(runtime_url, tenant, name).json()
        expect(notice).to_have_text(
            f"Created profile {name} (config version {created['version']}) and "
            f"deployed schema {created['tenant_schema_name']}."
        )
        assert _schema_deployed(config_manager, schema_loader, tenant, schema)

        detail = page.get_by_role("region", name=f"Profile {name}")
        expect(detail.locator("dt:text-is('Deployed as') + dd")).to_have_text(
            created["tenant_schema_name"]
        )
        delete = detail.get_by_role("group", name="Delete profile")
        delete.get_by_label(f"Also delete schema {schema}").check()
        delete.get_by_role("button", name="Delete").click()
        delete.get_by_label(f"Type {name} to delete this profile").fill(name)
        delete.get_by_role("button", name="Delete profile").click()
        expect(notice).to_have_text(
            f"Deleted profile {name} and schema {schema}.", timeout=DEPLOY_TIMEOUT_MS
        )
        assert _detail(runtime_url, tenant, name).status_code == 404
        assert not _schema_deployed(config_manager, schema_loader, tenant, schema)


class TestStartFromShippedProfile:
    def test_a_profile_started_from_a_shipped_one_carries_all_its_keys(
        self, page, web_url, runtime_url, tenant, config_manager
    ):
        """Every key of the shipped profile, the ones ingestion reads beside
        the named fields included, lands in the created profile."""
        template = "document_text_semantic"
        shipped = SHIPPED_PROFILES[template]
        name = f"web_{uuid.uuid4().hex[:8]}"
        _profiles_view(page, web_url, tenant)

        form = page.get_by_role("form", name="Create profile")
        form.get_by_label("Start from shipped profile").select_option(template)
        expect(form.get_by_label("Model loader")).to_have_value("colbert")
        expect(form.get_by_label("Schema name")).to_have_value(shipped["schema_name"])
        assert json.loads(
            form.get_by_label("Extra config", exact=True).input_value()
        ) == {
            "result_granularity": "source",
            "inference_services": {"embedding": "colbert_pylate"},
        }
        form.get_by_label("Profile name").fill(name)
        form.get_by_role("button", name="Create profile").click()
        expect(page.get_by_role("status")).to_contain_text(f"Created profile {name} ")

        created = _detail(runtime_url, tenant, name).json()
        assert {
            "type": created["type"],
            "description": created["description"],
            "schema_name": created["schema_name"],
            "embedding_model": created["embedding_model"],
            "pipeline_config": created["pipeline_config"],
            "strategies": created["strategies"],
            "embedding_type": created["embedding_type"],
            "model_loader": created["model_loader"],
            "schema_config": created["schema_config"],
            **created["extra_config"],
        } == shipped
        assert created["process_type"] is None

        detail = page.get_by_role("region", name=f"Profile {name}")
        expect(detail.locator("dt:text-is('Model loader') + dd")).to_have_text(
            "colbert"
        )
        expect(detail.locator("dt:text-is('Process type') + dd")).to_have_text(
            "inferred"
        )
        _delete_through_api(runtime_url, tenant, name)

        # A shipped name the tenant has its own profile under is not offered.
        _create_through_api(runtime_url, tenant, "image_colpali_mv")
        page.reload()
        _profiles_view(page, web_url, tenant)
        options = page.get_by_role("form", name="Create profile").get_by_label(
            "Start from shipped profile"
        )
        # Tenants also inherit the profiles stored under the system tenant.
        system_profiles = set(
            config_manager.get_stored_backend_config(
                tenant_id=SYSTEM_TENANT_ID
            ).profiles
        )
        expect(options.locator("option")).to_have_text(
            [
                "Blank profile",
                *sorted(
                    (set(SHIPPED_PROFILES) | system_profiles) - {"image_colpali_mv"}
                ),
            ]
        )
        _delete_through_api(runtime_url, tenant, "image_colpali_mv")


class TestRefusedCreate:
    def test_each_refusal_shows_its_reason_and_stores_nothing(
        self, page, web_url, runtime_url, tenant, config_manager
    ):
        name = f"web_{uuid.uuid4().hex[:8]}"
        _profiles_view(page, web_url, tenant)

        form = _fill_create(page, name, schema_config="[1]")
        expect(form.get_by_role("alert")).to_have_text(
            "Schema config must be a JSON object."
        )

        form = _fill_create(
            page,
            name,
            type_="film",
            strategies=json.dumps({"embedding": {"class": "NoSuchStrategy"}}),
        )
        valid_types = ProfileValidator(config_manager)._valid_profile_types
        expect(form.get_by_role("alert")).to_have_text(
            f"Profile validation failed: Invalid profile type 'film'. Must be one "
            f"of: {valid_types}; Strategy class 'NoSuchStrategy' not found. Ensure "
            "the class is importable from the configured module path."
        )
        assert _detail(runtime_url, tenant, name).status_code == 404

        _create_through_api(runtime_url, tenant, name)
        form = _fill_create(page, name, strategies=json.dumps(STRATEGIES))
        expect(form.get_by_role("alert")).to_have_text(
            f"Profile validation failed: Profile '{name}' already exists for "
            f"tenant '{tenant}'"
        )
        _delete_through_api(runtime_url, tenant, name)


class TestConcurrency:
    def test_two_operators_editing_two_profiles_both_land(
        self, browser, web_url, runtime_url, tenant
    ):
        """Both edits rewrite the tenant's one backend config document; each
        page's save lands and neither overwrites the other."""
        names = [f"web_{uuid.uuid4().hex[:8]}" for _ in range(2)]
        for name in names:
            _create_through_api(runtime_url, tenant, name)
        contexts = [browser.new_context() for _ in names]
        pages = [context.new_page() for context in contexts]
        try:
            forms = []
            for page, name in zip(pages, names):
                _profiles_view(page, web_url, tenant)
                page.get_by_role("region", name=f"Profiles of {tenant}").get_by_role(
                    "button", name=name, exact=True
                ).click()
                form = page.get_by_role("form", name=f"Edit profile {name}")
                form.get_by_label("Description").fill(f"Edited by the page for {name}")
                forms.append(form)
            # Both saves fire before either page waits on its write.
            for form in forms:
                form.get_by_role("button", name="Save changes").click(
                    no_wait_after=True
                )
            for page, name in zip(pages, names):
                expect(page.get_by_role("status")).to_contain_text(
                    f"Saved description of {name} (config version "
                )
        finally:
            for context in contexts:
                context.close()
        assert [
            _detail(runtime_url, tenant, name).json()["description"] for name in names
        ] == [f"Edited by the page for {name}" for name in names]
        for name in names:
            _delete_through_api(runtime_url, tenant, name)


class TestFaultContract:
    def test_a_down_runtime_reads_as_an_error_not_an_empty_list(
        self, page, built_client
    ):
        dead_runtime = f"http://127.0.0.1:{free_port()}"
        unreachable = (
            f"The Cogniverse runtime at {dead_runtime} did not answer (TypeError)."
        )
        with recording_telemetry_sink() as (sink_url, _):
            with serve_web(
                built_client, dead_runtime, KEY, telemetry_url=sink_url, built=True
            ) as url:
                _profiles_view(page, url, "acme:production")
                expect(
                    page.get_by_role("region", name="Tenant").get_by_role("alert")
                ).to_have_text(f"Tenant suggestions are unavailable: {unreachable}")
                profiles = page.get_by_role(
                    "region", name="Profiles of acme:production"
                )
                expect(profiles.get_by_role("alert")).to_have_text(unreachable)
                expect(profiles.get_by_role("table")).to_have_count(0)
                create = page.get_by_role(
                    "region", name="New profile for acme:production"
                )
                expect(create.get_by_role("alert")).to_have_text(unreachable)
                expect(
                    create.get_by_label("Start from shipped profile")
                ).to_be_disabled()
                expect(
                    profiles.get_by_text("No profiles created for acme:production.")
                ).to_have_count(0)
