"""The web client's Embedding atlas view, driven in Chromium.

Documents are ingested into real Vespa through the production pipeline under a
profile made from the shipped text template (a stand-in PyLate sidecar encodes
the tokens). The page's map is read back from Plotly and compared with the
runtime's own answer for the same profile.
"""

from __future__ import annotations

import re

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from tests.utils.document_ingest import ingest_texts
from tests.utils.pylate_stub import serve_pylate_stub
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "web-ops-harness-key"
TEXTS = {
    "rivers.txt": "Rivers carve canyons over thousands of years",
    "glaciers.txt": "Glaciers grind valleys into wide troughs",
    "volcanoes.txt": "Volcanoes build islands from cooling lava",
}


@pytest.fixture(scope="module")
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        config_manager, schema_loader, workflow_state_redis_url
    ) as url:
        yield url


@pytest.fixture(scope="module")
def tenant(runtime_url, config_manager, schema_loader, tmp_path_factory):
    with serve_pylate_stub() as encoder:
        system = config_manager.get_system_config()
        previous = dict(system.inference_service_urls)
        system.inference_service_urls["colbert_pylate"] = encoder
        config_manager.set_system_config(system)
        try:
            with httpx.Client(base_url=runtime_url, timeout=300) as client:
                yield ingest_texts(
                    client,
                    config_manager,
                    schema_loader,
                    tmp_path_factory.mktemp("atlas"),
                    TEXTS,
                )
        finally:
            system = config_manager.get_system_config()
            system.inference_service_urls = previous
            config_manager.set_system_config(system)


@pytest.fixture
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


@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


def _atlas_view(page: Page, web_url: str, tenant: str) -> None:
    page.goto(f"{web_url}/#/ops/atlas")
    expect(page.get_by_role("heading", name="Embedding atlas", level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Use tenant").click()
    expect(page.get_by_role("form", name="Map documents")).to_be_visible()


def _plot(page: Page, title: str):
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    expect(area).to_have_class(re.compile(r"\bjs-plotly-plot\b"))
    return area.evaluate(
        "el => el.data.map(t => ({type: t.type, x: t.x, y: t.y, text: t.text}))"
    )


class TestAtlas:
    def test_the_map_shows_each_document_where_the_runtime_places_it(
        self, page, web_url, runtime_url, tenant
    ):
        _atlas_view(page, web_url, tenant)
        form = page.get_by_role("form", name="Map documents")
        shipped = httpx.get(
            f"{runtime_url}/admin/profile-templates", params={"tenant_id": tenant}
        ).json()["templates"]
        expect(form.get_by_label("Profile").locator("option")).to_have_text(
            sorted({"notes", *(t["profile_name"] for t in shipped)})
        )
        form.get_by_label("Profile").select_option("notes")
        form.get_by_role("button", name="Show map").click()

        expected = httpx.get(
            f"{runtime_url}/admin/tenant/{tenant}/embeddings/atlas",
            params={"profile": "notes", "limit": 500},
        ).json()
        panel = page.get_by_role("region", name=f"Profile notes of {tenant}")
        facts = dict(
            zip(
                panel.locator("dt").all_inner_texts(),
                panel.locator("dd").all_inner_texts(),
                strict=True,
            )
        )
        shares = expected["explained_variance"]
        assert facts == {
            "Schema": expected["schema_name"],
            "Embedding": "embedding, 128 dimensions",
            "Documents mapped": "3",
            "Without an embedding": "0",
            "Variance shown": f"{shares[0] * 100:.1f}% across, "
            f"{shares[1] * 100:.1f}% up",
        }
        (drawn,) = _plot(page, "Documents of notes by embedding")
        assert drawn["type"] == "scatter"
        # Each request reads the documents in Vespa's visit order.
        plotted = {t: (x, y) for t, x, y in zip(drawn["text"], drawn["x"], drawn["y"])}
        assert sorted(plotted) == sorted(TEXTS)
        for point in expected["points"]:
            assert plotted[point["title"]] == pytest.approx(
                (point["x"], point["y"]), abs=1e-9
            )

        table = panel.get_by_role("table", name="Mapped documents")
        by_title = sorted(expected["points"], key=lambda p: p["title"])
        expect(table.locator("tbody tr td:first-child")).to_have_text(
            [p["title"] for p in by_title]
        )
        panel.get_by_label("Find documents").fill("CANYONS")
        expect(table.locator("tbody tr")).to_have_count(1)
        rivers = next(p for p in expected["points"] if p["title"] == "rivers.txt")
        expect(table.locator("tbody tr").first.get_by_role("cell")).to_have_text(
            [
                "rivers.txt",
                f"{rivers['x']:.3f}",
                f"{rivers['y']:.3f}",
                TEXTS["rivers.txt"],
            ]
        )

    def test_a_refused_map_shows_the_runtimes_reason(self, page, web_url, tenant):
        _atlas_view(page, web_url, tenant)
        form = page.get_by_role("form", name="Map documents")
        form.get_by_label("Documents").fill("0")
        form.get_by_role("button", name="Show map").click()
        expect(form.get_by_role("alert")).to_have_text(
            "Documents must be a whole number above 0."
        )
        form.get_by_label("Documents").fill("10")
        form.get_by_label("Profile").select_option("image_colpali_mv")
        form.get_by_role("button", name="Show map").click()
        expect(form.get_by_role("alert")).to_have_text(
            f"Schema 'image_colpali_mv' of profile 'image_colpali_mv' is not "
            f"deployed for tenant '{tenant}'"
        )
        expect(page.get_by_role("figure")).to_have_count(0)


class TestFaultContract:
    def test_a_down_runtime_reads_as_an_error_not_as_no_profiles(
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
                page.goto(f"{url}/#/ops/atlas")
                chooser = page.get_by_role("form", name="Choose tenant")
                chooser.get_by_label("Tenant ID").fill("acme:production")
                chooser.get_by_role("button", name="Use tenant").click()
                panel = page.get_by_role(
                    "region", name="Map documents of acme:production"
                )
                expect(panel.get_by_role("alert")).to_have_text(unreachable)
                expect(page.get_by_role("form", name="Map documents")).to_have_count(0)
