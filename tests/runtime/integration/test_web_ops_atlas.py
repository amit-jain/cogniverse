"""The web client's Embedding atlas view, driven in Chromium.

Documents are ingested into real Vespa through the production pipeline under a
profile made from the shipped text template (a stand-in PyLate sidecar encodes
the tokens, documents and queries alike). The page's maps are read back from
Plotly and compared with the runtime's own answer for the same profile; a
lasso is drawn on the UMAP map with the mouse.
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from tests.utils.document_ingest import ingest_texts
from tests.utils.http_fault_proxy import HTTPFaultProxy
from tests.utils.profile_payload import profile_create_payload
from tests.utils.pylate_stub import serve_pylate_stub
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import register_tenant, serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TEXTS = {
    "rivers.txt": "Rivers carve canyons over thousands of years",
    "glaciers.txt": "Glaciers grind valleys into wide troughs",
    "volcanoes.txt": "Volcanoes build islands from cooling lava",
}
RIVERS = {
    "rivers-1.txt": "rivers carve deep canyons over thousands of years",
    "rivers-2.txt": "rivers carve canyons through soft rock over years",
    "rivers-3.txt": "old rivers carve wide canyons over many years",
    "rivers-4.txt": "rivers carve canyons and valleys over years",
}
VOLCANOES = {
    "volcanoes-1.txt": "volcanoes build islands from cooling lava flows",
    "volcanoes-2.txt": "volcanoes build new islands from lava",
    "volcanoes-3.txt": "lava from volcanoes build islands slowly",
    "volcanoes-4.txt": "volcanoes erupt lava that build islands",
}


@pytest.fixture(scope="module")
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        config_manager, schema_loader, workflow_state_redis_url
    ) as url:
        yield url


@pytest.fixture(scope="module")
def encoder(config_manager):
    with serve_pylate_stub() as url:
        system = config_manager.get_system_config()
        previous = dict(system.inference_service_urls)
        system.inference_service_urls["colbert_pylate"] = url
        config_manager.set_system_config(system)
        try:
            yield url
        finally:
            system = config_manager.get_system_config()
            system.inference_service_urls = previous
            config_manager.set_system_config(system)


def _ingest(runtime_url, config_manager, schema_loader, work_dir, texts):
    """Ingest on a thread of its own: the pipeline runs its own event loop,
    which cannot start on a thread a Playwright browser already drives."""

    def ingest():
        with httpx.Client(base_url=runtime_url, timeout=300) as client:
            return ingest_texts(client, config_manager, schema_loader, work_dir, texts)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ingest).result()


@pytest.fixture(scope="module")
def tenant(runtime_url, config_manager, schema_loader, encoder, tmp_path_factory):
    return _ingest(
        runtime_url,
        config_manager,
        schema_loader,
        tmp_path_factory.mktemp("atlas"),
        TEXTS,
    )


@pytest.fixture(scope="module")
def topics(runtime_url, config_manager, schema_loader, encoder, tmp_path_factory):
    """A tenant whose documents are about rivers or about volcanoes."""
    return _ingest(
        runtime_url,
        config_manager,
        schema_loader,
        tmp_path_factory.mktemp("topics"),
        {**RIVERS, **VOLCANOES},
    )


@pytest.fixture
def web_url(built_client, runtime_url):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
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
    register_tenant(tenant)
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
        labels = [
            "Schema",
            "Embedding",
            "Documents mapped",
            "Without an embedding",
            "Variance shown",
        ]
        # The panel renders its facts as the map loads; read them once every
        # label has its value.
        expect(panel.locator("dt")).to_have_text(labels)
        expect(panel.locator("dd")).to_have_count(len(labels))
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
                built_client, dead_runtime, telemetry_url=sink_url, built=True
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


# Budget for a process's first UMAP layout, which compiles numba kernels
# (measured 10-25 s on this host's runs of this module).
UMAP_FIRST_LAYOUT_S = 120


def _umap_form(page: Page, web_url: str, tenant: str, queries: str = ""):
    _atlas_view(page, web_url, tenant)
    form = page.get_by_role("form", name="Map documents")
    form.get_by_label("Profile").select_option("notes")
    form.get_by_label("Projection").select_option("UMAP with clusters")
    if queries:
        form.get_by_label("Queries (one per line)").fill(queries)
    return form


def _traces(page: Page, title: str):
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    expect(area).to_have_class(re.compile(r"\bjs-plotly-plot\b"))
    return area.evaluate(
        "el => el.data.map(t => ({type: t.type, name: t.name, x: t.x, y: t.y,"
        " text: t.text, customdata: t.customdata}))"
    )


def _bars(page: Page, title: str):
    figure = page.get_by_role("figure", name=title, exact=True)
    return list(
        zip(
            figure.locator(".bar-label").all_inner_texts(),
            figure.locator(".bar-value").all_inner_texts(),
            strict=True,
        )
    )


def _lasso(page: Page, title: str, points):
    """Draw a lasso with the mouse around ``points`` (data coordinates)."""
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    area.scroll_into_view_if_needed()
    pixels = area.evaluate(
        """(el, pts) => {
            const xa = el._fullLayout.xaxis, ya = el._fullLayout.yaxis;
            const box = el.getBoundingClientRect();
            return pts.map(([x, y]) => [
                box.left + xa._offset + xa.l2p(x),
                box.top + ya._offset + ya.l2p(y),
            ]);
        }""",
        points,
    )
    xs, ys = [p[0] for p in pixels], [p[1] for p in pixels]
    left, right, top, bottom = min(xs) - 12, max(xs) + 12, min(ys) - 12, max(ys) + 12
    page.mouse.move(left, top)
    page.mouse.down()
    for x, y in [(right, top), (right, bottom), (left, bottom), (left, top + 1)]:
        page.mouse.move(x, y, steps=8)
    page.mouse.up()


class TestUmapAtlas:
    def test_the_umap_map_clusters_places_queries_and_answers_a_lasso(
        self, page, web_url, runtime_url, topics
    ):
        queries = ["rivers carve canyons", "volcanoes build islands"]
        form = _umap_form(page, web_url, topics, "\n".join(queries))
        form.get_by_role("button", name="Show map").click()
        panel = page.get_by_role("region", name=f"UMAP map of notes for {topics}")
        # A first layout in a process compiles UMAP's numba kernels.
        expect(panel.locator('dl[aria-label="Map facts"]')).to_be_visible(
            timeout=UMAP_FIRST_LAYOUT_S * 1000
        )
        expected = httpx.post(
            f"{runtime_url}/admin/tenant/{topics}/embeddings/atlas/umap",
            json={"profile": "notes", "limit": 500, "queries": queries},
            timeout=300,
        ).json()
        labels = {c["id"]: c["label"] for c in expected["clusters"]}
        assert sorted(labels.values()) == [
            "rivers, canyons, carve",
            "volcanoes, build, islands",
        ]
        facts = panel.locator('dl[aria-label="Map facts"]')
        laid_out = page.evaluate(
            "iso => new Date(iso).toLocaleString()", expected["computed_at"]
        )
        assert dict(
            zip(
                facts.locator("dt").all_inner_texts(),
                facts.locator("dd").all_inner_texts(),
                strict=True,
            )
        ) == {
            "Schema": expected["schema_name"],
            "Embedding": "embedding, 128 dimensions",
            "Documents mapped": "8",
            "Without an embedding": "0",
            "Clusters": "2",
            "Queries placed": "2",
            "Laid out": laid_out,
        }

        title = "Documents of notes by UMAP"
        drawn = _traces(page, title)
        by_cluster = {
            labels[c]: sorted(
                (p["title"], p["x"], p["y"])
                for p in expected["points"]
                if p["cluster"] == c
            )
            for c in labels
        }
        assert [(t["type"], t["name"]) for t in drawn] == [
            ("scatter", labels[c]) for c in sorted(labels)
        ] + [("scatter", "Queries")]
        for trace in drawn[:-1]:
            assert (
                sorted(
                    (d[0], x, y)
                    for d, x, y in zip(trace["customdata"], trace["x"], trace["y"])
                )
                == by_cluster[trace["name"]]
            )
            assert {d[1] for d in trace["customdata"]} == {trace["name"]}
            assert {d[4] for d in trace["customdata"]} == {"Document"}
        rivers_trace = next(t for t in drawn if t["name"] == "rivers, canyons, carve")
        assert {d[0] for d in rivers_trace["customdata"]} == set(RIVERS)
        assert drawn[-1]["text"] == ["Query 1", "Query 2"]
        assert (drawn[-1]["x"], drawn[-1]["y"]) == (
            [q["x"] for q in expected["queries"]],
            [q["y"] for q in expected["queries"]],
        )

        analysis = page.get_by_role("region", name="Query analysis")
        for query in expected["queries"]:
            expect(
                analysis.get_by_role(
                    "list", name=f"Documents most similar to {query['label']}"
                ).locator("li")
            ).to_have_text(
                [
                    f"{d['title']} (similarity {d['similarity']:.3f})"
                    for d in query["similar"]
                ]
            )
        assert _bars(page, "Documents per cluster") == [
            (labels[c], "4") for c in sorted(labels)
        ]
        assert _bars(page, "Points by kind") == [("Documents", "8"), ("Queries", "2")]

        table = panel.get_by_role("table", name="Mapped documents")
        expect(table.locator("tbody tr")).to_have_count(8)
        rivers_points = [
            [p["x"], p["y"]]
            for p in expected["points"]
            if labels[p["cluster"]] == "rivers, canyons, carve"
        ]
        _lasso(page, title, rivers_points)
        expect(panel.get_by_label("Selection")).to_have_text(
            "Selection: 4 documents of 8.Clear selection"
        )
        expect(table.locator("tbody tr td:first-child")).to_have_text(sorted(RIVERS))
        expect(table.locator("tbody tr td:nth-child(2)")).to_have_text(
            ["rivers, canyons, carve"] * 4
        )
        assert _bars(page, "Documents per cluster") == [("rivers, canyons, carve", "4")]
        assert _bars(page, "Points by kind")[0] == ("Documents", "4")
        panel.get_by_role("button", name="Clear selection").click()
        expect(table.locator("tbody tr")).to_have_count(8)

        panel.get_by_label("Density").check()
        page.get_by_role("figure", name=title, exact=True).locator(
            ".plot-area"
        ).evaluate(
            "el => new Promise(resolve => { const done = () =>"
            " el.data[0].type === 'histogram2dcontour' ? resolve() :"
            " setTimeout(done, 50); done(); })"
        )
        density, *rest = _traces(page, title)
        assert (density["type"], len(density["x"]), len(rest)) == (
            "histogram2dcontour",
            8,
            3,
        )

    def test_a_layout_is_kept_until_recomputed(self, page, web_url, topics):
        form = _umap_form(page, web_url, topics)
        form.get_by_role("button", name="Show map").click()
        facts = page.get_by_role(
            "region", name=f"UMAP map of notes for {topics}"
        ).locator('dl[aria-label="Map facts"]')
        laid_out = facts.locator("dd").last
        expect(laid_out).not_to_have_text("", timeout=UMAP_FIRST_LAYOUT_S * 1000)
        first = laid_out.inner_text()
        form.get_by_role("button", name="Show map").click()
        expect(laid_out).to_have_text(first)
        page.wait_for_timeout(1100)
        form.get_by_role("button", name="Recompute").click()
        expect(laid_out).not_to_have_text(first)

    def test_too_few_documents_and_too_many_queries_are_refused(
        self, page, web_url, tenant
    ):
        form = _umap_form(page, web_url, tenant, "\n".join(["q"] * 11))
        form.get_by_role("button", name="Show map").click()
        expect(form.get_by_role("alert")).to_have_text(
            "Place at most 10 queries, one per line."
        )
        form.get_by_label("Queries (one per line)").fill("rivers")
        form.get_by_role("button", name="Show map").click()
        expect(form.get_by_role("alert")).to_have_text(
            "A UMAP map needs at least 4 documents with an embedding; profile "
            f"'notes' of tenant '{tenant}' has 3."
        )
        expect(page.get_by_role("figure")).to_have_count(0)


class TestEmptyAndFailedMaps:
    def test_a_profile_without_documents_says_so(self, page, web_url, runtime_url):
        empty = f"atlasempty{uuid.uuid4().hex[:6]}:main"
        templates = httpx.get(
            f"{runtime_url}/admin/profile-templates", params={"tenant_id": empty}
        ).json()["templates"]
        template = next(
            t["config"]
            for t in templates
            if t["profile_name"] == "document_text_semantic"
        )
        created = httpx.post(
            f"{runtime_url}/admin/profiles",
            json=profile_create_payload("notes", template, empty),
            timeout=300,
        )
        assert created.status_code == 201, created.text
        _atlas_view(page, web_url, empty)
        form = page.get_by_role("form", name="Map documents")
        form.get_by_label("Profile").select_option("notes")
        form.get_by_role("button", name="Show map").click()
        panel = page.get_by_role("region", name=f"Profile notes of {empty}")
        expect(panel.locator("p.muted")).to_have_text(
            "No documents with an embedding under notes."
        )
        expect(panel.get_by_role("figure")).to_have_count(0)

    def test_a_refused_document_read_shows_the_runtimes_reason(
        self, page, web_url, tenant, config_manager, vespa_instance
    ):
        from cogniverse_core.registries.backend_registry import BackendRegistry

        _atlas_view(page, web_url, tenant)
        form = page.get_by_role("form", name="Map documents")
        form.get_by_label("Profile").select_option("notes")
        http_port = vespa_instance["http_port"]
        registry = BackendRegistry.get_instance()
        with HTTPFaultProxy(lambda path: f"http://localhost:{http_port}") as proxy:
            system = config_manager.get_system_config()
            system.backend_port = proxy.port
            config_manager.set_system_config(system)
            registry.clear_instances()
            try:
                proxy.arm(
                    lambda method, path, body: (
                        method == "GET" and path.startswith("/document/v1/")
                    ),
                    failure=True,
                )
                proxy.release.set()
                form.get_by_role("button", name="Show map").click()
                expect(form.get_by_role("alert")).to_have_text(
                    "Reading the documents of profile 'notes' failed; the runtime "
                    "log names the cause."
                )
            finally:
                system = config_manager.get_system_config()
                system.backend_port = http_port
                config_manager.set_system_config(system)
                registry.clear_instances()
        assert proxy.entered.is_set()
        expect(page.get_by_role("figure")).to_have_count(0)
