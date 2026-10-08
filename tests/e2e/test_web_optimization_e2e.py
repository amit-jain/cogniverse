"""The deployed web client's optimization and review tools, driven in Chromium
against the e2e cluster.

Training examples uploaded through the Optimization runs view are read back
from the runtime's review history and shown again in the Approvals view; the
optimization report streamed from the cluster's ``detailed_report_agent`` is
compared with the JSON the page downloads, for the session's seeded tenant.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import expect

from cogniverse_core.approval.interfaces import approved_synthetic_dataset_name
from cogniverse_synthetic.approval.uploads import upload_templates
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER
from tests.e2e.cluster import RUNTIME, TENANT_ID
from tests.e2e.web_client import (
    RUN_TIMEOUT_MS,
    VIEW_TIMEOUT_MS,
    choose_tenant,
    minted_tenant,
    open_view,
)

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

# Phoenix serves spans and annotations shortly after they are written.
HISTORY_SETTLE_S = 120.0
KILN = [
    {
        "query": "kiln firing schedule",
        "enhanced_query": "stoneware kiln firing schedule cone six bisque",
        "expansion_terms": ["cone six", "bisque"],
        "synonyms": ["kiln program"],
        "context": "ceramics",
        "reasoning": "names the ware, the cone and the firing stage",
    },
    {
        "query": "glaze crazing",
        "enhanced_query": "glaze crazing cooling quartz inversion",
        "expansion_terms": ["quartz inversion"],
        "synonyms": ["glaze cracking"],
        "context": "ceramics",
        "reasoning": "crazing follows from cooling through quartz inversion",
    },
]


def _approved_history(tenant_id: str, batch: str) -> list[dict]:
    """The runtime's approved history items of ``batch``, once both are
    served."""
    deadline = time.monotonic() + HISTORY_SETTLE_S
    while True:
        response = httpx.get(
            f"{RUNTIME}/admin/tenant/{tenant_id}/approvals/history", timeout=120.0
        )
        assert response.status_code == 200, response.text
        items = [
            item for item in response.json()["approved"] if item["batch_id"] == batch
        ]
        if len(items) == len(KILN) or time.monotonic() > deadline:
            return sorted(items, key=lambda item: item["item_id"])
        time.sleep(5.0)


class TestTrainingExamples:
    def test_uploaded_examples_are_approved_and_listed_as_reviewed(
        self, page, tmp_path
    ):
        tenant_id = minted_tenant("webupload")
        open_view(page, "optimization")
        choose_tenant(page, tenant_id, "Show runs")
        panel = page.get_by_role("region", name="Upload training examples")
        expect(panel.get_by_label("Template optimizer").locator("option")).to_have_text(
            sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER), timeout=VIEW_TIMEOUT_MS
        )
        panel.get_by_label("Template optimizer").select_option("query_enhancement")
        with page.expect_download() as downloaded:
            panel.get_by_role("button", name="Download template").click()
        assert json.loads(Path(downloaded.value.path()).read_text()) == {
            "optimizer": "query_enhancement",
            "examples": [upload_templates()["query_enhancement"]["example"]],
        }

        kiln = tmp_path / "kiln.json"
        kiln.write_text(
            json.dumps({"optimizer": "query_enhancement", "examples": KILN})
        )
        panel.get_by_label("Examples files (JSON)").set_input_files([kiln])
        file = panel.get_by_role("region", name="File kiln.json")
        expect(file.locator("p.muted")).to_have_text(
            "Valid query_enhancement examples file (2 examples)."
        )
        panel.get_by_label("Uploaded by").fill("e2e-operator@example.com")
        file.get_by_role("button", name="Upload kiln.json").click()
        notice = page.get_by_role("status")
        prefix = (
            "Approved 2 query_enhancement examples from kiln.json into "
            f"{approved_synthetic_dataset_name(tenant_id)} as batch "
        )
        expect(notice).to_contain_text(prefix, timeout=RUN_TIMEOUT_MS)
        batch = re.fullmatch(
            re.escape(prefix) + r"(upload_query_enhancement_[0-9a-f]{32})\.",
            notice.inner_text(),
        ).group(1)

        history = _approved_history(tenant_id, batch)
        assert [
            (item["item_id"], item["status"], item["query"], item["reviewer"])
            for item in history
        ] == [
            (
                f"{batch}_{index}",
                "approved",
                example["query"],
                "e2e-operator@example.com",
            )
            for index, example in enumerate(KILN)
        ]

        open_view(page, "approvals")
        choose_tenant(page, tenant_id, "Show review queue")
        page.get_by_role("navigation", name="Approval sections").get_by_role(
            "button", name="Approved", exact=True
        ).click()
        approved = page.get_by_role("region", name=f"Approved items of {tenant_id}")
        table = approved.get_by_role("table", name="Approved items")
        expect(table.locator("tbody tr")).to_have_count(2, timeout=VIEW_TIMEOUT_MS)
        assert sorted(
            row.get_by_role("cell").all_inner_texts()
            for row in table.locator("tbody tr").all()
        ) == [
            [
                item["item_id"],
                item["query"],
                "1.00",
                "approved",
                "e2e-operator@example.com",
                item["reviewed_at"],
            ]
            for item in history
        ]


class TestOptimizationReport:
    def test_the_downloaded_report_is_the_one_the_page_shows(self, page):
        open_view(page, "optimization")
        choose_tenant(page, TENANT_ID, "Show runs")
        panel = page.get_by_role("region", name="Optimization report")
        expect(panel.locator("p.muted").first).to_have_text(
            "detailed_report_agent is registered with the runtime and writes the "
            "report.",
            timeout=VIEW_TIMEOUT_MS,
        )
        panel.get_by_role("button", name="Generate report").click()
        download = panel.get_by_role("button", name="Download report")
        expect(download).to_be_visible(timeout=RUN_TIMEOUT_MS)
        expect(panel.get_by_role("alert")).to_have_count(0)
        with page.expect_download() as downloaded:
            download.click()
        assert re.fullmatch(
            r"optimization_report_\d{8}_\d{6}\.json",
            downloaded.value.suggested_filename,
        )
        report = json.loads(Path(downloaded.value.path()).read_text())
        body = report if "executive_summary" in report else report["result"]
        shown = panel.get_by_label("Report", exact=True)
        expect(shown.locator("p.report-text")).to_have_text(body["executive_summary"])
        expect(shown.get_by_role("listitem")).to_have_text(
            [item for item in body.get("recommendations", []) if isinstance(item, str)]
        )
