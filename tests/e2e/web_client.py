"""Drive the deployed web client (``clients/web``) from e2e tests.

The web client's Node server runs in the cluster behind the ``web`` Service,
published on the host at ``WEB``. It talks to the runtime with the chart's
harness key, and the runtime maps that key to one tenant: every agent run the
page starts is that tenant's. The operations views name their tenant
explicitly, in each view's own chooser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import yaml
from playwright.sync_api import Page, expect

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from tests.e2e.cluster import K3S_VALUES, RUNTIME, TENANT_DEPLOY_TIMEOUT_S
from tests.e2e.conftest import (
    SAMPLE_VIDEO_CONTENT_ID,
    SAMPLE_VIDEO_PATH,
    WEB,
    _ensure_sample_content_ingested,
)
from tests.e2e.sample_corpus import _sample_video_media_type
from tests.e2e.test_api_e2e import PROFILE, _deploy_profile_for_tenant
from tests.e2e.test_pi_harness_e2e import rendered_config

CHART_VALUES = Path(K3S_VALUES).with_name("values.yaml")

# A view renders after one runtime round trip; a search or an agent turn pays
# the model's cold start (measured 49.7s cold against 1.8s warm), and the
# runtime's own agent call gives up at 120s.
VIEW_TIMEOUT_MS = 30_000
RUN_TIMEOUT_MS = 300_000

# The sidebar's Operations section, in the order the client declares it
# (clients/web/src/client/ops/views.ts). Restated so a view that disappears
# or is renamed fails here rather than being absorbed by a derivation.
OPS_VIEWS = (
    ("tenants", "Tenants"),
    ("profiles", "Backend profiles"),
    ("config", "Configuration"),
    ("ingestion", "Ingestion"),
    ("optimization", "Optimization runs"),
    ("memory", "Memory"),
    ("approvals", "Approvals"),
    ("annotations", "Annotation queue"),
    ("workflows", "Workflow reviews"),
    ("profile-metrics", "Profile metrics"),
    ("rlm-ab", "RLM A/B"),
    ("analytics", "Analytics"),
    ("evaluation", "Evaluation"),
    ("atlas", "Embedding atlas"),
    ("routing", "Routing evaluation"),
)


def web_harness_key() -> str:
    """The harness key the deployed web server sends: the chart's
    ``web.harnessKey``, as the k3s overlay leaves it."""
    base = yaml.safe_load(CHART_VALUES.read_text())["web"]
    overlay = yaml.safe_load(Path(K3S_VALUES).read_text()).get("web") or {}
    key = overlay.get("harnessKey", base["harnessKey"])
    assert isinstance(key, str) and key.strip(), (
        f"{K3S_VALUES} deploys the web client without a harness key"
    )
    return key


def web_tenant() -> str:
    """The tenant the shipped config maps the web server's key to."""
    keys = rendered_config()["harness"]["api_keys"]
    return canonical_tenant_id(keys["$COGNIVERSE_HARNESS_API_KEY"])


WEB_TENANT = web_tenant()


def ensure_web_tenant_corpus() -> str:
    """Deploy the video profile for the web tenant and ingest the tracked
    sample video into it, idempotently by content id; returns the content id.

    The web tenant is the deployment's own tenant (``config.tenants``), so the
    session keeps it rather than deleting it at teardown.
    """
    with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        _deploy_profile_for_tenant(client, PROFILE, WEB_TENANT)
    content_id = _ensure_sample_content_ingested(
        SAMPLE_VIDEO_PATH,
        profile=PROFILE,
        media_type=_sample_video_media_type(SAMPLE_VIDEO_PATH),
        tenant_id=WEB_TENANT,
    )
    assert content_id == SAMPLE_VIDEO_CONTENT_ID, content_id
    return content_id


def agent_label(name: str) -> str:
    """The client's display name for an agent (``agentLabel`` in api.ts)."""
    words = [word for word in re.sub(r"_agent$", "", name).split("_") if word]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def registered_agents() -> list[str]:
    response = httpx.get(f"{RUNTIME}/agents/", timeout=60.0)
    assert response.status_code == 200, response.text
    return response.json()["agents"]


def open_agent(page: Page, agent: str) -> str:
    """Open ``agent``'s workspace; returns the thread the page opened."""
    label = agent_label(agent)
    page.goto(f"{WEB}/#/agents/{agent}", timeout=VIEW_TIMEOUT_MS)
    expect(page.get_by_role("heading", name=label, level=1)).to_be_visible(
        timeout=VIEW_TIMEOUT_MS
    )
    expect(page.get_by_placeholder(f"Ask {label}…")).to_be_visible()
    page.wait_for_function(
        f"window.location.hash.startsWith('#/agents/{agent}/')",
        timeout=VIEW_TIMEOUT_MS,
    )
    return thread_of(page, agent)


def thread_of(page: Page, agent: str) -> str:
    prefix = f"#/agents/{agent}/"
    hash_ = page.evaluate("window.location.hash")
    assert hash_.startswith(prefix), hash_
    return hash_.removeprefix(prefix)


def send(page: Page, agent: str, text: str) -> None:
    box = page.get_by_placeholder(f"Ask {agent_label(agent)}…")
    box.fill(text)
    box.press("Enter")


def user_messages(page: Page):
    return page.get_by_test_id("copilot-user-message")


def assistant_messages(page: Page):
    return page.get_by_test_id("copilot-assistant-message")


def saved_turns(thread: str) -> dict:
    """The web tenant's saved turns of ``thread``, read back from the runtime."""
    response = httpx.get(
        f"{RUNTIME}/ag-ui/threads/{thread}",
        headers={"Authorization": f"Bearer {web_harness_key()}"},
        timeout=60.0,
    )
    assert response.status_code == 200, response.text
    return response.json()


def run_agent_and_capture_state(page: Page, agent: str, text: str) -> dict:
    """Send ``text`` to ``agent`` and return the run's final state, as the
    browser received it in the run's ``STATE_SNAPSHOT`` event."""
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and response.url.split("?", 1)[0].endswith(
                f"/ui-api/copilotkit/agent/{agent}/run"
            )
        ),
        timeout=RUN_TIMEOUT_MS,
    ) as captured:
        send(page, agent, text)
    snapshots = [
        event["snapshot"]
        for event in sse_events(captured.value.text())
        if event.get("type") == "STATE_SNAPSHOT"
    ]
    assert len(snapshots) == 1, (
        f"the {agent} run must carry exactly one STATE_SNAPSHOT; got {snapshots}"
    )
    return snapshots[0]


def sse_events(body: str) -> list[dict]:
    """The JSON events of a server-sent event stream, in order."""
    events = []
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line.removeprefix("data:").strip()
            if payload:
                events.append(json.loads(payload))
    return events


def result_cards(state: dict) -> list[dict]:
    """The cards the client renders for a search run's final state
    (``resultsOf`` in ResultCards.tsx): id, the id a rating names, title and
    the score the set is ranked by, formatted as the card shows it."""

    def text(value):
        return value if isinstance(value, str) and value.strip() else None

    def number(value):
        return (
            value
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    cards = []
    for hit in (state.get("result") or {}).get("results") or []:
        if not isinstance(hit, dict):
            continue
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        card_id = (
            text(hit.get("id"))
            or text(hit.get("document_id"))
            or text(hit.get("image_id"))
            or text(hit.get("audio_id"))
            or text(hit.get("video_id"))
            or text(metadata.get("video_id"))
        )
        if card_id is None:
            continue
        score = next(
            (
                number(hit.get(key))
                for key in ("rrf_score", "score", "relevance_score")
                if number(hit.get(key)) is not None
            ),
            None,
        )
        title = (
            text(hit.get("title"))
            or text(metadata.get("title"))
            or text(metadata.get("video_title"))
        )
        cards.append(
            {
                "id": card_id,
                "rating_id": text(hit.get("document_id"))
                or text(hit.get("documentid"))
                or text(hit.get("id"))
                or text(hit.get("source_id"))
                or text(hit.get("image_id"))
                or text(hit.get("audio_id"))
                or text(hit.get("video_id")),
                "title": title or card_id,
                "score": None if score is None else f"{score:.3f}",
            }
        )
    return cards


def open_view(page: Page, view: str) -> None:
    """Open the operations view ``view`` by its id."""
    label = dict(OPS_VIEWS)[view]
    page.goto(f"{WEB}/#/ops/{view}", timeout=VIEW_TIMEOUT_MS)
    expect(page.get_by_role("heading", name=label, level=1)).to_be_visible(
        timeout=VIEW_TIMEOUT_MS
    )


def choose_tenant(page: Page, tenant: str, action: str) -> None:
    """Pick ``tenant`` in the open view's tenant chooser."""
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name=action).click()


def facts(region) -> dict[str, str]:
    """A ``<dl>``'s terms and values, as rendered."""
    return dict(
        zip(
            region.locator("dt").all_inner_texts(),
            region.locator("dd").all_inner_texts(),
            strict=True,
        )
    )
