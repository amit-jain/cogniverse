"""End-to-end: retrieved video keyframes reach the answer LLM.

The full multimodal flow against the live cluster, in one test:

  upload a frame-profile video
    -> real ingestion extracts keyframes, uploads them to MinIO, and indexes
       every segment with an ``s3://`` source_url in Vespa
    -> POST the detailed-report agent
    -> the agent runs a real search, resolves each hit's keyframe from object
       storage, and attaches the frames to the answer LLM

and asserts the frames actually reached the model
(``keyframes_attached == min(len(search_results), 4)``).
The unit tests only exercise this with a faked MediaLocator / LM; this proves the
contract end to end across the real ingestion pipeline, MinIO, Vespa, the search
agent, and the report agent's LLM call.

The autouse E2E fixture provisions the stack. An unavailable runtime or tracked
video fixture fails with its exact endpoint or path.
"""

import time
from pathlib import Path

import httpx
import pytest
import requests

from cogniverse_runtime.ingestion_worker.status_api import TERMINAL_STATES
from tests.e2e.test_api_e2e import PROFILE
from tests.e2e.test_ingestion_upload_e2e import EXPECTED_KEYFRAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = REPO_ROOT / "tests/system/resources/videos/v_-D1gdv_gQyw.mp4"
RUNTIME_URL = "http://localhost:33000"
VESPA_URL = "http://localhost:33080"
TENANT_FULL_ID = "flywheel_org:production"
SCHEMA_NAME = f"{PROFILE}_{TENANT_FULL_ID.replace(':', '_')}"


pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module", autouse=True)
def _require_multimodal_prerequisites() -> None:
    if not SAMPLE_VIDEO.is_file():
        pytest.fail(
            f"multimodal fixture video unavailable at {SAMPLE_VIDEO}",
            pytrace=False,
        )
    health_url = f"{RUNTIME_URL}/health/live"
    try:
        response = requests.get(health_url, timeout=10)
    except requests.RequestException as exc:
        pytest.fail(
            f"cogniverse runtime unavailable at {health_url}: "
            f"{type(exc).__name__}: {exc}",
            pytrace=False,
        )
    if response.status_code != 200:
        pytest.fail(
            f"cogniverse runtime unavailable at {health_url}: "
            f"HTTP {response.status_code} body={response.text[:300]!r}",
            pytrace=False,
        )


def _wait_terminal(ingest_id: str, deadline_s: int = 2400) -> dict:
    # Same bound as the ingestion-upload e2e: a fresh force=true ingest of
    # the sample video re-runs the full pipeline, whose per-segment KG claim
    # extraction spends 60-90s of LM time per segment (~19 segments ≈
    # 20-30 min). 600s timed out mid-pipeline.
    deadline = time.time() + deadline_s
    last = None
    while time.time() < deadline:
        resp = requests.get(f"{RUNTIME_URL}/ingestion/{ingest_id}/status", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        last = payload.get("latest", {})
        # The worker emits queued / running / complete / failed; the terminal
        # pair matches status_api.TERMINAL_STATES.
        if payload["state"] in TERMINAL_STATES:
            return payload
        time.sleep(5)
    pytest.fail(f"ingest {ingest_id} did not reach terminal in {deadline_s}s: {last}")


@pytest.fixture(scope="module")
def ingested_video() -> dict:
    """Upload the frame-profile video, wait for terminal, return its identity.

    ``force=true`` so a prior ingest of the same bytes doesn't dedupe this run
    (the whole point is a fresh ingest that writes keyframes to MinIO)."""
    with open(SAMPLE_VIDEO, "rb") as f:
        resp = requests.post(
            f"{RUNTIME_URL}/ingestion/upload",
            files={"file": (SAMPLE_VIDEO.name, f, "video/mp4")},
            data={"profile": PROFILE, "tenant_id": TENANT_FULL_ID},
            params={"force": "true"},
            timeout=60,
        )
    assert resp.status_code == 200, (
        f"upload failed: {resp.status_code} {resp.text[:300]}"
    )
    upload = resp.json()
    assert upload["source_url"].startswith(f"s3://cogniverse-ingest/{TENANT_FULL_ID}/")
    final = _wait_terminal(upload["ingest_id"])
    assert final["state"] == "complete", final
    result = final["latest"]["result"]
    assert result["keyframes"] == EXPECTED_KEYFRAMES, (
        f"expected {EXPECTED_KEYFRAMES} keyframes, got {result}"
    )
    return {
        "video_id": result["video_id"],
        "source_url": upload["source_url"],
    }


def _search_results_for_ingested_video(ingested_video: dict) -> list[dict]:
    """Fetch the clip's own search hits from the production search endpoint."""
    resp = requests.post(
        f"{RUNTIME_URL}/search",
        json={
            "query": "describe the outdoor scene and what the person is doing",
            "profile": PROFILE,
            "tenant_id": TENANT_FULL_ID,
            "top_k": 20,
            "result_granularity": "segment",
            "filters": {"video_id": ingested_video["video_id"]},
        },
        timeout=120,
    )
    assert resp.status_code == 200, (
        f"search failed: {resp.status_code} {resp.text[:300]}"
    )
    body = resp.json()
    results = body["results"]
    assert body["results_count"] == len(results), (
        "search response must report the exact number of returned hits; "
        f"got body={body}"
    )
    return results


def test_indexed_segments_carry_s3_source_url(ingested_video):
    """Ingestion side of the contract: every indexed segment records the
    ``s3://`` source_url the answer path derives the keyframe bucket from —
    not the worker's local temp path."""
    video_id = ingested_video["video_id"]
    yql = f'select source_url from sources {SCHEMA_NAME} where video_id contains "{video_id}"'
    resp = httpx.post(f"{VESPA_URL}/search/", json={"yql": yql, "hits": 50}, timeout=15)
    resp.raise_for_status()
    children = resp.json().get("root", {}).get("children", [])
    source_urls = {c.get("fields", {}).get("source_url") for c in children}
    assert source_urls, f"no indexed segments found for video_id={video_id}"
    assert source_urls == {ingested_video["source_url"]}, (
        f"every segment must record the s3:// upload URL {ingested_video['source_url']}; "
        f"got {source_urls} — a file:// value means keyframes are unfetchable at answer time"
    )


def test_report_agent_attaches_retrieved_keyframes_to_llm(ingested_video):
    """Answer side of the contract: the report agent resolves a supplied hit
    list, attaches the hits' keyframes from MinIO, and grounds the summary in
    the firewood/fire clip."""
    search_results = _search_results_for_ingested_video(ingested_video)
    resp = requests.post(
        f"{RUNTIME_URL}/agents/detailed_report_agent/process",
        json={
            "agent_name": "detailed_report_agent",
            "query": "describe the outdoor scene and what the person is doing",
            "context": {
                "tenant_id": TENANT_FULL_ID,
                "search_results": search_results,
            },
        },
        timeout=300,
    )
    assert resp.status_code == 200, (
        f"report failed: {resp.status_code} {resp.text[:300]}"
    )
    body = resp.json()
    assert body["status"] == "success", body
    result = body["result"]
    metadata = result["metadata"]

    # The report is now grounded in the exact hit list the test supplied, so
    # the agent metadata must reflect that list, not a live search result set.
    assert metadata["results_analyzed"] == len(search_results), (
        "detailed-report dispatch must analyze the supplied hits exactly; "
        f"got metadata={metadata}, supplied={len(search_results)}"
    )
    assert metadata["keyframes_attached"] == min(len(search_results), 4), (
        "keyframes attached to the LLM must equal min(len(search_results), 4); "
        f"got metadata={metadata}, supplied={len(search_results)}"
    )

    # The report must be a REAL grounded summary, not the templated fallback
    # the agent emits when the answer LM call fails (e.g. a keyframe payload
    # overflow). Without this, a broken multimodal report is indistinguishable
    # from a real one — the fallback stub even echoes the query, so the
    # content checks below would be fooled.
    assert metadata.get("report_degraded") is False, (
        "detailed report degraded to the fallback stub — the answer LM call "
        f"failed: {metadata.get('report_degraded_reason')!r}"
    )

    # Content: the report answers "describe the outdoor scene and what the
    # person is doing" over the tracked clip (ground truth: a man in a yellow
    # t-shirt kneels in a wooded area and lights a stack of firewood with a
    # knife and fire starter), so the grounded summary must describe the
    # person, the outdoor setting, and the firewood/fire activity the attached
    # keyframes show — not just be 50+ chars of anything. Robust to phrasing
    # via concept membership. (The query is stripped first so a bare echo of
    # it can't satisfy these on its own.)
    summary_lc = (
        result["executive_summary"]
        .lower()
        .replace("describe the outdoor scene and what the person is doing", "")
    )
    assert any(
        t in summary_lc
        for t in ("man", "person", "individual", "figure", "people", "someone")
    ), f"report does not describe the person in the clip: {summary_lc!r}"
    assert any(
        t in summary_lc
        for t in (
            "outdoor",
            "outside",
            "nature",
            "wooded",
            "forest",
            "dirt",
            "ground",
            "field",
            "wilderness",
            "gravel",
            "terrain",
            "scrub",
        )
    ), f"report does not describe the outdoor scene: {summary_lc!r}"
    assert any(
        t in summary_lc
        for t in (
            "fire",
            "smoke",
            "flame",
            "burn",
            "spark",
            "kindl",
            "ignit",
            "firewood",
            "wood",
            "log",
            "knife",
            "camp",
            "survival",
        )
    ), f"report does not describe the firewood/fire activity: {summary_lc!r}"
