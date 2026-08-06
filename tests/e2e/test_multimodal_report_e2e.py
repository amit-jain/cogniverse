"""End-to-end: retrieved video keyframes reach the answer LLM.

The full multimodal flow against the live cluster, in one test:

  upload a frame-profile video
    -> real ingestion extracts keyframes, uploads them to MinIO, and indexes
       every segment with an ``s3://`` source_url in Vespa
    -> POST the detailed-report agent
    -> the agent runs a real search, resolves each hit's keyframe from object
       storage, and attaches the frames to the answer LLM

and asserts the frames actually reached the model (``keyframes_attached > 0``).
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
from tests.e2e.test_ingestion_upload_e2e import EXPECTED_KEYFRAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = REPO_ROOT / "tests/system/resources/videos/v_-D1gdv_gQyw.mp4"
RUNTIME_URL = "http://localhost:33000"
VESPA_URL = "http://localhost:33080"
TENANT_FULL_ID = "flywheel_org:production"
PROFILE = "video_colpali_smol500_mv_frame"
SCHEMA_NAME = "video_colpali_smol500_mv_frame_flywheel_org_production"


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
    """Answer side of the contract: the report agent searches, resolves the
    hits' keyframes from MinIO, and attaches them to the LLM call. The
    ``keyframes_attached`` count in the response is the deterministic proof the
    frames reached the model (unit tests only fake this)."""
    resp = requests.post(
        f"{RUNTIME_URL}/agents/detailed_report_agent/process",
        json={
            "agent_name": "detailed_report_agent",
            "query": "describe the outdoor scene and what the person is doing",
            "context": {"tenant_id": TENANT_FULL_ID},
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

    # The dispatch ran a real search — the agent was not fed an empty result
    # set — bounded by the agent's max_results_to_analyze cap (20).
    results_analyzed = metadata["results_analyzed"]
    assert 1 <= results_analyzed <= 20, (
        f"detailed-report dispatch must run a real search within the "
        f"max_results_to_analyze cap, got metadata={metadata}"
    )
    # THE contract, as an exact value: keyframes attached == the analyzed
    # results capped at max_keyframes_to_llm (4). Every top hit's keyframe
    # resolves from MinIO for a just-ingested video, so the cap — not a
    # missing frame — is what bounds it.
    assert metadata["keyframes_attached"] == min(results_analyzed, 4), (
        "keyframes attached to the LLM must equal min(results_analyzed, 4); "
        f"got metadata={metadata}"
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
    # person is doing" over a clip of a man lighting a fire outdoors, so the
    # grounded summary must describe the person, the outdoor setting, and the
    # fire activity — not just be 50+ chars of anything. Robust to phrasing
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
        for t in ("fire", "smoke", "flame", "burn", "spark", "kindl", "ignit")
    ), f"report does not describe the fire-lighting activity: {summary_lc!r}"
