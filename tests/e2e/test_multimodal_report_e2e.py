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
"""

import time
from pathlib import Path

import httpx
import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = REPO_ROOT / "data/testset/evaluation/sample_videos/v_-D1gdv_gQyw.mp4"
RUNTIME_URL = "http://localhost:33000"
VESPA_URL = "http://localhost:33080"
TENANT_FULL_ID = "flywheel_org:production"
PROFILE = "video_colpali_smol500_mv_frame"
SCHEMA_NAME = "video_colpali_smol500_mv_frame_flywheel_org_production"


def _service_up(url: str, timeout: float = 3.0) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _tenant_registered() -> bool:
    try:
        resp = requests.get(
            f"{RUNTIME_URL}/admin/organizations/flywheel_org/tenants", timeout=5
        )
        if resp.status_code != 200:
            return False
        return any(
            t.get("tenant_full_id") == TENANT_FULL_ID
            for t in resp.json().get("tenants", [])
        )
    except requests.RequestException:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not SAMPLE_VIDEO.exists(),
        reason=f"sample video missing: {SAMPLE_VIDEO}",
    ),
]


def _wait_terminal(ingest_id: str, deadline_s: int = 600) -> dict:
    deadline = time.time() + deadline_s
    last = None
    while time.time() < deadline:
        resp = requests.get(f"{RUNTIME_URL}/ingestion/{ingest_id}/status", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        last = payload.get("latest", {})
        if payload.get("state") in ("complete", "completed", "failed", "error"):
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
    assert final["state"] in ("complete", "completed"), final
    result = final["latest"]["result"]
    assert result["keyframes"] == 19, f"expected 19 keyframes, got {result}"
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

    # The dispatch ran a real search — the agent was not fed an empty result set.
    assert metadata["results_analyzed"] > 0, (
        f"detailed-report dispatch must run a real search, got metadata={metadata}"
    )
    # THE contract: retrieved keyframes reached the answer LLM.
    assert metadata["keyframes_attached"] > 0, (
        "no keyframes reached the answer LLM — frames did not flow from the "
        f"search hits into the report agent. metadata={metadata}"
    )
    # And the model produced a real report over them.
    assert len(result["executive_summary"]) > 50, (
        f"executive_summary too short to be grounded: {result['executive_summary']!r}"
    )
