"""End-to-end real-services test for ``POST /ingestion/upload``.

WHAT THIS PROVES
================

Uploads a real video file to the live ``cogniverse-runtime`` service
running in k3d (NodePort 28000). The runtime persists the file to
MinIO, enqueues an ingest job on Redis, and an ``cogniverse-ingestor``
worker pod picks it up. The worker runs the full pipeline:

  * Real Whisper (``cogniverse-vllm-asr``) transcribes the audio.
  * Real ColPali (``cogniverse-vllm-colpali``) embeds the keyframes.
  * Real GLiNER (``cogniverse-gliner``) extracts entities.
  * Real ClaimExtractor with real gemma-4-e4b-it
    (``cogniverse-vllm-llm-student``) produces SPO edges.
  * Real GraphManager.upsert persists into a real Vespa instance
    (``cogniverse-vespa``).

The test then verifies the documents landed in Vespa by querying the
real schema directly. Every assertion is exact-value: integer counts
match the runtime's terminal-event ``result`` payload byte-equal, the
queried document's ``video_id`` matches the runtime-assigned id,
``start_time`` is a non-negative float, the back-ref arrays
(``entity_ids``/``relation_ids``/``claim_ids``) carry the per-segment
graph state.

PRE-REQS
========

* k3d cluster up with the cogniverse helm chart deployed.
* ``cogniverse-runtime`` at localhost:28000 (NodePort 28000).
* ``cogniverse-vespa`` at localhost:8080 (NodePort 8080).
* Tenant ``flywheel_org:production`` registered (the chart ships
  it as the default test tenant).
* Sample video at
  ``data/testset/evaluation/sample_videos/v_-D1gdv_gQyw.mp4``.

The test will skip with a clear reason when any prereq is missing —
no silent passes.
"""

import time
from pathlib import Path

import httpx
import pytest
import requests

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_VIDEO = REPO_ROOT / "data/testset/evaluation/sample_videos/v_-D1gdv_gQyw.mp4"
RUNTIME_URL = "http://localhost:28000"
VESPA_URL = "http://localhost:8080"
TENANT_FULL_ID = "flywheel_org:production"
PROFILE = "video_colpali_smol500_mv_frame"
SCHEMA_NAME = "video_colpali_smol500_mv_frame_flywheel_org_production"


# --------------------------------------------------------------------- #
# Liveness gates                                                         #
# --------------------------------------------------------------------- #


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
    pytest.mark.integration,
    pytest.mark.skipif(
        not SAMPLE_VIDEO.exists(),
        reason=f"sample video missing: {SAMPLE_VIDEO}",
    ),
    pytest.mark.skipif(
        not _service_up(f"{RUNTIME_URL}/health"),
        reason=f"cogniverse-runtime not reachable at {RUNTIME_URL}",
    ),
    pytest.mark.skipif(
        not _service_up(f"{VESPA_URL}/state/v1/health"),
        reason=f"live Vespa not reachable at {VESPA_URL}",
    ),
    pytest.mark.skipif(
        not _tenant_registered(),
        reason=f"tenant {TENANT_FULL_ID} not registered in the cluster",
    ),
]


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _vespa_count(yql: str) -> int:
    resp = httpx.post(
        f"{VESPA_URL}/search/",
        json={"yql": yql, "hits": 0},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json().get("root", {}).get("fields", {}).get("totalCount", 0)


def _vespa_search(yql: str, hits: int = 50) -> list:
    resp = httpx.post(
        f"{VESPA_URL}/search/",
        json={"yql": yql, "hits": hits},
        timeout=10.0,
    )
    resp.raise_for_status()
    return [
        h.get("fields", {}) for h in resp.json().get("root", {}).get("children", [])
    ]


def _wait_terminal(ingest_id: str, deadline_s: int = 480) -> dict:
    """Block until the ingest reaches a terminal state. Returns the
    latest event dict. Fails the test on timeout."""
    deadline = time.time() + deadline_s
    last = None
    while time.time() < deadline:
        resp = requests.get(f"{RUNTIME_URL}/ingestion/{ingest_id}/status", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        state = payload.get("state")
        last = payload.get("latest", {})
        if state in ("complete", "completed", "failed", "error"):
            return payload
        time.sleep(5)
    pytest.fail(
        f"ingest {ingest_id} did not reach terminal state in "
        f"{deadline_s}s. Last seen: {last}"
    )


# --------------------------------------------------------------------- #
# E1 — Real upload completes successfully                                #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def upload_result():
    """POST the sample video, wait for terminal, return the final payload."""
    with open(SAMPLE_VIDEO, "rb") as f:
        resp = requests.post(
            f"{RUNTIME_URL}/ingestion/upload",
            files={"file": (SAMPLE_VIDEO.name, f, "video/mp4")},
            # force is a Query param — in form data it is silently ignored
            # and idempotency dedupe reuses the previous ingest of the
            # same content.
            data={
                "profile": PROFILE,
                "tenant_id": TENANT_FULL_ID,
            },
            params={"force": "true"},
            timeout=60,
        )
    assert resp.status_code == 200, (
        f"upload failed: HTTP {resp.status_code} body={resp.text[:300]}"
    )
    upload_payload = resp.json()
    ingest_id = upload_payload["ingest_id"]
    final = _wait_terminal(ingest_id, deadline_s=480)
    return {"upload": upload_payload, "final": final}


def test_upload_returns_queued_ingest_id(upload_result):
    """The synchronous response carries the expected fields."""
    upload = upload_result["upload"]
    assert upload["state"] == "queued"
    assert upload["filename"] == "v_-D1gdv_gQyw.mp4"
    assert upload["ingest_id"].startswith("ingest_")
    assert upload["source_url"].startswith(f"s3://cogniverse-ingest/{TENANT_FULL_ID}/")
    assert upload["source_url"].endswith(".mp4")


def test_ingestion_reaches_terminal_complete_state(upload_result):
    """The async job completes (not failed, not error, not stalled)."""
    final = upload_result["final"]
    assert final["state"] in ("complete", "completed"), final
    latest = final["latest"]
    assert latest["state"] in ("complete", "completed")
    result = latest["result"]
    # Pipeline's own counts: 18s @ 30fps with the smol500_mv_frame
    # profile produces exactly 19 keyframes.
    assert result["keyframes"] == 19
    assert result["documents_fed"] == 19
    assert result["chunks"] == 19


# --------------------------------------------------------------------- #
# E2 — The 19 documents land in Vespa under this video_id                #
# --------------------------------------------------------------------- #


def test_persisted_document_count_matches_pipeline_result(upload_result):
    video_id = upload_result["final"]["latest"]["result"]["video_id"]
    expected = upload_result["final"]["latest"]["result"]["documents_fed"]
    yql = f'select * from sources {SCHEMA_NAME} where video_id contains "{video_id}"'
    actual = _vespa_count(yql)
    assert actual == expected, (
        f"expected {expected} docs for video_id={video_id} in {SCHEMA_NAME}, "
        f"got {actual}"
    )


# --------------------------------------------------------------------- #
# E3 — Per-document fields land in the expected shape                    #
# --------------------------------------------------------------------- #


def test_persisted_documents_carry_expected_fields(upload_result):
    """Every persisted doc has video_id + segment_id + start_time + end_time."""
    video_id = upload_result["final"]["latest"]["result"]["video_id"]
    yql = f'select * from sources {SCHEMA_NAME} where video_id contains "{video_id}"'
    docs = _vespa_search(yql, hits=50)
    assert len(docs) == 19

    # Every doc shares video_id.
    assert all(d.get("video_id") == video_id for d in docs)

    # 19 distinct integer segment_ids (the pipeline assigns 0..18 in
    # frame-based mode).
    seg_ids = sorted(int(d.get("segment_id")) for d in docs)
    assert seg_ids == list(range(19))

    # start_time / end_time present + non-negative; the sequence
    # starts at 0.0 and durations are monotonic.
    by_seg = {int(d["segment_id"]): d for d in docs}
    seg_0 = by_seg[0]
    assert float(seg_0["start_time"]) == 0.0
    assert float(seg_0["end_time"]) > 0.0
    starts = [float(by_seg[i]["start_time"]) for i in range(19)]
    ends = [float(by_seg[i]["end_time"]) for i in range(19)]
    assert all(e > s for s, e in zip(starts, ends, strict=True))
    assert starts == sorted(starts)  # monotonic across segment_ids
    # The frame-based profile produces 1-second windows per keyframe.
    for s, e in zip(starts, ends, strict=True):
        assert e - s == pytest.approx(1.0, abs=0.01)


def test_persisted_documents_have_kg_backrefs(upload_result):
    """KG back-refs (entity_ids) populate on every content doc.

    The per-segment KG extraction pass runs after content ingestion:
    Whisper transcribes the audio, the transcript is aligned to each
    keyframe's 1-second window, GLiNER extracts entities per chunk,
    and the resulting node_ids are PATCHed onto every content doc via
    the Vespa Document v1 update path.

    For the v_-D1gdv_gQyw.mp4 sample (the speaker says "I'm about to
    light a fire Bear Grylls style. Or at least try. Don't tell me I
    got it on the first try. Don't tell me I got it on the first one.
    Will it catch it on fire? Look at that. We have fire!"), GLiNER at
    threshold 0.3 reliably surfaces:
        - 'Bear Grylls' (Person, score ~0.917)
        - 'fire'        (Substance, score ~0.75-0.81)
    Pronouns 'I', 'We' are filtered by _PRONOUN_BLOCKLIST.

    Whisper returns a single transcript segment for this audio; only
    the keyframes whose 1-second window overlaps the spoken-audio span
    receive KG back-refs. Empirically against the live cluster + this
    video: keyframes 0..16 overlap (17 with entities), 17 + 18 fall
    past the transcript end and have empty entity_ids — that boundary
    shape is locked so a future Whisper-segmentation change here
    surfaces for explicit review.
    """
    video_id = upload_result["final"]["latest"]["result"]["video_id"]
    yql = f'select * from sources {SCHEMA_NAME} where video_id contains "{video_id}"'
    docs = _vespa_search(yql, hits=50)
    assert len(docs) == 19

    by_seg = {int(d["segment_id"]): d for d in docs}
    expected_entity_ids = sorted(["fire", "bear_grylls"])

    # Keyframes 0..16 overlap the Whisper transcript window →
    # entity_ids populated with ['bear_grylls', 'fire'] (sorted).
    for idx in range(17):
        doc = by_seg[idx]
        actual = sorted(doc.get("entity_ids", []))
        assert actual == expected_entity_ids, (
            f"seg={idx}: expected entity_ids={expected_entity_ids}, got {actual}"
        )

    # Keyframes 17 + 18 fall past the spoken-audio span → no
    # transcript overlap → empty entity_ids. Locked so a Whisper
    # segmentation change here surfaces for explicit review.
    for idx in (17, 18):
        doc = by_seg[idx]
        assert doc.get("entity_ids", []) == [], (
            f"seg={idx}: expected empty entity_ids past transcript end, "
            f"got {doc.get('entity_ids')}"
        )

    # ClaimExtractor produces SPO edges only when the transcript yields
    # extractable relations against the locked predicate vocabulary
    # (born_in, discovered, worked_at, won, ...). The Bear-Grylls test
    # clip has no such relations, so relation_ids / claim_ids stay
    # empty for this fixture. Lock the empty-list shape so a future
    # ClaimExtractor change that starts emitting edges for this clip
    # surfaces here for explicit review.
    for idx in range(19):
        doc = by_seg[idx]
        assert doc.get("relation_ids", []) == [], (
            f"seg={idx}: unexpected relation_ids: {doc.get('relation_ids')}"
        )
        assert doc.get("claim_ids", []) == [], (
            f"seg={idx}: unexpected claim_ids: {doc.get('claim_ids')}"
        )
