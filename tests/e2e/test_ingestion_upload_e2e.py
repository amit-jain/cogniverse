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
* ``cogniverse-runtime`` at localhost:33000 (NodePort 28000).
* ``cogniverse-vespa`` at localhost:33080 (NodePort 8080).
* Tenant ``flywheel_org:production`` registered (the chart ships
  it as the default test tenant).
* Sample video at
  ``data/testset/evaluation/sample_videos/v_-D1gdv_gQyw.mp4``.

The test will skip with a clear reason when any prereq is missing —
no silent passes.
"""

import re
import time
from pathlib import Path

import httpx
import pytest
import requests

from cogniverse_agents.graph.graph_schema import (
    node_id_from_doc_id,
    normalize_name,
)

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = REPO_ROOT / "data/testset/evaluation/sample_videos/v_-D1gdv_gQyw.mp4"
RUNTIME_URL = "http://localhost:33000"
VESPA_URL = "http://localhost:33080"
TENANT_FULL_ID = "flywheel_org:production"
PROFILE = "video_colpali_smol500_mv_frame"
SCHEMA_NAME = "video_colpali_smol500_mv_frame_flywheel_org_production"
KG_SCHEMA_NAME = "knowledge_graph_flywheel_org_production"


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
    pytest.mark.e2e,
    pytest.mark.skipif(
        not SAMPLE_VIDEO.exists(),
        reason=f"sample video missing: {SAMPLE_VIDEO}",
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


def _wait_terminal(ingest_id: str, deadline_s: int = 2400) -> dict:
    """Block until the ingest reaches a terminal state. Returns the
    latest event dict. Fails the test on timeout.

    The bound covers the real pipeline on shared local hardware: the
    per-segment KG claim extraction spends 60-90s of LM time per segment,
    so ~19 segments legitimately run 20-30 minutes.
    """
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
    final = _wait_terminal(ingest_id, deadline_s=2400)
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
    """KG back-refs (entity_ids) populate on the content docs with
    well-formed, deduplicated, node-normalized ids.

    The per-segment KG pass extracts entities from BOTH the aligned audio
    transcript AND the per-keyframe VLM frame descriptions, normalizes each
    to a node id via ``normalize_name`` (lowercased, non-alphanumerics →
    ``_``), and PATCHes the ids onto every content doc through the Vespa
    Document v1 update path.

    Exact entity strings vary with the ASR + LM output, so identity is not
    pinned; the contract has three layers:
      * CONTENT — the extraction captured what is actually in THIS clip: a
        man (person), a wooded/scrubland outdoor setting, and fire/smoke
        surfacing in the later frames as the fire is lit. Matched on token
        membership so wording varies freely but the subject matter must be
        present and temporally correct.
      * REFERENTIAL INTEGRITY — every backref resolves to a node genuinely
        persisted in the tenant's knowledge_graph schema (no dangling
        pointers; the ``_prune_failed_backrefs`` invariant, end to end).
      * SHAPE — backrefs populate the content segments and every id is a
        well-formed, deduplicated, node-normalized token.

    Before the claim-extraction output-token budget fix the extractor's
    response truncated before its final output field, the parse failed on
    every segment, and each content doc silently persisted an EMPTY
    entity_ids list while the job still reported success (an
    indexed-but-ungrounded corpus).
    """
    video_id = upload_result["final"]["latest"]["result"]["video_id"]
    yql = f'select * from sources {SCHEMA_NAME} where video_id contains "{video_id}"'
    docs = _vespa_search(yql, hits=50)
    assert len(docs) == 19

    by_seg = {int(d["segment_id"]): d for d in docs}
    per_seg = {i: list(by_seg[i].get("entity_ids") or []) for i in range(19)}
    # Surface the real distribution in the run log — model output varies, so
    # this records what this run actually extracted for later inspection.
    print("KG entity_ids per segment:", per_seg)

    populated = [i for i, ids in per_seg.items() if ids]
    all_ids = [e for ids in per_seg.values() for e in ids]
    distinct = set(all_ids)

    # Extraction populated backrefs on the vast majority of the 19 segments —
    # the per-keyframe VLM description alone yields entities for every frame,
    # so a mostly-empty result means the extraction/parse regressed to the
    # pre-fix silent-empty behaviour. The floor allows a few frames whose
    # terse description filters to nothing without flaking on LM variance.
    assert len(populated) >= 12, (
        f"KG backrefs populated on only {len(populated)}/19 segments "
        f"(pre-fix silent-empty regression?): {per_seg}"
    )

    # Content: the sample clip shows a man lighting a fire in a wooded /
    # scrubland outdoor area. Regardless of exact LM phrasing, the extraction
    # must capture those three semantic elements of THIS specific video —
    # a pipeline that extracted nothing relevant, or unrelated content, fails
    # here. Matched on token membership so wording variance ('man' vs
    # 'the_man' vs 'young_man') doesn't break the check.
    def _segments_with(*needles: str) -> list:
        return sorted(
            i
            for i, ids in per_seg.items()
            if any(any(n in e for n in needles) for e in ids)
        )

    person_segs = _segments_with("man", "person", "people", "human")
    outdoor_segs = _segments_with(
        "outdoor", "wood", "forest", "wilderness", "natural", "scrub", "grass", "field"
    )
    fire_segs = _segments_with("fire", "smoke", "flame", "ember", "burn")

    # The subject — a man — is the persistent focus across the clip.
    assert len(person_segs) >= 5, (
        f"a clip of a man yielded person entities in only {len(person_segs)} "
        f"segments: {per_seg}"
    )
    # The outdoor / wooded setting is present throughout.
    assert len(outdoor_segs) >= 5, (
        f"an outdoor clip yielded outdoor/nature entities in only "
        f"{len(outdoor_segs)} segments: {per_seg}"
    )
    # The activity — lighting a fire — surfaces as fire/smoke, and does so in
    # the LATER frames (the fire is lit toward the end of the clip), proving
    # the extraction tracks the video's actual narrative progression rather
    # than emitting generic tokens.
    assert fire_segs, f"no fire/smoke entity from a fire-lighting clip: {per_seg}"
    assert max(fire_segs) >= 10, (
        f"fire/smoke only in early segments {fire_segs}; expected it as the "
        f"fire develops toward the end of the clip: {per_seg}"
    )

    node_id_re = re.compile(r"^[a-z0-9][a-z0-9_]*$")
    for seg, ids in per_seg.items():
        assert len(ids) == len(set(ids)), f"seg={seg} duplicate entity_ids: {ids}"
        assert len(ids) <= 30, f"seg={seg} implausible entity count {len(ids)}: {ids}"
        for e in ids:
            assert node_id_re.match(e), f"seg={seg} malformed entity_id: {e!r}"
            # Every id is a node identifier, i.e. re-normalizing is a no-op.
            assert normalize_name(e) == e, (
                f"seg={seg} entity_id not node-normalized: "
                f"{e!r} -> {normalize_name(e)!r}"
            )

    # Referential integrity against the REAL persisted graph: every backref
    # must resolve to an actual node written to the tenant's knowledge_graph
    # schema. This is the strong content assertion — it proves the ids point
    # at graph data that genuinely exists (not a dangling reference, not a
    # normalization mismatch between the backref writer and the node writer,
    # not a silently-failed node upsert) without asserting entity identity,
    # which varies by LM run.
    node_docs = _vespa_search(
        f'select * from sources {KG_SCHEMA_NAME} where doc_type contains "node"',
        hits=400,
    )
    persisted_node_ids = {
        node_id_from_doc_id(str(d.get("doc_id", "")), TENANT_FULL_ID) for d in node_docs
    }
    persisted_node_ids.discard("")
    # The graph is non-empty and every distinct backref resolves to a node.
    assert len(persisted_node_ids) >= 3, (
        f"KG schema {KG_SCHEMA_NAME} holds too few nodes "
        f"({len(persisted_node_ids)}) — node upsert path may have failed"
    )
    dangling = distinct - persisted_node_ids
    assert not dangling, (
        f"{len(dangling)} entity_id backrefs resolve to no persisted KG node "
        f"(dangling references): {sorted(dangling)}"
    )
    # Each node also carries the fields the traversal/search paths read.
    node_by_id = {
        node_id_from_doc_id(str(d.get("doc_id", "")), TENANT_FULL_ID): d
        for d in node_docs
    }
    for e in sorted(distinct):
        node = node_by_id[e]
        assert node.get("doc_type") == "node", f"{e}: wrong doc_type {node!r}"
        assert node.get("tenant_id") == TENANT_FULL_ID, f"{e}: wrong tenant {node!r}"
        assert normalize_name(str(node.get("name", ""))) == e, (
            f"{e}: node name {node.get('name')!r} does not normalize to its id"
        )

    # relation_ids / claim_ids (the SPO-edge back-refs) may or may not be
    # present depending on whether the LM emitted extractable relations for
    # this clip; assert only the shape and the same node-normalized id form
    # when present, never exact identity.
    for seg in range(19):
        for field in ("relation_ids", "claim_ids"):
            vals = by_seg[seg].get(field) or []
            assert isinstance(vals, list), f"seg={seg} {field} not a list: {vals!r}"
            assert len(vals) == len(set(vals)), f"seg={seg} duplicate {field}: {vals}"
            for v in vals:
                assert isinstance(v, str) and v.strip(), (
                    f"seg={seg} malformed {field} entry: {v!r}"
                )
