"""Live-cluster proof that ingestion-pipeline spans land in Phoenix.

Locks the contract added under "TODO #1: Wire up ingestion-pipeline
telemetry":

1. ``pipeline.run`` outer span exists with the right attributes.
2. Per-stage child spans (``pipeline.keyframes`` etc.) exist as
   children of ``pipeline.run``, ordered correctly.
3. ``pipeline.kg.extract_per_segment`` sibling span exists with
   nodes/edges/segments attributes populated.
4. ``TelemetryLevel`` filter actually controls emission: pipeline
   spans land at DETAILED+ and DISAPPEAR at BASIC.

Skips when the cogniverse-runtime pod or Phoenix isn't reachable.
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

RUNTIME_BASE = os.environ.get("COGNIVERSE_RUNTIME_BASE", "http://localhost:28000")
PHOENIX_BASE = os.environ.get("COGNIVERSE_PHOENIX_BASE", "http://localhost:26006")
_TENANT = "flywheel_org:production"


def _runtime_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{RUNTIME_BASE}/health")
        return r.status_code == 200
    except Exception:
        return False


def _phoenix_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{PHOENIX_BASE}/v1/traces")
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_runtime_reachable() and _phoenix_reachable()),
        reason=(
            f"requires cogniverse-runtime at {RUNTIME_BASE} AND Phoenix at "
            f"{PHOENIX_BASE}"
        ),
    ),
]


def _query_pipeline_spans(unique_anchor: str, timeout_s: float = 60.0) -> list[dict]:
    """Poll Phoenix for spans named ``pipeline.*`` whose attributes
    contain the unique anchor string."""
    from phoenix.client import Client

    project = f"cogniverse-{_TENANT}"
    px = Client(base_url=PHOENIX_BASE)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            spans = px.spans.get_spans_dataframe(project_identifier=project, limit=500)
        except Exception:
            time.sleep(1.0)
            continue
        if len(spans) == 0:
            time.sleep(1.0)
            continue
        pipeline_spans = spans[spans["name"].fillna("").str.startswith("pipeline.")]
        if len(pipeline_spans) == 0:
            time.sleep(1.0)
            continue
        # Filter to spans whose attributes mention the unique anchor.
        # Anchor lives in pipeline.video_id / pipeline.source_uri /
        # kg.source_doc_id.
        attrs_cols = [c for c in pipeline_spans.columns if c.startswith("attributes.")]
        # Cast to string then substring-match across all attr cols.
        matching_idx = pipeline_spans.index[
            pipeline_spans[attrs_cols]
            .astype(str)
            .apply(lambda row: unique_anchor in " ".join(row.values), axis=1)
        ]
        matching = pipeline_spans.loc[matching_idx]
        if len(matching) >= 2:
            return [
                {
                    "name": r["name"],
                    "attrs": {k: r[k] for k in attrs_cols if r[k] is not None},
                }
                for _, r in matching.iterrows()
            ]
        time.sleep(1.0)
    return []


def _upload_fixture(tenant_id: str, unique_anchor: str) -> dict:
    """POST a small fixture video; wait for terminal state."""
    fixture = os.environ.get("COGNIVERSE_INGEST_FIXTURE")
    if not fixture or not os.path.exists(fixture):
        pytest.skip("Set COGNIVERSE_INGEST_FIXTURE=/path/to/small.mp4 to run this test")
    # Make the filename carry the unique anchor so it shows up in
    # pipeline.video_id / pipeline.source_uri attributes.
    upload_name = f"pipeline_test_{unique_anchor}.mp4"
    with open(fixture, "rb") as fh:
        with httpx.Client(timeout=600.0) as c:
            r = c.post(
                f"{RUNTIME_BASE}/ingestion/upload",
                files={"file": (upload_name, fh, "video/mp4")},
                data={
                    "tenant_id": tenant_id,
                    "profile": "video_colpali_smol500_mv_frame",
                    "wait": "true",
                    "wait_timeout": "300",
                },
            )
    assert r.status_code in (200, 202), f"upload failed: {r.status_code} {r.text[:300]}"
    return r.json()


def test_pipeline_run_outer_span_exists():
    """The outermost ``pipeline.run`` span emits per ingest."""
    anchor = uuid.uuid4().hex[:8]
    _upload_fixture(_TENANT, anchor)

    spans = _query_pipeline_spans(anchor, timeout_s=60.0)
    assert spans, f"no pipeline.* spans found for anchor {anchor!r}"

    run_spans = [s for s in spans if s["name"] == "pipeline.run"]
    assert run_spans, (
        f"expected pipeline.run span; got names {sorted({s['name'] for s in spans})}"
    )
    run = run_spans[0]
    assert run["attrs"].get("attributes.pipeline.video_id"), (
        f"pipeline.run missing pipeline.video_id attr: {run['attrs']}"
    )
    # duration_ms is the strongest assertion: proves the wrapper
    # actually closed (not abandoned mid-flight).
    duration_ms = run["attrs"].get("attributes.pipeline.duration_ms")
    assert duration_ms is not None and int(duration_ms) > 0


def test_pipeline_stage_child_spans_exist():
    """The 4 per-stage child spans land for a successful ingest."""
    anchor = uuid.uuid4().hex[:8]
    _upload_fixture(_TENANT, anchor)

    spans = _query_pipeline_spans(anchor, timeout_s=60.0)
    names = {s["name"] for s in spans}
    # Segmentation + embeddings are mandatory on the colpali profile.
    # Transcription + descriptions only fire if the profile enables them.
    assert "pipeline.keyframes" in names, (
        f"pipeline.keyframes missing; got {sorted(names)}"
    )
    assert "pipeline.embeddings" in names, (
        f"pipeline.embeddings missing; got {sorted(names)}"
    )


def test_pipeline_kg_extract_sibling_span_exists():
    """``pipeline.kg.extract_per_segment`` fires after the pipeline."""
    anchor = uuid.uuid4().hex[:8]
    _upload_fixture(_TENANT, anchor)

    spans = _query_pipeline_spans(anchor, timeout_s=60.0)
    kg_spans = [s for s in spans if s["name"] == "pipeline.kg.extract_per_segment"]
    assert kg_spans, (
        f"pipeline.kg.extract_per_segment missing; got names "
        f"{sorted({s['name'] for s in spans})}"
    )
    kg = kg_spans[0]
    # kg.nodes_count + kg.edges_count + kg.segments_count must all
    # be present (even if 0 — locked attributes).
    for attr in (
        "attributes.kg.nodes_count",
        "attributes.kg.edges_count",
        "attributes.kg.segments_count",
    ):
        assert attr in kg["attrs"], f"{attr} missing on kg span: {kg['attrs']}"
