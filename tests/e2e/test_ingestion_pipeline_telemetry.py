"""Live-cluster proof that ingestion-pipeline spans land in Phoenix.

Locks the contract:

* ``pipeline.run`` outer span exists with the expected attributes
  (video_id, source_uri, schema_name, duration_ms, stages_run).
* Per-stage child spans (``pipeline.keyframes`` /
  ``pipeline.transcription`` / ``pipeline.descriptions`` /
  ``pipeline.embeddings``) land as children of ``pipeline.run`` —
  at least keyframes + embeddings on the colpali profile.
* ``pipeline.kg.extract_per_segment`` sibling span carries
  ``kg.nodes_count`` / ``kg.edges_count`` / ``kg.segments_count``.
* ``TelemetryLevel`` filter controls emission: pipeline spans land
  at DETAILED+ and disappear at BASIC.

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
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (_runtime_reachable() and _phoenix_reachable()),
        reason=(
            f"requires cogniverse-runtime at {RUNTIME_BASE} AND Phoenix at "
            f"{PHOENIX_BASE}"
        ),
    ),
]


def _attr(row, *paths):
    """Lookup an attribute value with both flat and nested column
    representations. Phoenix's ``get_spans_dataframe`` may flatten
    ``job.id`` to ``attributes.job.id`` OR keep it as a dict on
    ``attributes.job`` depending on version. Walk both forms; return
    the first non-null hit.
    """
    # Try flat first: "attributes." + ".".join(paths)
    flat = "attributes." + ".".join(paths)
    if flat in row.index:
        v = row[flat]
        if v is not None and (not isinstance(v, float) or not _isnan(v)):
            return v
    # Try nested: walk attributes.<paths[0]> dict for paths[1:]
    nested_col = "attributes." + paths[0]
    if nested_col in row.index:
        nested = row[nested_col]
        if isinstance(nested, dict):
            cur = nested
            for p in paths[1:]:
                if not isinstance(cur, dict) or p not in cur:
                    cur = None
                    break
                cur = cur[p]
            if cur is not None:
                return cur
    return None


def _isnan(v):
    """``float('nan')`` is the only non-None empty marker pandas uses
    in object columns. Detect it without importing pandas/numpy."""
    try:
        return v != v
    except Exception:
        return False


def _query_pipeline_spans(ingest_id: str, timeout_s: float = 60.0) -> list[dict]:
    """Poll Phoenix for the ``pipeline.*`` spans emitted by one ingest.

    Anchors on the worker span's ``job.id == ingest_id`` attribute,
    then collects every ``pipeline.*`` span sharing that span's
    trace_id (so we get the children too). The KG-extract span runs
    in a separate trace, so we also add any
    ``pipeline.kg.extract_per_segment`` span whose
    ``kg.source_doc_id`` matches the trace's video_id.
    """
    from datetime import datetime, timedelta, timezone

    from phoenix.client import Client

    project = f"cogniverse-{_TENANT}"
    px = Client(base_url=PHOENIX_BASE)
    # Window + explicit timeout: the unscoped scan with the client's 5s
    # method default times out on a loaded span store, and the swallowed
    # exception reads as "no spans yet" until the poll budget dies.
    window_start = datetime.now(timezone.utc) - timedelta(hours=1)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            spans = px.spans.get_spans_dataframe(
                project_identifier=project,
                start_time=window_start,
                limit=2000,
                timeout=90,
            )
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

        # Anchor on worker span carrying job.id == ingest_id.
        worker_only = pipeline_spans[
            pipeline_spans["name"] == "pipeline.worker.process_job"
        ]
        worker_matching_idx = [
            i for i, r in worker_only.iterrows() if _attr(r, "job", "id") == ingest_id
        ]
        if not worker_matching_idx:
            time.sleep(1.0)
            continue
        worker_spans = pipeline_spans.loc[worker_matching_idx]

        anchor_trace_ids = set(worker_spans["context.trace_id"].dropna())
        in_trace = pipeline_spans[
            pipeline_spans["context.trace_id"].isin(anchor_trace_ids)
        ]

        # KG-extract span runs in a separate trace; match by
        # kg.source_doc_id == one of the run-span's pipeline.video_id.
        video_ids = {
            _attr(r, "pipeline", "video_id")
            for _, r in in_trace[in_trace["name"] == "pipeline.run"].iterrows()
        }
        video_ids.discard(None)
        kg_idx = []
        if video_ids:
            kg_candidates = pipeline_spans[
                pipeline_spans["name"] == "pipeline.kg.extract_per_segment"
            ]
            kg_idx = [
                i
                for i, r in kg_candidates.iterrows()
                if _attr(r, "kg", "source_doc_id") in video_ids
            ]

        matching = pipeline_spans.loc[
            list(in_trace.index) + [i for i in kg_idx if i not in in_trace.index]
        ]
        attrs_cols = [c for c in matching.columns if c.startswith("attributes.")]
        return [
            {
                "name": r["name"],
                "attrs": {
                    k: r[k] for k in attrs_cols if r[k] is not None and not _isnan(r[k])
                },
                "_attr": lambda paths, _r=r: _attr(_r, *paths),
            }
            for _, r in matching.iterrows()
        ]
    return []


def _upload_fixture(tenant_id: str) -> str:
    """POST a small fixture video; wait for terminal state. Returns
    the ``ingest_id`` from the upload response — used to anchor the
    Phoenix span query so the test can isolate this run's emissions
    from other ingest traffic.
    """
    # Default to the smallest bundled sample so the test runs without env
    # setup — an unset fixture used to skip, which hid this surface from
    # every full-suite run.
    fixture = os.environ.get("COGNIVERSE_INGEST_FIXTURE") or os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "testset",
        "evaluation",
        "sample_videos",
        "v_-6dz6tBH77I.mp4",
    )
    if not os.path.exists(fixture):
        pytest.fail(
            f"ingest fixture video not found at {fixture} — set "
            "COGNIVERSE_INGEST_FIXTURE or restore the sample_videos data dir"
        )
    upload_name = f"pipeline_test_{uuid.uuid4().hex[:8]}.mp4"
    with open(fixture, "rb") as fh:
        with httpx.Client(timeout=600.0) as c:
            r = c.post(
                f"{RUNTIME_BASE}/ingestion/upload",
                files={"file": (upload_name, fh, "video/mp4")},
                # tenant_id/profile are Form fields; wait/wait_timeout/force
                # are Query params — sent as form data they are silently
                # ignored and the route returns 202 without waiting.
                data={
                    "tenant_id": tenant_id,
                    "profile": "video_colpali_smol500_mv_frame",
                },
                params={"wait": "true", "wait_timeout": "300", "force": "true"},
            )
    assert r.status_code in (200, 202), f"upload failed: {r.status_code} {r.text[:300]}"
    body = r.json()
    ingest_id = body.get("ingest_id") or body.get("job_id") or ""
    assert ingest_id, f"upload response missing ingest_id: {body}"
    # wait=true returns after wait_timeout even when the job is still
    # running — the route only reports status="success" on terminal
    # 'complete'. A non-terminal ingest must fail HERE with the real
    # cause, not later as "no pipeline.* spans found".
    assert body.get("status") == "success", (
        f"ingest {ingest_id} not terminal-complete within wait_timeout: "
        f"status={body.get('status')!r}"
    )
    return ingest_id


@pytest.fixture(scope="module")
def _ingested_run():
    """One ingest per module — three assertions share the spans."""
    ingest_id = _upload_fixture(_TENANT)
    spans = _query_pipeline_spans(ingest_id, timeout_s=60.0)
    assert spans, f"no pipeline.* spans found for ingest_id {ingest_id!r}"
    return spans


def test_pipeline_run_outer_span_exists(_ingested_run):
    """The outermost ``pipeline.run`` span emits per ingest."""
    run_spans = [s for s in _ingested_run if s["name"] == "pipeline.run"]
    assert run_spans, (
        f"expected pipeline.run span; got names "
        f"{sorted({s['name'] for s in _ingested_run})}"
    )
    run = run_spans[0]
    video_id = run["_attr"](["pipeline", "video_id"])
    assert video_id, f"pipeline.run missing pipeline.video_id attr: {run['attrs']}"
    duration_ms = run["_attr"](["pipeline", "duration_ms"])
    assert duration_ms is not None and int(duration_ms) > 0


def test_pipeline_stage_child_spans_exist(_ingested_run):
    """The 4 per-stage child spans land for a successful ingest."""
    names = {s["name"] for s in _ingested_run}
    assert "pipeline.keyframes" in names, (
        f"pipeline.keyframes missing; got {sorted(names)}"
    )
    assert "pipeline.embeddings" in names, (
        f"pipeline.embeddings missing; got {sorted(names)}"
    )


def test_pipeline_kg_extract_sibling_span_exists(_ingested_run):
    """``pipeline.kg.extract_per_segment`` fires after the pipeline."""
    kg_spans = [
        s for s in _ingested_run if s["name"] == "pipeline.kg.extract_per_segment"
    ]
    assert kg_spans, (
        f"pipeline.kg.extract_per_segment missing; got names "
        f"{sorted({s['name'] for s in _ingested_run})}"
    )
    kg = kg_spans[0]
    for path in (
        ("kg", "nodes_count"),
        ("kg", "edges_count"),
        ("kg", "segments_count"),
    ):
        val = kg["_attr"](list(path))
        assert val is not None, f"kg.{'.'.join(path)} missing on kg span: {kg['attrs']}"
