"""The dashboard page body against a real runtime socket.

``app.py`` is the Streamlit entry script, so its ingestion, search, chat and
annotation behaviour only exists when the script runs. These tests drive the
real script with ``AppTest`` and answer its HTTP calls from a real uvicorn
server that speaks the runtime's routes, so what is asserted is the exact
wire request the page makes and the exact widgets it renders back.

Three contracts are pinned here:

* Ingestion submits the uploaded bytes to ``POST /ingestion/upload`` and
  reports the job's terminal state — never a success banner over a failure.
* Interactive Search renders exactly one result list for the one search it
  ran, labelled with the profile the runtime reported.
* Switching the active tenant drops the previous tenant's search results,
  chat and annotations before any tab renders.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
import pytest
import streamlit as st
import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from streamlit.testing.v1 import AppTest

from cogniverse_dashboard.ingestion import submit_video_ingestion
from cogniverse_dashboard.tabs.optimization import OPTIMIZATION_RUN_COLUMNS
from cogniverse_dashboard.utils.runtime_client import get_runtime_client

APP_PATH = "libs/dashboard/cogniverse_dashboard/app.py"

VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42cogniverse-test-video-bytes"
VIDEO_NAME = "clip.mp4"
DEFAULT_PROFILE = "video_colpali_smol500_mv_frame"


@dataclass
class RuntimeRecorder:
    """What the page sent, and what the runtime is scripted to answer."""

    registered_tenants: set = field(default_factory=lambda: {"acme:a", "acme:b"})
    uploads: List[Dict[str, Any]] = field(default_factory=list)
    status_polls: List[str] = field(default_factory=list)
    job_for_profile: Dict[str, str] = field(default_factory=dict)
    concurrent_uploads: int = 0
    max_concurrent_uploads: int = 0
    upload_barrier: Any = None
    upload_extra: Dict[str, Any] = field(default_factory=dict)
    searches: List[Dict[str, Any]] = field(default_factory=list)
    # ingest_id -> ordered status payloads, last one repeats
    status_script: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    upload_response: Dict[str, Any] = field(
        default_factory=lambda: {"status": 202, "body": None}
    )
    status_response_status: int = 200
    search_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    search_profile: str = "video_colqwen_omni_mv_chunk_30s"
    search_degraded: List[Dict[str, str]] = field(default_factory=list)
    search_span_id: str = "0123456789abcdef"
    # tenant_id -> the optimization runs GET .../optimize/runs answers with
    optimize_runs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    optimize_runs_status: int = 200
    optimize_runs_requests: List[Dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _build_app(recorder: RuntimeRecorder) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/agents/{agent_name}")
    async def agent(agent_name: str) -> Dict[str, str]:
        return {"name": agent_name}

    @app.get("/admin/tenants/{tenant_id}")
    async def tenant(tenant_id: str):
        if tenant_id in recorder.registered_tenants:
            return {"tenant_id": tenant_id}
        return JSONResponse(status_code=404, content={"detail": "unknown tenant"})

    @app.post("/ingestion/upload")
    async def upload(
        file: UploadFile,
        profile: str = Form(...),
        backend: str = Form(...),
        tenant_id: str = Form(...),
    ):
        content = await file.read()
        with recorder.lock:
            index = len(recorder.uploads)
            recorder.uploads.append(
                {
                    "filename": file.filename,
                    "content": content,
                    "content_type": file.content_type,
                    "profile": profile,
                    "backend": backend,
                    "tenant_id": tenant_id,
                }
            )
            recorder.job_for_profile[profile] = f"ingest-{index}"
            recorder.concurrent_uploads += 1
            recorder.max_concurrent_uploads = max(
                recorder.max_concurrent_uploads, recorder.concurrent_uploads
            )
        try:
            if recorder.upload_barrier is not None:
                await asyncio.wait_for(recorder.upload_barrier.wait(), timeout=60)
        finally:
            with recorder.lock:
                recorder.concurrent_uploads -= 1
        scripted = recorder.upload_response
        if scripted["status"] not in (200, 202):
            return JSONResponse(
                status_code=scripted["status"], content=scripted["body"]
            )
        body: Dict[str, Any] = {"ingest_id": f"ingest-{index}", "state": "queued"}
        body.update(recorder.upload_extra)
        return JSONResponse(status_code=scripted["status"], content=body)

    @app.get("/ingestion/{ingest_id}/status")
    async def status(ingest_id: str):
        with recorder.lock:
            recorder.status_polls.append(ingest_id)
        if recorder.status_response_status != 200:
            return JSONResponse(
                status_code=recorder.status_response_status,
                content={"detail": "ingestion status store unavailable"},
            )
        script = recorder.status_script.get(ingest_id)
        if not script:
            return JSONResponse(status_code=404, content={"detail": "no such ingest"})
        payload = script[0] if len(script) == 1 else script.pop(0)
        return payload

    @app.get("/admin/tenant/{tenant_id}/optimize/runs")
    async def optimize_runs(tenant_id: str, limit: int = 20):
        with recorder.lock:
            recorder.optimize_runs_requests.append(
                {"tenant_id": tenant_id, "limit": limit}
            )
        if recorder.optimize_runs_status != 200:
            return JSONResponse(
                status_code=recorder.optimize_runs_status,
                content={"detail": "Argo API unreachable: connection refused"},
            )
        return {"runs": recorder.optimize_runs.get(tenant_id, [])[:limit]}

    @app.post("/a2a/")
    async def a2a(request: Request):
        body = await request.json()
        metadata = body["params"]["metadata"]
        with recorder.lock:
            recorder.searches.append(metadata)
        tenant_id = metadata["tenant_id"]
        results = recorder.search_results.get(tenant_id, [])
        final = {
            "type": "final",
            "data": {
                "query": body["params"]["message"]["parts"][0]["text"],
                "search_mode": "single_profile",
                "profile": recorder.search_profile,
                "degraded_profiles": recorder.search_degraded,
                "results": results,
                "total_results": len(results),
                "span_id": recorder.search_span_id,
            },
        }

        def stream():
            frame = {
                "result": {
                    "status": {"message": {"parts": [{"text": json.dumps(final)}]}}
                }
            }
            yield f"data: {json.dumps(frame)}\n\n".encode()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


@pytest.fixture
def runtime():
    recorder = RuntimeRecorder()
    config = uvicorn.Config(
        _build_app(recorder), host="127.0.0.1", port=0, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started is True, "runtime stand-in did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    recorder.url = f"http://127.0.0.1:{port}"
    try:
        yield recorder
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def page(runtime, monkeypatch):
    """``app.py`` wired to the runtime stand-in, with caches isolated."""
    import cogniverse_foundation.config.utils as config_utils
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from tests.utils.memory_store import InMemoryConfigStore

    def _factory() -> ConfigManager:
        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_system_config(SystemConfig(agent_registry_url=runtime.url))
        return manager

    monkeypatch.setattr(config_utils, "create_default_config_manager", _factory)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield lambda: AppTest.from_file(APP_PATH, default_timeout=300)
    st.cache_data.clear()
    st.cache_resource.clear()


def _open_tenant(page, tenant_id: str) -> AppTest:
    app = page()
    app.run()
    app.text_input(key="active_tenant_input").set_value(tenant_id).run()
    assert [e.message for e in app.exception] == []
    return app


def _switch_tenant(app: AppTest, tenant_id: str) -> AppTest:
    app.text_input(key="active_tenant_input").set_value(tenant_id).run()
    assert [e.message for e in app.exception] == []
    return app


def _ingestion_messages(elements) -> List[str]:
    """Only the Ingestion Testing tab's own banners.

    The page also renders sidebar agent-status banners and per-tab
    telemetry errors, so an unfiltered list would not pin this tab.
    """
    return [
        element.value
        for element in elements
        if element.value.startswith(("video_", "Ingestion failed for ", "All "))
    ]


def _button(app: AppTest, label: str):
    matches = [b for b in app.button if b.label == label]
    assert len(matches) == 1, f"{label} -> {[b.label for b in app.button]}"
    return matches[0]


def _complete(documents_fed: int, video_id: str, chunks: int) -> Dict[str, Any]:
    return {
        "state": "complete",
        "latest": {
            "state": "complete",
            "result": {
                "video_id": video_id,
                "documents_fed": documents_fed,
                "chunks": chunks,
            },
        },
    }


def _upload_video(app: AppTest) -> AppTest:
    uploader = next(
        u for u in app.file_uploader if u.label == "Upload test video for ingestion"
    )
    uploader.set_value((VIDEO_NAME, VIDEO_BYTES, "video/mp4")).run()
    return app


def _result(document_id: str, video_id: str, score: float) -> Dict[str, Any]:
    return {
        "document_id": document_id,
        "score": score,
        "metadata": {"video_id": video_id, "description": f"{video_id} description"},
        "temporal_info": {"start_time": 0.0, "end_time": 3.0},
    }


# --------------------------------------------------------------------------
# B8 — ingestion submits bytes and reports the terminal job outcome
# --------------------------------------------------------------------------


def test_process_video_uploads_the_bytes_and_reports_the_fed_documents(page, runtime):
    runtime.status_script["ingest-0"] = [_complete(7, "clip_9f2", 3)]
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    _button(app, "🔄 Process Video").click().run()

    assert [e.message for e in app.exception] == []
    assert runtime.uploads == [
        {
            "filename": VIDEO_NAME,
            "content": VIDEO_BYTES,
            "content_type": "video/mp4",
            "profile": DEFAULT_PROFILE,
            "backend": "vespa",
            "tenant_id": "acme:a",
        }
    ]
    assert runtime.status_polls == ["ingest-0"]
    successes = [s.value for s in app.success]
    assert f"{DEFAULT_PROFILE}: fed 7 documents as clip_9f2" in successes
    assert "All 1 profiles ingested" in successes
    assert app.session_state["processing_results"] == [
        {
            "status": "success",
            "profile": DEFAULT_PROFILE,
            "ingest_id": "ingest-0",
            "video_id": "clip_9f2",
            "documents_fed": 7,
            "chunks_created": 3,
            "deduplicated": False,
        }
    ]


def test_failed_ingestion_is_reported_as_failure_not_a_success_banner(page, runtime):
    runtime.status_script["ingest-0"] = [
        {
            "state": "failed",
            "latest": {"state": "failed", "error": "keyframe extraction crashed"},
        }
    ]
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    _button(app, "🔄 Process Video").click().run()

    assert [e.message for e in app.exception] == []
    assert _ingestion_messages(app.success) == []
    assert _ingestion_messages(app.error) == [
        f"{DEFAULT_PROFILE}: Ingestion ingest-0 failed: keyframe extraction crashed",
        f"Ingestion failed for {DEFAULT_PROFILE} (1 of 1 profiles)",
    ]
    assert app.session_state["processing_results"][0]["status"] == "error"


def test_rejected_upload_names_the_http_failure_and_never_polls(page, runtime):
    runtime.upload_response = {
        "status": 503,
        "body": {"message": "object store unavailable"},
    }
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    _button(app, "🔄 Process Video").click().run()

    assert runtime.status_polls == []
    assert _ingestion_messages(app.success) == []
    reported = _ingestion_messages(app.error)
    assert len(reported) == 2
    assert reported[0].startswith(f"{DEFAULT_PROFILE}: Upload rejected: HTTP 503:")
    assert "object store unavailable" in reported[0]
    assert reported[1] == f"Ingestion failed for {DEFAULT_PROFILE} (1 of 1 profiles)"


def test_status_outage_mid_job_is_an_error_not_a_completed_ingestion(page, runtime):
    runtime.status_script["ingest-0"] = [_complete(4, "clip_x", 2)]
    runtime.status_response_status = 503
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    _button(app, "🔄 Process Video").click().run()

    assert _ingestion_messages(app.success) == []
    reported = _ingestion_messages(app.error)
    assert reported[0].startswith(
        f"{DEFAULT_PROFILE}: Ingestion status for ingest-0: HTTP 503:"
    )
    assert app.session_state["processing_results"][0]["status"] == "error"


def test_each_profile_gets_its_own_job_and_outcomes_do_not_cross(page, runtime):
    second = "video_xclip_sv_chunk_6s"
    runtime.status_script["ingest-0"] = [_complete(5, "clip_a", 2)]
    runtime.status_script["ingest-1"] = [
        {"state": "failed", "latest": {"state": "failed", "error": "profile missing"}}
    ]
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    next(m for m in app.multiselect if m.label == "Select profiles to test").set_value(
        [DEFAULT_PROFILE, second]
    ).run()
    _button(app, "🔄 Process Video").click().run()

    assert [u["profile"] for u in runtime.uploads] == [DEFAULT_PROFILE, second]
    assert [u["content"] for u in runtime.uploads] == [VIDEO_BYTES, VIDEO_BYTES]
    assert runtime.status_polls == ["ingest-0", "ingest-1"]
    assert [
        (r["profile"], r["status"], r.get("documents_fed"))
        for r in app.session_state["processing_results"]
    ] == [(DEFAULT_PROFILE, "success", 5), (second, "error", None)]
    assert _ingestion_messages(app.error) == [
        f"{second}: Ingestion ingest-1 failed: profile missing",
        f"Ingestion failed for {second} (1 of 2 profiles)",
    ]
    assert _ingestion_messages(app.success) == [
        f"{DEFAULT_PROFILE}: fed 5 documents as clip_a"
    ]


def test_reupload_of_ingested_bytes_is_reported_as_already_ingested(page, runtime):
    """MinIO is content-addressed, so the same bytes/profile/tenant dedupe.

    The runtime answers the resubmit with the earlier run's id and state and
    none of its counts; the tab reports it as ingested, not as a failure.
    """
    runtime.upload_extra = {
        "ingest_id": "ingest-earlier",
        "state": "complete",
        "existing": True,
    }
    runtime.status_script["ingest-earlier"] = [
        {
            "state": "complete",
            "latest": {
                "state": "complete",
                "ingest_id": "ingest-earlier",
                "existing": True,
            },
        }
    ]
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    _button(app, "🔄 Process Video").click().run()

    assert [e.message for e in app.exception] == []
    assert runtime.status_polls == ["ingest-earlier"]
    assert _ingestion_messages(app.error) == []
    assert _ingestion_messages(app.success) == [
        f"{DEFAULT_PROFILE}: Already ingested as ingest-earlier; nothing was re-fed",
        "All 1 profiles ingested",
    ]
    assert app.session_state["processing_results"] == [
        {
            "status": "success",
            "profile": DEFAULT_PROFILE,
            "ingest_id": "ingest-earlier",
            "deduplicated": True,
            "video_id": None,
            "documents_fed": None,
            "chunks_created": None,
            "message": ("Already ingested as ingest-earlier; nothing was re-fed"),
        }
    ]


def test_reupload_keeps_the_earlier_runs_counts_when_its_trail_survives(page, runtime):
    runtime.upload_extra = {
        "ingest_id": "ingest-earlier",
        "state": "complete",
        "existing": True,
    }
    runtime.status_script["ingest-earlier"] = [_complete(7, "clip_9f2", 3)]
    app = _open_tenant(page, "acme:a")
    _upload_video(app)
    _button(app, "🔄 Process Video").click().run()

    assert _ingestion_messages(app.error) == []
    assert app.session_state["processing_results"] == [
        {
            "status": "success",
            "profile": DEFAULT_PROFILE,
            "ingest_id": "ingest-earlier",
            "deduplicated": True,
            "video_id": "clip_9f2",
            "documents_fed": 7,
            "chunks_created": 3,
            "message": ("Already ingested as ingest-earlier; nothing was re-fed"),
        }
    ]


def test_submission_reports_every_state_the_job_passes_through(runtime):
    runtime.status_script["ingest-0"] = [
        {"state": "queued", "latest": {"state": "queued"}},
        {"state": "running", "latest": {"state": "running"}},
        _complete(3, "clip_t", 1),
    ]
    seen: List[str] = []
    with httpx.Client() as client:
        outcome = submit_video_ingestion(
            client,
            runtime.url,
            filename=VIDEO_NAME,
            content=VIDEO_BYTES,
            content_type="video/mp4",
            profile=DEFAULT_PROFILE,
            tenant_id="acme:a",
            sleep=lambda _seconds: None,
            on_state=seen.append,
        )

    assert seen == ["queued", "running", "complete"]
    assert runtime.status_polls == ["ingest-0", "ingest-0", "ingest-0"]
    assert outcome["documents_fed"] == 3


def test_concurrent_uploads_through_the_shared_client_keep_their_own_tenants(runtime):
    """The dashboard process holds one pooled client for every interaction.

    Two ingestions for two tenants overlapping inside the runtime must come
    back with their own job ids and their own counts, and each upload must
    carry its own tenant; a client that serialised them would never release
    the barrier.
    """
    second = "video_xclip_sv_chunk_6s"
    tenants = {DEFAULT_PROFILE: "acme:a", second: "acme:b"}
    runtime.upload_barrier = asyncio.Barrier(2)
    runtime.status_script["ingest-0"] = [_complete(5, "clip_0", 2)]
    runtime.status_script["ingest-1"] = [_complete(9, "clip_1", 4)]
    expected = {"ingest-0": (5, "clip_0"), "ingest-1": (9, "clip_1")}

    client = get_runtime_client()
    start = threading.Barrier(2)
    outcomes: Dict[str, Dict[str, Any]] = {}

    def submit(profile: str) -> None:
        start.wait(timeout=30)
        outcomes[profile] = submit_video_ingestion(
            client,
            runtime.url,
            filename=VIDEO_NAME,
            content=VIDEO_BYTES,
            content_type="video/mp4",
            profile=profile,
            tenant_id=tenants[profile],
            sleep=lambda _seconds: None,
        )

    threads = [
        threading.Thread(target=submit, args=(profile,))
        for profile in (DEFAULT_PROFILE, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert [thread.is_alive() for thread in threads] == [False, False]

    assert runtime.max_concurrent_uploads == 2
    assert sorted(runtime.job_for_profile) == sorted([DEFAULT_PROFILE, second])
    assert sorted(runtime.job_for_profile.values()) == ["ingest-0", "ingest-1"]
    assert sorted(runtime.status_polls) == ["ingest-0", "ingest-1"]
    assert {row["profile"]: row["tenant_id"] for row in runtime.uploads} == tenants
    assert sorted(outcomes) == sorted([DEFAULT_PROFILE, second])
    for profile, outcome in outcomes.items():
        job = runtime.job_for_profile[profile]
        assert (
            outcome["profile"],
            outcome["ingest_id"],
            outcome["documents_fed"],
            outcome["video_id"],
        ) == (profile, job, *expected[job])


# --------------------------------------------------------------------------
# M26 — one search, one result list, the profile the runtime actually used
# --------------------------------------------------------------------------


def _run_search(app: AppTest, query: str) -> AppTest:
    next(t for t in app.text_input if t.label == "Enter your search query").set_value(
        query
    ).run()
    _button(app, "🔍 Search").click().run()
    assert [e.message for e in app.exception] == []
    return app


def test_search_renders_one_result_list_for_the_single_executed_operation(
    page, runtime
):
    runtime.search_results["acme:a"] = [
        _result("doc-1", "video_a", 0.91),
        _result("doc-2", "video_b", 0.42),
    ]
    app = _open_tenant(page, "acme:a")
    _run_search(app, "robots")

    assert "Select profiles to test" in [m.label for m in app.multiselect]
    assert "Ranking Strategies" not in [m.label for m in app.multiselect]
    assert "Processing Profile" not in [s.label for s in app.selectbox]
    assert runtime.searches[-1]["top_k"] == 5
    assert "profile" not in runtime.searches[-1]
    assert [m.value for m in app.markdown if m.value.startswith("### 📊 Results")] == [
        "### 📊 Results (single_profile)"
    ]
    assert [s.value for s in app.success if s.value.startswith("Found")] == [
        "Found 2 results for 'robots'"
    ]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Results"] == "2"
    assert metrics["Profile"] == "video_colqwen_omni_mv_chunk_30s"
    assert len([b for b in app.button if b.label == "💾 Save Annotation"]) == 2
    stored = app.session_state["current_search_results"]
    assert stored["tenant_id"] == "acme:a"
    assert stored["profile"] == "video_colqwen_omni_mv_chunk_30s"
    assert stored["degraded_profiles"] == []
    assert [r["document_id"] for r in stored["results"]] == ["doc-1", "doc-2"]
    assert [w.value for w in app.warning if w.value.startswith("Partial results")] == []
    assert app.session_state["conversation_history"] == [
        {
            "query": "robots",
            "profile": "video_colqwen_omni_mv_chunk_30s",
            "timestamp": stored["timestamp"],
            "result_count": 2,
        }
    ]


# --------------------------------------------------------------------------
# M25 — a tenant switch shows and writes nothing from the previous tenant
# --------------------------------------------------------------------------


def test_tenant_switch_drops_the_previous_tenants_search_chat_and_annotations(
    page, runtime
):
    runtime.search_results["acme:a"] = [_result("doc-1", "video_a", 0.91)]
    runtime.search_results["acme:b"] = []
    app = _open_tenant(page, "acme:a")
    _run_search(app, "robots")
    app.session_state["chat_messages"] = [{"role": "user", "content": "hello A"}]
    app.session_state["search_annotations"] = [
        {"tenant_id": "acme:a", "query": "robots", "result_id": 0}
    ]
    app.session_state["orch_spans"] = ["A-span"]
    first_session_id = app.session_state["session_id"]

    _switch_tenant(app, "acme:b")

    assert app.session_state["current_tenant"] == "acme:b"
    assert "current_search_results" not in app.session_state
    assert app.session_state["chat_messages"] == []
    assert app.session_state["conversation_history"] == []
    assert app.session_state["search_annotations"] == []
    assert "orch_spans" not in app.session_state
    assert app.session_state["session_id"] != first_session_id
    assert [b.label for b in app.button if b.label == "💾 Save Annotation"] == []
    assert [m.value for m in app.markdown if m.value.startswith("### 📊 Results")] == []
    assert "video_a" not in "".join(
        str(m.value) for m in app.markdown if isinstance(m.value, str)
    )


def test_search_after_a_switch_is_scoped_to_the_new_tenant(page, runtime):
    runtime.search_results["acme:a"] = [_result("doc-1", "video_a", 0.91)]
    runtime.search_results["acme:b"] = [_result("doc-9", "video_b", 0.77)]
    app = _open_tenant(page, "acme:a")
    _run_search(app, "robots")
    _switch_tenant(app, "acme:b")
    _run_search(app, "robots")

    assert [s["tenant_id"] for s in runtime.searches] == ["acme:a", "acme:b"]
    stored = app.session_state["current_search_results"]
    assert stored["tenant_id"] == "acme:b"
    assert [r["document_id"] for r in stored["results"]] == ["doc-9"]
    assert {m.label: m.value for m in app.metric}["Results"] == "1"


def test_two_sessions_on_different_tenants_never_see_each_others_results(page, runtime):
    """Two browser sessions share this process's render caches.

    A cache keyed without the tenant would hand the second session the
    first session's answer, and a re-render of either would then show the
    other tenant's results.
    """
    runtime.search_results["acme:a"] = [_result("doc-1", "video_a", 0.91)]
    runtime.search_results["acme:b"] = [
        _result("doc-9", "video_b", 0.77),
        _result("doc-8", "video_c", 0.55),
    ]
    first = _open_tenant(page, "acme:a")
    _run_search(first, "robots")
    second = _open_tenant(page, "acme:b")
    _run_search(second, "robots")

    # Re-render each session after the other has run: a process-wide cache
    # hit would surface here.
    _button(first, "🔄 Refresh Now").click().run()
    _button(second, "🔄 Refresh Now").click().run()

    assert [row["tenant_id"] for row in runtime.searches] == ["acme:a", "acme:b"]
    assert first.session_state["current_search_results"]["tenant_id"] == "acme:a"
    assert second.session_state["current_search_results"]["tenant_id"] == "acme:b"
    assert [
        row["document_id"]
        for row in first.session_state["current_search_results"]["results"]
    ] == ["doc-1"]
    assert [
        row["document_id"]
        for row in second.session_state["current_search_results"]["results"]
    ] == ["doc-9", "doc-8"]
    assert {m.label: m.value for m in first.metric}["Results"] == "1"
    assert {m.label: m.value for m in second.metric}["Results"] == "2"
    assert first.session_state["session_id"] != second.session_state["session_id"]


def test_partially_failed_ensemble_is_marked_degraded_not_a_complete_result(
    page, runtime
):
    runtime.search_results["acme:a"] = [_result("doc-1", "video_a", 0.91)]
    runtime.search_degraded = [
        {"profile": "video_xclip_sv_chunk_6s", "reason": "query encoder unavailable"}
    ]
    app = _open_tenant(page, "acme:a")
    _run_search(app, "robots")

    assert app.session_state["current_search_results"]["degraded_profiles"] == [
        {"profile": "video_xclip_sv_chunk_6s", "reason": "query encoder unavailable"}
    ]
    # Streamlit lifts a leading emoji out of the body into the icon slot.
    assert [w.value for w in app.warning if w.value.startswith("Partial results")] == [
        "Partial results: video_xclip_sv_chunk_6s did not run "
        "(query encoder unavailable)"
    ]


def _tenant_id_boxes(app: AppTest):
    return [box for box in app.text_input if box.label == "Tenant ID"]


def test_the_config_tab_cannot_switch_the_tenant_behind_the_sidebar(page, runtime):
    """The sidebar is the only tenant selector.

    Its change detection drops the previous tenant's state and its gate
    refuses an unregistered tenant. A second free-text box would move the
    tenant-management, memory and A/B tabs past both.
    """
    runtime.search_results["acme:a"] = [_result("doc-1", "video_a", 0.91)]
    app = _open_tenant(page, "acme:a")
    _run_search(app, "robots")

    boxes = _tenant_id_boxes(app)
    assert [box.value for box in boxes] == ["acme:a"]
    assert [box.disabled for box in boxes] == [True]

    boxes[0].set_value("acme:b").run()

    assert [e.message for e in app.exception] == []
    assert app.session_state["current_tenant"] == "acme:a"
    assert app.session_state["active_tenant"] == "acme:a"
    assert app.session_state["current_search_results"]["tenant_id"] == "acme:a"
    assert [
        item.value for item in app.info if item.value.startswith("Current tenant:")
    ] == ["Current tenant: **acme:a**"]


def test_a_result_landing_after_a_switch_is_refused_not_rendered(page, runtime):
    """A search that completes after the switch writes under the old tenant.

    The reset runs at switch time, so a late write repopulates the key; the
    page refuses that record where it would render or export it, and every
    other tab keeps working.
    """
    runtime.search_results["acme:a"] = [_result("doc-1", "video_a", 0.91)]
    runtime.search_results["acme:b"] = []
    app = _open_tenant(page, "acme:a")
    _run_search(app, "robots")
    inflight = app.session_state["current_search_results"]

    _switch_tenant(app, "acme:b")
    assert "current_search_results" not in app.session_state

    app.session_state["current_search_results"] = inflight
    app.run()

    assert [e.message for e in app.exception] == []
    assert [
        e.value for e in app.error if e.value.startswith("Search result belongs")
    ] == ["Search result belongs to acme:a, not acme:b"]
    assert [m.value for m in app.markdown if m.value.startswith("### 📊 Results")] == []
    assert "video_a" not in "".join(
        str(m.value) for m in app.markdown if isinstance(m.value, str)
    )
    assert [b.label for b in app.button if b.label == "💾 Save Annotation"] == []
    assert len([b for b in app.button if b.label == "🔍 Search"]) == 1
    assert [box.value for box in _tenant_id_boxes(app)] == ["acme:b"]


def _recorded_runs(now: datetime) -> List[Dict[str, Any]]:
    """Two runs as the runtime's optimize/runs route returns them, newest
    first. The newest started 90 minutes ago, so its tile reads "1h ago"."""
    return [
        {
            "workflow_name": "manual-optimize-simba-x7k2p",
            "mode": "simba",
            "trigger": "manual",
            "phase": "Running",
            "started_at": (now - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": None,
        },
        {
            "workflow_name": "cogniverse-agent-optimization-1758009600",
            "mode": None,
            "trigger": "scheduled",
            "phase": "Succeeded",
            "started_at": "2026-09-15T03:00:00Z",
            "finished_at": "2026-09-15T03:42:00Z",
        },
    ]


def _history_rows(app: AppTest) -> List[List[Any]]:
    frames = [
        frame.value
        for frame in app.dataframe
        if list(frame.value.columns) == list(OPTIMIZATION_RUN_COLUMNS)
    ]
    assert len(frames) == 1, [list(f.value.columns) for f in app.dataframe]
    return frames[0].values.tolist()


def test_optimization_overview_reads_the_runtimes_runs(page, runtime):
    """The run count, last-run and history elements come from
    ``GET /admin/tenant/{id}/optimize/runs`` — nothing writes them locally."""
    now = datetime.now(timezone.utc)
    recorded = _recorded_runs(now)
    runtime.optimize_runs["acme:a"] = recorded

    app = _open_tenant(page, "acme:a")

    assert [
        (request["tenant_id"], request["limit"])
        for request in runtime.optimize_runs_requests
    ] == [("acme:a", 10)]
    tiles = {m.label: m.value for m in app.metric}
    assert tiles["Optimization Runs"] == "2"
    assert tiles["Last Optimization"] == "1h ago (Running)"
    assert _history_rows(app) == [
        [
            "manual-optimize-simba-x7k2p",
            "simba",
            "manual",
            "Running",
            recorded[0]["started_at"],
            "—",
        ],
        [
            "cogniverse-agent-optimization-1758009600",
            "—",
            "scheduled",
            "Succeeded",
            "2026-09-15T03:00:00Z",
            "2026-09-15T03:42:00Z",
        ],
    ]


def test_a_run_argo_has_not_started_reads_as_not_started(page, runtime):
    """A submitted Workflow the controller has not picked up has no
    ``startedAt`` and no phase; the tile says so rather than "unknown"."""
    runtime.optimize_runs["acme:a"] = [
        {
            "workflow_name": "manual-optimize-profile-6hfm5",
            "mode": "profile",
            "trigger": "manual",
            "phase": None,
            "started_at": None,
            "finished_at": None,
        }
    ]

    app = _open_tenant(page, "acme:a")

    tiles = {m.label: m.value for m in app.metric}
    assert tiles["Optimization Runs"] == "1"
    assert tiles["Last Optimization"] == "not started (Pending)"


def test_tenant_switch_reads_the_new_tenants_optimization_runs(page, runtime):
    """The tiles follow the sidebar's active tenant, never the previous
    tenant's runs."""
    now = datetime.now(timezone.utc)
    runtime.optimize_runs["acme:a"] = _recorded_runs(now)
    runtime.optimize_runs["acme:b"] = []

    app = _open_tenant(page, "acme:a")
    assert {m.label: m.value for m in app.metric}["Optimization Runs"] == "2"

    _switch_tenant(app, "acme:b")

    tiles = {m.label: m.value for m in app.metric}
    assert tiles["Optimization Runs"] == "0"
    assert tiles["Last Optimization"] == "Never"
    assert [request["tenant_id"] for request in runtime.optimize_runs_requests][
        -1
    ] == "acme:b"
    assert [
        frame.value
        for frame in app.dataframe
        if list(frame.value.columns) == list(OPTIMIZATION_RUN_COLUMNS)
    ] == []


def test_optimization_overview_reports_a_runtime_outage_not_zero_runs(page, runtime):
    """A 503 from the runtime shows the error; the count must not read 0."""
    runtime.optimize_runs["acme:a"] = _recorded_runs(datetime.now(timezone.utc))
    runtime.optimize_runs_status = 503

    app = _open_tenant(page, "acme:a")

    tiles = {m.label: m.value for m in app.metric}
    assert tiles["Optimization Runs"] == "—"
    assert tiles["Last Optimization"] == "—"
    assert [
        e.value
        for e in app.error
        if e.value.startswith("Optimization runs unavailable")
    ] == [
        "Optimization runs unavailable: HTTP 503: "
        "Argo API unreachable: connection refused"
    ]
    assert [
        w.value for w in app.warning if w.value.startswith("History unavailable")
    ] == ["History unavailable while the runtime cannot list runs."]
    assert [
        frame.value
        for frame in app.dataframe
        if list(frame.value.columns) == list(OPTIMIZATION_RUN_COLUMNS)
    ] == []
