"""Video ingestion submission for the dashboard's Ingestion Testing tab.

Kept out of ``app.py`` so it is importable (and testable) without executing
``app.py``'s top-level Streamlit UI body.

The dashboard and the runtime are separate Deployments with no shared
filesystem, so the upload goes over the wire as multipart bytes to
``POST /ingestion/upload`` and the job is then polled to a terminal state
through ``GET /ingestion/{ingest_id}/status``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import httpx

# The upload is queued and the worker does keyframe extraction, transcription
# and embedding on it, so a minutes-long wait is the normal case.
DEFAULT_POLL_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 2.0

TERMINAL_STATES = frozenset({"complete", "failed"})


def submit_video_ingestion(
    client: httpx.Client,
    runtime_url: str,
    *,
    filename: str,
    content: bytes,
    content_type: str,
    profile: str,
    tenant_id: str,
    poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Ingest ``content`` under ``profile`` and return the terminal outcome.

    ``status`` is ``success`` only when the worker reported ``complete`` and
    fed at least one document. Every other outcome — a rejected upload, a
    failed job, a poll that ran out of budget, an empty feed — comes back as
    ``error`` with the reason, so the caller has no way to render a failure
    as a success.
    """
    try:
        response = client.post(
            f"{runtime_url}/ingestion/upload",
            files={"file": (filename, content, content_type)},
            data={"profile": profile, "backend": "vespa", "tenant_id": tenant_id},
        )
    except httpx.HTTPError as exc:
        return {
            "status": "error",
            "profile": profile,
            "message": f"Upload to {runtime_url}/ingestion/upload failed: {exc}",
        }

    if response.status_code not in (200, 202):
        return {
            "status": "error",
            "profile": profile,
            "message": (
                f"Upload rejected: HTTP {response.status_code}: {response.text}"
            ),
        }

    body = response.json()
    ingest_id = body.get("ingest_id")
    if not ingest_id:
        return {
            "status": "error",
            "profile": profile,
            "message": f"Upload response carried no ingest_id: {body}",
        }

    state = body.get("state", "queued")
    latest: Dict[str, Any] = {}
    deadline = monotonic() + poll_timeout_s
    while state not in TERMINAL_STATES and monotonic() < deadline:
        sleep(POLL_INTERVAL_S)
        try:
            status = client.get(f"{runtime_url}/ingestion/{ingest_id}/status")
        except httpx.HTTPError as exc:
            return {
                "status": "error",
                "profile": profile,
                "ingest_id": ingest_id,
                "message": f"Ingestion status for {ingest_id} unreadable: {exc}",
            }
        if status.status_code != 200:
            return {
                "status": "error",
                "profile": profile,
                "ingest_id": ingest_id,
                "message": (
                    f"Ingestion status for {ingest_id}: HTTP "
                    f"{status.status_code}: {status.text}"
                ),
            }
        payload = status.json()
        state = payload.get("state", state)
        latest = payload.get("latest", {}) or {}

    if state == "failed":
        return {
            "status": "error",
            "profile": profile,
            "ingest_id": ingest_id,
            "message": (
                f"Ingestion {ingest_id} failed: "
                f"{latest.get('error', 'no error reported')}"
            ),
        }
    if state not in TERMINAL_STATES:
        return {
            "status": "error",
            "profile": profile,
            "ingest_id": ingest_id,
            "message": (
                f"Ingestion {ingest_id} did not reach a terminal state within "
                f"{poll_timeout_s:.0f}s; last state was {state!r}"
            ),
        }

    result = latest.get("result", {}) or {}
    documents_fed = result.get("documents_fed", 0)
    if not isinstance(documents_fed, int) or isinstance(documents_fed, bool):
        return {
            "status": "error",
            "profile": profile,
            "ingest_id": ingest_id,
            "message": (
                f"Ingestion {ingest_id} completed with invalid "
                f"documents_fed={documents_fed!r}"
            ),
        }
    if documents_fed <= 0:
        return {
            "status": "error",
            "profile": profile,
            "ingest_id": ingest_id,
            "message": (
                f"Ingestion {ingest_id} completed without feeding any documents"
            ),
        }
    return {
        "status": "success",
        "profile": profile,
        "ingest_id": ingest_id,
        "video_id": result.get("video_id"),
        "documents_fed": documents_fed,
        "chunks_created": result.get("chunks", result.get("keyframes", 0)),
        "processing_time": result.get("processing_time", 0),
    }
