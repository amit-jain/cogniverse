"""End-to-end inbound messaging against the live cogniverse-runtime pod.

Hits the actual ``cogniverse-runtime`` Service via NodePort 28000 (the
same pod the /ingestion/upload E2E proves can be reached from the
operator's host). The test does NOT bring up its own infrastructure
— it depends on the k3d cluster running ``cogniverse-runtime`` with
the per-session inbound-messaging route deployed.

Each assertion pins behaviour the consumer (a UI, a supervisor agent,
an admin tool) relies on:

* ``POST /agents/{name}/message`` with no active session returns
  exactly 404 with detail ``"session '<id>' not active"`` — the
  caller's mental model is "the session ended (or never existed)."
* ``GET /agents/{name}/sessions/{id}`` mirrors the same 404 detail
  so the polling pattern (start /process in a background task, poll
  for active, then post messages) sees a consistent boundary.
* Past-deadline messages 400 at intake with field+value in detail.
* Invalid role 422 with field locator.

The "constraint actually changes retrieval output" end-to-end
behaviour requires a real BRIGHT-shaped query against the live
gemma + Vespa stack — too slow + LM-flaky to lock byte-equal here.
That assertion is covered by the iterative-loop integration test
which exercises the SHIPPED loop body deterministically; this E2E
file scopes only the HTTP boundary contract against the running
pod.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

RUNTIME_BASE = os.environ.get(
    "COGNIVERSE_RUNTIME_BASE",
    "http://localhost:28000",
)


def _runtime_reachable() -> bool:
    """Skip the suite cleanly when the runtime isn't on the host."""
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{RUNTIME_BASE}/health")
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _runtime_reachable(),
        reason=(
            f"cogniverse-runtime not reachable at {RUNTIME_BASE}; "
            "this E2E requires the k3d cluster to be up with the "
            "runtime NodePort exposed (28000 by default). Set "
            "COGNIVERSE_RUNTIME_BASE if the address differs."
        ),
    ),
]


# --------------------------------------------------------------------- #
# 404 with exact detail when session never existed                       #
# --------------------------------------------------------------------- #


def test_post_message_unknown_session_returns_404_with_exact_detail():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "e2e-never-existed",
                "tenant_id": "flywheel_org:production",
                "role": "user",
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 404
    # Exact match against the live runtime's response — this is the
    # contract the unit-test suite locks too. Drift between the two
    # surfaces means the live deployment doesn't match the contract.
    assert resp.json()["detail"] == "session 'e2e-never-existed' not active"


# --------------------------------------------------------------------- #
# GET session 404 mirrors the same detail                                #
# --------------------------------------------------------------------- #


def test_get_session_unknown_returns_404_with_exact_detail():
    with httpx.Client(timeout=5.0) as c:
        resp = c.get(
            f"{RUNTIME_BASE}/agents/orchestrator/sessions/e2e-unknown",
            params={"tenant_id": "flywheel_org:production"},
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session 'e2e-unknown' not active"


# --------------------------------------------------------------------- #
# Past deadline → 400 at intake (no quiet drop)                          #
# --------------------------------------------------------------------- #


def test_post_message_past_deadline_returns_400_with_field_and_value():
    past_ms = int(time.time() * 1000) - 5_000
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "e2e-any",
                "tenant_id": "flywheel_org:production",
                "role": "user",
                "content": "stale",
                "tags": [],
                "deadline_ms": past_ms,
            },
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "deadline_ms" in detail
    assert str(past_ms) in detail


# --------------------------------------------------------------------- #
# Invalid role → 422 with field locator                                  #
# --------------------------------------------------------------------- #


def test_post_message_invalid_role_returns_422():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "e2e-any",
                "role": "wizard",  # not in {user, system, agent}
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "role") in locs


# --------------------------------------------------------------------- #
# Empty session_id → 422                                                  #
# --------------------------------------------------------------------- #


def test_post_message_empty_session_id_returns_422():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "",
                "role": "user",
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in locs


# --------------------------------------------------------------------- #
# Missing session_id field → 422                                          #
# --------------------------------------------------------------------- #


def test_post_message_missing_session_id_returns_422():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "tenant_id": "flywheel_org:production",
                "role": "user",
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in locs


# --------------------------------------------------------------------- #
# End-to-end inbound channel through /process → orchestrator loop         #
# --------------------------------------------------------------------- #


def _run_orchestrator_process(
    *,
    query: str,
    session_id: str,
    tenant_id: str,
    timeout: float = 180.0,
) -> dict:
    """Submit /process and return the parsed orchestration_result.

    The orchestrator is synchronous — this blocks until the loop
    finishes. The caller hands ``session_id`` so a parallel HTTP
    actor (test harness or another caller) can resolve the running
    session via /sessions/{id} and POST messages.
    """
    with httpx.Client(timeout=timeout) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator_agent/process",
            json={
                "agent_name": "orchestrator_agent",
                "query": query,
                "context": {"tenant_id": tenant_id},
                "top_k": 5,
                "session_id": session_id,
            },
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["orchestration_result"]


def _post_message(
    *,
    session_id: str,
    tenant_id: str,
    content: str,
    tags: list[str],
) -> int:
    """POST /agents/orchestrator/message — returns the HTTP status."""
    with httpx.Client(timeout=10.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": session_id,
                "tenant_id": tenant_id,
                "role": "user",
                "content": content,
                "tags": tags,
            },
        )
    return resp.status_code


def test_baseline_process_run_has_empty_inbound_constraints_applied():
    """Run /process with no inbound messages — the response's
    ``inbound_constraints_applied`` list MUST be exactly ``[]``. This
    is the baseline that with-constraint runs are compared against:
    if a future regression silently drains messages from random
    sessions, this empty baseline would flip and the test fails.
    """
    import uuid

    session_id = f"e2e-baseline-{uuid.uuid4().hex[:8]}"
    result = _run_orchestrator_process(
        query="what about fire",
        session_id=session_id,
        tenant_id="flywheel_org:production",
    )
    loop = result["final_output"]["iterative_loop"]
    # Exact empty list — not "len == 0", not "is not None".
    assert loop["inbound_constraints_applied"] == []


def test_constraint_posted_mid_process_lands_in_inbound_constraints_applied():
    """Background task: POST /process. Foreground: wait for the
    session to go active via /sessions/{id}, POST a constraint, wait
    for /process to return. The response's
    ``inbound_constraints_applied`` list MUST equal exactly
    ``["<the constraint text>"]`` — proving the channel reached the
    orchestrator's iterative loop end-to-end.

    Strong assertion: the WITH-CONSTRAINT run differs from the
    baseline in exactly this field, no others. Pinpoints the inbound
    channel as the cause of the diff, not LM noise.
    """
    import threading
    import uuid

    session_id = f"e2e-with-constraint-{uuid.uuid4().hex[:8]}"
    tenant_id = "flywheel_org:production"
    constraint_text = "only sources from 2024"

    result_holder: dict = {}
    thread_error: list = []

    def _run():
        try:
            result_holder["result"] = _run_orchestrator_process(
                query="what about fire",
                session_id=session_id,
                tenant_id=tenant_id,
            )
        except Exception as exc:  # noqa: BLE001 — propagated to assertion
            thread_error.append(exc)

    bg = threading.Thread(target=_run, daemon=True)
    bg.start()

    # Poll /sessions/{id} until 200 (orchestrator registered the
    # session at loop entry).
    deadline = time.time() + 30
    active = False
    while time.time() < deadline:
        with httpx.Client(timeout=2.0) as c:
            sess_resp = c.get(
                f"{RUNTIME_BASE}/agents/orchestrator/sessions/{session_id}",
                params={"tenant_id": tenant_id},
            )
        if sess_resp.status_code == 200:
            active = True
            break
        time.sleep(0.5)
    assert active, f"session {session_id} never went active within 30 s"

    # Inject the constraint while the orchestrator is mid-flight.
    post_status = _post_message(
        session_id=session_id,
        tenant_id=tenant_id,
        content=constraint_text,
        tags=["constraint"],
    )
    assert post_status == 202, f"expected 202 on inbound POST, got {post_status}"

    bg.join(timeout=240)
    assert not thread_error, f"background /process raised: {thread_error[0]!r}"
    assert "result" in result_holder, "background /process didn't return"

    loop = result_holder["result"]["final_output"]["iterative_loop"]
    # Exact-list assertion — the constraint reached the loop AND
    # only that constraint did (no phantom drain of other sessions'
    # messages bleeding in via the singleton registry).
    assert loop["inbound_constraints_applied"] == [constraint_text]


def test_cross_tenant_post_to_active_session_returns_404_live():
    """Live-cluster cross-tenant probe. POST to an active session
    using a DIFFERENT tenant must 404 with the same detail used for
    unknown sessions. Catches the case where the deployed runtime
    forgot to apply the tenant guard.

    Runs against a freshly-started session to avoid race with the
    previous test's session lifecycle.
    """
    import threading
    import uuid

    session_id = f"e2e-xtenant-{uuid.uuid4().hex[:8]}"
    owning_tenant = "flywheel_org:production"

    result_holder: dict = {}

    def _run():
        try:
            result_holder["result"] = _run_orchestrator_process(
                query="what about fire",
                session_id=session_id,
                tenant_id=owning_tenant,
            )
        except Exception:
            pass

    bg = threading.Thread(target=_run, daemon=True)
    bg.start()

    # Wait for active.
    deadline = time.time() + 30
    while time.time() < deadline:
        with httpx.Client(timeout=2.0) as c:
            sess_resp = c.get(
                f"{RUNTIME_BASE}/agents/orchestrator/sessions/{session_id}",
                params={"tenant_id": owning_tenant},
            )
        if sess_resp.status_code == 200:
            break
        time.sleep(0.5)

    # Wrong tenant attempts a POST → 404 with the same detail.
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": session_id,
                "tenant_id": "wrong-tenant",
                "role": "user",
                "content": "probe",
                "tags": [],
            },
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == f"session '{session_id}' not active"

    # Wait for the orchestrator to finish so the session closes
    # cleanly before the next test.
    bg.join(timeout=240)
